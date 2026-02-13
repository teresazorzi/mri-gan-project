# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import argparse
from argparse import Namespace
import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
import yaml
import numpy as np
import random

# Add the directory containing this script to the Python path.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import MRINiftiDataset
from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D, weights_init
from src.trainer import Trainer

def set_seed(seed):
    """
    Ensures reproducibility by fixing random seeds for all libraries.

    Parameters
    ----------
    seed : int
        The integer value used to initialize the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_config(config_path):
    """
    Loads the simulation parameters from YAML configuration file.

    Parameters
    ----------
    config_path : str
        The filesystem path to the .yaml configuration file.

    Returns
    -------
    dict
        A dictionary containing the parsed configuration parameters.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_arguments():
    """
    CLI interface to provide the configuration file path.

    Returns
    -------
    argparse.Namespace
        Object containing the path to the configuration file.
    """

    parser = argparse.ArgumentParser(description="3D MRI GAN Training Launcher")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file."
    )

    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--lr", type=float, help="Override learning rate.")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda).")
    parser.add_argument("--data_root", type=str, help="Override dataset root path.")
   
    return parser.parse_args()

def validate_config(config):
    """
    Performs checks to ensure that all required parameters are present and that their values 
    are physically or mathematically meaningful.

    Parameters
    ----------
    config : dict
        The configuration dictionary loaded from the YAML file.

    Returns
    -------
    None
        The function returns nothing. It continues execution if valid.

    Raises
    ------
    KeyError
        If a required section or parameter is missing from the configuration.
    ValueError
        If a parameter has an invalid value (e.g., negative batch size).
    FileNotFoundError
        If the data paths specified do not exist.
    """
    
    required_params = {
        'dataset': ['data_root', 'file_pattern', 'target_shape', 'num_workers'],
        'model': ['latent_dim', 'ngf', 'ndf', 'num_classes'],
        'training': ['epochs', 'batch_size', 'lr', 'n_critic', 'lambda_gp', 'device', 'seed'],
        'output': ['save_dir', 'sample_interval', 'checkpoint_interval']
    }

    # check for required sections and keys
    for section, keys in required_params.items():
        if section not in config:
            raise KeyError(f"Configuration file is missing required section: '{section}'")        
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing key '{key}' in section '{section}'")

    # Logical checks for values 
    # Dataset Checks
    if not os.path.exists(config['dataset']['data_root']):
        raise FileNotFoundError(f"Data root path does not exist: {config['dataset']['data_root']}")
    if not isinstance(config['dataset']['file_pattern'], str) or not config['dataset']['file_pattern']:
        raise ValueError("file_pattern must be a non-empty string") 
    shape = config['dataset']['target_shape']
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"target_shape must be a list of 3 integers (Depth, Height, Width), got {shape}")
    if config['dataset']['num_workers'] < 0:
        raise ValueError("num_workers must be >= 0")
    
    # Model Checks
    model = config['model']
    if model['latent_dim'] < 1: raise ValueError("Latent dimension must be >= 1")
    if model['ngf'] < 1 or model['ndf'] < 1:
        raise ValueError("Number of filters (ngf, ndf) must be >= 1")
    if model['num_classes'] < 1:
        raise ValueError("num_classes must be >= 1")

    # Training Checks
    train = config['training']
    if train['batch_size'] < 1: raise ValueError("Batch size must be >= 1")
    if train['epochs'] < 1: raise ValueError("Epochs must be >= 1")
    if train['lr'] <= 0: raise ValueError("Learning rate must be > 0")
    if train['n_critic'] < 1: raise ValueError("n_critic must be >= 1")
    if train['lambda_gp'] < 0: raise ValueError("lambda_gp must be >=0")
    valid_devices = ['cpu', 'cuda']
    if train['device'] not in valid_devices and 'cuda:' not in train['device']:
        raise ValueError(f"Invalid device '{train['device']}'. Must be one of {valid_devices} or 'cuda:id'")
 
    # Output Checks
    out = config['output']
    if out['sample_interval'] < 1: raise ValueError("Sample interval must be >= 1")
    if out['checkpoint_interval'] < 1: raise ValueError("Checkpoint interval must be >= 1")
    if not isinstance(out['save_dir'], str) or not out['save_dir']:
        raise ValueError("save_dir must be a valid directory path string")

def main():
    """
    Main execution flow.
    Steps:
    1. Parse command-line arguments and load configuration.
    2. Validate configuration parameters for logical consistency.
    3. Prepare the execution environment (set seeds, create directories).
    4. Load datasets for each class and concatenate them.
    5. Initialize the Generator and Discriminator models.
    6. Create a Trainer instance and start the training loop.
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If any of the configuration parameters fail validation checks.
    """

    # Parse and Load
    args = parse_arguments()
    config = load_config(args.config)
    if args.epochs:     config['training']['epochs'] = args.epochs
    if args.batch_size: config['training']['batch_size'] = args.batch_size
    if args.lr:         config['training']['lr'] = args.lr
    if args.device:     config['training']['device'] = args.device
    if args.data_root:  config['dataset']['data_root'] = args.data_root

    # Perform validation checks on the configuration parameters
    validate_config(config)
    
    # Prepare the execution environment
    # 1. Set Reproducibility
    set_seed(config['training']['seed'])
    
    # 2. Create Output Directory
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    data_root = config['dataset']['data_root']
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    target_shape = tuple(config['dataset']['target_shape'])
    num_classes = config['model']['num_classes']

    print(f"\n{'='*50}")
    print(f"MRI GAN TRAINING LAUNCHER")
    print(f"{'='*50}")
    print(f"Config File  : {args.config}")
    print(f"Data Root    : {data_root}")
    print(f"Device       : {device}")
    print(f"Epochs       : {config['training']['epochs']}") 
    print(f"Batch Size   : {config['training']['batch_size']}")
    print(f"Seed         : {config['training']['seed']}")
    print(f"{'='*50}\n")

    # --- 1. DATASET SETUP ---
    print("Loading datasets...")
    try:
        class_folders = sorted([
            d for d in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, d))
        ])

        # check that the number of class folders matches the expected number of classes
        if len(class_folders) != num_classes:
            raise ValueError(
                f"CONFIG MISMATCH: 'num_classes' in config is {num_classes}, "
                f"but found {len(class_folders)} subfolders in '{data_root}': {class_folders}.\n"
                f"Check your config.yaml file or the folder structure."
            )
        
        print(f"Class mapping: { {i: name for i, name in enumerate(class_folders)} }")
        
        datasets = []
        for label_idx, class_name in enumerate(class_folders): 
            ds = MRINiftiDataset(
                root_dir=os.path.join(data_root, class_name) , 
                label=label_idx, 
                file_pattern=config['dataset']['file_pattern'], 
                target_shape=target_shape
            )
            
            if len(ds) == 0:
                print(f"WARNING: Dataset for class '{class_name}' is empty. Ignored.")
            else:
                print(f" - Loaded Class '{class_name}' (Label {label_idx}): {len(ds)} images")
                datasets.append(ds)

        if not datasets:
             raise RuntimeError("No dataset loaded. Unable to start training.")

        full_dataset = ConcatDataset(datasets)
        print(f"Total training images: {len(full_dataset)}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        return

    dataloader = DataLoader(
        full_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['dataset']['num_workers']
    )

    # --- 2. MODEL INITIALIZATION ---
    print("Initializing models...")
    generator = CPUOptimizedGenerator3D(
        latent_dim=config['model']['latent_dim'], 
        num_classes=num_classes,
        target_shape=target_shape,
        ngf=config['model']['ngf']
    ).to(device)

    discriminator = CPUOptimizedDiscriminator3D(
        num_classes=num_classes,
        ndf=config['model']['ndf'],
        input_shape=target_shape
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # --- 3. TRAINING LOOP ---
    # Create a Namespace for Trainer compatibility
    trainer_config = Namespace(
        lr=config['training']['lr'],
        epochs=config['training']['epochs'],
        latent_dim=config['model']['latent_dim'],
        num_classes=num_classes,
        n_critic=config['training']['n_critic'],
        lambda_gp=config['training']['lambda_gp'],         
        save_dir=save_dir,
        sample_interval=config['output']['sample_interval'],        
        checkpoint_interval=config['output']['checkpoint_interval']
    )

    trainer = Trainer(generator, discriminator, dataloader, device, config=trainer_config)
    trainer.train()

if __name__ == "__main__":
    main()