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
from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
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
    # Path Checks
    if not os.path.exists(config['dataset']['data_root']):
        raise FileNotFoundError(f"Data root path does not exist: {config['dataset']['data_root']}")

    # Training Checks
    train = config['training']
    if train['batch_size'] < 1: raise ValueError("Batch size must be >= 1")
    if train['epochs'] < 1: raise ValueError("Epochs must be >= 1")
    if train['lr'] <= 0: raise ValueError("Learning rate must be > 0")
    if train['n_critic'] < 1: raise ValueError("n_critic must be >= 1")
    if train['lambda_gp'] < 0: raise ValueError("lambda_gp must be >=0")
    
    # Device Check 
    valid_devices = ['cpu', 'cuda']
    if train['device'] not in valid_devices and 'cuda:' not in train['device']:
        raise ValueError(f"Invalid device '{train['device']}'. Must be one of {valid_devices} or 'cuda:id'")

    # Model Checks
    if config['model']['latent_dim'] < 1: raise ValueError("Latent dimension must be >= 1")
    if config['model']['ngf'] < 1 or config['model']['ndf'] < 1:
        raise ValueError("Number of filters (ngf, ndf) must be >= 1")

    # Dataset Checks
    shape = config['dataset']['target_shape']
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"target_shape must be a list of 3 integers (Depth, Height, Width), got {shape}")

    # Output Checks
    out = config['output']
    if out['sample_interval'] < 1: raise ValueError("Sample interval must be >= 1")
    if out['checkpoint_interval'] < 1: raise ValueError("Checkpoint interval must be >= 1")
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

    print(f"\n{'='*50}")
    print(f"MRI GAN TRAINING LAUNCHER")
    print(f"{'='*50}")
    print(f"Config File  : {args.config}")
    print(f"Data Root    : {data_root}")
    print(f"Device       : {device}")
    print(f"Seed         : {config['training']['seed']}")
    print(f"{'='*50}\n")

    # --- 1. DATASET SETUP ---
    print("Loading datasets...")
    try:
        file_pattern = config['dataset']['file_pattern']
        target_shape = tuple(config['dataset']['target_shape'])
        # Initialize specific class datasets using the flexible file pattern.
        # This makes the code usable for T1, T2, or other registered modalities.
        ds_ad = MRINiftiDataset(os.path.join(data_root, 'AD'), label=0, file_pattern=file_pattern, target_shape=target_shape)
        ds_cn = MRINiftiDataset(os.path.join(data_root, 'CN'), label=1, file_pattern=file_pattern, target_shape=target_shape)
        ds_lmci = MRINiftiDataset(os.path.join(data_root, 'LMCI'), label=2, file_pattern=file_pattern, target_shape=target_shape)

        full_dataset = ConcatDataset([ds_ad, ds_cn, ds_lmci])

        # Verification step: ensure data was actually loaded.
        if len(full_dataset) == 0:
            raise RuntimeError(f"No files found matching '{file_pattern}' in {data_root}. "
                               f"Please check your data directory structure.")

        print(f"Total images found: {len(full_dataset)}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
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
        ngf=config['model']['ngf']
    ).to(device)

    discriminator = CPUOptimizedDiscriminator3D(
        ndf=config['model']['ndf']
    ).to(device)

    # --- 3. TRAINING LOOP ---
    # Create a Namespace for Trainer compatibility
    trainer_config = Namespace(
        lr=config['training']['lr'],
        epochs=config['training']['epochs'],
        latent_dim=config['model']['latent_dim'],
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