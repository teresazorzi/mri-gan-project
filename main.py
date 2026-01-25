# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import argparse
import os
import torch
import sys
from torch.utils.data import DataLoader, ConcatDataset

# Add the directory containing this script to the Python path.
# This ensures that 'import src...' works regardless of where the script is launched from.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import MRINiftiDataset
from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
from src.trainer import Trainer

def parse_arguments():
    """
    Parses command line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train a GAN for 3D MRI Synthesis")
    
    # --- 1. Data and Output Configuration ---
    parser.add_argument("--data_root", type=str, default="data", 
                        help="Path to the root folder containing class subfolders (AD, CN, LMCI)")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory where results, checkpoints, and logs will be saved")
    
    # New argument for generalization (Modality-Agnostic)
    parser.add_argument("--file_pattern", type=str, default="MPRAGE_MNI_norm.nii.gz",
                        help="Filename pattern to search for (e.g. '*.nii' or specific filenames). "
                             "Allows using the code with different datasets/modalities.")
    
    # --- 2. Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (reduce if Out Of Memory)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of the latent vector z")
    parser.add_argument("--n_critic", type=int, default=5, help="Number of Discriminator steps per Generator step")
    
    # --- 3. System Configuration ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation ('cuda' or 'cpu')")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of subprocesses for data loading (0 for Windows)")

    return parser.parse_args()

def main():
    """
    Main execution flow:
    1. Parse arguments and validate inputs.
    2. Setup data loaders with the specified file pattern.
    3. Initialize Generator and Discriminator.
    4. Start the training loop.
    """
    args = parse_arguments()

    # --- DEFENSIVE PROGRAMMING Checks ---
    # Validate critical inputs before starting heavy operations
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"The data path '{args.data_root}' does not exist. Please check the path.")
    
    if args.batch_size < 1:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    
    if args.epochs < 1:
        raise ValueError(f"Epochs must be positive, got {args.epochs}")

    # Create output directories structure
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "progress_images"), exist_ok=True)

    device = torch.device(args.device)

    print(f"\n{'='*50}")
    print(f"MRI GAN TRAINING LAUNCHER")
    print(f"{'='*50}")
    print(f"Data Root    : {args.data_root}")
    print(f"File Pattern : {args.file_pattern}")
    print(f"Device       : {device}")
    print(f"Epochs       : {args.epochs}")
    print(f"{'='*50}\n")

    # --- 1. DATASET SETUP ---
    print("Loading datasets...")
    try:
        # Initialize specific class datasets using the flexible file pattern
        # This makes the code usable for T1, T2, or other registered modalities
        ds_ad = MRINiftiDataset(os.path.join(args.data_root, 'AD'), label=0, file_pattern=args.file_pattern)
        ds_cn = MRINiftiDataset(os.path.join(args.data_root, 'CN'), label=1, file_pattern=args.file_pattern)
        ds_lmci = MRINiftiDataset(os.path.join(args.data_root, 'LMCI'), label=2, file_pattern=args.file_pattern)
        
        # Combine them into a single dataset
        full_dataset = ConcatDataset([ds_ad, ds_cn, ds_lmci])
        
        # Check if data was actually found
        if len(full_dataset) == 0:
            raise RuntimeError(f"No files found matching '{args.file_pattern}' in {args.data_root}. "
                               f"Please check your data directory structure.")
            
        print(f"Total images found: {len(full_dataset)}")
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        return # Exit cleanly

    dataloader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )

    # --- 2. MODEL INITIALIZATION ---
    print("Initializing models...")
    generator = CPUOptimizedGenerator3D(latent_dim=args.latent_dim).to(device)
    discriminator = CPUOptimizedDiscriminator3D().to(device)

    # --- 3. TRAINING LOOP ---
    # Create a configuration object to pass hyperparameters to the Trainer
    class Config:
        pass
    config = Config()
    config.lr = args.lr
    config.epochs = args.epochs
    config.latent_dim = args.latent_dim
    config.n_critic = args.n_critic # Pass n_critic specifically for WGAN
    
    # Initialize the Trainer Engine
    trainer = Trainer(generator, discriminator, dataloader, device, config)
    
    # Start the training process
    trainer.train()

if __name__ == "__main__":
    main()