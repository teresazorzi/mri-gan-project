import argparse
import os
import torch
import sys

# Add the current directory to python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trainer import train_model

def main():
    """
    Main entry point for the MRI GAN training application.
    Parses command line arguments and initiates the training process.
    """
    parser = argparse.ArgumentParser(description="Train a GAN for 3D MRI Synthesis")
    
    # --- 1. Data and Output Paths (CRITICO: No percorsi assoluti nel codice!) ---
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Path to the folder containing class subfolders (AD, CN, LMCI)")
    parser.add_argument("--output_dir", type=str, default="results", 
                        help="Directory where results and checkpoints will be saved")
    
    # --- 2. Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (reduce if OOM)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimension of the latent vector z")
    parser.add_argument("--n_critic", type=int, default=5, help="Critic steps per generator step")
    
    # --- 3. System Config ---
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use: 'cuda' or 'cpu'")

    args = parser.parse_args()

    # Create config dictionary
    config = {
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'latent_dim': args.latent_dim,
        'n_critic': args.n_critic
    }

    # Validate paths (Esame: "raise an appropriate exception")
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"The data path '{args.data_root}' does not exist. Please check the path.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run Training
    print(f"\n{'='*40}")
    print(f"MRI GAN TRAINING LAUNCHER")
    print(f"{'='*40}")
    print(f"Data Root : {args.data_root}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Device    : {args.device}")
    print(f"Config    : {config}")
    print(f"{'='*40}\n")
    
    device = torch.device(args.device)
    
    # Start the engine
    train_model(config, args.data_root, device, args.output_dir)

if __name__ == "__main__":
    main()