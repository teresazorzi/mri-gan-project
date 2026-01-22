import os
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Import our custom modules
from .dataset import MRINiftiDataset
from .models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D, weights_init
from .utils import compute_gradient_penalty, save_fake_slice

def train_model(config, data_root, device, output_dir):
    """
    Main training loop for the MRI GAN.
    
    Args:
        config (dict): Dictionary containing training hyperparameters (lr, epochs, latent_dim, etc.).
        data_root (str): Path to the directory containing class folders.
        device (torch.device): Device to run the training on (CPU or CUDA).
        output_dir (str): Path where checkpoints and logs will be saved.
    """
    
    # --- 1. SETUP DIRECTORIES ---
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    images_dir = os.path.join(output_dir, "progress_images")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # --- 2. DATASET PREPARATION ---
    # We scan the data_root folder to find class subfolders
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print(f"Found classes: {classes}")
    
    train_subsets = []
    target_shape = (64, 64, 64) # Fixed shape as per architecture

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_root, class_name)
        # We use a simple 80/20 split logic or use full dataset depending on config.
        # Here we load the full dataset for training as in the original script logic.
        ds = MRINiftiDataset(class_dir, label=i, target_shape=target_shape, augment=True)
        train_subsets.append(ds)
        print(f"Class {class_name}: {len(ds)} samples loaded.")

    if not train_subsets:
        raise ValueError("No data found! Check the data_root path.")

    train_dataset = ConcatDataset(train_subsets)
    loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # --- 3. MODEL INITIALIZATION ---
    G = CPUOptimizedGenerator3D(latent_dim=config['latent_dim'], num_classes=len(classes)).to(device)
    D = CPUOptimizedDiscriminator3D(num_classes=len(classes)).to(device)
    
    # Initialize weights
    G.apply(weights_init)
    D.apply(weights_init)

    # Optimizers (Adam is standard for GANs)
    opt_G = optim.Adam(G.parameters(), lr=config['lr'], betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=config['lr'] * 0.1, betas=(0.5, 0.9))

    # --- 4. TRAINING LOOP ---
    print("Starting training...")
    history = {'epoch': [], 'd_loss': [], 'g_loss': []}
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_d_loss = []
        epoch_g_loss = []
        
        for i, (real, labels) in enumerate(loader):
            real, labels = real.to(device), labels.to(device)
            batch_size = real.size(0)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_D.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, config['latent_dim'], device=device)
            fake = G(z, labels).detach() # Detach to avoid training G here
            
            # WGAN-GP Loss
            d_real = D(real, labels)
            d_fake = D(fake, labels)
            gp = compute_gradient_penalty(D, real, fake, labels, device)
            
            loss_D = -torch.mean(d_real) + torch.mean(d_fake) + 10 * gp
            loss_D.backward()
            opt_D.step()
            epoch_d_loss.append(loss_D.item())

            # -----------------
            #  Train Generator
            # -----------------
            # Train G every n_critic steps
            if i % config['n_critic'] == 0:
                opt_G.zero_grad()
                # Re-generate fakes (with gradients attached this time)
                gen_fake = G(z, labels)
                loss_G = -torch.mean(D(gen_fake, labels))
                loss_G.backward()
                opt_G.step()
                epoch_g_loss.append(loss_G.item())

        # --- LOGGING ---
        avg_d = np.mean(epoch_d_loss) if epoch_d_loss else 0
        avg_g = np.mean(epoch_g_loss) if epoch_g_loss else 0
        
        history['epoch'].append(epoch)
        history['d_loss'].append(avg_d)
        history['g_loss'].append(avg_g)
        
        print(f"Epoch {epoch}/{config['epochs']} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

        # Save Checkpoint & CSV every 10 epochs
        if epoch % 10 == 0:
            # Save metrics
            pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
            
            # Save model weights
            torch.save(G.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth"))
            
            # Save preview image
            save_fake_slice(G, config, os.path.join(images_dir, f"epoch_{epoch}.png"), device)

    print("Training finished.")