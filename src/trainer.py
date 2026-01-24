# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import torch
import torch.optim as optim
import os
import sys

# Ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import compute_gradient_penalty, save_fake_slice

class Trainer:
    """
    Trainer class to manage the GAN training loop.
    It handles the WGAN-GP optimization logic, logging, and checkpoint saving.
    """
    def __init__(self, generator, discriminator, dataloader, device, config):
        """
        Args:
            generator: The Generator network.
            discriminator: The Discriminator network.
            dataloader: PyTorch DataLoader.
            device: Training device (cpu or cuda).
            config: Object containing hyperparameters (lr, n_critic, etc.).
        """
        self.G = generator
        self.D = discriminator
        self.dataloader = dataloader
        self.device = device
        self.config = config
        
        # WGAN-GP Parameters
        self.lambda_gp = 10 
        self.n_critic = getattr(config, 'n_critic', 5) # Default to 5 if not set
        
        # Optimizers (Adam is standard for WGAN-GP)
        # Betas (0.5, 0.9) are recommended for GAN stability
        self.opt_G = optim.Adam(self.G.parameters(), lr=config.lr, betas=(0.5, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.5, 0.9))

        # Fixed noise for monitoring progress (consistency across epochs)
        self.fixed_noise = torch.randn(3, config.latent_dim).to(device)
        self.fixed_labels = torch.tensor([0, 1, 2]).to(device)

    def train(self):
        """
        Executes the main training loop.
        """
        print("Starting training process...")
        
        for epoch in range(self.config.epochs):
            for i, (real_imgs, labels) in enumerate(self.dataloader):
                
                # Move data to device
                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)
                
                batch_size = real_imgs.size(0)

                # ---------------------
                #  1. Train Discriminator
                # ---------------------
                self.opt_D.zero_grad()
                
                # Generate fake images
                z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
                fake_imgs = self.G(z, labels).detach() # Detach to avoid G gradients
                
                # WGAN Loss: D(fake) - D(real) (To minimize D_loss is to maximize D(real) - D(fake))
                d_real = self.D(real_imgs, labels).mean()
                d_fake = self.D(fake_imgs, labels).mean()
                
                # Gradient Penalty (Enforce Lipschitz constraint)
                gp = compute_gradient_penalty(self.D, real_imgs, fake_imgs, labels, self.device)
                
                d_loss = d_fake - d_real + self.lambda_gp * gp
                
                # --- SAFETY CHECK: NaN Loss (Defensive Programming) ---
                if torch.isnan(d_loss):
                    raise ValueError("Training diverged: Discriminator Loss is NaN. Try lowering the learning rate.")

                d_loss.backward()
                self.opt_D.step()

                # -----------------
                #  2. Train Generator (every n_critic steps)
                # -----------------
                if i % self.n_critic == 0:
                    self.opt_G.zero_grad()
                    
                    # Generate fresh fakes (with gradients)
                    gen_imgs = self.G(z, labels)
                    
                    # Generator wants to maximize D(fake) -> minimize -D(fake)
                    g_loss = -torch.mean(self.D(gen_imgs, labels))
                    
                    # --- SAFETY CHECK: NaN Loss ---
                    if torch.isnan(g_loss):
                        raise ValueError("Training diverged: Generator Loss is NaN.")

                    g_loss.backward()
                    self.opt_G.step()

                # Print Log every 10 batches
                if i % 10 == 0:
                    g_loss_val = g_loss.item() if 'g_loss' in locals() else 0.0
                    print(f"[Epoch {epoch}/{self.config.epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss_val:.4f}]")

            # --- End of Epoch Actions ---
            
            # 1. Save progress image (using the robust util function)
            try:
                save_fake_slice(self.G, self.fixed_noise, self.fixed_labels, epoch, 
                                output_dir=os.path.join("results", "progress_images"))
            except Exception as e:
                print(f"Warning: Could not save progress image: {e}")
            
            # 2. Save Model Checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_dir = os.path.join("results", "checkpoints")
                torch.save(self.G.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.pth"))
                print(f"Checkpoint saved for epoch {epoch+1}")