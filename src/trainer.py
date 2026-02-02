# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import os
import torch
import torch.optim as optim
from src.utils import compute_gradient_penalty, save_fake_slice


class Trainer:
    """
    Trainer class to manage the GAN training loop.

    It handles the WGAN-GP optimization logic, logging, checkpoint saving,
    and visualization of progress.
    """

    def __init__(self, generator, discriminator, dataloader, device, config):
        """
        Initialize the Trainer.

        Parameters
        ----------
        generator : nn.Module
            The Generator network.
        discriminator : nn.Module
            The Discriminator network.
        dataloader : torch.utils.data.DataLoader
            DataLoader providing the training data.
        device : torch.device
            Training device (cpu or cuda).
        config : argparse.Namespace
            Object containing hyperparameters (lr, n_critic, latent_dim, epochs, save_dir).
        """
        self.G = generator
        self.D = discriminator
        self.dataloader = dataloader
        self.device = device
        self.config = config

        # WGAN-GP Hyperparameters
        self.lambda_gp = 10  # Standard weight for gradient penalty in WGAN-GP papers
        self.n_critic = getattr(config, 'n_critic', 5)

        # Optimizers.
        # Beta1=0.5, Beta2=0.9 are empirically recommended for GAN stability (Radford et al.).
        self.opt_G = optim.Adam(self.G.parameters(), lr=config.lr, betas=(0.5, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.5, 0.9))

        # Fixed noise and labels for monitoring progress.
        # Using fixed input allows visualizing how the generator improves on the SAME latent space over time.
        self.fixed_noise = torch.randn(3, config.latent_dim).to(device)
        self.fixed_labels = torch.tensor([0, 1, 2]).to(device)

        # Setup Output Directories
        self.checkpoint_dir = os.path.join(config.save_dir, "checkpoints")
        self.images_dir = os.path.join(config.save_dir, "progress_images")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def train(self):
        """
        Execute the main training loop.

        Iterates through epochs and batches, updating the Discriminator and Generator
        according to the WGAN-GP schedule. Saves checkpoints and progress images.

        Raises
        ------
        ValueError
            If the loss becomes NaN (divergence).
        """
        print(f"Starting training process on {self.device}...")

        for epoch in range(self.config.epochs):
            for i, (real_imgs, labels) in enumerate(self.dataloader):

                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_imgs.size(0)

                # ==================================================================
                # 1. Train Discriminator (Critic)
                # ==================================================================
                self.opt_D.zero_grad()

                # Sample noise and generate fake images
                z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
                
                # Detach prevents gradients from flowing back into the Generator during D training
                fake_imgs = self.G(z, labels).detach()

                # WGAN Loss Calculation
                # The Critic tries to maximize (D(real) - D(fake)).
                # Since optimizers minimize, we minimize -(D(real) - D(fake)) = D(fake) - D(real).
                d_real = self.D(real_imgs, labels).mean()
                d_fake = self.D(fake_imgs, labels).mean()

                # Enforce 1-Lipschitz continuity via Gradient Penalty
                # This prevents mode collapse and stabilizes training compared to standard WGAN clipping.
                gp = compute_gradient_penalty(self.D, real_imgs, fake_imgs, labels, self.device)

                d_loss = d_fake - d_real + self.lambda_gp * gp

                # Defensive Programming: Check for divergence
                if torch.isnan(d_loss):
                    raise ValueError("Training diverged: Discriminator Loss is NaN. Try lowering the learning rate.")

                d_loss.backward()
                self.opt_D.step()

                # ==================================================================
                # 2. Train Generator (every n_critic steps)
                # ==================================================================
                g_loss_val = 0.0  # Initialize for logging safety

                if i % self.n_critic == 0:
                    self.opt_G.zero_grad()

                    # Generate fresh fakes (this time we need gradients, so no detach)
                    gen_imgs = self.G(z, labels)

                    # WGAN Generator Loss
                    # The Generator wants to maximize D(fake).
                    # Minimizing -D(fake) achieves this.
                    g_loss = -torch.mean(self.D(gen_imgs, labels))

                    if torch.isnan(g_loss):
                        raise ValueError("Training diverged: Generator Loss is NaN.")

                    g_loss.backward()
                    self.opt_G.step()
                    
                    # Store value for logging
                    g_loss_val = g_loss.item()

                # Log progress
                if i % 10 == 0:
                    # If g_loss hasn't been updated yet in this loop, use 0.0 or previous value
                    print(f"[Epoch {epoch+1}/{self.config.epochs}] [Batch {i}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss_val:.4f}]")

            # ==================================================================
            # End of Epoch Actions
            # ==================================================================
            
            # 1. Visualize Progress
            try:
                save_fake_slice(
                    self.G, 
                    self.fixed_noise, 
                    self.fixed_labels, 
                    epoch + 1,
                    output_dir=self.images_dir
                )
            except Exception as e:
                print(f"Warning: Could not save progress image: {e}")

            # 2. Save Model Checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    self.G.state_dict(), 
                    os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch+1}.pth")
                )
                print(f"Checkpoint saved for epoch {epoch+1}")