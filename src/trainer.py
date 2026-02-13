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
    Core engine for the WGAN-GP training process.

    This class manages the adversarial optimization loop, enforcing 
    Lipschitz continuity via gradient penalty.
    """

    def __init__(self, generator, discriminator, dataloader, device, config):
        """
        Initialize the training environment.

        Parameters
        ----------
        generator : nn.Module
            The Generator network.
        discriminator : nn.Module
            The Discriminator network.
        dataloader : torch.utils.data.DataLoader
            Iterator providing volumetric MRI batches and labels.
        device : torch.device
            Computation target (CPU or CUDA).
        config : argparse.Namespace
            Container for hyperparameters and output paths.
        """
        self.G = generator
        self.D = discriminator
        self.dataloader = dataloader
        self.device = device
        self.config = config

        self.lambda_gp = config.lambda_gp
        self.n_critic = config.n_critic

        # Adam optimizers with momentum (0.5, 0.9) optimized for WGAN-GP equilibrium
        self.opt_G = optim.Adam(self.G.parameters(), lr=config.lr, betas=(0.5, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=config.lr, betas=(0.5, 0.9))

        # Fixed noise ensures consistent progress tracking across epochs
        self.fixed_noise = torch.randn(config.num_classes, config.latent_dim).to(device)
        self.fixed_labels = torch.arange(config.num_classes).to(device)

        self.checkpoint_dir = os.path.join(config.save_dir, "checkpoints")
        self.images_dir = os.path.join(config.save_dir, "progress_images")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def train(self):
        """
        Execute the adversarial optimization loop.

        Implements the WGAN-GP training protocol, alternating between
        Critic updates and Generator synthesis.

        Raises
        ------
        ValueError
            If numerical divergence (NaN loss) occurs in either network.
        """
        print(f"Starting training process on {self.device}...")

        for epoch in range(self.config.epochs):
            for i, (real_imgs, labels) in enumerate(self.dataloader):

                real_imgs = real_imgs.to(self.device)
                labels = labels.to(self.device)
                batch_size = real_imgs.size(0)

                # 1. Train Discriminator
                self.opt_D.zero_grad()

                z = torch.randn(batch_size, self.config.latent_dim).to(self.device)
                
                # Detach prevents computing gradients for G during the Discriminator update
                fake_imgs = self.G(z, labels).detach()

                d_real = self.D(real_imgs, labels).mean()
                d_fake = self.D(fake_imgs, labels).mean()

                # Penalty enforces 1-Lipschitz continuity to prevent mode collapse
                gp = compute_gradient_penalty(self.D, real_imgs, fake_imgs, labels, self.device)
                d_loss = d_fake - d_real + self.lambda_gp * gp

                if torch.isnan(d_loss):
                    raise ValueError("Training diverged: Discriminator Loss is NaN.")

                d_loss.backward()
                self.opt_D.step()

                # 2. Train Generator
                g_loss_val = 0.0  

                if i % self.n_critic == 0:
                    self.opt_G.zero_grad()

                    gen_imgs = self.G(z, labels)
                    g_loss = -torch.mean(self.D(gen_imgs, labels))

                    if torch.isnan(g_loss):
                        raise ValueError("Training diverged: Generator Loss is NaN.")

                    g_loss.backward()
                    self.opt_G.step()
                    g_loss_val = g_loss.item()

                if i % 10 == 0:
                    print(f"[Epoch {epoch+1}/{self.config.epochs}] [Batch {i}] "
                          f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss_val:.4f}]")

            # --- Periodic Visualization & Checkpointing ---
            # 1. Visualization (Progress Images)
            if (epoch + 1) % self.config.sample_interval == 0:
                try:
                    save_fake_slice(
                        self.G, self.fixed_noise, self.fixed_labels, 
                        epoch + 1, output_dir=self.images_dir
                    )
                except Exception as e:
                    # Prevent I/O failures from interrupting the scientific computation
                    print(f"Warning: Progress visualization failed: {e}")

            # 2. Model Checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                gen_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch+1}.pth")
                disc_path = os.path.join(self.checkpoint_dir, f"discriminator_epoch_{epoch+1}.pth")
                
                torch.save(self.G.state_dict(), gen_path)
                torch.save(self.D.state_dict(), disc_path)
                print(f"System checkpoint persisted at epoch {epoch+1}")