# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np

def compute_gradient_penalty(D, real, fake, labels, device):
    """
    Calculates the Gradient Penalty for WGAN-GP.
    Enforces the Lipschitz constraint on the discriminator.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=device)
    
    # Interpolation between real and fake
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    
    fake_output = torch.ones_like(d_interpolates, requires_grad=False)
    
    # Get gradients
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def save_fake_slice(G, config, filename, device="cpu"):
    """
    Generates a sample batch and saves the middle slice as a PNG image.
    Used for monitoring training progress.
    """
    G.eval()
    with torch.no_grad():
        # Generate 3 images (one for each class if classes=3)
        z = torch.randn(3, config['latent_dim'], device=device)
        labels = torch.arange(3, device=device) % 3 # Ensure valid labels
        
        fakes = G(z, labels).squeeze(1).cpu().numpy()
        
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        mid = fakes.shape[2] // 2 
        img = fakes[i, mid, :, :]
        
        # Normalize for visualization [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Class {i}")
        axes[i].axis('off')
    
    plt.savefig(filename)
    plt.close(fig)
    G.train()