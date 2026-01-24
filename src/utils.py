# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import torch
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    """
    Calculates the gradient penalty loss for WGAN-GP.
    Enforces the Lipschitz constraint (gradient norm must be <= 1).
    """
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = D(interpolates, labels)
    
    # Create a tensor of ones for grad_outputs (needed for autograd)
    fake = torch.ones(d_interpolates.shape).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def save_fake_slice(generator, fixed_noise, fixed_labels, epoch, output_dir="results/progress_images"):
    """
    Saves a slice of the generated 3D volume for visualization.
    
    Args:
        generator: The Generator model.
        fixed_noise: Constant noise vector to see progress on same 'seed'.
        fixed_labels: Labels corresponding to the noise.
        epoch: Current epoch number.
        output_dir: Directory where the image will be saved.
    """
    # --- FIX: Ensure directory exists ---
    os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        # Generate volume and move to CPU
        fake_volumes = generator(fixed_noise, fixed_labels).cpu().numpy()
    generator.train()

    # Take the middle slice of the first generated volume
    # Shape is (Batch, Channel, Depth, Height, Width) -> we want (H, W) of the middle Depth
    vol = fake_volumes[0, 0, :, :, :]
    
    # Select the middle slice along the depth axis
    mid_slice_idx = vol.shape[0] // 2
    mid_slice = vol[mid_slice_idx, :, :]

    # Plot and save
    plt.figure(figsize=(5, 5))
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f'Epoch {epoch}')
    plt.axis('off')
    
    filename = os.path.join(output_dir, f"epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()