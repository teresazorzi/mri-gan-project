# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import os
import torch
import torch.autograd as autograd
import matplotlib.pyplot as plt

def compute_gradient_penalty(D, real_samples, fake_samples, labels, device):
    """
    Calculates the gradient penalty loss for WGAN-GP.

    This penalizes the norm of the gradient of the critic with respect to its input.
    It enforces the 1-Lipschitz constraint required for the Wasserstein metric to be valid.

    Parameters
    ----------
    D : nn.Module
        The Discriminator (Critic) network.
    real_samples : torch.Tensor
        Batch of real images.
    fake_samples : torch.Tensor
        Batch of generated images.
    labels : torch.Tensor
        Class labels for conditional generation.
    device : torch.device
        Device to perform computations on (cpu or cuda).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the gradient penalty term.
    """
    # Random weight term 'alpha' for interpolation.
    # We sample random points along the straight lines between real and fake pairs.
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(device)

    # Get random interpolation between real and fake samples.
    # requires_grad_(True) is essential here because we need to differentiate 
    # the Critic's output w.r.t. these interpolated images.
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates, labels)

    # Dummy tensor of ones.
    # Autograd requires a 'grad_outputs' argument matching the shape of 'outputs'
    # to compute gradients for non-scalar outputs.
    fake = torch.ones(d_interpolates.shape).to(device)

    # Calculate gradients of the Critic's scores w.r.t. the interpolated images.
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,   # Required for higher-order derivatives (since we differentiate the gradient itself)
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Flatten the gradients to (Batch_Size, Feature_Dim) to compute the norm easily.
    gradients = gradients.view(gradients.size(0), -1)
    
    # The penalty is the mean squared distance of the gradient norm from 1.
    # This softly enforces the Lipschitz constraint.
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def save_fake_slice(generator, fixed_noise, fixed_labels, epoch, output_dir):
    """
    Generate a sample volume and save a 2D middle slice for visualization.

    Parameters
    ----------
    generator : nn.Module
        The Generator model.
    fixed_noise : torch.Tensor
        Constant noise vector to visualize progress on the same latent 'seed'.
    fixed_labels : torch.Tensor
        Labels corresponding to the noise vector.
    epoch : int
        Current epoch number (used for filename).
    output_dir : str
        Directory path where the image will be saved.
    """
    # Ensure the target directory exists before saving to prevent IOErrors.
    os.makedirs(output_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        # Generate volume and move to CPU (NumPy requires CPU tensors).
        fake_volumes = generator(fixed_noise, fixed_labels).cpu().numpy()
    generator.train()

    # Extract the number of samples in the batch to determine how many subplots we need.
    num_samples = fake_volumes.shape[0]
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        # Extract the i-th volume
        vol = fake_volumes[i, 0, :, :, :]
        # Select the middle slice along the depth axis (Coronal/Axial view depending on orientation).
        mid_slice = vol[vol.shape[0] // 2, :, :]
        
        axes[i].imshow(mid_slice, cmap='gray')
        axes[i].set_title(f'Class {fixed_labels[i].item()}')
        axes[i].axis('off')

    plt.tight_layout()
    filename = os.path.join(output_dir, f"epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()