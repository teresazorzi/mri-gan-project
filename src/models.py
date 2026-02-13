# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(module):
    """
    Custom weights initialization for Generator and Discriminator.
    
    Follows DCGAN guidelines to prevent vanishing/exploding gradients.
    This function is applied recursively to every layer in the network via .apply().
    
    It is designed to be 'selective': it only modifies layers with learnable 
    parameters (weights and biases) and silently ignores all other layers or where 
    parameters are disabled by the user (e.g., bias=False).

    Parameters
    ----------
    module : nn.Module
        The layer being initialized.
    """
    classname = module.__class__.__name__
    
    # Initialize Conv layers with Normal(0, 0.02).
    # This breaks symmetry without introducing large weights.
    if 'Conv' in classname:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    
    # Initialize InstanceNorm: scale=1, bias=0.
    # This ensures the layer acts as an identity function at the start of training.
    elif 'InstanceNorm' in classname:
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    
    # All other layers (ReLU, Sequential, Dropout, etc.) are ignored.
    # We do not raise errors because these layers do not have learnable parameters.
    else:
        pass

class CPUOptimizedGenerator3D(nn.Module):
    """
    3D Generator Network for MRI volume synthesis.
    """
    def __init__(self, latent_dim, num_classes, ngf, target_shape):
        """
        Initialize the Generator.

        Parameters
        ----------
        latent_dim : int
            Dimension of the random noise vector (z).
        num_classes : int
            Number of target classes for conditional generation.
        ngf : int
            Number of generator filters in the last conv layer.
        target_shape : tuple
            Expected output dimensions (D, H, W).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.target_shape = target_shape
        
        self.label_embedding = nn.Embedding(num_classes, latent_dim // 2)
        input_dim = latent_dim + latent_dim // 2
        
        # Reshape flat input vector into a 3D tensor to enable 3D transposed convolutions.
        self.initial = nn.Sequential(
            nn.ConvTranspose3d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.InstanceNorm3d(ngf * 8, affine=True),
            nn.ReLU(True)
        )

        # Progressive resolution doubling (Upsampling) to grow volume from low-res to 64x64x64.
        self.main = nn.Sequential(
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ngf * 4, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ngf * 2, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.InstanceNorm3d(ngf, affine=True),
            nn.ReLU(True),

            nn.ConvTranspose3d(ngf, 1, 4, 2, 1, bias=False),
            # Tanh activation maps output to [-1, 1], matching the preprocessing of real MRI data.
            nn.Tanh() 
        )

    def forward(self, z, labels):
        """
        Forward pass of the Generator.

        Parameters
        ----------
        z : torch.Tensor
            Input latent vector (Batch, Latent).
        labels : torch.Tensor
            Class labels (Batch,).

        Returns
        -------
        torch.Tensor
            Generated 3D volume (Batch, 1, D, H, W).
        
        Raises
        ------
        ValueError
            If 'z' is not 2D or 'labels' is not 1D.
        """
        # Prevent obscure runtime errors by enforcing strict shape requirements early.
        if z.dim() != 2:
            raise ValueError(f"Generator input 'z' must be 2D (Batch, Latent), got shape {z.shape}")
        if labels.dim() != 1:
            raise ValueError(f"Generator input 'labels' must be 1D (Batch,), got shape {labels.shape}")

        emb = self.label_embedding(labels)
        
        # Fuse noise vector with class information (Conditional Generation) before convolution starts.
        x = torch.cat([z, emb], dim=1).view(-1, self.latent_dim + self.latent_dim // 2, 1, 1, 1)
        
        x = self.initial(x)
        x = self.main(x)
        
        # Ensure exact target_shape output, handling potential rounding discrepancies in conv arithmetic.
        return F.interpolate(x, size=self.target_shape, mode='trilinear', align_corners=False)

class CPUOptimizedDiscriminator3D(nn.Module):
    """
    3D Discriminator Network for MRI volume evaluation.
    Implements WGAN-GP logic (InstanceNorm, No Sigmoid).
    """
    def __init__(self, num_classes, ndf, input_shape):
        """
        Initialize the Discriminator.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        ndf : int
            Number of Discriminator Filters in the first layer.
        input_shape : tuple
            Shape of input volume (D, H, W). MUST be (64, 64, 64).

        Raises
        ------
        ValueError
            If input_shape is strictly not (64, 64, 64). 
            The fixed 5-layer architecture mathematically requires this size 
            to reduce dimensions to a scalar score.
        """
        super().__init__()

        # --- ARCHITECTURE CONSTRAINT CHECK ---
        expected_shape = (64, 64, 64)
        if input_shape[-3:] != expected_shape:
            raise ValueError(
                f"Architecture Constraint: Input shape must be exactly {expected_shape}, "
                f"got {input_shape}. \n"
            )

        self.input_shape = input_shape
        
        # Project label embedding to match input image dimensions (D*H*W) for concatenation.
        self.label_embedding = nn.Embedding(num_classes, 32)
        self.label_proj = nn.Linear(32, input_shape[0]*input_shape[1]*input_shape[2])
        
        # Compress spatial information into high-level features for validity assessment.
        self.main = nn.Sequential(
            # Input channels = 2 (Image + Label Map).
            nn.Conv3d(2, ndf, 4, 2, 1), 
            nn.LeakyReLU(0.2),

            nn.Conv3d(ndf, ndf*2, 4, 2, 1), 
            # Use InstanceNorm instead of BatchNorm to maintain sample independence (critical for WGAN-GP).
            nn.InstanceNorm3d(ndf*2), 
            nn.LeakyReLU(0.2),

            nn.Conv3d(ndf*2, ndf*4, 4, 2, 1), 
            nn.InstanceNorm3d(ndf*4), 
            nn.LeakyReLU(0.2),

            nn.Conv3d(ndf*4, ndf*8, 4, 2, 1), 
            nn.InstanceNorm3d(ndf*8), 
            nn.LeakyReLU(0.2),

            # Output raw scalar (criticism score) without Sigmoid, as WGAN minimizes Wasserstein distance.
            nn.Conv3d(ndf*8, 1, 4, 1, 0)
        )

    def forward(self, x, labels):
        """
        Forward pass of the Discriminator.

        Parameters
        ----------
        x : torch.Tensor
            Input image volume (Batch, Channel, D, H, W).
        labels : torch.Tensor
            Class labels (Batch,).

        Returns
        -------
        torch.Tensor
            Validity scores (Batch,).
            
        Raises
        ------
        ValueError
            If 'x' is not 5D or 'labels' is not 1D.
        """
        # Enforce strict shape requirements.
        if x.dim() != 5:
            raise ValueError(f"Discriminator input image must be 5D (Batch, Channel, D, H, W), got shape {x.shape}")
        if labels.dim() != 1:
            raise ValueError(f"Discriminator input 'labels' must be 1D (Batch,), got shape {labels.shape}")

        emb = self.label_embedding(labels)
        lp = self.label_proj(emb).view(-1, 1, *self.input_shape) 
        
        # Inject class information directly into the input space via channel concatenation.
        combined_input = torch.cat([x, lp], dim=1)
        
        return self.main(combined_input).view(-1)