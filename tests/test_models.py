# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import torch

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D


# --- TEST GENERATOR ---

def test_generator_output_shape():
    """
    Verify that the Generator produces a 5D tensor with the correct target dimensions.
    
    This ensures the output format (Batch, Channel, Depth, Height, Width) 
    is compatible with the Discriminator and 3D plotting tools.
    """
    batch_size = 2
    latent_dim = 64
    
    z = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, 3, (batch_size,))
    
    G = CPUOptimizedGenerator3D(latent_dim=latent_dim, target_shape=(64, 64, 64))
    output = G(z, labels)
    
    # Verify consistency with (Batch, Channel, D, H, W) format
    assert output.shape == (batch_size, 1, 64, 64, 64)


def test_generator_raises_error_on_wrong_input():
    """
    Verify that the Generator raises ValueError when input dimensions are incorrect.
    
    This tests the defensive programming checks implemented in the forward pass.
    """
    G = CPUOptimizedGenerator3D()
    
    # The generator expects 2D input (Batch, Latent), providing 3D should trigger an error
    wrong_z = torch.randn(2, 64, 1) 
    labels = torch.randint(0, 3, (2,))
    
    with pytest.raises(ValueError):
        G(wrong_z, labels)


# --- TEST DISCRIMINATOR ---

def test_discriminator_output_shape():
    """
    Verify that the Discriminator outputs a scalar score for each image in the batch.
    """
    batch_size = 2
    input_shape = (64, 64, 64)
    
    fake_img = torch.randn(batch_size, 1, *input_shape)
    labels = torch.randint(0, 3, (batch_size,))
    
    D = CPUOptimizedDiscriminator3D(input_shape=input_shape)
    output = D(fake_img, labels)
    
    # The output should be a flat vector of scores (Batch,), one validity score per image
    assert output.shape == (batch_size,)


def test_discriminator_raises_error_on_wrong_input():
    """
    Verify that the Discriminator raises ValueError for inputs with missing dimensions.
    
    This ensures we don't accidentally pass 4D tensors (missing channel dim) 
    which would cause silent broadcasting errors in convolutions.
    """
    D = CPUOptimizedDiscriminator3D()
    
    # Input missing the Channel dimension: (Batch, D, H, W) instead of (Batch, C, D, H, W)
    wrong_img = torch.randn(2, 64, 64, 64) 
    labels = torch.randint(0, 3, (2,))
    
    with pytest.raises(ValueError):
        D(wrong_img, labels)