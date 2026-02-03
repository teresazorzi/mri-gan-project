# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import torch
import torch.nn as nn

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D, weights_init

# --- INITIALIZATION TESTS ---

def test_weights_initialization_logic():
    """
    Verify that the custom weights_init applies specific Gaussian distributions.
    
    Validates DCGAN-standard initialization (Normal 0, 0.02) to ensure 
    numerical stability in 3D convolutional layers.
    """
    model = CPUOptimizedGenerator3D(latent_dim=64)
    model.apply(weights_init)
    
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose3d):
            assert torch.isclose(m.weight.std(), torch.tensor(0.02), atol=1e-2)
            break

def test_weights_init_with_bias():
    """
    Ensure weights_init correctly zeroes out biases when present.
    
    Validates that layers with bias terms start from a neutral state 
    to facilitate initial optimization steps.
    """
    layer = nn.Conv3d(1, 1, 3, bias=True)
    layer.bias.data.fill_(5.0)
    weights_init(layer)
    assert torch.all(layer.bias == 0)

# --- GENERATOR TESTS ---

def test_generator_output_shape():
    """
    Verify the Generator produces a 5D tensor with correct target dimensions.
    
    Validates the volumetric output format (Batch, Channel, Depth, Height, Width) 
    required for 3D GAN processing.
    """
    batch_size = 2
    latent_dim = 64
    z = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, 3, (batch_size,))
    
    G = CPUOptimizedGenerator3D(latent_dim=latent_dim, target_shape=(64, 64, 64))
    output = G(z, labels)
    
    assert output.shape == (batch_size, 1, 64, 64, 64)

def test_generator_input_validation_full_coverage():
    """
    Verify the Generator enforces structural integrity for latent and label inputs.
    
    Validates strict dimensionality checks to prevent matrix mismatch errors 
    during conditional 3D synthesis.
    """
    gen = CPUOptimizedGenerator3D()
    
    # Case 1: Invalid latent dimensions (2D expectation)
    with pytest.raises(ValueError, match="must be 2D"):
        gen(torch.randn(1, 64, 1), torch.tensor([0]))

    # Case 2: Invalid label dimensions (1D expectation)
    with pytest.raises(ValueError, match="must be 1D"):
        gen(torch.randn(1, 64), torch.tensor([[0]]))

# --- DISCRIMINATOR TESTS ---

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
    
    assert output.shape == (batch_size,)

def test_discriminator_input_validation_full_coverage():
    """
    Verify that the Discriminator rejects malformed image volumes or label tensors.
    
    Ensures the Critic only processes volumetric data consistent with 3D 
    medical imaging standards.
    """
    disc = CPUOptimizedDiscriminator3D()
    
    # Case 1: Invalid image dimensions (5D expectation)
    with pytest.raises(ValueError, match="must be 5D"):
        disc(torch.randn(1, 1, 64, 64), torch.tensor([0]))
        
    # Case 2: Invalid label dimensions (1D expectation)
    with pytest.raises(ValueError, match="must be 1D"):
        disc(torch.randn(1, 1, 64, 64, 64), torch.tensor([[0]]))