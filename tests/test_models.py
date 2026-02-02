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
    
    Correct initialization (Normal 0, 0.02) is critical for WGAN stability to 
    prevent gradients from vanishing in the early stages of 3D convolution.
    """
    model = CPUOptimizedGenerator3D(latent_dim=64)
    model.apply(weights_init)
    
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose3d):
            # DCGAN standard: weights must follow N(0, 0.02)
            assert torch.isclose(m.weight.std(), torch.tensor(0.02), atol=1e-2)
            break

def test_weights_init_with_bias():
    """
    Ensure weights_init correctly zeroes out biases when present.
    
    This validates the logic for potential architecture variants where bias 
    terms are enabled, ensuring a neutral starting point for optimization.
    """
    layer = nn.Conv3d(1, 1, 3, bias=True)
    layer.bias.data.fill_(5.0)
    weights_init(layer)
    assert torch.all(layer.bias == 0)

# --- GENERATOR TESTS ---

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
    
    assert output.shape == (batch_size, 1, 64, 64, 64)

def test_generator_input_validation_full_coverage():
    """
    Verify that the Generator enforces structural integrity for both latent and label inputs.
    
    Ensuring strict dimensionality (2D for z, 1D for labels) prevents 
    downstream matrix mismatch errors in the 3D convolution pipeline.
    """
    gen = CPUOptimizedGenerator3D()
    
    # Case 1: Invalid latent dimensions (covers line 112)
    with pytest.raises(ValueError, match="must be 2D"):
        gen(torch.randn(1, 64, 1), torch.tensor([0]))

    # Case 2: Invalid label dimensions (covers line 114)
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
    
    Strict 5D validation ensures the Critic only processes volumetric data 
    consistent with 3D medical imaging standards.
    """
    disc = CPUOptimizedDiscriminator3D()
    
    # Case 1: Invalid image dimensions (covers line 194)
    with pytest.raises(ValueError, match="must be 5D"):
        disc(torch.randn(1, 1, 64, 64), torch.tensor([0]))
        
    # Case 2: Invalid label dimensions (covers line 196)
    with pytest.raises(ValueError, match="must be 1D"):
        disc(torch.randn(1, 1, 64, 64, 64), torch.tensor([[0]]))