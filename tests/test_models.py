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

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D, weights_init


def test_weights_initialization_logic():
    """
    Verify that the custom weights_init applies specific Gaussian distributions.
    
    Correct initialization (Normal 0, 0.02) is critical for WGAN stability to 
    prevent gradients from vanishing in the early stages of 3D convolution.
    """
    model = CPUOptimizedGenerator3D(latent_dim=64)
    model.apply(weights_init)
    
    for m in model.modules():
        if isinstance(m, torch.nn.ConvTranspose3d):
            # DCGAN standard: weights must follow N(0, 0.02)
            assert torch.isclose(m.weight.std(), torch.tensor(0.02), atol=1e-2)
            break

def test_weights_init_with_bias():
    """
    Ensure weights_init correctly zeroes out biases when present.
    """
    layer = torch.nn.Conv3d(1, 1, 3, bias=True)
    layer.bias.data.fill_(5.0)
    weights_init(layer)
    assert torch.all(layer.bias == 0)
        
# --- TEST GENERATOR ---

def test_generator_input_validation():
    """
    Verify that the Generator enforces strict input dimensionality.
    
    This defensive check prevents obscure matrix multiplication errors 
    further down the PyTorch computational graph.
    """
    gen = CPUOptimizedGenerator3D(latent_dim=64)
    z_bad = torch.randn(1, 64, 1, 1) 
    labels = torch.tensor([0])
    
    with pytest.raises(ValueError, match="Generator input 'z' must be 2D"):
        gen(z_bad, labels)

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


def test_generator_raises_error_on_wrong_input():
    """
    Verify that the Generator raises ValueError when input dimensions are incorrect.
    
    This tests the defensive programming checks implemented in the forward pass.
    """
    G = CPUOptimizedGenerator3D()
    wrong_z = torch.randn(2, 64, 1) 
    labels = torch.randint(0, 3, (2,))
    
    with pytest.raises(ValueError):
        G(wrong_z, labels)


# --- TEST DISCRIMINATOR ---

def test_discriminator_input_validation():
    """
    Verify that the Discriminator rejects non-5D volumetric data.
    """
    disc = CPUOptimizedDiscriminator3D()
    x_bad = torch.randn(1, 1, 64, 64) 
    labels = torch.tensor([0])
    
    with pytest.raises(ValueError, match="Discriminator input image must be 5D"):
        disc(x_bad, labels)

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


def test_discriminator_raises_error_on_wrong_input():
    """
    Verify that the Discriminator raises ValueError for inputs with missing dimensions.
    
    This ensures we don't accidentally pass 4D tensors (missing channel dim) 
    which would cause silent broadcasting errors in convolutions.
    """
    D = CPUOptimizedDiscriminator3D()
    wrong_img = torch.randn(2, 64, 64, 64) 
    labels = torch.randint(0, 3, (2,))
    
    with pytest.raises(ValueError):
        D(wrong_img, labels)