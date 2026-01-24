# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import torch

# Fix imports path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D

# --- TEST GENERATOR ---

def test_generator_output_shape():
    """
    GIVEN a Generator
    WHEN passed a latent vector z (Batch, 64)
    THEN it should output a 5D tensor (Batch, 1, 64, 64, 64).
    """
    batch_size = 2
    latent_dim = 64
    z = torch.randn(batch_size, latent_dim)
    labels = torch.randint(0, 3, (batch_size,))
    
    G = CPUOptimizedGenerator3D(latent_dim=latent_dim)
    output = G(z, labels)
    
    assert output.shape == (batch_size, 1, 64, 64, 64)

def test_generator_raises_error_on_wrong_input():
    """
    GIVEN a Generator
    WHEN passed a 3D tensor instead of 2D for z
    THEN it should raise ValueError.
    """
    G = CPUOptimizedGenerator3D()
    wrong_z = torch.randn(2, 64, 1) # Error: 3 Dimensions
    labels = torch.randint(0, 3, (2,))
    
    with pytest.raises(ValueError):
        G(wrong_z, labels)

# --- TEST DISCRIMINATOR ---

def test_discriminator_raises_error_on_wrong_input():
    """
    GIVEN a Discriminator
    WHEN passed a 4D tensor (missing channel dim)
    THEN it should raise ValueError.
    """
    D = CPUOptimizedDiscriminator3D()
    wrong_img = torch.randn(2, 64, 64, 64) # Error: Missing channel
    labels = torch.randint(0, 3, (2,))
    
    with pytest.raises(ValueError):
        D(wrong_img, labels)