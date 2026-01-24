# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import torch
import shutil

# Fix imports path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import compute_gradient_penalty, save_fake_slice
from src.models import CPUOptimizedDiscriminator3D, CPUOptimizedGenerator3D

# --- FIXTURES ---

@pytest.fixture
def mock_data_device():
    """Returns fake data and cpu device."""
    batch_size = 2
    # Fake images 64x64x64 to be fast
    real = torch.randn(batch_size, 1, 64, 64, 64)
    fake = torch.randn(batch_size, 1, 64, 64, 64)
    labels = torch.randint(0, 3, (batch_size,))
    device = torch.device("cpu")
    return real, fake, labels, device

# --- TEST CASES ---

def test_gradient_penalty_returns_scalar(mock_data_device):
    """
    GIVEN real and fake samples
    WHEN compute_gradient_penalty is called
    THEN it should return a single scalar value (0-dim tensor).
    """
    real, fake, labels, device = mock_data_device
    
    # We need a discriminator to pass to the function
    D = CPUOptimizedDiscriminator3D(input_shape=(64, 64, 64))
    
    gp = compute_gradient_penalty(D, real, fake, labels, device)
    
    # Check it is a scalar (dimension 0) or size 1
    assert gp.numel() == 1
    assert gp.item() >= 0 # Gradient penalty must be positive

def test_save_fake_slice_creates_file(tmp_path):
    """
    GIVEN a generator and a destination folder
    WHEN save_fake_slice is called
    THEN a PNG file should be created in that folder.
    """
    # Setup
    output_dir = tmp_path / "img_test"
    # We pass the string path to the function
    output_dir_str = str(output_dir)
    
    latent_dim = 10
    G = CPUOptimizedGenerator3D(latent_dim=latent_dim, target_shape=(64, 64, 64))
    
    fixed_noise = torch.randn(3, latent_dim)
    fixed_labels = torch.tensor([0, 1, 2])
    
    # Execution
    save_fake_slice(G, fixed_noise, fixed_labels, epoch=99, output_dir=output_dir_str)
    
    # Check
    expected_file = output_dir / "epoch_99.png"
    assert expected_file.exists()