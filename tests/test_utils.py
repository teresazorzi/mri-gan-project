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
# allowing imports from 'src' regardless of where the test is run.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import compute_gradient_penalty, save_fake_slice
from src.models import CPUOptimizedDiscriminator3D, CPUOptimizedGenerator3D

# --- FIXTURES ---

@pytest.fixture
def mock_data_device():
    """
    Generate fake batches of data and provide the CPU device.
    This setup isolates the mathematical logic from real data IO.
    """
    batch_size = 2
    
    # 64x64x64 dimensions match the default model configuration
    real = torch.randn(batch_size, 1, 64, 64, 64)
    fake = torch.randn(batch_size, 1, 64, 64, 64)
    labels = torch.randint(0, 3, (batch_size,))
    device = torch.device("cpu")
    
    return real, fake, labels, device

# --- TEST CASES ---

def test_gradient_penalty_returns_scalar(mock_data_device):
    """
    Verify that the gradient penalty calculation returns a valid non-negative scalar.
    This ensures the penalty can be directly added to the scalar discriminator loss.
    """
    real, fake, labels, device = mock_data_device
    
    D = CPUOptimizedDiscriminator3D(input_shape=(64, 64, 64))
    
    gp = compute_gradient_penalty(D, real, fake, labels, device)
    
    assert gp.numel() == 1
    
    # Gradient penalty is a squared norm, so it must mathematically be non-negative
    assert gp.item() >= 0

def test_save_fake_slice_creates_file(tmp_path):
    """
    Verify that the visualization utility successfully generates and saves an image file.
    """
    output_dir = tmp_path / "img_test"
    # Convert path object to string to ensure compatibility with os.path functions inside the utility
    output_dir_str = str(output_dir)
    
    latent_dim = 10
    G = CPUOptimizedGenerator3D(latent_dim=latent_dim, target_shape=(64, 64, 64))
    
    fixed_noise = torch.randn(3, latent_dim)
    fixed_labels = torch.tensor([0, 1, 2])
    
    save_fake_slice(G, fixed_noise, fixed_labels, epoch=99, output_dir=output_dir_str)
    
    expected_file = output_dir / "epoch_99.png"
    assert expected_file.exists()