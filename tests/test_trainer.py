# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Fix imports path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
from src.trainer import Trainer

# --- MOCK OBJECTS (Fake Objects for Testing) ---

class MockConfig:
    """A simple configuration object to simulate command line arguments."""
    def __init__(self):
        self.lr = 0.001
        self.epochs = 1        # Run only 1 epoch for testing
        self.latent_dim = 10   # Small latent dim
        self.n_critic = 1      # Update G every step to speed up test
        self.batch_size = 2

@pytest.fixture
def mock_dataloader():
    """Creates a dataloader with random fake MRI data."""
    # Create fake 3D images: (Batch=4, Channel=1, D=64, H=64, W=64)
    data = torch.randn(4, 1, 64, 64, 64)
    labels = torch.randint(0, 3, (4,)) # Random labels 0-2
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=2)

@pytest.fixture
def models():
    """Initialize fresh models for testing."""
    G = CPUOptimizedGenerator3D(latent_dim=10, target_shape=(64, 64, 64))
    D = CPUOptimizedDiscriminator3D(input_shape=(64, 64, 64))
    return G, D

# --- TEST CASES ---

def test_trainer_runs_without_crash(tmp_path, mock_dataloader, models):
    """
    GIVEN a Trainer setup with fake data and models
    WHEN we run train() for 1 epoch
    THEN it should complete without errors and produce a checkpoint.
    """
    generator, discriminator = models
    config = MockConfig()
    
    # We change the working directory to tmp_path so 'results/' are created there and don't pollute the real project folder.
    os.chdir(tmp_path) 
    
    device = torch.device("cpu") # Force CPU for testing 
    
    trainer = Trainer(generator, discriminator, mock_dataloader, device, config)

    trainer.train()
    
    # ASSERTIONS
    
    # 1. Check if "results" folder was created
    assert os.path.exists("results")
    
    # 2. Check if progress images were saved (Since config.epochs=1, it might execute save_fake_slice once)
    assert os.path.exists(os.path.join("results", "progress_images"))
    
    # Note: Checkpoints are saved every 10 epochs in the original code, so with 1 epoch we won't find a .pth file, but the fact that 
    # trainer.train() finished without errors is the real test here.

def test_trainer_handles_nan_loss(mock_dataloader, models):
    """
    GIVEN a Trainer
    WHEN a NaN loss occurs (simulated)
    THEN it should raise a ValueError (Defensive Programming).
    """
    generator, discriminator = models
    config = MockConfig()
    device = torch.device("cpu")
    
    trainer = Trainer(generator, discriminator, mock_dataloader, device, config)
    
    # Trick: force the discriminator weights to be NaN
    for p in discriminator.parameters():
        p.data.fill_(float('nan'))
        
    # When we run, D(real) will be NaN, causing d_loss to be NaN
    with pytest.raises(ValueError, match="NaN"):
        trainer.train()