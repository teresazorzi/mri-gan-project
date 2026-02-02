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

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
from src.trainer import Trainer


class MockConfig:
    """
    A simple configuration object to simulate command line arguments.
    """
    def __init__(self, save_dir):
        """
        Initialize the mock configuration.

        Parameters
        ----------
        save_dir : str
            Path where the trainer should output results.
        """
        self.lr = 0.001
        self.epochs = 1        # Run only 1 epoch to keep tests fast
        self.latent_dim = 10   # Small latent dim for efficiency
        self.n_critic = 1      # Update G every step to maximize code coverage
        self.batch_size = 2
        self.save_dir = save_dir # Crucial: Trainer expects this attribute


@pytest.fixture
def mock_dataloader():
    """
    Create a dataloader with random fake MRI data.
    
    Returns
    -------
    DataLoader
        A PyTorch DataLoader serving random tensors of shape (Batch, 1, 64, 64, 64).
    """
    # Create fake 3D images: (Batch=4, Channel=1, D=64, H=64, W=64)
    data = torch.randn(4, 1, 64, 64, 64)
    labels = torch.randint(0, 3, (4,)) 
    
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=2)


@pytest.fixture
def models():
    """
    Initialize fresh, small-scale models for testing.

    Returns
    -------
    tuple
        (Generator, Discriminator) initialized on CPU.
    """
    # Initialize lightweight models for testing speed
    G = CPUOptimizedGenerator3D(latent_dim=10, target_shape=(64, 64, 64))
    D = CPUOptimizedDiscriminator3D(input_shape=(64, 64, 64))
    return G, D


def test_trainer_runs_without_crash(tmp_path, mock_dataloader, models):
    """
    Verify that the training loop runs for one epoch without errors.

    This acts as an integration test for the Trainer class, ensuring
    it correctly interacts with the models, dataloader, and file system.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a clean temporary directory.
    mock_dataloader : DataLoader
        Fixture providing fake data.
    models : tuple
        Fixture providing (G, D) models.
    """
    generator, discriminator = models
    
    # Configure the trainer to write into the temporary test directory
    # to avoid polluting the project root with test artifacts.
    config = MockConfig(save_dir=str(tmp_path))
    
    # Force CPU to ensure tests pass on environments without GPUs (CI/CD)
    device = torch.device("cpu") 
    
    trainer = Trainer(generator, discriminator, mock_dataloader, device, config)

    trainer.train()
    
    # Verify that the expected directory structure was created.
    # The Trainer should create 'progress_images' inside the save_dir.
    expected_img_dir = tmp_path / "progress_images"
    assert expected_img_dir.exists(), "Trainer failed to create progress_images directory."


def test_trainer_handles_nan_loss(tmp_path, mock_dataloader, models):
    """
    Verify that the Trainer raises a ValueError if training diverges (NaN loss).

    This tests the defensive programming logic inside the training loop.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture.
    mock_dataloader : DataLoader
        Fixture providing fake data.
    models : tuple
        Fixture providing (G, D) models.
    """
    generator, discriminator = models
    config = MockConfig(save_dir=str(tmp_path))
    device = torch.device("cpu")
    
    trainer = Trainer(generator, discriminator, mock_dataloader, device, config)
    
    # Sabotage the discriminator weights by filling them with NaN.
    # This guarantees that the forward pass will output NaN, triggering the safety check.
    for p in discriminator.parameters():
        p.data.fill_(float('nan'))
        
    with pytest.raises(ValueError, match="NaN"):
        trainer.train()