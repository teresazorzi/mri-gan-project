# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import argparse
import pytest
import torch
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
from src.trainer import Trainer

@pytest.fixture
def trainer_setup(tmp_path):
    """
    Standardize the trainer environment for consistent integration testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.

    Returns
    -------
    tuple
        (Generator, Discriminator, DataLoader, Device, Namespace)
    """
    data = torch.randn(2, 1, 64, 64, 64)
    labels = torch.randint(0, 3, (2,)) 
    dataloader = DataLoader(TensorDataset(data, labels), batch_size=2)
    
    G = CPUOptimizedGenerator3D(latent_dim=10, target_shape=(64, 64, 64))
    D = CPUOptimizedDiscriminator3D(input_shape=(64, 64, 64))
    device = torch.device("cpu")
    
    config = argparse.Namespace(
        lr=0.001,
        epochs=1,
        latent_dim=10,
        n_critic=1,
        batch_size=2,
        save_dir=str(tmp_path)
    )
    
    return G, D, dataloader, device, config

def test_trainer_integration_and_checkpoints(tmp_path, trainer_setup):
    """
    Verify the full training lifecycle, including checkpoint persistence.
    
    Validates that the periodic model serialization logic correctly interacts 
    with the filesystem at the specified epoch intervals.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Fixture for verifying file creation on disk.
    trainer_setup : tuple
        Standardized training components.
    """
    G, D, dl, device, config = trainer_setup
    config.epochs = 10 
    
    trainer = Trainer(G, D, dl, device, config)
    trainer.train()
    
    checkpoint_path = tmp_path / "checkpoints" / "generator_epoch_10.pth"
    assert checkpoint_path.exists()
    assert (tmp_path / "progress_images").exists()

def test_trainer_discriminator_divergence_guard(trainer_setup):
    """
    Verify that the Trainer halts execution if the Discriminator diverges.
    
    Validates the fail-fast mechanism that monitors numerical stability 
    within the Critic's loss calculation to prevent resource exhaustion.

    Parameters
    ----------
    trainer_setup : tuple
        Standardized training components.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    
    for p in D.parameters():
        p.data.fill_(float('nan'))
        
    with pytest.raises(ValueError, match="Discriminator Loss"):
        trainer.train()

def test_trainer_generator_nan_specific(trainer_setup, monkeypatch):
    """
    Verify the Generator's specific safety branch for numerical divergence.
    
    Utilizes a stateful mock to isolate the Generator's loss calculation 
    from the Discriminator's update phase, ensuring targeted exception handling.

    Parameters
    ----------
    trainer_setup : tuple
        Standardized training components.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Pytest fixture for runtime attribute modification.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    
    call_state = {"count": 0}
    
    def smart_mock_forward(*args, **kwargs):
        call_state["count"] += 1
        if call_state["count"] > 2:
            return torch.tensor([float('nan')], requires_grad=True).to(device)
        return torch.tensor([0.0], requires_grad=True).to(device)

    monkeypatch.setattr(D, "forward", smart_mock_forward)
    monkeypatch.setattr("src.trainer.compute_gradient_penalty", lambda *a, **k: torch.tensor(0.0))

    with pytest.raises(ValueError, match="Generator Loss is NaN"):
        trainer.train()

def test_trainer_save_image_exception_handling(trainer_setup):
    """
    Verify Trainer robustness against non-critical filesystem failures.
    
    Simulates I/O exceptions during progress visualization to ensure that 
    the core training loop remains resilient to peripheral errors.

    Parameters
    ----------
    trainer_setup : tuple
        Standardized training components.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    
    with patch('src.trainer.save_fake_slice') as mock_save:
        mock_save.side_effect = Exception("Simulated disk error")
        trainer.train()
        
    mock_save.assert_called()