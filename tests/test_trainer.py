# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import os
import pytest
import torch
import argparse
from unittest.mock import patch, MagicMock
from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D
from src.trainer import Trainer

# --- FIXTURES ---

@pytest.fixture
def trainer_setup(tmp_path):
    """
    Standardize the trainer environment.

    This fixture initializes 3D GAN models and a mock dataloader.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.

    Returns
    -------
    G : CPUOptimizedGenerator3D
        A small generator (ngf=1).
    D : CPUOptimizedDiscriminator3D
        A small discriminator (ndf=1).
    dataloader : list
        A mock dataloader containing a single batch of random data.
    device : torch.device
        The computation device (CPU).
    config : argparse.Namespace
        Configuration object containing training hyperparameters and paths.
    """
    torch.manual_seed(42)
    latent_dim = 10
    num_classes = 3
    target_shape = (64, 64, 64)
    
    # Models with small filters for fast testing
    G = CPUOptimizedGenerator3D(latent_dim, num_classes, ngf=1, target_shape=target_shape)
    D = CPUOptimizedDiscriminator3D(num_classes, ndf=1, input_shape=target_shape)
    
    # Mock DataLoader: one batch of random MRI-like data
    real_batch = torch.randn(2, 1, *target_shape)
    labels = torch.randint(0, num_classes, (2,))
    dataloader = [(real_batch, labels)]
    
    config = argparse.Namespace(
        epochs=1,
        latent_dim=latent_dim,
        num_classes=num_classes,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_dir=str(tmp_path),
        lr=0.0002,
        n_critic=1,
        lambda_gp=10.0,
        device="cpu",
        sample_interval=1,
        checkpoint_interval=1
    )
    
    return G, D, dataloader, torch.device("cpu"), config

# --- LIFECYCLE & PERSISTENCE TESTS  ---

def test_trainer_checkpoint_existence(trainer_setup):
    """Verify that the Generator checkpoint is serialized correctly.

    GIVEN: A Trainer instance configured to execute for one epoch.
    WHEN: The train() method completes its execution cycle.
    THEN: A valid .pth file for the generator exists in the checkpoint directory.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    trainer.train()
    
    ckpt_path = os.path.join(config.checkpoint_dir, "generator_epoch_1.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint was not found at {ckpt_path}"

def test_trainer_checkpoint_non_empty(trainer_setup):
    """Verify that the Generator checkpoint is serialized correctly and not empty.

    GIVEN: A Trainer instance configured to execute for one epoch.
    WHEN: The train() method completes its execution cycle.
    THEN: A valid .pth file exists AND its size is greater than zero bytes.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    trainer.train()
    
    ckpt_path = os.path.join(config.checkpoint_dir, "generator_epoch_1.pth")

    ckpt_size = os.path.getsize(ckpt_path)
    assert ckpt_size > 0, f"Checkpoint file at {ckpt_path} is empty (0 bytes)."


def test_trainer_progress_image_persistence(trainer_setup):
    """Verify that the trainer creates the progress visualization directory.

    GIVEN: A Trainer with a valid sample directory path in its configuration.
    WHEN: The training loop starts.
    THEN: The 'progress_images' subdirectory is created on the filesystem
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)

    expected_dir = os.path.join(config.save_dir, "progress_images")
    assert os.path.exists(expected_dir), "Progress images directory not found."

# --- NUMERICAL STABILITY TESTS ---

def test_trainer_discriminator_nan_error(trainer_setup):
    """Verify the fail-fast mechanism for Discriminator numerical divergence.

    GIVEN: A Discriminator state where weights are corrupted with NaNs.
    WHEN: The Trainer attempts to compute the loss during a training step.
    THEN: A ValueError is raised with a message containing 'Discriminator Loss is NaN'.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    
    for p in D.parameters():
        p.data.fill_(float('nan'))
        
    with pytest.raises(ValueError, match="Discriminator Loss is NaN"):
        trainer.train()

def test_trainer_generator_nan_error(trainer_setup, monkeypatch):
    """Verify the fail-fast mechanism for Generator numerical divergence.

    GIVEN: A Discriminator that returns NaN only during the Generator update.
    WHEN: The training loop reaches the Generator optimization step.
    THEN: A ValueError is raised with the message 'Generator Loss is NaN'.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    
    # 1. mock compute_gradient_penalty to return 0.0 to ensure d_loss is finite and the training loop proceeds to the Generator step.
    monkeypatch.setattr("src.trainer.compute_gradient_penalty", 
                        lambda *args, **kwargs: torch.tensor(0.0, requires_grad=True))

    # 2. Mock the Discriminator's forward method to return 0.0 for the first two calls (d_real and d_fake) and NaN for the third call (g_loss).
    mock_responses = [
        torch.tensor([0.0], requires_grad=True), # d_real
        torch.tensor([0.0], requires_grad=True), # d_fake
        torch.tensor([float('nan')], requires_grad=True) # g_loss 
    ]
    
    # magickmock returns the predefined responses in sequence for each call to D.forward
    mock_forward = MagicMock(side_effect=mock_responses)
    monkeypatch.setattr(D, "forward", mock_forward)
    
    with pytest.raises(ValueError, match="Generator Loss is NaN"):
        trainer.train()


# --- FAULT TOLERANCE TESTS ---

def test_trainer_tolerance_to_io_errors(trainer_setup):
    """Verify that the Trainer is tolerant to non-critical filesystem errors.

    GIVEN: A simulated disk I/O failure during the progress image saving phase.
    WHEN: the external save_fake_slice function raises an Exception.
    THEN: The Trainer catches the exception and continues the training loop.
    """
    G, D, dl, device, config = trainer_setup
    trainer = Trainer(G, D, dl, device, config)
    
    # Patch the image saving utility to simulate an error
    with patch('src.trainer.save_fake_slice') as mock_save:
        mock_save.side_effect = Exception("Simulated disk error")
        # Training should not crash
        trainer.train()
        
    assert mock_save.called, "The Trainer should have attempted to save images despite the failure."
