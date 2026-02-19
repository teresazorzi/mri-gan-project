# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import torch
import numpy as np
import random
import matplotlib
import gc
from unittest.mock import patch

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import compute_gradient_penalty, save_fake_slice
from src.models import CPUOptimizedDiscriminator3D, CPUOptimizedGenerator3D

# --- FIXTURES ---

@pytest.fixture
def real_models_and_data():
    """
    Initialize model instances and data compatible with the 3D architecture.
    
    To keep tests fast despite the large spatial size, the channel depth is 
    minimized using ngf=1 and ndf=1.

    Parameters
    ----------
    None : This is a pytest fixture.

    Returns
    -------
    G : CPUOptimizedGenerator3D
        The initialized Generator model.
    D : CPUOptimizedDiscriminator3D
        The initialized Discriminator model.
    real_imgs : torch.Tensor
        A batch of simulated real MRI volumes (Batch, 1, 64, 64, 64).
    fake_imgs : torch.Tensor
        A batch of simulated fake MRI volumes.
    labels : torch.Tensor
        A 1D tensor of class labels.
    device : torch.device
        the computation device (CPU).
    """
    # 1. Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 2. Configuration matching models.py constraints
    # Discriminator architecture implies 64->32->16->8->4->1 logic.
    img_shape = (64, 64, 64) 
    batch_size = 2
    latent_dim = 10
    num_classes = 3
    ngf = 1 
    ndf = 1 
    
    device = torch.device("cpu")

    # Generator signature: (latent_dim, num_classes, ngf, target_shape)
    G = CPUOptimizedGenerator3D(
        latent_dim=latent_dim, 
        num_classes=num_classes, 
        ngf=ngf, 
        target_shape=img_shape
    )
    
    # Discriminator signature: (num_classes, ndf, input_shape)
    D = CPUOptimizedDiscriminator3D(
        num_classes=num_classes, 
        ndf=ndf, 
        input_shape=img_shape
    )
    
    # 4. Generate compatible data
    # Real images must be (Batch, Channel, D, H, W) -> Channel is 1 for MRI
    real_imgs = torch.randn(batch_size, 1, *img_shape)
    fake_imgs = torch.randn(batch_size, 1, *img_shape)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    yield G, D, real_imgs, fake_imgs, labels, device

    # Cleanup
    del G
    del D
    del real_imgs
    del fake_imgs
    del labels
    del device

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()


# --- GRADIENT PENALTY TESTS ---

def test_gradient_penalty_output_type(real_models_and_data):
    """Verify that the gradient penalty function returns a torch Tensor.

    GIVEN: A real Discriminator architecture and a batch of real and fake data.
    WHEN: The compute_gradient_penalty function is called.
    THEN: The returned object is an instance of torch.Tensor.
    """
    _, D, real, fake, labels, device = real_models_and_data
    gp = compute_gradient_penalty(D, real, fake, labels, device)
    
    assert isinstance(gp, torch.Tensor), "Gradient penalty must be a torch.Tensor"

def test_gradient_penalty_is_scalar(real_models_and_data):
    """Verify that the gradient penalty is a scalar value.

    GIVEN: Valid discriminator and data batches.
    WHEN: The gradient penalty is computed.
    THEN: The resulting tensor has 0 dimensions.
    """
    _, D, real, fake, labels, device = real_models_and_data
    gp = compute_gradient_penalty(D, real, fake, labels, device)
    
    assert gp.dim() == 0, f"Expected a scalar (0-dim), but got a {gp.dim()}-dim tensor."

def test_gradient_penalty_graph_attachment(real_models_and_data):
    """Verify that the gradient penalty maintains the computational graph.

    GIVEN: A Discriminator and data requiring gradients.
    WHEN: compute_gradient_penalty is executed.
    THEN: The output tensor has a grad_fn, allowing for backpropagation.
    """
    _, D, real, fake, labels, device = real_models_and_data
    gp = compute_gradient_penalty(D, real, fake, labels, device)
    
    assert gp.grad_fn is not None, "Gradient penalty is detached from the graph; backprop would fail."

def test_gradient_penalty_non_negative(real_models_and_data):
    """Verify that the gradient penalty value is never negative.

    GIVEN: Real and fake MRI volumes.
    WHEN: compute_gradient_penalty is called.
    THEN: The penalty value is greater than or equal to 0.0.
    """
    _, D, real, fake, labels, device = real_models_and_data
    gp = compute_gradient_penalty(D, real, fake, labels, device)
    
    assert gp.item() >= 0.0, f"Expected non-negative penalty, but got {gp.item()}."


# --- SAVE_FAKE_SLICE TESTS ---

def test_save_fake_slice_file_creation(real_models_and_data, tmp_path):
    """Verify that save_fake_slice creates a PNG file.

    GIVEN: A generator and a temporary output directory.
    WHEN: save_fake_slice is executed for a specific epoch.
    THEN: A file named 'epoch_1.png' exists in the directory.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "creation_test"
    noise = torch.randn(len(labels), G.latent_dim)
    
    save_fake_slice(G, noise, labels, epoch=5, output_dir=str(out_dir))
    
    output_file = out_dir / "epoch_5.png"
    assert output_file.exists(), "The output PNG file was not created."

def test_save_fake_slice_single_sample_logic(real_models_and_data, tmp_path):
    """Verify that save_fake_slice handles a batch of size 1 correctly.

    GIVEN: A noise vector and label for a single sample.
    WHEN: save_fake_slice is executed.
    THEN: The function creates a PNG file for the single sample.
    """
    G, _, _, _, _, _ = real_models_and_data
    out_dir = tmp_path / "single_sample"
    noise = torch.randn(1, G.latent_dim)
    label = torch.tensor([0])
    
    save_fake_slice(G, noise, label, 1, str(out_dir))
    assert (out_dir / "epoch_1.png").exists(), "Failed to create PNG for single sample batch."

def test_save_fake_slice_determinism(real_models_and_data, tmp_path):
    """Verify that identical inputs produce identical image files.

    GIVEN: The same noise vector and generator state.
    WHEN: Generating slices twice for two different epochs.
    THEN: The resulting PNG files are identical in their byte content.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "determinism_test"
    noise = torch.randn(len(labels), G.latent_dim)
    
    save_fake_slice(G, noise, labels, epoch=1, output_dir=str(out_dir))
    save_fake_slice(G, noise, labels, epoch=2, output_dir=str(out_dir))
    
    with open(out_dir / "epoch_1.png", "rb") as f1, open(out_dir / "epoch_2.png", "rb") as f2:
        assert f1.read() == f2.read(), "Images differ despite identical noise."

def test_save_fake_slice_mode_restoration(real_models_and_data, tmp_path):
    """Verify that the generator is restored to training mode after execution.

    GIVEN: A generator explicitly set to .train() mode.
    WHEN: save_fake_slice is called (which uses .eval() internally).
    THEN: The generator's .training attribute is True after the function returns.
    """
    G, _, _, _, labels, _ = real_models_and_data
    G.train()
    
    save_fake_slice(G, torch.randn(len(labels), G.latent_dim), labels, 1, str(tmp_path))
    
    assert G.training is True, "The generator was not restored to training mode."

def test_save_fake_slice_directory_auto_creation(tmp_path):
    """Verify that the function automatically creates non-existent directories.

    GIVEN: A path to a nested directory that does not yet exist.
    WHEN: save_fake_slice is called with that path.
    THEN: The directory is created successfully.
    """
    G = CPUOptimizedGenerator3D(10, 2, 1, (64, 64, 64))
    nested_dir = tmp_path / "automatic" / "subfolder"
    
    save_fake_slice(G, torch.randn(1, 10), torch.tensor([0]), 1, str(nested_dir))
    
    assert nested_dir.exists(), "The output directory was not automatically created."

def test_save_fake_slice_noise_sensitivity(real_models_and_data, tmp_path):
    """Verify that different noise vectors produce different images.

    GIVEN: Two distinct noise vectors (ones vs zeros).
    WHEN: Slices are saved for each.
    THEN: The resulting PNG files are not identical.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "noise_sensitivity_test"
    n1, n2 = torch.ones(len(labels), G.latent_dim), torch.zeros(len(labels), G.latent_dim)
    
    save_fake_slice(G, n1, labels, epoch=1, output_dir=str(out_dir))
    save_fake_slice(G, n2, labels, epoch=2, output_dir=str(out_dir))
    
    with open(out_dir / "epoch_1.png", "rb") as f1, open(out_dir / "epoch_2.png", "rb") as f2:
        assert f1.read() != f2.read(), "Generator produced identical images for different noise inputs."

def test_save_fake_slice_file_size_is_positive(real_models_and_data, tmp_path):
    """Verify that the generated PNG file is not empty.

    GIVEN: A valid generator and output parameters.
    WHEN: save_fake_slice is executed.
    THEN: The resulting file has a size greater than 0 bytes.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "size_test"
    noise = torch.randn(len(labels), G.latent_dim)
    
    save_fake_slice(G, noise, labels, epoch=1, output_dir=str(out_dir))
    
    output_file = out_dir / "epoch_1.png"
    assert output_file.stat().st_size > 0, "The generated PNG file is empty (0 bytes)."

def test_save_fake_slice_calls_generator_once(real_models_and_data, tmp_path):
    """Verify that the generator is explicitly called.

    GIVEN: A generator and valid noise/label tensors.
    WHEN: save_fake_slice is executed.
    THEN: The generator's forward method is called exactly one time.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "call_test"
    noise = torch.randn(len(labels), G.latent_dim)
    
    with patch.object(G, 'forward', wraps=G.forward) as mock_forward:
        save_fake_slice(G, noise, labels, 1, str(out_dir))
        mock_forward.assert_called_once()
        call_count = mock_forward.call_count
        assert call_count == 1, f"Expected generator to be called exactly once, but it was called {call_count} times."

def test_save_fake_slice_passes_correct_noise(real_models_and_data, tmp_path):
    """Verify that the exact noise tensor is passed to the generator.

    GIVEN: A generator and a specific noise tensor.
    WHEN: save_fake_slice is executed.
    THEN: The noise received by the generator matches the input noise.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "noise_test"
    noise = torch.randn(len(labels), G.latent_dim)
    
    with patch.object(G, 'forward', wraps=G.forward) as mock_forward:
        save_fake_slice(G, noise, labels, 1, str(out_dir))
        
        args, _ = mock_forward.call_args
        passed_noise = args[0]
        
        assert torch.equal(passed_noise, noise), "The provided noise was modified or not passed to the Generator."

def test_save_fake_slice_passes_correct_labels(real_models_and_data, tmp_path):
    """Verify that the exact label tensor is passed to the generator.

    GIVEN: A generator and specific class labels.
    WHEN: save_fake_slice is executed.
    THEN: The labels received by the generator match the input labels.
    """
    G, _, _, _, labels, _ = real_models_and_data
    out_dir = tmp_path / "labels_test"
    noise = torch.randn(len(labels), G.latent_dim)
    
    with patch.object(G, 'forward', wraps=G.forward) as mock_forward:
        save_fake_slice(G, noise, labels, 1, str(out_dir))
        
        args, _ = mock_forward.call_args
        passed_labels = args[1]
        
        assert torch.equal(passed_labels, labels), "The provided labels were modified or not passed to the Generator."