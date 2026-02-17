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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import CPUOptimizedGenerator3D, CPUOptimizedDiscriminator3D, weights_init

# --- FIXTURES ---

@pytest.fixture
def model_params():
    """
    Provides deterministic parameters and resets seeds for every test.

    Parameters
    ----------
    None : This is a pytest fixture.

    Returns
    -------
    params : dict
        A dictionary containing latent_dim (int), num_classes (int), 
        ngf (int), ndf (int), and the spatial shape (tuple).
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    return {
        "latent_dim": 100,
        "num_classes": 3,
        "ngf": 8, 
        "ndf": 8,
        "shape": (64, 64, 64)
    }

# --- INITIALIZATION TESTS (weights_init) ---

def test_weights_init_modifies_conv_weights():
    """Verify that weights_init effectively changes Conv3d weights from zero.

    GIVEN: A Convolutional 3D layer with weights manually forced to zero.
    WHEN: The weights_init function is applied to the layer.
    THEN: The weight tensor is no longer all zeros.
    """
    layer = nn.Conv3d(1, 10, kernel_size=3)
    nn.init.constant_(layer.weight, 0.0)
    weights_init(layer)
    assert torch.any(layer.weight != 0.0), "Conv3d weights were not modified by weights_init."

def test_weights_init_safety_no_bias():
    """Verify that weights_init handles layers without bias attributes correctly.

    GIVEN: A Conv3d layer initialized with bias=False (bias attribute is None).
    WHEN: The weights_init function is applied.
    THEN: No AttributeError is raised and the bias remains None.
    """
    layer = nn.Conv3d(1, 10, 3, bias=False)
    
    try:
        weights_init(layer)
    except AttributeError as e:
        pytest.fail(f"weights_init failed on a layer with bias=False: {e}")
    
    assert layer.bias is None, "Bias attribute should remain None."

def test_weights_init_zeros_conv_bias():
    """Verify that weights_init sets Conv3d bias to exactly zero when present.

    GIVEN: A Convolutional 3D layer with randomized bias.
    WHEN: The weights_init function is applied to the layer.
    THEN: All bias parameters are initialized to exactly zero.
    """
    layer = nn.Conv3d(1, 10, kernel_size=3)
    nn.init.constant_(layer.bias, 1.0)
    
    weights_init(layer)
    
    assert torch.all(layer.bias == 0), "Conv3d bias should be reset to zero by weights_init."

def test_weights_init_instancenorm_scaling():
    """Verify InstanceNorm3d initialization for affine weights.

    GIVEN: An InstanceNorm3d layer with affine parameters enabled.
    WHEN: The weights_init function is applied to the layer.
    THEN: The weight mean is approximately 1.0, following DCGAN/WGAN-GP standards.
    """
    layer = nn.InstanceNorm3d(10, affine=True)
    nn.init.constant_(layer.weight, 0.0)
    weights_init(layer)
    assert torch.isclose(layer.weight.mean(), torch.tensor(1.0), atol=0.1), \
        f"InstanceNorm weight mean should be ~1.0, got {layer.weight.mean().item()}."

def test_weights_init_ignores_non_parametric_layers():
    """Verify the 'else' branch of weights_init for non-parametric layers.

    GIVEN: A ReLU activation layer which contains no weights or biases.
    WHEN: The weights_init function is applied to the layer.
    THEN: The function completes successfully without modifying the layer.
    """
    layer = nn.ReLU()
    try:
        weights_init(layer)
    except Exception as e:
        pytest.fail(f"weights_init crashed on ReLU with error: {e}")


# --- GENERATOR TESTS ---

def test_generator_output_spatial_dimensions(model_params):
    """Verify the Generator produces the correct 3D spatial shape.

    GIVEN: A Generator instance, a valid latent vector z and a class label.
    WHEN: The forward pass of the Generator is executed.
    THEN: The output tensor matches the expected dimensions (Batch, Channel, D, H, W).
    """
    G = CPUOptimizedGenerator3D(model_params["latent_dim"], model_params["num_classes"], model_params["ngf"], model_params["shape"])
    z, l = torch.randn(1, 100), torch.tensor([0])
    output = G(z, l)
    expected = (1, 1, 64, 64, 64)
    assert output.shape == expected, f"Expected shape {expected}, but got {output.shape}."

def test_generator_output_range_min(model_params):
    """Verify Generator output min value is consistent with Tanh activation.

    GIVEN: A Generator instance and valid random inputs.
    WHEN: A synthetic 3D volume is generated.
    THEN: All intensity values are greater than or equal to -1.0.
    """
    G = CPUOptimizedGenerator3D(model_params["latent_dim"], model_params["num_classes"], model_params["ngf"], model_params["shape"])
    output = G(torch.randn(1, 100), torch.tensor([0]))
    assert output.min() >= -1.0, f"Generator output below -1.0: {output.min()}"

def test_generator_output_range_max(model_params):
    """Verify Generator output max value is consistent with Tanh activation.

    GIVEN: A Generator instance and valid random inputs.
    WHEN: A synthetic 3D volume is generated.
    THEN: All intensity values are smaller than or equal to 1.0.
    """
    G = CPUOptimizedGenerator3D(model_params["latent_dim"], model_params["num_classes"], model_params["ngf"], model_params["shape"])
    output = G(torch.randn(1, 100), torch.tensor([0]))
    assert output.max() <= 1.0, f"Generator output above 1.0: {output.max()}"

def test_generator_determinism_purity(model_params):
    """Verify functional purity: same input must yield identical output.

    GIVEN: A Generator and a fixed pair of (z, labels).
    WHEN: The forward pass is executed two times with the exact same tensors.
    THEN: The resulting synthetic volumes are bitwise identical.
    """
    G = CPUOptimizedGenerator3D(model_params["latent_dim"], model_params["num_classes"], model_params["ngf"], model_params["shape"])
    z, l = torch.randn(1, 100), torch.tensor([0])
    assert torch.equal(G(z, l), G(z, l)), "Generator output is non-deterministic."

def test_generator_raises_error_on_invalid_z_dim(model_params):
    """Verify defensive check for non-2D latent inputs in Generator.

    GIVEN: A latent tensor z with an invalid number of dimensions (e.g., 3D).
    WHEN: The Generator forward pass is called with this tensor.
    THEN: A ValueError is raised with a message containing 'must be 2D'.
    """
    G = CPUOptimizedGenerator3D(model_params["latent_dim"], model_params["num_classes"], model_params["ngf"], model_params["shape"])
    with pytest.raises(ValueError, match="must be 2D"):
        G(torch.randn(1, 100, 1), torch.tensor([0]))

def test_generator_raises_error_on_invalid_label_dim(model_params):
    """Verify defensive check for non-1D label inputs in Generator.

    GIVEN: A label tensor with an invalid number of dimensions (e.g., 2D).
    WHEN: The Generator forward pass is called with this tensor.
    THEN: A ValueError is raised with a message containing 'must be 1D'.
    """
    G = CPUOptimizedGenerator3D(model_params["latent_dim"], model_params["num_classes"], model_params["ngf"], model_params["shape"])
    with pytest.raises(ValueError, match="must be 1D"):
        G(torch.randn(1, 100), torch.tensor([[0]]))

# --- DISCRIMINATOR TESTS ---

def test_discriminator_architecture_constraint(model_params):
    """Verify that the Discriminator rejects unsupported input shapes.

    GIVEN: An input spatial shape (32x32x32) that does not match the 5-layer reduction logic.
    WHEN: The Discriminator is initialized.
    THEN: A ValueError is raised with 'Architecture Constraint' in the message.
    """
    with pytest.raises(ValueError, match="Architecture Constraint"):
        CPUOptimizedDiscriminator3D(3, 8, (32, 32, 32))

def test_discriminator_output_is_1d(model_params):
    """Verify that the Discriminator returns a 1D tensor of scores.

    GIVEN: A valid 5D MRI input volume batch.
    WHEN: The forward pass of the Discriminator is executed.
    THEN: The output tensor has exactly one dimension.
    """
    D = CPUOptimizedDiscriminator3D(model_params["num_classes"], model_params["ndf"], model_params["shape"])
    x = torch.randn(2, 1, 64, 64, 64)
    l = torch.tensor([0, 1])
    output = D(x, l)
    assert output.dim() == 1, f"Discriminator should return 1D tensor, got {output.dim()}D."

def test_discriminator_batch_size_consistency(model_params):
    """Verify that Discriminator produces one score per batch element.

    GIVEN: A valid input batch of size 3.
    WHEN: The forward pass of the Discriminator is executed.
    THEN: The length of the output tensor is equal to the batch size.
    """
    D = CPUOptimizedDiscriminator3D(model_params["num_classes"], model_params["ndf"], model_params["shape"])
    x = torch.randn(3, 1, 64, 64, 64)
    l = torch.tensor([0, 1, 2])
    output = D(x, l)
    assert len(output) == 3, f"Expected 3 scores for batch size 3, got {len(output)}."

def test_discriminator_raises_error_on_invalid_x_dim(model_params):
    """Verify defensive check for non-5D image inputs in Discriminator.

    GIVEN: An input image tensor x with an invalid number of dimensions (e.g., 4D).
    WHEN: The Discriminator forward pass is called with this tensor.
    THEN: A ValueError is raised with a message containing 'must be 5D'.
    """
    D = CPUOptimizedDiscriminator3D(model_params["num_classes"], model_params["ndf"], model_params["shape"])
    with pytest.raises(ValueError, match="must be 5D"):
        D(torch.randn(1, 1, 64, 64), torch.tensor([0]))

def test_discriminator_raises_error_on_invalid_label_dim(model_params):
    """Verify defensive check for non-1D label inputs in Discriminator.

    GIVEN: A label tensor with an invalid number of dimensions (e.g., 2D).
    WHEN: The Discriminator forward pass is called with this tensor.
    THEN: A ValueError is raised with a message containing 'must be 1D'.
    """
    D = CPUOptimizedDiscriminator3D(model_params["num_classes"], model_params["ndf"], model_params["shape"])
    with pytest.raises(ValueError, match="must be 1D"):
        D(torch.randn(1, 1, 64, 64, 64), torch.tensor([[0]]))