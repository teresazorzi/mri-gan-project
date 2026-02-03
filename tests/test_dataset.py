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
import nibabel as nib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import MRINiftiDataset

# --- FIXTURES ---

@pytest.fixture
def mock_dataset_folder(tmp_path):
    """
    Create a temporary directory with dummy NIfTI files for standard testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.

    Returns
    -------
    str
        The absolute path to the temporary data directory.
    """
    class_dir = tmp_path / "AD"
    class_dir.mkdir()
    
    for i in range(2):
        patient_dir = class_dir / f"Patient_{i:03d}"
        patient_dir.mkdir()
        
        data = np.random.rand(64, 64, 64).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, patient_dir / "MPRAGE_MNI_norm.nii.gz")
        
    return str(class_dir)

@pytest.fixture
def custom_file_folder(tmp_path):
    """
    Create a folder with non-standard filenames to verify modality-agnosticism.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.

    Returns
    -------
    str
        The absolute path to the temporary custom data directory.
    """
    d = tmp_path / "T2_Images"
    d.mkdir()
    p = d / "Scan_01"
    p.mkdir()
    
    data = np.random.rand(32, 32, 32).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, p / "brain_t2.nii") 
    
    return str(d)

# --- TEST CASES ---

def test_initialization_valid_path(mock_dataset_folder):
    """
    Verify that the dataset correctly locates and counts files in a standard structure.

    Parameters
    ----------
    mock_dataset_folder : str
        Path to the temporary directory provided by the fixture.
    """
    dataset = MRINiftiDataset(class_dir=mock_dataset_folder, label=0)
    assert len(dataset) == 2

def test_custom_file_pattern(custom_file_folder):
    """
    Verify that the dataset can load files with custom patterns for flexible modalities.

    Parameters
    ----------
    custom_file_folder : str
        Path to the temporary custom directory provided by the fixture.
    """
    dataset = MRINiftiDataset(class_dir=custom_file_folder, label=1, file_pattern="brain_t2.nii")
    assert len(dataset) == 1
    
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)

def test_raises_error_on_missing_path():
    """
    Verify that initializing with a non-existent path raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        MRINiftiDataset(class_dir="invalid/path/test", label=0)

def test_data_shape_consistency(mock_dataset_folder):
    """
    Verify that the output tensor dimensions match the expected (1, D, H, W) format.

    Parameters
    ----------
    mock_dataset_folder : str
        Path to the temporary directory provided by the fixture.
    """
    dataset = MRINiftiDataset(class_dir=mock_dataset_folder, label=0)
    image, label = dataset[0]
    
    assert image.shape == (1, 64, 64, 64)

def test_dataset_handles_flat_images(tmp_path):
    """
    Verify normalization robustness against blank scans to prevent division by zero.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.
    """
    flat_dir = tmp_path / "FlatData"
    flat_dir.mkdir()
    patient_dir = flat_dir / "Patient_Empty"
    patient_dir.mkdir()
    
    data = np.zeros((64, 64, 64), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, patient_dir / "MPRAGE_MNI_norm.nii.gz")
    
    dataset = MRINiftiDataset(class_dir=str(flat_dir), label=0)
    image, _ = dataset[0]
    
    assert torch.all(image == -1)

def test_dataset_fault_tolerance_on_corruption(tmp_path):
    """
    Verify robustness against corrupted NIfTI headers through safe fallback logic.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.
    """
    corrupt_dir = tmp_path / "CorruptData"
    corrupt_dir.mkdir()
    bad_file = corrupt_dir / "invalid_scan.nii.gz"
    bad_file.write_text("This is not a valid NIfTI header.")
    
    dataset = MRINiftiDataset(class_dir=str(corrupt_dir), label=0, file_pattern="invalid_scan.nii.gz")
    
    image, _ = dataset[0]
    
    assert torch.all(image == 0)
    assert image.shape == (1, 64, 64, 64)

def test_dataset_warning_on_empty_folder(tmp_path):
    """
    Verify dataset identification of empty search results based on specific patterns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Built-in pytest fixture for temporary directory management.
    """
    empty_dir = tmp_path / "Empty"
    empty_dir.mkdir()
    
    dataset = MRINiftiDataset(class_dir=str(empty_dir), label=0, file_pattern="*.txt")
    
    assert len(dataset) == 0