# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os

# Add the project root directory to Python's search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
import numpy as np
import nibabel as nib
from src.dataset import MRINiftiDataset

# --- Creation of temporary data for tests ---

@pytest.fixture
def mock_dataset_folder(tmp_path):
    """Creates a temporary directory with dummy NIfTI files for standard testing."""
    class_dir = tmp_path / "AD"
    class_dir.mkdir()
    
    # Create 2 fake patients
    for i in range(2):
        patient_dir = class_dir / f"Patient_{i:03d}"
        patient_dir.mkdir()
        # Random 3D image
        data = np.random.rand(64, 64, 64).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        # Save with default file name
        nib.save(img, patient_dir / "MPRAGE_MNI_norm.nii.gz")
        
    return str(class_dir)

@pytest.fixture
def custom_file_folder(tmp_path):
    """Creates a folder with weird filenames to test the 'file_pattern' flexibility."""
    d = tmp_path / "T2_Images"
    d.mkdir()
    p = d / "Scan_01"
    p.mkdir()
    
    # Fake image but with a different name
    data = np.random.rand(32, 32, 32).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, p / "brain_t2.nii") # <--- Custom name!
    
    return str(d)

# --- TEST CASES ---

def test_initialization_valid_path(mock_dataset_folder):
    """
    GIVEN a valid folder with default files
    WHEN MRINiftiDataset is initialized
    THEN it should find the files and have correct length.
    """
    dataset = MRINiftiDataset(class_dir=mock_dataset_folder, label=0)
    assert len(dataset) == 2

def test_custom_file_pattern(custom_file_folder):
    """
    GIVEN a folder with non-standard filenames (e.g. 'brain_t2.nii')
    WHEN initialized with file_pattern='brain_t2.nii'
    THEN it should find the files correctly.
    (This proves the code is modality-agnostic!)
    """
    dataset = MRINiftiDataset(class_dir=custom_file_folder, label=1, file_pattern="brain_t2.nii")
    assert len(dataset) == 1
    # Also verify it loads something
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)

def test_raises_error_on_missing_path():
    """
    GIVEN a non-existent path
    WHEN initialized
    THEN it should raise FileNotFoundError (Defensive Programming).
    """
    with pytest.raises(FileNotFoundError):
        MRINiftiDataset(class_dir="percorso/inesistente/assurdo", label=0)

def test_getitem_output_shape(mock_dataset_folder):
    """
    GIVEN a dataset
    WHEN getitem is called
    THEN output tensor should have shape (1, 64, 64, 64).
    """
    dataset = MRINiftiDataset(class_dir=mock_dataset_folder, label=0)
    image, label = dataset[0]
    # Check dimensions: (Channels, Depth, Height, Width)
    assert image.shape == (1, 64, 64, 64)