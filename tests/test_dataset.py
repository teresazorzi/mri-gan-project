# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import os
import pytest
import numpy as np
import nibabel as nib
import torch
import random
from src.dataset import MRINiftiDataset, find_mri_files

# --- FIXTURES ---

@pytest.fixture
def mock_mri_root(tmp_path):
    """
    Create a temporary recursive directory structure with deterministic dummy NIfTI files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory.

    Returns
    -------
    root_dir : str
        The absolute path to the root directory containing mock MRI data for AD class.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    root_dir = tmp_path / "data_root"
    class_dir = root_dir / "AD"
    class_dir.mkdir(parents=True)
    
    shapes = [(64, 64, 64), (40, 48, 56)]
    for i, shape in enumerate(shapes):
        patient_dir = class_dir / f"Patient_{i:03d}"
        patient_dir.mkdir()
        data = np.random.rand(*shape).astype(np.float32) * 1000
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, str(patient_dir / "MPRAGE_MNI_norm.nii.gz"))
        
    return str(root_dir)

# --- TEST CASES ---

#  SORTING TESTS
def test_find_mri_files_is_sorted(tmp_path):
    """Verify that the discovery logic returns a list that is sorted.

    GIVEN: A directory with files created in non-alphabetical order.
    WHEN: find_mri_files is executed.
    THEN: The returned list is exactly equal to its sorted version.
    """
    (tmp_path / "z_scan.nii.gz").write_text("dummy")
    (tmp_path / "a_scan.nii.gz").write_text("dummy")
    
    results = find_mri_files(str(tmp_path), "*.nii.gz")
    
    assert results == sorted(results), "The file list is not sorted alphabetically."

def test_find_mri_files_first_element_is_correct(tmp_path):
    """Verify that the sorting logic correctly identifies the first alphabetical element.

    GIVEN: A directory containing 'z_scan' and 'a_scan'.
    WHEN: find_mri_files is executed.
    THEN: The first element of the list is 'a_scan'.
    """
    (tmp_path / "z_scan.nii.gz").write_text("dummy")
    (tmp_path / "a_scan.nii.gz").write_text("dummy")
    
    results = find_mri_files(str(tmp_path), "*.nii.gz")
    
    assert results[0].endswith("a_scan.nii.gz"), f"Sorting failed: expected 'a_scan...' but got '{os.path.basename(results[0])}'."

# FILTERING TESTS
def test_find_mri_files_count_is_correct(tmp_path):
    """Verify that the number of found files matches the number of valid NIfTI files.

    GIVEN: A directory with one valid NIfTI file and one JSON file.
    WHEN: find_mri_files is called with the NIfTI pattern.
    THEN: The length of the returned list is exactly 1.
    """
    (tmp_path / "valid.nii.gz").write_text("dummy")
    (tmp_path / "metadata.json").write_text("dummy")
    
    results = find_mri_files(str(tmp_path), "*.nii.gz")
    
    assert len(results) == 1, f"Filtering failed: expected 1 file, found {len(results)}."

def test_find_mri_files_extension_is_correct(tmp_path):
    """Verify that the files returned by the search actually have the requested extension.

    GIVEN: A directory containing a valid NIfTI file.
    WHEN: find_mri_files is called.
    THEN: The returned file path ends with '.nii.gz'.
    """
    (tmp_path / "valid.nii.gz").write_text("dummy")
    
    results = find_mri_files(str(tmp_path), "*.nii.gz")
    
    assert results[0].endswith(".nii.gz"), f"Extension mismatch: expected .nii.gz, got {results[0]}."

def test_find_mri_files_recursive_discovery(tmp_path):
    """
    Verify that the discovery logic can traverse nested directory structures.
    
    GIVEN: A nested hierarchy of folders containing NIfTI files.
    WHEN: find_mri_files is called with a recursive glob pattern (**/).
    THEN: Files from all subdirectories are collected into a single flat list.
    """
    level_1 = tmp_path / "subject_01"
    level_2 = level_1 / "session_A"
    level_2.mkdir(parents=True)
    
    (level_2 / "brain_mri.nii.gz").write_text("dummy")
    (tmp_path / "root_scan.nii.gz").write_text("dummy")

    results = find_mri_files(str(tmp_path), "**/*.nii.gz")

    assert len(results) == 2, f"Recursion failed: expected to find 2 files, found {len(results)}."
    assert any("session_A" in path for path in results), "Recursion failed: could not find the file in the subdirectory."

def test_dataset_sample_count(mock_mri_root):
    """Test that the dataset identifies the correct number of samples.

    GIVEN: A directory structure containing exactly 2 NIfTI files.
    WHEN: The MRINiftiDataset is initialized with a recursive pattern.
    THEN: The length of the dataset is equal to 2.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=0, file_pattern="**/*.nii.gz")
    assert len(dataset) == 2, f"Expected 2 files, but found {len(dataset)}."

def test_dataset_label_assignment(mock_mri_root):
    """Test that the dataset correctly stores the assigned class label.

    GIVEN: A valid directory and a specific numerical label (e.g., 5).
    WHEN: The MRINiftiDataset is initialized.
    THEN: The internal label attribute matches the input value.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=5, file_pattern="**/*.nii.gz")
    assert dataset.label == 5, f"Expected label 5, but got {dataset.label}."

def test_dataset_output_shape(mock_mri_root):
    """Verify that retrieval produces a tensor with the specified target shape.

    GIVEN: A target_shape of (32, 32, 32).
    WHEN: An item is retrieved via __getitem__.
    THEN: The output tensor shape is (1, 32, 32, 32).
    """
    target = (32, 32, 32)
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=0, target_shape=target, file_pattern="**/*.nii.gz")
    img, _ = dataset[0]
    assert img.shape == (1, 32, 32, 32), f"Expected shape (1, 32, 32, 32), but got {img.shape}."

def test_dataset_normalization_minimum(mock_mri_root):
    """Verify that the minimum intensity value is normalized to -1.0.

    GIVEN: Raw MRI data with arbitrary positive values.
    WHEN: The normalization pipeline is applied.
    THEN: The minimum value of the output tensor is greater than or equal to -1.0.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=0, file_pattern="**/*.nii.gz")
    img, _ = dataset[0]
    assert img.min() >= -1.0, f"Intensity underflow: {img.min()} < -1.0"

def test_dataset_normalization_maximum(mock_mri_root):
    """Verify that the maximum intensity value is normalized to 1.0.

    GIVEN: Raw MRI data with arbitrary positive values.
    WHEN: The normalization pipeline is applied.
    THEN: The maximum value of the output tensor is less than or equal to 1.0.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=0, file_pattern="**/*.nii.gz")
    img, _ = dataset[0]
    assert img.max() <= 1.0, f"Intensity overflow: {img.max()} > 1.0"

def test_dataset_image_dtype(mock_mri_root):
    """Verify that the image tensor has the correct float32 dtype.

    GIVEN: A valid MRINiftiDataset.
    WHEN: A sample is retrieved.
    THEN: The image tensor dtype is torch.float32.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=0, file_pattern="**/*.nii.gz")
    img, _ = dataset[0]
    assert img.dtype == torch.float32, f"Expected float32, got {img.dtype}."

def test_dataset_label_dtype(mock_mri_root):
    """Verify that the label tensor has the correct long dtype.

    GIVEN: A valid MRINiftiDataset.
    WHEN: A sample is retrieved.
    THEN: The label tensor dtype is torch.long.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=0, file_pattern="**/*.nii.gz")
    _, label = dataset[0]
    assert label.dtype == torch.long, f"Expected long dtype, got {label.dtype}."

def test_dataset_handles_flat_image_values(tmp_path):
    """Verify that constant images result in a zero tensor.

    GIVEN: A NIfTI file containing only zeros.
    WHEN: Normalization is applied.
    THEN: All elements in the output tensor are exactly zero.
    """
    flat_dir = tmp_path / "Flat"
    flat_dir.mkdir()
    img = nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4))
    nib.save(img, str(flat_dir / "zero.nii.gz"))
    dataset = MRINiftiDataset(root_dir=str(flat_dir), label=0, file_pattern="zero.nii.gz")
    image, _ = dataset[0]
    assert torch.all(image == 0), "Output should be all zeros for constant input."

def test_dataset_corrupted_file_raises_ioerror(tmp_path):
    """Verify that corrupted files raise an IOError during loading.

    GIVEN: A file with valid extension but invalid content.
    WHEN: __getitem__ attempts to load the file.
    THEN: An IOError is raised with the loading error message.
    """
    corrupt_dir = tmp_path / "Corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / "bad.nii.gz").write_text("Corrupted content")
    dataset = MRINiftiDataset(root_dir=str(corrupt_dir), label=0, file_pattern="bad.nii.gz")
    with pytest.raises(IOError, match="Error loading file"):
        _ = dataset[0]

def test_dataset_empty_file_list_on_invalid_pattern(tmp_path):
    """Verify that an empty list is returned if no files match the pattern.

    GIVEN: An empty directory.
    WHEN: The dataset is initialized with a non-matching pattern.
    THEN: The length of the dataset is 0.
    """
    empty_dir = tmp_path / "Empty"
    empty_dir.mkdir()
    dataset = MRINiftiDataset(root_dir=str(empty_dir), label=0, file_pattern="*.txt")
    assert len(dataset) == 0, "Dataset should be empty when no files match."

def test_dataset_filter_non_nifti_files(tmp_path):
    """Verify that files not matching the NIfTI pattern are ignored.

    GIVEN: A directory with one .nii.gz file and one .json file.
    WHEN: Initialized with the NIfTI pattern.
    THEN: The JSON file is excluded from the file list.
    """
    wrong_files_dir = tmp_path / "WrongFiles"
    wrong_files_dir.mkdir()
    nib.save(nib.Nifti1Image(np.eye(10), np.eye(4)), str(wrong_files_dir / "scan.nii.gz"))
    (wrong_files_dir / "info.json").write_text("{}")
    dataset = MRINiftiDataset(root_dir=str(wrong_files_dir), label=0, file_pattern="*.nii.gz")
    assert "info.json" not in dataset.file_list[0], "JSON file should have been filtered out."

def test_dataset_invalid_path_raises_error():
    """Verify that a FileNotFoundError is raised for non-existent directories.

    GIVEN: A path that does not exist on the filesystem.
    WHEN: The MRINiftiDataset is instantiated with this path.
    THEN: A FileNotFoundError is promptly raised.
    """
    with pytest.raises(FileNotFoundError):
        MRINiftiDataset(root_dir="/tmp/invalid/path/test/NonExistent", label=0)

def test_dataset_label_tensor_value(mock_mri_root):
    """Verify that the retrieved label tensor contains the correct value that was initialized.

    GIVEN: A dataset initialized with label 1.
    WHEN: A sample is retrieved.
    THEN: The label tensor value must be exactly 1.
    """
    dataset = MRINiftiDataset(root_dir=mock_mri_root, label=1, file_pattern="**/*.nii.gz")
    _, label = dataset[0]
    assert label.item() == 1, f"Expected label value 1, but got {label.item()}."

def test_dataset_handles_flat_image_shape(tmp_path):
    """Verify that flat images maintain correct dimensions.

    GIVEN: A NIfTI file containing only zeros.
    WHEN: The item is retrieved.
    THEN: The output tensor shape must be (1, 64, 64, 64).
    """
    flat_dir = tmp_path / "Flat"
    flat_dir.mkdir()
    img = nib.Nifti1Image(np.zeros((64, 64, 64), dtype=np.float32), np.eye(4))
    nib.save(img, str(flat_dir / "zero.nii.gz"))
    dataset = MRINiftiDataset(root_dir=str(flat_dir), label=0, file_pattern="zero.nii.gz")
    image, _ = dataset[0]
    assert image.shape == (1, 64, 64, 64), f"Expected shape (1, 64, 64, 64), got {image.shape}."