# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import scipy.ndimage


def find_mri_files(root_dir, file_pattern):
    """
    Search for NIfTI files and return a sorted list of absolute paths.

    Parameters
    ----------
    root_dir : str
        The base directory to search.
    file_pattern : str
        The glob pattern to match (e.g., "**/*.nii.gz").

    Returns
    -------
    list of str
        Sorted absolute paths to the found files.
    """
    search_path = os.path.join(root_dir, file_pattern)
    
    return sorted(glob.glob(search_path, recursive=True))

class MRINiftiDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing 3D MRI Nifti files.

    This dataset scans a directory for Nifti files, normalizes their intensity,
    and spatially resizes them to a fixed target shape suitable for the GAN.
    """

    def __init__(self, root_dir, label, file_pattern="*.nii.gz", target_shape=None):
        """
        Initialize the MRI Dataset by scanning and sorting available NIfTI files.

        Parameters
        ----------
        root_dir : str
            Path to the directory containing the MRI NIfTI files for a specific class.
        label : int
            Numerical class label associated with the data (e.g., 0 for AD, 1 for CN, 2 for LMCI).
        file_pattern : str, optional
            A glob-style string pattern used to filter files within `root_dir`.
            Defaults to "*.nii.gz".
        target_shape : tuple of int, optional
            The desired output dimensions (Depth, Height, Width) for 3D resampling.

        Raises
        ------
        FileNotFoundError
            If the specified `root_dir` does not exist on the filesystem.
        """
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"The directory '{root_dir}' does not exist. Please check the path.")

        self.root_dir = root_dir
        self.label = label
        self.target_shape = target_shape

        # Build path search string
        search_path = os.path.join(root_dir, file_pattern)

        # Recursively search for Nifti files.
        self.file_list = find_mri_files(root_dir, file_pattern)
        
        if len(self.file_list) == 0:
            print(f"Warning: No files found in {search_path}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns
        -------
        int
            Number of Nifti files found.
        """
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Load, normalize, and resize the MRI volume.

        The method handles the full preprocessing pipeline: loading the NIfTI file, applying spatial resampling (interpolation),
        and performing intensity normalization to the [-1, 1] range.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve from the sorted file list.

        Returns
        -------
        img_tensor : torch.Tensor
            The preprocessed 3D image tensor with shape (1, D, H, W).
        label_tensor : torch.Tensor
            The class label as a long integer tensor.

        Raises
        ------
        IOError
            If the NIfTI file is corrupted, missing, or cannot be parsed by nibabel.
        RuntimeError
            If resampling fails due to incompatible dimensions.
        """

        img_path = self.file_list[idx]
        try:
            img = nib.load(img_path)
            img_data = img.get_fdata().astype(np.float32)
        except Exception as e:
            # Handle corrupted files
            raise IOError(f"Error loading file {img_path}: {e}")

        # 3. Resize (if target_shape is provided)
        if self.target_shape is not None and img_data.shape != self.target_shape:
            zoom_factors = [t / s for t, s in zip(self.target_shape, img_data.shape)]
            img_data = scipy.ndimage.zoom(img_data, zoom_factors, order=1) # Order 1 = Linear interpolation

        # 4. Normalization to [-1, 1]
        # GANs typically require input in range [-1, 1] or [0, 1]..
        min_val = np.min(img_data)
        max_val = np.max(img_data)

        if max_val - min_val > 0:
            img_data = (img_data - min_val) / (max_val - min_val) # [0, 1]
            img_data = (img_data * 2) - 1 # [-1, 1]
        else:
            # Fallback for empty/constant images (shouldn't happen in MRI)
            img_data = np.zeros_like(img_data)

        # 5. Convert to Tensor
        # PyTorch 3D Conv expects: (Channel, Depth, Height, Width)
        # We add the Channel dimension using np.expand_dims
        img_tensor = torch.from_numpy(np.expand_dims(img_data, axis=0))
        
        # 6. Label Tensor
        label_tensor = torch.tensor(self.label, dtype=torch.long)

        return img_tensor, label_tensor