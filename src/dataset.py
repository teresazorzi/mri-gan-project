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
import torch.nn.functional as F
from torch.utils.data import Dataset


class MRINiftiDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing 3D MRI Nifti files.

    This dataset scans a directory for Nifti files, normalizes their intensity,
    and spatially resizes them to a fixed target shape suitable for the GAN.
    """

    def __init__(self, class_dir, label, target_shape=(64, 64, 64), file_pattern="MPRAGE_MNI_norm.nii.gz"):
        """
        Initialize the MRI Dataset.

        Parameters
        ----------
        class_dir : str
            Path to the folder containing the specific class images.
        label : int
            Integer label associated with this class (e.g., 0 for CN, 1 for AD).
        target_shape : tuple, optional
            Desired output shape (Depth, Height, Width) (default is (64, 64, 64)).
        file_pattern : str, optional
            Filename pattern to search for (default is 'MPRAGE_MNI_norm.nii.gz').
            Can be changed to '*.nii' or specific filenames for other datasets.

        Raises
        ------
        FileNotFoundError
            If the provided `class_dir` does not exist.
        """
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"The directory '{class_dir}' does not exist. Please check the path.")

        self.label = label
        self.target_shape = target_shape

        # Recursively search for Nifti files to handle complex folder structures.
        self.file_list = glob.glob(os.path.join(class_dir, "**", file_pattern), recursive=True)

        if len(self.file_list) == 0:
            print(f"Warning: No files matching '{file_pattern}' found in {class_dir}")

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
        Load, normalize, and resize the MRI volume at the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: The preprocessed 3D image of shape (1, D, H, W).
            - torch.Tensor: The class label.

        Notes
        -----
        If a file is corrupted or cannot be loaded, a dummy zero-tensor is returned
        to prevent the entire training loop from crashing.
        """
        try:
            img_path = self.file_list[idx]
            img = nib.load(img_path)
            data = img.get_fdata().astype(np.float32)

            # Robust Min-Max Normalization to [0, 1].
            # This standardizes intensity ranges across different patients/scanners.
            mi, ma = data.min(), data.max()

            # Prevent division by zero if the image is flat/empty.
            if ma - mi > 1e-8:
                data = (data - mi) / (ma - mi + 1e-8)
            else:
                # Handle corrupted/empty scans safely by returning a zero volume.
                data = np.zeros_like(data)

            # Rescale to [-1, 1].
            # This matches the Tanh output range of the Generator, critical for WGAN stability.
            data = (data * 2) - 1

            # Convert to Tensor and add Batch/Channel dimensions for interpolation.
            # Input to F.interpolate must be (Batch, Channel, D, H, W).
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)

            # Resize to target shape.
            # Ensures all inputs have consistent dimensions for the neural network.
            tensor = F.interpolate(tensor, size=self.target_shape, mode='trilinear', align_corners=False)

            # Return tensor: remove Batch dim -> (Channel, D, H, W).
            return tensor.squeeze(0), torch.tensor(self.label, dtype=torch.long)

        except Exception as e:
            # Log the error but continue execution (Fault Tolerance).
            print(f"Error loading file {self.file_list[idx]}: {e}")
            return torch.zeros((1, *self.target_shape)), torch.tensor(self.label, dtype=torch.long)