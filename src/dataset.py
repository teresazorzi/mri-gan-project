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
import torch.nn.functional as F
import scipy.ndimage as ndimage

class MRINiftiDataset(Dataset):
    """
    Custom Dataset for loading 3D MRI Nifti files.
    
    This dataset scans a directory for Nifti files, normalizes them to [-1, 1],
    and resizes them to the target shape.
    It supports diverse 3D imaging protocols (e.g., T1, T2, PET), assuming 
    the input data is spatially registered
    """

    def __init__(self, class_dir, label, target_shape=(64, 64, 64),file_pattern="MPRAGE_MNI_norm.nii.gz"):
        """
        Args:
            class_dir (str): Path to the folder containing the specific class images.
            label (int): Integer label associated with this class.
            target_shape (tuple): Desired output shape (Depth, Height, Width).
            file_pattern (str): Filename pattern to search for (default: 'MPRAGE_MNI_norm.nii.gz').
                                Can be changed to '*.nii' or specific filenames for other datasets.
        """
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"The directory '{class_dir}' does not exist. Please check the path.")
        
        self.label = label
        self.target_shape = target_shape
        
        # Search for specific Nifti files recursively
        # This allows reusability for other datasets/modalities.
        self.file_list = glob.glob(os.path.join(class_dir, "**", file_pattern), recursive=True)
        
        if len(self.file_list) == 0:
            print(f"Warning: No files matching '{file_pattern}' found in {class_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            # Load Nifti file
            img_path = self.file_list[idx]
            img = nib.load(img_path)
            data = img.get_fdata().astype(np.float32)
            
            # Robust Normalization to [0, 1]
            mi, ma = data.min(), data.max()

            # Safety check for flat images (division by zero protection)
            if ma - mi > 1e-8:
                data = (data - mi) / (ma - mi + 1e-8)
            else:
                # Handle corrupted/empty scans safely
                data = np.zeros_like(data)

            # Rescale to [-1, 1] for GAN stability (Tanh activation in Generator)
            data = (data * 2) - 1
            
            # Convert to Tensor (Add Channel and Batch dimensions for interpolation)
            # Input to interpolate must be (Batch, Channel, D, H, W)
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0) 
            
            # Resize to target shape
            tensor = F.interpolate(tensor, size=self.target_shape, mode='trilinear', align_corners=False)

            # Return tensor: remove Batch dim -> (Channel, D, H, W)
            return tensor.squeeze(0), torch.tensor(self.label, dtype=torch.long)
            
        except Exception as e:
            # In production/exams, it is better to log the error rather than silently returning zeros
            print(f"Error loading file {self.file_list[idx]}: {e}")
            return torch.zeros((1, *self.target_shape)), torch.tensor(self.label, dtype=torch.long)