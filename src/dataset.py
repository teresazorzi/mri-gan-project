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
    """

    def __init__(self, class_dir, label, target_shape=(64, 64, 64), augment=False):
        """
        Args:
            class_dir (str): Path to the folder containing the specific class images.
            label (int): Integer label associated with this class.
            target_shape (tuple): Desired output shape (Depth, Height, Width).
        """
        self.label = label
        self.target_shape = target_shape
        
        # Search for specific Nifti files recursively
        # Pattern: looks for 'MPRAGE_MNI_norm.nii.gz' inside subfolders
        self.file_list = glob.glob(os.path.join(class_dir, "**", "MPRAGE_MNI_norm.nii.gz"), recursive=True)
        
        if len(self.file_list) == 0:
            print(f"Warning: No files found in {class_dir}")

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
            data = (data - mi) / (ma - mi + 1e-8)

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