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
    
    This dataset scans a directory for Nifti files, normalizes them,
    and optionally applies lightweight data augmentation (rotation/shift).
    """

    def __init__(self, class_dir, label, target_shape=(64, 64, 64), augment=False):
        """
        Args:
            class_dir (str): Path to the folder containing the specific class images.
            label (int): Integer label associated with this class.
            target_shape (tuple): Desired output shape (Depth, Height, Width).
            augment (bool): If True, applies random rotation and shift to the data.
        """
        self.label = label
        self.target_shape = target_shape
        self.augment = augment 
        
        # Search for specific Nifti files recursively
        # Ideally, the specific filename pattern should also be a parameter, 
        # but we keep it here for simplicity based on the original project.
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

            # --- LIGHTWEIGHT AUGMENTATION ---
            if self.augment:
                # 1. Subtle rotation (+/- 1 degree)
                angle = np.random.uniform(-1, 1)
                axes = np.random.choice([(0,1), (0,2), (1,2)]) 
                data = ndimage.rotate(data, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0.0)
                
                # 2. Small random shift
                shift_x = np.random.randint(-2, 3) 
                shift_y = np.random.randint(-2, 3)
                shift_z = np.random.randint(-2, 3)
                data = ndimage.shift(data, shift=[shift_x, shift_y, shift_z], order=1, mode='constant', cval=0.0)

            # Rescale to [-1, 1] for GAN stability (Tanh activation in Generator)
            data = (data * 2) - 1
            
            # Convert to Tensor and resize
            tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0) 
            tensor = F.interpolate(tensor, size=self.target_shape, mode='trilinear', align_corners=False)
            
            return tensor.squeeze(0), torch.tensor(self.label, dtype=torch.long)
            
        except Exception as e:
            # In production/exams, it is better to log the error rather than silently returning zeros
            print(f"Error loading file {self.file_list[idx]}: {e}")
            return torch.zeros((1, *self.target_shape)), torch.tensor(self.label, dtype=torch.long)