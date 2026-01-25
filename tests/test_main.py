# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import subprocess
import shutil
import numpy as np
import nibabel as nib

# --- FIXTURES FOR TEMPORARY DATA ---

@pytest.fixture
def temp_data_structure(tmp_path):
    """
    Creates a temporary valid dataset structure (AD, CN, LMCI) 
    so main.py has something to load.
    """
    root_dir = tmp_path / "data"
    root_dir.mkdir()
    
    classes = ["AD", "CN", "LMCI"]
    for cls in classes:
        cls_dir = root_dir / cls
        cls_dir.mkdir()
        # Create one fake patient with one fake image
        patient_dir = cls_dir / "Patient_001"
        patient_dir.mkdir()
        
        # Create fake 64x64x64 nifti (random noise)
        data = np.random.rand(64, 64, 64).astype(np.float32)
        # Nifti header identity matrix
        img = nib.Nifti1Image(data, np.eye(4))
        # Save as .nii.gz
        nib.save(img, patient_dir / "mri.nii.gz")
        
    return str(root_dir)

# --- TEST CASE ---

def test_main_execution_cli(temp_data_structure):
    """
    GIVEN a valid dataset path
    WHEN we run 'python main.py' via command line with minimal arguments
    THEN it should execute without errors (exit code 0).
    """
    
    # 1. Locate main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    main_script = os.path.join(project_root, 'main.py')
    
    # Verify main.py exists
    if not os.path.exists(main_script):
        pytest.fail(f"Could not find main.py at {main_script}")

    # 2. Build the command
    cmd = [
        sys.executable, main_script,
        "--data_root", temp_data_structure,
        "--file_pattern", "mri.nii.gz", 
        "--epochs", "1",           
        "--batch_size", "2",       
        "--n_critic", "1",         
        "--lr", "0.001",
        "--latent_dim", "10"
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # 3. Run the process
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True
    )
    
    # 4. Debugging Output (Only shows if test fails)
    if result.returncode != 0:
        print("\n--- STDOUT (Logs) ---")
        print(result.stdout)
        print("\n--- STDERR (Errors) ---")
        print(result.stderr)
        
    # 5. Assert Success
    assert result.returncode == 0, "main.py crashed! See STDERR above for details."