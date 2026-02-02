# -*- coding: utf-8 -*-
'''
Author: Teresa Zorzi
Date: January 2026
'''

import sys
import os
import pytest
import subprocess
import numpy as np
import nibabel as nib


@pytest.fixture
def temp_data_structure(tmp_path):
    """
    Create a temporary valid dataset structure (AD, CN, LMCI) for integration testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory unique to the test invocation.

    Returns
    -------
    str
        The absolute path to the temporary data root directory.
    """
    root_dir = tmp_path / "data"
    root_dir.mkdir()
    
    classes = ["AD", "CN", "LMCI"]
    for cls in classes:
        cls_dir = root_dir / cls
        cls_dir.mkdir()
        
        patient_dir = cls_dir / "Patient_001"
        patient_dir.mkdir()
        
        # Create a fake 64x64x64 Nifti image (random noise)
        data = np.random.rand(64, 64, 64).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, patient_dir / "mri.nii.gz")
        
    return str(root_dir)


def test_main_execution_cli(temp_data_structure, tmp_path):
    """
    Test the full execution of main.py via Command Line Interface (CLI).
    
    This verifies that the script runs from start to finish (exit code 0)
    given valid arguments, effectively acting as an integration test.

    Parameters
    ----------
    temp_data_structure : str
        Path to the temporary data root created by the fixture.
    tmp_path : pathlib.Path
        Pytest fixture for creating temporary output directories.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    main_script = os.path.join(project_root, 'main.py')
    
    if not os.path.exists(main_script):
        pytest.fail(f"Could not find main.py at {main_script}")

    temp_output_dir = tmp_path / "test_results"

    # We use sys.executable to ensure we use the same Python environment 
    # currently running the tests (avoids venv mismatch).
    cmd = [
        sys.executable, main_script,
        "--data_root", temp_data_structure,
        "--save_dir", str(temp_output_dir),
        "--file_pattern", "mri.nii.gz", 
        "--epochs", "1",           
        "--batch_size", "2",       
        "--n_critic", "1",         
        "--lr", "0.001",
        "--latent_dim", "10",
        "--device", "cpu",         # Force CPU to ensure stability in CI environments without GPUs
        "--num_workers", "0"       # Avoids multiprocessing overhead/freezing on Windows
    ]
    
    print(f"Executing command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode != 0:
        print("\n--- STDOUT (Logs) ---")
        print(result.stdout)
        print("\n--- STDERR (Errors) ---")
        print(result.stderr)
        
    assert result.returncode == 0, "main.py crashed! See STDERR above for details."
    
    # Verify that the script actually produced output, not just exited silently.
    assert os.path.exists(temp_output_dir), "Output directory was not created."