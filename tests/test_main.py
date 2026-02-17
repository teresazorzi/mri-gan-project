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
import yaml

# --- FIXTURES ---

@pytest.fixture
def integration_env(tmp_path):
    """
    Setup a complete environment for integration testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest fixture providing a temporary directory unique to the test.

    Returns
    -------
    tuple
        (config_path, save_dir) - Paths to the generated config and output folder.
    """

    # 1. Create Mock Dataset (2 classes: AD and CN)
    data_root = tmp_path / "mock_data"
    for cls in ["AD", "CN"]:
        cls_dir = data_root / cls
        cls_dir.mkdir(parents=True)
        # Create a 64x64x64 NIfTI
        img_data = np.random.rand(64, 64, 64).astype(np.float32)
        img = nib.Nifti1Image(img_data, np.eye(4))
        nib.save(img, str(cls_dir / "test_scan.nii.gz"))

    # 2. Define Output Directory
    save_dir = tmp_path / "output"
    save_dir.mkdir()

    # 3. Create Mock Config
    config = {
        'dataset': {
            'data_root': str(data_root),
            'file_pattern': 'test_scan.nii.gz',
            'target_shape': [64, 64, 64],
            'num_workers': 0
        },
        'model': {
            'latent_dim': 4,
            'ngf': 2,
            'ndf': 2,
            'num_classes': 2
        },
        'training': {
            'epochs': 1,
            'batch_size': 1,
            'lr': 0.0002,
            'n_critic': 1,
            'lambda_gp': 0.0,
            'device': 'cpu',
            'seed': 42
        },
        'output': {
            'save_dir': str(save_dir),
            'sample_interval': 1,
            'checkpoint_interval': 1
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    return str(config_path), str(save_dir)

# --- INTEGRATION TESTS ---

def test_main_cli_successful_run(integration_env):
    """Verify that the main script executes without errors using a valid configuration.

    GIVEN: A valid configuration file and mock NIfTI dataset.
    WHEN: The main.py script is executed via CLI.
    THEN: The process exit code is 0 (Success).
    """
    config_path, _ = integration_env
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    main_script = os.path.join(project_root, 'main.py')

    cmd = [sys.executable, main_script, "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, f"Main execution failed!\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"

def test_main_creates_checkpoints_directory(integration_env):
    """Verify that the training generates the checkpoints directory.

    GIVEN: A valid integration environment.
    WHEN: The main.py script is executed successfully.
    THEN: The 'checkpoints' folder is created within the save directory.
    """
    config_path, save_dir = integration_env
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, 'main.py')

    subprocess.run([sys.executable, main_script, "--config", config_path], check=True)

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    assert os.path.exists(ckpt_dir), f"Checkpoint directory not created at: {ckpt_dir}"

def test_main_creates_samples_directory(integration_env):
    """Verify that the training generates the progress images directory.

    GIVEN: A valid integration environment.
    WHEN: The main.py script is executed successfully.
    THEN: The 'progress_images' folder is created within the save directory.
    """
    config_path, save_dir = integration_env
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(project_root, 'main.py')

    subprocess.run([sys.executable, main_script, "--config", config_path], check=True)
    
    imgs_dir = os.path.join(save_dir, "progress_images")
    assert os.path.exists(imgs_dir), f"Progress images directory not created at: {imgs_dir}"

def test_main_num_classes_mismatch_error(integration_env):
    """Verify defensive check when folder structure doesn't match config.

    GIVEN: A config expecting 3 classes but a dataset with only 2 folders.
    WHEN: The main.py script attempts to load the dataset.
    THEN: A specific 'CONFIG MISMATCH' error message is printed to output.
    """
    config_path, _ = integration_env
    # Modify config to expect 3 classes
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['model']['num_classes'] = 3
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    main_script = os.path.join(project_root, 'main.py')

    result = subprocess.run([sys.executable, main_script, "--config", config_path], capture_output=True, text=True)
    
    assert "CONFIG MISMATCH" in result.stdout or "CONFIG MISMATCH" in result.stderr, \
        "The script failed to detect the number of classes mismatch."