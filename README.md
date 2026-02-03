# 3D MRI GAN - Synthetic Alzheimer's MRI Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-ee4c2c.svg)](https://pytorch.org/)

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Repository Structure](#repository-structure)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Testing](#testing)
- [License](#license)

## Project Overview

This repository contains a PyTorch-based toolkit for generating, manipulating, and analyzing 3D medical images. It implements a **3D Generative Adversarial Network (GAN)** designed to synthesize high-resolution structural MRI brain volumes.

The project focuses on generating synthetic data for different stages of Alzheimer's Disease:
- **AD**: Alzheimer's Disease
- **LMCI**: Late Mild Cognitive Impairment
- **CN**: Cognitively Normal

**Key Features:**
- **Full 3D Synthesis:** Generates volumetric data (64x64x64), not just 2D slices.
- **WGAN Training Loop:** Uses Wasserstein Loss for robust convergence on CPU/Entry-level GPUs.
- **NIfTI Support:** Native handling of `.nii.gz` medical formats.
- **Augmentation:** Aims to improve classification performance on small medical datasets.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/mri-gan-project.git
    cd mri-gan-project
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (**Python ≥ 3.12**). Dependencies are strictly pinned to ensure total reproducibility. 
    
    Install the full environment using:
    ```bash
    pip install -r requirements.txt
    ``` 
    **Core packages included:**
    * `torch==2.10.0`
    * `numpy==1.26.4`
    * `nibabel==5.3.3`
    * `matplotlib==3.10.7`
    * `pytest==8.4.2`
    * `pytest-cov==7.0.0`

## Dataset & Preprocessing

The model is designed to work with **T1-weighted MPRAGE sequences** (originally sourced from the IDA by LONI database).

**Important**: The input data must be pre-processed using the specific pipeline described below to ensure convergence.

### Preprocessing Pipeline

All training images must undergo the following standardized workflow before being used:

1. **Format Conversion**: DICOM to NIfTI (`.nii.gz`) conversion
2. **Standardization**: Reorientation to RAS standard and Field-of-View (FOV) reduction.
3. **Skull Stripping**: Brain extraction using FSL BET.
4. **Bias Correction**: N4 Bias-Field Correction.
5. **Spatial Normalization**: Registration to the MNI152 template.
6. **Intensity Normalization**: Scaled to interval `[-1, 1]`.

### Data Directory Structure

To train the model, organize your pre-processed data as follows:
```
data/
├── AD/                          # Alzheimer's Disease (Label 0)
│   ├── Patient_001/
│   │   └── MPRAGE_MNI_norm.nii.gz
│   └── ...
├── CN/                          # Cognitively Normal (Label 1)
│   ├── Patient_002/
│   │   └── MPRAGE_MNI_norm.nii.gz
│   └── ...
└── LMCI/                        # Late Mild Cognitive Impairment (Label 2)
    └── ...
```

## Repository Structure

This repository contains the following folders and files:

- **`data/`**: Placeholder folder for input datasets. Users must organize their `.nii.gz` files here.
- **`src/`**: Contains the source code modules for the project.
  - `dataset.py`: Handles loading and preprocessing of 3D NIfTI MRI volumes.
  - `models.py`: Defines the Generator and Discriminator neural network architectures.
  - `trainer.py`: Implements the training loop, loss calculation, and backpropagation.
  - `utils.py`: Utility functions for Gradient Penalty calculation and visualization of 3D orthogonal slices.
- **`notebooks/`**: Contains interactive Jupyter Notebooks (e.g., `demo_generation.ipynb`) for visualization.
- **`tests/`**: Contains unit tests to verify model shapes and data loading logic.
- **`results/`**: Destination folder for generated images, logs, and saved models (created automatically).
- **`main.py`**: Main script for launching the training process via command line.
- **`requirements.txt`**: Contains the list of dependencies required for the project.

## Scripts Overview

### `src/models.py`
Defines the core GAN architecture optimized for 3D medical data. It includes:
- **Generator**: Transforms latent vectors into 3D volumes (64 × 64 × 64).
- **Discriminator**: Evaluates the authenticity of the scans (acts as a Critic in WGAN).

### `src/dataset.py`
Manages the loading of medical data. It searches for files matching the specified pattern (default: `MPRAGE_MNI_norm.nii.gz`) within the data directory and maps folder names (AD, CN, LMCI) to class labels.

### `src/trainer.py`
Orchestrates the training process. It handles the epoch loops, batch processing, Wasserstein loss calculation, and the alternating updates of Generator and Discriminator weights.

### `main.py`
The entry point of the application. It parses command-line arguments to set hyperparameters (epochs, learning rate, batch size, file pattern) and initiates the training process without requiring code modifications.

## Usage

The script `main.py` can be run from the command line with different configurations. Below are common usage examples.

### 1️⃣ Train the Model
Use `main.py` to start the training process.

**Standard run:**
```bash
python main.py --data_root "./data"
```

**Advanced Configuration:**
```bash
python main.py \
  --data_root "data" \
  --file_pattern "mri.nii.gz" \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.0002 \
  --n_critic 5 \
  --save_dir "results" \
  --latent_dim 64 \
  --device "cuda"
```

**Available Arguments:**
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--data_root` | Path to the dataset root folder (must contain AD/CN/LMCI subfolders) | `data` |
| `--save_dir` | Directory where results and checkpoints will be saved | `results` |
| `--file_pattern`| Specific filename to search for (e.g., `mri.nii.gz`) | `MPRAGE_MNI_norm.nii.gz` |
| `--epochs` | Total number of training epochs | `50` |
| `--batch_size` | Number of volumes per batch (reduce if OOM error occurs) | `4` |
| `--lr` | Learning rate for the optimizer | `0.0002` |
| `--n_critic` | Number of Critic (Discriminator) updates per Generator update | `5` |
| `--latent_dim` | Size of the latent noise vector (z) | `64` |
| `--device` | Computation device (`cpu` or `cuda`) | `cuda (auto-detected)` |

### 2️⃣ Demo & Visualization

You can generate synthetic brains using the trained model without running a full training loop.

**Interactive Notebook** For a more interactive experience, open the Jupyter Notebook:

1. Navigate to notebooks/

2. Open demo_generation.ipynb

3. Run the cells to load the latest checkpoint and visualize the output.

### 3️⃣ Training Outputs
During training, the software will automatically generate inside the directory specified by`--save_dir` (default: `results`):

- **Checkpoints**: Saved models (`.pth` files) in `checkpoints/` inside the specified save directory.

- **Progress Images**: PNG slices of generated brains in `progress_images/`, inside the specified save directory, for visual monitoring.

## Testing

This project follows a rigorous testing protocol to ensure reliability and maintainability.

**To run the tests:**
```bash
pytest tests/
```

**Test Coverage:**

The project achieves 100% code coverage on all core logic modules (`dataset.py`, `models.py`, `trainer.py`, `utils.py`), verifying every functional branch and numerical safety guard.
```bash
pytest --cov=src --cov-report=term-missing tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Disclaimer**: This software is for research and educational purposes only. Not intended for clinical use.