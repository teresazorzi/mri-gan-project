# 3D MRI GAN - Alzheimer's Disease Synthesis

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Repository Structure](#repository-structure)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Testing](#testing)
- [License](#license)

## Project Overview

This repository contains a PyTorch-based toolkit for generating, manipulating, and analyzing 3D medical images. It implements a 3D Generative Adversarial Network (GAN) designed to synthesize high-resolution structural MRI brain volumes.

The project focuses on generating synthetic data for different stages of Alzheimer's Disease:
- **AD**: Alzheimer's Disease
- **LMCI**: Late Mild Cognitive Impairment
- **CN**: Cognitively Normal

**Key Features:**
- Train a lightweight 3D GAN on consumer hardware (CPU/Entry-level GPU)
- Generate synthetic 3D NIfTI volumes indistinguishable from real pre-processed scans
- Augment small medical datasets to improve classification model performance

## Installation

To use this repository, first clone it with:
```bash
git clone https://github.com/YOUR_USERNAME/mri-gan-project.git
cd mri-gan-project
```

## Dependencies

This project requires Python ≥ 3.8 and standard deep learning libraries.

You can install all required dependencies using:
```bash
pip install -r requirements.txt
```

Main requirements include:
- `torch` (PyTorch)
- `numpy`
- `nibabel` (Medical image processing)
- `pandas`
- `scikit-image`
- `matplotlib`
- `pytest` (for testing)

## Dataset & Preprocessing

The model is designed to work with T1-weighted MPRAGE sequences (originally sourced from the IDA by LONI database).

⚠️ **Important**: The input data must be pre-processed using the specific pipeline described below to ensure convergence.

### Preprocessing Pipeline

All training images must undergo the following standardized workflow before being used:

1. **Format Conversion**: DICOM to NIfTI (`.nii.gz`) conversion
2. **Standardization**: Reorientation to RAS standard and Field-of-View (FOV) reduction
3. **Skull Stripping**: Brain extraction using FSL BET (`-R -f 0.5`)
4. **Bias Correction**: N4 Bias-Field Correction (N4ITK) to remove magnetic field inhomogeneities
5. **Spatial Normalization**: Registration to the MNI152 1mm standard template (Rigid + Affine transformation)
6. **Intensity Normalization**:
   - Voxel intensities are robustly clipped between the 0.1th and 99.8th percentiles
   - Linearly scaled to the interval [-1, 1]
   - Background voxels (outside the brain mask) are explicitly set to -1

### Data Directory Structure

To train the model, organize your pre-processed data (`MPRAGE_MNI_norm.nii.gz`) as follows:
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

- **`data/`**: Placeholder folder for input datasets. Users must organize their `.nii.gz` files here
- **`src/`**: Contains the source code modules for the project
  - `dataset.py`: Handles loading and preprocessing of 3D NIfTI MRI volumes
  - `models.py`: Defines the Generator and Discriminator neural network architectures
  - `trainer.py`: Implements the training loop, loss calculation, and backpropagation
  - `utils.py`: Utility functions for saving checkpoints and visualizing 3D slices
- **`tests/`**: Contains unit tests to verify model shapes and data loading logic
- **`results/`**: Destination folder for generated images, logs, and saved models (created automatically)
- **`main.py`**: Main script for launching the training process via command line
- **`requirements.txt`**: Contains the list of dependencies required for the project

## Scripts Overview

### `src/models.py`
Defines the core GAN architecture optimized for 3D medical data. It includes:
- **Generator**: Transforms latent vectors into 3D volumes (64 × 64 × 64)
- **Discriminator**: Evaluates the authenticity of the scans

### `src/dataset.py`
Manages the loading of medical data. It searches for files named `MPRAGE_MNI_norm.nii.gz` within the data directory and maps folder names (AD, CN, LMCI) to class labels.

### `src/trainer.py`
Orchestrates the training process. It handles the epoch loops, batch processing, and the alternating updates of Generator and Discriminator weights.

### `main.py`
The entry point of the application. It parses command-line arguments to set hyperparameters (epochs, learning rate, batch size) and initiates the training process without requiring code modifications.

## Usage

The script `main.py` can be run from the command line with different configurations. Below are common usage examples.

### 1️⃣ Train the Model

Run the training loop specifying the path to your data folder.

**Standard run:**
```bash
python main.py --data_root ./data --epochs 100 --batch_size 4
```

**Custom hyperparameters:**
```bash
python main.py --data_root ./data --epochs 200 --lr 0.0001 --device cuda
```

**Available Arguments:**
- `--data_root`: (Required) Path to the directory containing class subfolders
- `--output_dir`: Directory to save checkpoints and images (default: `./results`)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 4). Reduce this if you run out of memory
- `--lr`: Learning rate (default: 0.0002)
- `--device`: Force usage of `cuda` or `cpu` (default: auto-detect)

### 2️⃣ Training Outputs

During training, the software will automatically generate:

- **Checkpoints**: Saved models in `results/checkpoints/`
- **Progress Images**: PNG slices of generated brains in `results/progress_images/` for visual monitoring
- **History**: A CSV file (`training_history.csv`) tracking Generator and Discriminator loss

## Testing

This project includes a suite of unit tests to verify the architecture integrity before training.

**To run the tests:**
```bash
pytest tests/
```

**Test Coverage:**

To check how much code is covered by tests (requires `pytest-cov`):
```bash
pytest --cov=src tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Disclaimer**: This software is for research and educational purposes only. Not intended for clinical use.