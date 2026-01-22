# 3D MRI GAN - Alzheimer's Disease Synthesis

A PyTorch implementation of a 3D Generative Adversarial Network (GAN) designed to synthesize high-resolution structural MRI brain volumes. This project focuses on generating synthetic data for different stages of Alzheimer's Disease: **Alzheimer's Disease (AD)**, **Late Mild Cognitive Impairment (LMCI)**, and **Cognitively Normal (CN)**.

## Project Overview

This software allows researchers to:
- Train a lightweight 3D GAN on consumer hardware (CPU/Entry-level GPU).
- Generate synthetic 3D NIfTI volumes indistinguishable from real pre-processed scans.
- Augment small medical datasets to improve classification model performance.

## Dataset & Preprocessing

The model is designed to work with **T1-weighted MPRAGE** sequences (originally sourced from the IDA by LONI database).

**Important:** The input data must be pre-processed using the specific pipeline described below to ensure convergence.

### Preprocessing Pipeline
All training images must undergo the following standardized workflow before being used:

1. **Format Conversion:** DICOM to NIfTI (`.nii.gz`) conversion.
2. **Standardization:** Reorientation to RAS standard and Field-of-View (FOV) reduction.
3. **Skull Stripping:** Brain extraction using FSL BET (`-R -f 0.5`).
4. **Bias Correction:** N4 Bias-Field Correction (N4ITK) to remove magnetic field inhomogeneities.
5. **Spatial Normalization:** Registration to the **MNI152 1mm standard template** (Rigid + Affine transformation).
6. **Intensity Normalization:**
    - Voxel intensities are robustly clipped between the 0.1th and 99.8th percentiles.
    - Linearly scaled to the interval **[-1, 1]**.
    - Background voxels (outside the brain mask) are explicitly set to **-1**.

### Directory Structure
To train the model, organize your pre-processed data (`MPRAGE_MNI_norm.nii.gz`) as follows:

```text
/path/to/your/data/
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
        ├── Patient_003/
        │   └── MPRAGE_MNI_norm.nii.gz
        └── ...
```
## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mri-gan-project.git
   cd mri-gan-project 
   ```

2. **Install dependencies: It is recommended to use a virtual environment.**

    ```bash
    pip install -r requirements.txt 
    ```

## Usage
The project provides a Command Line Interface (CLI) for easy execution without modifying the source code.

**Training the Model**
Run the training loop specifying the path to your data folder:

```bash
python main.py --data_root ./data --epochs 100 --batch_size 4  
```

**Available Arguments:**
- `--data_root`: (Required) Path to the directory containing class subfolders.
- `--output_dir`: Directory to save checkpoints and images (default: `./results`).
- `--epochs`: Number of training epochs (default: `100`).
- `--batch_size`: Batch size (default: `4`). Reduce this if you run out of memory.
- `--lr`: Learning rate (default: `0.0002`).
- `--device`: Force usage of `cuda` or `cpu` (default: auto-detect).

**Output**
During training, the software will generate:

- Checkpoints: Saved models in results/checkpoints/.

- Progress Images: PNG slices of generated brains in results/progress_images/ for visual monitoring.

- History: A CSV file (training_history.csv) tracking Generator and Discriminator loss.

## Testing
This project includes a suite of unit tests to verify the architecture integrity before training. To run the tests:

```bash
pytest tests/
```
## Project Structure
- `src/`: Core source code (Dataset loader, Models, Trainer).

- `data/`: Placeholder folder for your local dataset (ignored by Git).

- `tests/`: Unit tests for continuous integration.

- `main.py`: Entry point for the CLI.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Disclaimer: This software is for research and educational purposes only. Not intended for clinical use.