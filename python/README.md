# Protein Reconstruction with ChebNet

This repository contains three Python scripts designed for protein structure reconstruction using ChebNet graph convolutional networks:
*   `chebnet_blind.py` (Base Case)
*   `chebnet_aug_noisy.py`
*   `chebnet_aug_denoise.py`

All scripts implement a similar pipeline involving an HNO encoder and a Decoder2 model.

## Core Pipeline

The common steps across all scripts are:
1.  **Data Loading and Preprocessing**: Parses PDB and JSON files to load protein coordinates, renumbers atoms and residues, and aligns frames.
2.  **Graph Dataset Construction**: Builds PyTorch Geometric `Data` objects, creating k-NN graphs from the protein coordinates.
3.  **HNO Encoder Training**: Trains a ChebNet-based encoder to learn latent representations of the protein structures.
4.  **Decoder2 Training**: Trains a decoder to reconstruct protein coordinates from the HNO embeddings, optionally using a conditioner (X_ref or z_ref).
5.  **Output Export**: Exports various results (ground truth, HNO reconstructions, full decoder reconstructions, HNO embeddings, pooled embeddings) to HDF5 files.

## Script Variants and Their Distinctions

The three scripts differ primarily in their handling of **data augmentation** and **output organization**.

### 1. `chebnet_blind.py` (Base Case)

This is the foundational implementation without advanced features:
*   **Data Augmentation**: Does **not** support data augmentation. The `build_graph_dataset` function uses original, clean coordinates as both input (`x`) and target (`y`).
*   **Output Export**: Exports all results directly into the `structures/` and `latent_reps/` root directories. No subfolders are created.

### 2. `chebnet_aug_noisy.py`

This variant introduces data augmentation where the model learns to reconstruct the *noisy* input:
*   **Data Augmentation**: Supports data augmentation. When enabled, the `build_graph_dataset` function applies noise to input structures. The model's target (`y`) for reconstruction is the **noisy (perturbed) input coordinates** (`Data(x=noisy_coords_cpu, y=noisy_coords_cpu)`).
*   **Output Export**: Uses an enhanced export function that creates `structures/original/`, `latent_reps/original/`, `structures/augmented/`, and `latent_reps/augmented/` subdirectories for organized output.

### 3. `chebnet_aug_denoise.py`

This variant also introduces data augmentation, but the model learns to reconstruct the *clean* structure from noisy input (denoising):
*   **Data Augmentation**: Supports data augmentation. When enabled, the `build_graph_dataset` function applies noise to input structures. The model's target (`y`) for reconstruction is the **original (clean) coordinates** (`Data(x=noisy_coords_cpu, y=coords_cpu)`).
*   **Output Export**: Uses the same enhanced export function as `chebnet_aug_noisy.py`, creating `structures/original/`, `latent_reps/original/`, `structures/augmented/`, and `latent_reps/augmented/` subdirectories.

## Data Augmentation and Export Logic

The behavior of data augmentation and the enhanced export is controlled by the `data_augmentation.enabled` and `data_augmentation.augmentation_factor` parameters in the `param.yaml` configuration file.

*   **When `data_augmentation.enabled: true` and `augmentation_factor > 0` (for `chebnet_aug_noisy.py` and `chebnet_aug_denoise.py`)**:
    *   The dataset will include both original and newly generated augmented samples.
    *   The export function will create and populate two sets of directories:
        *   `structures/original/` and `latent_reps/original/`: Containing results for the unperturbed original frames.
        *   `structures/augmented/` and `latent_reps/augmented/`: Containing results for the full dataset (original + augmented samples).

*   **When `data_augmentation.enabled: false` (for `chebnet_aug_noisy.py` and `chebnet_aug_denoise.py`)**:
    *   No data augmentation is applied. The dataset consists only of the original frames.
    *   The export function will only create and populate `structures/original/` and `latent_reps/original/` directories. The `augmented/` directories will **not** be created.

## Configuration (`param.yaml`)

Key parameters for controlling data augmentation and other aspects are found in `param.yaml`:

```yaml
data_augmentation:
  enabled: true # Set to true to enable data augmentation, false to disable
  augmentation_factor: 1 # Number of augmented samples per original sample. Set to 0 for no augmentation.
  initial_noise_std_range: [0.1, 0.8] # Range for initial noise standard deviation
  target_ca_rmsd: 1.0 # Target C-alpha RMSD for noise scaling
```

## Usage Examples

To run any of the scripts, you need to provide a configuration file (e.g., `param.yaml`).

**1. Run the Base Case (`chebnet_blind.py`):**

Ensure `param.yaml` does NOT have a `data_augmentation` section or has `data_augmentation.enabled: false`.

```bash
python chebnet_blind.py --config param.yaml
```
*Output*: Files directly in `structures/` and `latent_reps/`.

**2. Run `chebnet_aug_noisy.py` or `chebnet_aug_denoise.py` with Data Augmentation Enabled:**

Ensure `param.yaml` has `data_augmentation.enabled: true` and `augmentation_factor` set to `1` or higher.

```bash
python chebnet_aug_noisy.py --config param.yaml
# OR
python chebnet_aug_denoise.py --config param.yaml
```
*Output*: Files in `structures/original/`, `latent_reps/original/`, `structures/augmented/`, and `latent_reps/augmented/`.

**3. Run `chebnet_aug_noisy.py` or `chebnet_aug_denoise.py` with Data Augmentation Disabled:**

Ensure `param.yaml` has `data_augmentation.enabled: false`.

```bash
python chebnet_aug_noisy.py --config param.yaml
# OR
python chebnet_aug_denoise.py --config param.yaml
```
*Output*: Files only in `structures/original/` and `latent_reps/original/`.
