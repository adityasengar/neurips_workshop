# Protein Reconstruction with ChebNet

This repository contains two Python scripts, `chebnet_aug_noisy.py` and `chebnet_aug_denoise.py`, designed for protein structure reconstruction using ChebNet graph convolutional networks. Both scripts implement a similar training pipeline involving an HNO encoder and a Decoder2 model, with options for data augmentation and dihedral loss.

## Core Functionality

The scripts perform the following main steps:
1.  **Data Loading and Preprocessing**: Parses PDB and JSON files to load protein coordinates, renumbers atoms and residues, and aligns frames.
2.  **Graph Dataset Construction**: Builds PyTorch Geometric `Data` objects, creating k-NN graphs from the protein coordinates.
3.  **HNO Encoder Training**: Trains a ChebNet-based encoder to learn latent representations of the protein structures.
4.  **Decoder2 Training**: Trains a decoder to reconstruct protein coordinates from the HNO embeddings, optionally using a conditioner (X_ref or z_ref).
5.  **Output Export**: Exports various results (ground truth, HNO reconstructions, full decoder reconstructions, HNO embeddings, pooled embeddings) to HDF5 files.

## Relationship to Base `chebnet_blind.py` and Key Modifications

Both `chebnet_aug_noisy.py` and `chebnet_aug_denoise.py` are direct evolutions of an original base script, `chebnet_blind.py`. The primary modifications from this base version are:

1.  **Introduction of Data Augmentation**: The `build_graph_dataset` function was significantly enhanced to support data augmentation. The base `chebnet_blind.py` only created graphs from the original input coordinates. The current scripts introduce parameters (`data_augmentation.enabled`, `data_augmentation.augmentation_factor`, `data_augmentation.target_ca_rmsd`, `data_augmentation.initial_noise_std_range`) to apply noise to input structures and multiply the dataset>

2.  **Conditional Target (`y`) for Data Augmentation**: This is the fundamental difference that led to the creation of two distinct scripts from the base. When data augmentation is enabled:
    *   In the original `chebnet_blind.py` (and now `chebnet_aug_noisy.py`), the model's target (`y`) for reconstruction is the **noisy (perturbed) input coordinates**. The model learns to map noisy input to noisy output.
    *   In `chebnet_aug_denoise.py` (which evolved from `chebnet_blind_unnoisy.py`), the model's target (`y`) is the **original (clean) coordinates**. This trains the model to perform a denoising task, reconstructing a clean structure from a noisy input.

3.  **Enhanced Output Export**: The `export_final_outputs` function has been modified to provide a more organized output structure. The base `chebnet_blind.py` exported all results directly into the `structures/` and `latent_reps/` root directories. The updated scripts now create `original/` and `augmented/` subdirectories within these, allowing for clear separation of results based on whether they correspond to the unperturbed original data or the full dataset includ>

## Key Differences: `chebnet_aug_noisy.py` vs. `chebnet_aug_denoise.py`

The primary distinction between the two scripts lies in how they handle the target coordinates (`y`) during the data augmentation step within the `build_graph_dataset` function:

*   **`chebnet_aug_noisy.py`**: When data augmentation is enabled, the model is trained to reconstruct the **noisy (perturbed) input coordinates**. This means `Data(x=noisy_coords_cpu, y=noisy_coords_cpu)`. The model learns to map noisy input to noisy output.
*   **`chebnet_aug_denoise.py`**: When data augmentation is enabled, the model is trained to reconstruct the **original (clean) coordinates** from the noisy input. This means `Data(x=noisy_coords_cpu, y=coords_cpu)`. This effectively trains the model to perform a denoising task.

## New Export Functionality

A significant new feature has been introduced in the `export_final_outputs` function to provide more organized output based on data augmentation status.

Previously, all exported structures and latent representations were saved directly into the `structures/` and `latent_reps/` directories. Now, the output is categorized into subdirectories:

*   **`structures/original/` and `latent_reps/original/`**: These directories will always contain the results corresponding to the **original, unperturbed dataset**.
*   **`structures/augmented/` and `latent_reps/augmented/`**: These directories will contain the results corresponding to the **full dataset, including both original and augmented (perturbed) samples**. These folders are only created and populated if data augmentation is enabled and the `augmentation_factor` is greater than 0.

### Behavior with Data Augmentation

The export behavior is controlled by the `data_augmentation.enabled` and `data_augmentation.augmentation_factor` parameters in the `param.yaml` configuration file.

*   **When `data_augmentation.enabled: true` and `augmentation_factor > 0`**:
    *   The `export_final_outputs` function is called twice.
    *   First, it exports data for the `original` set (every `(1 + augmentation_factor)`-th sample from the full dataset, representing the unperturbed frames).
    *   Second, it exports data for the `augmented` set (the entire dataset, which includes both original and newly generated augmented samples).
    *   You will find both `structures/original`, `latent_reps/original` and `structures/augmented`, `latent_reps/augmented` directories populated.

*   **When `data_augmentation.enabled: false` (Base Case)**:
    *   The `export_final_outputs` function is called only once.
    *   It exports data for the `original` set. Since no augmentation occurred, the "original" dataset is simply the full input dataset.
    *   Only `structures/original/` and `latent_reps/original/` directories will be created and populated. The `augmented/` directories will **not** be created.

## Configuration (`param.yaml`)

The behavior of data augmentation and export is configured via the `param.yaml` file. Key parameters are:

```yaml
data_augmentation:
  enabled: true # Set to true to enable data augmentation, false to disable
  augmentation_factor: 1 # Number of augmented samples per original sample. Set to 0 or 1 for testing.
  # ... other augmentation parameters
```

## Usage Examples

To run the scripts, you need to provide a configuration file (e.g., `param.yaml`).

**1. Run with Data Augmentation Enabled (and factor 1):**

First, ensure `param.yaml` has `data_augmentation.enabled: true` and `data_augmentation.augmentation_factor: 1`.

```bash
python chebnet_aug_noisy.py --config param.yaml
# OR
python chebnet_aug_denoise.py --config param.yaml
```
This will create both `structures/original`, `latent_reps/original` and `structures/augmented`, `latent_reps/augmented` directories.

**2. Run with Data Augmentation Disabled (Base Case):**

First, ensure `param.yaml` has `data_augmentation.enabled: false`.

```bash
python chebnet_aug_noisy.py --config param.yaml
# OR
python chebnet_aug_denoise.py --config param.yaml
```
This will only create and populate `structures/original` and `latent_reps/original` directories.

