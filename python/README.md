## Key Differences: `chebnet_blind.py` vs. `chebnet_blind_unnoisy.py`

The primary distinction between the two scripts lies in how they handle the target coordinates (`y`) during the data augmentation step within the `build_graph_dataset` function:

*   **`chebnet_blind.py`**: When data augmentation is enabled, the model is trained to reconstruct the **noisy (perturbed) input coordinates**. This means `Data(x=noisy_coords_cpu, y=noisy_coords_cpu)`. The model learns to map noisy input to noisy output.
*   **`chebnet_blind_unnoisy.py`**: When data augmentation is enabled, the model is trained to reconstruct the **original (clean) coordinates** from the noisy input. This means `Data(x=noisy_coords_cpu, y=coords_cpu)`. This effectively trains the model to perform a denoising task.

## New Export Functionality

A significant new feature has been introduced in the `export_final_outputs` function to provide more organized output based on data augmentation status.

Previously, all exported structures and latent representations were saved directly into the `structures/` and `latent_reps/` directories. Now, the output is categorized into subdirectories:

*   **`structures/original/` and `latent_reps/original/`**: These directories will always contain the results corresponding to the **original, unperturbed dataset**.
*   **`structures/augmented/` and `latent_reps/augmented/`**: These directories will contain the results corresponding to the **full dataset, including both original and augmented (perturbed) samples**. These folders are only created and populated if data augmentation is enabled and the `augmentation_factor` is g>

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
python chebnet_blind.py --config param.yaml
# OR
python chebnet_blind_unnoisy.py --config param.yaml
```
This will create both `structures/original`, `latent_reps/original` and `structures/augmented`, `latent_reps/augmented` directories.

**2. Run with Data Augmentation Disabled (Base Case):**

First, ensure `param.yaml` has `data_augmentation.enabled: false`.

```bash
python chebnet_blind.py --config param.yaml
# OR
python chebnet_blind_unnoisy.py --config param.yaml
```
This will only create and populate `structures/original` and `latent_reps/original` directories.
