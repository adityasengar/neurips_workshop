# Latent Space Dynamics and Structure Reconstruction

This project focuses on modeling the temporal evolution of molecular systems within a learned latent space and subsequently reconstructing their 3D structures. It provides tools for simulating dynamics using various approaches and for converting latent space trajectories back into interpretable atomic coordinates.

## 1. Latent Space Dynamics Modeling

Dynamics in the latent space can be performed using three primary methods: Langevin Dynamics, a linear Koopman Operator, or a non-linear Neural Network Propagator. These methods operate on time-series data, such as pooled embeddings or diffusion-generated embeddings.

### 1.1. Langevin Dynamics (`simulate_dynamics.py`)

This script performs Langevin dynamics simulations directly within the latent space learned by a diffusion model. It uses a pre-trained diffusion model to estimate the "forces" (score function) that guide the movement of the latent vector, allowing for the generation of dynamic trajectories of protein conformations.

**Key Arguments:**

*   `--diff_config`: Path to the `param_diff.yaml` file used for training the diffusion model.
*   `--diff_ckpt`: Path to the trained diffusion model checkpoint (e.g., `checkpoints/diffusion_checkpoint_exp1.pth`).
*   `--start_emb_file_name`: (Optional) Absolute path to the HDF5 file containing the starting latent embedding. If provided, `--start_emb_dir`, `--start_exp_idx`, and `--start_conservative_mode` are ignored for filename construction.
*   `--start_emb_dir`: (Used if `--start_emb_file_name` is not provided) Directory containing the starting generated embedding file (e.g., `latent_reps`).
*   `--start_exp_idx`: Index of the specific embedding within the HDF5 file to use as the starting point. Defaults to `0` (first embedding). If `--start_emb_file_name` is not provided, this index is also used to construct the default filename (e.g., `generated_embeddings_expX_...h5`).
*   `--start_conservative_mode`: (Used if `--start_emb_file_name` is not provided) Use this flag if the starting embedding is from a conservative diffusion run (e.g., `generated_embeddings_expX_conservative.h5`). Omit it for non-conservative runs.
*   `--num_steps`: The total number of simulation steps to perform.
*   `--temperature`: A float representing the temperature for the Langevin dynamics diffusion term. Set to `0.0` for deterministic-like trajectories (no thermal noise).
*   `--output_file`: The absolute path for the output HDF5 file where the Langevin trajectory will be saved.
*   `--time_step_size`: The time step size (`dt`) for Langevin dynamics (e.g., `1e-8`).
*   `--noise_level_t`: The integer `t` for `noise_level_sigma` (which is `sqrt_one_minus_alphas_cumprod[t]`). This parameter controls the noise level at which the score function is evaluated.
*   `--save_frequency`: Frequency (in steps) at which to save states to the output HDF5 file. Default is `1` (save every step).
*   `--score_clip_value`: Optional. Maximum norm for the score/force vector to prevent instability. If omitted, no clipping is applied.

**Example Usage:**

```bash
python simulate_dynamics.py \
    --diff_config param_diff.yaml \
    --diff_ckpt checkpoints/diffusion_checkpoint_exp1.pth \
    --start_emb_dir latent_reps \
    --start_exp_idx 1 \
    --num_steps 100 \
    --temperature 0.0 \
    --output_file latent_reps/langevin_trajectory_trial.h5 \
    --time_step_size 1e-8 \
    --noise_level_t 0 \
    --save_frequency 1
```

**Small Example Run (Langevin -> Inference -> XTC):**

1.  **Simulate Langevin Dynamics:**
    ```bash
    python simulate_dynamics.py \
        --diff_config param_diff.yaml \
        --diff_ckpt checkpoints/diffusion_checkpoint_exp1.pth \
        --start_emb_file_name latent_reps/generated_embeddings_exp1_non_conservative.h5 \
        --start_emb_key generated_embeddings \
        --num_steps 100 \
        --temperature 0.0 \
        --output_file latent_reps/langevin_trajectory_example.h5 \
        --time_step_size 1e-8 \
        --noise_level_t 0 \
        --save_frequency 1
    ```

2.  **Run Inference on Langevin Trajectory:**
    ```bash
    python inference_old.py \
        --config param.yaml \
        --hno_ckpt checkpoints/hno_checkpoint.pth \
        --decoder2_ckpt checkpoints/decoder2_checkpoint.pth \
        --diff_emb_file latent_reps/langevin_trajectory_example.h5 \
        --diff_emb_key langevin_trajectory \
        --conditioner_x_ref_pt structures/X_ref_coords.pt \
        --conditioner_z_ref_pt latent_reps/z_ref_embedding.pt \
        --output_file structures/full_coords_langevin_trajectory_example.h5 \
        --output_key full_coords_langevin_trajectory_example
    ```

3.  **Convert HDF5 to XTC Trajectory:**
    ```bash
    python convert_h5_to_xtc.py \
        --h5_file structures/full_coords_langevin_trajectory_example.h5 \
        --h5_key full_coords_langevin_trajectory_example \
        --ref_pdb /scratch/asengar/long_sim/atlas/kinase_1ptq_perturn/heavy_chain.pdb \
        --output_xtc structures/full_coords_langevin_trajectory_example.xtc
    ```

### 1.2. Linear Dynamics with a Koopman Operator (`fit_koopman_model.py`)

This method fits a single linear matrix `A` that approximates the system's dynamics according to `z_{t+1} ≈ A * z_t`. It uses Dynamic Mode Decomposition (DMD) to find the best-fit linear operator, providing a fast and interpretable model.

**Key Arguments:**

*   `--h5_file`: (Optional) Path to the HDF5 file containing the latent space embeddings (e.g., `pooled_embedding.h5`). Defaults to `latent_reps/pooled_embedding.h5`.
*   `--train_fraction`: (Optional) The fraction of the input time-series data to use for training. Defaults to `1.0`.
*   `--svd_rank`: (Optional) Number of dominant singular values (modes) to retain during SVD for de-noising and dimensionality reduction. If omitted, the full rank is used.
*   `--rollout_noise_std`: (Optional) Standard deviation of Gaussian noise to add at each step of the predictive rollout. Defaults to `0.0`.
*   `--output_file`: (Optional) Path to save the learned Koopman operator matrix (`A`) as a NumPy array. Defaults to `koopman_operator.npy`.

**Example Usage:**

```bash
python fit_koopman_model.py --train_fraction 0.8 --svd_rank 20 --rollout_noise_std 0.01
```

**Small Example Run (Koopman -> Inference -> XTC):**

1.  **Fit Koopman Model and Generate Rollout:**
    ```bash
    python fit_koopman_model.py \
        --h5_file latent_reps/pooled_embedding.h5 \
        --train_fraction 0.8 \
        --svd_rank 20 \
        --rollout_noise_std 0.01 \
        --output_file koopman_operator_example.npy
    ```
    *Output H5 file: `latent_reps/koopman_rollout_train0p80_rank20.h5` (default naming based on parameters)*

2.  **Run Inference on Koopman Trajectory:**
    ```bash
    python inference_old.py \
        --config param.yaml \
        --hno_ckpt checkpoints/hno_checkpoint.pth \
        --decoder2_ckpt checkpoints/decoder2_checkpoint.pth \
        --diff_emb_file latent_reps/koopman_rollout_train0p80_rank20.h5 \
        --diff_emb_key koopman_rollout \
        --conditioner_x_ref_pt structures/X_ref_coords.pt \
        --conditioner_z_ref_pt latent_reps/z_ref_embedding.pt \
        --output_file structures/full_coords_koopman_rollout_example.h5 \
        --output_key full_coords_koopman_rollout_example
    ```

3.  **Convert HDF5 to XTC Trajectory:**
    ```bash
    python convert_h5_to_xtc.py \
        --h5_file structures/full_coords_koopman_rollout_example.h5 \
        --h5_key full_coords_koopman_rollout_example \
        --ref_pdb /scratch/asengar/long_sim/atlas/kinase_1ptq_perturn/heavy_chain.pdb \
        --output_xtc structures/full_coords_koopman_rollout_example.xtc
    ```

### 1.3. Non-Linear Dynamics with a Neural Network Propagator (`train_neural_propagator.py`)

This method trains a neural network to learn a potentially non-linear function `f` that maps `z_t` to `z_{t+1}`, i.e., `z_{t+1} ≈ f(z_t)`. This approach can capture more complex dynamics.

**Key Arguments:**

*   `--h5_file`: (Optional) Path to the HDF5 file containing the latent space embeddings (e.g., `pooled_embedding.h5`). Defaults to `latent_reps/pooled_embedding.h5`.
*   `--train_fraction`: (Optional) The fraction of the input time-series data to use for training. Defaults to `0.8`.
*   `--frame_skip`: (Optional) The number of frames (time steps) to predict into the future (`z_{t+n}`). Defaults to `1`.
*   `--rollout_noise_std`: (Optional) Standard deviation of Gaussian noise to add at each step of the predictive rollout. Defaults to `0.0`.
*   `--epochs`: (Optional) The number of training epochs. Defaults to `100`.
*   `--output_model_path`: (Optional) Path to save the trained neural network model's weights. Defaults to `neural_propagator.pth`.

**Example Usage:**

```bash
python train_neural_propagator.py --train_fraction 0.8 --frame_skip 5 --rollout_noise_std 0.01
```

**Small Example Run (Neural Propagator -> Inference -> XTC):**

1.  **Train Neural Propagator and Generate Rollout:**
    ```bash
    python train_neural_propagator.py \
        --h5_file latent_reps/pooled_embedding.h5 \
        --train_fraction 0.8 \
        --frame_skip 5 \
        --rollout_noise_std 0.01 \
        --epochs 10 \
        --output_model_path neural_propagator_example.pth
    ```
    *Output H5 file: `latent_reps/nn_rollout_train0p80_skip5_noise0p01.h5` (default naming based on parameters)*

2.  **Run Inference on Neural Propagator Trajectory:**
    ```bash
    python inference_old.py \
        --config param.yaml \
        --hno_ckpt checkpoints/hno_checkpoint.pth \
        --decoder2_ckpt checkpoints/decoder2_checkpoint.pth \
        --diff_emb_file latent_reps/nn_rollout_train0p80_skip5_noise0p01.h5 \
        --diff_emb_key nn_rollout \
        --conditioner_x_ref_pt structures/X_ref_coords.pt \
        --conditioner_z_ref_pt latent_reps/z_ref_embedding.pt \
        --output_file structures/full_coords_nn_rollout_example.h5 \
        --output_key full_coords_nn_rollout_example
    ```

3.  **Convert HDF5 to XTC Trajectory:**
    ```bash
    python convert_h5_to_xtc.py \
        --h5_file structures/full_coords_nn_rollout_example.h5 \
        --h5_key full_coords_nn_rollout_example \
        --ref_pdb /scratch/asengar/long_sim/atlas/kinase_1ptq_perturn/heavy_chain.pdb \
        --output_xtc structures/full_coords_nn_rollout_example.xtc
    ```

## 2. Running Inference on Latent Trajectories (`inference_old.py`)

After simulating dynamics in the latent space, you will have a trajectory of latent embeddings. To visualize these, you need to decode them back into 3D protein coordinates. For Langevin trajectories, `inference_old.py` is recommended due to its flexibility in accepting direct HDF5 input files.

**Key Arguments:**

*   `--config`: Path to the main configuration file (e.g., `param_old.yaml`).
*   `--hno_ckpt`: Path to the trained HNO (Hierarchical Neural Operator) checkpoint (e.g., `checkpoints/hno_checkpoint.pth`).
*   `--decoder2_ckpt`: Path to the trained decoder checkpoint (e.g., `checkpoints/decoder2_checkpoint.pth`).
*   `--diff_emb_file`: Path to the HDF5 file containing the latent embeddings to decode (e.g., `latent_reps/langevin_trajectory_trial.h5`).
*   `--diff_emb_key`: The key within the HDF5 file that corresponds to the latent embeddings (e.g., `langevin_trajectory`).
*   `--conditioner_x_ref_pt`: Path to the reference coordinates for conditioning (e.g., `structures/X_ref_coords.pt`).
*   `--conditioner_z_ref_pt`: Path to the reference latent embedding for conditioning (e.g., `latent_reps/z_ref_embedding.pt`).
*   `--output_file`: The absolute path for the output HDF5 file where the 3D coordinates will be saved.
*   `--output_key`: The key within the output HDF5 file for the 3D coordinates.

**Example Usage:**

```bash
python inference_old.py \
    --config param_old.yaml \
    --hno_ckpt checkpoints/hno_checkpoint.pth \
    --decoder2_ckpt checkpoints/decoder2_checkpoint.pth \
    --diff_emb_file latent_reps/langevin_trajectory_trial.h5 \
    --diff_emb_key langevin_trajectory \
    --conditioner_x_ref_pt structures/X_ref_coords.pt \
    --conditioner_z_ref_pt latent_reps/z_ref_embedding.pt \
    --output_file structures/full_coords_langevin_trajectory_trial.h5 \
    --output_key full_coords_langevin_trajectory_trial
```

## 3. Converting HDF5 to XTC Trajectory (`convert_h5_to_xtc.py`)

The final 3D coordinates are stored in an HDF5 file. For visualization in molecular dynamics software (like VMD or PyMOL), it's often useful to convert this HDF5 file into an XTC trajectory format.

**Key Arguments:**

*   `--h5_file`: Path to the input HDF5 file containing the 3D coordinates (e.g., `structures/full_coords_langevin_trajectory_trial.h5`).
*   `--h5_key`: The key within the HDF5 file that corresponds to the 3D coordinates.
*   `--ref_pdb`: Path to a reference PDB file (e.g., `/scratch/asengar/long_sim/atlas/kinase_1ptq_perturn/heavy_chain.pdb`). This is often used for topology information.
*   `--output_xtc`: The absolute path for the output XTC file.

**Example Usage:**

```bash
python convert_h5_to_xtc.py \
    --h5_file structures/full_coords_langevin_trajectory_trial.h5 \
    --h5_key full_coords_langevin_trajectory_trial \
    --ref_pdb /scratch/asengar/long_sim/atlas/kinase_1ptq_perturn/heavy_chain.pdb \
    --output_xtc structures/full_coords_langevin_trajectory_trial.xtc
```








