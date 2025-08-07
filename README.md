# Latent Space Dynamics Simulation and Structure Reconstruction Guide

This guide focuses on the `simulate_dynamics.py` script for performing Langevin dynamics in the learned latent space, and the subsequent steps to reconstruct and visualize the generated protein structures.

## 1. Simulating Langevin Dynamics (`simulate_dynamics.py`)

### What it Does
The `simulate_dynamics.py` script performs Langevin dynamics simulations directly within the latent space learned by the diffusion model. It uses a pre-trained diffusion model (either conservative or non-conservative) to estimate the "forces" (score function) that guide the movement of the latent vector. This allows for the generation of dynamic traj>

The script infers crucial model parameters (like `model_type`, `hidden_dim`, `diffusion_steps`, `beta_start`, `beta_end`, and `conservative` flag) directly from the attributes stored within the input HDF5 embedding file. This ensures compatibility with the specific diffusion model that generated the starting embedding.

### How it is Operated
The script requires a pre-trained diffusion model checkpoint and a starting latent embedding. The starting embedding is typically an output from `new_diff.py`.

**Key Arguments:**
*   `--diff_config`: Path to the `param_diff.yaml` file used for training the diffusion model.
*   `--diff_ckpt`: Path to the trained diffusion model checkpoint (e.g., `checkpoints/diffusion_checkpoint_exp1.pth`).
*   `--start_emb_dir`: Directory containing the starting generated embedding file (e.g., `latent_reps`).
*   `--start_exp_idx`: The experiment index used in the filename of the starting generated embedding (e.g., `1` for `generated_embeddings_exp1_non_conservative.h5`). This also implicitly selects the first sample (index 0) from within that HDF5 file.
*   `--start_conservative_mode`: Use this flag if the starting embedding is from a conservative diffusion run. Omit it for non-conservative runs.
*   `--num_steps`: The total number of simulation steps to perform.
*   `--temperature`: A float representing the temperature for the Langevin dynamics diffusion term. Set to `0.0` for deterministic-like trajectories (no thermal noise).
*   `--output_file`: The absolute path for the output HDF5 file where the Langevin trajectory will be saved.
*   `--time_step_size`: The time step size (`dt`) for Langevin dynamics (e.g., `1e-8`).
*   `--noise_level_t`: The integer `t` for `noise_level_sigma` (which is `sqrt_one_minus_alphas_cumprod[t]`). This parameter controls the noise level at which the score function is evaluated.
*   `--save_frequency`: Frequency (in steps) at which to save states to the output HDF5 file. Default is `1` (save every step).
*   `--score_clip_value`: Optional. Maximum norm for the score/force vector to prevent instability. If omitted, no clipping is applied.

**Example Usage:**
To run a short simulation starting from `latent_reps/generated_embeddings_exp1_non_conservative.h5`:

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

This command will generate an HDF5 file (e.g., `latent_reps/langevin_trajectory_trial.h5`) containing the sequence of latent vectors over the simulation.

## 2. Running Inference on Langevin Trajectories (`inference_old.py`)

After simulating dynamics in the latent space, you will have a trajectory of latent embeddings. To visualize these, you need to decode them back into 3D protein coordinates. For Langevin trajectories, the `inference_old.py` script is recommended due to its flexibility in accepting direct HDF5 input files.

**Example Usage (Decoding the trial Langevin trajectory):**

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
This will produce an HDF5 file (e.g., `structures/full_coords_langevin_trajectory_trial.h5`) containing the 3D coordinates of the protein over the simulated trajectory.

## 3. Converting HDF5 to XTC Trajectory (`convert_h5_to_xtc.py`)

The final 3D coordinates are stored in an HDF5 file. For visualization in molecular dynamics software (like VMD or PyMOL), it's often useful to convert this HDF5 file into an XTC trajectory format.

**Example Usage (Converting the decoded Langevin trajectory):**

```bash
python convert_h5_to_xtc.py \
    --h5_file structures/full_coords_langevin_trajectory_trial.h5 \
    --h5_key full_coords_langevin_trajectory_trial \
    --ref_pdb /scratch/asengar/long_sim/atlas/kinase_1ptq_perturn/heavy_chain.pdb \
    --output_xtc structures/full_coords_langevin_trajectory_trial.xtc
```
This command will generate an XTC file (e.g., `structures/full_coords_langevin_trajectory_trial.xtc`) which can be loaded into molecular visualization tools.


