# Project Overview

This README provides an overview of utility scripts for analyzing and comparing molecular dynamics simulation data, focusing on latent space embeddings and coordinate data.

---

# `compare_outputs.py`

Compares two HDF5 files containing 3D atomic coordinates by calculating the Mean Squared Error (MSE) between datasets. Useful for quantifying similarity.

## How to Run

```bash
python compare_outputs.py <file1_path> <key1> <file2_path> <key2>
```

---

# `t-test.py`

Analyzes and visualizes the separation of latent space embedding datasets (clusters). It performs PCA for 2D visualization, calculates Euclidean distances between cluster centroids, and conducts t-tests on principal components to assess statistical differences. Outputs include a PCA scatter plot and statistical comparisons.

## How to Run

```bash
python t-test.py <h5_file1> <h5_file2> [h5_file3] [--output_plot <plot_filename>] [--output_stats <stats_filename>]
```

---

# `calculate_frame_variance.py`

Analyzes the internal variability within each time step (frame) of a Langevin dynamics trajectory stored in an HDF5 file. It calculates the Mean Squared Error (MSE) between consecutive samples within each frame to quantify latent space diversity.

## How to Run

```bash
python latent_reps/calculate_frame_variance.py <path_to_langevin_trajectory_h5_file>
```

---

# `inspect_h5.py`

A simple utility for quickly inspecting the contents of an HDF5 file. It lists all top-level datasets (keys), their shapes, and data types (`dtype`) without loading full data.

## How to Run

```bash
python latent_reps/inspect_h5.py <path_to_your_h5_file>
```

