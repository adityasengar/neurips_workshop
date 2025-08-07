# Project Overview

This README provides an overview of several utility scripts used in this project for analyzing and comparing molecular dynamics simulation data, particularly focusing on latent space embeddings and coordinate data.

---

# `compare_outputs.py`

This script is a utility for quantitatively comparing two HDF5 files containing coordinate data, typically 3D atomic coordinates of molecular structures.

## What it Does

It loads coordinate data from two specified HDF5 files and their respective dataset keys. It then checks if their shapes match. If they do, it computes the Mean Squared Error (MSE) between the two datasets. A lower MSE indicates greater similarity between the coordinate sets.

## How to Run

```bash
python compare_outputs.py <file1_path> <key1> <file2_path> <key2>
```

## Example Usage

```bash
python compare_outputs.py structures/full_coords_diff_exp1_conservative.h5 full_coords_diff structures/full_coords_diff_old_exp1_conservative.h5 full_coords_diff_old
```

**Expected Output Example:**

```
Comparing structures/full_coords_diff_exp1_conservative.h5 (key: full_coords_diff) and structures/full_coords_diff_old_exp1_conservative.h5 (key: full_coords_diff_old)
Loaded structures/full_coords_diff_exp1_conservative.h5 with shape: (100, 402, 3)
Loaded structures/full_coords_diff_old_exp1_conservative.h5 with shape: (100, 402, 3)
Mean Squared Error (MSE) between the two datasets: 0.00012345
```

---

# `t-test.py`

This script is designed to analyze and visualize the separation of different latent space embedding datasets (referred to as "clusters").

## What it Does

1.  **Data Loading and Preparation**: Loads data from 2 or 3 specified HDF5 files. If 3D, it reshapes the data to 2D.
2.  **Dimensionality Reduction (PCA)**: Combines all loaded datasets, scales them, and applies Principal Component Analysis (PCA) to reduce the data to two principal components (PC1 and PC2) for 2D visualization.
3.  **Cluster Comparison**: Performs pairwise comparisons between the input datasets:
    *   **Euclidean Distance**: Calculates the distance between the centroids of the clusters in the PCA space.
    *   **T-test**: Conducts an independent t-test on PC1 and PC2 values to determine if the cluster means are statistically different.
4.  **Visualization**: Generates a scatter plot of the data in the 2D PCA space, with each dataset represented by a different color and label, allowing for visual inspection of their overlap or separation. The plot is saved as a PNG image.
5.  **Statistics Output**: The pairwise comparison statistics (Euclidean distance and t-test p-values) are saved to a text file.

## How to Run

```bash
python t-test.py <h5_file1> <h5_file2> [h5_file3] [--output_plot <plot_filename>] [--output_stats <stats_filename>]
```

*   `<h5_file1>`, `<h5_file2>`, `[h5_file3]`: Paths to 2 or 3 HDF5 files containing latent space embeddings.
*   `--output_plot`: (Optional) Filename to save the PCA overlap plot (default: `pca_overlap_plot.png`).
*   `--output_stats`: (Optional) Filename to save the cluster comparison statistics (default: `cluster_stats.txt`).

## Example Usage

**Comparing 2 HDF5 files:**

```bash
python t-test.py \
    latent_reps/generated_embeddings_exp1_non_conservative.h5 \
    latent_reps/pooled_embedding.h5 \
    --output_plot pca_comparison_2files.png \
    --output_stats stats_2files.txt
```

**Comparing 3 HDF5 files:**

```bash
python t-test.py \
    latent_reps/generated_embeddings_exp1_non_conservative.h5 \
    latent_reps/pooled_embedding.h5 \
    latent_reps/langevin_trajectory_exp2_non_conservative_short_temp0.1_dt1e-5.h5 \
    --output_plot pca_comparison_3files.png \
    --output_stats stats_3files.txt
```

**Expected Output (console and files):**

Console output will show loading messages. Statistics will be saved to the specified `--output_stats` file, and the plot to `--output_plot`.

Example `stats_3files.txt` content:

```
----------------------------------------
  Comparison: 'generated_embeddings_exp1_non_conservative.h5' vs. 'pooled_embedding.h5'
----------------------------------------
Distance between centroids: 12.34
T-test on PC1: p-value = 1.23e-10
T-test on PC2: p-value = 4.56e-08

----------------------------------------
  Comparison: 'generated_embeddings_exp1_non_conservative.h5' vs. 'langevin_trajectory_exp2_non_conservative_short_temp0.1_dt1e-5.h5'
----------------------------------------
Distance between centroids: 5.67
T-test on PC1: p-value = 7.89e-05
T-test on PC2: p-value = 1.01e-03

----------------------------------------
  Comparison: 'pooled_embedding.h5' vs. 'langevin_trajectory_exp2_non_conservative_short_temp0.1_dt1e-5.h5'
----------------------------------------
Distance between centroids: 8.90
T-test on PC1: p-value = 2.34e-07
T-test on PC2: p-value = 5.67e-06
```

---

# `calculate_frame_variance.py`

This script analyzes the internal variability within each "frame" (or time step) of a Langevin dynamics trajectory stored in an HDF5 file.

## What it Does

It loads a dataset named `langevin_trajectory` from the specified HDF5 file. For each time frame, it calculates the Mean Squared Error (MSE) between consecutive samples within that frame. This helps quantify the diversity or spread of the latent space embeddings at each specific time point in the simulation.

## How to Run

```bash
python latent_reps/calculate_frame_variance.py <path_to_langevin_trajectory_h5_file>
```

## Example Usage

```bash
python latent_reps/calculate_frame_variance.py latent_reps/langevin_trajectory_example.h5
```

**Expected Output Example:**

```
Calculating MSE between consecutive samples within each frame for latent_reps/langevin_trajectory_example.h5:
  Frame 0:
    MSE between sample 0 and sample 1: 1.234567e-04
    MSE between sample 1 and sample 2: 9.876543e-05
  Frame 1:
    MSE between sample 0 and sample 1: 5.432109e-05
    ...
```

---

# `inspect_h5.py`

This script is a simple utility for quickly inspecting the contents of an HDF5 file.

## What it Does

It takes a single HDF5 file path as a command-line argument and then lists all top-level datasets (keys) within that file, along with their shapes and data types (`dtype`). This provides a quick overview of the HDF5 file's structure without loading the full data into memory.

## How to Run

```bash
python latent_reps/inspect_h5.py <path_to_your_h5_file>
```

## Example Usage

```bash
python latent_reps/inspect_h5.py latent_reps/pooled_embedding.h5
```

**Expected Output Example:**

```
Keys in latent_reps/pooled_embedding.h5:
- pooled_embedding (shape: (20002, 100), dtype: float32)
```
