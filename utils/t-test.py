import argparse
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from numpy.linalg import norm
import itertools
import os
import sys

def load_and_prepare_data(filepath):
    """
    Loads data from an HDF5 file and reshapes it to 2D if necessary.
    """
    with h5py.File(filepath, 'r') as f:
        key = list(f.keys())[0]
        print(f"Loading key '{key}' from {filepath}")
        data = f[key][:]
    
    print(f"  > Original shape: {data.shape}")

    if data.ndim == 3:
        num_features = data.shape[-1]
        data = data.reshape(-1, num_features)
        print(f"  > Reshaped to: {data.shape}")
    
    if data.ndim != 2:
        raise ValueError(f"Data from {filepath} is not 2D or 3D, but {data.ndim}D. Cannot process.")
        
    return data

def compare_clusters(cluster1, cluster2, name1, name2):
    """Calculates and prints the Euclidean distance and t-test results for two clusters."""
    print("-" * 40)
    print(f"  Comparison: '{name1}' vs. '{name2}'")
    print("-" * 40)
    
    # Euclidean Distance
    centroid1 = np.mean(cluster1, axis=0)
    centroid2 = np.mean(cluster2, axis=0)
    distance = norm(centroid1 - centroid2)
    print(f"Distance between centroids: {distance:.2f}")

    # Statistical T-test
    # Ensure there are enough samples for t-test
    if cluster1.shape[0] > 1 and cluster2.shape[0] > 1:
        ttest_pc1 = ttest_ind(cluster1[:, 0], cluster2[:, 0], equal_var=False)
        ttest_pc2 = ttest_ind(cluster1[:, 1], cluster2[:, 1], equal_var=False)
        print(f"T-test on PC1: p-value = {ttest_pc1.pvalue:.2e}")
        print(f"T-test on PC2: p-value = {ttest_pc2.pvalue:.2e}\n")
    else:
	print("Not enough samples in one or both clusters for t-test (min 2 samples required).\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize latent space embeddings from multiple HDF5 files.")
    parser.add_argument("h5_files", nargs='+', help="Paths to 2 or 3 HDF5 files containing latent space embeddings.")
    parser.add_argument("--output_plot", type=str, default="pca_overlap_plot.png", help="Filename to save the PCA overlap plot.")
    parser.add_argument("--output_stats", type=str, default="cluster_stats.txt", help="Filename to save the cluster comparison statistics.")

    args = parser.parse_args()

    if not (2 <= len(args.h5_files) <= 3):
        print("Error: Please provide exactly 2 or 3 HDF5 file paths.")
        parser.print_help()
        return

    # Define a set of distinct colors for plotting
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    try:
        # --- Load, Scale, and PCA ---
        all_data_list = [load_and_prepare_data(path) for path in args.h5_files]
        data_lengths = [data.shape[0] for data in all_data_list]
        combined_data = np.vstack(all_data_list)
        
       	scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
       	pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        
       	split_indices = np.cumsum(data_lengths)[:-1]
        pc_data_split = np.split(principal_components, split_indices)
        
        # Generate plot labels from filenames
        plot_labels = [os.path.basename(f) for f in args.h5_files]
        plot_colors = default_colors[:len(args.h5_files)]

        # --- Perform Pairwise Comparisons ---
        print("\n" + "="*40)
        print("  Pairwise Cluster Separation Analysis")
        print("="*40)
        
        # Redirect print output to a file for statistics
        original_stdout = sys.stdout
        with open(args.output_stats, 'w') as f_stats:
            sys.stdout = f_stats
            for i, j in itertools.combinations(range(len(pc_data_split)), 2):
                compare_clusters(pc_data_split[i], pc_data_split[j], plot_labels[i], plot_labels[j])
        sys.stdout = original_stdout # Restore stdout
        print(f"Cluster comparison statistics saved to {args.output_stats}")

        # --- Create the Plot ---
        plt.style.use('seaborn-whitegrid') # Use a more widely compatible seaborn style
        fig, ax = plt.subplots(figsize=(12, 9))

        for i, pc_data in enumerate(pc_data_split):
            ax.scatter(pc_data[:, 0], pc_data[:, 1], 
                       label=plot_labels[i], color=plot_colors[i],
                       alpha=0.6, s=15)

        ax.set_title('Multi-File PCA Overlap Plot', fontsize=16)
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
        ax.legend(loc='best', markerscale=2, fontsize=10)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"PCA overlap plot saved to {args.output_plot}")

    except FileNotFoundError as e:
        print(f"\nERROR: File not found -> {e.filename}")
        print("Please ensure all provided HDF5 file paths are correct.")
    except ValueError as e:
        print(f"\nERROR: Data processing issue -> {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()


