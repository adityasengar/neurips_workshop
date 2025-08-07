import h5py
import numpy as np
import sys

file_path = sys.argv[1]

try:
    with h5py.File(file_path, 'r') as f:
        if 'langevin_trajectory' in f:
            trajectory_data = f['langevin_trajectory'][:] # Read all data into memory
            
            if trajectory_data.ndim < 2:
                print("Error: Data does not have enough dimensions to be treated as frames with samples.")
                sys.exit(1)

            num_frames = trajectory_data.shape[0]
            num_samples_per_frame = trajectory_data.shape[1]

            if num_samples_per_frame < 2:
                print("Not enough samples per frame to calculate pairwise MSE (need at least 2).")
                sys.exit(0)

            print(f"Calculating MSE between consecutive samples within each frame for {file_path}:")
            for i in range(num_frames):
                frame_data = trajectory_data[i]
                print(f"  Frame {i}:")
                for j in range(num_samples_per_frame - 1):
                    sample1 = frame_data[j]
                    sample2 = frame_data[j+1]
                    mse = np.mean(np.square(sample1 - sample2))
                    print(f"    MSE between sample {j} and sample {j+1}: {mse:.6e}")
        else:
            print("Dataset 'langevin_trajectory' not found in the H5 file.")
except Exception as e:
    print(f"Error processing H5 file: {e}")


