import h5py
import numpy as np

def compare_h5_coords(file1: str, key1: str, file2: str, key2: str):
    """Loads coordinate data from two HDF5 files and computes MSE."""
    print(f"Comparing {file1} (key: {key1}) and {file2} (key: {key2})")
    
    try:
        with h5py.File(file1, 'r') as f1:
            coords1 = f1[key1][:]
        print(f"Loaded {file1} with shape: {coords1.shape}")
    except Exception as e:
        print(f"Error loading {file1}: {e}")
        return

    try:
        with h5py.File(file2, 'r') as f2:
            coords2 = f2[key2][:]
        print(f"Loaded {file2} with shape: {coords2.shape}")
    except Exception as e:
        print(f"Error loading {file2}: {e}")
        return

    if coords1.shape != coords2.shape:
        print(f"Error: Shapes mismatch! {coords1.shape} vs {coords2.shape}")
        return

    # Compute Mean Squared Error
    mse = np.mean((coords1 - coords2)**2)
    print(f"Mean Squared Error (MSE) between the two datasets: {mse:.8f}")

    # Optionally, compute Root Mean Squared Deviation (RMSD) if desired
    # rmsd = np.sqrt(mse)
    # print(f"Root Mean Squared Deviation (RMSD): {rmsd:.8f}")

if __name__ == "__main__":
    file_new = "structures/full_coords_diff_exp1_conservative.h5"
    key_new = "full_coords_diff"
    file_old = "structures/full_coords_diff_old_exp1_conservative.h5"
    key_old = "full_coords_diff_old"

    compare_h5_coords(file_new, key_new, file_old, key_old)

