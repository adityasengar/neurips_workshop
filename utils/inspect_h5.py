import h5py
import sys

file_path = sys.argv[1]

try:
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}:")
        for key in f.keys():
            print(f"- {key} (shape: {f[key].shape}, dtype: {f[key].dtype})")
except Exception as e:
    print(f"Error reading H5 file: {e}")

