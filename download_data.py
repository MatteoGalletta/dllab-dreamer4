import os
import zstandard as zstd
import h5py
from huggingface_hub import hf_hub_download

decompressed_file_path = "pusht_expert_train.h5"
if not os.path.exists(decompressed_file_path):
    # --- 1. Download the compressed dataset ---
    print("Downloading dataset from Hugging Face (this might take a while, it's 13GB)...")
    compressed_file_path = hf_hub_download(
        repo_id="quentinll/lewm-pusht", 
        filename="pusht_expert_train.h5.zst", 
        repo_type="dataset"
    )


    # --- 2. Decompress the .zst file ---
    print("Decompressing the .zst file to .h5...")
    with open(compressed_file_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with open(decompressed_file_path, 'wb') as uncompressed_file:
            dctx.copy_stream(compressed_file, uncompressed_file)
    print("Decompression complete!")
else:
    print("Skipping download and decompression.")

# --- 3. Import and inspect the HDF5 data ---
print("Loading HDF5 dataset...")
# Open in read-only mode
dataset = h5py.File(decompressed_file_path, 'r')

# Print the top-level keys to see how the dataset is structured
print("\nDataset Keys:", list(dataset.keys()))

# Example: Accessing the actual data arrays (keys might vary slightly based on repo structure)
# Usually, HDF5 robotics datasets have keys like 'data', 'actions', 'images', or 'obs'
for key in dataset.keys():
    print(f"Shape of {key}: {dataset[key].shape}")

# Remember to close the file when you are completely done using it in your dataloader
# dataset.close()