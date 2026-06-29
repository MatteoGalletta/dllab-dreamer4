import os
import gdown
import zipfile

dataset_path = "pusht_cchi_v7_replay.zarr.zip"
extracted_dataset_path = "pusht_cchi_v7_replay.zarr"

# Download the demonstration dataset from Google Drive
if not os.path.isfile(dataset_path):
    file_id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=file_id, output=dataset_path, quiet=False)

# Extract the dataset
if not os.path.isdir(extracted_dataset_path):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dataset_path)
