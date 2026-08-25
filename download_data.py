#!/usr/bin/env python3
"""Download the legacy LeWM PushT HDF5 dataset.

The main experiments in this repository use the smaller public NPZ replay
dataset downloaded by ``scripts/download_pusht_npz.py``. This utility is kept
for reproducing legacy HDF5-based experiments and can require substantial disk
space after decompression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import zstandard as zstd
from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "quentinll/lewm-pusht"
DEFAULT_FILENAME = "pusht_expert_train.h5.zst"
DEFAULT_OUTPUT = "data/expert_trajectories/pusht_expert_train.h5"


def download_and_decompress(*, output_path: Path, repo_id: str, filename: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"Dataset already exists at {output_path}; skipping download.")
        return

    print(f"Downloading {filename} from Hugging Face dataset {repo_id}.")
    compressed_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
    )
    print(f"Decompressing {compressed_path} -> {output_path}")
    with Path(compressed_path).open("rb") as compressed_file:
        with output_path.open("wb") as uncompressed_file:
            zstd.ZstdDecompressor().copy_stream(compressed_file, uncompressed_file)


def inspect_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as dataset:
        print("Dataset keys:", list(dataset.keys()))
        for key, value in dataset.items():
            shape = getattr(value, "shape", None)
            print(f"  {key}: {shape if shape is not None else type(value).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--no-inspect", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_and_decompress(
        output_path=args.output,
        repo_id=str(args.repo_id),
        filename=str(args.filename),
    )
    if not args.no_inspect:
        inspect_hdf5(args.output)


if __name__ == "__main__":
    main()
