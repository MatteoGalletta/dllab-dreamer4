#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import zarr


DEFAULT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


def convert_zarr_to_npz(zarr_dir: Path, npz_path: Path) -> None:
    dataset_root = zarr.open(str(zarr_dir), mode="r")
    states = np.asarray(dataset_root["data"]["state"][:])
    actions = np.asarray(dataset_root["data"]["action"][:])
    images = np.asarray(dataset_root["data"]["img"][:])
    episode_ends = np.asarray(dataset_root["meta"]["episode_ends"][:], dtype=np.int64)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        states=states,
        actions=actions,
        images=images,
        episode_ends=episode_ends,
    )
    print(
        f"Saved {npz_path} | images={tuple(images.shape)} actions={tuple(actions.shape)} "
        f"states={tuple(states.shape)} episodes={int(len(episode_ends))}"
    )


def obtain_expert_trajectories(*, data_dir: Path, url: str, keep_zip: bool, force_redownload: bool) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "pusht.zip"
    extract_dir = data_dir / "pusht"
    zarr_dir = extract_dir / "pusht_cchi_v7_replay.zarr"
    npz_path = data_dir / "pusht_expert.npz"

    if force_redownload:
        zip_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        npz_path.unlink(missing_ok=True)

    if not npz_path.exists():
        if not zarr_dir.exists():
            if not zip_path.exists():
                print(f"Downloading expert PushT dataset from {url}")
                urllib.request.urlretrieve(url, zip_path)
                print(f"Downloaded archive to {zip_path}")
            print(f"Extracting {zip_path} -> {extract_dir}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            if not keep_zip:
                zip_path.unlink(missing_ok=True)
        convert_zarr_to_npz(zarr_dir, npz_path)
    else:
        print(f"NPZ already exists at {npz_path}")

    return npz_path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_data_dir = repo_root / "data" / "expert_trajectories"
    parser = argparse.ArgumentParser(description="Download and convert PushT expert trajectories to NPZ.")
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir))
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = obtain_expert_trajectories(
        data_dir=Path(args.data_dir),
        url=str(args.url),
        keep_zip=bool(args.keep_zip),
        force_redownload=bool(args.force_redownload),
    )
    print(f"Done. NPZ dataset is ready at: {npz_path}")


if __name__ == "__main__":
    main()
