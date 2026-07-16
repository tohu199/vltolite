#!/usr/bin/env python3
"""Download and extract MNIST into data/kd_datasets/11_MNIST/."""

import argparse
from pathlib import Path
from typing import Tuple

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.kd_dataloader import MNISTDataset


def download_mnist(data_dir: Path) -> Tuple[int, int]:
    data_dir.mkdir(parents=True, exist_ok=True)
    root = str(data_dir)

    MNISTDataset._ensure_downloaded(root)
    train = MNISTDataset(root, split="train")
    test = MNISTDataset(root, split="test")
    return len(train), len(test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MNIST for VL2Lite.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/kd_datasets/11_MNIST"),
        help="Directory to store MNIST (default: data/kd_datasets/11_MNIST)",
    )
    args = parser.parse_args()

    print(f"Downloading MNIST to {args.data_dir.resolve()} ...")
    n_train, n_test = download_mnist(args.data_dir)
    print(f"Done. train={n_train}, test={n_test}")


if __name__ == "__main__":
    main()
