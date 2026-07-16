#!/usr/bin/env python3
"""Download and extract CIFAR-10 into data/kd_datasets/10_CIFAR10/."""

import argparse
from pathlib import Path
from typing import Tuple

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.kd_dataloader import CIFAR10Dataset


def download_cifar10(data_dir: Path) -> Tuple[int, int]:
    data_dir.mkdir(parents=True, exist_ok=True)
    root = str(data_dir)

    CIFAR10Dataset._ensure_downloaded(root)
    train = CIFAR10Dataset(root, split="train")
    test = CIFAR10Dataset(root, split="test")
    return len(train), len(test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 for VL2Lite.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/kd_datasets/10_CIFAR10"),
        help="Directory to store CIFAR-10 (default: data/kd_datasets/10_CIFAR10)",
    )
    args = parser.parse_args()

    print(f"Downloading CIFAR-10 to {args.data_dir.resolve()} ...")
    n_train, n_test = download_cifar10(args.data_dir)
    print(f"Done. train={n_train}, test={n_test}")


if __name__ == "__main__":
    main()
