#!/usr/bin/env python3
"""Compute the absolute step budget for a fixed epoch count."""

from __future__ import annotations

import argparse
import math


def compute_max_steps(epochs: int, train_rows_per_epoch: int, global_batch_size: int) -> int:
    """Return ceil(epochs * rows / global_batch), with strict positive inputs."""

    epochs = int(epochs)
    rows = int(train_rows_per_epoch)
    batch = int(global_batch_size)
    if epochs <= 0 or rows <= 0 or batch <= 0:
        raise ValueError("epochs, train_rows_per_epoch and global_batch_size must be positive")
    return int(math.ceil(epochs * rows / batch))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--global-batch", type=int, required=True)
    args = parser.parse_args()
    print(compute_max_steps(args.epochs, args.train_rows, args.global_batch))


if __name__ == "__main__":
    main()
