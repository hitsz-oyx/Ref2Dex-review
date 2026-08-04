"""Create a deterministic sequence-level split for a GRAB Cm Stage 4 cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--path-prefix", default="", help="Dataset directory prefix stored in split entries.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0 or not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("Ratios must satisfy 0 < train_ratio < 1 and 0 <= val_ratio < 1.")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    data_root = args.data_root.resolve()
    sequence_dirs = sorted(
        path.parent for path in data_root.glob("*/*/shared.npz")
        if (path.parent / "left.npz").exists() or (path.parent / "right.npz").exists()
    )
    if len(sequence_dirs) < 3:
        raise ValueError(f"Need at least 3 complete sequences under {data_root}, found {len(sequence_dirs)}.")

    sequence_dirs = list(sequence_dirs)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sequence_dirs)
    total = len(sequence_dirs)
    train_count = max(1, int(round(total * args.train_ratio)))
    val_count = max(1, int(round(total * args.val_ratio)))
    if train_count + val_count >= total:
        val_count = total - train_count - 1
    train_dirs = sequence_dirs[:train_count]
    val_dirs = sequence_dirs[train_count:train_count + val_count]
    test_dirs = sequence_dirs[train_count + val_count:]

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path_prefix = str(args.path_prefix).strip("/")
    for name, dirs in (("train", train_dirs), ("val", val_dirs), ("test", test_dirs)):
        entries: list[str] = []
        for sequence_dir in sorted(dirs):
            for side in ("left", "right"):
                hand_path = sequence_dir / f"{side}.npz"
                if hand_path.exists():
                    entry = hand_path.relative_to(data_root).as_posix()
                    entries.append(f"{path_prefix}/{entry}" if path_prefix else entry)
        (output_root / f"{name}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    summary = {
        "name": output_root.name,
        "task": "cm_v3",
        "dataset": [path_prefix or "grab"],
        "seed": args.seed,
        "group_by": "sequence",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "num_sequences": total,
        "num_train_sequences": len(train_dirs),
        "num_val_sequences": len(val_dirs),
        "num_test_sequences": len(test_dirs),
        "train_split": "train.txt",
        "val_split": "val.txt",
        "test_split": "test.txt",
    }
    (output_root / "split.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
