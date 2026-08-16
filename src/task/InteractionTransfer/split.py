"""InteractionTransfer sequence-level split (V0.8).

生成分案要求的 sequence 级 80/10/10 split，每行只写 sequence 相对路径
（如 ``s1/airplane_fly_1``），不区分左右手；dataset 侧仍然只读 right。
划分逻辑与 ``process/GRAB/build_cm_split.py`` 一致（sorted glob + seed shuffle），
因此与 Cm 的 full_grab_v1 split 保持同一 sequence 划分。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="GRAB stage4 cache 根目录（其下为 s*/seq/shared.npz）")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="例如 src/task/InteractionTransfer/splits/grab_seed42")
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
    split = {
        "train": sequence_dirs[:train_count],
        "val": sequence_dirs[train_count:train_count + val_count],
        "test": sequence_dirs[train_count + val_count:],
    }

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for name, dirs in split.items():
        entries = sorted(d.relative_to(data_root).as_posix() for d in dirs)
        (output_root / f"{name}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")

    summary = {
        "name": output_root.name,
        "task": "interaction_transfer",
        "seed": args.seed,
        "group_by": "sequence",
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "num_sequences": total,
        "num_train_sequences": len(split["train"]),
        "num_val_sequences": len(split["val"]),
        "num_test_sequences": len(split["test"]),
    }
    (output_root / "split.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
