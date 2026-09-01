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
    parser.add_argument(
        "--source-split-json",
        type=Path,
        default=None,
        help="Reuse an existing sequence assignment and filter it to a Scene Cache root.",
    )
    return parser.parse_args()


def _read_source_scene_assignment(
    data_root: Path,
    source_split_json: Path,
) -> tuple[dict[str, list[Path]], dict]:
    source_split_json = source_split_json.resolve()
    source = json.loads(source_split_json.read_text(encoding="utf-8"))
    available = {
        sequence_dir.relative_to(data_root).as_posix(): sequence_dir
        for sequence_dir in sorted(
            path.parent.parent for path in data_root.glob("*/*/shared/raw_frame_id.npy")
        )
    }
    if len(available) < 3:
        raise ValueError(f"Need at least 3 Scene Cache sequences under {data_root}, found {len(available)}.")

    assignments: dict[str, list[Path]] = {}
    assigned: set[str] = set()
    for split_name in ("train", "val", "test"):
        list_name = source.get(f"{split_name}_split")
        if not list_name:
            raise ValueError(f"{source_split_json} is missing {split_name}_split")
        list_path = Path(str(list_name))
        if not list_path.is_absolute():
            list_path = source_split_json.parent / list_path
        keys: set[str] = set()
        for line in list_path.read_text(encoding="utf-8").splitlines():
            parts = Path(line.strip()).parts
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(f"Cannot derive sequence from split entry {line!r}")
            keys.add(Path(parts[-3], parts[-2]).as_posix())
        overlap = assigned & keys
        if overlap:
            raise ValueError(f"Source sequence splits overlap: {sorted(overlap)[:3]}")
        assigned.update(keys)
        assignments[split_name] = [available[key] for key in sorted(keys & available.keys())]

    missing_assignment = set(available) - assigned
    if missing_assignment:
        raise ValueError(f"Scene sequences missing from source split: {sorted(missing_assignment)[:3]}")
    return assignments, source


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0 or not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("Ratios must satisfy 0 < train_ratio < 1 and 0 <= val_ratio < 1.")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    data_root = args.data_root.resolve()
    source = None
    if args.source_split_json is not None:
        assignments, source = _read_source_scene_assignment(data_root, args.source_split_json)
        train_dirs = assignments["train"]
        val_dirs = assignments["val"]
        test_dirs = assignments["test"]
        scene_cache = True
    else:
        scene_sequence_dirs = sorted(
            path.parent.parent for path in data_root.glob("*/*/shared/raw_frame_id.npy")
        )
        if scene_sequence_dirs:
            sequence_dirs = scene_sequence_dirs
            scene_cache = True
        else:
            sequence_dirs = sorted(
                path.parent for path in data_root.glob("*/*/shared.npz")
                if (path.parent / "left.npz").exists() or (path.parent / "right.npz").exists()
            )
            scene_cache = False
        if len(sequence_dirs) < 3:
            raise ValueError(f"Need at least 3 complete sequences under {data_root}, found {len(sequence_dirs)}.")
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

    total = len(train_dirs) + len(val_dirs) + len(test_dirs)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path_prefix = str(args.path_prefix).strip("/")
    for name, dirs in (("train", train_dirs), ("val", val_dirs), ("test", test_dirs)):
        entries: list[str] = []
        for sequence_dir in sorted(dirs):
            if scene_cache:
                # Cm scene loader expects sequence directories; the shared
                # frame-id file is only the discovery marker.
                entry = sequence_dir.relative_to(data_root).as_posix()
                entries.append(f"{path_prefix}/{entry}" if path_prefix else entry)
            else:
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
        "seed": int(source.get("seed", args.seed)) if source is not None else args.seed,
        "group_by": "sequence",
        "train_ratio": len(train_dirs) / total,
        "val_ratio": len(val_dirs) / total,
        "test_ratio": len(test_dirs) / total,
        "num_sequences": total,
        "num_train_sequences": len(train_dirs),
        "num_val_sequences": len(val_dirs),
        "num_test_sequences": len(test_dirs),
        "train_split": "train.txt",
        "val_split": "val.txt",
        "test_split": "test.txt",
    }
    if source is not None:
        summary["source_split_json"] = str(args.source_split_json.resolve())
        summary["source_num_sequences"] = int(source.get("num_sequences", total))
    (output_root / "split.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
