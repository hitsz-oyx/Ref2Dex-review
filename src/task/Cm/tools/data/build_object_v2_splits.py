"""Create deterministic, dataset-stratified sequence splits for object-v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.task.Cm.dataset_object_v2 import _sequence_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    root = args.root.resolve()
    groups: Dict[str, List[Path]] = {}
    for sequence in _sequence_dirs(root):
        payload = json.loads((sequence / "shared" / "meta.json").read_text(encoding="utf-8"))
        groups.setdefault(str(payload.get("dataset_name", "unknown")), []).append(sequence)
    rng = np.random.default_rng(args.seed)
    splits = {"train": [], "val": [], "test": []}
    for name, values in sorted(groups.items()):
        values = list(values); rng.shuffle(values)
        n_test = max(1, round(len(values) * args.test_fraction)) if len(values) >= 3 else 0
        n_val = max(1, round(len(values) * args.val_fraction)) if len(values) - n_test >= 3 else 0
        splits["test"].extend(values[:n_test]); splits["val"].extend(values[n_test:n_test+n_val]); splits["train"].extend(values[n_test+n_val:])
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("train", "val", "test"):
        rels = sorted(str(path.relative_to(root)) for path in splits[name])
        path = args.output / f"{name}.txt"
        path.write_text("\n".join(rels) + "\n", encoding="utf-8")
        files[name] = str(path.name)
    descriptor = {"schema_name": "ref2dex_cm_object_v2_split", "root": str(root), "seed": args.seed,
                  "val_fraction": args.val_fraction, "test_fraction": args.test_fraction,
                  "train_split": files["train"], "val_split": files["val"], "test_split": files["test"],
                  "counts": {name: len(values) for name, values in splits.items()}}
    (args.output / "splits.json").write_text(json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(descriptor, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
