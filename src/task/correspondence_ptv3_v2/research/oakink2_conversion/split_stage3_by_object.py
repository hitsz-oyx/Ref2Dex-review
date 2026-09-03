"""Create deterministic object-disjoint train/val views of OakInk2 Stage3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()
    source = Path(args.source_root).resolve()
    out = Path(args.output_root).resolve()
    files = sorted(source.glob("*.npz"))
    if not files:
        raise FileNotFoundError(source)
    object_ids: set[str] = set()
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            seq_id = str(np.asarray(data["seq_id"]).item())
        parts = seq_id.split("/")
        if len(parts) < 3:
            raise ValueError(f"Unexpected seq_id in {path}: {seq_id}")
        object_id = parts[-2]
        object_ids.add(object_id)
        rows.append((path, object_id, seq_id))
    ordered = sorted(object_ids)
    n_val = max(1, int(round(len(ordered) * float(args.val_fraction)))) if len(ordered) > 1 else 0
    val_objects = set(ordered[-n_val:]) if n_val else set()
    train_objects = set(ordered) - val_objects
    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for path, object_id, seq_id in rows:
        split = "val" if object_id in val_objects else "train"
        target = out / split / path.name
        if not target.exists():
            os.symlink(path, target)
        counts[split] += 1
    manifest = {
        "source_root": str(source), "output_root": str(out),
        "num_files": len(files), "num_objects": len(ordered),
        "train_objects": sorted(train_objects), "val_objects": sorted(val_objects),
        "file_counts": counts,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("num_files", "num_objects", "file_counts")}, indent=2))


if __name__ == "__main__":
    main()
