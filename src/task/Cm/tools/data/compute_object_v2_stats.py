"""Compute E1 per-dataset statistics for a Cm object-v2 cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.task.Cm.dataset_object_v2 import CmObjectV2Dataset, write_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--num-obj-points", type=int, default=512)
    parser.add_argument("--min-stride", type=int, default=1)
    parser.add_argument("--max-stride", type=int, default=10)
    parser.add_argument("--include-inactive", action="store_false", dest="active_only", default=True)
    args = parser.parse_args()
    dataset = CmObjectV2Dataset(
        args.root,
        num_obj_points=args.num_obj_points,
        num_hand_points=1538,
        min_stride=args.min_stride,
        max_stride=args.max_stride,
        active_only=args.active_only,
    )
    output = args.output or (args.root / "object_v2_statistics.json")
    payload = write_statistics(dataset, output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[cm-object-v2] wrote {output.resolve()}")


if __name__ == "__main__":
    main()
