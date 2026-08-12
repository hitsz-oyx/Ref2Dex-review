"""生成 V17.1 time-to-effect 元数据并审计分布。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.task.InteractionDynamics.dataset_v17 import (
    dominant_sequences, split_sequences, time_to_effect_for_paths)


def summary(values: torch.Tensor) -> dict:
    valid = values >= 0
    buckets = {"missing": ~valid, "0_10": valid & (values < 10),
               "10_30": (values >= 10) & (values < 30),
               "30_100": (values >= 30) & (values < 100), "100_plus": values >= 100}
    present = values[valid].float()
    return {"samples": len(values), "valid_ratio": float(valid.float().mean()),
            "median": float(present.median()) if len(present) else None,
            "mean": float(present.mean()) if len(present) else None,
            "p90": float(torch.quantile(present, .9)) if len(present) else None,
            "bucket_counts": {name: int(mask.sum()) for name, mask in buckets.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v17-config", default="src/task/InteractionDynamics/configs/v17_full.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.v17_config).read_text())
    paths = dominant_sequences(config["data"]["root"], config["data"]["dominant_hand_manifest"])
    split = split_sequences(paths, config["data"]["split_seed"])
    result = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        values = time_to_effect_for_paths(config["data"]["root"], getattr(split, name))
        torch.save(values, args.output / f"{name}.pt")
        result[name] = summary(values)
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
