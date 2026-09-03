"""Evaluate an ObjectInteractionCm checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import cleanup_distributed, load_checkpoint, load_config, task_config_from_dict
from src.task.ObjectInteractionCm.runner import ObjectInteractionCmRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ObjectInteractionCm checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--distributed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
    else:
        payload = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = task_config_from_dict(payload["config"])
    cfg.train.device = args.device
    if args.distributed:
        cfg.train.distributed.enable = True
    try:
        runner = ObjectInteractionCmRunner(cfg, mode="eval", checkpoint=args.checkpoint)
        metrics = runner.evaluate_all() if args.split == "val" else runner.evaluate_test_all()
        if runner.is_primary:
            for key, value in metrics.items():
                print(f"{key}: {value:.6g}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
