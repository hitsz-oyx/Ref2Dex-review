from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import load_config
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner


DEFAULT_CONFIG = "src.task.correspondence_ptv3.config:Config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PTv3 correspondence model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Python config reference.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted key=value syntax.")
    parser.add_argument("--data", default=None, help="Optional override for data.train_path.")
    parser.add_argument("--output-dir", default=None, help="Optional override for train.output_dir.")
    parser.add_argument("--device", default=None, help="Optional override for train.device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.data is not None:
        cfg.data.train_path = args.data
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.device is not None:
        cfg.train.device = args.device
    if not str(cfg.data.train_path).strip():
        raise ValueError("data.train_path is empty. Pass --data or set data.train_path to a Stage 3 dataset root.")

    CorrespondencePTV3Runner(cfg, mode="train").run()


if __name__ == "__main__":
    main()
