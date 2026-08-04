from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import cleanup_distributed, load_config
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner


DEFAULT_CONFIG = "src.task.correspondence_ptv3_v2.config:Config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PTv3 v2 correspondence model.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Python config reference.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted key=value syntax.")
    parser.add_argument("--data", default=None, help="Optional override for data.train_path.")
    parser.add_argument("--device", default=None, help="Optional override for train.device.")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed training logic.")
    parser.add_argument("--local-rank", "--local_rank", default=None, type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _override_keys(overrides: list[str]) -> set[str]:
    keys: set[str] = set()
    for override in overrides:
        item = override.strip()
        if item.startswith("--"):
            item = item[2:]
        if "=" not in item:
            continue
        key, _ = item.split("=", 1)
        keys.add(key.strip())
    return keys


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    override_keys = _override_keys(args.set)
    if args.data is not None:
        cfg.data.train_path = args.data
        override_keys.add("data.train_path")
    if args.device is not None:
        cfg.train.device = args.device
        override_keys.add("train.device")
    if args.distributed:
        cfg.train.distributed.enable = True
        override_keys.add("train.distributed.enable")
    setattr(cfg, "_explicit_override_keys", set(override_keys))
    setattr(cfg, "_explicit_name", "name" in override_keys)
    setattr(cfg.wandb, "_explicit_name", "wandb.name" in override_keys)
    if not str(cfg.data.train_path).strip():
        raise ValueError("data.train_path is empty. Pass --data or set data.train_path.")
    try:
        CorrespondencePTV3V2Runner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
