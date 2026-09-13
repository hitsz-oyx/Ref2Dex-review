"""Train ObjectInteractionCm through the shared BaseRunner."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import cleanup_distributed, load_config
from src.task.ObjectInteractionCm.runner import ObjectInteractionCmRunner


DEFAULT_CONFIG = "src.task.ObjectInteractionCm.config:Config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ObjectInteractionCm V1.2.1.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Python config reference or YAML path.")
    parser.add_argument("--set", action="append", default=[], help="Override config with dotted key=value syntax.")
    parser.add_argument("--data", default=None, help="Optional override for data.root and data.train_path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", default=None, type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def _override_keys(overrides: list[str]) -> set[str]:
    return {item.split("=", 1)[0].lstrip("-").strip() for item in overrides if "=" in item}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    override_keys = _override_keys(args.set)
    if args.data is not None:
        cfg.data.root = args.data
        cfg.data.train_path = args.data
        override_keys.update({"data.root", "data.train_path"})
    if args.device is not None:
        cfg.train.device = args.device
        override_keys.add("train.device")
    if args.distributed:
        cfg.train.distributed.enable = True
        override_keys.add("train.distributed.enable")
    setattr(cfg, "_explicit_override_keys", override_keys)
    has_index = str(getattr(cfg.data, "index_path", "") or "").strip()
    has_root = str(getattr(cfg.data, "root", "") or getattr(cfg.data, "train_path", "")).strip()
    if not has_index and not has_root:
        raise ValueError("Set data.index_path or data.root/data.train_path.")
    try:
        ObjectInteractionCmRunner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
