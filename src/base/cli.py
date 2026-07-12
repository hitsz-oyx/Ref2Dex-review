from __future__ import annotations

import argparse

from .base_config import load_config


def build_train_parser(
    *,
    description: str,
    default_config: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="Python config reference.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dotted key=value syntax.",
    )
    parser.add_argument("--data", default=None, help="Optional override for data.train_path.")
    parser.add_argument("--output-dir", default=None, help="Optional override for train.output_dir.")
    parser.add_argument("--device", default=None, help="Optional override for train.device.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training logic. Launch with torchrun for multi-GPU execution.",
    )
    parser.add_argument("--local-rank", "--local_rank", default=None, type=int, help=argparse.SUPPRESS)
    return parser


def collect_override_keys(
    overrides: list[str],
) -> set[str]:
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


def load_train_config_from_args(
    args: argparse.Namespace,
):
    cfg = load_config(args.config, overrides=args.set)
    override_keys = collect_override_keys(args.set)

    if args.data is not None:
        cfg.data.train_path = args.data
        override_keys.add("data.train_path")
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
        override_keys.add("train.output_dir")
    if args.device is not None:
        cfg.train.device = args.device
        override_keys.add("train.device")
    if args.distributed:
        cfg.train.distributed.enable = True
        override_keys.add("train.distributed.enable")

    setattr(cfg, "_explicit_override_keys", set(override_keys))
    setattr(cfg, "_explicit_name", "name" in override_keys)
    setattr(cfg.wandb, "_explicit_name", "wandb.name" in override_keys)
    setattr(
        cfg.train,
        "_explicit_output_dir",
        bool(args.output_dir is not None or "train.output_dir" in override_keys),
    )
    return cfg


__all__ = [
    "build_train_parser",
    "collect_override_keys",
    "load_train_config_from_args",
]
