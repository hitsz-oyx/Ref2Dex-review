"""Train CmDecoderv2 through the shared BaseRunner."""
from __future__ import annotations

import argparse

from src.base import cleanup_distributed, load_config

from .runner import CmDecoderV2Runner


DEFAULT_CONFIG = "src.task.CmDecoderv2.config:Config"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device", default=None)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", default=None, type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides=args.set)
    override_keys = {item.split("=", 1)[0].strip().lstrip("-") for item in args.set if "=" in item}
    if args.device is not None:
        cfg.train.device = args.device
        override_keys.add("train.device")
    if args.distributed:
        cfg.train.distributed.enable = True
        override_keys.add("train.distributed.enable")
    setattr(cfg, "_explicit_override_keys", override_keys)
    try:
        CmDecoderV2Runner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
