"""Train frozen-Cm Inspire F1 action reconstruction."""
from __future__ import annotations

import argparse

from src.base import cleanup_distributed, load_config
from src.task.CmDecoder.runner import CmDecoderRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src.task.CmDecoder.config:Config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.device is not None:
        cfg.train.device = args.device
    try:
        CmDecoderRunner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
