"""Train the V1.1.16 B1 FieldRealizer through BaseRunner."""
from __future__ import annotations

import argparse

from src.base import cleanup_distributed, load_config

from .field_runner import FieldRealizerRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src.task.CmDecoderv2.field_config:Config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    if args.device is not None:
        cfg.train.device = args.device
    try:
        FieldRealizerRunner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
