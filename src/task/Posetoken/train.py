"""训练 synthetic MANO Static PoseToken。"""
from __future__ import annotations

import argparse

from src.base import cleanup_distributed, load_config
from src.task.Posetoken.config import validate_config
from src.task.Posetoken.runner import PoseTokenRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src.task.Posetoken.config:Config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    validate_config(cfg)
    if args.device:
        cfg.train.device = args.device
    try:
        PoseTokenRunner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
