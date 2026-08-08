"""Train InteractionDynamics V1 through the shared BaseRunner."""
from __future__ import annotations

import argparse

from src.base import cleanup_distributed, load_config
from src.task.InteractionDynamics.config import validate_config
from src.task.InteractionDynamics.runner import InteractionDynamicsRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src.task.InteractionDynamics.config:Config")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device"); parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    validate_config(cfg)
    override_keys = {item.lstrip("-").split("=", 1)[0] for item in args.set if "=" in item}
    if args.device:
        cfg.train.device = args.device; override_keys.add("train.device")
    if args.distributed:
        cfg.train.distributed.enable = True; override_keys.add("train.distributed.enable")
    setattr(cfg, "_explicit_override_keys", override_keys)
    try:
        InteractionDynamicsRunner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
