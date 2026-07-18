"""Evaluate a BaseRunner CmAction checkpoint."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import cleanup_distributed, load_checkpoint, load_config, task_config_from_dict
from src.task.Cm.runner import CmActionRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CmAction checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", default=None, type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is None:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = task_config_from_dict(checkpoint["config"])
    else:
        cfg = load_config(args.config)
    cfg.train.device = args.device
    if args.distributed:
        cfg.train.distributed.enable = True
    try:
        CmActionRunner(cfg, mode="eval", checkpoint=args.checkpoint).run()
    finally:
        cleanup_distributed()
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
