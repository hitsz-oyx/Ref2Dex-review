from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import load_checkpoint, load_config, task_config_from_dict
from src.task.corresponse_v1.runner import CorrResponseRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a StaticHOCNet checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or checkpoint.pt path.")
    parser.add_argument("--config", default=None, help="Optional Python config reference.")
    parser.add_argument("--device", default="auto", help="Device override, for example cpu or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config is None:
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        cfg = task_config_from_dict(checkpoint["config"])
    else:
        cfg = load_config(args.config)

    cfg.train.device = args.device
    CorrResponseRunner(cfg, mode="eval", checkpoint=args.checkpoint).run()
    print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
