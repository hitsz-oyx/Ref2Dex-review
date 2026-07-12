from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import cleanup_distributed
from src.base.cli import build_train_parser, load_train_config_from_args

from .runner import CorrespondencePTV3Runner


DEFAULT_CONFIG = "src.task.correspondence_ptv3.config:Config"


def main() -> None:
    parser = build_train_parser(
        description="Train the correspondence task.",
        default_config=DEFAULT_CONFIG,
    )
    args = parser.parse_args()
    cfg = load_train_config_from_args(args)

    if not str(cfg.data.train_path).strip():
        raise ValueError(
            "data.train_path is empty. Pass --data or set data.train_path to a Stage 3 dataset root."
        )

    try:
        CorrespondencePTV3Runner(cfg, mode="train").run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
