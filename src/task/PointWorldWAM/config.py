from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: str) -> Namespace:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"配置必须是 mapping: {path}")

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return Namespace(**{key: convert(item) for key, item in value.items()})
        return value

    cfg = convert(raw)
    cfg.data.root = str(_resolve_path(cfg.data.root))
    if hasattr(cfg.data, "grab_raw_root"):
        cfg.data.grab_raw_root = str(_resolve_path(cfg.data.grab_raw_root))
    for field in (
        "pointworld_root",
        "pointworld_checkpoint",
        "norm_stats_path",
        "grab_raw_root",
        "mano_model_dir",
        "left_vtemplate",
        "right_vtemplate",
    ):
        if hasattr(cfg.model, field):
            setattr(cfg.model, field, str(_resolve_path(getattr(cfg.model, field))))
    cfg.train.output_dir = str(_resolve_path(cfg.train.output_dir))
    return cfg


def config_to_dict(value: Any) -> Any:
    if isinstance(value, Namespace):
        return {key: config_to_dict(item) for key, item in vars(value).items()}
    return value
