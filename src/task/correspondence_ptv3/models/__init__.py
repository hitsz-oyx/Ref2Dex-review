from __future__ import annotations

from typing import Any

from src.task.correspondence_ptv3.config import resolve_correspondence_model_config

from .common import PTv3DenseBackbone
from .ptv3_concat import CorrespondencePTV3Model, PTv3ConcatModel


def build_correspondence_model(
    model_cfg: Any,
    meta_cfg: Any,
) -> object:
    config = resolve_correspondence_model_config(model_cfg)
    if config.name == "ptv3_concat":
        wrapper = type("ModelConfigWrapper", (), {})()
        wrapper.meta = meta_cfg
        wrapper.model = model_cfg
        return PTv3ConcatModel(wrapper)
    raise ValueError(f"Unsupported correspondence model: {config.name!r}.")


__all__ = [
    "CorrespondencePTV3Model",
    "PTv3ConcatModel",
    "PTv3DenseBackbone",
    "build_correspondence_model",
]
