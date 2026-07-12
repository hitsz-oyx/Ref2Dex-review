from __future__ import annotations

from src.task.correspondence_ptv3.composition import (
    ContactSupervisionConfig,
    CorrespondenceModelConfig,
    resolve_correspondence_model_config,
)
from src.task.correspondence_ptv3.supervision import build_contact_supervision

from .common import PTv3DenseBackbone
from .ptv3_concat import CorrespondencePTV3Model, PTv3ConcatModel


def build_correspondence_model(
    model_config: CorrespondenceModelConfig,
    *,
    meta_cfg,
    contact_supervision_config: ContactSupervisionConfig,
) -> object:
    config = resolve_correspondence_model_config(model_config)
    if config.name == "ptv3_concat":
        wrapper = type("ModelConfigWrapper", (), {})()
        wrapper.meta = meta_cfg
        wrapper.model = config
        return PTv3ConcatModel(
            wrapper,
            contact_supervision=build_contact_supervision(contact_supervision_config),
        )
    raise ValueError(f"Unsupported correspondence model: {config.name!r}.")


__all__ = [
    "CorrespondencePTV3Model",
    "PTv3ConcatModel",
    "PTv3DenseBackbone",
    "build_correspondence_model",
]
