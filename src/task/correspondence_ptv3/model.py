"""Legacy import compatibility for correspondence model APIs.

New model implementations live in:
    src.task.correspondence_ptv3.models

Do not add new architectures here.
"""

from __future__ import annotations

from src.task.correspondence_ptv3.composition import resolve_correspondence_components
from src.task.correspondence_ptv3.models import (
    CorrespondencePTV3Model,
    PTv3ConcatModel,
    PTv3DenseBackbone,
    build_correspondence_model,
)
from src.task.correspondence_ptv3.supervision import build_contact_supervision


class StaticHOCPTv3(PTv3ConcatModel):
    def __init__(self, cfg, **kwargs):
        self.backbone_cls = PTv3DenseBackbone
        components = resolve_correspondence_components(
            cfg,
            explicit_override_keys=getattr(cfg, "_explicit_override_keys", None),
        )
        super().__init__(
            cfg,
            contact_supervision=build_contact_supervision(
                components.contact_supervision,
            ),
            **kwargs,
        )


__all__ = [
    "CorrespondencePTV3Model",
    "PTv3ConcatModel",
    "PTv3DenseBackbone",
    "StaticHOCPTv3",
    "build_correspondence_model",
]
