from __future__ import annotations

from src.task.correspondence_ptv3.models import (
    CorrespondencePTV3Model,
    PTv3ConcatModel,
    PTv3DenseBackbone,
    build_correspondence_model,
)


class StaticHOCPTv3(PTv3ConcatModel):
    def __init__(self, cfg, **kwargs):
        self.backbone_cls = PTv3DenseBackbone
        super().__init__(cfg, **kwargs)


__all__ = [
    "CorrespondencePTV3Model",
    "PTv3ConcatModel",
    "PTv3DenseBackbone",
    "StaticHOCPTv3",
    "build_correspondence_model",
]
