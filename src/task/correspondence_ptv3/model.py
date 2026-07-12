"""Legacy import compatibility for correspondence model APIs.

New model implementations live in:
    src.task.correspondence_ptv3.models

Do not add new architectures here.
"""

from __future__ import annotations

from hydra.utils import instantiate

from src.task.correspondence_ptv3.models import (
    CorrespondencePTV3Model,
    PTv3ConcatModel,
    PTv3DenseBackbone,
)
from src.task.correspondence_ptv3.supervision import SoftContactSupervision


class StaticHOCPTv3(PTv3ConcatModel):
    def __init__(self, cfg, **kwargs):
        self.backbone_cls = PTv3DenseBackbone
        supervision_cfg = getattr(cfg, "contact_supervision", None)
        if isinstance(supervision_cfg, dict) and "_target_" in supervision_cfg:
            contact_supervision = instantiate(supervision_cfg)
        else:
            contact_supervision = SoftContactSupervision()
        super().__init__(
            cfg,
            contact_supervision=contact_supervision,
            **kwargs,
        )


__all__ = [
    "CorrespondencePTV3Model",
    "PTv3ConcatModel",
    "PTv3DenseBackbone",
    "StaticHOCPTv3",
]
