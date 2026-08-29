"""Cm 数据集规范入口。

Stage4 实现位于 ``stage4.py``；本包提供 Cm 数据集的公共导入面。
"""

from .stage4 import (
    Stage4CmDataset,
    _ensure_sequence_disjoint_splits,
    _normal_world_to_hand,
    _sequence_group_key,
    _world_to_hand,
    make_dataloaders,
    scalar_string,
)

__all__ = ["Stage4CmDataset", "make_dataloaders"]
