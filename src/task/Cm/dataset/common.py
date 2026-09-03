"""Stage4 数据集共享的小型工具函数。

实现位于 :mod:`src.task.Cm.dataset.stage4`；这里仅提供稳定的内部导入面，
避免在数据模块之间通过包级通配符形成隐式循环依赖。
"""

from .stage4 import (
    _normal_world_to_hand,
    _world_to_hand,
    scalar_string,
)

__all__ = ["_normal_world_to_hand", "_world_to_hand", "scalar_string"]
