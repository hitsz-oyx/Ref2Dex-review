"""ObjectV2 数据集兼容入口。"""

from src.task.Cm.dataset_object_v2 import *  # noqa: F401,F403
from src.task.Cm import dataset_object_v2 as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
