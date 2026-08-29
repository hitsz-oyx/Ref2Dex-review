"""HRDexDB 数据集兼容入口。"""

from src.task.Cm.dataset_hrdexdb import *  # noqa: F401,F403
from src.task.Cm import dataset_hrdexdb as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
