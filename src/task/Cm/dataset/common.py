"""通用 Stage4 数据工具的兼容入口。"""

from src.task.Cm.dataset import *  # noqa: F401,F403
from src.task.Cm import dataset as _legacy


def __getattr__(name):
    return getattr(_legacy, name)
