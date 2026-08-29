"""Cm 的规范运行入口。

当前阶段保留旧顶层模块作为兼容实现；本包提供后续迁移使用的稳定入口。
"""

from src.task.Cm.model import CmFlowModel
from src.task.Cm.runner import CmActionRunner

__all__ = ["CmActionRunner", "CmFlowModel"]
