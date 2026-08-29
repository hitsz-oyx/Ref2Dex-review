"""数据工厂兼容入口。

真实 dispatch 仍由旧 runner 维护；后续迁移时将把 schema 路由集中到这里。
"""

from src.task.Cm.dataset import make_dataloaders

__all__ = ["make_dataloaders"]
