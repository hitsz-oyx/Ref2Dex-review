"""Viewer server 兼容入口。"""

from src.task.Cm.viewer.server import *  # noqa: F401,F403
from src.task.Cm.viewer.server import main


if __name__ == "__main__":
    main()
