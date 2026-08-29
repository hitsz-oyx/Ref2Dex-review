"""规范特征提取入口，暂时转发到旧入口。"""

from src.task.Cm.extract import *  # noqa: F401,F403
from src.task.Cm.extract import main


if __name__ == "__main__":
    main()
