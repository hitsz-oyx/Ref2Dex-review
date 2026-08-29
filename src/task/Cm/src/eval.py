"""规范评估入口，暂时转发到旧入口。"""

from src.task.Cm.eval import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
