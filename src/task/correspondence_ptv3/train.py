# =============================================================================
# correspondence_ptv3 训练入口脚本
# =============================================================================
# 本文件是 `correspondence_ptv3` 任务的命令行训练入口。
# 通过 `python -m src.task.correspondence_ptv3.train` 即可启动训练。
#
# 主要职责：
#   1. 解析命令行参数（配置文件、覆盖项、数据路径、输出目录、设备等）；
#   2. 加载并合并配置（支持通过 `--set key=value` 临时覆盖任何配置项）；
#   3. 创建 CorrespondencePTV3Runner 并以 "train" 模式启动训练流程。
#
# 与 `correspondence_v1.train` 接口完全一致，便于切换不同主干网络时
# 复用训练脚本的批处理/调度逻辑。
# =============================================================================

# 启用 Python 3.7+ 的延迟注解求值。
from __future__ import annotations

# 命令行参数解析。
import argparse
# 用于把项目根目录加入 sys.path，支持以模块方式运行。
import sys
# 面向对象的路径处理工具。
from pathlib import Path

# 当本文件以 "脚本方式"（而非包方式）执行时，__package__ 会为 None 或空串。
# 此时需要手动把工程根目录添加到 sys.path，确保 `from src.xxx import ...`
# 这种相对导入写法依然可以工作。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 加载配置对象的工具函数。
from src.base import load_config
# 训练运行器入口。
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner


# 默认的 Python 配置引用，格式为 "模块路径:对象名"。
# 当用户不显式指定 --config 时，会使用本任务的默认配置。
DEFAULT_CONFIG = "src.task.correspondence_ptv3.config:Config"


def parse_args() -> argparse.Namespace:
    """构造并解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    # 创建一个参数解析器，并设置脚本说明。
    parser = argparse.ArgumentParser(description="Train the PTv3 correspondence model.")
    # 指定要使用的配置。可以是 Python 引用（如默认 DEFAULT_CONFIG），
    # 也可以是 YAML/JSON 等其他格式（取决于 load_config 的实现）。
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Python config reference.")
    # 通过 `--set key=value` 的形式覆盖任意嵌套配置项，可重复传入。
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config with dotted key=value syntax.",
    )
    # 快速覆盖训练数据路径，等价于 `--set data.train_path=...`。
    parser.add_argument("--data", default=None, help="Optional override for data.train_path.")
    # 快速覆盖输出目录，等价于 `--set train.output_dir=...`。
    parser.add_argument("--output-dir", default=None, help="Optional override for train.output_dir.")
    # 快速覆盖训练设备，例如 cpu / cuda:0。
    parser.add_argument("--device", default=None, help="Optional override for train.device.")
    # 执行解析并返回 Namespace 对象。
    return parser.parse_args()


def main() -> None:
    """脚本主入口：解析参数 → 加载配置 → 启动训练。"""
    # 第一步：解析命令行参数。
    args = parse_args()

    # 第二步：加载基础配置，并应用所有 `--set key=value` 形式的覆盖项。
    # load_config 会自动 import DEFAULT_CONFIG 指向的 Config 类并实例化。
    cfg = load_config(args.config, overrides=args.set)

    # 第三步：应用快速覆盖参数（与 --set 相比写起来更短）。
    if args.data is not None:
        cfg.data.train_path = args.data
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.device is not None:
        cfg.train.device = args.device

    # 第四步：安全检查——train_path 必须非空，否则数据集无法加载。
    if not str(cfg.data.train_path).strip():
        raise ValueError(
            "data.train_path is empty. Pass --data or set data.train_path to a Stage 3 dataset root."
        )

    # 第五步：创建 Runner 并以 "train" 模式启动完整训练流程。
    # Runner 内部会负责构建数据加载器、初始化模型、构建优化器、
    # 训练循环、评估、保存检查点、记录 wandb 等所有工作。
    CorrespondencePTV3Runner(cfg, mode="train").run()


# 当脚本被直接执行（而非被 import）时，进入主流程。
if __name__ == "__main__":
    main()
