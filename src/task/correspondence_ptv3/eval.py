# =============================================================================
# correspondence_ptv3 评估入口脚本
# =============================================================================
# 本文件是 `correspondence_ptv3` 任务的命令行评估入口。
# 通过 `python -m src.task.correspondence_ptv3.eval` 即可加载已训练好的
# 检查点并在验证集/测试集上运行推理与评估。
#
# 与 `train.py` 最大的差异在于：
#   - 必须显式指定一个已保存的检查点（目录或 checkpoint.pt 文件）；
#   - 如果没有显式提供 --config，会从检查点中反序列化保存时的配置；
#   - 启动时把 Runner 的 mode 切换为 "eval"，从而走评估/推理分支。
# =============================================================================

# 启用 Python 3.7+ 的延迟注解求值。
from __future__ import annotations

# 命令行参数解析。
import argparse
# 用于把项目根目录加入 sys.path，支持以脚本方式运行。
import sys
# 面向对象的路径处理工具。
from pathlib import Path

# 当以脚本方式直接运行本文件时，__package__ 会为 None 或空串，
# 此时手动将工程根目录加入 sys.path，保证 `from src.xxx import ...`
# 这种相对导入写法依然可用。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# 加载检查点/配置的辅助函数。
from src.base import load_checkpoint, load_config, task_config_from_dict
# 评估时所使用的运行器入口（与训练共用同一个类）。
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner


def parse_args() -> argparse.Namespace:
    """构造并解析评估相关的命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    # 创建参数解析器，并设置脚本说明。
    parser = argparse.ArgumentParser(description="Evaluate a PTv3 correspondence checkpoint.")
    # 待评估的检查点路径：可以是保存的目录（自动找其中的 checkpoint.pt），
    # 也可以直接指向一个 .pt 文件。
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or checkpoint.pt path.")
    # 可选的 Python 配置引用。如果不指定，则会从检查点中读取保存的配置。
    parser.add_argument("--config", default=None, help="Optional Python config reference.")
    # 设备覆盖项，例如 cpu / cuda:0；缺省沿用检查点中的设备设置。
    parser.add_argument("--device", default="auto", help="Device override, for example cpu or cuda:0.")
    # 解析并返回 Namespace。
    return parser.parse_args()


def main() -> None:
    """脚本主入口：解析参数 → 加载配置 → 加载检查点 → 启动评估。"""
    # 第一步：解析命令行参数。
    args = parse_args()

    # 第二步：决定使用哪一份配置。
    # - 如果用户显式提供了 --config，就从外部加载（通常用于在新环境复现实验）；
    # - 否则从检查点中读取当时保存的配置（确保评估时的模型结构与训练时一致）。
    if args.config is None:
        # 读取检查点（仅加载到 CPU，避免在没有 GPU 的机器上直接分配显存）。
        checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
        # 把 checkpoint["config"] 字典还原为 TaskConfig 对象。
        cfg = task_config_from_dict(checkpoint["config"])
    else:
        # 通过 Python 引用加载外部配置。
        cfg = load_config(args.config)

    # 第三步：用命令行传入的设备覆盖配置中的设备设置。
    cfg.train.device = args.device

    # 第四步：创建 Runner 并以 "eval" 模式启动评估流程。
    # Runner 在 eval 模式下会加载模型权重、跑前向推理、计算验证指标等。
    CorrespondencePTV3Runner(cfg, mode="eval", checkpoint=args.checkpoint).run()

    # 第五步：在终端输出所评估的检查点路径，便于人工确认。
    print(f"checkpoint: {args.checkpoint}")


# 当脚本被直接执行（而非被 import）时，进入主流程。
if __name__ == "__main__":
    main()
