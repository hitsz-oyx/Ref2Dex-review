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
import os
import sys
# 用于把项目根目录加入 sys.path，支持以脚本方式运行。
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.base import cleanup_distributed, load_checkpoint

from .config_loader import load_correspondence_config
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PTv3 correspondence checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or checkpoint.pt path.")
    parser.add_argument("--config", default=None, help="Optional Hydra config name or saved config path.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Optional Hydra-style override, for example edge_sampler=stratified.",
    )
    parser.add_argument("--device", default="auto", help="Device override, for example cpu or cuda:0.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed evaluation logic. Launch with torchrun for multi-GPU execution.",
    )
    parser.add_argument("--local-rank", "--local_rank", default=None, type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config_source = args.config if args.config is not None else checkpoint["config"]
    cfg = load_correspondence_config(
        config_source,
        overrides=list(args.set),
    )
    cfg.train.device = args.device
    if args.distributed:
        cfg.train.distributed.enable = True

    try:
        CorrespondencePTV3Runner(cfg, mode="eval", checkpoint=args.checkpoint).run()
    finally:
        cleanup_distributed()

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"checkpoint: {args.checkpoint}")
if __name__ == "__main__":
    main()
