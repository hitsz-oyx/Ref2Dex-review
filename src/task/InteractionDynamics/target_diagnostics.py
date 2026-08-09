"""量化 hand/object patch motion target 中真正的 patch-local 信号。"""
from __future__ import annotations

import argparse
import json

import torch

from src.base import build_runner_from_checkpoint
from src.task.InteractionDynamics.runner import InteractionDynamicsRunner, patch_motion_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def _accumulate_target(stats: dict[str, float], prefix: str, target: torch.Tensor) -> None:
    global_motion = target.mean(2, keepdim=True)
    local = target - global_motion
    stats[f"{prefix}_total_sq"] += float(target.square().sum())
    stats[f"{prefix}_global_sq"] += float(global_motion.square().expand_as(target).sum())
    stats[f"{prefix}_local_sq"] += float(local.square().sum())
    stats[f"{prefix}_count"] += target.numel()


def diagnose(runner: InteractionDynamicsRunner, split: str, max_batches: int) -> dict[str, float]:
    loader = runner.train_loader if split == "train" else (
        runner.val_loaders["val/"] if split == "val" else runner.test_loaders["test/"])
    stats = {f"{prefix}_{key}": 0.0 for prefix in ("hand", "object")
             for key in ("total_sq", "global_sq", "local_sq", "count")}
    runner.eval_mode()
    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = runner.prepare_batch(raw_batch)
        with runner.eval_context():
            prediction = runner.model(batch)
        scale = float(runner.cfg.meta.motion_scale)
        hand = patch_motion_target(batch["hand_disp_chunk"].float(), prediction["hand_knn_idx"], scale)
        obj = patch_motion_target(batch["obj_disp_chunk_gt"].float(), prediction["obj_knn_idx"], scale)
        _accumulate_target(stats, "hand", hand)
        _accumulate_target(stats, "object", obj)
    result: dict[str, float] = {}
    for prefix in ("hand", "object"):
        total = stats[f"{prefix}_total_sq"]
        count = stats[f"{prefix}_count"]
        result[f"{prefix}/rms_cm"] = (total / count) ** .5
        result[f"{prefix}/local_rms_cm"] = (stats[f"{prefix}_local_sq"] / count) ** .5
        result[f"{prefix}/local_energy_ratio"] = stats[f"{prefix}_local_sq"] / max(total, 1e-12)
        result[f"{prefix}/global_energy_ratio"] = stats[f"{prefix}_global_sq"] / max(total, 1e-12)
    return result


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=args.config, mode="eval", device=args.device)
    if not isinstance(runner, InteractionDynamicsRunner):
        raise ValueError("target_diagnostics.py 只支持 InteractionDynamicsRunner checkpoint")
    print(json.dumps(diagnose(runner, args.split, args.max_batches), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
