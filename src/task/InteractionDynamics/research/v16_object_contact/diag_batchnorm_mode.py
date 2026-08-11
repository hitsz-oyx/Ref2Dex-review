"""比较 V16 checkpoint 在标准 eval 与 world BatchNorm 使用批统计时的指标。"""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from src.base import build_runner_from_checkpoint, load_config
from src.base.checkpoint import unwrap_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.data.intervention_on_train = True
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=cfg, mode="eval", device=args.device)
    model = unwrap_model(runner.model)

    normal = runner.evaluate_loader(runner.train_loader, prefix="train/")
    runner.eval_mode()
    batchnorm_count = 0
    for module in model.world.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
            batchnorm_count += 1
    totals = {}
    count = 0
    for batch in runner.train_loader:
        batch = runner.prepare_batch(batch)
        with torch.no_grad(), runner.eval_context():
            output = runner.step(runner.model, batch, mode="eval")
        batch_size = output.batch_size or runner.batch_size(batch)
        count += batch_size
        for key, value in output.metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
    batch_stats = {f"train/{key}": value / count for key, value in totals.items()}
    keys = ("train/object_contact/mse", "train/object_contact/relative_improvement",
            "train/object_contact/f1", "train/object_contact/prediction_active_fraction")
    print(json.dumps({
        "world_batchnorm_count": batchnorm_count,
        "eval_running_stats": {key: normal[key] for key in keys},
        "batch_stats": {key: batch_stats[key] for key in keys},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
