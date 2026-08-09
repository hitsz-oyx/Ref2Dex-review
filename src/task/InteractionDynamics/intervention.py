"""评估 Effect Decoder 对 interaction token 局部结构的依赖。"""
from __future__ import annotations

import argparse
import json

import torch

from src.base import build_runner_from_checkpoint
from src.base.checkpoint import unwrap_model
from src.task.InteractionDynamics.runner import InteractionDynamicsRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def evaluate_interventions(runner: InteractionDynamicsRunner, split: str) -> dict[str, dict[str, float]]:
    prefix = f"{split}/"
    loaders = runner.val_loaders if split == "val" else runner.test_loaders
    loader = loaders[prefix]
    effect = unwrap_model(runner.model).effect
    results: dict[str, dict[str, float]] = {}

    for mode in ("normal", "mean", "shuffle", "cross_sample"):
        handle = None
        if mode != "normal":
            def intervene(_module, args, intervention=mode):
                values = list(args)
                tokens = values[3]
                if intervention == "mean":
                    tokens = tokens.mean(1, keepdim=True).expand_as(tokens)
                elif intervention == "shuffle":
                    tokens = tokens.flip(1)
                elif intervention == "cross_sample":
                    if tokens.shape[0] < 2:
                        raise RuntimeError("cross_sample intervention requires batch_size >= 2")
                    tokens = tokens.roll(1, 0)
                values[3] = tokens
                return tuple(values)
            handle = effect.register_forward_pre_hook(intervene)
        metrics = runner.evaluate_loader(loader, prefix=prefix)
        if handle is not None:
            handle.remove()
        results[mode] = {
            "ade_mm": metrics[f"{prefix}object/ade_mm"],
            "fde_mm": metrics[f"{prefix}object/fde_mm"],
        }
    return results


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=args.config, mode="eval", device=args.device,
    )
    if not isinstance(runner, InteractionDynamicsRunner):
        raise ValueError("intervention.py 只支持 InteractionDynamicsRunner checkpoint")
    print(json.dumps(evaluate_interventions(runner, args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
