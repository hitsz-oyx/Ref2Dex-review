"""评估 Effect Decoder 对 interaction token 局部结构的依赖。"""
from __future__ import annotations

import argparse
import json

import torch

from src.base import build_runner_from_checkpoint, load_config
from src.base.checkpoint import unwrap_model
from src.task.InteractionDynamics.runner import InteractionDynamicsRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def evaluate_interventions(runner: InteractionDynamicsRunner, split: str) -> dict[str, dict[str, float]]:
    prefix = f"{split}/"
    if split == "train":
        loader = runner.train_loader
    else:
        loaders = runner.val_loaders if split == "val" else runner.test_loaders
        loader = loaders[prefix]
    model = unwrap_model(runner.model)
    effect = model.effect
    results: dict[str, dict[str, float]] = {}

    is_v14 = hasattr(model, "action_adapter") and hasattr(model, "action_intervention_mode")
    modes = (("normal", "root_zero", "articulation_zero", "articulation_mean",
              "articulation_shuffle", "cross_sample") if is_v14
             else ("normal", "mean", "shuffle", "cross_sample"))
    for mode in modes:
        handle = None
        uses_model_intervention = is_v14 or bool(getattr(model, "num_interaction_slots", 0))
        if uses_model_intervention:
            model.action_intervention_mode = mode
        elif mode != "normal":
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
                batch, patches, _ = tokens.shape
                values[5] = model.patch_effect(tokens).reshape(
                    batch, patches, model.effect.time_embed.shape[1], 3).transpose(1, 2)
                return tuple(values)
            handle = effect.register_forward_pre_hook(intervene)
        metrics = runner.evaluate_loader(loader, prefix=prefix)
        if handle is not None:
            handle.remove()
        if uses_model_intervention:
            model.action_intervention_mode = "normal"
        results[mode] = {
            "ade_mm": metrics[f"{prefix}object/ade_mm"],
            "fde_mm": metrics[f"{prefix}object/fde_mm"],
        }
    return results


def main() -> None:
    args = parse_args()
    config = args.config
    if args.split == "train":
        if config is None:
            raise ValueError("train intervention requires --config")
        config = load_config(config)
        config.data.intervention_on_train = True
    runner = build_runner_from_checkpoint(
        args.checkpoint, config=config, mode="eval", device=args.device,
    )
    if not isinstance(runner, InteractionDynamicsRunner):
        raise ValueError("intervention.py 只支持 InteractionDynamicsRunner checkpoint")
    print(json.dumps(evaluate_interventions(runner, args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
