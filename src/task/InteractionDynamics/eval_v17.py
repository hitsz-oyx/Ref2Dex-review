"""评估 V17 best checkpoint 的分层误差与 Goal 依赖。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.task.InteractionDynamics.dataset_v17 import CachedV17Dataset
from src.task.InteractionDynamics.residual_interaction_diffusion import (
    ResidualInteractionDiffusion, sample_residual_v)
from src.task.InteractionDynamics.residual_interaction_regression import (
    ResidualInteractionRegressor, persistence_future)
from src.task.InteractionDynamics.train_residual_interaction_regression import intervene_goal


SUBSETS = ("overall", "dynamic", "static", "active", "inactive",
           "dynamic_active", "far_30cm", "far_60cm", "far_100cm")


class Metrics:
    def __init__(self) -> None:
        self.values = {name: {key: 0. for key in ("u", "r", "d", "n", "tp", "pp", "gp")}
                       for name in SUBSETS}

    def update(self, prediction: torch.Tensor, target: torch.Tensor,
               masks: dict[str, torch.Tensor]) -> None:
        prediction = prediction.reshape(*prediction.shape[:-1], 4, 7)
        target = target.reshape(*target.shape[:-1], 4, 7)
        for name, mask in masks.items():
            p, t = prediction[mask], target[mask]
            if not len(p):
                continue
            row = self.values[name]
            row["u"] += (p[..., :3] - t[..., :3]).square().sum().item()
            row["r"] += (p[..., 3:6] - t[..., 3:6]).square().sum().item()
            row["d"] += (p[..., 6] - t[..., 6]).square().sum().item()
            row["n"] += p[..., 6].numel()
            predicted_contact = torch.exp(-p[..., 6].square() / 2) > .5
            target_contact = torch.exp(-t[..., 6].square() / 2) > .5
            row["tp"] += (predicted_contact & target_contact).sum().item()
            row["pp"] += predicted_contact.sum().item()
            row["gp"] += target_contact.sum().item()

    def result(self) -> dict:
        result = {}
        for name, row in self.values.items():
            if not row["n"]:
                continue
            precision = row["tp"] / max(row["pp"], 1)
            recall = row["tp"] / max(row["gp"], 1)
            result[name] = {
                "samples_x_anchors_x_horizon": int(row["n"]),
                "u_rmse_cm": (row["u"] / (3 * row["n"])) ** .5,
                "r_rmse_cm": (row["r"] / (3 * row["n"])) ** .5,
                "d_rmse_cm": (row["d"] / row["n"]) ** .5,
                "contact_f1": 2 * precision * recall / max(precision + recall, 1e-8),
            }
        return result


def masks_for(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    dynamic = batch["residual"].square().mean((1, 2)).sqrt() >= .2
    active = batch["goal"][:, 0] > .5
    distance = batch["distance_cm"]
    return {"overall": torch.ones_like(dynamic), "dynamic": dynamic, "static": ~dynamic,
            "active": active, "inactive": ~active, "dynamic_active": dynamic & active,
            "far_30cm": distance >= 30, "far_60cm": distance >= 60,
            "far_100cm": distance >= 100}


@torch.inference_mode()
def evaluate(checkpoint: dict, loader: DataLoader, device: torch.device,
             sampling_steps: int) -> dict:
    cfg = checkpoint["config"]; model_cfg = cfg["model"]
    deterministic = ResidualInteractionRegressor(
        4, 4, model_cfg["dim"], model_cfg["heads"], model_cfg["layers"]).to(device)
    diffusion = ResidualInteractionDiffusion(
        4, 4, model_cfg["dim"], model_cfg["heads"], model_cfg["layers"]).to(device)
    deterministic.load_state_dict(checkpoint["deterministic"])
    diffusion.load_state_dict(checkpoint["diffusion"])
    deterministic.eval(); diffusion.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    modes = ("correct", "motion_zero", "motion_shuffle", "motion_reverse", "active_zero")
    accumulators = {"persistence": Metrics()}
    accumulators.update({f"deterministic/{mode}": Metrics() for mode in modes})
    accumulators.update({f"diffusion/{mode}": Metrics() for mode in modes})
    generator = torch.Generator(device=device).manual_seed(42)
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        state, future = batch["state"], batch["future"]
        normalized_state = (state - stats["state_mean"]) / stats["state_std"]
        persistence = persistence_future(state, 4)
        masks = masks_for(batch)
        accumulators["persistence"].update(persistence, future, masks)
        noise = torch.randn(batch["residual"].shape, device=device, generator=generator)
        for mode in modes:
            goal = intervene_goal(batch["goal"], mode)
            normalized_goal = (goal - stats["goal_mean"]) / stats["goal_std"]
            residual = deterministic(normalized_state, batch["anchors_cm"],
                                     batch["object_patches"], normalized_goal)
            prediction = persistence + residual * stats["residual_std"] + stats["residual_mean"]
            accumulators[f"deterministic/{mode}"].update(prediction, future, masks)
            residual, _ = sample_residual_v(
                diffusion, normalized_state, batch["anchors_cm"], batch["object_patches"],
                normalized_goal, cfg["diffusion"]["steps"], initial_noise=noise,
                sampling_steps=sampling_steps)
            prediction = persistence + residual * stats["residual_std"] + stats["residual_mean"]
            accumulators[f"diffusion/{mode}"].update(prediction, future, masks)
    return {name: metrics.result() for name, metrics in accumulators.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path,
                        default=Path("output/research/InteractionDynamics/v17/cache"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sampling-steps", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    dataset = CachedV17Dataset(args.cache, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    result = {"checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
              "split": args.split, "samples": len(dataset),
              "sampling_steps": args.sampling_steps,
              "metrics": evaluate(checkpoint, loader, torch.device("cuda"), args.sampling_steps)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
