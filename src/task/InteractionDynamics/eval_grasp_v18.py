"""V18 stable-grasp、persistence 与 best-of-K 评估。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.task.InteractionDynamics.dataset_grasp_v18 import CachedGraspDataset
from src.task.InteractionDynamics.grasp_interaction_diffusion import (
    GraspInteractionDiffusion, sample_grasp_v)
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future


def trajectory_statistics(future: torch.Tensor) -> dict[str, torch.Tensor]:
    values = future.reshape(*future.shape[:-1], 8, 7)
    terminal = values[..., -3:, :]
    contact = (terminal[..., 6] < 2.).sum(-2).float()
    u_rms = terminal[..., :3].square().mean((-1, -2, -3)).sqrt()
    success = (contact >= 4).all(-1) & (u_rms < .3)
    return {"success": success, "contact": contact[..., -1],
            "d": terminal[..., -1, 6].mean(-1), "u_rms": u_rms}


@torch.inference_mode()
def evaluate(model, loader, stats, device, samples=1, sampling_steps=10):
    totals = {key: 0. for key in ("n", "gt_success", "persistence_success", "sample_success",
                                  "any_success", "formation_n", "formation_gt_success",
                                  "formation_persistence_success", "formation_sample_success",
                                  "formation_any_success",
                                  "single_squared", "best_squared", "elements",
                                  "pairwise_squared", "pairs", "contact", "d", "u")}
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        state, target = batch["state"], batch["future"]
        normalized_state = (state - stats["state_mean"]) / stats["state_std"]
        persistence = persistence_future(state, 8)
        predictions = []
        for _ in range(samples):
            residual = sample_grasp_v(model, normalized_state, batch["anchors_cm"],
                                      batch["object_patches"], sampling_steps=sampling_steps)
            predictions.append(persistence + residual * stats["residual_std"]
                               + stats["residual_mean"])
        predictions = torch.stack(predictions)
        errors = (predictions - target[None]).square().mean((2, 3))
        generated = trajectory_statistics(predictions)
        gt = trajectory_statistics(target); baseline = trajectory_statistics(persistence)
        initial_no_contact = (state[..., 3] < 2.).sum(-1) < 4
        totals["n"] += len(state); totals["elements"] += errors.numel()
        totals["gt_success"] += gt["success"].sum().item()
        totals["persistence_success"] += baseline["success"].sum().item()
        totals["sample_success"] += generated["success"].sum().item()
        totals["any_success"] += generated["success"].any(0).sum().item()
        totals["formation_n"] += initial_no_contact.sum().item()
        totals["formation_gt_success"] += gt["success"][initial_no_contact].sum().item()
        totals["formation_persistence_success"] += baseline["success"][initial_no_contact].sum().item()
        totals["formation_sample_success"] += generated["success"][:, initial_no_contact].sum().item()
        totals["formation_any_success"] += generated["success"][:, initial_no_contact].any(0).sum().item()
        totals["single_squared"] += errors[0].sum().item()
        totals["best_squared"] += errors.min(0).values.sum().item()
        totals["contact"] += generated["contact"].sum().item()
        totals["d"] += generated["d"].sum().item(); totals["u"] += generated["u_rms"].sum().item()
        if samples > 1:
            pairs = []
            for i in range(samples):
                for j in range(i): pairs.append((predictions[i] - predictions[j]).square().mean((1, 2)))
            pairwise = torch.stack(pairs)
            totals["pairwise_squared"] += pairwise.sum().item(); totals["pairs"] += pairwise.numel()
    n, draws = totals["n"], totals["n"] * samples
    formation = totals["formation_n"]
    return {"samples": int(n), "draws_per_state": samples,
            "gt_stable_success_rate": totals["gt_success"] / n,
            "persistence_stable_success_rate": totals["persistence_success"] / n,
            "diffusion_stable_success_rate": totals["sample_success"] / draws,
            "diffusion_any_of_k_stable_rate": totals["any_success"] / n,
            "initial_no_contact_samples": int(formation),
            "formation_gt_stable_rate": totals["formation_gt_success"] / max(formation, 1),
            "formation_persistence_stable_rate": totals["formation_persistence_success"] / max(formation, 1),
            "formation_diffusion_stable_rate": totals["formation_sample_success"] / max(formation * samples, 1),
            "formation_diffusion_any_of_k_rate": totals["formation_any_success"] / max(formation, 1),
            "single_sample_rmse_cm": (totals["single_squared"] / n) ** .5,
            "best_of_k_rmse_cm": (totals["best_squared"] / n) ** .5,
            "pairwise_trajectory_rms_cm": ((totals["pairwise_squared"] / totals["pairs"]) ** .5
                                             if totals["pairs"] else 0.),
            "terminal_contact_count": totals["contact"] / draws,
            "terminal_mean_d_cm": totals["d"] / draws,
            "terminal_u_rms_cm": totals["u"] / draws}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", default="output/research/InteractionDynamics/v18/cache")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]; model_cfg = config["model"]
    model = GraspInteractionDiffusion(model_cfg["horizon"], model_cfg["dim"],
                                     model_cfg["heads"], model_cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    loader = DataLoader(CachedGraspDataset(args.cache, args.split), batch_size=args.batch_size,
                        shuffle=False, num_workers=2)
    result = evaluate(model, loader, stats, device, args.samples,
                      config["diffusion"]["sampling_steps"])
    result.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
