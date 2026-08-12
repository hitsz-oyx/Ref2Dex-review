"""V18 stable-grasp、persistence 与 best-of-K 评估。"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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


def subset_masks(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current_contact = (batch["state"][..., 3] < 2.).sum(-1)
    before = batch["frame"] < batch["grasp_frame"]
    return {"overall": torch.ones_like(before),
            "formation": current_contact < 4,
            "transition": (current_contact >= 4) & before,
            "maintenance": ~before}


def _empty_subset() -> dict:
    return {"n": 0, "gt_success": 0, "persistence_success": 0,
            "sample_success": 0, "any_success": 0, "single_squared": 0.,
            "best_squared": 0., "gt_contact": [], "prediction_contact": []}


@torch.inference_mode()
def evaluate(model, loader, stats, device, samples=1, sampling_steps=10):
    totals = {key: 0. for key in ("n", "gt_success", "persistence_success", "sample_success",
                                  "any_success", "formation_n", "formation_gt_success",
                                  "formation_persistence_success", "formation_sample_success",
                                  "formation_any_success",
                                  "single_squared", "best_squared", "elements",
                                  "pairwise_squared", "pairs", "contact", "d", "u")}
    subsets = {name: _empty_subset() for name in
               ("overall", "formation", "transition", "maintenance")}
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
        masks = subset_masks(batch)
        for name, mask in masks.items():
            row = subsets[name]; count = int(mask.sum())
            row["n"] += count
            row["gt_success"] += int(gt["success"][mask].sum())
            row["persistence_success"] += int(baseline["success"][mask].sum())
            row["sample_success"] += int(generated["success"][:, mask].sum())
            row["any_success"] += int(generated["success"][:, mask].any(0).sum())
            row["single_squared"] += float(errors[0, mask].sum())
            row["best_squared"] += float(errors[:, mask].min(0).values.sum())
            row["gt_contact"].append(gt["contact"][mask].cpu())
            row["prediction_contact"].append(generated["contact"][:, mask].flatten().cpu())
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
    result = {"samples": int(n), "draws_per_state": samples,
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
    result["subsets"] = {}
    for name, row in subsets.items():
        count = row["n"]
        if not count: continue
        subset = {"samples": count,
                  "gt_stable_rate": row["gt_success"] / count,
                  "persistence_stable_rate": row["persistence_success"] / count,
                  "diffusion_stable_rate": row["sample_success"] / (count * samples),
                  "diffusion_any_of_k_rate": row["any_success"] / count,
                  "single_rmse_cm": (row["single_squared"] / count) ** .5,
                  "best_of_k_rmse_cm": (row["best_squared"] / count) ** .5}
        for source in ("gt", "prediction"):
            values = torch.cat(row[source + "_contact"]).float()
            subset[source + "_terminal_contact"] = {
                "mean": float(values.mean()), "median": float(values.median()),
                "p10": float(torch.quantile(values, .1)),
                "p90": float(torch.quantile(values, .9))}
        result["subsets"][name] = subset
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", default="output/research/InteractionDynamics/v18/cache")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--event-audit", type=Path,
                        help="audit_bilateral.py 生成的 events.json")
    parser.add_argument("--event-group", choices=("single_hand_clean", "bilateral"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]; model_cfg = config["model"]
    model = GraspInteractionDiffusion(model_cfg["horizon"], model_cfg["dim"],
                                     model_cfg["heads"], model_cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    dataset = CachedGraspDataset(args.cache, args.split)
    if args.event_group:
        if args.event_audit is None:
            parser.error("--event-group 需要同时提供 --event-audit")
        audit = {int(row["event_index"]): row["group"]
                 for row in json.loads(args.event_audit.read_text())}
        indices = [index for index in range(len(dataset))
                   if audit[dataset.event_indices[
                       bisect.bisect_right(dataset.offsets, index) - 1]] == args.event_group]
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=2)
    result = evaluate(model, loader, stats, device, args.samples,
                      config["diffusion"]["sampling_steps"])
    result.update({"split": args.split, "checkpoint_epoch": checkpoint["epoch"],
                   "event_group": args.event_group or "all"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
