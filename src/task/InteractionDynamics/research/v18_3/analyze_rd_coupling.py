"""在中等规模分层子集上分析 V18 的 r/d 几何耦合与锥投影修复。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.task.InteractionDynamics.dataset_grasp_v18 import CachedGraspDataset
from src.task.InteractionDynamics.eval_grasp_v18 import subset_masks
from src.task.InteractionDynamics.grasp_interaction_diffusion import (
    GraspInteractionDiffusion, sample_grasp_v)
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future


def unpack(future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = future.reshape(*future.shape[:-1], 8, 7)
    return values[..., 3:6], values[..., 6]


def quantiles(x: torch.Tensor) -> dict[str, float]:
    x = x.float().flatten()
    return {"mean": float(x.mean()), "median": float(x.median()),
            "p05": float(torch.quantile(x, .05)),
            "p95": float(torch.quantile(x, .95))}


def cone_project(r: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (r,d) 欧氏投影到二阶锥 ||r||<=d。"""
    norm = r.norm(dim=-1)
    inside = norm <= d
    opposite = norm <= -d
    projected_d = (norm + d).clamp_min(0) / 2
    scale = projected_d / norm.clamp_min(1e-12)
    projected_r = r * scale[..., None]
    projected_r = torch.where(inside[..., None], r, projected_r)
    projected_d = torch.where(inside, d, projected_d)
    projected_r = torch.where(opposite[..., None], torch.zeros_like(r), projected_r)
    projected_d = torch.where(opposite, torch.zeros_like(d), projected_d)
    return projected_r, projected_d


def contact_scores(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred, target = pred < 2, target < 2
    tp = (pred & target).sum().float(); fp = (pred & ~target).sum().float()
    fn = (~pred & target).sum().float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall).clamp_min(1e-12))}


def source_stats(r: torch.Tensor, d: torch.Tensor) -> dict:
    norm = r.norm(dim=-1); margin = d - norm
    violated = margin < -1e-6
    return {
        "margin_cm": quantiles(margin),
        "norm_r_cm": quantiles(norm),
        "d_cm": quantiles(d),
        "violation_rate": float(violated.float().mean()),
        "violation_depth_cm": quantiles((-margin[violated]).clamp_min(0)) if violated.any() else None,
        "near_boundary_rate_0.1cm": float((margin.abs() < .1).float().mean()),
        "pearson_norm_r_d": float(torch.corrcoef(torch.stack([norm.flatten(), d.flatten()]))[0, 1]),
        "violation_given_contact": float((margin[d < 2] < -1e-6).float().mean()),
        "violation_given_far": float((margin[d >= 2] < -1e-6).float().mean()),
    }


def repair_stats(r: torch.Tensor, d: torch.Tensor, gt_r: torch.Tensor,
                 gt_d: torch.Tensor) -> dict:
    originally_violated = r.norm(dim=-1) - d > 1e-6
    candidates = {
        "none": (r, d),
        "raise_d": (r, torch.maximum(d, r.norm(dim=-1))),
        "shrink_r": (r * (d.clamp_min(0) / r.norm(dim=-1).clamp_min(1e-12)).clamp_max(1)[..., None],
                     d.clamp_min(0)),
        "joint_cone": cone_project(r, d),
    }
    result = {}
    for name, (rr, dd) in candidates.items():
        anchor_error = torch.cat([rr - gt_r, (dd - gt_d)[..., None]], -1).square().mean(-1).sqrt()
        result[name] = {
            "r_rmse_cm": float((rr - gt_r).square().mean().sqrt()),
            "d_rmse_cm": float((dd - gt_d).square().mean().sqrt()),
            "rd_joint_rmse_cm": float(torch.cat([rr - gt_r, (dd - gt_d)[..., None]], -1).square().mean().sqrt()),
            "edit_r_rms_cm": float((rr - r).square().mean().sqrt()),
            "edit_d_rms_cm": float((dd - d).square().mean().sqrt()),
            "contact": contact_scores(dd, gt_d),
            "violation_rate": float(((rr.norm(dim=-1) - dd) > 1e-6).float().mean()),
            "rd_anchor_rmse_originally_valid_cm": float(anchor_error[~originally_violated].mean()),
            "rd_anchor_rmse_originally_violated_cm": float(anchor_error[originally_violated].mean()),
        }
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["config"]; model_cfg = cfg["model"]
    model = GraspInteractionDiffusion(model_cfg["horizon"], model_cfg["dim"],
                                      model_cfg["heads"], model_cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    dataset = CachedGraspDataset(args.cache, args.split)
    count = min(args.samples, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, count).round().long().unique().tolist()
    loader = DataLoader(Subset(dataset, indices), args.batch_size, num_workers=2)
    torch.manual_seed(args.seed)
    collected = {stage: {key: [] for key in ("gt_r", "gt_d", "gen_r", "gen_d")}
                 for stage in ("overall", "formation", "transition", "maintenance")}
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        residual = sample_grasp_v(
            model, (batch["state"] - stats["state_mean"]) / stats["state_std"],
            batch["anchors_cm"], batch["object_patches"],
            sampling_steps=cfg["diffusion"]["sampling_steps"])
        generated = persistence_future(batch["state"], 8) + residual * stats["residual_std"] + stats["residual_mean"]
        gt_r, gt_d = unpack(batch["future"]); gen_r, gen_d = unpack(generated)
        for stage, mask in subset_masks(batch).items():
            for key, value in (("gt_r", gt_r), ("gt_d", gt_d),
                               ("gen_r", gen_r), ("gen_d", gen_d)):
                collected[stage][key].append(value[mask].cpu())
    result = {"split": args.split, "samples": len(indices),
              "selection": "full_split_evenly_spaced", "checkpoint_epoch": checkpoint["epoch"],
              "implementation_note": "代码权重为 softmax(-||h-o||^2/tau^2)，无公式中的 1/2。"}
    result["groups"] = {}
    for stage, values in collected.items():
        values = {key: torch.cat(value) for key, value in values.items()}
        result["groups"][stage] = {
            "count": int(values["gt_d"].shape[0]),
            "gt": source_stats(values["gt_r"], values["gt_d"]),
            "generated": source_stats(values["gen_r"], values["gen_d"]),
            "repairs": repair_stats(values["gen_r"], values["gen_d"],
                                    values["gt_r"], values["gt_d"]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
