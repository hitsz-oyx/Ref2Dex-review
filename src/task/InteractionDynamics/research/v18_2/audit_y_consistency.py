"""全量审计 V18 GT/生成 Y 的解析与时序一致性。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.task.InteractionDynamics.dataset_grasp_v18 import CachedGraspDataset
from src.task.InteractionDynamics.eval_grasp_v18 import subset_masks
from src.task.InteractionDynamics.grasp_interaction_diffusion import (
    GraspInteractionDiffusion, sample_grasp_v)
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future


def unpack(state: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = future.reshape(*future.shape[:-1], 8, 7)
    u, r, d = values[..., :3], values[..., 3:6], values[..., 6]
    r0 = state[..., :3]
    delta_r = r - torch.cat([r0[..., None, :], r[..., :-1, :]], -2)
    return r, d, (u - delta_r).norm(dim=-1)


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.float().flatten()
    return {"mean": float(values.mean()), "median": float(values.median()),
            "p90": float(torch.quantile(values, .90)),
            "p95": float(torch.quantile(values, .95)), "max": float(values.max())}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
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
    loader = DataLoader(CachedGraspDataset(args.cache, args.split), args.batch_size,
                        shuffle=False, num_workers=2)
    torch.manual_seed(args.seed)
    values = {source: {stage: {"rd": [], "ur": []} for stage in
                       ("overall", "formation", "transition", "maintenance")}
              for source in ("gt", "generated")}
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        state = batch["state"]
        residual = sample_grasp_v(
            model, (state - stats["state_mean"]) / stats["state_std"],
            batch["anchors_cm"], batch["object_patches"],
            sampling_steps=cfg["diffusion"]["sampling_steps"])
        generated = persistence_future(state, 8) + residual * stats["residual_std"] + stats["residual_mean"]
        for source, future in (("gt", batch["future"]), ("generated", generated)):
            r, d, ur = unpack(state, future)
            rd = (r.norm(dim=-1) - d).clamp_min(0)
            for stage, mask in subset_masks(batch).items():
                values[source][stage]["rd"].append(rd[mask].cpu())
                values[source][stage]["ur"].append(ur[mask].cpu())
    result = {"split": args.split, "samples": len(loader.dataset), "checkpoint_epoch": checkpoint["epoch"]}
    result["groups"] = {}
    for source, stages in values.items():
        result["groups"][source] = {}
        for stage, metrics in stages.items():
            rd, ur = torch.cat(metrics["rd"]), torch.cat(metrics["ur"])
            result["groups"][source][stage] = {
                "rd": {"violation_rate": float((rd > 1e-6).float().mean()),
                       "mean_violation_cm": float(rd.mean()),
                       "p95_violation_cm": float(torch.quantile(rd.float(), .95)),
                       "max_violation_cm": float(rd.max())},
                "ur_cm": distribution(ur)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__": main()
