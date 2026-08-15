from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset_one_step import build_one_step_dataset
from src.task.PointWorldWAM.pointworld_one_step import GRABPointWorldOneStep
from src.task.PointWorldWAM.train import move_batch
from src.task.PointWorldWAM.train_one_step import evaluate_conditions, flow_metrics


ROOT = Path(__file__).resolve().parents[5]


def rigid_projection(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_centered = source - source.mean(1, keepdim=True)
    target_centered = target - target.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(source_centered.transpose(1, 2) @ target_centered)
    correction = torch.ones((len(source), 3), device=source.device, dtype=source.dtype)
    correction[:, -1] = torch.sign(torch.det(u @ vh))
    rotation = u @ torch.diag_embed(correction) @ vh
    return source_centered @ rotation + target.mean(1, keepdim=True)


@torch.no_grad()
def diagnose(model, loader, device):
    predictions, targets, points, motions = [], [], [], []
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(
            batch["object_points"],
            batch["object_normals"],
            batch["hand_points"],
            batch["hand_normals"],
            batch["hand_flow"],
        )["object_flow"]
        predictions.append(pred.cpu())
        targets.append(batch["object_flow"].cpu())
        points.append(batch["object_points"].cpu())
        motions.append(batch["motion"].cpu())
    pred = torch.cat(predictions)
    target = torch.cat(targets)
    source = torch.cat(points)
    motion = torch.cat(motions)
    pred_tracks = source + pred
    target_tracks = source + target
    pred_rigid = rigid_projection(source, pred_tracks)
    target_rigid = rigid_projection(source, target_tracks)
    per_transition = torch.linalg.vector_norm(pred - target, dim=-1).mean(1)
    return {
        "pred_nonrigid_residual_m": float(
            torch.linalg.vector_norm(pred_tracks - pred_rigid, dim=-1).mean()
        ),
        "target_nonrigid_residual_m": float(
            torch.linalg.vector_norm(target_tracks - target_rigid, dim=-1).mean()
        ),
        "rigid_projected_metrics": flow_metrics(pred_rigid - source, target, source),
        "per_transition_error_m": {
            "min": float(per_transition.min()),
            "median": float(per_transition.median()),
            "p90": float(torch.quantile(per_transition, 0.9)),
            "max": float(per_transition.max()),
        },
        "motion_error_correlation": float(
            np.corrcoef(motion.numpy(), per_transition.numpy())[0, 1]
        ),
    }


def main() -> None:
    config_path = ROOT / "src/task/PointWorldWAM/configs/grab_one_step_v02a.yaml"
    experiment = ROOT / "output/exp/pointworld_wam_one_step_v02a"
    output = ROOT / "output/research/pointworld_wam/one_step_v02a"
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(config_path))
    device = torch.device(cfg.train.device)
    dataset = build_one_step_dataset(cfg.data)
    loader = DataLoader(dataset, batch_size=cfg.train.eval_batch_size, shuffle=False)
    checkpoint = torch.load(experiment / "best.pt", map_location="cpu", weights_only=False)
    model = GRABPointWorldOneStep(cfg.model)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    conditions = evaluate_conditions(model, loader, device)
    report = {
        "best_step": int(checkpoint["step"]),
        "conditions": conditions,
        "diagnostics": diagnose(model, loader, device),
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    history = json.loads((experiment / "history.json").read_text(encoding="utf-8"))
    evaluations = [
        (int(row["step"]), 1000.0 * float(row["eval_gt_point_error_m"]))
        for row in history
        if "eval_gt_point_error_m" in row
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(*zip(*evaluations), marker="o", markersize=3)
    axes[0].axhspan(1.0, 2.0, color="tab:green", alpha=0.15, label="V0.2A gate")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("GT-action point error (mm)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    labels = ("GT", "static action", "shuffled", "zero flow")
    keys = ("gt", "static_action", "shuffled_action", "zero_flow_baseline")
    axes[1].bar(
        labels,
        [1000.0 * conditions[key]["point_error_m"] for key in keys],
    )
    axes[1].set_ylabel("point error (mm)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "diagnostics.png", dpi=180)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
