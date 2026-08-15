from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset import build_dataset
from src.task.PointWorldWAM.pointworld_forward import GRABPointWorldForward
from src.task.PointWorldWAM.train import evaluate_conditions, metrics, move_batch


ROOT = Path(__file__).resolve().parents[5]
EXPERIMENTS = {
    "v0_baseline": (
        "src/task/PointWorldWAM/configs/grab_forward_long.yaml",
        "output/exp/pointworld_wam_forward_long/pretrained_two_seq",
    ),
    "v0_1_geometry": (
        "src/task/PointWorldWAM/configs/grab_forward_geometry_v1.yaml",
        "output/exp/pointworld_wam_forward_geometry_v1",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_rigidity(model, loader, device):
    predictions, targets, projections = [], [], []
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["object_points"]
        pred = model(
            target,
            batch["object_normals"],
            batch["hand_points"],
            batch["hand_normals"],
        )["object_tracks"]
        batch_size, timesteps, points, _ = pred.shape
        source = target[:, :1].expand(-1, timesteps, -1, -1).reshape(-1, points, 3)
        flat_pred = pred.reshape(-1, points, 3)
        source_centered = source - source.mean(1, keepdim=True)
        pred_centered = flat_pred - flat_pred.mean(1, keepdim=True)
        u, _, vh = torch.linalg.svd(source_centered.transpose(1, 2) @ pred_centered)
        correction = torch.ones((len(source), 3), device=device, dtype=pred.dtype)
        correction[:, -1] = torch.sign(torch.det(u @ vh))
        rotation = u @ torch.diag_embed(correction) @ vh
        projected = source_centered @ rotation + flat_pred.mean(1, keepdim=True)
        predictions.append(pred.cpu())
        targets.append(target.cpu())
        projections.append(projected.view(batch_size, timesteps, points, 3).cpu())
    pred = torch.cat(predictions)
    target = torch.cat(targets)
    projected = torch.cat(projections)
    return {
        "pred_nonrigid_rmse_m": float(torch.linalg.vector_norm(pred - projected, dim=-1).mean()),
        "rigid_projected": metrics(projected, target),
    }


def main() -> None:
    report = {}
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, (config_path, experiment_path) in EXPERIMENTS.items():
        cfg = load_config(str(ROOT / config_path))
        set_seed(cfg.seed)
        device = torch.device(cfg.train.device)
        dataset = build_dataset(cfg.data)
        loader = DataLoader(dataset, batch_size=max(2, cfg.train.batch_size), shuffle=False)
        directory = ROOT / experiment_path
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        model = GRABPointWorldForward(cfg.model)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        conditions = evaluate_conditions(model, loader, device)
        rigidity = evaluate_rigidity(model, loader, device)
        gt = conditions["gt"]["ade_m"]
        static = conditions["static"]["ade_m"]
        shuffled = conditions["shuffled"]["ade_m"]
        report[name] = {
            "best_step": int(checkpoint["step"]),
            "conditions": conditions,
            "rigidity": rigidity,
            "gt_ade_better_than_static_pct": 100.0 * (static - gt) / static,
            "gt_ade_better_than_shuffled_pct": 100.0 * (shuffled - gt) / shuffled,
        }
        del model, checkpoint
        torch.cuda.empty_cache()

    baseline = report["v0_baseline"]["conditions"]["gt"]
    geometry = report["v0_1_geometry"]["conditions"]["gt"]
    report["geometry_vs_baseline"] = {
        key: 100.0 * (geometry[key] - baseline[key]) / baseline[key]
        for key in ("ade_m", "fde_m", "translation_m", "rotation_deg")
    }
    output = ROOT / "output/research/pointworld_wam/geometry_v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    names = list(EXPERIMENTS)
    labels = ["V0 baseline", "V0.1 geometry"]
    x = np.arange(len(names))
    width = 0.25
    for index, condition in enumerate(("gt", "static", "shuffled")):
        axes[0].bar(
            x + (index - 1) * width,
            [1000.0 * report[name]["conditions"][condition]["ade_m"] for name in names],
            width,
            label=condition,
        )
    axes[0].axhline(5.0, color="tab:green", linestyle="--", label="ADE gate")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("ADE (mm)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()
    metrics = ("ade_m", "fde_m", "translation_m")
    for index, name in enumerate(names):
        axes[1].bar(
            np.arange(len(metrics)) + (index - 0.5) * 0.35,
            [1000.0 * report[name]["conditions"]["gt"][key] for key in metrics],
            0.35,
            label=labels[index],
        )
    axes[1].set_xticks(np.arange(len(metrics)), ("ADE", "FDE", "Translation"))
    axes[1].set_ylabel("error (mm)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=180)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
