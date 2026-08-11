"""诊断 V15 endpoint 时间对齐，并用 endpoint-distance oracle 检查监督可拟合性。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.base import load_config
from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.runner import endpoint_contact_target
from src.task.InteractionDynamics.uni3d import gather_points, patchify


def build_dataset(config: str) -> tuple[object, InteractionDynamicsDataset]:
    cfg = load_config(config)
    data = cfg.data
    dataset = InteractionDynamicsDataset(
        data.train_path,
        dominant_hand_manifest=data.dominant_hand_manifest,
        hand_side=data.hand_side,
        num_effect_points=cfg.meta.num_effect_points,
        chunk_len=cfg.meta.chunk_len,
        temporal_stride=cfg.meta.temporal_stride,
        base_seed=cfg.train.seed,
        max_samples=data.max_train_samples,
        min_object_effect_norm=data.min_object_effect_norm,
    )
    return cfg, dataset


def contact_geometry(batch: dict[str, torch.Tensor], sigma_m: float
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, obj_knn = patchify(batch["world_obj_points_object"], 64, 32)
    hand = gather_points(batch["future_hand_points_object_endpoint"],
                         batch["action_patch_knn_idx"])
    current_obj = gather_points(batch["world_obj_points_object"], obj_knn).mean(2)
    endpoint_obj_points = batch["world_obj_points_object"] + batch["obj_disp_chunk_gt"][:, -1]
    endpoint_obj = gather_points(endpoint_obj_points, obj_knn).mean(2)
    old_distance = torch.linalg.vector_norm(
        hand[:, :, :, None] - current_obj[:, None, None], dim=-1).amin(2)
    endpoint_distance = torch.linalg.vector_norm(
        hand[:, :, :, None] - endpoint_obj[:, None, None], dim=-1).amin(2)
    prediction = {"obj_knn_idx": obj_knn}
    target = endpoint_contact_target(batch, prediction, sigma_m)
    torch.testing.assert_close(
        target, torch.exp(-endpoint_distance.square() / (2 * sigma_m ** 2)))
    return old_distance, endpoint_distance, target


def train_oracle(distance: torch.Tensor, target: torch.Tensor, steps: int,
                 device: torch.device) -> dict[str, float]:
    x = (distance.reshape(-1, 1) * 100).to(device)
    y = target.reshape(-1, 1).to(device)
    model = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(),
                          nn.Linear(64, 1), nn.Sigmoid()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    generator = torch.Generator(device=device).manual_seed(42)
    initial = None
    for step in range(steps):
        index = torch.randint(len(x), (min(8192, len(x)),), generator=generator, device=device)
        prediction = model(x[index])
        loss = torch.nn.functional.mse_loss(prediction, y[index])
        if initial is None:
            initial = float(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = model(x)
        mse = torch.nn.functional.mse_loss(prediction, y)
        zero = y.square().mean()
        active_gt = y > .5
        active_pred = prediction > .5
        tp = (active_gt & active_pred).sum().float()
        precision = tp / active_pred.sum().clamp_min(1)
        recall = tp / active_gt.sum().clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {"initial_mse": float(initial), "mse": float(mse), "zero_mse": float(zero),
            "relative_improvement": float(1 - mse / zero), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/grab_v15_contact_overfit.yaml")
    parser.add_argument("--output", default="output/research/InteractionDynamics/v15_1_contact_alignment")
    parser.add_argument("--oracle-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg, dataset = build_dataset(args.config)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    old_distance, distance, target = contact_geometry(batch, float(cfg.meta.contact_sigma_m))
    old_target = torch.exp(-old_distance.square() / (2 * float(cfg.meta.contact_sigma_m) ** 2))
    result = {
        "samples": len(dataset),
        "sigma_m": float(cfg.meta.contact_sigma_m),
        "old_active_fraction": float((old_target > .5).float().mean()),
        "aligned_active_fraction": float((target > .5).float().mean()),
        "old_zero_mse": float(old_target.square().mean()),
        "aligned_zero_mse": float(target.square().mean()),
        "mean_object_endpoint_shift_mm": float(torch.linalg.vector_norm(
            batch["obj_disp_chunk_gt"][:, -1], dim=-1).mean() * 1000),
    }
    result["oracle"] = train_oracle(distance, target, args.oracle_steps, torch.device(args.device))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    count = min(10, len(dataset))
    figure, axes = plt.subplots(count, 2, figsize=(8, 3 * count), squeeze=False)
    for index in range(count):
        axes[index, 0].imshow(old_target[index], vmin=0, vmax=1, cmap="magma")
        axes[index, 0].set_title(f"sample {index}: H(t+8) vs O(t)")
        axes[index, 1].imshow(target[index], vmin=0, vmax=1, cmap="magma")
        axes[index, 1].set_title(f"sample {index}: H(t+8) vs O(t+8)")
    figure.tight_layout()
    figure.savefig(output / "contact_alignment.png", dpi=140)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
