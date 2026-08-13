"""V20.4 Phase 1：在冻结预测缓存上训练 velocity fusion gate。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.task.InteractionDynamics.fusion_gate_v20_4 import (
    FusionPredictionDataset, VelocityFusionGate, fuse_velocity, fusion_features,
)
from src.task.InteractionDynamics.train_h_realizer_v20_2 import error_metrics


@torch.no_grad()
def evaluate(model, loader, device):
    predictions, targets, gates = [], [], []
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        gate = model(fusion_features(batch["y_free"], batch["y_h"]))
        predictions.append(fuse_velocity(batch["y_free"], batch["y_h"], gate).cpu())
        targets.append(batch["future_y"].cpu()); gates.append(gate.cpu())
    prediction, target, gate = torch.cat(predictions), torch.cat(targets), torch.cat(gates)
    return {"error": error_metrics(prediction[..., :7], target), "gate_mean": float(gate.mean()),
            "gate_std": float(gate.std()), "gate_by_timestep": gate.mean((0, 1, 3)).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    args = parser.parse_args(); torch.manual_seed(42); device = torch.device("cuda")
    train = FusionPredictionDataset(args.cache / "train.pt")
    val = FusionPredictionDataset(args.cache / "val.pt")
    loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=0)
    validation = DataLoader(val, args.batch_size, shuffle=False, num_workers=0)
    field_std = train.data["field_std"].to(device); v_std = field_std[..., 4:7]
    model = VelocityFusionGate(hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    iterator = iter(loader); best = float("inf"); args.output.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try: raw = next(iterator)
        except StopIteration: iterator = iter(loader); raw = next(iterator)
        batch = {key: value.to(device) for key, value in raw.items()}
        gate = model(fusion_features(batch["y_free"], batch["y_h"]))
        prediction = fuse_velocity(batch["y_free"], batch["y_h"], gate)
        velocity_loss = ((prediction[..., 4:7] - batch["future_y"][..., 4:7]) / v_std).square().mean()
        smooth = (gate[..., 1:, :] - gate[..., :-1, :]).square().mean()
        loss = velocity_loss + args.smooth_weight * smooth
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step == 1:
            gradients = [value.grad for value in model.parameters() if value.grad is not None]
            print(json.dumps({"gradient_gate": {"finite": all(torch.isfinite(x).all() for x in gradients),
                  "nonzero": any(float(x.abs().sum()) > 0 for x in gradients)}}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model.eval(), validation, device); model.train()
            row = {"step": step, "loss": float(loss), "velocity_loss": float(velocity_loss),
                   "smooth": float(smooth), "validation": metrics}; print(json.dumps(row), flush=True)
            payload = {"model": model.state_dict(), "args": vars(args), "step": step,
                       "validation": metrics}
            torch.save(payload, args.output / "latest.pt")
            score = metrics["error"]["v_rmse_cm"]
            if score < best:
                best = score; torch.save(payload, args.output / "best.pt")


if __name__ == "__main__":
    main()
