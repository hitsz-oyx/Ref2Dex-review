"""V18.4 deterministic MANO-H residual regression 训练。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition


def evenly_spaced(dataset, count: int | None):
    if count is None or count >= len(dataset): return dataset
    indices = torch.linspace(0, len(dataset) - 1, count).round().long().unique().tolist()
    return Subset(dataset, indices)


def statistics(dataset) -> dict[str, torch.Tensor]:
    values = torch.stack([dataset[i]["future_delta_h"] for i in range(len(dataset))]).double()
    return {"delta_mean": values.mean((0, 1)).float().reshape(1, 1, 30),
            "delta_std": values.std((0, 1)).float().clamp_min(1e-3).reshape(1, 1, 30)}


@torch.inference_mode()
def evaluate(model, loader, stats, device) -> dict[str, float]:
    squared = torch.zeros(3, device=device); count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(batch["state"], batch["anchors_cm"], batch["object_patches"],
                           batch["current_h"]) * stats["delta_std"] + stats["delta_mean"]
        error = prediction - batch["future_delta_h"]
        squared += torch.stack([error[..., :3].square().sum(), error[..., 3:6].square().sum(),
                                error[..., 6:].square().sum()])
        count += error.shape[0] * error.shape[1]
    return {"translation_rmse_cm": float((squared[0] / (count * 3)).sqrt()),
            "rotation_rmse_rad": float((squared[1] / (count * 3)).sqrt()),
            "pose_rmse": float((squared[2] / (count * 24)).sqrt())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--controlled32", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); device = torch.device("cuda")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    full_train = CachedManoHDataset(args.cache, "train")
    train = (Subset(full_train, range(min(32, len(full_train)))) if args.controlled32
             else evenly_spaced(full_train, args.train_samples))
    val = train if args.controlled32 else evenly_spaced(CachedManoHDataset(args.cache, "val"), args.val_samples)
    stats = {key: value.to(device) for key, value in statistics(train).items()}
    model = ManoHandTransition(dim=args.dim, layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val, args.batch_size, shuffle=False, num_workers=2)
    args.output.mkdir(parents=True, exist_ok=True); iterator = iter(loader); best = float("inf")
    for step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        target = (batch["future_delta_h"] - stats["delta_mean"]) / stats["delta_std"]
        prediction = model(batch["state"], batch["anchors_cm"], batch["object_patches"], batch["current_h"])
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if step % 250 == 0 or step == args.steps:
            metrics = evaluate(model.eval(), val_loader, stats, device); model.train()
            score = metrics["translation_rmse_cm"] + 10 * metrics["rotation_rmse_rad"] + metrics["pose_rmse"]
            row = {"step": step, "train_loss": float(loss), "val": metrics}; print(json.dumps(row), flush=True)
            payload = {"model": model.state_dict(), "stats": {k: v.cpu() for k, v in stats.items()},
                       "step": step, "args": vars(args), "val": metrics}
            torch.save(payload, args.output / "latest.pt")
            if score < best: best = score; torch.save(payload, args.output / "best.pt")


if __name__ == "__main__": main()
