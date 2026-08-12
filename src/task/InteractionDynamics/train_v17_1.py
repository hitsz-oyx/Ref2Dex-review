"""双卡训练 V17.1 deterministic baseline/time-to-effect 对照。"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.task.InteractionDynamics.dataset_v17 import CachedV17Dataset
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future
from src.task.InteractionDynamics.residual_interaction_regression_v17_1 import (
    TemporalResidualInteractionRegressor, temporal_goal)
from src.task.InteractionDynamics.train_v17 import stats_for


class TemporalDataset(Dataset):
    def __init__(self, cache_root: str | Path, time_root: str | Path, split: str) -> None:
        self.base = CachedV17Dataset(cache_root, split)
        self.time_to_effect = torch.load(Path(time_root) / f"{split}.pt", map_location="cpu")
        if len(self.base) != len(self.time_to_effect):
            raise ValueError(f"{split} cache/time metadata length mismatch")

    def __len__(self): return len(self.base)

    def __getitem__(self, index):
        row = self.base[index]
        row["time_to_effect"] = self.time_to_effect[index]
        return row


def tau_statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.log1p(values[values >= 0].float())
    if not len(values):
        raise ValueError("training split has no valid time-to-effect")
    return values.mean(), values.std().clamp_min(.01)


def bucket_masks(time_to_effect: torch.Tensor, dynamic: torch.Tensor) -> dict[str, torch.Tensor]:
    valid = time_to_effect >= 0
    return {"dynamic": dynamic, "missing": dynamic & ~valid,
            "0_10": dynamic & valid & (time_to_effect < 10),
            "10_30": dynamic & (time_to_effect >= 10) & (time_to_effect < 30),
            "30_100": dynamic & (time_to_effect >= 30) & (time_to_effect < 100),
            "100_plus": dynamic & (time_to_effect >= 100)}


@torch.inference_mode()
def evaluate(model, loader, stats, tau_mean, tau_std, device, use_time):
    totals = {source: {name: {key: 0. for key in ("r", "d", "n")}
                       for name in ("dynamic", "missing", "0_10", "10_30", "30_100", "100_plus")}
              for source in ("persistence", "model")}
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        state, future, residual = batch["state"], batch["future"], batch["residual"]
        persistence = persistence_future(state, 4)
        normalized_goal = (batch["goal"] - stats["goal_mean"]) / stats["goal_std"]
        condition = temporal_goal(normalized_goal, batch["time_to_effect"],
                                  tau_mean, tau_std, use_time)
        prediction = model((state - stats["state_mean"]) / stats["state_std"],
                           batch["anchors_cm"], batch["object_patches"], condition)
        prediction = persistence + prediction * stats["residual_std"] + stats["residual_mean"]
        masks = bucket_masks(batch["time_to_effect"], residual.square().mean((1, 2)).sqrt() >= .2)
        for source, value in (("persistence", persistence), ("model", prediction)):
            value = value.reshape(*value.shape[:-1], 4, 7)
            target = future.reshape(*future.shape[:-1], 4, 7)
            for name, mask in masks.items():
                p, t = value[mask], target[mask]
                if not len(p): continue
                row = totals[source][name]
                row["r"] += (p[..., 3:6] - t[..., 3:6]).square().sum().item()
                row["d"] += (p[..., 6] - t[..., 6]).square().sum().item()
                row["n"] += p[..., 6].numel()
    result = {}
    for source, buckets in totals.items():
        result[source] = {}
        for name, row in buckets.items():
            if row["n"]:
                result[source][name] = {"r_rmse_cm": (row["r"] / (3 * row["n"])) ** .5,
                                        "d_rmse_cm": (row["d"] / row["n"]) ** .5,
                                        "elements": int(row["n"])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/v17_1_time_to_effect.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--use-time-to-effect", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local = (int(os.getenv(name, default)) for name, default in
                          (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if world > 1:
        torch.cuda.set_device(local); dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    seed = config["seed"] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = config["data"]
    train = TemporalDataset(data["cache_root"], data["time_to_effect_root"], "train")
    val = TemporalDataset(data["cache_root"], data["time_to_effect_root"], "val")
    test = TemporalDataset(data["cache_root"], data["time_to_effect_root"], "test")
    args.output.mkdir(parents=True, exist_ok=True)
    stats_path = args.output / "train_statistics.pt"
    if rank == 0: stats = stats_for(train, stats_path)
    if world > 1: dist.barrier(); stats = torch.load(stats_path, map_location="cpu")
    stats = {key: value.to(device) for key, value in stats.items()}
    tau_mean, tau_std = tau_statistics(train.time_to_effect)
    tau_mean, tau_std = tau_mean.to(device), tau_std.to(device)
    model_config = config["model"]
    model = TemporalResidualInteractionRegressor(
        4, 4, model_config["dim"], model_config["heads"], model_config["layers"]).to(device)
    if world > 1: model = DDP(model, device_ids=[local])
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"],
                                  weight_decay=training["weight_decay"])
    sampler = DistributedSampler(train, world, rank, shuffle=True) if world > 1 else None
    loader = DataLoader(train, batch_size=training["per_gpu_batch"], sampler=sampler,
                        shuffle=sampler is None, num_workers=training["num_workers"],
                        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=training["per_gpu_batch"], shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    best = float("inf"); global_step = 0
    for epoch in range(training["epochs"]):
        if sampler: sampler.set_epoch(epoch)
        model.train(); started = time.time()
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            state = (batch["state"] - stats["state_mean"]) / stats["state_std"]
            goal = (batch["goal"] - stats["goal_mean"]) / stats["goal_std"]
            goal = temporal_goal(goal, batch["time_to_effect"], tau_mean, tau_std,
                                 args.use_time_to_effect)
            target = (batch["residual"] - stats["residual_mean"]) / stats["residual_std"]
            permutation = torch.stack([torch.randperm(128, device=device) for _ in range(len(state))])
            def gather(value):
                index = permutation.reshape(*permutation.shape, *([1] * (value.ndim - 2)))
                return value.gather(1, index.expand_as(value))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(gather(state), gather(batch["anchors_cm"]),
                                   gather(batch["object_patches"]), goal)
                loss = torch.nn.functional.mse_loss(prediction, gather(target))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); global_step += 1
        if world > 1: dist.barrier()
        if rank == 0:
            raw = model.module if world > 1 else model; raw.eval()
            metrics = evaluate(raw, val_loader, stats, tau_mean, tau_std, device,
                               args.use_time_to_effect)
            dynamic = metrics["model"]["dynamic"]
            score = dynamic["r_rmse_cm"] + dynamic["d_rmse_cm"]
            row = {"epoch": epoch + 1, "global_step": global_step,
                   "seconds": time.time() - started, "train_loss": float(loss), "val": metrics}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            checkpoint = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch + 1, "global_step": global_step,
                          "use_time_to_effect": args.use_time_to_effect,
                          "tau_mean": tau_mean.cpu(), "tau_std": tau_std.cpu(),
                          "stats": {key: value.cpu() for key, value in stats.items()},
                          "config": config, "val": metrics}
            torch.save(checkpoint, args.output / "latest.pt")
            if score < best: best = score; torch.save(checkpoint, args.output / "best.pt")
        if world > 1: dist.barrier()
    if rank == 0:
        checkpoint = torch.load(args.output / "best.pt", map_location=device)
        raw = model.module if world > 1 else model
        raw.load_state_dict(checkpoint["model"]); raw.eval()
        test_metrics = evaluate(raw, DataLoader(test, batch_size=training["per_gpu_batch"],
                                shuffle=False, num_workers=2), stats, tau_mean, tau_std,
                                device, args.use_time_to_effect)
        result = {"best_epoch": checkpoint["epoch"], "use_time_to_effect": args.use_time_to_effect,
                  "val": checkpoint["val"], "test": test_metrics}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"final": result}, ensure_ascii=False), flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__":
    main()
