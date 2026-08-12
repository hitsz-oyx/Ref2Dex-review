"""V18 stable-grasp residual v-diffusion 四卡训练。"""
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
from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.task.InteractionDynamics.dataset_grasp_v18 import CachedGraspDataset
from src.task.InteractionDynamics.eval_grasp_v18 import evaluate
from src.task.InteractionDynamics.grasp_interaction_diffusion import GraspInteractionDiffusion
from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
from src.task.InteractionDynamics.residual_interaction_diffusion import make_v_target


def statistics(dataset, path: Path):
    if path.exists(): return torch.load(path, map_location="cpu")
    result = {}
    for key in ("state", "residual"):
        values = torch.cat([dataset[i][key] for i in range(len(dataset))], 0).double()
        result[key + "_mean"] = values.mean(0).float().reshape(1, 1, -1)
        result[key + "_std"] = values.std(0).float().clamp_min(.05).reshape(1, 1, -1)
    torch.save(result, path); return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/InteractionDynamics/configs/v18_grasp.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args(); config = yaml.safe_load(Path(args.config).read_text())
    rank, world, local = (int(os.getenv(name, default)) for name, default in
                          (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if world > 1: torch.cuda.set_device(local); dist.init_process_group("nccl")
    device = torch.device("cuda", local); seed = config["seed"] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train = CachedGraspDataset(config["data"]["cache_root"], "train")
    val = CachedGraspDataset(config["data"]["cache_root"], "val")
    if args.max_samples:
        train = Subset(train, range(min(args.max_samples, len(train))))
        val = Subset(train, range(len(train)))
    args.output.mkdir(parents=True, exist_ok=True); stats_path = args.output / "statistics.pt"
    if rank == 0: stats = statistics(train, stats_path)
    if world > 1: dist.barrier(); stats = torch.load(stats_path, map_location="cpu")
    stats = {key: value.to(device) for key, value in stats.items()}
    model_cfg = config["model"]
    model = GraspInteractionDiffusion(model_cfg["horizon"], model_cfg["dim"],
                                     model_cfg["heads"], model_cfg["layers"]).to(device)
    if world > 1: model = DDP(model, device_ids=[local])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"])
    start_epoch = 0; global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        (model.module if world > 1 else model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
    sampler = DistributedSampler(train, world, rank, shuffle=True) if world > 1 else None
    loader = DataLoader(train, batch_size=config["training"]["per_gpu_batch"], sampler=sampler,
                        shuffle=sampler is None, num_workers=config["training"]["num_workers"],
                        pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=config["training"]["per_gpu_batch"],
                            shuffle=False, num_workers=2)
    _, alpha_bar = cosine_schedule(config["diffusion"]["steps"], device)
    epochs = args.epochs or config["training"]["epochs"]
    best_rmse = float("inf"); best_stable = (-1., float("inf"))
    checkpoint_dir = args.output / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    for epoch in range(start_epoch, epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train(); started = time.time(); losses = []
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            state = (batch["state"] - stats["state_mean"]) / stats["state_std"]
            residual = (batch["residual"] - stats["residual_mean"]) / stats["residual_std"]
            timestep = torch.randint(config["diffusion"]["steps"], (len(state),), device=device)
            noise = torch.randn_like(residual)
            noisy, target = make_v_target(residual, noise, alpha_bar[timestep][:, None, None])
            permutation = torch.stack([torch.randperm(128, device=device) for _ in range(len(state))])
            def gather(value):
                index = permutation.reshape(*permutation.shape, *([1] * (value.ndim - 2)))
                return value.gather(1, index.expand_as(value))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(gather(noisy), gather(state), gather(batch["anchors_cm"]),
                                   gather(batch["object_patches"]), timestep)
                loss = torch.nn.functional.mse_loss(prediction, gather(target))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            global_step += 1; losses.append(float(loss))
            if args.max_steps and global_step >= args.max_steps: break
        if world > 1: dist.barrier()
        should_evaluate = (epoch + 1) % args.eval_every == 0 or epoch + 1 == epochs or (args.max_steps and global_step >= args.max_steps)
        if rank == 0 and should_evaluate:
            raw = model.module if world > 1 else model; raw.eval()
            metrics = evaluate(raw, val_loader, stats, device, samples=1,
                               sampling_steps=config["diffusion"]["sampling_steps"])
            score = metrics["single_sample_rmse_cm"]
            row = {"epoch": epoch + 1, "global_step": global_step,
                   "seconds": time.time() - started, "train_loss": float(np.mean(losses)), "val": metrics}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            checkpoint = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                          "epoch": epoch + 1, "global_step": global_step,
                          "stats": {key: value.cpu() for key, value in stats.items()},
                          "config": config, "val": metrics}
            torch.save(checkpoint, args.output / "latest.pt")
            if (epoch + 1) % args.save_every == 0:
                torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")
            if score < best_rmse:
                best_rmse = score; torch.save(checkpoint, args.output / "best_rmse.pt")
            stable_key = (metrics["diffusion_stable_success_rate"], -score)
            if stable_key > (best_stable[0], -best_stable[1]):
                best_stable = (stable_key[0], score)
                torch.save(checkpoint, args.output / "best.pt")
        if world > 1: dist.barrier()
        if args.max_steps and global_step >= args.max_steps: break
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
