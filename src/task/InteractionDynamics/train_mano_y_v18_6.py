"""V18.6 full-data 多卡 MANO-constrained Y-supervised 训练。"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.mano_hand_transition import CachedManoHDataset, ManoHandTransition
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import decode_mano_y
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future
from src.task.InteractionDynamics.train_mano_h_v18_4 import statistics
from src.task.InteractionDynamics.train_mano_y_v18_5 import ManoLayers, evaluate, move, residual_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--per-gpu-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prior-weight", type=float, default=0.)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="ref2dex")
    parser.add_argument("--wandb-group", default="interaction-dynamics-v18-6")
    parser.add_argument("--wandb-name", default="v18-6-mano-y-full-ddp")
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    args = parser.parse_args()
    rank = int(os.getenv("RANK", "0")); world = int(os.getenv("WORLD_SIZE", "1")); local = int(os.getenv("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local); dist.init_process_group("nccl")
    device = torch.device("cuda", local); torch.manual_seed(42 + rank)
    train = CachedManoHDataset(args.cache, "train"); val = CachedManoHDataset(args.cache, "val")
    args.output.mkdir(parents=True, exist_ok=True); stats_path = args.output / "statistics.pt"
    if rank == 0 and not stats_path.exists():
        payload = statistics(train); payload["residual_std"] = residual_std(train)
        torch.save(payload, stats_path)
    if world > 1: dist.barrier()
    payload = torch.load(stats_path, map_location="cpu")
    delta_stats = {key: payload[key].to(device) for key in ("delta_mean", "delta_std")}
    r_std = payload["residual_std"].to(device)
    model = ManoHandTransition(dim=args.dim, layers=args.layers).to(device)
    if world > 1: model = DDP(model, device_ids=[local])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sampler = DistributedSampler(train, world, rank, shuffle=True) if world > 1 else None
    loader = DataLoader(train, args.per_gpu_batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val, args.per_gpu_batch, shuffle=False, num_workers=0)
    mano_layers = ManoLayers(args.grab_root, args.mano_path, device)
    run = None
    if rank == 0 and args.wandb:
        import wandb
        wandb_config = {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()}
        wandb_config.update({"world_size": world,
                             "global_batch": world * args.per_gpu_batch})
        run = wandb.init(project=args.wandb_project, group=args.wandb_group,
                         name=args.wandb_name, mode="online",
                         config=wandb_config)
    best = float("inf"); global_step = 0; training_started = time.time()
    for epoch in range(args.epochs):
        if sampler: sampler.set_epoch(epoch)
        model.train(); epoch_started = time.time(); losses = []
        for raw in loader:
            batch = move(raw, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = model(batch["state"], batch["anchors_cm"],
                                   batch["object_patches"], batch["current_h"])
            delta = normalized.float() * delta_stats["delta_std"] + delta_stats["delta_mean"]
            future, _ = decode_mano_y(delta, batch, mano_layers)
            pred_residual = future - persistence_future(batch["state"], 8)
            loss_y = ((pred_residual - batch["residual"]) / r_std).square().mean()
            prior = torch.relu(normalized.float().abs() - 3).square().mean()
            loss = loss_y + args.prior_weight * prior
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss_y)); global_step += 1
            if args.max_steps and global_step >= args.max_steps: break
        if world > 1: dist.barrier()
        should_eval = ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs
                       or (args.max_steps and global_step >= args.max_steps))
        if rank == 0:
            row = {"epoch": epoch + 1, "global_step": global_step,
                   "epoch_seconds": time.time() - epoch_started,
                   "elapsed_hours": (time.time() - training_started) / 3600,
                   "train_loss_y": sum(losses) / max(len(losses), 1)}
            if should_eval:
                raw_model = model.module if world > 1 else model
                metrics = evaluate(raw_model.eval(), val_loader, delta_stats, r_std,
                                   mano_layers, device); row["val"] = metrics
                score = (metrics["r_rmse_cm"] + metrics["u_rmse_cm"] + metrics["d_rmse_cm"]
                         + .02 * metrics["contact_mean_abs_error"])
                checkpoint = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                    "delta_stats": {key: value.cpu() for key, value in delta_stats.items()},
                    "residual_std": r_std.cpu(), "args": vars(args), "epoch": epoch + 1,
                    "global_step": global_step, "val": metrics}
                torch.save(checkpoint, args.output / "latest.pt")
                if score < best: best = score; torch.save(checkpoint, args.output / "best.pt")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if run:
                flat = {key: value for key, value in row.items() if key != "val"}
                if "val" in row: flat.update({"val/" + key: value for key, value in row["val"].items()})
                run.log(flat, step=global_step)
        if world > 1: dist.barrier()
        if args.max_steps and global_step >= args.max_steps: break
    if run: run.finish()
    if world > 1: dist.destroy_process_group()


if __name__ == "__main__": main()
