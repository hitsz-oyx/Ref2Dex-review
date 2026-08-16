"""V1.0 正式训练：geometry cache + online sampling 全量 transition。

用法::

    python -m src.task.InteractionTransfer.train_full \\
        --stage4-root data/processed_data/stage4/data/grab \\
        --geometry-root data/processed_data/stage4/interactiontransfer_geometry_cache \\
        --split-dir src/task/InteractionTransfer/splits/grab_seed42 \\
        --epochs 20 --batch-size 8 --num-workers 4 --device cuda:0 \\
        --output outputs/InteractionTransfer/v10_full_20ep

V1.0（指导 §一次性升级）：随机 MANO surface sampling（769+769，每 epoch
在线重采）、随机 gap ∈ {1,2,4,8}、9D action（含 Δt）、双手 split KNN 8+8。
static 部分不再离线缓存，每次 forward 在线跑 frozen PTv3（V1.0 接受的
代价）。train split 每个 epoch 随机 gap/采样；val/test 按 index 播种可
复现。训练 loss 保持 SmoothL1；模型选择与报告只看 EPE。每个 epoch 结束
做一次 validation，保存 best.pt / last.pt；训练结束后用 best checkpoint
在 test split 上报告 GT/zero/reverse/cross 的 EPE。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.task.InteractionTransfer.dataset import GRABRandomTransitionDataset
from src.task.InteractionTransfer.losses import flow_loss
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.model import InteractionTransfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", required=True)
    parser.add_argument("--geometry-root", required=True,
                        help="geometry_cache.py 生成的 MANO 顶点 cache 目录")
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--gaps", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--hand-points-per-side", type=int, default=769)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/InteractionTransfer/v10_full_20ep"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--skip-interventions", action="store_true",
                        help="最终 test 只评 GT EPE，跳过 zero/reverse/cross")
    return parser.parse_args()


class FlowIntervention(Dataset):
    """test 阶段对 hand_flow 做干预；cross 同时替换 Δt（flow 与 gap 配套）。"""

    def __init__(self, base: Dataset, mode: str):
        self.base, self.mode = base, mode
        self.perm = np.roll(np.arange(len(base)), 1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        if self.mode == "zero":
            sample["hand_flow"] = torch.zeros_like(sample["hand_flow"])
        elif self.mode == "reverse":
            sample["hand_flow"] = -sample["hand_flow"]
        elif self.mode == "cross":
            other = self.base[int(self.perm[index])]
            sample["hand_flow"] = other["hand_flow"].clone()
            sample["gap"] = other["gap"].clone()
        return sample


def model_output(model: InteractionTransfer, batch: dict) -> torch.Tensor:
    n_left = batch["hand_points"].shape[1] // 2
    return model(batch["object_points"], batch["object_normals"],
                 batch["hand_points"][:, :n_left], batch["hand_normals"][:, :n_left],
                 batch["hand_points"][:, n_left:], batch["hand_normals"][:, n_left:],
                 batch["hand_flow"], batch["gap"])["object_flow"]


def make_loader(dataset, args, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=True,
                      persistent_workers=args.num_workers > 0, drop_last=False)


def build_datasets(args):
    def read(name):
        return [line.strip() for line in (args.split_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    splits = {name: read(name) for name in ("train", "val", "test")}
    common = dict(stage4_root=args.stage4_root, geometry_root=args.geometry_root,
                  object_points=512, hand_points_per_side=args.hand_points_per_side,
                  gaps=tuple(args.gaps), seed=args.seed)
    return {name: GRABRandomTransitionDataset(sequences=seqs, deterministic=name != "train", **common)
            for name, seqs in splits.items()}


@torch.no_grad()
def evaluate_epe(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
        pred = model_output(model, batch)
        total += epe(pred, batch["object_flow"]) * batch["object_flow"].shape[0]
        count += batch["object_flow"].shape[0]
    return total / count


def save_checkpoint(path: Path, model, optimizer, epoch, best_val_epe, run_config, split_info):
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {key: value.detach().cpu() for key, value in model.state_dict().items() if key in trainable}
    torch.save({"epoch": epoch, "best_val_epe": best_val_epe, "model": state,
                "optimizer": optimizer.state_dict(), "config": run_config, "split": split_info}, path)


def load_trainable(model, ckpt):
    current = model.state_dict()
    current.update(ckpt["model"])
    model.load_state_dict(current)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(args)
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} transitions (starts)", flush=True)

    split_info = {name: {"num_transitions": len(dataset)} for name, dataset in datasets.items()}
    run_config = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                  "seed": args.seed, "stage4_root": args.stage4_root,
                  "geometry_root": args.geometry_root,
                  "gaps": list(args.gaps), "hand_points_per_side": args.hand_points_per_side,
                  "dense_checkpoint": args.dense_checkpoint,
                  "dataset_config": datasets["train"].config}
    (args.output / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (args.output / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    train_loader = make_loader(datasets["train"], args, shuffle=True)
    val_loader = make_loader(datasets["val"], args, shuffle=False)

    model = InteractionTransfer(dense_checkpoint=args.dense_checkpoint).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    metrics = {"epoch": [], "train_loss": [], "val_epe_mm": []}
    start_epoch, best_val_epe = 1, float("inf")
    last_path, best_path = args.output / "last.pt", args.output / "best.pt"
    if last_path.is_file():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        load_trainable(model, ckpt)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_epe = float(ckpt["best_val_epe"])
        metrics_path = args.output / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"从 last.pt 恢复：epoch {ckpt['epoch']}，best val EPE {best_val_epe * 1000:.2f} mm", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        assert not model.static_encoder.training
        loss_sum, count = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad(set_to_none=True)
            pred = model_output(model, batch)
            loss = flow_loss(pred, batch["object_flow"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}: {loss}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * batch["object_flow"].shape[0]
            count += batch["object_flow"].shape[0]
        train_loss = loss_sum / count
        val_epe = evaluate_epe(model, val_loader, device)
        metrics["epoch"].append(epoch)
        metrics["train_loss"].append(train_loss)
        metrics["val_epe_mm"].append(val_epe * 1000)
        is_best = val_epe < best_val_epe
        if is_best:
            best_val_epe = val_epe
        save_checkpoint(last_path, model, optimizer, epoch, best_val_epe, run_config, split_info)
        if is_best:
            save_checkpoint(best_path, model, optimizer, epoch, best_val_epe, run_config, split_info)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        tag = " | BEST" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.5f} | "
              f"val_EPE={val_epe * 1000:.2f} mm{tag}", flush=True)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    load_trainable(model, ckpt)
    print(f"加载 best.pt（epoch {ckpt['epoch']}，val EPE {ckpt['best_val_epe'] * 1000:.2f} mm）", flush=True)

    test_loaders = {"gt": make_loader(datasets["test"], args, shuffle=False)}
    if not args.skip_interventions:
        for mode in ("zero", "reverse", "cross"):
            test_loaders[mode] = make_loader(FlowIntervention(datasets["test"], mode), args, shuffle=False)
    results = {}
    for mode, loader in test_loaders.items():
        results[mode] = evaluate_epe(model, loader, device) * 1000
        print(f"Test EPE [{mode}] = {results[mode]:.2f} mm", flush=True)
    (args.output / "test_metrics.json").write_text(
        json.dumps({"best_epoch": int(ckpt["epoch"]), "best_val_epe_mm": ckpt["best_val_epe"] * 1000,
                    "test_epe_mm": results}, indent=2), encoding="utf-8")
    print(json.dumps({"test_epe_mm": results}), flush=True)


if __name__ == "__main__":
    main()
