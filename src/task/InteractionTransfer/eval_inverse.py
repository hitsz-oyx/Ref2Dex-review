"""V1.0 inverse generalization：test split 上批量 inverse optimization 的 EPE。

流程（每个 test transition）::

    target = GT object_flow
    zero init -> optimize (rigid 12D / free) -> optimized action -> EPE

输出对比表：Zero action / GT action / Optimized rigid / Optimized free。
rigid 为双手各自 SE(3)（ξ_L, ξ_R 共 12 维，指导 §8）。

用法::

    python -m src.task.InteractionTransfer.eval_inverse \\
        --stage4-root data/processed_data/stage4/data/grab \\
        --geometry-root data/processed_data/stage4/interactiontransfer_geometry_cache \\
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \\
        --checkpoint outputs/InteractionTransfer/v10_full_20ep/best.pt \\
        --device cuda:0 --num-samples 500 --batch-size 16 \\
        --steps 300 --lr 0.01 \\
        --output outputs/InteractionTransfer/v10_full_20ep/test_inverse_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.task.InteractionTransfer.dataset import GRABRandomTransitionDataset
from src.task.InteractionTransfer.inverse_optimize import optimize_hand_flow
from src.task.InteractionTransfer.model import InteractionTransfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=500, help="0 = 全部 test transitions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--target-kind", choices=("effect", "c_obj"), default="effect")
    parser.add_argument("--init", choices=("zero", "random", "cross", "gt"), default="zero")
    parser.add_argument("--gaps", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--parameterizations", nargs="+", default=("rigid", "free"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def make_init_flow(kind: str, gt_flow: torch.Tensor) -> torch.Tensor:
    if kind == "zero":
        return torch.zeros_like(gt_flow)
    if kind == "random":
        return (0.01 * torch.randn_like(gt_flow)).clamp(-0.03, 0.03)
    if kind == "gt":
        return gt_flow.clone()
    raise ValueError(f"init {kind} 需要 cross flow，由 viewer 提供；批量 eval 请用 zero/random/gt")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sequences = [line.strip() for line in args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = GRABRandomTransitionDataset(args.stage4_root, args.geometry_root, sequences,
                                          gaps=tuple(args.gaps), seed=args.seed, deterministic=True)
    if args.num_samples and args.num_samples < len(dataset):
        indices = np.linspace(0, len(dataset) - 1, args.num_samples).astype(int).tolist()
    else:
        indices = list(range(len(dataset)))
    print(f"test transitions: {len(dataset)}, sampled: {len(indices)}")

    model = InteractionTransfer(dense_checkpoint=args.dense_checkpoint).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current = model.state_dict()
    current.update(ckpt["model"])
    model.load_state_dict(current)
    epoch = int(ckpt.get("epoch", -1))
    print(f"checkpoint: {args.checkpoint} (epoch {epoch})")

    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=4)

    summaries = {f"opt_{p}": [] for p in args.parameterizations}
    zero_epe, gt_epe = [], []
    sample_history = None
    for param in args.parameterizations:
        zero_epe, gt_epe = [], []
        print(f"\n=== parameterization: {param} (init={args.init}, target={args.target_kind}, "
              f"steps={args.steps}, lr={args.lr}) ===")
        for bi, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.no_grad():
                n_left = batch["hand_points"].shape[1] // 2
                static = model.encode_static(batch["object_points"], batch["object_normals"],
                                             batch["hand_points"][:, :n_left],
                                             batch["hand_normals"][:, :n_left],
                                             batch["hand_points"][:, n_left:],
                                             batch["hand_normals"][:, n_left:])
            init_flow = make_init_flow(args.init, batch["hand_flow"])
            result = optimize_hand_flow(model, static, batch["hand_points"], batch["object_flow"],
                                        gt_flow=batch["hand_flow"], init_flow=init_flow,
                                        dt=batch["gap"], parameterization=param,
                                        target_kind=args.target_kind,
                                        steps=args.steps, lr=args.lr, history_every=50)
            zero_epe.append(result["initial"]["epe_mm"])
            gt_epe.append(result["gt"]["epe_mm"])
            summaries[f"opt_{param}"].append(result["optimized"]["epe_mm"])
            if bi == 0 and sample_history is None:
                sample_history = result["history"]
            done = min((bi + 1) * args.batch_size, len(indices))
            print(f"  [{done}/{len(indices)}] zero {np.mean(zero_epe):.2f} mm | "
                  f"opt {np.mean(summaries[f'opt_{param}']):.2f} mm | gt {np.mean(gt_epe):.2f} mm", flush=True)

    table = {
        "checkpoint": str(args.checkpoint), "epoch": epoch,
        "num_samples": len(indices), "init": args.init,
        "target_kind": args.target_kind, "steps": args.steps, "lr": args.lr,
        "gaps": list(args.gaps),
        "test_epe_mm": {
            "zero_action": float(np.mean(zero_epe)),
            **{f"optimized_{p}": float(np.mean(summaries[f"opt_{p}"])) for p in args.parameterizations},
            "gt_action": float(np.mean(gt_epe)),
        },
        "sample_history": sample_history,
    }
    print("\n=== Test inverse EPE ===")
    for k, v in table["test_epe_mm"].items():
        print(f"{k:>18}: {v:.2f} mm")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(table, indent=2), encoding="utf-8")
        print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
