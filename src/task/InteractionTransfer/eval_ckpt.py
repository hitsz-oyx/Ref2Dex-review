"""V0.9 checkpoint 定量验证：test split 上 GT/Zero/Reverse/Cross 的 EPE。

用法（cache 模式，推荐）::

    python -m src.task.InteractionTransfer.eval_ckpt \\
        --cache-root data/processed_data/stage4/interactiontransfer_static_cache \\
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \\
        --checkpoint outputs/InteractionTransfer/v08_full_20ep/best.pt \\
        --device cuda:0 --batch-size 8 \\
        --output outputs/InteractionTransfer/v08_full_20ep/test_metrics.json

也可用 ``--root`` 直接读 raw GRAB cache（每次 forward 重跑 frozen PTv3，慢）。
Frozen DenseToken 从原 DenseToken checkpoint 加载；trainable modules 从
--checkpoint 加载。指标只算 Object Flow EPE（mm）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.task.InteractionTransfer.cached_dataset import CachedTransitionDataset
from src.task.InteractionTransfer.dataset import GRABOneStepDataset
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.model import InteractionTransfer
from src.task.InteractionTransfer.train_full import STATIC_KEYS, FlowIntervention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache-root", help="static cache 目录（推荐）")
    source.add_argument("--root", help="raw GRAB stage4 cache（慢速参照）")
    parser.add_argument("--split", type=Path, required=True, help="test.txt 路径")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt / last.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--output", type=Path, default=None, help="结果 json 输出路径")
    return parser.parse_args()


@torch.no_grad()
def evaluate_mode(model, dataset, mode, args, device, cached) -> float:
    base = dataset if mode == "gt" else FlowIntervention(dataset, mode)
    loader = DataLoader(base, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    total, count = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
        if cached:
            pred = model.forward_core(**{k: batch[k] for k in STATIC_KEYS},
                                      hand_flow=batch["hand_flow"])["object_flow"]
        else:
            pred = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                         batch["hand_normals"], batch["hand_flow"])["object_flow"]
        total += epe(pred, batch["object_flow"]) * batch["object_flow"].shape[0]
        count += batch["object_flow"].shape[0]
    return total / count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    sequences = [line.strip() for line in args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.cache_root:
        dataset = CachedTransitionDataset(args.cache_root, sequences)
        cached = True
    else:
        dataset = GRABOneStepDataset(args.root, sequences, max_transitions=0)
        cached = False

    model = InteractionTransfer(dense_checkpoint=args.dense_checkpoint).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    current = model.state_dict()
    current.update(ckpt["model"])
    model.load_state_dict(current)
    epoch = int(ckpt.get("epoch", -1))

    print(f"checkpoint: {args.checkpoint} (epoch {epoch})")
    print(f"test transitions: {len(dataset)}")

    results = {}
    for mode in ("gt", "zero", "reverse", "cross"):
        results[mode] = evaluate_mode(model, dataset, mode, args, device, cached) * 1000
        print(f"Test EPE [{mode:>7}] = {results[mode]:.2f} mm")

    gate = results["gt"] < min(results["zero"], results["reverse"], results["cross"])
    print(f"Quantitative gate (GT < zero/reverse/cross): {'PASS' if gate else 'FAIL'}")

    payload = {"checkpoint": str(args.checkpoint), "epoch": epoch,
               "num_test_transitions": len(dataset), "test_epe_mm": results,
               "gt_beats_interventions": bool(gate)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
