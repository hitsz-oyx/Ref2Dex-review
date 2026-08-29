"""Benchmark online-online versus online-cache DenseToken variability.

This is the V1.1.1 gate for a real scene root.  It keeps the dataset view
fixed, runs the frozen encoder twice online, and compares both feature-level
and Cm-head-level differences against the cached features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.base import task_config_from_dict
from src.task.Cm.dataset.scene import Stage4CmSceneDataset
from src.task.Cm.src.model import CmFlowModel


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _head(model: CmFlowModel, batch: dict[str, torch.Tensor], z_obj: torch.Tensor,
          z_hand: torch.Tensor, contact: torch.Tensor) -> dict[str, torch.Tensor]:
    return model.head(
        z_obj=z_obj,
        z_hand=z_hand,
        dense_hand_contact=contact,
        obj_points=batch["obj_points"],
        obj_normals=batch["obj_normals"],
        hand_points=batch["hand_points"],
        hand_normals=batch["hand_normals"],
        hand_flow=batch["hand_flow"],
        obj_valid_mask=batch["obj_valid_mask"],
        delta_time_s=batch["delta_time_s"],
    )


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.float() ** 2)).cpu())


def _epe_mm(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
    error = torch.linalg.norm(pred.float() - target.float(), dim=-1)
    return float((error * valid.float()).sum().cpu() / valid.float().sum().clamp_min(1).cpu() * 1000.0)


def _update_accumulator(acc: dict[str, list[float]], key: str, value: float) -> None:
    acc.setdefault(key, []).append(float(value))


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = task_config_from_dict(checkpoint["config"])
    cfg.data.root = str(Path(args.root).resolve())
    cfg.data.train_path = str(Path(args.root).resolve())

    device = torch.device(args.device)
    model = CmFlowModel(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = Stage4CmSceneDataset(
        args.root,
        num_obj_points=int(cfg.meta.num_obj_points),
        num_hand_points=int(cfg.meta.num_hand_points),
        base_seed=int(cfg.train.seed),
        min_stride=1,
        max_stride=10,
        fixed_stride=int(args.stride),
        sampling_bank_size=int(cfg.data.sampling_bank_size),
        fixed_bank=int(args.bank),
        max_samples=int(args.samples),
        use_dense_cache=True,
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    stats: dict[str, list[float]] = {}
    processed = 0

    with torch.no_grad():
        for batch_cpu in loader:
            batch = _move_batch(batch_cpu, device)
            online_batch = {key: value for key, value in batch.items() if not key.startswith("cached_")}
            z_obj_1, z_hand_1, contact_1 = model.dense_encoder(
                obj_points=online_batch["obj_points"],
                obj_normals=online_batch["obj_normals"],
                hand_points=online_batch["hand_points"],
                hand_normals=online_batch["hand_normals"],
                obj_valid_mask=online_batch["obj_valid_mask"],
            )
            z_obj_2, z_hand_2, contact_2 = model.dense_encoder(
                obj_points=online_batch["obj_points"],
                obj_normals=online_batch["obj_normals"],
                hand_points=online_batch["hand_points"],
                hand_normals=online_batch["hand_normals"],
                obj_valid_mask=online_batch["obj_valid_mask"],
            )
            z_obj_cache = batch["cached_z_obj"]
            z_hand_cache = batch["cached_z_hand"]
            contact_cache = batch["cached_hand_contact"]

            pred_oo_1 = _head(model, batch, z_obj_1, z_hand_1, contact_1)
            pred_oo_2 = _head(model, batch, z_obj_2, z_hand_2, contact_2)
            pred_cache = _head(model, batch, z_obj_cache, z_hand_cache, contact_cache)
            valid = batch["obj_valid_mask"].bool()
            target = batch["obj_flow_gt"]

            for name, first, second in (
                ("feature_z_obj_oo_rms", z_obj_1, z_obj_2),
                ("feature_z_obj_oc_rms", z_obj_1, z_obj_cache),
                ("feature_z_hand_oo_rms", z_hand_1, z_hand_2),
                ("feature_z_hand_oc_rms", z_hand_1, z_hand_cache),
                ("feature_contact_oo_rms", contact_1, contact_2),
                ("feature_contact_oc_rms", contact_1, contact_cache),
                ("output_flow_oo_rms", pred_oo_1["pred_obj_flow"], pred_oo_2["pred_obj_flow"]),
                ("output_flow_oc_rms", pred_oo_1["pred_obj_flow"], pred_cache["pred_obj_flow"]),
            ):
                _update_accumulator(stats, name, _rms(first - second))
            for name, prediction in (
                ("epe_online_1_mm", pred_oo_1),
                ("epe_online_2_mm", pred_oo_2),
                ("epe_cached_mm", pred_cache),
            ):
                _update_accumulator(stats, name, _epe_mm(prediction["pred_obj_flow"], target, valid))
            processed += int(valid.shape[0])
            if processed >= int(args.samples):
                break

    result: dict[str, Any] = {
        "root": str(Path(args.root).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples": processed,
        "stride": int(args.stride),
        "bank": int(args.bank),
        "device": str(device),
        "metrics": {
            key: {"mean": float(np.mean(values)), "max": float(np.max(values))}
            for key, values in sorted(stats.items())
        },
    }
    result["ratios"] = {
        "feature_z_obj_oc_over_oo": result["metrics"]["feature_z_obj_oc_rms"]["mean"]
        / max(result["metrics"]["feature_z_obj_oo_rms"]["mean"], 1e-12),
        "feature_z_hand_oc_over_oo": result["metrics"]["feature_z_hand_oc_rms"]["mean"]
        / max(result["metrics"]["feature_z_hand_oo_rms"]["mean"], 1e-12),
        "output_flow_oc_over_oo": result["metrics"]["output_flow_oc_rms"]["mean"]
        / max(result["metrics"]["output_flow_oo_rms"]["mean"], 1e-12),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--bank", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
