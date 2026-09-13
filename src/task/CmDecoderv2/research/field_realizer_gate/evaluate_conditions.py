"""Offline B1/C0/Cs sensitivity diagnostic on one explicit checkpoint."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.base.base_config import task_config_from_dict
from ...field_dataset import FieldRealizerDataset
from ...field_realizer import FieldRealizerModel, spatial_shuffle_f7, zero_f7


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run_log = (args.output / "train.log").open("w", buffering=1)
    os.dup2(run_log.fileno(), 1)
    os.dup2(run_log.fileno(), 2)
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "task": "CmDecoderv2",
        "modification_version": "V1.1.16", "operation_category": ["diagnostic", "operation"],
        "run_id": args.output.name, "run_status": "RUNNING", "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "command": sys.argv, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest(args.checkpoint), "script_sha256": digest(Path(__file__)),
        "config_snapshot": "config.json", "metadata_snapshot": "metadata.json",
        "scope": "intermediate-checkpoint offline sensitivity; not contact50 or final Gate",
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = task_config_from_dict(checkpoint["config"])
        (args.output / "config.json").write_text(json.dumps(checkpoint["config"], indent=2) + "\n")
        index_path = Path(cfg.data.index_path)
        index = json.loads(index_path.read_text())
        dataset = FieldRealizerDataset(
            index["sequences"]["val"], field_root=cfg.data.field_root,
            urdf_path=cfg.data.urdf_path, window_size=cfg.meta.window_size,
            active_only=cfg.data.active_only,
        )
        metadata = {
            "index_path": str(index_path), "index_sha256": digest(index_path),
            "field_root": cfg.data.field_root, "windows": len(dataset),
            "checkpoint_epoch": checkpoint["epoch"], "checkpoint_step": checkpoint["step"],
            "split": "val", "conditions": ["B1", "C0", "Cs"],
            "shuffle": "seed42; independent permutation per window, shared across K; anchors fixed",
            "aggregation": "micro over active sample-horizons; h1 additionally reported",
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        model_cfg = copy.copy(cfg.model)
        model_cfg.meta = cfg.meta
        model = FieldRealizerModel(model_cfg).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)
        generator = torch.Generator().manual_seed(seed)
        totals = {c: {"epe": 0., "epe_h1": 0., "q": 0., "wrist": 0., "q_change": 0., "wrist_change": 0.}
                  for c in ("B1", "C0", "Cs")}
        count = count_h1 = 0
        with torch.inference_mode(), (args.output / "metrics.jsonl").open("w") as log:
            for batch_id, cpu in enumerate(loader):
                batch = {k: v.cuda(non_blocking=True) for k, v in cpu.items()}
                original = batch["f7"]
                permutation = torch.stack([torch.randperm(128, generator=generator)
                                           for _ in range(len(original))]).cuda()
                valid = batch["active_mask"]
                count += int(valid.sum())
                count_h1 += int(valid[:, 0].sum())
                baseline = None
                for condition in totals:
                    batch["f7"] = (original if condition == "B1" else zero_f7(original)
                                   if condition == "C0" else spatial_shuffle_f7(original, permutation))
                    out = model(batch)
                    if baseline is None:
                        baseline = out
                    epe = (out["pred_hand_points_object"] - batch["target_hand_points_object"]).norm(dim=-1).mean(-1) * 1000
                    values = {
                        "epe": epe,
                        "q": (out["pred_q_delta"] - batch["target_q_delta"]).abs().mean(-1),
                        "wrist": (out["pred_wrist_translation"] - batch["target_wrist_translation"]).norm(dim=-1) * 1000,
                        "q_change": (out["pred_q_delta"] - baseline["pred_q_delta"]).abs().mean(-1),
                        "wrist_change": (out["pred_wrist_translation"] - baseline["pred_wrist_translation"]).norm(dim=-1) * 1000,
                    }
                    if not all(torch.isfinite(x).all() for x in values.values()):
                        raise ValueError("Non-finite evaluation output")
                    for key, value in values.items():
                        totals[condition][key] += float(value[valid].sum())
                    totals[condition]["epe_h1"] += float(epe[:, 0][valid[:, 0]].sum())
                    log.write(json.dumps({"batch": batch_id, "condition": condition,
                                          "active_horizons": int(valid.sum()),
                                          "point_epe_mm": float(epe[valid].mean())}) + "\n")
                if batch_id % 100 == 0:
                    print(f"batch={batch_id}/{len(loader)}", flush=True)
        results = {c: {k: v / (count_h1 if k == "epe_h1" else count)
                       for k, v in values.items()} for c, values in totals.items()}
        (args.output / "evaluation.json").write_text(json.dumps({
            "conditions": results, "active_horizons": count, "active_h1": count_h1,
            "conclusion": "INCONCLUSIVE", "scope": manifest["scope"],
        }, indent=2) + "\n")
        print(json.dumps(results, indent=2), flush=True)
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
