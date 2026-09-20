"""Encode fixed demonstration Cm windows; do not export a target-action bank."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.base import load_config
from src.task.CmDecoderv2.dataset import CmDecoderV2Dataset, _collate_cm_decoder
from src.task.CmDecoderv2.kinematics import QUERY_LINKS
from src.task.CmDecoderv2.rl.online_base import load_frozen_decoder, sha256

ROOT = Path(__file__).resolve().parents[5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default="src/task/CmDecoderv2/configs/active/mano_actual_finetune_v1_batch8.yaml")
    parser.add_argument("--checkpoint", default="outputs/cmdecoderv2/cm_decoder_v2_mano_actual_finetune_v1_batch8_20260913_092251/checkpoints/best.pt")
    parser.add_argument("--checkpoint-sha256", default="0814bdabcbdf484d90c6855a50e3bfebc1053b4d085ebde152b15f65aa8c4494")
    parser.add_argument("--dexplore-assets", type=Path, default=ROOT.parent / "dexplore/dexplore/data/assets/mjcf")
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = load_config(args.config)
    index_path = ROOT / cfg.data.index_path
    index = json.loads(index_path.read_text())
    entry = next(e for e in index["sequences"]["train"] if e["id"] == "s1/airplane_lift")
    dataset = CmDecoderV2Dataset(
        [entry], urdf_path=ROOT / cfg.data.urdf_path, window_size=4,
        num_obj_points=cfg.meta.num_obj_points, num_hand_points=cfg.meta.num_hand_points,
        hand_stream_mode=cfg.meta.hand_stream_mode, knn_k=cfg.meta.knn_k,
        hand_supervision_radius_m=cfg.meta.hand_supervision_radius_m,
        seed=42, perturb=False, active_only=False,
    )
    model = load_frozen_decoder(args.config, args.checkpoint, args.checkpoint_sha256, args.device)
    if not 0 <= args.start_frame < len(dataset):
        raise ValueError(f"start-frame must be in [0, {len(dataset) - 1}], got {args.start_frame}")
    arrays = {"cm_tokens": [], "anchor_pos": [], "anchor_normal": [], "sample_valid": []}
    with torch.inference_mode():
        for row in range(args.start_frame, len(dataset)):
            batch = _collate_cm_decoder([dataset[row]])
            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            encoded = model._encode_cm_window(batch)
            for key, value in zip(arrays, encoded):
                arrays[key].append(value[0].cpu().numpy())
            if row % 100 == 0:
                print(f"Encoded source window {row}/{len(dataset)}", flush=True)
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    seq = dataset.sequences[0]
    geometry_manifest = json.loads((Path(entry["geometry_root"]) / "manifest.json").read_text())
    tensor_path = Path(geometry_manifest["rl_q"]["tensor"])
    raw = torch.load(tensor_path, map_location="cpu", weights_only=False)
    initial_q = np.asarray(seq.q_native[args.start_frame]).copy()
    initial_links = dataset.kinematics.link_transforms_native(initial_q)
    arrays.update(
        initial_native_q=initial_q,
        initial_object_pose=np.asarray(seq.object_pose[args.start_frame]).copy(),
        initial_table_pose=raw[0, 238:245].numpy().astype(np.float32),
        initial_link_poses=np.stack([initial_links[k] for k in QUERY_LINKS]).astype(np.float32),
        zero_base_inverse=dataset.kinematics._zero_hand_base_inverse.astype(np.float32),
    )
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Non-finite reference artifact")
    np.savez(args.output / "source.npz", **arrays)
    asset_link = ROOT / "src/task/CmDecoderv2/assets/dexplore_objects"
    if not asset_link.exists():
        asset_link.symlink_to(args.dexplore_assets.resolve(), target_is_directory=True)
    if asset_link.resolve() != args.dexplore_assets.resolve():
        raise ValueError("Existing object asset link has a different target")
    assets = {name: sha256(asset_link / name) for name in ["airplane.urdf", "table.urdf", "objects/airplane/airplane.obj", "objects/table/table.obj"]}
    manifest = {
        "schema_name": "ref2dex_cm_online_source_v1", "work_version": "V1.1.15",
        "sequence_id": entry["id"], "split": "train", "seed": 42,
        "checkpoint_sha256": args.checkpoint_sha256, "checkpoint": args.checkpoint,
        "oicm_checkpoint_sha256": model.oicm_checkpoint_sha256,
        "data_file": "source.npz", "data_sha256": sha256(args.output / "source.npz"),
        "source_index": str(index_path), "source_index_sha256": sha256(index_path),
        "source_manifest": entry["paired_source_manifest"],
        "source_type": "paired_view_mano_parent_surface_at_actual_reference_object_pose",
        "initial_state_tensor": str(tensor_path), "initial_state_frame": args.start_frame,
        "window_start_frame": args.start_frame, "window_count": len(dataset) - args.start_frame,
        "fps": 30, "online_closed_loop": True,
        "target_state_usage": "selected frame reset only; no future Inspire target bank",
        "sample_valid_ratio": float(arrays["sample_valid"].mean()),
        "asset_root": str(asset_link.relative_to(ROOT)), "asset_sha256": assets,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run = {
        "manifest_schema": "ref2dex.run.v1", "task": "CmDecoderv2", "mode": "prepare_online_source",
        "run_id": args.output.name, "work_version": "V1.1.15", "operation_category": ["data", "operation"],
        "created_at": datetime.now(timezone.utc).isoformat(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "config_snapshot": "config.json", "metadata_snapshot": "manifest.json", "seed": 42,
        "initial_checkpoint": args.checkpoint, "output_dir": str(args.output.resolve()),
    }
    (args.output / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n")
    (args.output / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
