"""Build a derived multi-frame horizon cache from the v4 geometry layer.

The first use is a 3 Hz horizon (stride=10 at the recorded 30 Hz pose rate).
No mesh/FK work is repeated: hand/object world geometry and q_full are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.task.CmDecoder.dataset import _rotate_to_frame, _to_frame


def build(args: argparse.Namespace) -> None:
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    out_manifest = {**source_manifest, "schema": "cmdecoder_layered_v4", "horizon_stride": int(args.stride),
                    "horizon_hz": float(args.source_hz) / float(args.stride), "cache_dirs": {}}
    for episode, source_rel in source_manifest["cache_dirs"].items():
        source_episode = args.source_root / source_rel
        source_geometry = source_episode / "geometry"
        out_rel = f"v1/episodes/{source_rel.rsplit('/', 1)[-1]}"
        out_manifest["cache_dirs"][episode] = out_rel
        out_episode = output_root / out_rel
        task_dir = out_episode / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        q = np.load(source_geometry / "q_full.npy", mmap_mode="r", allow_pickle=False)[:, 6:]
        times = np.load(source_geometry / "frame_time.npy", mmap_mode="r", allow_pickle=False)
        source_ids = np.load(source_geometry / "source_frame_id.npy", mmap_mode="r", allow_pickle=False)
        hand = np.load(source_geometry / "hand_points_world.npy", mmap_mode="r", allow_pickle=False)
        hand_normals = np.load(source_geometry / "hand_normals_world.npy", mmap_mode="r", allow_pickle=False)
        obj = np.load(source_geometry / "obj_points_world.npy", mmap_mode="r", allow_pickle=False)
        obj_normals = np.load(source_geometry / "obj_normals_world.npy", mmap_mode="r", allow_pickle=False)
        wrist = np.load(source_geometry / "wrist_pose_world.npy", mmap_mode="r", allow_pickle=False)
        idx = np.arange(0, len(q) - int(args.stride), dtype=np.int64)
        jdx = idx + int(args.stride)
        keep = (source_ids[jdx] - source_ids[idx]) == int(args.stride)
        idx, jdx = idx[keep], jdx[keep]
        hand_t = np.asarray([_to_frame(hand[i], wrist[i]) for i in idx], dtype=np.float32)
        hand_n = np.asarray([_rotate_to_frame(hand_normals[i], wrist[i]) for i in idx], dtype=np.float32)
        hand_next = np.asarray([_to_frame(hand[i], wrist[i]) for i in jdx], dtype=np.float32)
        obj_t = np.asarray([_to_frame(obj[i], wrist[i]) for i in idx], dtype=np.float32)
        obj_n = np.asarray([_rotate_to_frame(obj_normals[i], wrist[i]) for i in idx], dtype=np.float32)
        arrays = {
            "hand_points": hand_t,
            "hand_normals": hand_n,
            "hand_flow": hand_next - hand_t,
            "obj_points": obj_t,
            "obj_normals": obj_n,
            "obj_valid_mask": np.ones((len(idx), obj_t.shape[1]), dtype=np.bool_),
            "q_t": np.asarray(q[idx], dtype=np.float32),
            "q_next": np.asarray(q[jdx], dtype=np.float32),
            "delta_time_s": np.asarray(times[jdx] - times[idx], dtype=np.float32),
            "source_frame_delta": np.asarray(source_ids[jdx] - source_ids[idx], dtype=np.int64),
            "is_30hz_pair": np.asarray((source_ids[jdx] - source_ids[idx]) == 1, dtype=np.bool_),
            "q_delta_abs_max": np.max(np.abs(q[jdx] - q[idx]), axis=1).astype(np.float32),
        }
        for name, value in arrays.items():
            np.save(task_dir / f"{name}.npy", value, allow_pickle=False)
        task_manifest = {"schema": "cmdecoder_layered_v4", "layer": "cmdecoder_task_horizon",
                         "episode": episode, "horizon_stride": int(args.stride), "horizon_hz": float(args.source_hz) / float(args.stride),
                         "samples": int(len(idx)), "fields": sorted(arrays)}
        (task_dir / "manifest.json").write_text(json.dumps(task_manifest, indent=2), encoding="utf-8")
        print(f"built {episode}: {len(idx)} pairs", flush=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz"))
    parser.add_argument("--output-manifest", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v1/selection_20_seed42.json"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--source-hz", type=float, default=30.0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
