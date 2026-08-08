"""Build the isolated InteractionDynamics V1 sequence cache from raw GRAB."""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from process.GRAB.raw import (
    DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABRawAdapter,
    load_manifest_seq_paths, resolve_grab_sequence_root,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data/processed_data/interaction_dynamics_v1/data/grab"
SHARED_SCHEMA_NAME = "ref2dex_interaction_dynamics_shared"
HAND_SCHEMA_NAME = "ref2dex_interaction_dynamics_hand"
SCHEMA_VERSION = "1.0.0"
SOURCE_FPS = 120.0


def _sequences(args: argparse.Namespace) -> list[Path]:
    if args.raw_file:
        return [Path(args.raw_file).resolve()]
    if args.manifest:
        return [Path(path) for path in load_manifest_seq_paths(args.manifest, args.grab_root)]
    root = resolve_grab_sequence_root(args.grab_root)
    paths = sorted(root.glob("*/*.npz"))
    if not args.seq:
        return paths
    target = args.seq[:-4] if args.seq.endswith(".npz") else args.seq
    return [p for p in paths if target in p.relative_to(root).with_suffix("").as_posix()]


def _candidate_mask(obj: np.ndarray, hand: np.ndarray, threshold: float,
                    batch_size: int, device: torch.device) -> np.ndarray:
    result = np.empty(obj.shape[:2], dtype=bool)
    for start in range(0, len(obj), batch_size):
        end = min(start + batch_size, len(obj))
        o = torch.from_numpy(obj[start:end]).to(device)
        h = torch.from_numpy(hand[start:end]).to(device)
        result[start:end] = (torch.cdist(o, h).amin(-1) <= threshold).cpu().numpy()
    return result


def build_shared(source: dict[str, Any], source_path: Path, grab_root: Path,
                 ds_rate: int) -> dict[str, np.ndarray]:
    try:
        source_raw_file = source_path.resolve().relative_to(grab_root.resolve()).as_posix()
    except ValueError:
        source_raw_file = str(source_path.resolve())
    return {
        "schema_name": np.asarray(SHARED_SCHEMA_NAME), "schema_version": np.asarray(SCHEMA_VERSION),
        "source_raw_file": np.asarray(source_raw_file), "dataset_name": np.asarray("grab"),
        "seq_id": np.asarray(source["seq_id"]), "subject_id": np.asarray(source["subject_id"]),
        "seq_name": np.asarray(source["seq_name"]), "object_name": np.asarray(source["object_name"]),
        "raw_frame_id": np.asarray(source["raw_frame_id"], np.int32),
        "ds_rate": np.asarray(ds_rate, np.int32), "source_fps": np.asarray(SOURCE_FPS, np.float32),
        "coordinate_frame": np.asarray("world"),
        "obj_points_world": np.asarray(source["obj_points_world"], np.float32),
        "obj_normals_world": np.asarray(source["obj_normals_world"], np.float32),
        "obj_point_id": np.asarray(source["obj_point_id"], np.int32),
        "obj_root_pose_world": np.asarray(source["obj_root_pose"], np.float32),
    }


def build_hand(source: dict[str, Any], side: str, threshold: float,
               batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    root_key = f"{side}_hand_root_pose"
    if root_key not in source:
        raise ValueError(f"{source['seq_id']} has no reconstructed {side} hand")
    obj = np.asarray(source["obj_points_world"], np.float32)
    hand = np.asarray(source[f"{side}_hand_points_world"], np.float32)
    return {
        "schema_name": np.asarray(HAND_SCHEMA_NAME), "schema_version": np.asarray(SCHEMA_VERSION),
        "side": np.asarray(side),
        "hand_root_pose_world": np.asarray(source[root_key], np.float32),
        "hand_points_world": hand,
        "hand_normals_world": np.asarray(source[f"{side}_hand_normals_world"], np.float32),
        "hand_point_id": np.asarray(source[f"{side}_hand_point_id"], np.int32),
        "hand_cano_points": np.asarray(source[f"{side}_hand_cano_points"], np.float32),
        "hand_finger_id": np.asarray(source[f"{side}_hand_finger_id"], np.int32),
        "hand_region_id": np.asarray(source[f"{side}_hand_region_id"], np.int32),
        "obj_candidate_mask_5cm": _candidate_mask(obj, hand, threshold, batch_size, device),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    p.add_argument("--mano-path", default=DEFAULT_MANO_MODEL_DIR)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--raw-file"); p.add_argument("--seq"); p.add_argument("--manifest")
    p.add_argument("--side", choices=["left", "right", "both"], default="both")
    p.add_argument("--ds-rate", type=int, default=4); p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--frame-start", type=int, default=0); p.add_argument("--frame-batch-size", type=int, default=4)
    p.add_argument("--device", default="auto"); p.add_argument("--nn-batch-size", type=int, default=8)
    p.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    p.add_argument("--candidate-threshold", type=float, default=0.05); p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if sum(value is not None for value in (args.raw_file, args.seq, args.manifest)) > 1:
        raise SystemExit("Use only one of --raw-file, --seq, and --manifest")
    if args.ds_rate != 4:
        raise SystemExit("InteractionDynamics V1 requires --ds-rate=4")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    paths = _sequences(args)
    if not paths:
        raise FileNotFoundError("No matching GRAB sequences")
    adapter = GRABRawAdapter(num_obj_points=4096, device=str(device), ds_rate=4,
        max_frames=args.max_frames or None, frame_start=args.frame_start, grab_root=args.grab_root,
        mano_path=args.mano_path, obj_unit=args.obj_unit, nn_batch_size=args.nn_batch_size)
    stats = {"sequences": len(paths), "shared_written": 0, "hands_written": 0, "failed": 0}
    for raw_path in paths:
        try:
            source = adapter.process_sequence(str(raw_path))
            out = args.output_root / source["subject_id"] / source["seq_name"]
            out.mkdir(parents=True, exist_ok=True)
            shared_path = out / "shared.npz"
            if args.overwrite or not shared_path.exists():
                np.savez(shared_path, **build_shared(source, raw_path, Path(args.grab_root), 4))
                stats["shared_written"] += 1
            for side in (("left", "right") if args.side == "both" else (args.side,)):
                path = out / f"{side}.npz"
                if path.exists() and not args.overwrite:
                    continue
                try:
                    payload = build_hand(source, side, args.candidate_threshold, args.frame_batch_size, device)
                except ValueError as exc:
                    print(f"[interaction-cache] skip {side}: {exc}")
                    continue
                np.savez(path, **payload)
                stats["hands_written"] += 1
                counts = payload["obj_candidate_mask_5cm"].sum(1)
                print(f"[interaction-cache] wrote {path} frames={len(counts)} candidates={int(counts.min())}/{int(np.median(counts))}/{int(counts.max())}")
        except Exception:
            stats["failed"] += 1
            traceback.print_exc()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "meta.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"[interaction-cache] {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
