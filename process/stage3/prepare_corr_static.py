from __future__ import annotations

import argparse
import json
import pickle
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE2_ROOT = ROOT / "processed_data" / "generated" / "stage2" / "grab_initonly_4096"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "processed_data" / "generated" / "stage3" / "grab_initonly_4096"
)
SCHEMA_NAME = "train_corr_static"
SCHEMA_VERSION = "1.0.0"


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def _load_stage2(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Stage 2 payload must be a dict: {path}")
    if str(payload.get("schema_name", "")) != "ref2dex_opti":
        raise ValueError(f"Unsupported Stage 2 schema in {path}: {payload.get('schema_name')!r}")
    return payload


def _require_array(
    payload: dict[str, Any],
    key: str,
    *,
    ndim: int | None = None,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing required Stage 2 field: {key}")
    array = np.asarray(payload[key], dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{key}: expected ndim={ndim}, got shape={array.shape}")
    return array


def _points_world_to_obj(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    rotation = poses[:, :3, :3]
    translation = poses[:, :3, 3]
    return np.einsum("tji,tpj->tpi", rotation, points - translation[:, None]).astype(np.float32)


def _normals_world_to_obj(normals: np.ndarray, poses: np.ndarray) -> np.ndarray:
    rotation = poses[:, :3, :3]
    result = np.einsum("tji,tpj->tpi", rotation, normals)
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    return (result / np.clip(norm, 1e-8, None)).astype(np.float32)


def _points_world_to_hand_root(points: np.ndarray, hand_root_poses: np.ndarray) -> np.ndarray:
    """把 world 坐标变换到 hand-root frame:

        x^{hand_root} = R_root^T @ (x^{world} - t_wrist)

    其中:
        hand_root_poses = T_world_from_hand_root, shape (T, 4, 4)
        R_root  = hand_root_poses[:, :3, :3]
        t_wrist = hand_root_poses[:, :3, 3]
    """
    rotation = hand_root_poses[:, :3, :3]
    translation = hand_root_poses[:, :3, 3]
    return np.einsum("tji,tpj->tpi", rotation, points - translation[:, None]).astype(np.float32)


def _normals_world_to_hand_root(normals: np.ndarray, hand_root_poses: np.ndarray) -> np.ndarray:
    """法向只做旋转：n^{hand_root} = R_root^T @ n^{world}."""
    rotation = hand_root_poses[:, :3, :3]
    result = np.einsum("tji,tpj->tpi", rotation, normals)
    norm = np.linalg.norm(result, axis=-1, keepdims=True)
    return (result / np.clip(norm, 1e-8, None)).astype(np.float32)


def _compute_candidate_knn(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    *,
    k_cross: int,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames, num_obj, _ = obj_points.shape
    num_hand = hand_points.shape[1]
    if k_cross <= 0 or k_cross > num_hand:
        raise ValueError(f"k_cross must be in [1, {num_hand}], got {k_cross}")

    min_dist = np.empty((num_frames, num_obj), dtype=np.float32)
    candidate_mask = np.empty((num_frames, num_obj), dtype=bool)
    knn_idx = np.full((num_frames, num_obj, k_cross), -1, dtype=np.int16)

    batch_size = max(1, int(frame_batch_size))
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        obj = torch.from_numpy(obj_points[start:end]).to(device=device, dtype=torch.float32)
        hand = torch.from_numpy(hand_points[start:end]).to(device=device, dtype=torch.float32)
        dist = torch.cdist(obj, hand)
        knn_dist, idx = torch.topk(dist, k=k_cross, dim=-1, largest=False, sorted=True)
        batch_min = knn_dist[..., 0]
        batch_candidate = batch_min <= float(candidate_threshold)
        idx = torch.where(batch_candidate.unsqueeze(-1), idx, torch.full_like(idx, -1))

        min_dist[start:end] = batch_min.cpu().numpy().astype(np.float32)
        candidate_mask[start:end] = batch_candidate.cpu().numpy()
        knn_idx[start:end] = idx.cpu().numpy().astype(np.int16)
        del dist, knn_dist, idx, obj, hand

    return min_dist, candidate_mask, knn_idx


def _validate_stage2_geometry(
    payload: dict[str, Any],
    *,
    num_obj_pool: int,
    num_hand_points: int,
) -> None:
    raw_frame_id = _require_array(payload, "raw_frame_id", ndim=1)
    num_frames = raw_frame_id.shape[0]
    expected = {
        "obj_points_world": (num_frames, num_obj_pool, 3),
        "obj_normals_world": (num_frames, num_obj_pool, 3),
        "obj_root_pose": (num_frames, 4, 4),
        "hand_points_world": (num_frames, num_hand_points, 3),
        "hand_normals_world": (num_frames, num_hand_points, 3),
        "obj_point_id": (num_obj_pool,),
        "hand_point_id": (num_hand_points,),
        "hand_cano_points": (num_hand_points, 3),
        "hand_finger_id": (num_hand_points,),
        "hand_region_id": (num_hand_points,),
    }
    for key, shape in expected.items():
        value = _require_array(payload, key)
        if value.shape != shape:
            raise ValueError(f"{key}: expected shape={shape}, got {value.shape}")
    for key in ("obj_points_world", "obj_normals_world", "hand_points_world", "hand_normals_world"):
        if not np.isfinite(np.asarray(payload[key])).all():
            raise ValueError(f"{key} contains non-finite values")


def build_stage3_sequence(
    payload: dict[str, Any],
    *,
    source_path: Path,
    stage2_root: Path,
    num_obj_pool: int,
    num_hand_points: int,
    k_cross: int,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
    mirror_left_to_right: bool,
    coordinate_frame: str = "object",
) -> dict[str, np.ndarray]:
    """Build one Stage 3 sample.

    ``coordinate_frame`` selects the canonical frame the model sees:
      - ``"object"`` (default): points are transformed by ``obj_root_pose``
        (object root at origin).
      - ``"hand_root"``: points are transformed by ``hand_root_pose``
        (MANO wrist at origin, orientation = MANO global_orient).
        Requires the Stage 2 payload to contain ``hand_root_pose``; raises
        ``KeyError`` otherwise.

    Distance / KNN fields (``obj_to_hand_min_dist``, ``obj_candidate_mask_5cm``,
    ``gt_obj_to_hand_knn_idx``) are frame-invariant and reused as-is.
    """
    if coordinate_frame not in {"object", "hand_root"}:
        raise ValueError(
            f"coordinate_frame must be 'object' or 'hand_root', got {coordinate_frame!r}"
        )
    _validate_stage2_geometry(
        payload,
        num_obj_pool=num_obj_pool,
        num_hand_points=num_hand_points,
    )
    poses = _require_array(payload, "obj_root_pose", ndim=3, dtype=np.float32)
    if coordinate_frame == "object":
        obj_points = _points_world_to_obj(
            _require_array(payload, "obj_points_world", ndim=3, dtype=np.float32),
            poses,
        )
        obj_normals = _normals_world_to_obj(
            _require_array(payload, "obj_normals_world", ndim=3, dtype=np.float32),
            poses,
        )
        hand_points = _points_world_to_obj(
            _require_array(payload, "hand_points_world", ndim=3, dtype=np.float32),
            poses,
        )
        hand_normals = _normals_world_to_obj(
            _require_array(payload, "hand_normals_world", ndim=3, dtype=np.float32),
            poses,
        )
        frame_pose_field = "T_world_from_obj"
        frame_pose_value = poses
    else:  # "hand_root"
        if "hand_root_pose" not in payload:
            raise KeyError(
                "Stage 2 payload missing 'hand_root_pose'; re-run Stage 2 with a "
                "preprocessor that emits hand_root_pose (e.g. updated process/ARCTIC/raw.py "
                "or process/GRAB/raw.py)."
            )
        hand_root_poses = _require_array(
            payload, "hand_root_pose", ndim=3, dtype=np.float32
        )
        obj_points = _points_world_to_hand_root(
            _require_array(payload, "obj_points_world", ndim=3, dtype=np.float32),
            hand_root_poses,
        )
        obj_normals = _normals_world_to_hand_root(
            _require_array(payload, "obj_normals_world", ndim=3, dtype=np.float32),
            hand_root_poses,
        )
        hand_points = _points_world_to_hand_root(
            _require_array(payload, "hand_points_world", ndim=3, dtype=np.float32),
            hand_root_poses,
        )
        hand_normals = _normals_world_to_hand_root(
            _require_array(payload, "hand_normals_world", ndim=3, dtype=np.float32),
            hand_root_poses,
        )
        frame_pose_field = "T_world_from_hand_root"
        frame_pose_value = hand_root_poses
    hand_cano_points = _require_array(payload, "hand_cano_points", ndim=2, dtype=np.float32).copy()
    side = str(payload["side"])
    if mirror_left_to_right and side == "left":
        obj_points[..., 0] *= -1
        obj_normals[..., 0] *= -1
        hand_points[..., 0] *= -1
        hand_normals[..., 0] *= -1
        hand_cano_points[..., 0] *= -1

    min_dist, candidate_mask, clean_knn_idx = _compute_candidate_knn(
        obj_points,
        hand_points,
        k_cross=k_cross,
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    source_rel = source_path.resolve().relative_to(stage2_root.resolve()).as_posix()

    out: dict[str, np.ndarray] = {
        "schema_name": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION),
        "source_stage2_file": np.asarray(source_rel),
        "processing_mode": np.asarray(str(payload["processing_mode"])),
        "dataset_name": np.asarray(str(payload["dataset_name"])),
        "seq_id": np.asarray(str(payload["seq_id"])),
        "subject_id": np.asarray(str(payload["subject_id"])),
        "seq_name": np.asarray(str(payload["seq_name"])),
        "object_name": np.asarray(str(payload["object_name"])),
        "side": np.asarray(side),
        "raw_frame_id": _require_array(payload, "raw_frame_id", ndim=1, dtype=np.int32),
        "stage2_frame_idx": _require_array(payload, "stage2_frame_idx", ndim=1, dtype=np.int32),
        "obj_points": obj_points,
        "obj_normals": obj_normals,
        "obj_point_id": _require_array(payload, "obj_point_id", ndim=1, dtype=np.int32),
        "hand_points": hand_points,
        "hand_normals": hand_normals,
        "hand_point_id": _require_array(payload, "hand_point_id", ndim=1, dtype=np.int32),
        "hand_cano_points": hand_cano_points,
        "hand_finger_id": _require_array(payload, "hand_finger_id", ndim=1, dtype=np.int32),
        "hand_region_id": _require_array(payload, "hand_region_id", ndim=1, dtype=np.int32),
        "obj_to_hand_min_dist": min_dist,
        "obj_candidate_mask_5cm": candidate_mask,
        "gt_obj_to_hand_knn_idx": clean_knn_idx,
        frame_pose_field: frame_pose_value,
        "coordinate_frame": np.asarray(coordinate_frame),
    }
    return out


def _resolve_files(stage2_root: Path, seq_id: str | None, side: str | None) -> list[Path]:
    files = sorted(stage2_root.glob("*/*.pkl"))
    if side is not None:
        files = [path for path in files if path.stem.endswith(f"_{side}")]
    if seq_id is None:
        return files
    target = seq_id.strip("/")
    selected = []
    for path in files:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if str(payload.get("seq_id", "")) == target:
            selected.append(path)
    return selected


def _write_meta(
    output_root: Path,
    *,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> None:
    meta = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_stage2_root": str(Path(args.stage2_root).resolve()),
        "output_root": str(output_root.resolve()),
        "sample_unit": "single_sequence_single_hand",
        "coordinate_frame": str(args.coordinate_frame),
        "num_obj_pool": int(args.num_obj_pool),
        "num_obj_train": int(args.num_obj_train),
        "num_hand_points": int(args.num_hand_points),
        "k_cross": int(args.k_cross),
        "candidate_threshold": float(args.candidate_threshold),
        "padding_policy": "pad_invalid_without_replacement",
        "epoch_sampling": True,
        "mirror_left_to_right": bool(args.mirror_left_to_right),
        "knn_storage": {
            "field": "gt_obj_to_hand_knn_idx",
            "dtype": "int16",
            "non_candidate_value": -1,
            "distance_saved": False,
        },
        "stats": stats,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common Stage 2 payloads -> Stage 3 point pools")
    parser.add_argument("--stage2-root", default=str(DEFAULT_STAGE2_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seq-id", default=None)
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--num-obj-pool", type=int, default=4096)
    parser.add_argument("--num-obj-train", type=int, default=512)
    parser.add_argument("--num-hand-points", type=int, default=1538)
    parser.add_argument("--k-cross", type=int, default=32)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mirror-left-to-right", action="store_true")
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--coordinate-frame",
        choices=["object", "hand_root"],
        default="object",
        help=(
            "Frame the Stage 3 points are expressed in. "
            "'object' (default): transformed by obj_root_pose. "
            "'hand_root': transformed by hand_root_pose (MANO wrist at origin, "
            "orientation = MANO global_orient); requires Stage 2 payload to "
            "contain hand_root_pose."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_obj_pool != 4096 or args.num_obj_train != 512 or args.num_hand_points != 1538:
        raise SystemExit("Current schema requires num_obj_pool=4096, num_obj_train=512, num_hand_points=1538")
    stage2_root = Path(args.stage2_root).resolve()
    output_root = Path(args.output_root).resolve()
    files = _resolve_files(stage2_root, args.seq_id, args.side)
    if not files:
        raise FileNotFoundError(f"No Stage 2 pkl files found under {stage2_root}")
    device = _resolve_device(args.device)
    stats = {"source_files": len(files), "written": 0, "skipped": 0, "failed": 0, "frames": 0}

    for source_path in files:
        relative = source_path.relative_to(stage2_root).with_suffix(".npz")
        output_path = output_root / relative
        if output_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue
        try:
            payload = _load_stage2(source_path)
            stage3 = build_stage3_sequence(
                payload,
                source_path=source_path,
                stage2_root=stage2_root,
                num_obj_pool=args.num_obj_pool,
                num_hand_points=args.num_hand_points,
                k_cross=args.k_cross,
                candidate_threshold=args.candidate_threshold,
                frame_batch_size=args.frame_batch_size,
                device=device,
                mirror_left_to_right=args.mirror_left_to_right,
                coordinate_frame=args.coordinate_frame,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_compressed:
                np.savez_compressed(output_path, **stage3)
            else:
                np.savez(output_path, **stage3)
            num_frames = int(stage3["raw_frame_id"].shape[0])
            stats["written"] += 1
            stats["frames"] += num_frames
            candidate_counts = stage3["obj_candidate_mask_5cm"].sum(axis=1)
            print(
                f"[stage3] wrote {output_path} frames={num_frames} "
                f"candidate[min/median/max]="
                f"{int(candidate_counts.min())}/{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
            )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage3] failed {source_path}: {exc}")
            traceback.print_exc()

    _write_meta(output_root, args=args, stats=stats)
    print(f"[stage3] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
