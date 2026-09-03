"""Export HOCap annotation poses and meshes to Ref2Dex Stage 3.

HOCap stores world-space object poses and MANO PCA45 poses.  The converter
keeps the raw HOCap data untouched and writes one Ref2Dex sample per
sequence/object/hand.  RGB-D and image labels are intentionally optional:
the correspondence input is reconstructed from the clean object mesh and the
MANO pose, while contact distances are derived from geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# manopth/chumpy still references removed NumPy aliases on NumPy >= 1.24.
for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)

import torch
import trimesh
import yaml
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/HOCap/data")
DEFAULT_MANO_ROOT = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
DEFAULT_MANOPTH_ROOT = Path("/home/wbcd/workspace/GeneOH-Diffusion/manopth")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hocap_subject1_annotation_v1"
)
SCHEMA_NAME = "train_corr_static_v2"
SCHEMA_VERSION = "2.1.0"
NUM_OBJ_POOL = 4096
NUM_HAND_POINTS = 1538


def _configure_manopth(root: Path) -> None:
    root = root.expanduser().resolve()
    if not (root / "manopth" / "manolayer.py").exists():
        raise FileNotFoundError(
            f"manopth checkout not found under {root}; set --manopth-root or HOCAP_MANOPTH_ROOT"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_mano_layer(side: str, mano_root: Path, manopth_root: Path):
    _configure_manopth(manopth_root)
    from manopth.manolayer import ManoLayer  # type: ignore

    return ManoLayer(
        side=side,
        mano_root=str(mano_root),
        flat_hand_mean=False,
        ncomps=45,
        use_pca=True,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(payload).__name__}")
    return payload


def _seed(text: str) -> int:
    # Stable across Python processes and machines; do not use hash(text).
    import hashlib

    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if hasattr(g, "vertices") and hasattr(g, "faces")]
        if not meshes:
            raise ValueError(f"Mesh scene has no geometry: {path}")
        mesh = trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {path}: {type(mesh)!r}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid mesh shape for {path}: vertices={vertices.shape}, faces={faces.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(faces).all():
        raise ValueError(f"Non-finite mesh data: {path}")
    return vertices, faces


def _sample_surface(path: Path, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(*_load_mesh(path), process=False)
    try:
        points, face_idx = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    except TypeError:  # trimesh versions without the seed keyword
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        finally:
            np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return np.asarray(points, dtype=np.float32), normals.astype(np.float32)


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("HOCap object pose contains a zero-length quaternion")
    x, y, z, w = (q / norm).T
    matrix = np.zeros(q.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _object_pose_matrices(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Object poses must be (T,7), got {poses.shape}")
    out = np.broadcast_to(np.eye(4, dtype=np.float32), (len(poses), 4, 4)).copy()
    out[:, :3, :3] = _quat_xyzw_to_matrix(poses[:, :4])
    out[:, :3, 3] = poses[:, 4:7]
    return out


def _transform_points(points: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum("tij,nj->tni", poses[:, :3, :3], points, optimize=True) + poses[:, None, :3, 3]


def _transform_normals(normals: np.ndarray, poses: np.ndarray) -> np.ndarray:
    result = np.einsum("tij,nj->tni", poses[:, :3, :3], normals, optimize=True)
    return result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    values = np.asarray(axis_angle, dtype=np.float32)
    theta = np.linalg.norm(values, axis=-1, keepdims=True)
    axis = values / np.clip(theta, 1e-8, None)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = np.cos(theta[..., 0])
    s = np.sin(theta[..., 0])
    one = 1.0 - c
    out = np.empty(values.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = c + x * x * one
    out[..., 0, 1] = x * y * one - z * s
    out[..., 0, 2] = x * z * one + y * s
    out[..., 1, 0] = y * x * one + z * s
    out[..., 1, 1] = c + y * y * one
    out[..., 1, 2] = y * z * one - x * s
    out[..., 2, 0] = z * x * one - y * s
    out[..., 2, 1] = z * y * one + x * s
    out[..., 2, 2] = c + z * z * one
    return out


def _mano_forward(
    pose: np.ndarray,
    *,
    side: str,
    betas: np.ndarray,
    mano_root: Path,
    manopth_root: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pose.ndim != 2 or pose.shape[1] < 51:
        raise ValueError(f"HOCap MANO pose must be (T,51), got {pose.shape}")
    valid = ~np.all(pose[:, :51] == -1.0, axis=1)
    if not valid.all():
        raise ValueError(f"Requested HOCap hand {side} has invalid frames: {np.where(~valid)[0][:8].tolist()}")
    layer = _load_mano_layer(side, mano_root, manopth_root)
    torch_device = torch.device(device)
    layer = layer.to(torch_device)
    p = torch.from_numpy(pose[:, :48].astype(np.float32)).to(torch_device)
    b = torch.from_numpy(np.asarray(betas, dtype=np.float32)).view(1, 10).expand(len(pose), -1).to(torch_device)
    t = torch.from_numpy(pose[:, 48:51].astype(np.float32)).to(torch_device)
    with torch.no_grad():
        vertices_mm, joints_mm = layer(p, b, t)
    vertices = vertices_mm.detach().cpu().numpy().astype(np.float32) / 1000.0
    joints = joints_mm.detach().cpu().numpy().astype(np.float32) / 1000.0
    faces = layer.th_faces.detach().cpu().numpy().astype(np.int64)
    if faces.shape != (NUM_HAND_POINTS, 3):
        raise ValueError(f"Expected standard MANO faces {(NUM_HAND_POINTS, 3)}, got {faces.shape}")
    face_vertices = vertices[:, faces]
    hand_points = face_vertices.mean(axis=2).astype(np.float32)
    hand_normals = np.cross(face_vertices[:, :, 1] - face_vertices[:, :, 0], face_vertices[:, :, 2] - face_vertices[:, :, 0])
    hand_normals /= np.clip(np.linalg.norm(hand_normals, axis=-1, keepdims=True), 1e-8, None)
    root_pose = np.broadcast_to(np.eye(4, dtype=np.float32), (len(pose), 4, 4)).copy()
    root_pose[:, :3, :3] = _axis_angle_to_matrix(pose[:, :3])
    root_pose[:, :3, 3] = joints[:, 0]
    return hand_points, hand_normals.astype(np.float32), root_pose, vertices


def _hand_to_obj_min_dist(obj_points: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    result = np.empty((len(obj_points), hand_points.shape[1]), dtype=np.float32)
    for index, (obj, hand) in enumerate(zip(obj_points, hand_points)):
        result[index] = cKDTree(obj).query(hand, k=1, workers=1)[0].astype(np.float32)
    return result


def _world_to_root(points: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    return np.einsum("tji,tnj->tni", root_pose[:, :3, :3], points - root_pose[:, None, :3, 3], optimize=True).astype(np.float32)


def _world_normals_to_root(normals: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    result = np.einsum("tji,tnj->tni", root_pose[:, :3, :3], normals, optimize=True)
    return (result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def _load_betas(path: Path) -> np.ndarray:
    payload = _read_yaml(path)
    values = payload.get("betas", payload.get("beta"))
    if values is None:
        raise ValueError(f"No betas in {path}")
    betas = np.asarray(values, dtype=np.float32).reshape(-1)
    if betas.size != 10:
        raise ValueError(f"MANO betas must have 10 values, got {betas.shape}")
    return betas


def _sequence_paths(data_root: Path, sequence: str) -> tuple[Path, dict[str, Any]]:
    sequence_dir = (data_root / sequence).resolve()
    data_root = data_root.resolve()
    if data_root not in sequence_dir.parents:
        raise ValueError(f"Sequence escapes data root: {sequence}")
    meta_path = sequence_dir / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return sequence_dir, _read_yaml(meta_path)


def _write_sample(
    output_path: Path,
    *,
    seq_id: str,
    subject_id: str,
    sequence: str,
    object_id: str,
    side: str,
    raw_frame_id: np.ndarray,
    obj_points_world: np.ndarray,
    obj_normals_world: np.ndarray,
    obj_points_canonical: np.ndarray,
    obj_normals_canonical: np.ndarray,
    obj_root_pose: np.ndarray,
    hand_points_world: np.ndarray,
    hand_normals_world: np.ndarray,
    hand_root_pose: np.ndarray,
    mano_pose_raw: np.ndarray,
    mano_betas: np.ndarray,
    mano_v_template: np.ndarray,
) -> None:
    obj_points = _world_to_root(obj_points_world, hand_root_pose)
    obj_normals = _world_normals_to_root(obj_normals_world, hand_root_pose)
    hand_points = _world_to_root(hand_points_world, hand_root_pose)
    hand_normals = _world_normals_to_root(hand_normals_world, hand_root_pose)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_name=np.asarray(SCHEMA_NAME),
        schema_version=np.asarray(SCHEMA_VERSION),
        dataset_id=np.asarray("hocap"),
        dataset_name=np.asarray("HOCap"),
        subject_id=np.asarray(subject_id),
        sequence=np.asarray(sequence),
        object_id=np.asarray(object_id),
        seq_id=np.asarray(seq_id),
        side=np.asarray(side),
        raw_frame_id=raw_frame_id.astype(np.int32),
        obj_points=obj_points.astype(np.float32),
        obj_normals=obj_normals.astype(np.float32),
        hand_points=hand_points.astype(np.float32),
        hand_normals=hand_normals.astype(np.float32),
        hand_to_obj_min_dist=_hand_to_obj_min_dist(obj_points, hand_points),
        coordinate_frame=np.asarray("hand_root"),
        hand_root_pose=hand_root_pose.astype(np.float32),
        obj_repr=np.asarray("rigid_canonical"),
        obj_points_canonical=obj_points_canonical.astype(np.float32),
        obj_normals_canonical=obj_normals_canonical.astype(np.float32),
        obj_point_id=np.arange(len(obj_points_canonical), dtype=np.int32),
        obj_root_pose_world=obj_root_pose.astype(np.float32),
        mano_global_orient=mano_pose_raw[:, :3].astype(np.float32),
        mano_transl=mano_pose_raw[:, 48:51].astype(np.float32),
        mano_pose=mano_pose_raw[:, 3:48].astype(np.float32),
        mano_betas=np.broadcast_to(mano_betas, (len(raw_frame_id), 10)).astype(np.float32).copy(),
        mano_v_template=mano_v_template.astype(np.float32),
        mano_use_pca=np.asarray(True),
        mano_num_pca_comps=np.asarray(45, dtype=np.int64),
        mano_flat_hand_mean=np.asarray(False),
        mano_pose_repr=np.asarray("pca"),
    )


def _export_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    sequence_dir, meta = _sequence_paths(data_root, sequence)
    object_ids = [str(value) for value in (meta.get("object_ids") or [])]
    sides = [str(value).lower() for value in (meta.get("mano_sides") or [])]
    if args.object_id:
        object_ids = [value for value in object_ids if value == args.object_id]
    if args.side:
        sides = [value for value in sides if value == args.side]
    if not object_ids or not sides:
        raise ValueError(f"No requested object/hand in {sequence}: objects={object_ids}, sides={sides}")
    poses_m = np.load(sequence_dir / "poses_m.npy", allow_pickle=False).astype(np.float32)
    poses_o = np.load(sequence_dir / "poses_o.npy", allow_pickle=False).astype(np.float32)
    if poses_m.ndim != 3 or poses_m.shape[0] not in (1, 2) or poses_m.shape[-1] < 51:
        raise ValueError(f"Unsupported poses_m shape in {sequence}: {poses_m.shape}")
    if poses_o.ndim != 3 or poses_o.shape[0] != len(meta.get("object_ids") or []) or poses_o.shape[-1] != 7:
        raise ValueError(f"Unsupported poses_o shape in {sequence}: {poses_o.shape}")
    frame_count = min(int(meta.get("num_frames", poses_m.shape[1])), poses_m.shape[1], poses_o.shape[1])
    frame_indices = np.arange(0, frame_count, max(1, int(args.frame_stride)), dtype=np.int32)
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]
    if len(frame_indices) == 0:
        raise ValueError(f"No frames selected for {sequence}")
    subject_id = str(meta.get("subject_id", sequence_dir.parent.name))
    beta_path = data_root / "calibration" / "mano" / f"{subject_id}.yaml"
    betas = _load_betas(beta_path)
    mano_root = Path(args.mano_root).expanduser().resolve()
    manopth_root = Path(args.manopth_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    results: list[str] = []
    for side in sides:
        side_index = 0 if side == "right" else 1
        if side_index >= poses_m.shape[0]:
            raise ValueError(f"No pose slot for {side} in {sequence}: {poses_m.shape}")
        raw_pose = poses_m[side_index, frame_indices, :51]
        hand_points_world, hand_normals_world, hand_root_pose, _vertices = _mano_forward(
            raw_pose,
            side=side,
            betas=betas,
            mano_root=mano_root,
            manopth_root=manopth_root,
            device=args.device,
        )
        mano_layer = _load_mano_layer(side, mano_root, manopth_root)
        mano_v_template = mano_layer.th_v_template.detach().cpu().numpy().reshape(-1, 3).astype(np.float32)
        for object_id in object_ids:
            mesh_path = data_root / "models" / object_id / "cleaned_mesh_10000.obj"
            if not mesh_path.exists():
                mesh_path = data_root / "models" / object_id / "textured_mesh.obj"
            canonical_points, canonical_normals = _sample_surface(
                mesh_path, NUM_OBJ_POOL, _seed(f"hocap:{sequence}:{object_id}:{NUM_OBJ_POOL}"),
            )
            object_index = list(meta["object_ids"]).index(object_id)
            object_pose = _object_pose_matrices(poses_o[object_index, frame_indices])
            object_points_world = _transform_points(canonical_points, object_pose).astype(np.float32)
            object_normals_world = _transform_normals(canonical_normals, object_pose).astype(np.float32)
            seq_id = f"hocap:{sequence}/{object_id}"
            output_path = output_root / subject_id / sequence_dir.name / f"{object_id}_{side}.npz"
            if output_path.exists() and not args.overwrite:
                results.append(str(output_path))
                continue
            _write_sample(
                output_path,
                seq_id=seq_id,
                subject_id=subject_id,
                sequence=sequence,
                object_id=object_id,
                side=side,
                raw_frame_id=frame_indices,
                obj_points_world=object_points_world,
                obj_normals_world=object_normals_world,
                obj_points_canonical=canonical_points,
                obj_normals_canonical=canonical_normals,
                obj_root_pose=object_pose,
                hand_points_world=hand_points_world,
                hand_normals_world=hand_normals_world,
                hand_root_pose=hand_root_pose,
                mano_pose_raw=raw_pose,
                mano_betas=betas,
                mano_v_template=mano_v_template,
            )
            results.append(str(output_path))
            print(f"[hocap] wrote {output_path} frames={len(frame_indices)} object={object_id} side={side}", flush=True)
    return {
        "sequence": sequence,
        "files": results,
        # Each object/hand output contains the complete selected timeline;
        # report frame samples (not just source timeline frames) in metadata.
        "frames": len(frame_indices),
        "frame_samples": len(frame_indices) * len(results),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOCap annotation-only source -> Ref2Dex Stage 3")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sequence", action="append", default=None, help="relative path such as subject_1/20231025_165502; repeatable")
    parser.add_argument("--object-id", default=None)
    parser.add_argument("--side", choices=("left", "right"), default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mano-root", default=str(DEFAULT_MANO_ROOT))
    parser.add_argument("--manopth-root", default=str(DEFAULT_MANOPTH_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    sequences = args.sequence
    if not sequences:
        data_root = Path(args.data_root).expanduser().resolve()
        sequences = [str(path.parent.relative_to(data_root)) for path in sorted(data_root.glob("subject_*/*/meta.yaml"))]
    stats: dict[str, Any] = {"source_sequences": len(sequences), "written_files": 0, "frames": 0, "failed": 0, "failures": []}
    for sequence in sequences:
        try:
            result = _export_sequence(args, sequence)
            stats["written_files"] += len(result["files"])
            stats["frames"] += result["frame_samples"]
        except Exception as exc:  # noqa: BLE001
            stats["failed"] += 1
            stats["failures"].append({"sequence": sequence, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[hocap] failed {sequence}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "HOCap",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "output_root": str(output_root),
        "coordinate_frame": "hand_root",
        "sample_unit": "single_sequence_single_object_single_hand",
        "object_sampling": "cleaned_mesh_10000_surface_4096_seeded_by_sequence_object",
        "contact_target": "geometric_cKDTree_hand_to_object_min_distance",
        "mano_use_pca": True,
        "mano_num_pca_comps": 45,
        "mano_flat_hand_mean": False,
        "mano_pose_repr": "pca",
        "stats": stats,
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
