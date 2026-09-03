"""Export one HRDexDB robot episode to correspondence Stage 3.

This smoke adapter targets ``inspire_dftp`` first.  It samples a stable set
of 1538 hand surface points once from the URDF mesh, stores link-local
coordinates, and keeps the hand qpos/URDF metadata needed for q-space noise at
dataset time.  Only the six hand joints are perturbed; the arm/base remains
fixed so the current hand-root frame does not move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from dataset.HRDexDB.hrdexdb_contact_heatmaps import hrdexdb_io as io
from src.task.CmDecoder.dataset import (
    _eval_surface,
    _robot_hand_binding,
    _robot_hand_mesh,
    _surface_spec,
)


SCHEMA_NAME = "train_corr_static_v2"
SCHEMA_VERSION = "2.1.0"
NUM_OBJ_POOL = 4096
NUM_HAND_POINTS = 1538


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)


def _pose_archive(path: Path, frame_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files, key=lambda value: int(value.split("_")[-1]))
        poses = np.stack([np.asarray(archive[key], dtype=np.float32) for key in keys[:frame_count]])
    if poses.shape != (frame_count, 4, 4):
        raise ValueError(f"Invalid object pose archive: {poses.shape}")
    return poses


def _sample_object(path: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False)
    points, face_idx = trimesh.sample.sample_surface(mesh, NUM_OBJ_POOL, seed=seed)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return np.asarray(points, dtype=np.float32), normals


def _robot_q_limits(urdf_path: Path, q_dim: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(q_dim, -np.inf, dtype=np.float32)
    upper = np.full(q_dim, np.inf, dtype=np.float32)
    q_index = 0
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        joint_type = joint.attrib.get("type", "fixed")
        mimic = joint.find("mimic")
        if joint_type == "fixed" or mimic is not None:
            continue
        limit = joint.find("limit")
        if limit is not None:
            if "lower" in limit.attrib:
                lower[q_index] = float(limit.attrib["lower"])
            if "upper" in limit.attrib:
                upper[q_index] = float(limit.attrib["upper"])
        q_index += 1
    if q_index != q_dim:
        raise ValueError(f"URDF q dimension {q_index} does not match trajectory {q_dim}")
    return lower, upper


def export_episode(data_root: Path, episode: str, output_root: Path, *, max_frames: int = 0) -> Path:
    episode_root = (data_root / episode).resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(episode_root)
    hand_name = episode.split("/", 1)[0]
    if hand_name != "inspire_dftp":
        raise ValueError("This first robot adapter only supports inspire_dftp")
    urdf_path = (data_root.parent / "assets" / "robots" / "xarm_inspire_DFTP.urdf").resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    q_raw, _robot_time = io.load_robot_qpos(episode_root, "inspire")
    video_time, _video_frame_ids = io.load_video_timeline(episode_root, len(q_raw))
    q_video = np.stack([np.interp(video_time, _robot_time, q_raw[:, j]) for j in range(q_raw.shape[1])], axis=1).astype(np.float32)
    with np.load(episode_root / "object_6d_pose_v2.npz", allow_pickle=False) as archive:
        keys = sorted(archive.files, key=lambda value: int(value.split("_")[-1]))
    frame_count = len(keys)
    if max_frames > 0:
        frame_count = min(frame_count, max_frames)
    q_index = np.linspace(0, len(q_video) - 1, frame_count).round().astype(np.int64)
    q_frames = q_video[q_index]
    object_pose = _pose_archive(episode_root / "object_6d_pose_v2.npz", frame_count)
    object_name = episode_root.parent.name
    object_mesh_path = data_root / "assets" / "mesh" / object_name / f"{object_name}.obj"
    object_points_canonical, object_normals_canonical = _sample_object(object_mesh_path, seed=42)
    c2r = np.asarray(np.load(episode_root / "C2R.npy", allow_pickle=False), dtype=np.float32)
    urdf = io.parse_urdf(urdf_path)
    mesh_cache: dict = {}
    first_vertices, first_faces = _robot_hand_mesh(io, urdf, q_frames[0], mesh_cache)
    face, bary = _surface_spec(first_vertices, first_faces, NUM_HAND_POINTS, seed=991)
    point_groups, local_points, group_names = _robot_hand_binding(io, urdf, face, bary, mesh_cache)
    link_order = sorted({str(urdf.root_link), *(str(joint.parent) for joint in urdf.joints), *(str(joint.child) for joint in urdf.joints)})
    link_to_index = {name: index for index, name in enumerate(link_order)}
    point_link_index = np.asarray([link_to_index[str(group_names[int(group)])] for group in point_groups], dtype=np.int16)
    first_link_tfs = io.compute_link_transforms(urdf, q_frames[0])
    first_hand_points_robot, first_hand_normals_robot = _eval_surface(first_vertices, first_faces, face, bary)
    first_hand_normals_robot = _normalize_rows(first_hand_normals_robot)
    local_normals = np.empty_like(first_hand_normals_robot)
    for group_index, group_name in enumerate(group_names):
        mask = point_groups == group_index
        transform = first_link_tfs[str(group_name)]
        local_normals[mask] = first_hand_normals_robot[mask] @ transform[:3, :3]
    q_lower, q_upper = _robot_q_limits(urdf_path, q_frames.shape[1])
    hand_q_indices = np.arange(6, 12, dtype=np.int64)
    hand_noise_std = np.full(len(hand_q_indices), 0.03, dtype=np.float32)

    hand_points, hand_normals, root_poses = [], [], []
    object_points, object_normals = [], []
    for frame, q in enumerate(q_frames):
        vertices, faces = _robot_hand_mesh(io, urdf, q, mesh_cache)
        hp_robot, hn_robot = _eval_surface(vertices, faces, face, bary)
        hn_robot = _normalize_rows(hn_robot)
        hp_world = hp_robot @ c2r[:3, :3].T + c2r[:3, 3]
        hn_world = hn_robot @ c2r[:3, :3].T
        root_robot = io.compute_link_transforms(urdf, q).get("base_link", np.eye(4))
        root_world = c2r @ root_robot
        root_poses.append(root_world.astype(np.float32))
        hand_points.append((hp_world - root_world[:3, 3]) @ root_world[:3, :3])
        hand_normals.append(_normalize_rows(hn_world @ root_world[:3, :3]))
        op_world = object_points_canonical @ object_pose[frame, :3, :3].T + object_pose[frame, :3, 3]
        on_world = object_normals_canonical @ object_pose[frame, :3, :3].T
        object_points.append((op_world - root_world[:3, 3]) @ root_world[:3, :3])
        object_normals.append(_normalize_rows(on_world @ root_world[:3, :3]))
    hand_points = np.asarray(hand_points, dtype=np.float32)
    hand_normals = np.asarray(hand_normals, dtype=np.float32)
    object_points = np.asarray(object_points, dtype=np.float32)
    object_normals = np.asarray(object_normals, dtype=np.float32)
    root_poses = np.asarray(root_poses, dtype=np.float32)
    distances = np.stack([cKDTree(obj).query(hand, k=1, workers=1)[0] for obj, hand in zip(object_points, hand_points)]).astype(np.float32)

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"hrdexdb_{episode.replace('/', '_')}.npz"
    np.savez_compressed(
        output_path,
        schema_name=np.asarray(SCHEMA_NAME), schema_version=np.asarray(SCHEMA_VERSION),
        dataset_id=np.asarray("hrdexdb_robot"), dataset_name=np.asarray("HRDexDB"),
        seq_id=np.asarray(f"hrdexdb:{episode}"), side=np.asarray("right"),
        raw_frame_id=np.arange(frame_count, dtype=np.int32),
        obj_points=object_points, obj_normals=object_normals,
        hand_points=hand_points, hand_normals=hand_normals,
        hand_to_obj_min_dist=distances, coordinate_frame=np.asarray("hand_root"),
        hand_root_pose=root_poses, obj_repr=np.asarray("rigid_canonical"),
        obj_points_canonical=object_points_canonical, obj_normals_canonical=object_normals_canonical,
        obj_point_id=np.arange(NUM_OBJ_POOL, dtype=np.int32), obj_root_pose_world=object_pose,
        robot_repr=np.asarray("inspire_dftp_fk"), robot_qpos=q_frames,
        robot_hand_qpos_indices=hand_q_indices, robot_qpos_lower=q_lower, robot_qpos_upper=q_upper,
        robot_hand_qpos_noise_std=hand_noise_std, robot_hand_points_local=local_points,
        robot_hand_normals_local=local_normals, robot_hand_point_link_index=point_link_index,
        robot_link_names=np.asarray(link_order), robot_c2r=c2r, robot_urdf_path=np.asarray(str(urdf_path)),
    )
    meta = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "dataset_name": "HRDexDB", "episode": episode, "robot_repr": "inspire_dftp_fk",
        "coordinate_frame": "hand_root", "num_frames": frame_count,
        "num_obj_points": NUM_OBJ_POOL, "num_hand_points": NUM_HAND_POINTS,
        "robot_hand_qpos_indices": hand_q_indices.tolist(), "robot_noise_std_rad": hand_noise_std.tolist(),
        "robot_urdf": str(urdf_path), "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "frames": frame_count, "obj": object_points.shape, "hand": hand_points.shape, "dist_min": float(distances.min())}, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/HRDexDB/v0_nonvideo")
    parser.add_argument("--episode", default="inspire_dftp/mug_holder/2")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    export_episode(Path(args.data_root).resolve(), args.episode, Path(args.output_root).resolve(), max_frames=args.max_frames)


if __name__ == "__main__":
    main()
