"""Reconstruct and visualize one HRDexDB human episode.

The diagnostic is intentionally independent of the Stage 3 exporter.  It
reconstructs the source MANO JSON with the standard right-hand model and
compares the result to the saved 778-vertex OBJ before checking the derived
hand-root geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

for _name, _value in {
    "bool": np.bool_, "int": np.int_, "float": np.float64,
    "complex": np.complex128, "object": np.object_, "unicode": np.str_, "str": np.str_,
}.items():
    setattr(np, _name, _value)

import torch
from scipy.spatial.transform import Rotation
from smplx import MANO
import trimesh


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int64)
    if vertices_arr.shape != (778, 3) or faces_arr.shape != (1538, 3):
        raise ValueError(f"Unexpected MANO OBJ shapes at {path}: {vertices_arr.shape}, {faces_arr.shape}")
    return vertices_arr, faces_arr


def _params(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    global_orient = np.asarray(payload["global_orient"], dtype=np.float32).reshape(3, 3)
    hand_pose = np.asarray(payload["hand_pose"], dtype=np.float32).reshape(15, 3, 3)
    betas = np.asarray(payload["betas"], dtype=np.float32).reshape(10)
    transl = np.asarray(payload["transl"], dtype=np.float32).reshape(3)
    joints = np.asarray(payload["joints"], dtype=np.float32).reshape(21, 3)
    return global_orient, hand_pose, betas, transl, joints


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[:, faces]
    centers = tri.mean(axis=2).astype(np.float32)
    normals = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return centers, normals.astype(np.float32)


def _root_pose(global_orient: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = global_orient
    pose[:3, 3] = wrist
    return pose


def _to_root(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (pose[:3, :3].T @ (points - pose[:3, 3]).T).T.astype(np.float32)


def _plot_frame(path: Path, *, hand: np.ndarray, obj: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(8, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    obj_stride = max(1, len(obj) // 5000)
    ax.scatter(obj[::obj_stride, 0], obj[::obj_stride, 1], obj[::obj_stride, 2], s=1.0, c="#3b82f6", alpha=0.35, label="object")
    ax.scatter(hand[:, 0], hand[:, 1], hand[:, 2], s=2.0, c="#ef4444", alpha=0.65, label="MANO")
    all_points = np.concatenate([obj, hand], axis=0)
    center = all_points.mean(axis=0)
    radius = max(float(np.ptp(all_points, axis=0).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def diagnose(data_root: Path, episode: str, stage3_path: Path, output_root: Path, frames: list[int]) -> dict:
    episode_root = data_root / episode
    mano_paths = sorted((episode_root / "hand" / "mano").glob("*.obj"))
    frame_ids = [int(path.stem) for path in mano_paths]
    if not frame_ids:
        raise FileNotFoundError(f"No MANO OBJ files under {episode_root}")
    frame_to_index = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    selected = [frame for frame in frames if frame in frame_to_index]
    if not selected:
        selected = [frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]]

    saved_vertices = []
    global_orient = []
    hand_pose = []
    betas = []
    transl = []
    joints = []
    faces: np.ndarray | None = None
    for path in mano_paths:
        vertices, obj_faces = _read_obj(path)
        if faces is None:
            faces = obj_faces
        elif not np.array_equal(faces, obj_faces):
            raise ValueError(f"MANO topology changed at {path}")
        saved_vertices.append(vertices)
        g, h, b, t, j = _params(episode_root / "hand" / "mano_params" / f"{path.stem}.json")
        global_orient.append(g)
        hand_pose.append(h)
        betas.append(b)
        transl.append(t)
        joints.append(j)
    assert faces is not None
    saved_vertices_arr = np.stack(saved_vertices)
    global_orient_arr = np.stack(global_orient)
    hand_pose_arr = np.stack(hand_pose)
    betas_arr = np.stack(betas)
    transl_arr = np.stack(transl)
    joints_arr = np.stack(joints)

    mano_root = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
    layer = MANO(str(mano_root), is_rhand=True, use_pca=False, flat_hand_mean=True).eval()
    pose_global = Rotation.from_matrix(global_orient_arr).as_rotvec().astype(np.float32)
    pose_hand = Rotation.from_matrix(hand_pose_arr.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 45).astype(np.float32)
    with torch.no_grad():
        reconstructed = layer(
            global_orient=torch.from_numpy(pose_global),
            hand_pose=torch.from_numpy(pose_hand),
            betas=torch.from_numpy(betas_arr.astype(np.float32)),
            transl=torch.from_numpy(transl_arr.astype(np.float32)),
        ).vertices.detach().cpu().numpy().astype(np.float32)
    vertex_error = np.linalg.norm(reconstructed - saved_vertices_arr, axis=-1)
    root_poses = np.stack([_root_pose(g, j[0]) for g, j in zip(global_orient_arr, joints_arr)])
    root_joints = np.stack([_to_root(j, pose) for j, pose in zip(joints_arr[:, 0], root_poses)])

    with np.load(stage3_path, allow_pickle=False) as stage3:
        stored_hand = np.asarray(stage3["hand_points"], dtype=np.float32)
        stored_obj = np.asarray(stage3["obj_points"], dtype=np.float32)
        stored_frames = np.asarray(stage3["raw_frame_id"], dtype=np.int32)
    frame_lookup = {int(frame): i for i, frame in enumerate(stored_frames.tolist())}
    recon_hand_centers, _ = _face_geometry(reconstructed, faces)
    root_hand_centers = np.stack([_to_root(points, pose) for points, pose in zip(recon_hand_centers, root_poses)])
    root_hand_error = np.linalg.norm(root_hand_centers - stored_hand, axis=-1)

    mesh_path = data_root / "assets" / "mesh" / episode_root.parent.name / f"{episode_root.parent.name}.obj"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    object_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    with np.load(episode_root / "object_6d_pose_v2.npz", allow_pickle=False) as archive:
        for frame in selected:
            idx = frame_to_index[frame]
            pose = np.asarray(archive[f"frame_{frame}"], dtype=np.float32)
            world_obj = object_vertices @ pose[:3, :3].T + pose[:3, 3]
            world_hand = saved_vertices_arr[idx]
            root_obj = _to_root(world_obj, root_poses[idx])
            root_hand = _to_root(world_hand, root_poses[idx])
            frame_dir = output_root / f"frame_{frame:05d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            _plot_frame(frame_dir / "world.png", hand=world_hand, obj=world_obj, title=f"HRDexDB human/{episode} frame {frame} — world")
            _plot_frame(frame_dir / "hand_root.png", hand=root_hand, obj=root_obj, title=f"HRDexDB human/{episode} frame {frame} — hand_root")

    report = {
        "episode": episode,
        "side": "right (confirmed by HRDexDB paired annotation path)",
        "num_frames": len(frame_ids),
        "mano_forward": {
            "model": str(mano_root / "MANO_RIGHT.pkl"),
            "flat_hand_mean": True,
            "pose_representation": "axis_angle_45 converted from 3x3 matrices",
            "vertex_rmse_m": float(np.sqrt(np.mean(vertex_error**2))),
            "vertex_mean_m": float(vertex_error.mean()),
            "vertex_p95_m": float(np.percentile(vertex_error, 95)),
            "vertex_max_m": float(vertex_error.max()),
        },
        "hand_root": {
            "definition": "R_global_orient.T @ (x_world - joints[0])",
            "joint0_max_abs_m": float(np.abs(root_joints).max()),
            "rotation_det_min": float(np.linalg.det(global_orient_arr).min()),
            "rotation_orthogonality_max_abs": float(np.max(np.abs(np.einsum("tji,tjk->tik", global_orient_arr, global_orient_arr) - np.eye(3)))),
            "stage3_face_center_rmse_m": float(np.sqrt(np.mean(root_hand_error**2))),
            "stage3_face_center_max_m": float(root_hand_error.max()),
        },
        "visualized_frames": selected,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/HRDexDB/v0_nonvideo")
    parser.add_argument("--episode", default="human/apple/0")
    parser.add_argument("--stage3", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frames", nargs="*", type=int, default=[0, 100, 200])
    args = parser.parse_args()
    diagnose(Path(args.data_root).resolve(), args.episode, Path(args.stage3).resolve(), Path(args.output_root).resolve(), args.frames)


if __name__ == "__main__":
    main()
