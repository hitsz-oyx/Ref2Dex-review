"""Small HRDexDB-human -> Ref2Dex Stage 3 smoke exporter.

This adapter intentionally starts with the representation that is already
available in the transferred HRDexDB human episodes: per-frame MANO OBJ
meshes, MANO parameter JSON, object 6D poses and the clean object mesh.  It
does not enable MANO reconstruction yet; that requires an independent check
of the rotation-matrix/translation convention in the source parameters.

The exporter writes one Stage 3 v2.1 NPZ for one episode.  Human MANO and
object poses are already expressed in the same HRDexDB world frame, while
``C2R.npy`` is a camera-to-robot calibration used by the robot pipeline and
is deliberately not applied here.  The hand-root frame is built from the
MANO global rotation and the recorded wrist joint (``joints[0]``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

for _name, _value in {
    "bool": np.bool_, "int": np.int_, "float": np.float64,
    "complex": np.complex128, "object": np.object_, "unicode": np.str_, "str": np.str_,
}.items():
    setattr(np, _name, _value)


SCHEMA_NAME = "train_corr_static_v2"
SCHEMA_VERSION = "2.1.0"
NUM_OBJ_POOL = 4096
NUM_HAND_POINTS = 1538


def _load_obj_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type: {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.shape != (778, 3) or faces.shape != (1538, 3):
        raise ValueError(f"Expected MANO topology (778,1538), got {vertices.shape}, {faces.shape}: {path}")
    return vertices, faces


def _sample_object(mesh_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported object mesh type: {mesh_path}")
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, face_idx = trimesh.sample.sample_surface(mesh, NUM_OBJ_POOL)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return np.asarray(points, dtype=np.float32), normals


def _read_mano_obj(path: Path, expected_faces: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            indices = []
            for token in line.split()[1:4]:
                indices.append(int(token.split("/")[0]) - 1)
            faces.append(indices)
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if v.shape != (778, 3) or f.shape != (1538, 3):
        raise ValueError(f"Invalid MANO OBJ {path}: vertices={v.shape}, faces={f.shape}")
    if expected_faces is not None and not np.array_equal(f, expected_faces):
        raise ValueError(f"MANO topology changed at {path}")
    return v, f


def _load_pose_archive(path: Path, frame_ids: list[int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        poses = []
        for frame_id in frame_ids:
            key = f"frame_{frame_id}"
            if key not in archive.files:
                raise KeyError(f"{path}: missing {key}")
            pose = np.asarray(archive[key], dtype=np.float32)
            if pose.shape != (4, 4):
                raise ValueError(f"{path}:{key}: expected (4,4), got {pose.shape}")
            poses.append(pose)
    return np.stack(poses, axis=0)


def _load_params(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    global_orient = np.asarray(payload["global_orient"], dtype=np.float32).reshape(3, 3)
    hand_pose = np.asarray(payload["hand_pose"], dtype=np.float32).reshape(15, 3, 3)
    betas = np.asarray(payload["betas"], dtype=np.float32).reshape(10)
    transl = np.asarray(payload["transl"], dtype=np.float32).reshape(3)
    joints = np.asarray(payload["joints"], dtype=np.float32).reshape(21, 3)
    return global_orient, hand_pose, betas, transl, joints


def _world_to_hand_root(points: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    rotation = root_pose[:, :3, :3]
    translation = root_pose[:, :3, 3]
    return np.einsum("tji,tnj->tni", rotation, points - translation[:, None], optimize=True).astype(np.float32)


def _normals_to_hand_root(normals: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    result = np.einsum("tji,tnj->tni", root_pose[:, :3, :3], normals, optimize=True)
    result /= np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)
    return result.astype(np.float32)


def export_episode(
    data_root: Path,
    episode: str,
    output_root: Path,
    *,
    max_frames: int = 0,
    include_mano: bool = False,
) -> Path:
    episode_root = (data_root / episode).resolve()
    if data_root.resolve() not in episode_root.parents:
        raise ValueError(f"Episode escapes data root: {episode}")
    if not episode_root.is_dir():
        raise FileNotFoundError(episode_root)
    mano_paths = sorted((episode_root / "hand" / "mano").glob("*.obj"))
    param_paths = sorted((episode_root / "hand" / "mano_params").glob("*.json"))
    if not mano_paths or len(mano_paths) != len(param_paths):
        raise ValueError(f"MANO mesh/parameter count mismatch in {episode_root}")
    frame_ids = [int(path.stem) for path in mano_paths]
    pose_archive = episode_root / "object_6d_pose_v2.npz"
    if not pose_archive.is_file():
        raise FileNotFoundError(pose_archive)
    object_name = episode_root.parent.name
    object_mesh = data_root / "assets" / "mesh" / object_name / f"{object_name}.obj"
    if not object_mesh.is_file():
        raise FileNotFoundError(object_mesh)
    if max_frames > 0:
        mano_paths, param_paths, frame_ids = mano_paths[:max_frames], param_paths[:max_frames], frame_ids[:max_frames]
    object_pose = _load_pose_archive(pose_archive, frame_ids)
    seed = int.from_bytes(hashlib.blake2b(episode.encode("utf-8"), digest_size=8).digest(), "little") & 0x7FFFFFFF
    canonical_points, canonical_normals = _sample_object(object_mesh, seed=seed)

    hand_points_world: list[np.ndarray] = []
    hand_normals_world: list[np.ndarray] = []
    root_poses: list[np.ndarray] = []
    global_orient_values: list[np.ndarray] = []
    hand_pose_values: list[np.ndarray] = []
    betas_values: list[np.ndarray] = []
    transl_values: list[np.ndarray] = []
    topology: np.ndarray | None = None
    for mesh_path, params_path in zip(mano_paths, param_paths):
        vertices, topology = _read_mano_obj(mesh_path, topology)
        global_orient, hand_pose, betas, transl, joints = _load_params(params_path)
        wrist = joints[0]
        faces = topology
        face_vertices = vertices[faces]
        centers = face_vertices.mean(axis=1)
        normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
        hand_points_world.append(centers.astype(np.float32))
        hand_normals_world.append(normals.astype(np.float32))
        root = np.eye(4, dtype=np.float32)
        root[:3, :3] = global_orient
        root[:3, 3] = wrist
        root_poses.append(root)
        global_orient_values.append(global_orient)
        hand_pose_values.append(hand_pose)
        betas_values.append(betas)
        transl_values.append(transl)
    hand_world = np.stack(hand_points_world)
    hand_normals_world = np.stack(hand_normals_world)
    hand_root_pose = np.stack(root_poses)
    obj_world = np.einsum("tij,nj->tni", object_pose[:, :3, :3], canonical_points, optimize=True) + object_pose[:, None, :3, 3]
    obj_normals_world = np.einsum("tij,nj->tni", object_pose[:, :3, :3], canonical_normals, optimize=True)
    obj_points = _world_to_hand_root(obj_world, hand_root_pose)
    obj_normals = _normals_to_hand_root(obj_normals_world, hand_root_pose)
    hand_points = _world_to_hand_root(hand_world, hand_root_pose)
    hand_normals = _normals_to_hand_root(hand_normals_world, hand_root_pose)
    distances = np.stack([cKDTree(obj).query(hand, k=1, workers=1)[0] for obj, hand in zip(obj_points, hand_points)]).astype(np.float32)

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"hrdexdb_{episode.replace('/', '_')}_right.npz"
    payload = dict(
        schema_name=np.asarray(SCHEMA_NAME), schema_version=np.asarray(SCHEMA_VERSION),
        dataset_id=np.asarray("hrdexdb"), dataset_name=np.asarray("HRDexDB"),
        seq_id=np.asarray(f"hrdexdb:{episode}"), side=np.asarray("right"),
        raw_frame_id=np.asarray(frame_ids, dtype=np.int32),
        obj_points=obj_points, obj_normals=obj_normals,
        hand_points=hand_points, hand_normals=hand_normals,
        hand_to_obj_min_dist=distances, coordinate_frame=np.asarray("hand_root"),
        hand_root_pose=hand_root_pose,
        obj_repr=np.asarray("rigid_canonical"), obj_points_canonical=canonical_points,
        obj_normals_canonical=canonical_normals, obj_point_id=np.arange(NUM_OBJ_POOL, dtype=np.int32),
        obj_root_pose_world=object_pose,
    )
    if include_mano:
        # HRDexDB JSON stores full 3x3 rotations.  The source OBJ is exactly
        # reproduced by standard MANO_RIGHT with flat_hand_mean=True and the
        # raw transl/betas values, so expose the equivalent axis-angle45
        # representation for the train-time reconstruction path.
        mano_root = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
        from smplx import MANO

        mano_layer = MANO(str(mano_root), is_rhand=True, use_pca=False, flat_hand_mean=True).eval()
        payload.update(
            mano_global_orient=Rotation.from_matrix(np.stack(global_orient_values)).as_rotvec().astype(np.float32),
            mano_transl=np.stack(transl_values).astype(np.float32),
            mano_pose=Rotation.from_matrix(np.stack(hand_pose_values).reshape(-1, 3, 3)).as_rotvec().reshape(len(frame_ids), 45).astype(np.float32),
            mano_betas=np.stack(betas_values).astype(np.float32),
            mano_v_template=mano_layer.v_template.detach().cpu().numpy().reshape(778, 3).astype(np.float32),
            mano_use_pca=np.asarray(False), mano_num_pca_comps=np.asarray(0, dtype=np.int64),
            mano_flat_hand_mean=np.asarray(True), mano_pose_repr=np.asarray("axis_angle"),
        )
    np.savez_compressed(output_path, **payload)
    meta = {
        "schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "dataset_name": "HRDexDB", "episode": episode,
        "coordinate_frame": "hand_root", "hand_root_translation": "joints[0]",
        "hand_root_rotation": "MANO global_orient matrix",
        "object_pose": "object_6d_pose_v2 frame_N, direct world transform",
        "c2r": "not applied to human episode; reserved for robot pipeline",
        "num_frames": len(frame_ids), "num_obj_points": NUM_OBJ_POOL,
        "num_hand_points": NUM_HAND_POINTS, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mano_fields": bool(include_mano),
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "frames": len(frame_ids), "obj": obj_points.shape, "hand": hand_points.shape, "dist_min": float(distances.min()), "dist_p01": float(np.percentile(distances, 1))}, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset/HRDexDB/v0_nonvideo")
    parser.add_argument("--episode", default="human/apple/0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--include-mano", action="store_true", help="write validated axis-angle45 MANO fields")
    args = parser.parse_args()
    export_episode(
        Path(args.data_root).expanduser().resolve(), args.episode,
        Path(args.output_root).expanduser().resolve(), max_frames=args.max_frames,
        include_mano=args.include_mano,
    )


if __name__ == "__main__":
    main()
