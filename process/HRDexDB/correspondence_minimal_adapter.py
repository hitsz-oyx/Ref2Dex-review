"""Adapt the HRDexDB ``minimal_allhands`` layout to correspondence Stage 3.

The minimal archive keeps object poses in a shared directory and stores robot
assets under ``assets/mesh_v2``.  This adapter normalizes that layout into the
existing Stage 3 v2.1 contract.  Human episodes use validated MANO fields;
robot episodes use URDF/FK fields and never enter the MANO path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# smplx/chumpy in the training environment still imports the removed NumPy
# aliases.  Install them before the lazy MANO import in the human branch.
for _name, _value in {
    "bool": np.bool_, "int": np.int_, "float": np.float64,
    "complex": np.complex128, "object": np.object_, "unicode": np.str_, "str": np.str_,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)

from dataset.HRDexDB.hrdexdb_contact_heatmaps import hrdexdb_io as io
from src.task.CmDecoder.dataset import _eval_surface, _robot_hand_binding, _robot_hand_mesh, _surface_spec


SCHEMA_NAME = "train_corr_static_v2"
SCHEMA_VERSION = "2.1.0"
NUM_OBJ_POOL = 4096
NUM_HAND_POINTS = 1538
ROBOT_SPECS: dict[str, dict[str, str]] = {
    "inspire_dftp": {"io_hand": "inspire", "urdf": "assets/robots/xarm_inspire_DFTP.urdf"},
    "inspire_f1": {"io_hand": "inspire_f1", "urdf": "assets/robots/xarm_inspire_f1_right.urdf"},
    "allegro_v5": {"io_hand": "allegro_v5", "urdf": "assets/robots/allegro_v5/xarm_allegro_v5.urdf"},
}
HAND_TYPES = ("human", "inspire_dftp", "inspire_f1", "allegro_v5")
_MANO_TEMPLATE: np.ndarray | None = None


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)


_normalize_robot_rows = _normalize_rows


def _episode_seed(episode: str) -> int:
    return int.from_bytes(hashlib.blake2b(episode.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


def _sample_object(mesh_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        points, face_idx = trimesh.sample.sample_surface(mesh, NUM_OBJ_POOL)
    finally:
        np.random.set_state(state)
    normals = np.asarray(mesh.face_normals[face_idx], dtype=np.float32)
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return np.asarray(points, dtype=np.float32), normals


def _resolve_mesh(data_root: Path, object_name: str) -> Path:
    mesh_dir = data_root / "assets" / "mesh_v2" / object_name
    candidates = [mesh_dir / f"{object_name}.obj", *sorted(mesh_dir.glob("*.obj"))]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No mesh_v2 OBJ for object {object_name!r}: {mesh_dir}")


def _load_pose_archive(data_root: Path, hand: str, object_name: str, seq: str) -> tuple[np.ndarray, str, Path]:
    filename = f"{object_name}_{seq}.npz"
    for version in ("v2", "v1"):
        path = data_root / f"object_6d_pose_{version}" / hand / filename
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            keys = sorted(archive.files, key=lambda value: int(value.split("_")[-1]))
            poses = np.stack([np.asarray(archive[key], dtype=np.float32) for key in keys])
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
            raise ValueError(f"Invalid object pose archive {path}: {poses.shape}")
        return poses, version, path
    raise FileNotFoundError(f"No v1/v2 object pose for {hand}/{object_name}/{seq}")


def _world_to_root(points: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    return np.einsum("tji,tnj->tni", root_pose[:, :3, :3], points - root_pose[:, None, :3, 3], optimize=True).astype(np.float32)


def _normals_to_root(normals: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    return _normalize_rows(np.einsum("tji,tnj->tni", root_pose[:, :3, :3], normals, optimize=True))


def _load_mano_params(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.asarray(payload["global_orient"], dtype=np.float32).reshape(3, 3),
        np.asarray(payload["hand_pose"], dtype=np.float32).reshape(15, 3, 3),
        np.asarray(payload["betas"], dtype=np.float32).reshape(10),
        np.asarray(payload["transl"], dtype=np.float32).reshape(3),
        np.asarray(payload["joints"], dtype=np.float32).reshape(21, 3),
    )


def _mano_template() -> np.ndarray:
    global _MANO_TEMPLATE
    if _MANO_TEMPLATE is None:
        from smplx import MANO
        mano_dir = Path("/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano")
        layer = MANO(str(mano_dir), is_rhand=True, use_pca=False, flat_hand_mean=True).eval()
        _MANO_TEMPLATE = layer.v_template.detach().cpu().numpy().reshape(778, 3).astype(np.float32)
    return _MANO_TEMPLATE


def _read_mano_obj(path: Path, expected_faces: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.shape != (778, 3) or faces_array.shape != (1538, 3):
        raise ValueError(f"Invalid MANO OBJ {path}: {vertices_array.shape}, {faces_array.shape}")
    if expected_faces is not None and not np.array_equal(faces_array, expected_faces):
        raise ValueError(f"MANO topology changed at {path}")
    return vertices_array, faces_array


def _robot_q_limits(urdf_path: Path, q_dim: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(q_dim, -np.inf, dtype=np.float32)
    upper = np.full(q_dim, np.inf, dtype=np.float32)
    index = 0
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        if joint.attrib.get("type", "fixed") == "fixed" or joint.find("mimic") is not None:
            continue
        limit = joint.find("limit")
        if limit is not None:
            if "lower" in limit.attrib:
                lower[index] = float(limit.attrib["lower"])
            if "upper" in limit.attrib:
                upper[index] = float(limit.attrib["upper"])
        index += 1
    if index != q_dim:
        raise ValueError(f"URDF q dimension {index} != trajectory q dimension {q_dim}: {urdf_path}")
    return lower, upper


def _resample_q(q_video: np.ndarray, target_len: int) -> np.ndarray:
    if len(q_video) == target_len:
        return np.asarray(q_video, dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(q_video))
    dst = np.linspace(0.0, 1.0, target_len)
    flat = np.asarray(q_video, dtype=np.float64)
    return np.stack([np.interp(dst, src, flat[:, i]) for i in range(flat.shape[1])], axis=1).astype(np.float32)


def _base_payload(*, episode: str, hand: str, object_pose: np.ndarray, object_points: np.ndarray, object_normals: np.ndarray, hand_points: np.ndarray, hand_normals: np.ndarray, root_pose: np.ndarray) -> dict[str, np.ndarray]:
    distances = np.stack([cKDTree(obj).query(hand, k=1, workers=1)[0] for obj, hand in zip(object_points, hand_points)]).astype(np.float32)
    return dict(
        schema_name=np.asarray(SCHEMA_NAME), schema_version=np.asarray(SCHEMA_VERSION),
        dataset_id=np.asarray("hrdexdb"), dataset_name=np.asarray("HRDexDB"),
        seq_id=np.asarray(f"hrdexdb:{episode}"), side=np.asarray("right"),
        raw_frame_id=np.arange(len(object_pose), dtype=np.int32),
        obj_points=object_points, obj_normals=object_normals,
        hand_points=hand_points, hand_normals=hand_normals,
        hand_to_obj_min_dist=distances, coordinate_frame=np.asarray("hand_root"),
        hand_root_pose=root_pose, obj_repr=np.asarray("rigid_canonical"),
    )


def _human_episode(data_root: Path, episode: str, include_mano: bool) -> dict[str, np.ndarray]:
    _, object_name, seq = episode.split("/", 2)
    ep_root = data_root / "human" / object_name / seq
    mano_paths = sorted((ep_root / "hand" / "mano").glob("*.obj"), key=lambda p: int(p.stem))
    param_paths = sorted((ep_root / "hand" / "mano_params").glob("*.json"), key=lambda p: int(p.stem))
    if not mano_paths or len(mano_paths) != len(param_paths):
        raise ValueError(f"MANO mesh/param mismatch: {ep_root}")
    object_pose, pose_version, pose_path = _load_pose_archive(data_root, "human", object_name, seq)
    frame_ids = [int(path.stem) for path in mano_paths]
    if max(frame_ids, default=-1) >= len(object_pose):
        raise ValueError(f"Object pose {pose_path} has {len(object_pose)} frames, needs {max(frame_ids)+1}")
    object_pose = object_pose[frame_ids]
    canonical_points, canonical_normals = _sample_object(_resolve_mesh(data_root, object_name), _episode_seed(episode))
    hand_world: list[np.ndarray] = []
    hand_normals_world: list[np.ndarray] = []
    roots: list[np.ndarray] = []
    global_orient: list[np.ndarray] = []
    hand_pose: list[np.ndarray] = []
    betas: list[np.ndarray] = []
    transl: list[np.ndarray] = []
    topology: np.ndarray | None = None
    for mesh_path, param_path in zip(mano_paths, param_paths):
        vertices, topology = _read_mano_obj(mesh_path, topology)
        root_rot, local_pose, beta, trans, joints = _load_mano_params(param_path)
        face_vertices = vertices[topology]
        normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        hand_world.append(face_vertices.mean(axis=1).astype(np.float32))
        hand_normals_world.append(_normalize_rows(normals))
        root = np.eye(4, dtype=np.float32); root[:3, :3] = root_rot; root[:3, 3] = joints[0]
        roots.append(root); global_orient.append(root_rot); hand_pose.append(local_pose); betas.append(beta); transl.append(trans)
    root_pose = np.stack(roots)
    obj_world = np.einsum("tij,nj->tni", object_pose[:, :3, :3], canonical_points, optimize=True) + object_pose[:, None, :3, 3]
    obj_normals_world = np.einsum("tij,nj->tni", object_pose[:, :3, :3], canonical_normals, optimize=True)
    hand_world_array = np.stack(hand_world)
    hand_normals_world_array = np.stack(hand_normals_world)
    payload = _base_payload(
        episode=episode, hand="human", object_pose=object_pose,
        object_points=_world_to_root(obj_world, root_pose), object_normals=_normals_to_root(obj_normals_world, root_pose),
        hand_points=_world_to_root(hand_world_array, root_pose), hand_normals=_normals_to_root(hand_normals_world_array, root_pose), root_pose=root_pose,
    )
    payload.update(
        obj_points_canonical=canonical_points, obj_normals_canonical=canonical_normals,
        obj_point_id=np.arange(NUM_OBJ_POOL, dtype=np.int32), obj_root_pose_world=object_pose,
    )
    if include_mano:
        payload.update(
            mano_global_orient=Rotation.from_matrix(np.stack(global_orient)).as_rotvec().astype(np.float32),
            mano_transl=np.stack(transl).astype(np.float32),
            mano_pose=Rotation.from_matrix(np.stack(hand_pose).reshape(-1, 3, 3)).as_rotvec().reshape(len(mano_paths), 45).astype(np.float32),
            mano_betas=np.stack(betas).astype(np.float32),
            mano_v_template=_mano_template(),
            mano_use_pca=np.asarray(False), mano_num_pca_comps=np.asarray(0, dtype=np.int64),
            mano_flat_hand_mean=np.asarray(True), mano_pose_repr=np.asarray("axis_angle"),
        )
    return payload


def _robot_episode(data_root: Path, episode: str) -> dict[str, np.ndarray]:
    hand, object_name, seq = episode.split("/", 2)
    spec = ROBOT_SPECS[hand]
    ep_root = data_root / hand / object_name / seq
    object_pose, pose_version, pose_path = _load_pose_archive(data_root, hand, object_name, seq)
    q_video, _, _ = io.load_robot_qpos_on_video_timeline(ep_root, spec["io_hand"])
    q_frames = _resample_q(q_video, len(object_pose))
    urdf_path = data_root.parent / spec["urdf"]
    urdf = io.parse_urdf(urdf_path)
    mesh_cache: dict[Any, Any] = {}
    first_vertices, first_faces = _robot_hand_mesh(io, urdf, q_frames[0], mesh_cache)
    face, bary = _surface_spec(first_vertices, first_faces, NUM_HAND_POINTS, seed=991)
    point_groups, local_points, group_names = _robot_hand_binding(io, urdf, face, bary, mesh_cache)
    link_order = sorted({str(urdf.root_link), *(str(joint.parent) for joint in urdf.joints), *(str(joint.child) for joint in urdf.joints)})
    link_to_index = {name: index for index, name in enumerate(link_order)}
    point_link_index = np.asarray([link_to_index[str(group_names[int(group)])] for group in point_groups], dtype=np.int16)
    first_tfs = io.compute_link_transforms(urdf, q_frames[0])
    _, first_normals = _eval_surface(first_vertices, first_faces, face, bary)
    local_normals = np.empty_like(first_normals, dtype=np.float32)
    for group_index, group_name in enumerate(group_names):
        mask = point_groups == group_index
        local_normals[mask] = _normalize_robot_rows(first_normals[mask]) @ np.asarray(first_tfs[str(group_name)][:3, :3], dtype=np.float32)
    q_lower, q_upper = _robot_q_limits(urdf_path, q_frames.shape[1])
    hand_q_indices = np.arange(6, q_frames.shape[1], dtype=np.int64)
    root_name = "base_link" if "base_link" in first_tfs else str(urdf.root_link)
    c2r = np.asarray(np.load(ep_root / "C2R.npy", allow_pickle=True), dtype=np.float32).reshape(4, 4)
    canonical_points, canonical_normals = _sample_object(_resolve_mesh(data_root, object_name), _episode_seed(episode))
    hand_points: list[np.ndarray] = []; hand_normals: list[np.ndarray] = []; obj_points: list[np.ndarray] = []; obj_normals: list[np.ndarray] = []; roots: list[np.ndarray] = []
    for frame, q in enumerate(q_frames):
        vertices, faces = _robot_hand_mesh(io, urdf, q, mesh_cache)
        hp_robot, hn_robot = _eval_surface(vertices, faces, face, bary)
        hn_robot = _normalize_robot_rows(hn_robot)
        hp_world = hp_robot @ c2r[:3, :3].T + c2r[:3, 3]
        hn_world = hn_robot @ c2r[:3, :3].T
        root_robot = np.asarray(io.compute_link_transforms(urdf, q).get(root_name), dtype=np.float32)
        root_world = c2r @ root_robot
        roots.append(root_world)
        hand_points.append((hp_world - root_world[:3, 3]) @ root_world[:3, :3])
        hand_normals.append(_normalize_robot_rows(hn_world @ root_world[:3, :3]))
        op_world = canonical_points @ object_pose[frame, :3, :3].T + object_pose[frame, :3, 3]
        on_world = canonical_normals @ object_pose[frame, :3, :3].T
        obj_points.append((op_world - root_world[:3, 3]) @ root_world[:3, :3])
        obj_normals.append(_normalize_robot_rows(on_world @ root_world[:3, :3]))
    root_pose = np.asarray(roots, dtype=np.float32)
    payload = _base_payload(
        episode=episode, hand=hand, object_pose=object_pose,
        object_points=np.asarray(obj_points, dtype=np.float32), object_normals=np.asarray(obj_normals, dtype=np.float32),
        hand_points=np.asarray(hand_points, dtype=np.float32), hand_normals=np.asarray(hand_normals, dtype=np.float32), root_pose=root_pose,
    )
    payload.update(
        obj_points_canonical=canonical_points, obj_normals_canonical=canonical_normals,
        obj_point_id=np.arange(NUM_OBJ_POOL, dtype=np.int32), obj_root_pose_world=object_pose,
        robot_repr=np.asarray(f"{hand}_fk"), robot_qpos=q_frames, robot_hand_qpos_indices=hand_q_indices,
        robot_qpos_lower=q_lower, robot_qpos_upper=q_upper,
        robot_hand_qpos_noise_std=np.full(len(hand_q_indices), 0.03, dtype=np.float32),
        robot_hand_points_local=np.asarray(local_points, dtype=np.float32),
        robot_hand_normals_local=np.asarray(local_normals, dtype=np.float32),
        robot_hand_point_link_index=point_link_index, robot_link_names=np.asarray(link_order),
        robot_c2r=c2r, robot_urdf_path=np.asarray(str(urdf_path)),
    )
    return payload


def enumerate_episodes(data_root: Path, hands: tuple[str, ...]) -> list[str]:
    episodes: list[str] = []
    for hand in hands:
        hand_root = data_root / hand
        if not hand_root.is_dir():
            continue
        for object_root in sorted(path for path in hand_root.iterdir() if path.is_dir()):
            for seq_root in sorted(path for path in object_root.iterdir() if path.is_dir()):
                episode = f"{hand}/{object_root.name}/{seq_root.name}"
                if hand != "human" and not (seq_root / "C2R.npy").is_file():
                    continue
                pose_exists = any(
                    (data_root / f"object_6d_pose_{version}" / hand / f"{object_root.name}_{seq_root.name}.npz").is_file()
                    for version in ("v2", "v1")
                )
                if not pose_exists:
                    continue
                episodes.append(episode)
    return episodes


def export_episode(data_root: Path, episode: str, output_root: Path, *, max_frames: int = 0) -> Path:
    hand = episode.split("/", 1)[0]
    payload = _human_episode(data_root, episode, include_mano=True) if hand == "human" else _robot_episode(data_root, episode)
    if max_frames > 0:
        original_frame_count = int(payload["raw_frame_id"].shape[0])
        for key, value in list(payload.items()):
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == original_frame_count:
                payload[key] = value[:max_frames]
        payload["raw_frame_id"] = payload["raw_frame_id"][:max_frames]
    out_dir = output_root / hand
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hrdexdb_{episode.replace('/', '_')}_right.npz"
    np.savez_compressed(out_path, **payload)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Extracted .../dataset/HRDexDB/v0_nonvideo")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episode", action="append", default=[])
    parser.add_argument("--hands", default=",".join(HAND_TYPES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    hands = tuple(value.strip() for value in args.hands.split(",") if value.strip())
    episodes = list(args.episode)
    if args.all:
        episodes = enumerate_episodes(data_root, hands)
    if not episodes:
        raise SystemExit("Specify --episode or --all")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / args.manifest_name
    errors_path = output_root / args.manifest_name.replace("manifest", "errors").replace(".jsonl", ".jsonl")
    done = failed = skipped = 0
    with manifest_path.open("a", encoding="utf-8") as manifest, errors_path.open("a", encoding="utf-8") as errors:
        for index, episode in enumerate(episodes, 1):
            hand = episode.split("/", 1)[0]
            out_path = output_root / hand / f"hrdexdb_{episode.replace('/', '_')}_right.npz"
            if args.resume and out_path.is_file():
                skipped += 1
                continue
            started = time.time()
            try:
                path = export_episode(data_root, episode, output_root, max_frames=args.max_frames)
                with np.load(path, allow_pickle=False) as data:
                    frames = int(data["obj_points"].shape[0])
                record = {"episode": episode, "status": "ok", "output": str(path), "frames": frames, "seconds": time.time() - started}
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n"); manifest.flush()
                done += 1
                print(json.dumps({"index": index, "total": len(episodes), **record}, ensure_ascii=False), flush=True)
            except Exception as exc:  # keep the full sweep going and make failures auditable
                record = {"episode": episode, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                errors.write(json.dumps(record, ensure_ascii=False) + "\n"); errors.flush()
                failed += 1
                print(json.dumps({"index": index, "total": len(episodes), **record}, ensure_ascii=False), flush=True)
    summary = {"requested": len(episodes), "converted": done, "skipped": skipped, "failed": failed, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    summary_name = args.manifest_name.replace("manifest", "summary").replace(".jsonl", ".json")
    (output_root / summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
