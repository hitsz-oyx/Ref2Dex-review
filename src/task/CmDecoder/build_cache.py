"""Build layered HRDexDB caches for CmDecoder before distributed training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.task.CmDecoder.dataset import (
    _eval_surface,
    _load_hrdex_io,
    _load_pose,
    _robot_hand_binding,
    _robot_hand_mesh,
    _urdf_link_order,
    _rotate_to_frame,
    _surface_spec,
    _to_frame,
    _to_world,
)


SCHEMA = "cmdecoder_layered_v4"
ROBOT_TYPES = ("human", "allegro_v5", "inspire_dftp", "inspire_f1")
REPO_ROOT = Path(__file__).resolve().parents[3]
HRDEXDB_ROOT = REPO_ROOT / "dataset" / "HRDexDB"


def _episode_kind(root: Path, episode: Path) -> str:
    """Return the HRDexDB top-level source kind for an episode."""
    relative = episode.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Expected <hand>/<object>/<scene>, got {relative}")
    kind = relative.parts[0]
    if kind not in ROBOT_TYPES:
        raise ValueError(f"Unsupported HRDexDB source kind {kind!r}: {episode}")
    return kind


def _resolve_pose_sources(root: Path, episode: Path) -> tuple[list[Path], str]:
    """Prefer official compact v2 poses, then v1, then legacy per-frame text."""
    kind = _episode_kind(root, episode)
    object_name = episode.parent.name
    scene = episode.name
    for version in ("v2", "v1"):
        compact = root / f"object_6d_pose_{version}" / kind / f"{object_name}_{scene}.npz"
        if compact.is_file():
            return [compact], f"compact_{version}"
    for pose_root in ("object_6d_pose_v2", "object_6d"):
        poses = sorted((episode / pose_root).glob("pose_*.txt"))
        if poses:
            return poses, f"legacy_{pose_root}"
    raise FileNotFoundError(f"No object poses: {episode}")


def _load_pose_sequence(paths: list[Path]) -> np.ndarray:
    if len(paths) == 1 and paths[0].suffix == ".npz":
        with np.load(paths[0], allow_pickle=False) as compact:
            def frame_index(key: str) -> int:
                prefix, separator, suffix = key.rpartition("_")
                if prefix != "frame" or not separator or not suffix.isdigit():
                    raise ValueError(f"Invalid compact pose key {key!r}: {paths[0]}")
                return int(suffix)

            keys = sorted(compact.files, key=frame_index)
            if not keys:
                raise ValueError(f"Empty compact pose archive: {paths[0]}")
            expected = list(range(len(keys)))
            actual = [frame_index(key) for key in keys]
            if actual != expected:
                raise ValueError(f"Non-contiguous compact pose frames: {paths[0]}")
            poses = np.stack([np.asarray(compact[key], dtype=np.float32) for key in keys])
    else:
        poses = np.stack([_load_pose(path) for path in paths]).astype(np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Invalid object pose sequence shape {poses.shape}: {paths[0]}")
    return poses.astype(np.float32, copy=False)


def _split_object_disjoint(
    episodes: list[str], *, seed: int, val_fraction: float, test_fraction: float
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    groups: dict[str, list[str]] = {}
    for episode in episodes:
        parts = Path(episode).parts
        if len(parts) < 3:
            raise ValueError(f"Expected <robot>/<object>/<scene>, got {episode!r}")
        groups.setdefault(parts[-2], []).append(episode)
    objects = sorted(groups)
    random.Random(seed + 1).shuffle(objects)
    targets = {
        "test": len(episodes) * float(test_fraction),
        "val": len(episodes) * float(val_fraction),
    }
    split_objects: dict[str, list[str]] = {"test": [], "val": [], "train": []}
    counts = {name: 0 for name in split_objects}
    for name in ("test", "val"):
        while objects and counts[name] < targets[name]:
            object_name = objects.pop()
            split_objects[name].append(object_name)
            counts[name] += len(groups[object_name])
    split_objects["train"] = objects
    splits = {
        name: [episode for object_name in names for episode in groups[object_name]]
        for name, names in split_objects.items()
    }
    return splits, split_objects


def _sha256_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        try:
            label = path.relative_to(root)
        except ValueError:
            label = path
        digest.update(str(label).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _select_episodes(root: Path, count: int, seed: int, robot_types: tuple[str, ...] = ("inspire_f1",)) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    io = _load_hrdex_io(root.parent)
    for kind in robot_types:
        if kind not in ROBOT_TYPES:
            raise ValueError(f"Unsupported robot type: {kind}")
    episodes = [episode for kind in robot_types for episode in sorted((root / kind).glob("*/*"))]
    for episode in episodes:
        if not episode.is_dir():
            continue
        kind = _episode_kind(root, episode)
        object_name = episode.parent.name
        mesh_exists = any((root / mesh_root / object_name / f"{object_name}.obj").exists() for mesh_root in ("assets/mesh_v2", "assets/mesh"))
        try:
            _resolve_pose_sources(root, episode)
            pose_exists = True
        except FileNotFoundError:
            pose_exists = False
        if kind == "human":
            hand_exists = any((episode / candidate).is_dir() and any((episode / candidate).glob("*.obj")) for candidate in ("hand/mano", "mano", "hand/mano/mano"))
            valid = hand_exists
        else:
            required = (episode / "raw" / "arm" / "time.npy", episode / "C2R.npy")
            hand_dir = episode / "raw" / "hand"
            hand_exists = any((hand_dir / name).exists() for name in ("right_joint_states.npy", "right_commands.npy", "position.npy", "action.npy"))
            required_files_exist = all(path.is_file() for path in required)
            try:
                # This catches layout/shape changes before workers are launched.
                io.load_robot_qpos(episode, {"inspire_dftp": "inspire", "allegro_v5": "allegro_v5"}.get(kind, kind))
                q_valid = True
            except Exception:
                q_valid = False
            valid = hand_exists and required_files_exist and q_valid
        if mesh_exists and pose_exists and valid:
            groups.setdefault(episode.parent.name, []).append(episode)
    if count <= 0:
        count = sum(len(values) for values in groups.values())
    rng = random.Random(seed)
    objects = sorted(groups)
    rng.shuffle(objects)
    for episodes in groups.values():
        rng.shuffle(episodes)
    selected: list[Path] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for object_name in objects:
            episodes = groups[object_name]
            if round_index < len(episodes):
                selected.append(episodes[round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    if len(selected) < count:
        raise ValueError(f"Requested {count} episodes, only found {len(selected)}")
    return selected[:count]


def _save_arrays(root: Path, arrays: dict[str, np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", np.asarray(value), allow_pickle=False)


def _episode_sources(root: Path, episode: Path, robot_urdf: Path | None) -> tuple[list[Path], str, Path, list[Path]]:
    poses, pose_source = _resolve_pose_sources(root, episode)
    kind = _episode_kind(root, episode)
    object_name = episode.parent.name
    mesh_path = root / "assets" / "mesh_v2" / object_name / f"{object_name}.obj"
    if not mesh_path.exists():
        mesh_path = root / "assets" / "mesh" / object_name / f"{object_name}.obj"
    timestamp_path = episode / "raw" / "timestamps" / "timestamp.npy"
    sources = [mesh_path, episode / "C2R.npy", timestamp_path,
               episode / "raw" / "timestamps" / "frame_id.npy",
               episode / "raw" / "arm" / "position.npy", episode / "raw" / "arm" / "time.npy"]
    if robot_urdf is not None:
        sources.append(robot_urdf)
    hand_dir = episode / "raw" / "hand"
    if kind == "human":
        sources.extend(sorted((episode / "hand" / "mano").glob("*.obj")))
        sources.extend(sorted((episode / "hand" / "mano_params").glob("*.json")))
    else:
        sources.extend(path for path in hand_dir.glob("*.npy") if path.is_file())
    sources.extend(poses)
    return poses, pose_source, mesh_path, [path for path in sources if path.exists()]


def _build_one(payload: dict) -> dict:
    started = time.time()
    root = Path(payload["dataset_root"])
    episode = root / payload["episode"]
    cache_root = Path(payload["cache_root"])
    kind = _episode_kind(root, episode)
    robot_urdf_value = payload.get("robot_urdf")
    robot_urdf = Path(robot_urdf_value) if robot_urdf_value else None
    seed = int(payload["seed"])
    num_hand = int(payload["num_hand_points"])
    num_obj = int(payload["num_obj_points"])
    num_obj_pool = int(payload.get("num_obj_pool", 4096))
    candidate_threshold_m = float(payload.get("candidate_threshold_m", 0.05))
    if num_obj_pool < num_obj:
        raise ValueError(f"num_obj_pool={num_obj_pool} must be >= num_obj_points={num_obj}")
    if candidate_threshold_m <= 0.0:
        raise ValueError("candidate_threshold_m must be positive")
    episode_id = hashlib.sha1(payload["episode"].encode()).hexdigest()[:16]
    output = cache_root / "v4" / "episodes" / episode_id
    complete = output / "task" / "manifest.json"
    poses, pose_source, mesh_path, sources = _episode_sources(root, episode, robot_urdf)
    source_fingerprint = _sha256_files(sources, root.parent)
    implementation_fingerprint = _sha256_files(
        [Path(__file__), Path(__file__).with_name("dataset.py")], Path(__file__).parents[3]
    )
    if complete.is_file():
        existing = json.loads(complete.read_text(encoding="utf-8"))
        implementation_matches = (
            existing.get("implementation_sha256") == implementation_fingerprint
            or bool(payload.get("reuse_schema_cache", False))
        )
        matches = (existing.get("schema") == SCHEMA and existing.get("source_sha256") == source_fingerprint
                   and implementation_matches
                   and existing.get("num_hand_points") == num_hand and existing.get("num_obj_points") == num_obj
                   and existing.get("num_obj_pool", num_obj) == num_obj_pool
                   and float(existing.get("candidate_threshold_m", candidate_threshold_m)) == candidate_threshold_m
                   and existing.get("sampling_seed") == seed)
        if matches:
            return {"episode": payload["episode"], "cache_dir": str(output), "status": "cached", "seconds": time.time() - started}

    io = _load_hrdex_io(root.parent)
    pose_seq = _load_pose_sequence(poses)
    timestamp_path = episode / "raw" / "timestamps" / "timestamp.npy"
    frame_id_path = episode / "raw" / "timestamps" / "frame_id.npy"
    video_times = None
    if timestamp_path.exists():
        video_times = np.asarray(np.load(timestamp_path, allow_pickle=True), dtype=np.float64).reshape(-1)
    if video_times is not None and len(video_times):
        source_video_index = np.linspace(0, len(video_times) - 1, len(pose_seq)).round().astype(np.int64)
        frame_times = video_times[source_video_index]
        if frame_id_path.exists():
            video_frame_ids = np.asarray(np.load(frame_id_path, allow_pickle=True), dtype=np.int64).reshape(-1)
            source_frame_id = video_frame_ids[np.minimum(source_video_index, len(video_frame_ids) - 1)]
        else:
            source_frame_id = source_video_index + 1
    else:
        source_video_index = np.arange(len(pose_seq), dtype=np.int64)
        frame_times = np.arange(len(pose_seq), dtype=np.float64) / 30.0
        source_frame_id = source_video_index

    c2r_path = episode / "C2R.npy"
    c2r = np.asarray(np.load(c2r_path, allow_pickle=False), dtype=np.float32) if c2r_path.exists() else np.eye(4, dtype=np.float32)
    object_name = episode.parent.name
    import trimesh
    object_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    object_vertices = np.asarray(object_mesh.vertices)
    object_faces = np.asarray(object_mesh.faces)
    object_seed = seed + zlib.crc32(object_name.encode()) % 100000
    obj_face_pool, obj_bary_pool = _surface_spec(object_vertices, object_faces, num_obj_pool, object_seed)
    obj_local_pool, obj_local_normals_pool = _eval_surface(object_vertices, object_faces, obj_face_pool, obj_bary_pool)
    # Keep the legacy 512-point task layer for CmDecoder compatibility.  Cm
    # itself consumes the larger stable pool plus the per-frame candidate mask
    # written below.
    obj_face, obj_bary = _surface_spec(object_vertices, object_faces, num_obj, object_seed)
    obj_local, obj_local_normals = _eval_surface(object_vertices, object_faces, obj_face, obj_bary)

    if kind == "human":
        mano_vertices, mano_faces, mano_ids, mano_dir = io.load_human_mano_sequence(episode)
        if int(num_hand) != len(mano_faces):
            raise ValueError(
                f"MANO provides {len(mano_faces)} fixed face points, but num_hand_points={num_hand}; "
                "use 1538 to preserve Cm correspondence"
            )
        if len(mano_vertices) != len(pose_seq):
            # Official compact poses and MANO OBJ names are both frame-indexed;
            # use numeric ids whenever a subset was exported, otherwise resample.
            pose_indices = np.clip(mano_ids, 0, len(pose_seq) - 1)
            pose_seq = pose_seq[pose_indices]
            frame_times = frame_times[np.clip(pose_indices, 0, len(frame_times) - 1)]
            source_video_index = source_video_index[np.clip(pose_indices, 0, len(source_video_index) - 1)]
            source_frame_id = mano_ids.astype(np.int64)
        else:
            source_frame_id = mano_ids.astype(np.int64)
        hand_face = np.arange(len(mano_faces), dtype=np.int64)
        hand_bary = np.full((len(hand_face), 3), 1.0 / 3.0, dtype=np.float32)
        hand_points_local = np.zeros((len(hand_face), 3), dtype=np.float32)
        hand_point_link_index = np.zeros(len(hand_face), dtype=np.int16)
        hand_binding_groups = ["mano"]
        link_order = ["mano"]
        hand_world, hand_normals_world, wrist_world = [], [], []
        params_dir = episode / "hand" / "mano_params"
        for frame, vertices in enumerate(mano_vertices):
            hp, hn = _eval_surface(vertices, mano_faces, hand_face, hand_bary)
            param_path = params_dir / f"{int(source_frame_id[frame]):05d}.json"
            wrist = np.eye(4, dtype=np.float32)
            if param_path.is_file():
                with param_path.open("r", encoding="utf-8") as handle:
                    params = json.load(handle)
                joints = np.asarray(params.get("joints", []), dtype=np.float32).reshape(-1, 3)
                orient = np.asarray(params.get("global_orient", []), dtype=np.float32).reshape(-1, 3, 3)
                if len(joints):
                    wrist[:3, 3] = joints[0]
                if len(orient):
                    wrist[:3, :3] = orient[0]
            hand_world.append(hp)
            hand_normals_world.append(hn)
            wrist_world.append(wrist)
        hand_world = np.asarray(hand_world, dtype=np.float32)
        hand_normals_world = np.asarray(hand_normals_world, dtype=np.float32)
        wrist_world = np.asarray(wrist_world, dtype=np.float32)
        q_full = np.zeros((len(hand_world), 12), dtype=np.float32)
        q_semantics = "unavailable_mano"
        hand_binding_semantics = "mano_face_center_v1"
    else:
        hand_name = {"inspire_dftp": "inspire", "allegro_v5": "allegro_v5"}.get(kind, kind)
        q_raw, robot_times = io.load_robot_qpos(episode, hand_name)
        robot_times = np.asarray(robot_times, dtype=np.float64)
        if video_times is None or not len(video_times):
            frame_times = np.arange(len(pose_seq), dtype=np.float64) / 30.0 + robot_times[0]
        q_full = np.stack([np.interp(frame_times, robot_times, q_raw[:, j]) for j in range(q_raw.shape[1])], axis=1).astype(np.float32)
        c2r = np.asarray(np.load(episode / "C2R.npy", allow_pickle=False), dtype=np.float32)
        urdf = io.parse_urdf(robot_urdf)
        mesh_cache: dict = {}
        first_vertices, first_faces = _robot_hand_mesh(io, urdf, q_full[0], mesh_cache)
        hand_face, hand_bary = _surface_spec(first_vertices, first_faces, num_hand, seed + 991)
        hand_point_group, hand_points_local, hand_binding_groups = _robot_hand_binding(io, urdf, hand_face, hand_bary, mesh_cache)
        link_order = _urdf_link_order(urdf)
        link_to_index = {name: index for index, name in enumerate(link_order)}
        hand_point_link_index = np.asarray([link_to_index[hand_binding_groups[int(group)]] for group in hand_point_group], dtype=np.int16)
        hand_world, hand_normals_world, wrist_world = [], [], []
        for frame in range(len(q_full)):
            vertices, faces = _robot_hand_mesh(io, urdf, q_full[frame], mesh_cache)
            hp, hn = _eval_surface(vertices, faces, hand_face, hand_bary)
            hand_world.append(_to_world(hp, c2r))
            hand_normals_world.append(hn @ c2r[:3, :3].T)
            wrist_robot = io.compute_link_transforms(urdf, q_full[frame]).get("base_link", np.eye(4))
            wrist_world.append(c2r @ wrist_robot)
        hand_world = np.asarray(hand_world, dtype=np.float32)
        hand_normals_world = np.asarray(hand_normals_world, dtype=np.float32)
        wrist_world = np.asarray(wrist_world, dtype=np.float32)
        q_semantics = f"{kind}_arm6_hand{q_full.shape[1] - 6}"
        hand_binding_semantics = "link_index_and_link_local_point_v1"
    object_world, object_normals_world = [], []
    object_pool_world, object_pool_normals_world = [], []
    for frame in range(len(q_full)):
        object_world.append(obj_local @ pose_seq[frame, :3, :3].T + pose_seq[frame, :3, 3])
        object_normals_world.append(obj_local_normals @ pose_seq[frame, :3, :3].T)
        object_pool_world.append(obj_local_pool @ pose_seq[frame, :3, :3].T + pose_seq[frame, :3, 3])
        object_pool_normals_world.append(obj_local_normals_pool @ pose_seq[frame, :3, :3].T)
    hand_world = np.asarray(hand_world, dtype=np.float32)
    hand_normals_world = np.asarray(hand_normals_world, dtype=np.float32)
    object_world = np.asarray(object_world, dtype=np.float32)
    object_normals_world = np.asarray(object_normals_world, dtype=np.float32)
    object_pool_world = np.asarray(object_pool_world, dtype=np.float32)
    object_pool_normals_world = np.asarray(object_pool_normals_world, dtype=np.float32)
    wrist_world = np.asarray(wrist_world, dtype=np.float32)

    # Candidate membership is invariant to the current wrist transform, so it
    # can be computed directly in the HRDexDB world frame.  Keep it aligned to
    # the stable object pool; the Cm loader samples from this mask at runtime.
    candidate_mask = np.empty((len(q_full), num_obj_pool), dtype=np.bool_)
    for frame in range(len(q_full)):
        # Query the exact Euclidean radius with a KD-tree instead of materializing
        # a (num_obj_pool, num_hand_points, 3) distance tensor per frame.  This
        # keeps the candidate semantics unchanged while cutting both CPU work and
        # peak memory substantially for the 4096 x 1538 pool/hand contract.
        candidate_mask[frame] = (
            cKDTree(hand_world[frame]).query_ball_point(
                object_pool_world[frame], r=candidate_threshold_m, return_length=True
            )
            > 0
        )

    geometry_arrays = {
        "frame_time": frame_times,
        "source_video_index": source_video_index,
        "source_frame_id": source_frame_id,
        "q_full": q_full,
        "wrist_pose_world": wrist_world,
        "hand_points_world": hand_world,
        "hand_normals_world": hand_normals_world,
        "hand_point_link_index": hand_point_link_index,
        "hand_points_local": hand_points_local,
        "obj_points_world": object_world,
        "obj_normals_world": object_normals_world,
        "obj_points_pool_world": object_pool_world,
        "obj_normals_pool_world": object_pool_normals_world,
        "obj_candidate_mask_5cm": candidate_mask,
    }
    _save_arrays(output / "geometry", geometry_arrays)

    task = {name: [] for name in ("hand_points", "hand_normals", "hand_flow", "obj_points", "obj_normals")}
    for frame in range(len(q_full) - 1):
        wrist = wrist_world[frame]
        hp = _to_frame(hand_world[frame], wrist)
        hp_next = _to_frame(hand_world[frame + 1], wrist)
        task["hand_points"].append(hp)
        task["hand_normals"].append(_rotate_to_frame(hand_normals_world[frame], wrist))
        task["hand_flow"].append(hp_next - hp)
        task["obj_points"].append(_to_frame(object_world[frame], wrist))
        task["obj_normals"].append(_rotate_to_frame(object_normals_world[frame], wrist))
    task_arrays = {name: np.asarray(values, dtype=np.float32) for name, values in task.items()}
    task_arrays.update({
        "obj_valid_mask": np.ones((len(q_full) - 1, num_obj), dtype=np.bool_),
        "q_t": q_full[:-1, 6:].astype(np.float32),
        "q_next": q_full[1:, 6:].astype(np.float32),
        "delta_time_s": np.diff(frame_times).astype(np.float32),
        "source_frame_delta": np.diff(source_frame_id).astype(np.int64),
        "is_30hz_pair": (np.diff(source_frame_id) == 1),
        "q_delta_abs_max": np.max(np.abs(q_full[1:, 6:] - q_full[:-1, 6:]), axis=1).astype(np.float32),
    })
    _save_arrays(output / "task", task_arrays)

    dt = np.diff(frame_times)
    common = {"schema": SCHEMA, "episode": payload["episode"], "robot_type": kind,
              "q_semantics": q_semantics, "source_sha256": source_fingerprint,
              "implementation_sha256": implementation_fingerprint,
              "frames": len(q_full), "num_hand_points": num_hand, "num_obj_points": num_obj,
              "num_obj_pool": num_obj_pool, "candidate_threshold_m": candidate_threshold_m,
              "sampling_seed": seed, "frame_mapping_strategy": "pose_index_to_nearest_normalized_video_index",
              "object_pose_source": pose_source,
              "video_frames": int(len(video_times)) if timestamp_path.exists() else None,
              "pose_frames": int(len(pose_seq)), "video_missing_frame_count": int(np.maximum(np.diff(source_frame_id) - 1, 0).sum()),
              "delta_time_median_s": float(np.median(dt)), "delta_time_max_s": float(np.max(dt)),
              "hand_binding_link_order": link_order,
              "hand_binding_groups": hand_binding_groups,
              "hand_binding_semantics": hand_binding_semantics}
    (output / "geometry" / "manifest.json").write_text(json.dumps({**common, "layer": "geometry", "fields": sorted(geometry_arrays)}, indent=2), encoding="utf-8")
    (output / "task" / "manifest.json").write_text(json.dumps({**common, "layer": "cmdecoder_task", "samples": len(q_full) - 1, "fields": sorted(task_arrays)}, indent=2), encoding="utf-8")
    return {"episode": payload["episode"], "cache_dir": str(output), "status": "built", "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=HRDEXDB_ROOT / "v0_nonvideo")
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1"))
    parser.add_argument("--robot-urdf", type=Path, default=HRDEXDB_ROOT / "assets" / "robots" / "xarm_inspire_f1_right.urdf")
    parser.add_argument("--robot-urdf-dir", type=Path, default=HRDEXDB_ROOT / "assets" / "robots")
    parser.add_argument("--robot-types", type=str, default="inspire_f1", help="comma-separated: human,allegro_v5,inspire_dftp,inspire_f1")
    parser.add_argument("--episodes", type=int, default=50, help="0 means all valid episodes")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-hand-points", type=int, default=1538)
    parser.add_argument("--num-obj-points", type=int, default=512)
    parser.add_argument("--num-obj-pool", type=int, default=4096)
    parser.add_argument("--candidate-threshold-m", type=float, default=0.05)
    parser.add_argument(
        "--reuse-schema-cache",
        action="store_true",
        help="Reuse complete cmdecoder_layered_v4 episodes when only the builder selector changed.",
    )
    parser.add_argument("--split-mode", choices=("episode_random", "object_disjoint"), default="episode_random")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if args.val_fraction <= 0 or args.test_fraction <= 0 or args.val_fraction + args.test_fraction >= 1:
        raise ValueError("val/test fractions must be positive and sum to less than 1")
    robot_types = tuple(item.strip() for item in args.robot_types.split(",") if item.strip())
    selected = _select_episodes(args.dataset_root, args.episodes, args.seed, robot_types)
    relative = [str(path.relative_to(args.dataset_root)) for path in selected]
    split_objects = None
    if args.split_mode == "object_disjoint":
        splits, split_objects = _split_object_disjoint(
            relative, seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction
        )
    else:
        split_rng = random.Random(args.seed + 1)
        split_rng.shuffle(relative)
        n_test = max(1, round(len(relative) * args.test_fraction))
        n_val = max(1, round(len(relative) * args.val_fraction))
        splits = {"test": relative[:n_test], "val": relative[n_test:n_test + n_val], "train": relative[n_test + n_val:]}
    urdf_map = {
        "inspire_f1": args.robot_urdf,
        "inspire_dftp": args.robot_urdf_dir / "xarm_inspire_DFTP.urdf",
        "allegro_v5": args.robot_urdf_dir / "allegro_v5" / "xarm_allegro_v5.urdf",
    }
    payloads = [{"dataset_root": str(args.dataset_root.resolve()), "cache_root": str(args.cache_root.resolve()),
                 "robot_urdf": None if episode.split("/", 1)[0] == "human" else str(urdf_map[episode.split("/", 1)[0]].resolve()),
                 "episode": episode, "seed": args.seed,
                 "num_hand_points": args.num_hand_points, "num_obj_points": args.num_obj_points,
                 "num_obj_pool": args.num_obj_pool, "candidate_threshold_m": args.candidate_threshold_m,
                 "reuse_schema_cache": args.reuse_schema_cache}
                for episode in relative]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_one, payload): payload["episode"] for payload in payloads}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result(); results.append(result)
            print(f"[{done:03d}/{len(futures):03d}] {result['status']} {result['episode']} {result['seconds']:.1f}s", flush=True)
    by_episode = {result["episode"]: str(Path(result["cache_dir"]).relative_to(args.cache_root.resolve())) for result in results}
    manifest = {"schema": SCHEMA, "selection": "object_stratified", "robot_types": list(robot_types),
                "split_mode": args.split_mode,
                "val_fraction": args.val_fraction, "test_fraction": args.test_fraction, "seed": args.seed,
                "episode_count": len(relative), "splits": splits, "cache_dirs": by_episode}
    if split_objects is not None:
        manifest["split_objects"] = split_objects
    selection_label = "all" if args.episodes <= 0 else str(args.episodes)
    split_label = "_object_disjoint" if args.split_mode == "object_disjoint" else ""
    manifest_path = args.cache_root / "v4" / f"selection_{selection_label}{split_label}_seed{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
