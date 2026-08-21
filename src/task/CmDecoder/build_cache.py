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

from src.task.CmDecoder.dataset import (
    _eval_surface,
    _load_hrdex_io,
    _load_pose,
    _robot_hand_mesh,
    _rotate_to_frame,
    _surface_spec,
    _to_frame,
    _to_world,
)


SCHEMA = "cmdecoder_layered_v4"


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


def _select_episodes(root: Path, count: int, seed: int) -> list[Path]:
    groups: dict[str, list[Path]] = {}
    for episode in sorted((root / "inspire_f1").glob("*/*")):
        object_name = episode.parent.name
        mesh_exists = any((root / mesh_root / object_name / f"{object_name}.obj").exists() for mesh_root in ("assets/mesh_v2", "assets/mesh"))
        pose_exists = any((episode / pose_root).is_dir() and next((episode / pose_root).glob("pose_*.txt"), None) is not None for pose_root in ("object_6d", "object_6d_pose_v2"))
        required = (episode / "raw" / "arm" / "position.npy", episode / "raw" / "arm" / "time.npy", episode / "C2R.npy")
        hand_dir = episode / "raw" / "hand"
        hand_exists = (hand_dir / "right_joint_states.npy").exists() or (hand_dir / "right_commands.npy").exists()
        if episode.is_dir() and mesh_exists and pose_exists and all(path.exists() for path in required) and hand_exists:
            groups.setdefault(episode.parent.name, []).append(episode)
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
    return selected


def _save_arrays(root: Path, arrays: dict[str, np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", np.asarray(value), allow_pickle=False)


def _episode_sources(root: Path, episode: Path, robot_urdf: Path) -> tuple[list[Path], Path, list[Path]]:
    poses = sorted((episode / "object_6d").glob("pose_*.txt"))
    if not poses:
        poses = sorted((episode / "object_6d_pose_v2").glob("pose_*.txt"))
    if not poses:
        raise FileNotFoundError(f"No object poses: {episode}")
    object_name = episode.parent.name
    mesh_path = root / "assets" / "mesh_v2" / object_name / f"{object_name}.obj"
    if not mesh_path.exists():
        mesh_path = root / "assets" / "mesh" / object_name / f"{object_name}.obj"
    timestamp_path = episode / "raw" / "timestamps" / "timestamp.npy"
    sources = [robot_urdf, mesh_path, episode / "C2R.npy", timestamp_path,
               episode / "raw" / "timestamps" / "frame_id.npy",
               episode / "raw" / "arm" / "position.npy", episode / "raw" / "arm" / "time.npy"]
    hand_dir = episode / "raw" / "hand"
    sources.extend(path for path in (hand_dir / "right_joint_states.npy", hand_dir / "right_joint_states_time.npy", hand_dir / "right_commands.npy", hand_dir / "right_commands_time.npy") if path.exists())
    sources.extend(poses)
    return poses, mesh_path, [path for path in sources if path.exists()]


def _build_one(payload: dict) -> dict:
    started = time.time()
    root = Path(payload["dataset_root"])
    episode = root / payload["episode"]
    cache_root = Path(payload["cache_root"])
    robot_urdf = Path(payload["robot_urdf"])
    seed = int(payload["seed"])
    num_hand = int(payload["num_hand_points"])
    num_obj = int(payload["num_obj_points"])
    episode_id = hashlib.sha1(payload["episode"].encode()).hexdigest()[:16]
    output = cache_root / "v4" / "episodes" / episode_id
    complete = output / "task" / "manifest.json"
    poses, mesh_path, sources = _episode_sources(root, episode, robot_urdf)
    source_fingerprint = _sha256_files(sources, root.parent)
    implementation_fingerprint = _sha256_files(
        [Path(__file__), Path(__file__).with_name("dataset.py")], Path(__file__).parents[3]
    )
    if complete.is_file():
        existing = json.loads(complete.read_text(encoding="utf-8"))
        matches = (existing.get("schema") == SCHEMA and existing.get("source_sha256") == source_fingerprint
                   and existing.get("implementation_sha256") == implementation_fingerprint
                   and existing.get("num_hand_points") == num_hand and existing.get("num_obj_points") == num_obj
                   and existing.get("sampling_seed") == seed)
        if matches:
            return {"episode": payload["episode"], "cache_dir": str(output), "status": "cached", "seconds": time.time() - started}

    io = _load_hrdex_io(root.parent)
    pose_seq = np.stack([_load_pose(path) for path in poses]).astype(np.float32)
    q_raw, robot_times = io.load_robot_qpos(episode, "inspire_f1")
    robot_times = np.asarray(robot_times, dtype=np.float64)
    timestamp_path = episode / "raw" / "timestamps" / "timestamp.npy"
    frame_id_path = episode / "raw" / "timestamps" / "frame_id.npy"
    if timestamp_path.exists():
        video_times = np.asarray(np.load(timestamp_path, allow_pickle=True), dtype=np.float64).reshape(-1)
        source_video_index = np.linspace(0, len(video_times) - 1, len(pose_seq)).round().astype(np.int64)
        frame_times = video_times[source_video_index]
        q_full = np.stack([np.interp(frame_times, robot_times, q_raw[:, j]) for j in range(12)], axis=1).astype(np.float32)
        if frame_id_path.exists():
            video_frame_ids = np.asarray(np.load(frame_id_path, allow_pickle=True), dtype=np.int64).reshape(-1)
            source_frame_id = video_frame_ids[np.minimum(source_video_index, len(video_frame_ids) - 1)]
        else:
            source_frame_id = source_video_index + 1
    else:
        frame_times = np.arange(len(pose_seq), dtype=np.float64) / 30.0 + robot_times[0]
        q_full = np.stack([np.interp(frame_times, robot_times, q_raw[:, j]) for j in range(12)], axis=1).astype(np.float32)
        source_video_index = np.arange(len(pose_seq), dtype=np.int64)
        source_frame_id = source_video_index + 1

    c2r_path = episode / "C2R.npy"
    c2r = np.asarray(np.load(c2r_path, allow_pickle=False), dtype=np.float32) if c2r_path.exists() else np.eye(4, dtype=np.float32)
    object_name = episode.parent.name
    import trimesh
    object_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    object_vertices = np.asarray(object_mesh.vertices)
    object_faces = np.asarray(object_mesh.faces)
    obj_face, obj_bary = _surface_spec(object_vertices, object_faces, num_obj, seed + zlib.crc32(object_name.encode()) % 100000)
    obj_local, obj_local_normals = _eval_surface(object_vertices, object_faces, obj_face, obj_bary)

    urdf = io.parse_urdf(robot_urdf)
    mesh_cache: dict = {}
    first_vertices, first_faces = _robot_hand_mesh(io, urdf, q_full[0], mesh_cache)
    hand_face, hand_bary = _surface_spec(first_vertices, first_faces, num_hand, seed + 991)
    hand_world, hand_normals_world, object_world, object_normals_world, wrist_world = [], [], [], [], []
    for frame in range(len(q_full)):
        vertices, faces = _robot_hand_mesh(io, urdf, q_full[frame], mesh_cache)
        hp, hn = _eval_surface(vertices, faces, hand_face, hand_bary)
        hand_world.append(_to_world(hp, c2r))
        hand_normals_world.append(hn @ c2r[:3, :3].T)
        object_world.append(obj_local @ pose_seq[frame, :3, :3].T + pose_seq[frame, :3, 3])
        object_normals_world.append(obj_local_normals @ pose_seq[frame, :3, :3].T)
        wrist_robot = io.compute_link_transforms(urdf, q_full[frame]).get("base_link", np.eye(4))
        wrist_world.append(c2r @ wrist_robot)
    hand_world = np.asarray(hand_world, dtype=np.float32)
    hand_normals_world = np.asarray(hand_normals_world, dtype=np.float32)
    object_world = np.asarray(object_world, dtype=np.float32)
    object_normals_world = np.asarray(object_normals_world, dtype=np.float32)
    wrist_world = np.asarray(wrist_world, dtype=np.float32)

    geometry_arrays = {
        "frame_time": frame_times,
        "source_video_index": source_video_index,
        "source_frame_id": source_frame_id,
        "q_full": q_full,
        "wrist_pose_world": wrist_world,
        "hand_points_world": hand_world,
        "hand_normals_world": hand_normals_world,
        "obj_points_world": object_world,
        "obj_normals_world": object_normals_world,
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
    common = {"schema": SCHEMA, "episode": payload["episode"], "source_sha256": source_fingerprint,
              "implementation_sha256": implementation_fingerprint,
              "frames": len(q_full), "num_hand_points": num_hand, "num_obj_points": num_obj,
              "sampling_seed": seed, "frame_mapping_strategy": "pose_index_to_nearest_normalized_video_index",
              "video_frames": int(len(video_times)) if timestamp_path.exists() else None,
              "pose_frames": int(len(pose_seq)), "video_missing_frame_count": int(np.maximum(np.diff(source_frame_id) - 1, 0).sum()),
              "delta_time_median_s": float(np.median(dt)), "delta_time_max_s": float(np.max(dt))}
    (output / "geometry" / "manifest.json").write_text(json.dumps({**common, "layer": "geometry", "fields": sorted(geometry_arrays)}, indent=2), encoding="utf-8")
    (output / "task" / "manifest.json").write_text(json.dumps({**common, "layer": "cmdecoder_task", "samples": len(q_full) - 1, "fields": sorted(task_arrays)}, indent=2), encoding="utf-8")
    return {"episode": payload["episode"], "cache_dir": str(output), "status": "built", "seconds": time.time() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/home2/wyy/oyx_ws/HRDexDB/v0"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/processed_data/cm_decoder/hrdexdb_inspire_f1"))
    parser.add_argument("--robot-urdf", type=Path, default=Path("/home2/wyy/oyx_ws/HRDexDB/assets/robots/xarm_inspire_f1_right.urdf"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-hand-points", type=int, default=1538)
    parser.add_argument("--num-obj-points", type=int, default=512)
    args = parser.parse_args()
    selected = _select_episodes(args.dataset_root, args.episodes, args.seed)
    relative = [str(path.relative_to(args.dataset_root)) for path in selected]
    split_rng = random.Random(args.seed + 1)
    split_rng.shuffle(relative)
    n_test = max(1, round(len(relative) * 0.1)); n_val = max(1, round(len(relative) * 0.1))
    splits = {"test": relative[:n_test], "val": relative[n_test:n_test + n_val], "train": relative[n_test + n_val:]}
    payloads = [{"dataset_root": str(args.dataset_root.resolve()), "cache_root": str(args.cache_root.resolve()),
                 "robot_urdf": str(args.robot_urdf.resolve()), "episode": episode, "seed": args.seed,
                 "num_hand_points": args.num_hand_points, "num_obj_points": args.num_obj_points}
                for episode in relative]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_build_one, payload): payload["episode"] for payload in payloads}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result(); results.append(result)
            print(f"[{done:03d}/{len(futures):03d}] {result['status']} {result['episode']} {result['seconds']:.1f}s", flush=True)
    by_episode = {result["episode"]: str(Path(result["cache_dir"]).relative_to(args.cache_root.resolve())) for result in results}
    manifest = {"schema": SCHEMA, "selection": "object_stratified", "seed": args.seed,
                "episode_count": len(relative), "splits": splits, "cache_dirs": by_episode}
    manifest_path = args.cache_root / "v4" / f"selection_{args.episodes}_seed{args.seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
