from __future__ import annotations

import importlib.util
import sys
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset


def _load_hrdex_io(root: Path):
    path = root / "hrdexdb_contact_heatmaps" / "hrdexdb_io.py"
    spec = importlib.util.spec_from_file_location("hrdexdb_io_runtime", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load HRDexDB geometry helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _inspire_f1_q(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    q = np.zeros((len(raw), 6), dtype=np.float32)
    q[:, 0] = (1800.0 - raw[:, 0]) * np.pi / 1800.0
    q[:, 1] = (1350.0 - raw[:, 1]) * np.pi / 1800.0
    q[:, 2:] = (1740.0 - raw[:, 2:]) * np.pi / 1800.0
    return q


def _load_pose(path: Path) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float32)
    value = value.reshape(4, 4) if value.size == 16 else value
    if value.shape != (4, 4):
        raise ValueError(f"Invalid object pose: {path}")
    return value


def _surface_spec(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int):
    rng = np.random.default_rng(seed)
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    probs = areas / max(float(areas.sum()), 1e-12)
    face = rng.choice(len(faces), size=count, replace=True, p=probs)
    uv = rng.random((count, 2), dtype=np.float32)
    over = (uv[:, 0] + uv[:, 1]) > 1.0
    uv[over] = 1.0 - uv[over]
    bary = np.concatenate([1.0 - uv.sum(1, keepdims=True), uv], axis=1).astype(np.float32)
    return face.astype(np.int64), bary


def _eval_surface(vertices: np.ndarray, faces: np.ndarray, face: np.ndarray, bary: np.ndarray):
    tri = vertices[faces[face]]
    points = (tri * bary[:, :, None]).sum(1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-8)
    return points.astype(np.float32), normals.astype(np.float32)


def _robot_hand_mesh(io: Any, urdf: Any, qpos: np.ndarray, mesh_cache: dict):
    link_tfs = io.compute_link_transforms(urdf, qpos)
    vertices, faces = [], []
    offset = 0
    for visual in urdf.visuals:
        if io.is_arm_visual(visual):
            continue
        key = (visual.mesh_path.resolve(), tuple(float(x) for x in visual.scale), visual.origin.tobytes())
        if key not in mesh_cache:
            mesh_cache[key] = io.transformed_mesh(io.load_mesh(visual.mesh_path), visual.origin, visual.scale)
        local = mesh_cache[key]
        value = io.apply_transform(local.vertices, link_tfs.get(visual.link, np.eye(4)))
        vertices.append(np.asarray(value, dtype=np.float32))
        faces.append(np.asarray(local.indices, dtype=np.int64) + offset)
        offset += len(value)
    if not vertices:
        raise RuntimeError("No hand visual mesh found in Inspire F1 URDF")
    return np.concatenate(vertices), np.concatenate(faces)


def _robot_hand_surface_mesh(io: Any, urdf: Any, mesh_cache: dict):
    """Return static link-local visual mesh and the link owning each face."""
    vertices, faces, face_links = [], [], []
    offset = 0
    for visual in urdf.visuals:
        if io.is_arm_visual(visual):
            continue
        key = (visual.mesh_path.resolve(), tuple(float(x) for x in visual.scale), visual.origin.tobytes())
        if key not in mesh_cache:
            mesh_cache[key] = io.transformed_mesh(io.load_mesh(visual.mesh_path), visual.origin, visual.scale)
        local = mesh_cache[key]
        value = np.asarray(local.vertices, dtype=np.float32)
        indices = np.asarray(local.indices, dtype=np.int64)
        vertices.append(value)
        faces.append(indices + offset)
        face_links.extend([str(visual.link)] * len(indices))
        offset += len(value)
    if not vertices:
        raise RuntimeError("No hand visual mesh found in Inspire F1 URDF")
    return np.concatenate(vertices), np.concatenate(faces), np.asarray(face_links, dtype=object)


def _robot_hand_binding(io: Any, urdf: Any, face: np.ndarray, bary: np.ndarray, mesh_cache: dict):
    """Recover static link groups and link-local sampled points for a surface spec.

    The surface spec is sampled on the concatenated hand mesh.  Each visual is
    already expressed in its link-local frame (including URDF visual origin and
    scale), so the selected face/barycentric samples can be stored once and
    reused for every q/frame.
    """
    vertices, faces, face_links = [], [], []
    offset = 0
    for visual in urdf.visuals:
        if io.is_arm_visual(visual):
            continue
        key = (visual.mesh_path.resolve(), tuple(float(x) for x in visual.scale), visual.origin.tobytes())
        if key not in mesh_cache:
            mesh_cache[key] = io.transformed_mesh(io.load_mesh(visual.mesh_path), visual.origin, visual.scale)
        local = mesh_cache[key]
        value = np.asarray(local.vertices, dtype=np.float32)
        indices = np.asarray(local.indices, dtype=np.int64)
        vertices.append(value)
        faces.append(indices + offset)
        face_links.extend([str(visual.link)] * len(indices))
        offset += len(value)
    all_vertices = np.concatenate(vertices)
    all_faces = np.concatenate(faces)
    selected_links = np.asarray(face_links, dtype=object)[np.asarray(face, dtype=np.int64)]
    group_names = list(dict.fromkeys(selected_links.tolist()))
    group_index = np.asarray([group_names.index(name) for name in selected_links], dtype=np.int16)
    triangles = all_vertices[all_faces[np.asarray(face, dtype=np.int64)]]
    local_points = (triangles * np.asarray(bary, dtype=np.float32)[:, :, None]).sum(axis=1).astype(np.float32)
    return group_index, local_points, group_names


def _urdf_link_order(urdf: Any) -> list[str]:
    """Return a deterministic link order shared by cache and differentiable FK."""
    names = {str(urdf.root_link)}
    for parent, joints in urdf.child_joints.items():
        names.add(str(parent))
        names.update(str(joint.child) for joint in joints)
    return sorted(names)


def _to_world(points: np.ndarray, c2r: np.ndarray) -> np.ndarray:
    """Transform robot-base points into HRDexDB's world frame."""
    return points @ c2r[:3, :3].T + c2r[:3, 3]


def _to_frame(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """Map world points to the current wrist/hand-root frame."""
    return (points - pose_world[:3, 3]) @ pose_world[:3, :3]


def _rotate_to_frame(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    result = normals @ pose_world[:3, :3]
    return result / np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1e-8)


class HRDexDBInspireDataset(Dataset):
    """30 Hz consecutive Inspire F1 frames with cached surface correspondences."""

    def __init__(self, *, root: str, cache_root: str, robot_urdf: str, num_hand_points: int, num_obj_points: int, seed: int, split: str, val_fraction: float, test_fraction: float, max_episodes: int | None = None, max_frames_per_episode: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.num_hand_points = int(num_hand_points)
        self.num_obj_points = int(num_obj_points)
        self.seed = int(seed)
        self.robot_urdf = Path(robot_urdf).expanduser().resolve()
        self.max_frames_per_episode = max_frames_per_episode
        self.io = _load_hrdex_io(self.root.parent)
        episodes = sorted((self.root / "inspire_f1").glob("*/*"))
        episodes = [p for p in episodes if p.is_dir()]
        n_test = int(len(episodes) * test_fraction)
        n_val = int(len(episodes) * val_fraction)
        if split == "test":
            episodes = episodes[:n_test]
        elif split == "val":
            episodes = episodes[n_test:n_test + n_val]
        else:
            episodes = episodes[n_test + n_val:]
        if max_episodes is not None:
            episodes = episodes[: int(max_episodes)]
        self.samples = []
        for episode in episodes:
            self.samples.extend(self._prepare_episode(episode))

    def _prepare_episode(self, episode: Path):
        cache_key = hashlib.sha1(("v2_world_arm_wristframe:" + str(episode.relative_to(self.root))).encode("utf-8")).hexdigest()[:16]
        cache_path = self.cache_root / f"{cache_key}_h{self.num_hand_points}_o{self.num_obj_points}_s{self.seed}.npz"
        if self.max_frames_per_episode is None and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                count = int(cached["q_t"].shape[0])
                return [
                    {
                        "hand_points": cached["hand_points"][i],
                        "hand_normals": cached["hand_normals"][i],
                        "hand_flow": cached["hand_flow"][i],
                        "obj_points": cached["obj_points"][i],
                        "obj_normals": cached["obj_normals"][i],
                        "obj_valid_mask": cached["obj_valid_mask"][i].astype(bool),
                        "q_t": cached["q_t"][i],
                        "q_next": cached["q_next"][i],
                        "delta_time_s": np.float32(1.0 / 30.0),
                    }
                    for i in range(count)
                ]
        poses = sorted((episode / "object_6d").glob("pose_*.txt"))
        if not poses:
            poses = sorted((episode / "object_6d_pose_v2").glob("pose_*.txt"))
        if not poses:
            return []
        pose_seq = np.stack([_load_pose(p) for p in poses], axis=0)
        # Align robot state to the recorded video timeline.  Robot streams
        # start before video capture; pairing their first samples directly
        # creates multi-second hand/object penetration in the viewer.
        q_full_raw, robot_times = self.io.load_robot_qpos(episode, "inspire_f1")
        if len(q_full_raw) < 2:
            return []
        robot_times = np.asarray(robot_times, dtype=np.float64).reshape(-1)
        ts_path = episode / "raw" / "timestamps" / "timestamp.npy"
        if ts_path.exists():
            video_times = np.asarray(np.load(ts_path, allow_pickle=True), dtype=np.float64).reshape(-1)
            q_video = np.stack([np.interp(video_times, robot_times, q_full_raw[:, j]) for j in range(12)], axis=1)
            idx = np.linspace(0, len(q_video) - 1, len(pose_seq)).round().astype(int)
            q30_full = q_video[idx].astype(np.float32)
        else:
            target_t = np.arange(float(robot_times[0]), float(robot_times[-1]) + 1e-6, 1.0 / 30.0)
            q30_full = np.stack([np.interp(target_t, robot_times, q_full_raw[:, j]) for j in range(12)], axis=1).astype(np.float32)
            n = min(len(q30_full), len(pose_seq))
            q30_full = q30_full[:n]
            pose_seq = pose_seq[:n]
        q30 = q30_full[:, 6:]
        if self.max_frames_per_episode is not None:
            q30_full = q30_full[: max(2, int(self.max_frames_per_episode))]
            q30 = q30_full[:, 6:]
        c2r = np.load(episode / "C2R.npy", allow_pickle=False) if (episode / "C2R.npy").exists() else np.eye(4, dtype=np.float32)
        pose_seq = pose_seq[:len(q30)].astype(np.float32)  # object poses are already in world coordinates
        object_name = episode.parent.name
        mesh_path = self.root / "assets" / "mesh_v2" / object_name / f"{object_name}.obj"
        if not mesh_path.exists():
            mesh_path = self.root / "assets" / "mesh" / object_name / f"{object_name}.obj"
        if not mesh_path.exists():
            return []
        import trimesh
        object_mesh = trimesh.load(mesh_path, force="mesh", process=False)
        object_seed = self.seed + zlib.crc32(object_name.encode("utf-8")) % 100000
        obj_face, obj_bary = _surface_spec(np.asarray(object_mesh.vertices), np.asarray(object_mesh.faces), self.num_obj_points, object_seed)
        urdf = self.io.parse_urdf(self.robot_urdf)
        mesh_cache = {}
        first_vertices, first_faces = _robot_hand_mesh(self.io, urdf, q30_full[0], mesh_cache)
        hand_face, hand_bary = _surface_spec(first_vertices, first_faces, self.num_hand_points, self.seed + 991)
        samples = []
        for i in range(len(q30) - 1):
            vertices_t, faces_t = _robot_hand_mesh(self.io, urdf, q30_full[i], mesh_cache)
            vertices_n, faces_n = _robot_hand_mesh(self.io, urdf, q30_full[i + 1], mesh_cache)
            hp, hn = _eval_surface(vertices_t, faces_t, hand_face, hand_bary)
            hp_next, _ = _eval_surface(vertices_n, faces_n, hand_face, hand_bary)
            wrist_robot = self.io.compute_link_transforms(urdf, q30_full[i]).get("base_link", np.eye(4))
            wrist_world = c2r @ wrist_robot
            hp = _to_frame(_to_world(hp, c2r), wrist_world).astype(np.float32)
            hp_next = _to_frame(_to_world(hp_next, c2r), wrist_world).astype(np.float32)
            hn = _rotate_to_frame(hn @ c2r[:3, :3].T, wrist_world).astype(np.float32)
            op_local, on = _eval_surface(np.asarray(object_mesh.vertices), np.asarray(object_mesh.faces), obj_face, obj_bary)
            op = op_local @ pose_seq[i, :3, :3].T + pose_seq[i, :3, 3]
            on = on @ pose_seq[i, :3, :3].T
            op = _to_frame(op, wrist_world).astype(np.float32)
            on = _rotate_to_frame(on, wrist_world).astype(np.float32)
            samples.append({"hand_points": hp, "hand_normals": hn, "hand_flow": hp_next - hp, "obj_points": op.astype(np.float32), "obj_normals": on.astype(np.float32), "obj_valid_mask": np.ones(self.num_obj_points, dtype=bool), "q_t": q30[i], "q_next": q30[i + 1], "delta_time_s": np.float32(1.0 / 30.0)})
        if self.max_frames_per_episode is None and samples:
            self.cache_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                hand_points=np.stack([x["hand_points"] for x in samples]),
                hand_normals=np.stack([x["hand_normals"] for x in samples]),
                hand_flow=np.stack([x["hand_flow"] for x in samples]),
                obj_points=np.stack([x["obj_points"] for x in samples]),
                obj_normals=np.stack([x["obj_normals"] for x in samples]),
                obj_valid_mask=np.stack([x["obj_valid_mask"] for x in samples]),
                q_t=np.stack([x["q_t"] for x in samples]),
                q_next=np.stack([x["q_next"] for x in samples]),
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else torch.as_tensor(value) for key, value in self.samples[index].items()}


class LayeredCacheDataset(Dataset):
    """Lazy mmap reader for caches built by ``build_cache.py``."""

    FIELDS = ("hand_points", "hand_normals", "hand_flow", "obj_points", "obj_normals", "obj_valid_mask", "q_t", "q_next", "delta_time_s")

    def __init__(self, cache_root: str, manifest_path: str, split: str, *, max_episodes: int | None = None,
                 max_frames_per_episode: int | None = None, active_motion_only: bool = False,
                 active_motion_threshold_deg: float = 0.5, require_30hz_pair: bool = True,
                 episode_filter: str | None = None, include_cm_tokens: bool = False,
                 include_point_bindings: bool = False):
        self.cache_root = Path(cache_root).expanduser().resolve()
        path = Path(manifest_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "cmdecoder_layered_v4":
            raise ValueError(f"Unsupported cache schema in {path}: {manifest.get('schema')}")
        self.entries: list[tuple[Path, int]] = []
        self._robot_types: dict[Path, str] = {}
        episodes = list(manifest["splits"][split])
        if episode_filter:
            episodes = [episode for episode in episodes if episode == episode_filter]
        if max_episodes is not None:
            episodes = episodes[:int(max_episodes)]
        self.include_cm_tokens = bool(include_cm_tokens)
        self.include_point_bindings = bool(include_point_bindings)
        threshold_rad = float(active_motion_threshold_deg) * np.pi / 180.0
        for episode_index, episode in enumerate(episodes):
            task_dir = self.cache_root / manifest["cache_dirs"][episode] / "task"
            task_manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
            self._robot_types[task_dir] = str(task_manifest.get("robot_type", Path(episode).parts[0]))
            count = int(task_manifest["samples"])
            keep = np.ones(count, dtype=bool)
            if require_30hz_pair:
                keep &= np.load(task_dir / "is_30hz_pair.npy", mmap_mode="r", allow_pickle=False)
            if active_motion_only:
                keep &= np.load(task_dir / "q_delta_abs_max.npy", mmap_mode="r", allow_pickle=False) >= threshold_rad
            selected = np.flatnonzero(keep)
            if max_frames_per_episode is not None:
                selected = selected[:int(max_frames_per_episode)]
            for index in selected:
                self.entries.append((task_dir, index))
        self._arrays: dict[Path, dict[str, np.ndarray]] = {}

    def _derive_wrist_motion(self, task_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """Derive current-to-target wrist SE(3) without invalidating Cm token sidecars."""
        cache_id = task_dir.parent.name
        candidates = [task_dir.parent / "geometry"]
        candidates.extend(self.cache_root.parent.glob(f"*/v4/episodes/{cache_id}/geometry"))
        geometry = next((path for path in candidates if (path / "wrist_pose_world.npy").is_file()), None)
        if geometry is None:
            raise FileNotFoundError(f"Could not locate geometry wrist poses for {task_dir}")
        wrist = np.load(geometry / "wrist_pose_world.npy", mmap_mode="r", allow_pickle=False)
        source_ids = np.load(geometry / "source_frame_id.npy", mmap_mode="r", allow_pickle=False)
        task_manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        sample_count = int(task_manifest["samples"])
        if task_manifest.get("layer") == "cmdecoder_task_horizon":
            stride = int(task_manifest["horizon_stride"])
            current = np.arange(0, len(wrist) - stride, dtype=np.int64)
            target = current + stride
            keep = (source_ids[target] - source_ids[current]) == stride
            current, target = current[keep], target[keep]
        else:
            current = np.arange(sample_count, dtype=np.int64)
            target = current + 1
        if len(current) != sample_count:
            raise ValueError(
                f"Wrist/task pair mismatch for {task_dir}: wrist={len(current)}, task={sample_count}"
            )
        relative = np.linalg.inv(np.asarray(wrist[current], dtype=np.float64)) @ np.asarray(
            wrist[target], dtype=np.float64
        )
        translation = relative[:, :3, 3].astype(np.float32)
        rotvec = Rotation.from_matrix(relative[:, :3, :3]).as_rotvec().astype(np.float32)
        return translation, rotvec

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        task_dir, frame = self.entries[index]
        if task_dir not in self._arrays:
            self._arrays[task_dir] = {
                field: np.load(task_dir / f"{field}.npy", mmap_mode="r", allow_pickle=False)
                for field in self.FIELDS
            }
            # The point-flow decoder does not consume q.  Keep a fixed six-
            # dimensional compatibility view so mixed hand types can be
            # collated without pretending Allegro/MANO q has Inspire meaning.
            robot_type = self._robot_types.get(task_dir, "inspire_f1")
            if robot_type not in {"inspire_f1", "inspire_dftp"}:
                count = len(self._arrays[task_dir]["q_t"])
                self._arrays[task_dir]["q_t"] = np.zeros((count, 6), dtype=np.float32)
                self._arrays[task_dir]["q_next"] = np.zeros((count, 6), dtype=np.float32)
            elif self._arrays[task_dir]["q_t"].shape[-1] != 6:
                self._arrays[task_dir]["q_t"] = np.asarray(self._arrays[task_dir]["q_t"][:, :6], dtype=np.float32)
                self._arrays[task_dir]["q_next"] = np.asarray(self._arrays[task_dir]["q_next"][:, :6], dtype=np.float32)
            translation, rotvec = self._derive_wrist_motion(task_dir)
            self._arrays[task_dir]["wrist_delta_translation"] = translation
            self._arrays[task_dir]["wrist_delta_rotvec"] = rotvec
            if self.include_cm_tokens:
                cm_path = task_dir.parent / "cm" / "cm_tokens.npy"
                if not cm_path.exists():
                    raise FileNotFoundError(f"Cm token cache not found: {cm_path}")
                self._arrays[task_dir]["cm_tokens"] = np.load(cm_path, mmap_mode="r", allow_pickle=False)
            if self.include_point_bindings:
                binding_fields = ("hand_point_link_index", "hand_points_local")
                missing = [field for field in binding_fields if not (task_dir / f"{field}.npy").is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Point binding cache missing {missing} for {task_dir}; rebuild the v4/v2 cache"
                    )
                self._arrays[task_dir].update({
                    field: np.load(task_dir / f"{field}.npy", mmap_mode="r", allow_pickle=False)
                    for field in binding_fields
                })
        static_fields = {"hand_point_link_index", "hand_points_local"}
        sample = {
            field: torch.from_numpy(np.array(array if field in static_fields else array[frame], copy=True))
            for field, array in self._arrays[task_dir].items()
        }
        return sample


class RandomHorizonGeometryDataset(Dataset):
    """Sample one deterministic pseudo-random horizon per current geometry frame.

    Unlike the task cache, this keeps only the frame-level geometry on disk and
    constructs the current-wrist pair online.  The stride assignment is stable
    for reproducible validation, while covering ``1..max_stride`` across the
    dataset without multiplying the number of samples by ten.
    """

    FIELDS = ("hand_points", "hand_normals", "hand_flow", "obj_points", "obj_normals",
              "obj_valid_mask", "q_t", "q_next", "delta_time_s")

    def __init__(self, cache_root: str, manifest_path: str, split: str, *, max_stride: int = 10,
                 stride_values: tuple[int, ...] | list[int] | None = None,
                 fixed_stride: int | None = None,
                 all_strides: bool = False,
                 coordinate_frame: str = "hand_root_t",
                 max_episodes: int | None = None, seed: int = 42):
        self.cache_root = Path(cache_root).expanduser().resolve()
        manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
        self.max_stride = int(max_stride)
        if self.max_stride < 1:
            raise ValueError("max_stride must be positive")
        self.coordinate_frame = str(coordinate_frame)
        if self.coordinate_frame not in {"hand_root_t", "object_pose_t"}:
            raise ValueError(f"Unsupported coordinate_frame={self.coordinate_frame!r}")
        manifest_frame = manifest.get("coordinate_frame")
        if manifest_frame is not None and str(manifest_frame) != self.coordinate_frame:
            raise ValueError(
                f"Cache manifest coordinate_frame={manifest_frame!r}, requested "
                f"{self.coordinate_frame!r}: {manifest_path}"
            )
        if fixed_stride is not None and int(fixed_stride) <= 0:
            raise ValueError("fixed_stride must be positive")
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        if stride_values is None:
            parsed_strides = tuple(range(1, self.max_stride + 1))
        else:
            parsed_strides = tuple(sorted({int(value) for value in stride_values}))
            if not parsed_strides or any(value <= 0 for value in parsed_strides):
                raise ValueError("stride_values must contain positive integers")
        self.stride_values = parsed_strides
        self.all_strides = bool(all_strides)
        if self.fixed_stride is not None and self.fixed_stride not in self.stride_values:
            self.stride_values = tuple(sorted(set(self.stride_values) | {self.fixed_stride}))
        episodes = list(manifest["splits"][split])
        if max_episodes is not None:
            episodes = episodes[:int(max_episodes)]
        self.entries: list[tuple[Path, int, int]] = []
        self._arrays: dict[Path, dict[str, np.ndarray]] = {}
        self._robot_types: dict[Path, str] = {}
        for episode_index, episode in enumerate(episodes):
            geometry = self.cache_root / manifest["cache_dirs"][episode] / "geometry"
            geometry_manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8"))
            geometry_frame = geometry_manifest.get("coordinate_frame")
            if geometry_frame is not None and str(geometry_frame) != "world":
                raise ValueError(f"Geometry cache must be world coordinates, got {geometry_frame!r}: {geometry}")
            if self.coordinate_frame == "object_pose_t" and not (geometry / "obj_pose_world.npy").is_file():
                raise ValueError(f"object_pose_t requires obj_pose_world.npy: {geometry}")
            self._robot_types[geometry] = str(geometry_manifest.get("robot_type", Path(episode).parts[0]))
            source_ids = np.load(geometry / "source_frame_id.npy", mmap_mode="r", allow_pickle=False)
            n = len(source_ids)
            if self.fixed_stride is not None or self.all_strides:
                candidate_strides = (self.fixed_stride,) if self.fixed_stride is not None else self.stride_values
                for frame in range(n - 1):
                    for stride in candidate_strides:
                        target = frame + int(stride)
                        if target < n and int(source_ids[target]) - int(source_ids[frame]) == int(stride):
                            self.entries.append((geometry, frame, int(stride)))
            else:
                # One current frame per sample; choose a stable pseudo-random stride.
                for frame in range(n - 1):
                    stride = self.stride_values[
                        (frame * 1009 + episode_index * 9176 + int(seed) * 7919) % len(self.stride_values)
                    ]
                    target = frame + int(stride)
                    if target < n and int(source_ids[target]) - int(source_ids[frame]) == int(stride):
                        self.entries.append((geometry, frame, int(stride)))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        geometry, frame, stride = self.entries[index]
        robot_type = self._robot_types.get(geometry, "inspire_f1")
        if geometry not in self._arrays:
            names = ("hand_points_world", "hand_normals_world", "obj_points_world",
                     "obj_normals_world", "wrist_pose_world", "q_full", "frame_time")
            if self.coordinate_frame == "object_pose_t":
                names = names + ("obj_pose_world",)
            self._arrays[geometry] = {
                name: np.load(geometry / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                for name in names
            }
        a = self._arrays[geometry]
        target = frame + stride
        wrist = np.asarray(
            a["obj_pose_world"][frame] if self.coordinate_frame == "object_pose_t" else a["wrist_pose_world"][frame],
            dtype=np.float32,
        )
        hand = np.asarray(a["hand_points_world"][frame], dtype=np.float32)
        hand_target = np.asarray(a["hand_points_world"][target], dtype=np.float32)
        obj = np.asarray(a["obj_points_world"][frame], dtype=np.float32)
        hand_normals = np.asarray(a["hand_normals_world"][frame], dtype=np.float32)
        obj_normals = np.asarray(a["obj_normals_world"][frame], dtype=np.float32)
        if robot_type not in {"inspire_f1", "inspire_dftp"}:
            q_t = np.zeros(6, dtype=np.float32)
            q_next = np.zeros(6, dtype=np.float32)
        else:
            q_t = np.asarray(a["q_full"][frame, 6:12], dtype=np.float32)
            q_next = np.asarray(a["q_full"][target, 6:12], dtype=np.float32)
        sample = {
            "hand_points": torch.from_numpy(_to_frame(hand, wrist)),
            "hand_normals": torch.from_numpy(_rotate_to_frame(hand_normals, wrist)),
            "hand_flow": torch.from_numpy(_to_frame(hand_target, wrist) - _to_frame(hand, wrist)),
            "obj_points": torch.from_numpy(_to_frame(obj, wrist)),
            "obj_normals": torch.from_numpy(_rotate_to_frame(obj_normals, wrist)),
            "obj_valid_mask": torch.ones(len(obj), dtype=torch.bool),
            "q_t": torch.from_numpy(np.array(q_t, dtype=np.float32, copy=True)),
            "q_next": torch.from_numpy(np.array(q_next, dtype=np.float32, copy=True)),
            "delta_time_s": torch.tensor(float(a["frame_time"][target] - a["frame_time"][frame]), dtype=torch.float32),
            "stride": torch.tensor(int(stride), dtype=torch.long),
        }
        return sample


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any = None):
    if getattr(data_cfg, "object_v2_filter", None):
        from src.task.CmDecoder.dataset_object_v2 import make_dataloaders as make_object_v2_dataloaders
        return make_object_v2_dataloaders(data_cfg, seed, meta_cfg=meta_cfg, distributed=distributed)
    manifest_path = getattr(data_cfg, "cache_manifest", None)
    if manifest_path:
        required_hand_flow_frame = getattr(meta_cfg, "required_hand_flow_frame", None)
        if required_hand_flow_frame is not None:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            actual_hand_flow_frame = manifest.get("hand_flow_frame")
            if actual_hand_flow_frame != required_hand_flow_frame:
                raise ValueError(
                    f"Cache hand_flow_frame={actual_hand_flow_frame!r}, required "
                    f"{required_hand_flow_frame!r}: {manifest_path}"
                )
        cache_kwargs = dict(max_episodes=getattr(data_cfg, "max_episodes", None), max_frames_per_episode=getattr(data_cfg, "max_frames_per_episode", None), active_motion_only=bool(getattr(data_cfg, "active_motion_only", False)), active_motion_threshold_deg=float(getattr(data_cfg, "active_motion_threshold_deg", 0.5)), require_30hz_pair=bool(getattr(data_cfg, "require_30hz_pair", True)), episode_filter=getattr(data_cfg, "episode_filter", None), include_cm_tokens=bool(getattr(meta_cfg, "use_cached_cm_tokens", False)), include_point_bindings=bool(getattr(meta_cfg, "use_cached_point_bindings", False)))
        random_horizon = int(getattr(data_cfg, "random_horizon_max_stride", 0))
        if random_horizon > 0:
            stride_values = getattr(data_cfg, "random_horizon_stride_values", None)
            if stride_values is not None:
                stride_values = tuple(int(value) for value in stride_values)
                random_horizon = max(random_horizon, max(stride_values))
            coordinate_frame = str(getattr(meta_cfg, "coordinate_frame", "hand_root_t"))
            random_kwargs = dict(
                max_stride=random_horizon,
                stride_values=stride_values,
                coordinate_frame=coordinate_frame,
                seed=seed,
            )
            train = RandomHorizonGeometryDataset(str(meta_cfg.cache_root), str(manifest_path), "train", **random_kwargs)
            val = RandomHorizonGeometryDataset(
                str(meta_cfg.cache_root), str(manifest_path), "val",
                **{**random_kwargs, "all_strides": bool(getattr(data_cfg, "eval_all_strides", False))}
            )
            test = RandomHorizonGeometryDataset(
                str(meta_cfg.cache_root), str(manifest_path), "test",
                **{**random_kwargs, "all_strides": bool(getattr(data_cfg, "eval_all_strides", False))}
            )
        else:
            train = LayeredCacheDataset(str(meta_cfg.cache_root), str(manifest_path), "train", **cache_kwargs)
            val = LayeredCacheDataset(str(meta_cfg.cache_root), str(manifest_path), "val", **cache_kwargs)
            test = LayeredCacheDataset(str(meta_cfg.cache_root), str(manifest_path), "test", **cache_kwargs)
        sampler = None
        if distributed is not None and getattr(distributed, "enabled", False):
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train, shuffle=True, seed=seed)
        train_loader = DataLoader(train, batch_size=int(data_cfg.batch_size), shuffle=sampler is None, sampler=sampler, num_workers=int(data_cfg.num_workers), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0, drop_last=bool(data_cfg.drop_last))
        val_loader = DataLoader(val, batch_size=int(data_cfg.val_batch_size), shuffle=False, num_workers=int(data_cfg.num_workers), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0)
        test_loader = DataLoader(test, batch_size=int(data_cfg.val_batch_size), shuffle=False, num_workers=int(data_cfg.num_workers), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0)
        return train_loader, val_loader, test_loader, {}, {"val/": val_loader}, {"test/": test_loader}
    kwargs = dict(root=str(data_cfg.root), cache_root=str(meta_cfg.cache_root), robot_urdf=str(meta_cfg.robot_urdf), num_hand_points=int(meta_cfg.num_hand_points), num_obj_points=int(meta_cfg.num_obj_points), seed=seed, val_fraction=float(data_cfg.val_fraction), test_fraction=float(data_cfg.test_fraction), max_episodes=getattr(data_cfg, "max_episodes", None), max_frames_per_episode=getattr(data_cfg, "max_frames_per_episode", None))
    train = HRDexDBInspireDataset(split="train", **kwargs)
    val = HRDexDBInspireDataset(split="val", **kwargs)
    test = HRDexDBInspireDataset(split="test", **kwargs)
    sampler = None
    if distributed is not None and getattr(distributed, "enabled", False):
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train, shuffle=True, seed=seed)
    train_loader = DataLoader(train, batch_size=int(data_cfg.batch_size), shuffle=sampler is None, sampler=sampler, num_workers=int(data_cfg.num_workers), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0, drop_last=bool(data_cfg.drop_last))
    val_loader = DataLoader(val, batch_size=int(data_cfg.val_batch_size), shuffle=False, num_workers=int(data_cfg.num_workers), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0)
    test_loader = DataLoader(test, batch_size=int(data_cfg.val_batch_size), shuffle=False, num_workers=int(data_cfg.num_workers), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0)
    return train_loader, val_loader, test_loader, {}, {"val/": val_loader}, {"test/": test_loader}
