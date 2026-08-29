"""Schema, fingerprint, and validation helpers for the Cm Scene Cache V1.

数据语义升级版（V1.md）：``obj_points`` 不再特指 manipulated object，而是
当前手附近的 local scene points（object + environment 一视同仁）。本模块
定义三级 mmap cache（scene geometry / sampling bank / dense bank）的磁盘
布局、指纹校验和只读加载接口。

Layout::

    cm_scene_v1/
    ├── meta.json
    └── s1/mug_lift_1/
        ├── shared/
        │   ├── meta.json            seq_id / ds_rate / source_fps / ...
        │   ├── raw_frame_id.npy     [T]        int32
        │   ├── obj_points_world.npy [T,N_O,3]  float32
        │   ├── obj_normals_world.npy[T,N_O,3]  float32
        │   ├── env_points_world.npy [N_E,3]    float32  (static_world)
        │   ├── env_normals_world.npy[N_E,3]    float32
        │   └── scene_source_id.npy  [N_O+N_E]  uint8    (0=object, 1=environment)
        ├── left|right/
        │   ├── hand_points_world.npy   [T,1538,3] float32
        │   ├── hand_normals_world.npy  [T,1538,3] float32
        │   ├── hand_root_pose_world.npy[T,4,4]    float32
        │   ├── candidate_offsets.npy   [T+1]      int64
        │   └── candidate_indices.npy   [K]        int32   (pool indices)
        ├── sampling_bank/
        │   └── left|right_indices.npy [T,B,512]   uint32  (0xFFFFFFFF = invalid)
        └── dense_bank/
            └── left|right/
                ├── z_scene.npy      [T_active,B,512,D]  float16
                ├── z_hand.npy       [T_active,B,1538,D] float16
                ├── hand_contact.npy [T_active,B,1538]   float16
                ├── active_frame_id.npy [T_active]       int32
                └── frame_to_dense.npy [T]                int32
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_NAME = "ref2dex_cm_scene_v1_1"
SCHEMA_VERSION = "1.1.0"

# Variable asset counts can exceed the uint16 index range.
INVALID_INDEX = 0xFFFFFFFF

SIDES = ("left", "right")
SHARED_FIELDS = (
    "raw_frame_id.npy",
    "obj_points_world.npy",
    "obj_normals_world.npy",
    "env_points_world.npy",
    "env_normals_world.npy",
    "scene_source_id.npy",
    "scene_asset_id.npy",
    "asset_offsets.npy",
    "meta.json",
)
SIDE_FIELDS = (
    "hand_points_world.npy",
    "hand_normals_world.npy",
    "hand_root_pose_world.npy",
    "candidate_offsets.npy",
    "candidate_indices.npy",
)
ENV_SOURCE_OBJECT = 0
ENV_SOURCE_ENVIRONMENT = 1
SUPPORTED_ENV_STORAGE = ("static_world",)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file's bytes; used to bind feature caches to their checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scene_cache_fingerprint(
    *,
    schema: str,
    points_per_asset: int,
    candidate_threshold_m: float,
    sampling_seed: int,
) -> str:
    """Fingerprint binding the sampling bank to the geometry cache contract."""
    payload = f"{schema}\0{int(points_per_asset)}\0{float(candidate_threshold_m):.9g}\0{int(sampling_seed)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dense_cache_fingerprint(
    *,
    checkpoint_sha: str,
    schema: str,
    sampling_fingerprint: str,
    num_scene_points: int,
    num_hand_points: int,
    bank_size: int,
) -> str:
    """Fingerprint binding dense features to checkpoint + sampling + point config.

    Dense cache 必须绑定 DenseToken checkpoint、scene sampling bank、
    normal 预处理、坐标约定和点数；任一变化都必须失效重建。
    """
    payload = (
        f"{checkpoint_sha}\0{schema}\0{sampling_fingerprint}\0"
        f"{int(num_scene_points)}\0{int(num_hand_points)}\0{int(bank_size)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_meta(root: str | Path) -> dict[str, Any]:
    meta_path = Path(root) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing scene-cache meta: {meta_path}")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{meta_path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{meta_path} must contain a JSON object.")
    return payload


def update_meta(root: str | Path, **fields: Any) -> dict[str, Any]:
    """Atomically merge fields into the root ``meta.json``."""
    root = Path(root)
    meta = read_meta(root) if (root / "meta.json").is_file() else {}
    meta.update(fields)
    root.mkdir(parents=True, exist_ok=True)
    tmp_path = root / "meta.json.tmp"
    tmp_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(root / "meta.json")
    return meta


def iter_sequence_dirs(root: str | Path) -> list[Path]:
    """Sequence dirs are two levels below the root (subject/sequence)."""
    root_path = Path(root)
    dirs = sorted(path for path in root_path.glob("*/*") if (path / "shared").is_dir())
    return dirs


def side_dir(sequence_dir: str | Path, side: str) -> Path:
    if side not in SIDES:
        raise ValueError(f"Unsupported side {side!r}")
    return Path(sequence_dir) / side


def load_mmap(path: str | Path) -> np.ndarray:
    return np.load(str(path), mmap_mode="r")


class SceneSequenceCache:
    """Lazy mmap view over one sequence directory of the scene cache."""

    def __init__(self, sequence_dir: str | Path) -> None:
        self.dir = Path(sequence_dir)
        shared_dir = self.dir / "shared"
        missing = [name for name in SHARED_FIELDS if not (shared_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{self.dir}: missing shared fields {missing}")
        self.shared_meta = json.loads((shared_dir / "meta.json").read_text(encoding="utf-8"))
        self.raw_frame_id: np.ndarray = load_mmap(shared_dir / "raw_frame_id.npy")
        self.obj_points_world: np.ndarray = load_mmap(shared_dir / "obj_points_world.npy")
        self.obj_normals_world: np.ndarray = load_mmap(shared_dir / "obj_normals_world.npy")
        self.env_points_world: np.ndarray = load_mmap(shared_dir / "env_points_world.npy")
        self.env_normals_world: np.ndarray = load_mmap(shared_dir / "env_normals_world.npy")
        self.scene_source_id: np.ndarray = load_mmap(shared_dir / "scene_source_id.npy")
        self.scene_asset_id: np.ndarray = load_mmap(shared_dir / "scene_asset_id.npy")
        self.asset_offsets: np.ndarray = load_mmap(shared_dir / "asset_offsets.npy")
        self._validate_shared()

    def _validate_shared(self) -> None:
        num_obj, num_env = self.obj_points_world.shape[1], self.env_points_world.shape[0]
        if self.obj_points_world.ndim != 3 or self.obj_points_world.shape[2] != 3:
            raise ValueError(f"{self.dir}: obj_points_world must be [T,N,3]")
        if self.env_points_world.shape != (num_env, 3):
            raise ValueError(f"{self.dir}: static env_points_world must be [N_E,3]")
        if self.obj_normals_world.shape != self.obj_points_world.shape:
            raise ValueError(f"{self.dir}: obj normals shape mismatch")
        if self.env_normals_world.shape != self.env_points_world.shape:
            raise ValueError(f"{self.dir}: env normals shape mismatch")
        if self.scene_source_id.shape != (num_obj + num_env,):
            raise ValueError(f"{self.dir}: scene_source_id must have length N_O+N_E")
        if self.scene_asset_id.shape != self.scene_source_id.shape:
            raise ValueError(f"{self.dir}: scene_asset_id must match scene pool length")
        if self.asset_offsets.ndim != 1 or self.asset_offsets.size < 2:
            raise ValueError(f"{self.dir}: asset_offsets must contain at least one asset")
        if int(self.asset_offsets[0]) != 0 or int(self.asset_offsets[-1]) != self.num_scene_pool:
            raise ValueError(f"{self.dir}: asset_offsets must span the complete scene pool")
        if np.any(np.diff(self.asset_offsets) <= 0):
            raise ValueError(f"{self.dir}: asset_offsets must be strictly increasing")
        if self.raw_frame_id.shape != (self.obj_points_world.shape[0],):
            raise ValueError(f"{self.dir}: raw_frame_id length must match obj frames")
        storage = str(self.shared_meta.get("environment_storage", "static_world"))
        if storage not in SUPPORTED_ENV_STORAGE:
            raise ValueError(f"{self.dir}: unsupported environment_storage {storage!r}")

    @property
    def num_obj_pool(self) -> int:
        return int(self.obj_points_world.shape[1])

    @property
    def num_env_pool(self) -> int:
        return int(self.env_points_world.shape[0])

    @property
    def num_scene_pool(self) -> int:
        return self.num_obj_pool + self.num_env_pool

    @property
    def frame_count(self) -> int:
        return int(self.obj_points_world.shape[0])

    def load_side(self, side: str, *, num_hand_points: int | None = None) -> dict[str, np.ndarray]:
        directory = side_dir(self.dir, side)
        missing = [name for name in SIDE_FIELDS if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{directory}: missing side fields {missing}")
        arrays = {
            "hand_points_world": load_mmap(directory / "hand_points_world.npy"),
            "hand_normals_world": load_mmap(directory / "hand_normals_world.npy"),
            "hand_root_pose_world": load_mmap(directory / "hand_root_pose_world.npy"),
            "candidate_offsets": load_mmap(directory / "candidate_offsets.npy"),
            "candidate_indices": load_mmap(directory / "candidate_indices.npy"),
        }
        frames = self.frame_count
        if arrays["hand_points_world"].shape[0] != frames:
            raise ValueError(f"{directory}: hand frame count mismatch")
        if num_hand_points is not None and arrays["hand_points_world"].shape[1] != int(num_hand_points):
            raise ValueError(f"{directory}: expected {num_hand_points} hand points")
        if arrays["candidate_offsets"].shape != (frames + 1,):
            raise ValueError(f"{directory}: candidate_offsets must have shape [T+1,{frames + 1}]")
        if arrays["candidate_offsets"][0] != 0:
            raise ValueError(f"{directory}: candidate_offsets must start at 0")
        if int(arrays["candidate_offsets"][-1]) != arrays["candidate_indices"].shape[0]:
            raise ValueError(f"{directory}: candidate_indices length must match offsets[-1]")
        return arrays

    def candidate_indices_at(self, side_arrays: dict[str, np.ndarray], frame: int) -> np.ndarray:
        """Ragged candidate pool indices for one frame (see V1.md §9)."""
        offsets = side_arrays["candidate_offsets"]
        start, end = int(offsets[int(frame)]), int(offsets[int(frame) + 1])
        return side_arrays["candidate_indices"][start:end]


def validate_scene_root(
    root: str | Path,
    *,
    num_scene_points: int | None = None,
    sampling_bank_size: int | None = None,
    use_dense_cache: bool = False,
    dense_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate the root meta and every sequence; fail loudly, never silently.

    ``num_scene_points`` is the per-bank model input size (512), used only to
    check the sampling-bank row width.  The full scene pool size
    (``N_O + N_E``) is validated against the root meta instead.

    见 V1.md §22：cache fingerprint 不匹配必须 ``raise``，绝不静默回退。
    """
    meta = read_meta(root)
    if str(meta.get("schema_name")) != SCHEMA_NAME:
        raise ValueError(f"{root}: expected schema {SCHEMA_NAME!r}, got {meta.get('schema_name')!r}")
    sequence_dirs = iter_sequence_dirs(root)
    if not sequence_dirs:
        raise ValueError(f"{root}: no sequence directories found")
    for sequence_dir in sequence_dirs:
        cache = SceneSequenceCache(sequence_dir)
        present_sides = tuple(side for side in SIDES if side_dir(sequence_dir, side).is_dir())
        for side in present_sides:
            cache.load_side(side)
        if sampling_bank_size is not None:
            bank_dir = sequence_dir / "sampling_bank"
            for side in present_sides:
                bank_path = bank_dir / f"{side}_indices.npy"
                if not bank_path.is_file():
                    raise FileNotFoundError(f"{bank_path}: sampling bank missing; run build_sampling_bank")
                bank = load_mmap(bank_path)
                expected = (cache.frame_count, int(sampling_bank_size), int(num_scene_points or 0))
                if bank.shape != expected:
                    raise ValueError(f"{bank_path}: expected shape {expected}, got {bank.shape}")
    if use_dense_cache:
        stored = meta.get("dense_cache", {})
        if not stored.get("enabled"):
            raise ValueError(f"{root}: dense cache not built; run build_dense_cache first")
        if dense_fingerprint is not None and str(stored.get("fingerprint")) != dense_fingerprint:
            raise ValueError(
                f"{root}: dense cache fingerprint mismatch: cache={stored.get('fingerprint')!r} "
                f"expected={dense_fingerprint!r}. Rebuild the dense cache (or the sampling bank); "
                "never silently reuse stale features."
            )
        for sequence_dir in sequence_dirs:
            for side in SIDES:
                dense_dir = sequence_dir / "dense_bank" / side
                if not side_dir(sequence_dir, side).is_dir():
                    continue
                for name in ("z_scene.npy", "z_hand.npy", "hand_contact.npy", "frame_to_dense.npy", "active_frame_id.npy"):
                    if not (dense_dir / name).is_file():
                        raise FileNotFoundError(f"{dense_dir / name}: dense bank file missing")
    return meta
