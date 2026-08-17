"""Cm Scene Cache V1 dataset: mmap scene pool + sampling bank + dense bank.

依照 ``src/task/Cm/docs/指导/V1.md``：``obj_points`` 的语义从 manipulated
object 升级为 local scene points（object + environment 一视同仁），但模型
输入规模仍是 512 scene + 1538 hand，batch key 命名与旧 ``Stage4CmDataset``
完全一致，模型和 loss 不感知差别。

``__getitem__`` 只做非常便宜的工作（V1.md §19）：选 sampling bank、读
512 个 selected_idx、运行时选 stride、mmap 两个端点的 scene 点并变换到
当前手根系、读 cached DenseToken。大的 Frozen PTv3 已移出 training loop。

mmap 句柄按 LRU 逐 worker 限量打开（默认每类 8 条序列），主要依赖
OS page cache + mmap 让多个 worker 共享磁盘页（V1.md §20）。
"""
from __future__ import annotations

import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base import make_default_eval_sampler, make_file_split_dataloaders
from src.task.Cm.cache_schema import (
    INVALID_INDEX,
    SCHEMA_NAME,
    SceneSequenceCache,
    dense_cache_fingerprint,
    read_meta,
    scene_cache_fingerprint,
    sha256_file,
    side_dir,
    validate_scene_root,
)
from src.task.correspondence_ptv3_v2.sampling import stable_frame_seed


# 每个 DataLoader worker 保留的 mmap 序列句柄上限（shared cache 与 side
# 记录各一份 LRU），其余交给 OS page cache。
_OPEN_SEQUENCE_LIMIT = 8
_DENSE_FILES = ("z_scene", "z_hand", "hand_contact", "frame_to_dense")


def _world_to_hand(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """``x_hand = R^T (x_world - t)``，与旧 ``Stage4CmDataset`` 完全一致。"""
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float32)
    translation = np.asarray(pose_world[:3, 3], dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normal_world_to_hand(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float32)
    result = np.asarray(normals, dtype=np.float32) @ rotation
    return (result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def gather_frame_scene_inputs(
    cache: SceneSequenceCache,
    side_arrays: dict[str, np.ndarray],
    bank_indices: np.ndarray,
    *,
    frame: int,
    bank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Current-frame 512 scene points/normals in ``hand_root_t`` + valid mask.

    Dataset 在线路径与 ``build_dense_cache`` 预计算路径共用本函数，保证
    cached DenseToken 与在线 Frozen PTv3 的输入逐位一致（parity Test D）。
    """
    selected = np.asarray(bank_indices[int(frame), int(bank)], dtype=np.int64)
    valid = selected != INVALID_INDEX
    safe = np.where(valid, selected, 0)
    scene_world = np.concatenate([
        np.asarray(cache.obj_points_world[int(frame)], dtype=np.float32),
        np.asarray(cache.env_points_world, dtype=np.float32),
    ])
    normals_world = np.concatenate([
        np.asarray(cache.obj_normals_world[int(frame)], dtype=np.float32),
        np.asarray(cache.env_normals_world, dtype=np.float32),
    ])
    pose = np.asarray(side_arrays["hand_root_pose_world"][int(frame)], dtype=np.float32)
    points = _world_to_hand(scene_world[safe], pose)
    normals = _normal_world_to_hand(normals_world[safe], pose)
    points[~valid] = 0.0
    normals[~valid] = 0.0
    return points, normals, valid


class Stage4CmSceneDataset(Dataset):
    """One active current frame over the unified scene pool, runtime stride.

    一个样本 = (sequence, side, current)。512 scene 点来自提前生成的
    sampling bank（B 套），bank 随 epoch 变化（V1.md §12）；stride 与旧
    数据集一样运行时决定（V1.md §13）；监督量在 ``hand_root_t`` 下构造。
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        base_seed: int = 42,
        active_only: bool = True,
        min_stride: int = 1,
        max_stride: int = 10,
        fixed_stride: int | None = None,
        max_samples: int | None = None,
        min_object_flow_norm: float = 0.0,
        train_strides: Sequence[int] | None = None,
        coordinate_frame: str = "hand_root_t",
        sampling_bank_size: int = 4,
        # 验证/测试固定 bank（例如 0）保证完全 deterministic；None 时随 epoch 变。
        fixed_bank: int | None = None,
        use_dense_cache: bool = False,
    ) -> None:
        if min_stride <= 0 or max_stride < min_stride:
            raise ValueError("Require 0 < min_stride <= max_stride.")
        if fixed_stride is not None and not min_stride <= fixed_stride <= max_stride:
            raise ValueError("fixed_stride must lie in [min_stride, max_stride].")
        if sampling_bank_size <= 0:
            raise ValueError("sampling_bank_size must be positive.")
        if fixed_bank is not None and not 0 <= fixed_bank < sampling_bank_size:
            raise ValueError("fixed_bank must lie in [0, sampling_bank_size).")
        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.base_seed = int(base_seed)
        self._epoch = mp.Value("q", 0, lock=True)
        self.active_only = bool(active_only)
        self.min_stride, self.max_stride = int(min_stride), int(max_stride)
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        self.coordinate_frame = str(coordinate_frame)
        self.min_object_flow_norm = float(min_object_flow_norm)
        if self.min_object_flow_norm < 0.0:
            raise ValueError("min_object_flow_norm must be non-negative.")
        self.train_strides = None if train_strides is None else tuple(sorted({int(v) for v in train_strides}))
        if self.train_strides is not None and (
            not self.train_strides
            or any(v < self.min_stride or v > self.max_stride for v in self.train_strides)
        ):
            raise ValueError("train_strides must be a non-empty subset of [min_stride, max_stride].")
        self.sampling_bank_size = int(sampling_bank_size)
        self.fixed_bank = None if fixed_bank is None else int(fixed_bank)
        self.use_dense_cache = bool(use_dense_cache)

        self.file_list = file_list
        self.sequence_dirs = self._resolve_sequence_dirs(file_list)
        self._seq_meta: dict[str, dict[str, Any]] = {}
        # key "seq|side" -> {"side_arrays", "bank", "dense"}；LRU 限量打开。
        self._open_records: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._open_caches: "OrderedDict[str, SceneSequenceCache]" = OrderedDict()
        self.ds_rate: int | None = None
        self.source_fps: float | None = None
        self.effective_fps: float | None = None
        self._samples: list[tuple[str, str, int]] = []
        for sequence_dir in self.sequence_dirs:
            cache = self._get_cache(sequence_dir)
            frame_count = cache.frame_count
            for side in self._available_sides(sequence_dir):
                record = self._load_record(sequence_dir, side)
                offsets = record["side_arrays"]["candidate_offsets"]
                counts = np.diff(np.asarray(offsets))
                filter_stride = self.fixed_stride if self.fixed_stride is not None else self.max_stride
                for current in range(0, frame_count - self.max_stride):
                    if self.active_only and counts[current] <= 0:
                        continue
                    if self.min_object_flow_norm > 0.0:
                        candidate = cache.candidate_indices_at(record["side_arrays"], current)
                        if candidate.size:
                            flow_norm = _scene_flow_norm(cache, current, current + filter_stride, candidate)
                            if float(flow_norm.mean()) < self.min_object_flow_norm:
                                continue
                    self._samples.append((str(sequence_dir), side, current))
        if self.ds_rate is None:
            raise ValueError(f"No complete scene sequences found under {self.data_path}")
        if max_samples is not None and int(max_samples) > 0:
            self._samples = self._samples[:int(max_samples)]
        if self.train_strides is not None:
            self._samples = [
                (seq, side, current, stride)
                for seq, side, current in self._samples
                for stride in self.train_strides
            ]
        if not self._samples:
            raise ValueError("No active frames with the configured maximum future stride.")

    # ---- construction / lazy loading -------------------------------------------
    def _resolve_sequence_dirs(self, file_list: list[str | Path] | None) -> list[str]:
        if file_list is not None:
            raw_paths = [Path(path) for path in file_list]
        else:
            raw_paths = sorted(self.data_path.glob("**/shared/raw_frame_id.npy"))
        dirs: list[str] = []
        for path in raw_paths:
            if path.name == "raw_frame_id.npy" and path.parent.name == "shared":
                candidate = path.parents[1]
            elif path.is_dir() and (path / "shared").is_dir():
                candidate = path
            elif path.parent.name == "shared":
                candidate = path.parents[1]
            else:
                raise ValueError(f"{path}: not a scene-cache sequence (expected shared/raw_frame_id.npy)")
            resolved = str(candidate.resolve())
            if resolved not in dirs:
                dirs.append(resolved)
        if not dirs:
            raise ValueError(f"No scene-cache sequences found in {self.data_path}")
        return dirs

    def _available_sides(self, sequence_dir: str | Path) -> tuple[str, ...]:
        return tuple(side for side in ("left", "right") if side_dir(sequence_dir, side).is_dir())

    def _get_cache(self, sequence_dir: str | Path) -> SceneSequenceCache:
        key = str(sequence_dir)
        cache = self._open_caches.get(key)
        if cache is None:
            cache = SceneSequenceCache(key)
            self._open_caches[key] = cache
            while len(self._open_caches) > _OPEN_SEQUENCE_LIMIT:
                self._open_caches.popitem(last=False)
        else:
            self._open_caches.move_to_end(key)
        if key not in self._seq_meta:
            meta = cache.shared_meta
            ds_rate, source_fps = int(meta["ds_rate"]), float(meta["source_fps"])
            if ds_rate <= 0 or source_fps <= 0.0:
                raise ValueError(f"{key}: ds_rate and source_fps must be positive")
            if self.ds_rate is None:
                self.ds_rate, self.source_fps = ds_rate, source_fps
                self.effective_fps = source_fps / ds_rate
            elif (self.ds_rate, self.source_fps) != (ds_rate, source_fps):
                raise ValueError("All scene sequences in one Dataset must use the same ds_rate and source_fps")
            self._seq_meta[key] = meta
        return cache

    def _load_record(self, sequence_dir: str | Path, side: str) -> dict[str, Any]:
        cache = self._get_cache(sequence_dir)
        side_arrays = cache.load_side(side, num_hand_points=self.num_hand_points)
        bank_path = Path(sequence_dir) / "sampling_bank" / f"{side}_indices.npy"
        if not bank_path.is_file():
            raise FileNotFoundError(f"{bank_path}: sampling bank missing; run build_sampling_bank first")
        bank = np.load(str(bank_path), mmap_mode="r")
        expected_shape = (cache.frame_count, self.sampling_bank_size, self.num_obj_points)
        if bank.shape != expected_shape:
            raise ValueError(f"{bank_path}: expected shape {expected_shape}, got {bank.shape}")
        record: dict[str, Any] = {"side_arrays": side_arrays, "bank": bank, "dense": None}
        if self.use_dense_cache:
            dense_dir = Path(sequence_dir) / "dense_bank" / side
            dense = {
                name: np.load(str(dense_dir / f"{name}.npy"), mmap_mode="r")
                for name in _DENSE_FILES
            }
            expected_dense = (cache.frame_count, self.sampling_bank_size)
            mapping = dense["frame_to_dense"]
            if mapping.shape != (cache.frame_count,):
                raise ValueError(f"{dense_dir / 'frame_to_dense.npy'}: expected shape {(cache.frame_count,)}, got {mapping.shape}")
            if np.any(mapping >= 0) and int(mapping[mapping >= 0].max()) >= dense["z_scene"].shape[0]:
                raise ValueError(f"{dense_dir}: frame_to_dense contains an out-of-range row")
            if dense["z_scene"].shape[1] != self.sampling_bank_size or dense["z_hand"].shape[1] != self.sampling_bank_size:
                raise ValueError(f"{dense_dir}: dense bank size does not match sampling_bank_size")
            record["dense"] = dense
        key = f"{sequence_dir}|{side}"
        self._open_records[key] = record
        self._open_records.move_to_end(key)
        while len(self._open_records) > _OPEN_SEQUENCE_LIMIT:
            self._open_records.popitem(last=False)
        return record

    def _get_record(self, sequence_dir: str, side: str) -> dict[str, Any]:
        key = f"{sequence_dir}|{side}"
        record = self._open_records.get(key)
        if record is None:
            record = self._load_record(sequence_dir, side)
        else:
            self._open_records.move_to_end(key)
        return record

    # ---- Dataset API ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def sample_location(self, index: int) -> tuple[Path, int]:
        """返回样本 (sequence 目录, 当前帧索引)，便于评估脚本按位置读取。"""
        sequence_dir, _side, current = self._samples[index][:3]
        return Path(sequence_dir), current

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        location = self._samples[index]
        sequence_dir, side, current = location[:3]
        cache = self._get_cache(sequence_dir)
        record = self._get_record(sequence_dir, side)
        meta = self._seq_meta[sequence_dir]
        raw_frame = int(cache.raw_frame_id[current])
        seed_args = dict(
            base_seed=self.base_seed,
            seq_id=str(meta.get("seq_id", Path(sequence_dir).name)),
            side=side,
            raw_frame_id=raw_frame,
            epoch=self.epoch,
        )
        stride_seed = stable_frame_seed(**seed_args, namespace="cm-stride")
        stride = (
            int(np.random.default_rng(stride_seed).choice(location[3]))
            if len(location) == 4 and isinstance(location[3], tuple) else
            int(location[3]) if len(location) == 4 else
            self.fixed_stride if self.fixed_stride is not None else
            int(np.random.default_rng(stride_seed).integers(self.min_stride, self.max_stride + 1))
        )
        future = current + stride
        bank = (
            self.fixed_bank if self.fixed_bank is not None else
            stable_frame_seed(**seed_args, namespace="cm-scene-bank") % self.sampling_bank_size
        )
        bank_indices = record["bank"]
        points, normals, valid = gather_frame_scene_inputs(
            cache, record["side_arrays"], bank_indices, frame=current, bank=bank,
        )
        # 未来端：同一个 selected_idx 访问 scene pool 的 future 行（V1.md §7）。
        selected = np.asarray(bank_indices[current, bank], dtype=np.int64)
        safe = np.where(selected != INVALID_INDEX, selected, 0)
        pose = np.asarray(record["side_arrays"]["hand_root_pose_world"][current], dtype=np.float32)
        scene_future_world = np.concatenate([
            np.asarray(cache.obj_points_world[future], dtype=np.float32),
            np.asarray(cache.env_points_world, dtype=np.float32),
        ])
        obj_future = _world_to_hand(scene_future_world[safe], pose)
        obj_future[~valid] = 0.0
        hand_current = _world_to_hand(np.asarray(record["side_arrays"]["hand_points_world"][current], dtype=np.float32), pose)
        hand_future = _world_to_hand(np.asarray(record["side_arrays"]["hand_points_world"][future], dtype=np.float32), pose)
        sample: dict[str, torch.Tensor] = {
            "obj_points": torch.from_numpy(points),
            "obj_normals": torch.from_numpy(normals),
            "obj_flow_gt": torch.from_numpy(obj_future - points),
            "hand_points": torch.from_numpy(hand_current),
            "hand_normals": torch.from_numpy(
                _normal_world_to_hand(np.asarray(record["side_arrays"]["hand_normals_world"][current], dtype=np.float32), pose)
            ),
            "hand_flow": torch.from_numpy(hand_future - hand_current),
            "obj_valid_mask": torch.from_numpy(valid),
            "selected_obj_idx": torch.from_numpy(selected.astype(np.int64)),
            # 0 = object, 1 = environment；绝不输入模型，仅用于 eval 拆分诊断。
            "scene_source_id": torch.from_numpy(np.asarray(cache.scene_source_id[safe], dtype=np.int64)),
            "sampling_bank": torch.tensor(int(bank)),
            "raw_frame_id": torch.tensor(raw_frame),
            "next_raw_frame_id": torch.tensor(int(cache.raw_frame_id[future])),
            "stride": torch.tensor(stride),
            "delta_time_s": torch.tensor(float(stride) / float(self.effective_fps), dtype=torch.float32),
        }
        if self.use_dense_cache:
            dense = record["dense"]
            dense_row = int(dense["frame_to_dense"][current])
            if dense_row < 0:
                raise RuntimeError(f"{sequence_dir}/{side} frame {current} has no dense-cache row")
            sample["cached_z_obj"] = torch.from_numpy(np.asarray(dense["z_scene"][dense_row, bank], dtype=np.float32))
            sample["cached_z_hand"] = torch.from_numpy(np.asarray(dense["z_hand"][dense_row, bank], dtype=np.float32))
            sample["cached_hand_contact"] = torch.from_numpy(np.asarray(dense["hand_contact"][dense_row, bank], dtype=np.float32))
        return sample


def _scene_flow_norm(
    cache: SceneSequenceCache,
    current: int,
    future: int,
    candidate: np.ndarray,
) -> np.ndarray:
    scene_current = np.concatenate([
        np.asarray(cache.obj_points_world[current], dtype=np.float32),
        np.asarray(cache.env_points_world, dtype=np.float32),
    ])
    scene_future = np.concatenate([
        np.asarray(cache.obj_points_world[future], dtype=np.float32),
        np.asarray(cache.env_points_world, dtype=np.float32),
    ])
    return np.linalg.norm(scene_future[candidate] - scene_current[candidate], axis=-1)


def _sequence_group_key(path: Path) -> str:
    """Keep both hand streams of one sequence in the same data split."""
    if path.name == "raw_frame_id.npy" and path.parent.name == "shared":
        return path.parents[1].as_posix()
    return path.parent.as_posix()


def _ensure_sequence_disjoint_splits(
    train_dirs: Sequence[str],
    val_dirs: Sequence[str],
    test_dirs: Sequence[str],
) -> None:
    named_sets = {"train": set(train_dirs), "val": set(val_dirs), "test": set(test_dirs)}
    overlaps = {
        "train_val": named_sets["train"] & named_sets["val"],
        "train_test": named_sets["train"] & named_sets["test"],
        "val_test": named_sets["val"] & named_sets["test"],
    }
    invalid = {name: paths for name, paths in overlaps.items() if paths}
    if invalid:
        raise ValueError(f"Cm scene sequence splits overlap: {sorted(next(iter(invalid.values())))[:5]}")


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any | None = None):
    """Scene-cache train/val/test loaders with fixed-stride fixed-bank eval views."""
    root = Path(str(getattr(data_cfg, "root", "") or data_cfg.train_path)).resolve()
    # Scene V1.1 uses one physical root and may intentionally leave
    # ``train_path`` empty.  The shared file-split helper otherwise interprets
    # an empty path as the repository cwd and can discover unrelated caches.
    if not str(getattr(data_cfg, "train_path", "") or "").strip():
        data_cfg.train_path = str(root)
    sampling_bank_size = int(getattr(data_cfg, "sampling_bank_size", 4))
    use_dense_cache = bool(getattr(data_cfg, "use_dense_cache", False))
    root_meta = read_meta(root)
    dense_fingerprint = None
    if use_dense_cache:
        dense_fingerprint = dense_cache_fingerprint(
            checkpoint_sha=sha256_file(str(meta_cfg.dense_checkpoint)),
            schema=SCHEMA_NAME,
            sampling_fingerprint=scene_cache_fingerprint(
                schema=SCHEMA_NAME,
                points_per_asset=int(root_meta["points_per_asset"]),
                candidate_threshold_m=float(root_meta["candidate_threshold_m"]),
                sampling_seed=int(root_meta.get("sampling_seed", -1)),
            ),
            num_scene_points=int(meta_cfg.num_obj_points),
            num_hand_points=int(meta_cfg.num_hand_points),
            bank_size=sampling_bank_size,
        )
    validate_scene_root(
        root,
        num_scene_points=int(meta_cfg.num_obj_points),
        sampling_bank_size=sampling_bank_size,
        use_dense_cache=use_dense_cache,
        dense_fingerprint=dense_fingerprint,
    )
    common = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "base_seed": int(seed),
        "active_only": bool(getattr(data_cfg, "active_only", True)),
        "min_stride": int(getattr(data_cfg, "min_stride", 1)),
        "max_stride": int(getattr(data_cfg, "max_stride", 10)),
        "coordinate_frame": str(meta_cfg.coordinate_frame),
        "min_object_flow_norm": float(getattr(data_cfg, "min_object_flow_norm", 0.0)),
        "sampling_bank_size": sampling_bank_size,
        "use_dense_cache": use_dense_cache,
    }
    train_loader, val_loader, test_loader, metadata = make_file_split_dataloaders(
        data_cfg, seed, dataset_cls=Stage4CmSceneDataset,
        file_pattern="**/shared/raw_frame_id.npy",
        train_dataset_kwargs={
            **common,
            "max_samples": getattr(data_cfg, "max_train_samples", None),
            "train_strides": getattr(data_cfg, "train_strides", None),
        },
        val_dataset_kwargs={**common, "fixed_stride": int(getattr(data_cfg, "val_stride", 1)),
                            "fixed_bank": int(getattr(data_cfg, "fixed_eval_bank", 0)),
                            "max_samples": getattr(data_cfg, "max_val_samples", None)},
        test_dataset_kwargs={**common, "fixed_stride": int(getattr(data_cfg, "test_stride", 1)),
                             "fixed_bank": int(getattr(data_cfg, "fixed_eval_bank", 0)),
                             "max_samples": getattr(data_cfg, "max_test_samples", None)},
        split_group_fn=_sequence_group_key if bool(getattr(data_cfg, "group_val_by_sequence", True)) else None,
        distributed=distributed,
    )
    _ensure_sequence_disjoint_splits(
        [Path(p).resolve().as_posix() for p in train_loader.dataset.sequence_dirs],
        [] if val_loader is None else [Path(p).resolve().as_posix() for p in val_loader.dataset.sequence_dirs],
        [] if test_loader is None else [Path(p).resolve().as_posix() for p in test_loader.dataset.sequence_dirs],
    )
    first = train_loader.dataset
    first_cache = first._get_cache(first.sequence_dirs[0])
    metadata.update({
        "schema_name": SCHEMA_NAME,
        "coordinate_frame": "hand_root_t",
        "num_obj_pool": first_cache.num_scene_pool,
        "num_env_pool": first_cache.num_env_pool,
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "sampling_bank_size": sampling_bank_size,
        "use_dense_cache": use_dense_cache,
        "min_stride": common["min_stride"],
        "max_stride": common["max_stride"],
        "ds_rate": int(first.ds_rate),
        "source_fps": float(first.source_fps),
        "effective_fps": float(first.effective_fps),
    })
    calibration_path = Path(metadata["train_path"]) / "metadata.json"
    if calibration_path.exists():
        try:
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{calibration_path} is not valid JSON.") from exc
        if not isinstance(calibration, dict):
            raise ValueError(f"{calibration_path} must contain a JSON object.")
        metadata.update({
            key: calibration[key]
            for key in (
                "flow_target_rms_m", "flow_target_scale", "statistics_split",
                "statistics_active_only", "statistics_num_obj_points",
                "statistics_stride_distribution", "statistics_stride_weighting",
                "statistics_point_weighting",
            )
            if key in calibration
        })

    def make_stride_loaders(loader: DataLoader | None, *, root_path: Any, prefix: str, max_samples: Any) -> dict[str, DataLoader]:
        if loader is None:
            return {}
        held_out_list = loader.dataset.file_list
        stride_loaders: dict[str, DataLoader] = {}
        for stride in tuple(getattr(data_cfg, "val_strides", (1, 2, 3, 4, 5))):
            stride = int(stride)
            if not common["min_stride"] <= stride <= common["max_stride"]:
                continue
            dataset = Stage4CmSceneDataset(
                root_path, file_list=held_out_list, fixed_stride=stride,
                fixed_bank=int(getattr(data_cfg, "fixed_eval_bank", 0)),
                max_samples=max_samples, **common,
            )
            loader_kwargs = {"batch_size": int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                             "shuffle": False, "num_workers": int(getattr(data_cfg, "num_workers", 0)),
                             "pin_memory": bool(getattr(data_cfg, "pin_memory", False)),
                             "sampler": make_default_eval_sampler(dataset, distributed=distributed)}
            if loader_kwargs["num_workers"] > 0:
                loader_kwargs["persistent_workers"] = bool(getattr(data_cfg, "persistent_workers", False))
                prefetch = getattr(data_cfg, "prefetch_factor", None)
                if prefetch is not None:
                    loader_kwargs["prefetch_factor"] = int(prefetch)
            stride_loaders[f"{prefix}/stride_{stride}/"] = DataLoader(dataset, **loader_kwargs)
        if not stride_loaders:
            stride_loaders[f"{prefix}/stride_1/"] = loader
        return stride_loaders

    split_root = getattr(data_cfg, "root", None)
    val_loaders = make_stride_loaders(
        val_loader, root_path=data_cfg.val_path or data_cfg.train_path or split_root,
        prefix="val", max_samples=getattr(data_cfg, "max_val_samples", None),
    )
    test_loaders = make_stride_loaders(
        test_loader, root_path=data_cfg.test_path or split_root,
        prefix="test", max_samples=getattr(data_cfg, "max_test_samples", None),
    )
    metadata["val_loader_names"] = sorted(val_loaders)
    metadata["test_loader_names"] = sorted(test_loaders)
    return train_loader, val_loader, test_loader, metadata, val_loaders, test_loaders
