"""Runtime-stride Cm samples from cached complete GRAB sequences.

从缓存的完整 GRAB 序列中按运行时 stride 采样，构造 Cm（contact-motion /
correspondence-motion）任务样本。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base import make_default_eval_sampler, make_file_split_dataloaders
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed


# Cm cache separates sequence-shared object states from hand-side states.
SHARED_SCHEMA_NAME = "ref2dex_cm_sequence_shared"
HAND_SCHEMA_NAME = "ref2dex_cm_sequence_hand"
REQUIRED_SHARED_FIELDS = {
    "schema_name", "raw_frame_id", "obj_points_world", "obj_normals_world", "obj_point_id",
    "ds_rate", "source_fps", "coordinate_frame", "seq_id",
}
REQUIRED_HAND_FIELDS = {
    "schema_name", "side", "hand_points_world", "hand_normals_world", "hand_root_pose_world",
    "obj_candidate_mask_5cm",
}


def scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
    """
    从 npz 字段中读取长度为 1 的字符串标量，找不到时返回 default。
    长度为 1 的字符串标量」= shape=() 、 size=1 、dtype 是字符串的 0-d 数组。
    它是 npz 里用来表示「单个字符串字段」的标准形式，取值时必须 .item() 才能拿到真正的 Python str 。
    """
    value = data.get(key)
    if value is None:
        return default
    value = np.asarray(value)
    return str(value.item()) if value.size == 1 else default


def _world_to_hand(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """Map world points to the current ``hand_root_t`` frame.

    把世界坐标下的点变换到当前帧的 ``hand_root_t``（手根坐标系）下：
    ``x_hand = R^T @ (x_world - t)``，其中 ``(R, t)`` 来自 ``hand_root_pose_world``。
    """
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float32)
    translation = np.asarray(pose_world[:3, 3], dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normal_world_to_hand(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """把世界坐标下的法向旋转到 ``hand_root_t`` 帧（法向只旋转不平移），并重新归一化。"""
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float32)
    result = np.asarray(normals, dtype=np.float32) @ rotation
    return (result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


class Stage4CmDataset(Dataset):
    """One active current frame, with endpoint stride chosen at runtime.

    一个样本对应「当前帧」加一个「未来帧」，两者 stride 在运行时按 seed 决定。
    缓存的 npz 里只存世界坐标系下的状态；监督量在两个端点都被表达在
    ``hand_root_t``（当前帧的手根系）下之后再构造。
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
        # 当前帧到未来帧之间的最小/最大间隔，按缓存后的时间轴计数。
        min_stride: int = 1,
        max_stride: int = 12,
        # 验证时若指定 fixed_stride，则每个样本都使用同一个 stride，便于对比
        fixed_stride: int | None = None,
        # 可选：只取前 N 个样本，方便做小型实验
        max_samples: int | None = None,
        coordinate_frame: str = "hand_root_t",
    ) -> None:
        if min_stride <= 0 or max_stride < min_stride:
            raise ValueError("Require 0 < min_stride <= max_stride.")
        if fixed_stride is not None and not min_stride <= fixed_stride <= max_stride:
            raise ValueError("fixed_stride must lie in [min_stride, max_stride].")
        self.data_path = Path(data_path)
        # data_root：若 data_path 是文件则回退到父目录，用于定位 meta.npz 等
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.base_seed = int(base_seed)
        # DataLoader 的 persistent worker 会拿到 Dataset 副本；用共享 Value
        # 让 BaseRunner.set_epoch 写入的 epoch 对所有副本都可见。
        self._epoch = mp.Value("q", 0, lock=True)
        self.active_only = bool(active_only)
        self.min_stride, self.max_stride = int(min_stride), int(max_stride)
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        self.coordinate_frame = str(coordinate_frame)
        # Collect hand-side files; their sibling shared.npz is merged on load.
        self.file_paths = (sorted(Path(path) for path in file_list) if file_list is not None else
                           (sorted(self.data_path.glob("**/*.npz")) if self.data_path.is_dir() else [self.data_path]))
        self.file_paths = [path for path in self.file_paths if path.name in {"left.npz", "right.npz"}]
        if not self.file_paths:
            raise ValueError(f"No Cm hand-side NPZ files found in {self.data_path}")
        self._cached_sequence_path: Path | None = None
        self._cached_shared_data: dict[str, np.ndarray] | None = None
        self._cached_side_path: Path | None = None
        self._cached_hand_data: dict[str, np.ndarray] | None = None
        self.ds_rate: int | None = None
        self.source_fps: float | None = None
        self.effective_fps: float | None = None
        # 预展开 (file, frame_idx) 样本列表，便于 __len__/__getitem__ 直接索引
        self._samples: list[tuple[Path, int]] = []
        for path in self.file_paths:
            shared_path = path.parent / "shared.npz"
            if not shared_path.exists():
                raise FileNotFoundError(f"{path}: missing shared sequence file {shared_path}")
            with np.load(shared_path, allow_pickle=False) as shared, np.load(path, allow_pickle=False) as hand:
                missing_shared = REQUIRED_SHARED_FIELDS.difference(shared.files)
                missing_hand = REQUIRED_HAND_FIELDS.difference(hand.files)
                if missing_shared:
                    raise KeyError(f"{shared_path}: missing shared fields {sorted(missing_shared)}")
                if missing_hand:
                    raise KeyError(f"{path}: missing hand fields {sorted(missing_hand)}")
                if str(np.asarray(shared["schema_name"]).item()) != SHARED_SCHEMA_NAME:
                    raise ValueError(f"{shared_path}: expected schema {SHARED_SCHEMA_NAME!r}")
                if str(np.asarray(hand["schema_name"]).item()) != HAND_SCHEMA_NAME:
                    raise ValueError(f"{path}: expected schema {HAND_SCHEMA_NAME!r}")
                if str(np.asarray(shared["coordinate_frame"]).item()) != "world":
                    raise ValueError(f"{shared_path}: cached sequence coordinates must be world-frame")
                ds_rate = int(np.asarray(shared["ds_rate"]).item())
                source_fps = float(np.asarray(shared["source_fps"]).item())
                if ds_rate <= 0 or source_fps <= 0.0:
                    raise ValueError(f"{shared_path}: ds_rate and source_fps must be positive")
                if shared["obj_points_world"].shape[1:] != (4096, 3):
                    raise ValueError(f"{shared_path}: expected obj_points_world [T,4096,3]")
                if hand["hand_points_world"].shape[1:] != (self.num_hand_points, 3):
                    raise ValueError(f"{path}: unexpected hand point shape")
                frame_count = int(shared["raw_frame_id"].shape[0])
                if hand["obj_candidate_mask_5cm"].shape != (frame_count, 4096):
                    raise ValueError(f"{path}: candidate mask must have shape [{frame_count},4096]")
                if hand["hand_points_world"].shape[0] != frame_count:
                    raise ValueError(f"{path}: hand frame count must match {shared_path}")
                if self.ds_rate is None:
                    self.ds_rate, self.source_fps = ds_rate, source_fps
                    self.effective_fps = source_fps / ds_rate
                elif (self.ds_rate, self.source_fps) != (ds_rate, source_fps):
                    raise ValueError("All Cm cache files in one Dataset must use the same ds_rate and source_fps")
                candidate = np.asarray(hand["obj_candidate_mask_5cm"], dtype=bool)
                for current in range(0, frame_count - self.max_stride):
                    if not self.active_only or candidate[current].any():
                        self._samples.append((path, current))
        if max_samples is not None and int(max_samples) > 0:
            self._samples = self._samples[:int(max_samples)]
        if not self._samples:
            raise ValueError("No active frames with the configured maximum future stride.")

    def __len__(self) -> int:
        return len(self._samples)

    def set_epoch(self, epoch: int) -> None:
        """供 Runner 调用的 epoch 切换接口；写入共享 Value 让所有 worker 副本可见。"""
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        """Load and merge one side file with its sequence-level shared state."""
        sequence_path = path.parent
        if self._cached_sequence_path != sequence_path or self._cached_shared_data is None:
            with np.load(sequence_path / "shared.npz", allow_pickle=False) as data:
                self._cached_shared_data = {key: np.asarray(data[key]) for key in data.files}
            self._cached_sequence_path = sequence_path
            self._cached_side_path = None
            self._cached_hand_data = None
        if self._cached_side_path != path or self._cached_hand_data is None:
            with np.load(path, allow_pickle=False) as data:
                self._cached_hand_data = {key: np.asarray(data[key]) for key in data.files}
            self._cached_side_path = path
        return {**self._cached_shared_data, **self._cached_hand_data}

    def sample_location(self, index: int) -> tuple[Path, int]:
        """返回样本 (npz 路径, 当前帧在文件内索引)，便于评估脚本按位置读取。"""
        return self._samples[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # 1) 读出当前帧 raw_frame_id，作为采样的稳定随机源
        path, current = self._samples[index]
        data = self._load_file(path)
        raw_frame = int(data["raw_frame_id"][current])
        # 用稳定 hash 派生两个独立随机流：stride 选择 + 物点采样
        seed_args = dict(base_seed=self.base_seed, seq_id=scalar_string(data, "seq_id", path.stem),
                         side=scalar_string(data, "side", ""), raw_frame_id=raw_frame, epoch=self.epoch)
        stride_seed = stable_frame_seed(**seed_args, namespace="cm-stride")
        object_seed = stable_frame_seed(**seed_args, namespace="cm-object-sampling")
        # 验证模式走 fixed_stride；训练模式从 [min_stride, max_stride] 闭区间随机
        stride = self.fixed_stride if self.fixed_stride is not None else int(
            np.random.default_rng(stride_seed).integers(self.min_stride, self.max_stride + 1)
        )
        future = current + stride
        # 当前帧手根位姿作为世界->hand_root_t 的变换源
        pose = data["hand_root_pose_world"][current]
        # 5cm 候选物点 mask：仅从这些点中再采 512 个作为运行时物点
        candidate = np.asarray(data["obj_candidate_mask_5cm"][current], dtype=bool)
        selected_idx, valid = sample_object_indices(candidate, num_samples=self.num_obj_points, seed=object_seed)
        safe = np.maximum(selected_idx, 0)
        # 把世界系下的物点 / 物点法向都转到当前帧 hand_root_t
        obj_current = _world_to_hand(data["obj_points_world"][current, safe], pose)
        obj_future = _world_to_hand(data["obj_points_world"][future, safe], pose)
        obj_normals = _normal_world_to_hand(data["obj_normals_world"][current, safe], pose)
        # 无效物点用 0 占位，模型/损失侧会用 valid mask 跳过
        obj_current[~valid] = obj_future[~valid] = obj_normals[~valid] = 0.0
        # 手部：当前帧与未来帧都转到 hand_root_t，flow = future - current
        hand_current = _world_to_hand(data["hand_points_world"][current], pose)
        hand_future = _world_to_hand(data["hand_points_world"][future], pose)
        return {
            "obj_points": torch.from_numpy(obj_current), "obj_normals": torch.from_numpy(obj_normals),
            "obj_flow_gt": torch.from_numpy(obj_future - obj_current),
            "hand_points": torch.from_numpy(hand_current),
            "hand_normals": torch.from_numpy(_normal_world_to_hand(data["hand_normals_world"][current], pose)),
            "hand_flow": torch.from_numpy(hand_future - hand_current),
            "obj_valid_mask": torch.from_numpy(valid), "selected_obj_idx": torch.from_numpy(selected_idx.astype(np.int64)),
            "raw_frame_id": torch.tensor(raw_frame), "next_raw_frame_id": torch.tensor(int(data["raw_frame_id"][future])),
            "stride": torch.tensor(stride),
        }


def _sequence_group_key(path: Path) -> str:
    """Keep left/right files of one shared sequence in the same data split."""
    return path.parent.as_posix()


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any | None = None):
    """构造训练 / 验证 DataLoader，并为每个 val_stride 单独构造一个 view。

    验证分多种 horizon 评估，所以同一个序列在不同 fixed_stride 下重复构造 dataset。
    """
    common = {"num_obj_points": int(meta_cfg.num_obj_points), "num_hand_points": int(meta_cfg.num_hand_points),
              "base_seed": int(seed), "active_only": bool(getattr(data_cfg, "active_only", True)),
              "min_stride": int(getattr(data_cfg, "min_stride", 1)), "max_stride": int(getattr(data_cfg, "max_stride", 12)),
              "coordinate_frame": str(meta_cfg.coordinate_frame)}
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg, seed, dataset_cls=Stage4CmDataset, file_pattern="**/*.npz",
        train_dataset_kwargs={**common, "max_samples": getattr(data_cfg, "max_train_samples", None)},
        val_dataset_kwargs={**common, "fixed_stride": int(getattr(data_cfg, "val_stride", 1)),
                            "max_samples": getattr(data_cfg, "max_val_samples", None)},
        split_group_fn=_sequence_group_key if bool(getattr(data_cfg, "group_val_by_sequence", True)) else None,
        distributed=distributed,
    )
    # Read shared metadata through the Dataset merger rather than side NPZ alone.
    first = train_loader.dataset.file_paths[0]
    data = train_loader.dataset._load_file(first)
    metadata.update({"schema_name": str(np.asarray(data["schema_name"]).item()), "coordinate_frame": "hand_root_t",
                     "num_obj_pool": int(data["obj_points_world"].shape[1]), "num_obj_points": int(meta_cfg.num_obj_points),
                     "num_hand_points": int(data["hand_points_world"].shape[1]), "min_stride": common["min_stride"],
                     "max_stride": common["max_stride"], "ds_rate": int(train_loader.dataset.ds_rate),
                     "source_fps": float(train_loader.dataset.source_fps),
                     "effective_fps": float(train_loader.dataset.effective_fps)})
    val_loaders: dict[str, DataLoader] = {}
    if val_loader is not None:
        # Evaluation is deterministic and reports each fixed horizon separately.
        # Constructing views from the held-out file list preserves the original
        # sequence-level split without materializing pair files.
        val_paths = val_loader.dataset.file_paths
        for stride in tuple(getattr(data_cfg, "val_strides", (1, 2, 3, 4, 5))):
            stride = int(stride)
            if not common["min_stride"] <= stride <= common["max_stride"]:
                continue
            dataset = Stage4CmDataset(
                data_cfg.val_path or data_cfg.train_path, file_list=val_paths,
                fixed_stride=stride, max_samples=getattr(data_cfg, "max_val_samples", None), **common,
            )
            loader_kwargs = {"batch_size": int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                             "shuffle": False, "num_workers": int(getattr(data_cfg, "num_workers", 0)),
                             "pin_memory": bool(getattr(data_cfg, "pin_memory", False)),
                             # Unlike the original shared val_loader, these
                             # per-stride views are constructed here.  Shard
                             # them too so DDP ranks do not duplicate the full
                             # 12-stride evaluation workload.
                             "sampler": make_default_eval_sampler(dataset, distributed=distributed)}
            if loader_kwargs["num_workers"] > 0:
                loader_kwargs["persistent_workers"] = bool(getattr(data_cfg, "persistent_workers", False))
                prefetch = getattr(data_cfg, "prefetch_factor", None)
                if prefetch is not None:
                    loader_kwargs["prefetch_factor"] = int(prefetch)
            val_loaders[f"val/stride_{stride}/"] = DataLoader(dataset, **loader_kwargs)
    if val_loader is not None and not val_loaders:
        val_loaders["val/stride_1/"] = val_loader
    metadata["val_loader_names"] = sorted(val_loaders)
    return train_loader, val_loader, metadata, val_loaders
