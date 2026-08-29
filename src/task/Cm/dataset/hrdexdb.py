"""HRDexDB-to-Cm adapter and configurable balanced source mixture.

The adapter consumes the normalized ``cmdecoder_layered_v4`` geometry cache
(``*_world.npy`` fields).  This is an input-geometry cache only: DenseToken
outputs are never read from disk, so the Cm model can unfreeze DenseToken and
backpropagate through its online forward pass.  It intentionally does not
depend on a particular robot hand: the cache builder is responsible for
producing the common 1538-hand-point contract, while the loader keeps the
embodiment name only as diagnostic metadata.  Object flow is constructed
online in the current hand-root frame, exactly like the object-v2 Cm dataset.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, make_default_train_sampler
from src.task.Cm.dataset import _normal_world_to_hand, _world_to_hand
from src.task.Cm.dataset.object_v2 import _MmapSequenceDataset, _resolve_splits, _sequence_dirs
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _resolve_manifest_episode(root: Path, relative: str) -> Path:
    """Resolve both the published ``v4/episodes/...`` and flat cache layouts."""
    rel = Path(relative)
    candidates = [root / rel, root.parent / rel, root.parent.parent / rel]
    for candidate in candidates:
        if (candidate / "geometry").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"HRDexDB manifest entry has no geometry cache: {relative}")


def _manifest_specs(
    root: Path,
    manifest: Path | None,
    split: str,
    source_prefix: str | None = None,
) -> list[tuple[str, Path]]:
    if manifest is None:
        manifests = sorted(root.glob("episodes/*/geometry/manifest.json"))
        if not manifests:
            raise FileNotFoundError(
                f"No HRDexDB geometry manifests under {root}; provide data.hrdexdb_manifest."
            )
        # A cache without a split descriptor is valid for a smoke run only.
        result = [(str(_load_json(path)["episode"]), path.parent.parent) for path in manifests]
        if source_prefix:
            prefix = source_prefix.rstrip("/") + "/"
            result = [item for item in result if item[0].startswith(prefix)]
        if not result:
            raise ValueError(f"No HRDexDB episodes match source_prefix={source_prefix!r} in split={split!r}")
        return result
    payload = _load_json(manifest)
    cache_dirs = payload.get("cache_dirs")
    splits = payload.get("splits")
    if not isinstance(cache_dirs, dict) or not isinstance(splits, dict):
        raise ValueError(f"HRDexDB manifest must contain cache_dirs and splits: {manifest}")
    episodes = splits.get(split)
    if not isinstance(episodes, list):
        raise ValueError(f"HRDexDB manifest has no {split!r} split: {manifest}")
    result = []
    for episode in episodes:
        if episode not in cache_dirs:
            raise KeyError(f"HRDexDB split episode missing cache_dirs entry: {episode}")
        result.append((str(episode), _resolve_manifest_episode(root, str(cache_dirs[episode]))))
    if source_prefix:
        prefix = source_prefix.rstrip("/") + "/"
        result = [item for item in result if item[0].startswith(prefix)]
    if not result:
        raise ValueError(f"No HRDexDB episodes match source_prefix={source_prefix!r} in split={split!r}")
    return result


class HrdexdbGeometryDataset(Dataset):
    """Random-horizon Cm samples from normalized HRDexDB geometry episodes."""

    def __init__(
        self,
        episodes: Sequence[tuple[str, Path]],
        *,
        num_obj_points: int = 512,
        num_obj_pool: int = 4096,
        num_hand_points: int = 1538,
        base_seed: int = 42,
        min_stride: int = 1,
        max_stride: int = 10,
        fixed_stride: int | None = None,
        active_only: bool = False,
        dataset_id: str = "hrdexdb",
    ) -> None:
        if num_obj_points != 512 or num_hand_points != 1538:
            raise ValueError("Cm/HRDexDB currently requires 512 object and 1538 hand points")
        if num_obj_pool < num_obj_points:
            raise ValueError("HRDexDB object pool must be at least the runtime object sample size")
        if min_stride <= 0 or max_stride < min_stride:
            raise ValueError("Invalid HRDexDB stride range")
        self.num_obj_points = int(num_obj_points)
        self.num_obj_pool = int(num_obj_pool)
        self.num_hand_points = int(num_hand_points)
        self.base_seed = int(base_seed)
        self.min_stride = int(min_stride)
        self.max_stride = int(max_stride)
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        self.active_only = bool(active_only)
        self.dataset_id = str(dataset_id)
        self._epoch = mp.Value("q", 0, lock=True)
        self.episodes = [(str(name), Path(path)) for name, path in episodes]
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
        self.rows: list[tuple[int, int]] = []
        for episode_index, (_, path) in enumerate(self.episodes):
            geometry = path / "geometry"
            required = [
                "hand_points_world.npy", "hand_normals_world.npy",
                "obj_points_pool_world.npy", "obj_normals_pool_world.npy",
                "wrist_pose_world.npy",
            ]
            missing = [name for name in required if not (geometry / name).is_file()]
            if missing:
                raise FileNotFoundError(f"HRDexDB episode {path} missing {missing}")
            frame_count = int(np.load(geometry / "frame_time.npy", mmap_mode="r").shape[0])
            candidate_path = geometry / "obj_candidate_mask_5cm.npy"
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    f"HRDexDB episode {path} has no 5cm candidate mask; rebuild the geometry layer with --num-obj-pool 4096"
                )
            candidate = np.load(candidate_path, mmap_mode="r")
            if candidate.shape != (frame_count, self.num_obj_pool):
                raise ValueError(
                    f"HRDexDB candidate mask must be [{frame_count},{self.num_obj_pool}], got {candidate.shape} at {path}"
                )
            # Keep a conservative max-stride row bank; the exact random stride
            # is selected at __getitem__ time for epoch-varying sampling.
            self.rows.extend(
                (episode_index, frame)
                for frame in range(max(0, frame_count - self.max_stride))
                if not self.active_only or bool(np.asarray(candidate[frame]).any())
            )
        if not self.rows:
            raise ValueError("No valid HRDexDB geometry transitions")

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, episode_index: int) -> dict[str, np.ndarray]:
        _, path = self.episodes[episode_index]
        if path not in self._cache:
            geometry = path / "geometry"
            loaded = {
                "hand": np.load(geometry / "hand_points_world.npy", mmap_mode="r"),
                "hand_normals": np.load(geometry / "hand_normals_world.npy", mmap_mode="r"),
                "obj_pool": np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r"),
                "obj_pool_normals": np.load(geometry / "obj_normals_pool_world.npy", mmap_mode="r"),
                "candidate": np.load(geometry / "obj_candidate_mask_5cm.npy", mmap_mode="r"),
                "pose": np.load(geometry / "wrist_pose_world.npy", mmap_mode="r"),
                "time": np.load(geometry / "frame_time.npy", mmap_mode="r"),
                "source": np.load(geometry / "source_frame_id.npy", mmap_mode="r")
                if (geometry / "source_frame_id.npy").is_file() else None,
            }
            if loaded["hand"].ndim != 3 or loaded["hand"].shape[1:] != (self.num_hand_points, 3):
                raise ValueError(f"HRDexDB hand geometry must be [T,{self.num_hand_points},3], got {loaded['hand'].shape} at {path}")
            if loaded["obj_pool"].ndim != 3 or loaded["obj_pool"].shape[1:] != (self.num_obj_pool, 3):
                raise ValueError(f"HRDexDB object pool must be [T,{self.num_obj_pool},3], got {loaded['obj_pool'].shape} at {path}")
            if loaded["candidate"].shape != (loaded["hand"].shape[0], self.num_obj_pool):
                raise ValueError(f"HRDexDB candidate mask must be [T,{self.num_obj_pool}], got {loaded['candidate'].shape} at {path}")
            if loaded["hand_normals"].shape != loaded["hand"].shape or loaded["obj_pool_normals"].shape != loaded["obj_pool"].shape:
                raise ValueError(f"HRDexDB normals do not match geometry at {path}")
            if loaded["pose"].shape != (loaded["hand"].shape[0], 4, 4):
                raise ValueError(f"HRDexDB wrist_pose_world must be [T,4,4], got {loaded['pose'].shape} at {path}")
            self._cache[path] = loaded
        return self._cache[path]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        episode_index, current = self.rows[index]
        episode_name, _ = self.episodes[episode_index]
        data = self._load(episode_index)
        seed = (self.base_seed * 1000003 + self.epoch * 9176 + index * 7919) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        stride = self.fixed_stride or int(rng.integers(self.min_stride, self.max_stride + 1))
        future = current + stride
        if future >= data["hand"].shape[0]:
            # Rows are built for max_stride, but fixed validation strides may be
            # larger than that configured range; fail loudly instead of wrapping.
            raise IndexError(f"HRDexDB transition exceeds episode length: {episode_name} +{stride}")
        pose = np.asarray(data["pose"][current], dtype=np.float32)
        candidate = np.asarray(data["candidate"][current], dtype=bool)
        selected, valid = sample_object_indices(
            candidate,
            num_samples=self.num_obj_points,
            seed=seed ^ 0x5EED5EED,
        )
        if self.active_only and not bool(valid.any()):
            raise ValueError(f"HRDexDB episode has no valid 5cm candidate: {episode_name} frame={current}")
        safe = np.maximum(selected, 0)
        obj_now = _world_to_hand(data["obj_pool"][current, safe], pose)
        obj_future = _world_to_hand(data["obj_pool"][future, safe], pose)
        hand_now = _world_to_hand(data["hand"][current], pose)
        hand_future = _world_to_hand(data["hand"][future], pose)
        obj_normals = _normal_world_to_hand(data["obj_pool_normals"][current, safe], pose)
        hand_normals = _normal_world_to_hand(data["hand_normals"][current], pose)
        valid &= np.isfinite(obj_now).all(axis=-1) & np.isfinite(obj_future).all(axis=-1)
        obj_now = np.nan_to_num(obj_now, nan=0.0)
        obj_future = np.nan_to_num(obj_future, nan=0.0)
        obj_normals = np.nan_to_num(obj_normals, nan=0.0)
        hand_now = np.nan_to_num(hand_now, nan=0.0)
        hand_future = np.nan_to_num(hand_future, nan=0.0)
        hand_normals = np.nan_to_num(hand_normals, nan=0.0)
        obj_now[~valid] = obj_future[~valid] = obj_normals[~valid] = 0.0
        source = data["source"]
        raw = int(source[current]) if source is not None else int(current)
        next_raw = int(source[future]) if source is not None else int(future)
        delta = float(data["time"][future] - data["time"][current])
        if not np.isfinite(delta) or delta <= 0.0:
            delta = float(stride) / 30.0
        embodiment = episode_name.split("/", 1)[0] if "/" in episode_name else episode_name
        return {
            "obj_points": torch.from_numpy(obj_now.astype(np.float32)),
            "obj_normals": torch.from_numpy(obj_normals.astype(np.float32)),
            "obj_flow_gt": torch.from_numpy((obj_future - obj_now).astype(np.float32)),
            "obj_valid_mask": torch.from_numpy(valid.astype(np.bool_)),
            "selected_obj_idx": torch.from_numpy(selected.astype(np.int64)),
            "hand_points": torch.from_numpy(hand_now.astype(np.float32)),
            "hand_normals": torch.from_numpy(hand_normals.astype(np.float32)),
            "hand_flow": torch.from_numpy((hand_future - hand_now).astype(np.float32)),
            "raw_frame_id": torch.tensor(raw, dtype=torch.long),
            "next_raw_frame_id": torch.tensor(next_raw, dtype=torch.long),
            "stride": torch.tensor(stride, dtype=torch.long),
            "delta_time_s": torch.tensor(delta, dtype=torch.float32),
            "dataset_id": self.dataset_id,
            "embodiment_id": embodiment,
            "episode_id": episode_name,
        }


class BalancedCmMixture(Dataset):
    """Virtual dataset with fixed outer source probabilities.

    Each virtual index draws one source according to ``probabilities`` and a
    deterministic local row.  DistributedSampler can therefore shard the
    virtual index space without changing the requested GRAB/ARCTIC/HRDexDB
    mixture.  HRDexDB's internal hand distribution is intentionally natural.
    """

    def __init__(self, sources: dict[str, Dataset], probabilities: dict[str, float], *, seed: int = 42):
        if not sources:
            raise ValueError("BalancedCmMixture requires at least one source")
        missing = [name for name in sources if name not in probabilities]
        if missing:
            raise ValueError(f"Missing source probabilities: {missing}")
        names = list(sources)
        weights = np.asarray([float(probabilities[name]) for name in names], dtype=np.float64)
        if np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("Source probabilities must be non-negative and non-zero")
        self.sources = sources
        self.names = names
        self.probabilities = weights / weights.sum()
        self.seed = int(seed)
        self._epoch = mp.Value("q", 0, lock=True)
        # One virtual epoch is the sum of physical rows, so each source gets
        # approximately p_i * total draws while preserving full throughput.
        self._length = max(1, sum(len(source) for source in sources.values()))

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)
        for source in self.sources.values():
            if hasattr(source, "set_epoch"):
                source.set_epoch(epoch)

    def __getitem__(self, index: int):
        seed = (self.seed * 1000003 + self.epoch * 9176 + int(index) * 7919) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        source_index = int(rng.choice(len(self.names), p=self.probabilities))
        name = self.names[source_index]
        source = self.sources[name]
        sample = dict(source[int(rng.integers(0, len(source)))])
        # Source-specific strings are useful in standalone HRDexDB loading but
        # cannot be default-collated with object-v2 rows that do not define
        # them.  dataset_id/source_bucket retain the required source identity.
        sample.pop("embodiment_id", None)
        sample.pop("episode_id", None)
        sample["source_bucket"] = name
        return sample

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)


def _object_split_dataset(root: Path, split_path: Path, split: str, *, data_cfg: Any, meta_cfg: Any, seed: int):
    sequence_dirs = _sequence_dirs(root)
    old_root = getattr(data_cfg, "split_json_path", None)
    try:
        data_cfg.split_json_path = str(split_path)
        train, val, test = _resolve_splits(data_cfg, root, sequence_dirs, seed)
    finally:
        if old_root is None:
            try:
                delattr(data_cfg, "split_json_path")
            except AttributeError:
                pass
        else:
            data_cfg.split_json_path = old_root
    dirs = {"train": train, "val": val, "test": test}[split]
    kwargs = dict(num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points),
                  base_seed=int(seed), min_stride=int(getattr(data_cfg, "min_stride", 1)),
                  max_stride=int(getattr(data_cfg, "max_stride", 10)),
                  active_only=bool(getattr(data_cfg, "active_only", True)),
                  sampling_bank_size=int(getattr(data_cfg, "sampling_bank_size", 4)),
                  fixed_eval_bank=int(getattr(data_cfg, "fixed_eval_bank", 0)))
    return _MmapSequenceDataset(dirs, fixed_stride=None if split == "train" else int(getattr(data_cfg, "val_stride", 1)), **kwargs)


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed=None):
    """Build the C64 HRDexDB fine-tune loaders.

    Required config fields are ``base_root``, ``base_split_json_path``,
    ``hrdexdb_root`` and ``hrdexdb_manifest``.  Base sequences are grouped by
    their existing ``dataset_id``. ``base_source_names`` selects the allowed
    base buckets, while ``hrdexdb_bucket_name`` names the filtered HRDexDB
    source in training and metrics.
    """
    # DenseToken outputs must be produced online when the backbone is
    # trainable.  Geometry on disk is allowed; ``cached_z_*`` is not.
    if bool(getattr(data_cfg, "use_dense_cache", False)):
        raise ValueError(
            "HRDexDB Cm fine-tuning requires online DenseToken execution; "
            "set data.use_dense_cache=false so gradients reach DenseToken."
        )
    if not bool(getattr(meta_cfg, "freeze_dense_encoder", True)) and bool(
        getattr(data_cfg, "use_dense_cache", False)
    ):
        raise ValueError("Trainable DenseToken cannot consume cached z_obj/z_hand features")
    hrdexdb_only = bool(getattr(data_cfg, "hrdexdb_only", False))
    source_prefix = getattr(data_cfg, "hrdexdb_source_prefix", None)
    if source_prefix is not None:
        source_prefix = str(source_prefix).strip() or None
    hrdexdb_bucket_name = str(
        getattr(data_cfg, "hrdexdb_bucket_name", "hrdexdb")
    ).strip().lower()
    if not hrdexdb_bucket_name:
        raise ValueError("data.hrdexdb_bucket_name must be non-empty")
    hroot = Path(str(getattr(data_cfg, "hrdexdb_root"))).resolve()
    manifest = Path(str(getattr(data_cfg, "hrdexdb_manifest"))).resolve()
    common = dict(num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points),
                  base_seed=int(seed), min_stride=int(getattr(data_cfg, "min_stride", 1)),
                  max_stride=int(getattr(data_cfg, "max_stride", 10)), active_only=bool(getattr(data_cfg, "active_only", True)),
                  sampling_bank_size=int(getattr(data_cfg, "sampling_bank_size", 4)), fixed_eval_bank=int(getattr(data_cfg, "fixed_eval_bank", 0)))
    hcommon = {key: common[key] for key in ("num_obj_points", "num_hand_points", "base_seed", "min_stride", "max_stride", "active_only")}
    hcommon["num_obj_pool"] = int(getattr(meta_cfg, "num_obj_pool", 4096))
    hcommon["dataset_id"] = hrdexdb_bucket_name

    if hrdexdb_only:
        htrain = HrdexdbGeometryDataset(
            _manifest_specs(hroot, manifest, "train", source_prefix), **hcommon
        )
        train_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
        train_kwargs.update(
            batch_size=int(data_cfg.batch_size),
            shuffle=distributed is None,
            sampler=None if distributed is None else make_default_train_sampler(
                htrain, shuffle=True, seed=seed, distributed=distributed, drop_last=False
            ),
        )
        train_loader = DataLoader(htrain, **train_kwargs)

        def eval_loader(dataset):
            kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
            kwargs.update(
                batch_size=int(getattr(data_cfg, "val_batch_size", data_cfg.batch_size)),
                shuffle=False,
                sampler=None if distributed is None else make_default_eval_sampler(dataset, distributed=distributed),
            )
            return DataLoader(dataset, **kwargs)

        val_strides = tuple(int(value) for value in getattr(data_cfg, "val_strides", (1, 5, 10)))
        test_strides = tuple(int(value) for value in getattr(data_cfg, "test_strides", val_strides))
        val_loaders = {}
        test_loaders = {}
        for stride in val_strides:
            dataset = HrdexdbGeometryDataset(
                _manifest_specs(hroot, manifest, "val", source_prefix),
                fixed_stride=stride,
                **hcommon,
            )
            val_loaders[f"val/stride_{stride}/{hrdexdb_bucket_name}/"] = eval_loader(dataset)
        for stride in test_strides:
            dataset = HrdexdbGeometryDataset(
                _manifest_specs(hroot, manifest, "test", source_prefix),
                fixed_stride=stride,
                **hcommon,
            )
            test_loaders[f"test/stride_{stride}/{hrdexdb_bucket_name}/"] = eval_loader(dataset)
        metadata = {
            "schema_name": "ref2dex_cm_hrdexdb_finetune_v1",
            "coordinate_frame": "hand_root_t",
            "num_obj_pool": int(getattr(meta_cfg, "num_obj_pool", 4096)),
            "num_obj_points": int(meta_cfg.num_obj_points),
            "num_hand_points": int(meta_cfg.num_hand_points),
            "source_probabilities": {hrdexdb_bucket_name: 1.0},
            "hrdexdb_manifest": str(manifest),
            "hrdexdb_source_prefix": source_prefix,
            "dataset_split": {
                "hrdexdb_train_episodes": len(_manifest_specs(hroot, manifest, "train", source_prefix)),
                "hrdexdb_val_episodes": len(_manifest_specs(hroot, manifest, "val", source_prefix)),
                "hrdexdb_test_episodes": len(_manifest_specs(hroot, manifest, "test", source_prefix)),
            },
            "val_strides": list(val_strides),
            "test_strides": list(test_strides),
        }
        return train_loader, next(iter(val_loaders.values())), next(iter(test_loaders.values())), metadata, val_loaders, test_loaders

    base_root = Path(str(getattr(data_cfg, "base_root"))).resolve()
    base_split = Path(str(getattr(data_cfg, "base_split_json_path"))).resolve()
    # Build the base split once, then keep the two source buckets separate.
    sequence_dirs = _sequence_dirs(base_root)
    old_split = getattr(data_cfg, "split_json_path", None)
    try:
        data_cfg.split_json_path = str(base_split)
        base_train, base_val, base_test = _resolve_splits(data_cfg, base_root, sequence_dirs, seed)
    finally:
        if old_split is None:
            try:
                delattr(data_cfg, "split_json_path")
            except AttributeError:
                pass
        else:
            data_cfg.split_json_path = old_split
    common = dict(num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points),
                  base_seed=int(seed), min_stride=int(getattr(data_cfg, "min_stride", 1)),
                  max_stride=int(getattr(data_cfg, "max_stride", 10)), active_only=bool(getattr(data_cfg, "active_only", True)),
                  sampling_bank_size=int(getattr(data_cfg, "sampling_bank_size", 4)), fixed_eval_bank=int(getattr(data_cfg, "fixed_eval_bank", 0)))
    hcommon = {key: common[key] for key in ("num_obj_points", "num_hand_points", "base_seed", "min_stride", "max_stride", "active_only")}
    hcommon["num_obj_pool"] = int(getattr(meta_cfg, "num_obj_pool", 4096))
    hcommon["dataset_id"] = hrdexdb_bucket_name
    def group_paths(dirs: Iterable[Path]) -> dict[str, list[Path]]:
        result: dict[str, list[Path]] = {}
        for path in dirs:
            try:
                name = str(json.loads((path / "shared" / "meta.json").read_text(encoding="utf-8")).get("dataset_name", "unknown")).lower()
            except (OSError, json.JSONDecodeError):
                name = "unknown"
            bucket = "grab" if "grab" in name else "arctic" if "arctic" in name else name
            result.setdefault(bucket, []).append(path)
        return {name: paths for name, paths in result.items() if paths}
    base_train_paths = group_paths(base_train)
    base_val_paths = group_paths(base_val)
    base_test_paths = group_paths(base_test)
    requested_base_sources = tuple(
        str(value).strip().lower()
        for value in getattr(data_cfg, "base_source_names", ("grab", "arctic"))
    )
    if not requested_base_sources or any(not value for value in requested_base_sources):
        raise ValueError("data.base_source_names must contain at least one non-empty source")
    expected_base_sources = set(requested_base_sources)
    if set(base_train_paths) != expected_base_sources:
        raise ValueError(
            "Base train buckets do not match data.base_source_names: "
            f"expected={sorted(expected_base_sources)}, got={sorted(base_train_paths)}"
        )
    for split_name, split_paths in (("val", base_val_paths), ("test", base_test_paths)):
        if set(split_paths) != expected_base_sources:
            raise ValueError(
                f"Base {split_name} buckets do not match data.base_source_names: "
                f"expected={sorted(expected_base_sources)}, got={sorted(split_paths)}"
            )
    htrain = HrdexdbGeometryDataset(_manifest_specs(hroot, manifest, "train", source_prefix), **hcommon)
    all_source_names = [*requested_base_sources, hrdexdb_bucket_name]
    if len(set(all_source_names)) != len(all_source_names):
        raise ValueError("Base sources and hrdexdb_bucket_name must be distinct")
    probs = {name: 1.0 / len(all_source_names) for name in all_source_names}
    custom_probs = getattr(data_cfg, "source_probabilities", None)
    if custom_probs is not None:
        values = custom_probs.items() if isinstance(custom_probs, dict) else vars(custom_probs).items()
        probs.update({str(k): float(v) for k, v in values})
    train_sources = {
        name: _MmapSequenceDataset(paths, **common)
        for name, paths in base_train_paths.items()
    }
    train_dataset = BalancedCmMixture(
        {**train_sources, hrdexdb_bucket_name: htrain}, probs, seed=seed
    )
    train_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
    train_kwargs.update(batch_size=int(data_cfg.batch_size), shuffle=distributed is None,
                        sampler=None if distributed is None else make_default_train_sampler(train_dataset, shuffle=True, seed=seed, distributed=distributed, drop_last=False))
    train_loader = DataLoader(train_dataset, **train_kwargs)
    def eval_loader(dataset):
        kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
        kwargs.update(batch_size=int(getattr(data_cfg, "val_batch_size", data_cfg.batch_size)), shuffle=False,
                      sampler=None if distributed is None else make_default_eval_sampler(dataset, distributed=distributed))
        return DataLoader(dataset, **kwargs)
    val_strides = tuple(int(value) for value in getattr(data_cfg, "val_strides", (1, 5, 10)))
    test_strides = tuple(int(value) for value in getattr(data_cfg, "test_strides", val_strides))
    if not val_strides or not test_strides:
        raise ValueError("HRDexDB fine-tuning requires non-empty val_strides and test_strides")
    val_loaders: dict[str, Any] = {}
    test_loaders: dict[str, Any] = {}
    for stride in val_strides:
        for name, paths in base_val_paths.items():
            ds = _MmapSequenceDataset(paths, **{**common, "fixed_stride": stride})
            val_loaders[f"val/stride_{stride}/{name}/"] = eval_loader(ds)
        hval = HrdexdbGeometryDataset(
            _manifest_specs(hroot, manifest, "val", source_prefix), fixed_stride=stride, **hcommon
        )
        val_loaders[f"val/stride_{stride}/{hrdexdb_bucket_name}/"] = eval_loader(hval)
    for stride in test_strides:
        for name, paths in base_test_paths.items():
            ds = _MmapSequenceDataset(paths, **{**common, "fixed_stride": stride})
            test_loaders[f"test/stride_{stride}/{name}/"] = eval_loader(ds)
        htest = HrdexdbGeometryDataset(
            _manifest_specs(hroot, manifest, "test", source_prefix), fixed_stride=stride, **hcommon
        )
        test_loaders[f"test/stride_{stride}/{hrdexdb_bucket_name}/"] = eval_loader(htest)
    metadata = {"schema_name": "ref2dex_cm_hrdexdb_finetune_v1", "coordinate_frame": "hand_root_t",
                "num_obj_pool": int(getattr(meta_cfg, "num_obj_pool", 4096)),
                "num_obj_points": int(meta_cfg.num_obj_points),
                "num_hand_points": int(meta_cfg.num_hand_points),
                "source_probabilities": probs, "hrdexdb_manifest": str(manifest),
                "hrdexdb_source_prefix": source_prefix,
                "hrdexdb_bucket_name": hrdexdb_bucket_name,
                "dataset_split": {"base_train_sequences": len(base_train), "hrdexdb_train_episodes": len(_manifest_specs(hroot, manifest, "train", source_prefix))},
                "val_strides": list(val_strides), "test_strides": list(test_strides)}
    first_val = next(iter(val_loaders.values()), None)
    first_test = next(iter(test_loaders.values()), None)
    return train_loader, first_val, first_test, metadata, val_loaders, test_loaders
