from __future__ import annotations

import json
import multiprocessing as mp
import os
import copy
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.base import make_file_split_dataloaders
from src.base.data import make_dataloader_kwargs, resolve_data_path
from src.base.distributed import make_default_eval_sampler, shard_sampler_for_distributed
from src.task.correspondence_ptv3_v2.sampling import (
    perturb_object_geometry,
    perturb_hand_root_geometry,
    sample_object_indices,
    sample_random_supervision_edges,
    stable_frame_seed,
)
from src.task.correspondence_ptv3_v2.hand_noise_profiles import (
    infer_stage3_dataset_id,
)
from src.task.correspondence_ptv3_v2.robot_recon import perturb_robot_hand_from_frame


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    """Accept both YAML dicts and the ConfigNode objects produced by load_config."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return dict(value)


# Cross-dataset MANO pose width. GRAB stores PCA24 and ARCTIC stores
# axis-angle45; the dataset pads everything to this length so PyTorch's
# default collate can stack the per-frame mano_pose vectors into a
# single (B, MANO_POSE_MAX_DIM) tensor. The runner slices back to the
# active representation length per group before calling smplx.MANO.
MANO_POSE_MAX_DIM = 45


class CorrStaticDatasetV2(Dataset):
    REQUIRED_FIELDS = {
        "seq_id",
        "side",
        "raw_frame_id",
        "obj_points",
        "obj_normals",
        "hand_points",
        "hand_normals",
        "hand_to_obj_min_dist",
        "coordinate_frame",
    }

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        num_supervision_edges: int = 128,
        contact_supervision_quotas: tuple[int, int, int, int] = (16, 16, 16, 16),
        contact_supervision_hard_negative_quota: int = 16,
        contact_supervision_hard_negative_distance_range: tuple[float, float] = (0.02, 0.03),
        contact_radius: float = 0.02,
        base_seed: int = 42,
        apply_obj_perturb: bool = True,
        apply_hand_perturb: bool = True,
        apply_hand_root_perturb: bool = False,
        hand_root_rot_std_deg: float = 10.0,
        hand_root_trans_std: float = 0.01,
        hand_root_perturb_prob: float = 1.0,
        exclusive_hand_object_perturb: bool = False,
        hand_perturb_prob: float = 0.8,
        obj_rot_std_deg: float = 10.0,
        obj_trans_std: float = 0.01,
        obj_perturb_prob: float = 1.0,
        runtime_resample_object: bool = True,
        # Fix #8 (docs/指导.md): opt-in strict schema gate. When True the
        # dataset refuses to load a Stage 3 root whose npz is below v2.1.0
        # or missing any required MANO field. When False the legacy v2.0
        # behaviour is preserved (hand_points are used as-is, no MANO
        # forward) so old pipelines keep working.
        use_mano_reconstruction: bool = False,
        use_robot_reconstruction: bool = False,
        apply_robot_perturb: bool = False,
        robot_perturb_prob: float = 1.0,
        mixed_hand_reconstruction: bool = False,
        robot_perturb_noise_scale_by_domain: dict[str, float] | None = None,
        exclusive_perturb_mode_probs: tuple[float, float, float] = (0.4, 0.4, 0.2),
        filter_non_interacting_frames: bool = False,
        interaction_max_distance_m: float = 0.05,
        blacklist_path: str | None = None,
        eval_sampling_epoch: int | None = None,
        coordinate_frame: str | None = None,
        transform_to_object_frame: bool = False,
        dataset_id: str | None = None,
        cache_index_path: str | Path | None = None,
        array_cache_path: str | Path | None = None,
        array_cache_required: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.num_supervision_edges = int(num_supervision_edges)
        if len(tuple(contact_supervision_quotas)) != 4:
            raise ValueError(
                f"contact_supervision_quotas must have 4 entries (weak/medium/strong/very_strong), "
                f"got {tuple(contact_supervision_quotas)}."
            )
        self.contact_supervision_quotas = tuple(int(q) for q in contact_supervision_quotas)
        self.contact_supervision_hard_negative_quota = int(contact_supervision_hard_negative_quota)
        self.contact_supervision_hard_negative_distance_range = tuple(
            float(x) for x in contact_supervision_hard_negative_distance_range
        )
        self.contact_radius = float(contact_radius)
        self.base_seed = int(base_seed)
        self.apply_obj_perturb = bool(apply_obj_perturb)
        self.apply_hand_perturb = bool(apply_hand_perturb)
        self.apply_hand_root_perturb = bool(apply_hand_root_perturb)
        self.hand_root_rot_std_deg = float(hand_root_rot_std_deg)
        self.hand_root_trans_std = float(hand_root_trans_std)
        self.hand_root_perturb_prob = float(hand_root_perturb_prob)
        self.exclusive_hand_object_perturb = bool(exclusive_hand_object_perturb)
        self.hand_perturb_prob = float(hand_perturb_prob)
        self.obj_rot_std_deg = float(obj_rot_std_deg)
        self.obj_trans_std = float(obj_trans_std)
        self.obj_perturb_prob = float(obj_perturb_prob)
        self.runtime_resample_object = bool(runtime_resample_object)
        self.use_mano_reconstruction = bool(use_mano_reconstruction)
        self.use_robot_reconstruction = bool(use_robot_reconstruction)
        self.apply_robot_perturb = bool(apply_robot_perturb)
        self.robot_perturb_prob = float(robot_perturb_prob)
        self.mixed_hand_reconstruction = bool(mixed_hand_reconstruction)
        self.exclusive_perturb_mode_probs = tuple(float(x) for x in exclusive_perturb_mode_probs)
        if len(self.exclusive_perturb_mode_probs) != 3 or any(x < 0 for x in self.exclusive_perturb_mode_probs):
            raise ValueError("exclusive_perturb_mode_probs must contain three non-negative values")
        if not np.isclose(sum(self.exclusive_perturb_mode_probs), 1.0, atol=1e-6):
            raise ValueError(
                "exclusive_perturb_mode_probs must sum to 1.0 "
                f"(got {self.exclusive_perturb_mode_probs})"
            )
        self.robot_perturb_noise_scale_by_domain = {
            str(key): float(value)
            for key, value in (robot_perturb_noise_scale_by_domain or {}).items()
        }
        self.filter_non_interacting_frames = bool(filter_non_interacting_frames)
        self.interaction_max_distance_m = float(interaction_max_distance_m)
        if self.interaction_max_distance_m <= 0.0:
            raise ValueError("interaction_max_distance_m must be positive.")
        self.dataset_id_override = None if dataset_id in {None, ""} else str(dataset_id)
        self.cache_index_path = None if cache_index_path in {None, ""} else Path(cache_index_path)
        self.array_cache_path = (
            None if array_cache_path in {None, ""} else Path(array_cache_path).expanduser().resolve()
        )
        self.array_cache_required = bool(array_cache_required)
        self.eval_sampling_epoch = None if eval_sampling_epoch is None else int(eval_sampling_epoch)
        self.transform_to_object_frame = bool(transform_to_object_frame)
        self._epoch = mp.Value("q", 0, lock=True)
        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None
        self.root_coordinate_frame = _load_root_coordinate_frame(self.data_root)

        if self.num_supervision_edges <= 0:
            raise ValueError("num_supervision_edges must be positive.")
        if self.contact_radius <= 0.0:
            raise ValueError("contact_radius must be positive.")

        paths = (
            sorted(Path(path) for path in file_list)
            if file_list is not None
            else (sorted(self.data_path.glob("**/*.npz")) if self.data_path.is_dir() else [self.data_path])
        )
        blacklist = _load_blacklist(blacklist_path)
        self.file_paths = [path for path in paths if not _is_blacklisted(path, self.data_root, blacklist)]
        if not self.file_paths:
            raise ValueError(f"No Stage 3 npz files found in {self.data_path}")

        # Per-NPZ coordinate_frame is the source of truth (not the root
        # meta.json, which is only written once at generation time). If the
        # caller specified an expected frame, every file must agree with it.
        # MANO fields are required only when the runtime will actually apply
        # hand perturbation.  ``schema_version`` is an identifier, not a
        # capability gate: clean legacy Stage 3 geometry remains valid when
        # the hand-noise path is disabled.
        per_file_frames: set[str] = set()
        require_mano_fields = bool(
            use_mano_reconstruction and apply_hand_perturb and not self.mixed_hand_reconstruction
        )
        mano_field_set = set(self._MANO_REQUIRED_FIELDS)
        robot_field_set = set(self._ROBOT_REQUIRED_FIELDS)
        for path in self.file_paths:
            with np.load(self._resolve_array_path(path), allow_pickle=False) as data:
                missing = self.REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
                if require_mano_fields:
                    self._validate_mano_schema(path, data)
                has_mano_fields = mano_field_set.issubset(set(data.files))
                has_robot_fields = robot_field_set.issubset(set(data.files))
                if self.mixed_hand_reconstruction:
                    if has_mano_fields == has_robot_fields:
                        raise KeyError(
                            f"{path}: mixed_hand_reconstruction requires exactly one complete "
                            f"MANO or robot field contract (has_mano={has_mano_fields}, "
                            f"has_robot={has_robot_fields})."
                        )
                elif self.use_robot_reconstruction:
                    missing_robot = sorted(robot_field_set.difference(data.files))
                    if missing_robot:
                        raise KeyError(
                            f"{path}: use_robot_reconstruction=True but the Stage 3 npz is missing "
                            f"robot fields {missing_robot}."
                        )
                frame = _resolve_coordinate_frame_from_npz(
                    data,
                    fallback=self.root_coordinate_frame,
                )
                if frame is None:
                    raise KeyError(
                        f"{path}: missing Stage 3 field 'coordinate_frame' and root meta.json "
                        "does not provide it. Re-run Stage 3 with the current "
                        "prepare_corr_static.py."
                    )
                per_file_frames.add(frame)

        if len(per_file_frames) != 1:
            raise ValueError(
                f"Inconsistent Stage 3 coordinate_frame across {self.data_root}: {sorted(per_file_frames)}"
            )
        self.source_coordinate_frame = next(iter(per_file_frames))
        self.coordinate_frame = self.source_coordinate_frame
        if coordinate_frame is not None and self.coordinate_frame != str(coordinate_frame):
            if not (
                self.transform_to_object_frame
                and str(coordinate_frame) == "object"
                and self.source_coordinate_frame == "hand_root"
            ):
                raise ValueError(
                    f"Stage 3 .npz at {self.data_root} was generated in "
                    f"coordinate_frame={self.coordinate_frame!r} but training expects "
                    f"{coordinate_frame!r}. Re-run Stage 3 with "
                    f"--coordinate-frame {coordinate_frame} on the same Stage 3 root."
                )
        if self.transform_to_object_frame and str(coordinate_frame) == "object":
            self.coordinate_frame = "object"

        self._samples: list[tuple[Path, int]] = []
        self.file_sample_ranges: list[tuple[int, int]] = []
        frame_counts = self._load_or_build_index(frame_paths=self.file_paths)
        filtered_frame_indices: list[np.ndarray] = []
        for path, num_frames in zip(self.file_paths, frame_counts):
            frame_indices = np.arange(int(num_frames), dtype=np.int64)
            if self.filter_non_interacting_frames:
                with np.load(self._resolve_array_path(path), allow_pickle=False) as data:
                    distances = np.asarray(data["hand_to_obj_min_dist"], dtype=np.float32)
                    if distances.ndim != 2 or distances.shape[0] != int(num_frames):
                        raise ValueError(
                            f"{path}: hand_to_obj_min_dist must have shape "
                            f"(T, H), got {distances.shape} for T={num_frames}."
                        )
                    frame_indices = np.flatnonzero(
                        np.min(distances, axis=1) <= self.interaction_max_distance_m
                    ).astype(np.int64)
            filtered_frame_indices.append(frame_indices)
        if self.filter_non_interacting_frames and not any(
            len(indices) > 0 for indices in filtered_frame_indices
        ):
            raise ValueError(
                f"All frames in {self.data_path} were filtered by interaction_max_distance_m="
                f"{self.interaction_max_distance_m}."
            )
        for path, frame_indices in zip(self.file_paths, filtered_frame_indices):
            start = len(self._samples)
            self._samples.extend((path, int(frame_idx)) for frame_idx in frame_indices.tolist())
            self.file_sample_ranges.append((start, len(self._samples)))

    def _load_or_build_index(self, *, frame_paths: list[Path]) -> list[int]:
        """Reuse a validated JSON file/frame-count index across training runs.

        The cache only accelerates dataset construction; sample arrays remain
        loaded through the existing per-worker NPZ cache, so stale or missing
        cache entries cannot change training data silently.
        """
        if self.cache_index_path is None:
            return [self._count_frames(path) for path in frame_paths]
        cache_path = self.cache_index_path
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            payload = {"version": 1, "files": {}}
        entries = payload.get("files", {})
        counts: list[int] = []
        changed = False
        for path in frame_paths:
            key = str(path.resolve())
            stat = path.stat()
            item = entries.get(key)
            valid = (
                item is not None
                and int(item.get("size", -1)) == int(stat.st_size)
                and float(item.get("mtime", -1)) == float(stat.st_mtime)
            )
            frames = int(item["frames"]) if valid else self._count_frames(path)
            counts.append(frames)
            if not valid:
                entries[key] = {
                    "size": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                    "frames": frames,
                }
                changed = True
        if changed:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "files": entries}
            tmp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
            tmp_path.replace(cache_path)
        return counts

    @staticmethod
    def _count_frames(path: Path) -> int:
        with np.load(path, allow_pickle=False) as data:
            return int(data["raw_frame_id"].shape[0])

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self._samples)

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        if self._cached_path == path and self._cached_data is not None:
            return self._cached_data
        with np.load(self._resolve_array_path(path), allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        # OakInk2 object-centered exports keep the frame-invariant canonical
        # object pool once per file as [P,3] instead of repeating it as
        # [T,P,3].  Materialize a broadcast view here so all downstream code
        # retains the regular per-frame indexing contract without duplicating
        # hundreds of gigabytes on disk.
        if "raw_frame_id" in payload:
            num_frames = int(np.asarray(payload["raw_frame_id"]).shape[0])
            for key in ("obj_points", "obj_normals"):
                value = payload.get(key)
                if value is not None and value.ndim == 2:
                    payload[key] = np.broadcast_to(value, (num_frames,) + value.shape)
        if self.transform_to_object_frame and self.source_coordinate_frame == "hand_root":
            if "obj_root_pose_world" not in payload or "hand_root_pose" not in payload:
                raise KeyError(f"{path}: runtime object-frame conversion requires both root poses")
            obj_pose = np.asarray(payload["obj_root_pose_world"], dtype=np.float64)
            hand_pose = np.asarray(payload["hand_root_pose"], dtype=np.float64)
            inv_obj = np.linalg.inv(obj_pose)
            rotation = np.einsum("tij,tjk->tik", inv_obj[:, :3, :3], hand_pose[:, :3, :3])
            translation = np.einsum("tij,tj->ti", inv_obj[:, :3, :3], hand_pose[:, :3, 3] - obj_pose[:, :3, 3])
            for key in ("obj_points", "hand_points"):
                points = np.asarray(payload[key], dtype=np.float64)
                payload[key] = np.einsum("tni,tji->tnj", points, rotation).astype(np.float32) + translation[:, None, :].astype(np.float32)
            for key in ("obj_normals", "hand_normals"):
                normals = np.asarray(payload[key], dtype=np.float64)
                transformed = np.einsum("tni,tji->tnj", normals, rotation)
                transformed /= np.clip(np.linalg.norm(transformed, axis=-1, keepdims=True), 1e-8, None)
                payload[key] = transformed.astype(np.float32)
            payload["coordinate_frame"] = np.asarray("object")
        self._cached_path = path
        self._cached_data = payload
        return payload

    def _resolve_array_path(self, source_path: Path) -> Path:
        if self.array_cache_path is None:
            return source_path
        try:
            relative = source_path.relative_to(self.data_root)
        except ValueError as exc:
            if self.array_cache_required:
                raise ValueError(
                    f"Source file {source_path} is outside dataset root {self.data_root}; "
                    "cannot resolve its array cache path."
                ) from exc
            return source_path
        cached = self.array_cache_path / relative
        if cached.is_file():
            # A sidecar is valid only if it was produced after the source
            # file. This cheap check prevents silently training on stale
            # arrays when a Stage 3 file is regenerated in place.
            try:
                if cached.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
                    return cached
            except OSError:
                pass
        if self.array_cache_required:
            raise FileNotFoundError(
                f"Required uncompressed array cache is missing or stale for {source_path}: {cached}"
            )
        return source_path

    @staticmethod
    def _scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
        value = data.get(key)
        if value is None:
            return default
        array = np.asarray(value)
        return str(array.item()) if array.size == 1 else default

    # ------------------------------------------------------------------
    # Fix #8 (docs/指导.md): strict schema validation when the train path
    # intends to re-run MANO forward. The Stage 3 .npz must carry every
    # required MANO field; partial data is a hard error, not a silent
    # fallback to the legacy hand_points field.
    # ------------------------------------------------------------------
    _MANO_REQUIRED_FIELDS: tuple[str, ...] = (
        "mano_global_orient",
        "mano_transl",
        "mano_pose",
        "mano_betas",
        "mano_v_template",
        "mano_use_pca",
        "mano_num_pca_comps",
        "mano_flat_hand_mean",
        "mano_pose_repr",
    )

    _ROBOT_REQUIRED_FIELDS: tuple[str, ...] = (
        "robot_qpos",
        "robot_hand_qpos_indices",
        "robot_qpos_lower",
        "robot_qpos_upper",
        "robot_hand_qpos_noise_std",
        "robot_hand_points_local",
        "robot_hand_normals_local",
        "robot_hand_point_link_index",
        "robot_link_names",
        "robot_c2r",
        "robot_urdf_path",
        "hand_root_pose",
    )

    @classmethod
    def _validate_mano_schema(
        cls,
        path: Path,
        data: Any,
    ) -> None:
        missing = [f for f in cls._MANO_REQUIRED_FIELDS if f not in data.files]
        if missing:
            raise KeyError(
                f"{path}: use_mano_reconstruction=True but the Stage 3 npz is missing "
                f"required MANO fields {missing}. Re-run Stage 3 with the current "
                "process/common/stage3_corr.py on the same Stage 2 root."
            )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, frame_idx = self._samples[index]
        data = self._load_file(path)
        seq_id = self._scalar_string(data, "seq_id", path.stem)
        side = self._scalar_string(data, "side", "")
        dataset_id = self.dataset_id_override or infer_stage3_dataset_id(data)
        raw_frame_id = int(np.asarray(data["raw_frame_id"])[frame_idx])
        epoch = self.eval_sampling_epoch if self.eval_sampling_epoch is not None else self.epoch

        sample_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=epoch,
        )
        # Legacy v2.0 data carries a clean-hand 5 cm candidate mask. New v2.1+
        # data deliberately omits it; use the full 4096 object pool as a
        # placeholder here and let the runner replace the 512 selected object
        # points with perturbed-hand proxy sampling.
        candidate_mask_data = data.get("obj_candidate_mask_5cm")
        has_legacy_candidate_mask = candidate_mask_data is not None
        # Runtime sampling must not consult clean GT candidate masks.  The
        # complete pool is perturbed first and sampled afterwards in the
        # runner, matching an online observation-time sampler.
        if self.runtime_resample_object:
            candidate_mask = np.ones((int(data["obj_points"].shape[-2]),), dtype=bool)
        elif has_legacy_candidate_mask:
            candidate_mask = np.asarray(candidate_mask_data[frame_idx], dtype=bool)
        else:
            candidate_mask = np.ones((int(data["obj_points"].shape[-2]),), dtype=bool)
        selected_idx, obj_valid = sample_object_indices(
            candidate_mask,
            num_samples=self.num_obj_points,
            seed=sample_seed,
        )
        safe_idx = np.maximum(selected_idx, 0)

        full_obj_points = np.asarray(data["obj_points"][frame_idx], dtype=np.float32).copy()
        full_obj_normals = np.asarray(data["obj_normals"][frame_idx], dtype=np.float32).copy()
        obj_points = np.asarray(full_obj_points[safe_idx], dtype=np.float32).copy()
        obj_normals = np.asarray(full_obj_normals[safe_idx], dtype=np.float32).copy()

        obj_points[~obj_valid] = 0
        obj_normals[~obj_valid] = 0

        clean_hand_points = np.asarray(data["hand_points"][frame_idx], dtype=np.float32).copy()
        clean_hand_normals = np.asarray(data["hand_normals"][frame_idx], dtype=np.float32).copy()
        hand_points = clean_hand_points.copy()
        hand_normals = clean_hand_normals.copy()
        robot_hand_input: tuple[np.ndarray, np.ndarray] | None = None
        robot_perturb_rms_m = 0.0
        # Exclusive augmentation modes: in the object-centered root/pose
        # protocol these are hand-root-only (0), hand-pose-only (1), clean
        # (2). Legacy hand/object configs retain their original meaning. A
        # stable frame seed makes the 40/40/20 split
        # independent of worker order and checkpoint resume.
        perturb_mode = 2
        if self.exclusive_hand_object_perturb and (
            self.apply_hand_perturb or self.apply_hand_root_perturb or self.apply_obj_perturb
        ):
            mode_seed = stable_frame_seed(
                base_seed=self.base_seed, seq_id=seq_id, side=side,
                raw_frame_id=raw_frame_id, epoch=epoch,
                namespace="exclusive-hand-object-perturbation",
            )
            mode_rng = np.random.default_rng(mode_seed)
            perturb_mode = int(mode_rng.choice(3, p=self.exclusive_perturb_mode_probs))
        if self.exclusive_hand_object_perturb and self.apply_hand_root_perturb:
            hand_root_perturb_this = bool(
                self.apply_hand_root_perturb and perturb_mode == 0
            )
            hand_perturb_this = bool(self.apply_hand_perturb and perturb_mode == 1)
            apply_obj_perturb_this = False
        elif self.exclusive_hand_object_perturb:
            hand_root_perturb_this = False
            hand_perturb_this = bool(self.apply_hand_perturb and perturb_mode == 0)
            apply_obj_perturb_this = bool(self.apply_obj_perturb and perturb_mode == 1)
        else:
            hand_root_perturb_this = bool(self.apply_hand_root_perturb)
            hand_perturb_this = bool(self.apply_hand_perturb)
            apply_obj_perturb_this = bool(self.apply_obj_perturb)

        hand_root_perturb_rms_m = 0.0
        if hand_root_perturb_this:
            if "hand_root_pose" not in data or "obj_root_pose_world" not in data:
                raise KeyError(
                    f"{path}: hand-root perturbation requires hand_root_pose and "
                    "obj_root_pose_world fields."
                )
            root_seed = stable_frame_seed(
                base_seed=self.base_seed,
                seq_id=seq_id,
                side=side,
                raw_frame_id=raw_frame_id,
                epoch=epoch,
                namespace="hand-root-perturbation",
            )
            perturbed_hand, perturbed_normals, _root_applied = perturb_hand_root_geometry(
                hand_points=hand_points,
                hand_normals=hand_normals,
                hand_root_pose_world=np.asarray(data["hand_root_pose"][frame_idx], dtype=np.float32),
                obj_root_pose_world=np.asarray(data["obj_root_pose_world"][frame_idx], dtype=np.float32),
                seed=root_seed,
                apply_perturb=True,
                rot_std_deg=self.hand_root_rot_std_deg,
                trans_std=self.hand_root_trans_std,
                perturb_prob=self.hand_root_perturb_prob,
            )
            hand_root_perturb_rms_m = float(np.sqrt(np.mean((perturbed_hand - hand_points) ** 2)))
            hand_points = perturbed_hand
            hand_normals = perturbed_normals
        has_robot_payload = all(key in data for key in self._ROBOT_REQUIRED_FIELDS)
        if self.use_robot_reconstruction and self.apply_robot_perturb and has_robot_payload and hand_perturb_this:
            robot_seed = stable_frame_seed(
                base_seed=self.base_seed,
                seq_id=seq_id,
                side=side,
                raw_frame_id=raw_frame_id,
                epoch=epoch,
                namespace="robot-q-perturbation",
            )
            robot_points, robot_normals, _robot_applied = perturb_robot_hand_from_frame(
                data,
                frame_idx=frame_idx,
                seed=robot_seed,
                apply_perturb=True,
                perturb_prob=self.robot_perturb_prob,
                noise_scale=self.robot_perturb_noise_scale_by_domain.get(
                    self.dataset_id_override or "", 1.0
                ),
                coordinate_frame=self.coordinate_frame,
            )
            robot_hand_input = (robot_points, robot_normals)
            robot_perturb_rms_m = float(
                np.sqrt(np.mean((robot_points - hand_points) ** 2))
            )
        hand_min_dist = _resolve_hand_to_obj_min_dist(data, frame_idx=frame_idx)
        if int(hand_min_dist.numel()) != int(self.num_hand_points):
            raise ValueError(
                f"{path}: hand_to_obj_min_dist has {int(hand_min_dist.numel())} entries, "
                f"but num_hand_points={self.num_hand_points}."
            )
        # Keep all clean hand points within the interaction window. The fixed
        # tensor width remains 1538 for batching; invalid slots are masked.
        hand_valid_mask = (hand_min_dist <= self.interaction_max_distance_m).bool()
        aug_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=epoch,
            namespace="augmentation",
        )
        geometry = perturb_object_geometry(
            obj_points=obj_points,
            obj_normals=obj_normals,
            hand_points=clean_hand_points,
            hand_normals=clean_hand_normals,
            seed=aug_seed,
            apply_obj_perturb=apply_obj_perturb_this,
            obj_rot_std_deg=self.obj_rot_std_deg,
            obj_trans_std=self.obj_trans_std,
            obj_perturb_prob=self.obj_perturb_prob,
        )
        if hand_root_perturb_this:
            geometry = replace(
                geometry,
                input_hand_points=hand_points,
                input_hand_normals=hand_normals,
            )
        if robot_hand_input is not None:
            geometry = replace(
                geometry,
                input_hand_points=robot_hand_input[0],
                input_hand_normals=robot_hand_input[1],
            )
        # Runtime resampling mode: perturb the full 4096 pool first; the
        # runner then draws a uniform 512-point subset from this observation.
        # ``full_geometry`` carries the clean
        # (``gt_*``) and the perturbed (``input_*``) versions of the
        # entire object pool; the hand arrays here are unused because
        # hand perturbation is performed by MANO forward in the runner.
        full_geometry = None
        # Keep the batch schema identical across datasets. Some regenerated
        # v2.1 sources (notably ContactPose) still carry the legacy candidate
        # mask while GRAB/ARCTIC may omit it. Runtime resampling deliberately
        # ignores that optional GT-derived field.
        runtime_resample_this = self.runtime_resample_object
        if runtime_resample_this:
            full_geometry = perturb_object_geometry(
                obj_points=full_obj_points,
                obj_normals=full_obj_normals,
                hand_points=clean_hand_points,
                hand_normals=clean_hand_normals,
                seed=aug_seed,
                apply_obj_perturb=apply_obj_perturb_this,
                obj_rot_std_deg=self.obj_rot_std_deg,
                obj_trans_std=self.obj_trans_std,
                obj_perturb_prob=self.obj_perturb_prob,
            )
        # hand_to_obj_min_dist is fully frame-invariant AND independent of
        # the 512 obj sampling, so it is the clean absolute contact
        # target for the hand contact head.

        supervision_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=epoch,
            namespace="supervision-edges",
        )
        random_edge_idx, random_edge_valid = sample_random_supervision_edges(
            num_obj_points=self.num_obj_points,
            num_hand_points=self.num_hand_points,
            obj_valid_mask=obj_valid,
            num_supervision_edges=self.num_supervision_edges,
            seed=supervision_seed,
        )

        contact_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=epoch,
            namespace="contact-supervision-edges",
        )
        contact_seed = int(contact_seed & ((1 << 63) - 1))

        input_points = np.concatenate([geometry.input_obj_points, geometry.input_hand_points], axis=0)
        input_normals = np.concatenate([geometry.input_obj_normals, geometry.input_hand_normals], axis=0)
        gt_points = np.concatenate([geometry.gt_obj_points, geometry.gt_hand_points], axis=0)
        gt_normals = np.concatenate([geometry.gt_obj_normals, geometry.gt_hand_normals], axis=0)
        point_valid_mask = np.concatenate(
            [obj_valid, hand_valid_mask.numpy()],
            axis=0,
        )

        result: dict[str, torch.Tensor] = {
            "points": torch.from_numpy(input_points).float(),
            "normals": torch.from_numpy(input_normals).float(),
            "gt_points": torch.from_numpy(gt_points).float(),
            "gt_normals": torch.from_numpy(gt_normals).float(),
            "point_valid_mask": torch.from_numpy(point_valid_mask),
            "runtime_obj_valid_mask": torch.from_numpy(obj_valid),
            "random_edge_idx": torch.from_numpy(random_edge_idx).long(),
            "random_edge_valid_mask": torch.from_numpy(random_edge_valid),
            "hand_min_dist": hand_min_dist.float(),
            "hand_valid_mask": hand_valid_mask,
            "contact_seed": torch.tensor(contact_seed, dtype=torch.long),
            "num_obj_points": torch.tensor(self.num_obj_points, dtype=torch.long),
            "num_hand_points": torch.tensor(self.num_hand_points, dtype=torch.long),
            "perturb_mode": torch.tensor(perturb_mode, dtype=torch.long),
            "robot_perturb_rms_m": torch.tensor(robot_perturb_rms_m, dtype=torch.float32),
            "hand_root_perturb_rms_m": torch.tensor(hand_root_perturb_rms_m, dtype=torch.float32),
            "apply_hand_root_perturb": torch.tensor(bool(hand_root_perturb_this), dtype=torch.bool),
        }
        has_mano_payload = all(key in data for key in self._MANO_REQUIRED_FIELDS)
        has_robot_payload = all(key in data for key in self._ROBOT_REQUIRED_FIELDS)
        result["has_mano"] = torch.tensor(
            bool(self.use_mano_reconstruction and has_mano_payload), dtype=torch.bool
        )
        result["has_robot"] = torch.tensor(
            bool(self.use_robot_reconstruction and has_robot_payload), dtype=torch.bool
        )
        # Keep default-collate keys/shapes identical when a batch mixes MANO
        # and robot samples. Robot rows carry inert MANO placeholders; the
        # runner selects only ``has_mano`` rows before reconstruction.
        result["apply_hand_perturb"] = torch.tensor(
            bool(self.use_mano_reconstruction and has_mano_payload and hand_perturb_this),
            dtype=torch.bool,
        )
        result["hand_perturb_seed"] = torch.tensor(
            int(sample_seed & ((1 << 63) - 1)), dtype=torch.long
        )
        result["__mano_side__"] = side
        if not has_mano_payload:
            result["mano_global_orient"] = torch.zeros(3, dtype=torch.float32)
            result["mano_transl"] = torch.zeros(3, dtype=torch.float32)
            result["mano_pose"] = torch.zeros(MANO_POSE_MAX_DIM, dtype=torch.float32)
            result["mano_pose_dim"] = torch.tensor(45, dtype=torch.long)
            result["mano_betas"] = torch.zeros(10, dtype=torch.float32)
            result["mano_v_template"] = torch.zeros((778, 3), dtype=torch.float32)
            result["mano_v_template_sha"] = "__robot_placeholder__"
            result["mano_use_pca"] = torch.tensor(False, dtype=torch.bool)
            result["mano_num_pca_comps"] = torch.tensor(0, dtype=torch.long)
            result["mano_flat_hand_mean"] = torch.tensor(True, dtype=torch.bool)
            result["hand_root_pose_world"] = torch.eye(4, dtype=torch.float32)
        # Keep the optional object-root pose key present for every sample so
        # default_collate can merge mixed domains (e.g. OakInk files without
        # an object pose and GRAB files carrying one).  Hand-root runs do not
        # consume this field, so an identity placeholder is the neutral value.
        # Object-frame runs must never silently fall back to identity: their
        # coordinate conversion is only valid when the source pose exists.
        if "obj_root_pose_world" in data:
            result["obj_root_pose_world"] = torch.from_numpy(
                np.asarray(data["obj_root_pose_world"][frame_idx], dtype=np.float32)
            ).float()
        elif self.coordinate_frame == "object":
            raise KeyError(
                f"{path}: coordinate_frame='object' requires obj_root_pose_world; "
                "regenerate this Stage 3 sample with object root pose fields."
            )
        else:
            result["obj_root_pose_world"] = torch.eye(4, dtype=torch.float32)
        if full_geometry is not None:
            result["full_input_obj_points"] = torch.from_numpy(
                full_geometry.input_obj_points
            ).float()
            result["full_input_obj_normals"] = torch.from_numpy(
                full_geometry.input_obj_normals
            ).float()
            result["full_gt_obj_points"] = torch.from_numpy(
                full_geometry.gt_obj_points
            ).float()
            result["full_gt_obj_normals"] = torch.from_numpy(
                full_geometry.gt_obj_normals
            ).float()
            result["runtime_resample_object"] = torch.tensor(True, dtype=torch.bool)
            result["object_seed"] = torch.tensor(int(aug_seed & ((1 << 63) - 1)), dtype=torch.long)

        # v2.1+: pass MANO parameters through so the runner can re-run
        # MANO forward on the GPU and apply hand PCA perturbations.
        # The runner groups samples by (side, use_pca, num_pca_comps,
        # flat_hand_mean, v_template_sha) so a single batch can mix
        # GRAB (PCA24) and ARCTIC (axis-angle45) without crashing.
        if self.use_mano_reconstruction and "mano_pose" in data and "mano_global_orient" in data:
            # Fix #1 (docs/指导.md): explicitly tell the runner this sample
            # has MANO parameters, so it actually runs the reconstruction
            # path instead of falling back to the legacy hand_points.
            result["has_mano"] = torch.tensor(True, dtype=torch.bool)
            # Fix #2: per-sample apply_hand_perturb flag so val_clean
            # does not get the same noise injection as train.
            result["apply_hand_perturb"] = torch.tensor(
                hand_perturb_this, dtype=torch.bool
            )
            # Fix #4: a per-frame, per-epoch, per-side, per-namespace seed
            # for the hand-PCA noise generator. The runner uses this so the
            # noise is independent of batch composition / DataLoader order
            # and stable across checkpoint resumes.
            hand_perturb_seed = stable_frame_seed(
                base_seed=self.base_seed,
                seq_id=seq_id,
                side=side,
                raw_frame_id=raw_frame_id,
                epoch=epoch,
                namespace="hand-perturbation",
            )
            result["hand_perturb_seed"] = torch.tensor(
                int(hand_perturb_seed & ((1 << 63) - 1)), dtype=torch.long
            )
            result["__mano_side__"] = side
            result["mano_global_orient"] = torch.from_numpy(
                np.asarray(data["mano_global_orient"][frame_idx], dtype=np.float32)
            ).float()
            result["mano_transl"] = torch.from_numpy(
                np.asarray(data["mano_transl"][frame_idx], dtype=np.float32)
            ).float()
            # Cross-dataset batching: GRAB stores PCA24 while ARCTIC stores
            # axis-angle45.  PyTorch's default collate cannot stack variable
            # length vectors, so we pad to the MANO axis-angle width.  The
            # runner slices back to the active representation length per
            # sample/group before calling smplx.MANO.
            mano_pose = np.asarray(data["mano_pose"][frame_idx], dtype=np.float32)
            if mano_pose.ndim != 1:
                raise ValueError(f"mano_pose frame must be 1D, got shape={mano_pose.shape}")
            mano_pose_dim = int(mano_pose.shape[0])
            if mano_pose_dim < MANO_POSE_MAX_DIM:
                mano_pose_padded = np.zeros((MANO_POSE_MAX_DIM,), dtype=np.float32)
                mano_pose_padded[:mano_pose_dim] = mano_pose
                mano_pose = mano_pose_padded
            result["mano_pose"] = torch.from_numpy(mano_pose).float()
            result["mano_pose_dim"] = torch.tensor(mano_pose_dim, dtype=torch.long)
            result["mano_betas"] = torch.from_numpy(
                np.asarray(data["mano_betas"][frame_idx], dtype=np.float32)
            ).float()
            # mano_v_template is per-subject (shape (778, 3)); we ship
            # the subject-specific template so the runner builds a
            # subject-aware MANO layer on first sight.
            v_template = data.get("mano_v_template")
            if v_template is not None:
                v_template_arr = np.asarray(v_template, dtype=np.float32)
                result["mano_v_template"] = torch.from_numpy(v_template_arr).float()
                # Fix #6: pre-compute the SHA1 digest on CPU so the
                # runner does not have to copy a GPU tensor and run
                # SHA1 per sample every batch.  The digest is what the
                # runner groups on; the actual ``mano_v_template``
                # tensor is still shipped so the layer can be built on
                # first sight.  Stored as a fixed-width ASCII string so
                # PyTorch's default collate stacks it into a list
                # without per-sample dtype/shape issues.
                import hashlib

                sha1 = hashlib.sha1(
                    np.ascontiguousarray(v_template_arr, dtype=np.float32).tobytes()
                ).hexdigest()
                result["mano_v_template_sha"] = sha1
            # hand_root_pose_world (T, 4, 4) is needed to bring the
            # MANO forward output from world back to the hand_root
            # frame the rest of the pipeline expects.
            if "hand_root_pose" in data:
                result["hand_root_pose_world"] = torch.from_numpy(
                    np.asarray(data["hand_root_pose"][frame_idx], dtype=np.float32)
                ).float()
            result["mano_use_pca"] = torch.tensor(
                bool(np.asarray(data.get("mano_use_pca", True)).item()),
                dtype=torch.bool,
            )
            result["mano_num_pca_comps"] = torch.tensor(
                int(np.asarray(data.get("mano_num_pca_comps", mano_pose_dim)).item()),
                dtype=torch.long,
            )
            result["mano_flat_hand_mean"] = torch.tensor(
                bool(np.asarray(data.get("mano_flat_hand_mean", True)).item()),
                dtype=torch.bool,
            )
        # Keep the domain tag present for clean mixed batches as well. It is
        # used for per-domain validation/logging and does not require MANO.
        result["__dataset_id__"] = dataset_id
        return result


class DomainConcatDataset(Dataset):
    """Concatenate per-domain datasets while retaining domain boundaries."""

    def __init__(self, datasets: list[CorrStaticDatasetV2], domain_ids: list[str]) -> None:
        if not datasets or len(datasets) != len(domain_ids):
            raise ValueError("DomainConcatDataset requires one id for every non-empty dataset.")
        if len(set(domain_ids)) != len(domain_ids):
            raise ValueError(f"Domain ids must be unique, got {domain_ids!r}.")
        self.datasets = list(datasets)
        self.domain_ids = [str(value) for value in domain_ids]
        self.cumulative_sizes: list[int] = []
        total = 0
        for dataset in self.datasets:
            if len(dataset) <= 0:
                raise ValueError("Domain datasets must not be empty.")
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        domain_idx = next(
            idx for idx, end in enumerate(self.cumulative_sizes) if index < end
        )
        start = 0 if domain_idx == 0 else self.cumulative_sizes[domain_idx - 1]
        return self.datasets[domain_idx][index - start]

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            dataset.set_epoch(epoch)


class DomainBalancedSampler(Sampler[int]):
    """Yield a fixed equal-domain stream for every local training batch.

    The sampler creates one independent stream per domain and shards each
    domain chunk across DDP ranks. Consequently every rank receives the same
    domain quota and the all-reduce sees a 1/3 mixture as well.
    """

    def __init__(
        self,
        dataset: DomainConcatDataset,
        *,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = True,
        sequence_locality: bool = True,
        domain_sampling_weights: tuple[float, ...] | list[float] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.drop_last = bool(drop_last)
        self.sequence_locality = bool(sequence_locality)
        if domain_sampling_weights in (None, (), []):
            weights = np.ones(len(dataset.datasets), dtype=np.float64)
        else:
            weights = np.asarray(domain_sampling_weights, dtype=np.float64)
            if weights.shape != (len(dataset.datasets),) or not np.all(np.isfinite(weights)):
                raise ValueError(
                    "domain_sampling_weights must contain one finite value per domain, "
                    f"got {domain_sampling_weights!r} for {len(dataset.datasets)} domains."
                )
            if np.any(weights <= 0.0):
                raise ValueError("domain_sampling_weights must be strictly positive.")
        normalized = weights / weights.sum()
        raw_quotas = normalized * self.batch_size
        quotas = np.floor(raw_quotas).astype(np.int64)
        remainder = int(self.batch_size - int(quotas.sum()))
        if remainder > 0:
            order = np.argsort(-(raw_quotas - quotas))
            quotas[order[:remainder]] += 1
        if np.any(quotas <= 0):
            raise ValueError(
                "Each domain must receive at least one sample per batch; "
                f"batch_size={self.batch_size}, weights={normalized.tolist()}."
            )
        self.domain_quotas = tuple(int(q) for q in quotas.tolist())
        self.domain_sampling_weights = tuple(float(x) for x in normalized.tolist())
        # ``steps_per_epoch`` is defined in global batches.  Each rank emits
        # one local quota, so account for world_size here; otherwise DDP
        # advances the epoch after roughly ``world_size`` dataset passes.
        self.steps_per_epoch = max(
            1,
            max(
                math.ceil(len(domain) / (quota * self.world_size))
                for domain, quota in zip(dataset.datasets, self.domain_quotas)
            ),
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch * self.batch_size

    @staticmethod
    def _cycle_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
        chunks: list[np.ndarray] = []
        remaining = int(count)
        while remaining > 0:
            permutation = rng.permutation(length)
            take = min(remaining, length)
            chunks.append(permutation[:take])
            remaining -= take
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.int64)

    @staticmethod
    def _sequence_local_indices(
        domain: CorrStaticDatasetV2,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample frames while keeping adjacent indices in the same NPZ.

        Domain balancing still happens at the batch level.  This only changes
        the order within each domain stream so the per-worker one-file cache
        can stay hot instead of reopening a random file for every sample.
        """
        ranges = domain.file_sample_ranges
        if not ranges:
            return np.empty((0,), dtype=np.int64)
        chunks: list[np.ndarray] = []
        remaining = int(count)
        while remaining > 0:
            for file_idx in rng.permutation(len(ranges)):
                start, end = ranges[int(file_idx)]
                if end <= start:
                    continue
                order = rng.permutation(np.arange(start, end, dtype=np.int64))
                take = min(remaining, len(order))
                chunks.append(order[:take])
                remaining -= take
                if remaining <= 0:
                    break
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.int64)

    def __iter__(self):
        offsets = [0]
        for domain in self.dataset.datasets[:-1]:
            offsets.append(offsets[-1] + len(domain))
        streams: list[np.ndarray] = []
        for domain_idx, domain in enumerate(self.dataset.datasets):
            rng = np.random.default_rng(
                stable_frame_seed(
                    base_seed=self.seed,
                    seq_id=f"domain-balanced-{domain_idx}",
                    side="",
                    raw_frame_id=0,
                    epoch=self.epoch,
                    namespace="sampler",
                )
            )
            domain_quota = self.domain_quotas[domain_idx]
            stream_count = self.steps_per_epoch * domain_quota * self.world_size
            if self.sequence_locality:
                streams.append(self._sequence_local_indices(domain, stream_count, rng))
            else:
                streams.append(self._cycle_indices(len(domain), stream_count, rng))

        for step in range(self.steps_per_epoch):
            for domain_idx, stream in enumerate(streams):
                domain_quota = self.domain_quotas[domain_idx]
                global_start = step * domain_quota * self.world_size
                local_start = global_start + self.rank * domain_quota
                local_indices = stream[local_start : local_start + domain_quota]
                offset = offsets[domain_idx]
                yield from (int(value) + offset for value in local_indices)


def _resolve_hand_to_obj_min_dist(
    data: dict[str, np.ndarray],
    *,
    frame_idx: int,
) -> torch.Tensor:
    cached = data.get("hand_to_obj_min_dist")
    if cached is None:
        raise KeyError(
            "Missing Stage 3 field 'hand_to_obj_min_dist'. Regenerate Stage 3 with the current "
            "prepare_corr_static.py before training v2."
        )
    return torch.from_numpy(np.asarray(cached[frame_idx], dtype=np.float32).copy())


def _load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    blacklist_path = Path(path)
    if not blacklist_path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")
    if blacklist_path.suffix.lower() == ".json":
        import json

        payload = json.loads(blacklist_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Blacklist JSON must contain a list")
        return {str(item) for item in payload}
    return {
        line.strip()
        for line in blacklist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _load_root_coordinate_frame(root: Path) -> str | None:
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None
    import json

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = payload.get("coordinate_frame")
    return None if frame in {None, ""} else str(frame)


def _resolve_coordinate_frame_from_npz(
    data: Any,
    *,
    fallback: str | None,
) -> str | None:
    if "coordinate_frame" in data.files:
        return str(np.asarray(data["coordinate_frame"]).item())
    return fallback


def _is_blacklisted(path: Path, root: Path, blacklist: set[str]) -> bool:
    if not blacklist:
        return False
    keys = {str(path), path.name, path.stem, path.as_posix()}
    try:
        keys.add(path.relative_to(root).as_posix())
    except ValueError:
        pass
    return bool(keys.intersection(blacklist))


def _sequence_group_key(path: Path) -> str:
    # OakInk conversion stores one NPZ per camera view.  The physical sample
    # identity is in seq_id (e.g. ``.../view0``), not in the filename alone;
    # group all views before assigning train/validation splits.
    try:
        with np.load(path, allow_pickle=False) as data:
            if "seq_id" in data.files:
                value = np.asarray(data["seq_id"])
                if value.size == 1:
                    import re
                    seq = str(value.item())
                    seq = re.sub(r"(?:/|_)view\d+$", "", seq)
                    if seq:
                        return seq
    except (OSError, ValueError, KeyError):
        pass
    stem = path.stem
    if stem.endswith("_left") or stem.endswith("_right"):
        stem = stem.rsplit("_", 1)[0]
    parent = path.parent.as_posix()
    return stem if parent in {"", "."} else f"{parent}/{stem}"


class SequenceLocalitySampler(Sampler[int]):
    def __init__(self, dataset: CorrStaticDatasetV2, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(
            stable_frame_seed(
                base_seed=self.seed,
                seq_id="sequence-locality-sampler",
                side="",
                raw_frame_id=0,
                epoch=self.epoch,
                namespace="frame-order",
            )
        )
        file_order = rng.permutation(len(self.dataset.file_sample_ranges))
        for file_idx in file_order:
            start, end = self.dataset.file_sample_ranges[int(file_idx)]
            yield from rng.permutation(np.arange(start, end, dtype=np.int64)).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def make_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any], dict[str, DataLoader]]:
    domain_specs = getattr(data_cfg, "domain_paths", None) or []
    if domain_specs:
        return _make_domain_balanced_dataloaders(
            data_cfg,
            seed,
            meta_cfg=meta_cfg,
            distributed=distributed,
            domain_specs=domain_specs,
        )

    expected_coordinate_frame = str(getattr(meta_cfg, "coordinate_frame", "hand_root"))
    runtime_resample_object = bool(getattr(meta_cfg, "runtime_resample_object", True))
    use_mano_reconstruction = bool(getattr(meta_cfg, "use_mano_reconstruction", False))
    use_robot_reconstruction = bool(getattr(meta_cfg, "use_robot_reconstruction", False))
    apply_robot_perturb = bool(getattr(meta_cfg, "apply_robot_perturb", False))
    robot_perturb_prob = float(getattr(meta_cfg, "robot_perturb_prob", 1.0))
    train_kwargs = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "num_supervision_edges": int(meta_cfg.num_supervision_edges),
        "contact_supervision_quotas": tuple(getattr(meta_cfg, "contact_supervision_quotas", (16, 16, 16, 16))),
        "contact_supervision_hard_negative_quota": int(
            getattr(meta_cfg, "contact_supervision_hard_negative_quota", 16)
        ),
        "contact_supervision_hard_negative_distance_range": tuple(
            getattr(meta_cfg, "contact_supervision_hard_negative_distance_range", (0.02, 0.03))
        ),
        "contact_radius": float(meta_cfg.contact_radius),
        "base_seed": int(seed),
        "apply_obj_perturb": bool(meta_cfg.apply_obj_perturb),
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", False)),
        "apply_hand_root_perturb": bool(getattr(meta_cfg, "apply_hand_root_perturb", False)),
        "hand_root_rot_std_deg": float(getattr(meta_cfg, "hand_root_rot_std_deg", 10.0)),
        "hand_root_trans_std": float(getattr(meta_cfg, "hand_root_trans_std", 0.01)),
        "hand_root_perturb_prob": float(getattr(meta_cfg, "hand_root_perturb_prob", 1.0)),
        "exclusive_hand_object_perturb": bool(
            getattr(meta_cfg, "exclusive_hand_object_perturb", False)
        ),
        "hand_perturb_prob": float(getattr(meta_cfg, "hand_perturb_prob", 1.0)),
        "obj_rot_std_deg": float(meta_cfg.obj_rot_std_deg),
        "obj_trans_std": float(meta_cfg.obj_trans_std),
        "obj_perturb_prob": float(meta_cfg.obj_perturb_prob),
        "runtime_resample_object": runtime_resample_object,
        "use_mano_reconstruction": use_mano_reconstruction,
        "use_robot_reconstruction": use_robot_reconstruction,
        "apply_robot_perturb": apply_robot_perturb,
        "robot_perturb_prob": robot_perturb_prob,
        "mixed_hand_reconstruction": bool(getattr(meta_cfg, "mixed_hand_reconstruction", False)),
        "robot_perturb_noise_scale_by_domain": _as_plain_mapping(
            getattr(meta_cfg, "robot_perturb_noise_scale_by_domain", {})
        ),
        "exclusive_perturb_mode_probs": tuple(
            getattr(meta_cfg, "exclusive_perturb_mode_probs", (0.4, 0.4, 0.2))
        ),
        "filter_non_interacting_frames": bool(
            getattr(meta_cfg, "filter_non_interacting_frames", False)
        ),
        "interaction_max_distance_m": float(
            getattr(meta_cfg, "interaction_max_distance_m", 0.05)
        ),
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
        "coordinate_frame": expected_coordinate_frame,
        "transform_to_object_frame": bool(getattr(meta_cfg, "transform_to_object_frame", False)),
        "dataset_id": getattr(data_cfg, "dataset_id", None),
        "cache_index_path": getattr(data_cfg, "cache_index_path", None),
        "array_cache_path": getattr(data_cfg, "array_cache_path", None),
        "array_cache_required": bool(getattr(data_cfg, "array_cache_required", False)),
    }
    val_clean_kwargs = {
        **train_kwargs,
        "apply_obj_perturb": False,
        "apply_hand_perturb": False,
        "apply_hand_root_perturb": False,
        "exclusive_hand_object_perturb": bool(getattr(meta_cfg, "exclusive_hand_object_perturb", False)) and not bool(getattr(meta_cfg, "apply_hand_root_perturb", False)),
        "apply_robot_perturb": False,
        "obj_perturb_prob": 0.0,
        "runtime_resample_object": runtime_resample_object,
        "eval_sampling_epoch": 0,
    }
    val_perturbed_kwargs = {
        **train_kwargs,
        "apply_obj_perturb": bool(getattr(meta_cfg, "val_apply_obj_perturb", True)),
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", False)),
        "apply_hand_root_perturb": False,
        "exclusive_hand_object_perturb": bool(getattr(meta_cfg, "exclusive_hand_object_perturb", False)) and not bool(getattr(meta_cfg, "apply_hand_root_perturb", False)),
        "apply_robot_perturb": apply_robot_perturb,
        "obj_perturb_prob": float(getattr(meta_cfg, "val_obj_perturb_prob", 1.0)),
        "runtime_resample_object": runtime_resample_object,
        "eval_sampling_epoch": 0,
    }
    train_loader, val_loader, test_loader, metadata = make_file_split_dataloaders(
        data_cfg,
        seed,
        dataset_cls=CorrStaticDatasetV2,
        file_pattern="**/*.npz",
        train_dataset_kwargs=train_kwargs,
        val_dataset_kwargs=val_clean_kwargs,
        split_group_fn=_sequence_group_key if bool(getattr(data_cfg, "group_val_by_sequence", True)) else None,
        distributed=distributed,
    )
    if bool(data_cfg.shuffle) and bool(getattr(data_cfg, "sequence_locality_shuffle", True)):
        train_sampler = SequenceLocalitySampler(train_loader.dataset, seed)
        train_sampler = shard_sampler_for_distributed(
            train_sampler,
            distributed=distributed,
            drop_last=bool(data_cfg.drop_last),
            pad=True,
        )
        loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(data_cfg.batch_size),
            shuffle=False,
            sampler=train_sampler,
            **make_dataloader_kwargs(data_cfg, loader_seed),
        )

    val_loaders: dict[str, DataLoader] = {}
    if val_loader is not None:
        val_loader.dataset.set_epoch(0)
        val_loaders["val_clean/"] = val_loader
        # Perturbed val loader — only if val_obj_perturb_prob > 0.
        if float(getattr(meta_cfg, "val_obj_perturb_prob", 1.0)) > 0:
            val_perturbed_dataset = CorrStaticDatasetV2(
                val_loader.dataset.data_root,
                file_list=val_loader.dataset.file_paths,
                **val_perturbed_kwargs,
            )
            val_perturbed_dataset.set_epoch(0)
            loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
            val_perturbed_loader = DataLoader(
                val_perturbed_dataset,
                batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                shuffle=False,
                sampler=make_default_eval_sampler(val_perturbed_dataset, distributed=distributed),
                **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
            )
            val_loaders["val_perturbed/"] = val_perturbed_loader
            if bool(getattr(meta_cfg, "val_apply_hand_root_perturb", False)):
                val_root_kwargs = {
                    **val_perturbed_kwargs,
                    "apply_obj_perturb": False,
                    "apply_hand_perturb": False,
                    "apply_hand_root_perturb": True,
                    "obj_perturb_prob": 0.0,
                }
                val_root_dataset = CorrStaticDatasetV2(
                    val_loader.dataset.data_root,
                    file_list=val_loader.dataset.file_paths,
                    **val_root_kwargs,
                )
                val_root_dataset.set_epoch(0)
                val_loaders["val_root_perturbed/"] = DataLoader(
                    val_root_dataset,
                    batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                    shuffle=False,
                    sampler=make_default_eval_sampler(val_root_dataset, distributed=distributed),
                    **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
                )

    first_path = train_loader.dataset.file_paths[0]
    with np.load(first_path, allow_pickle=False) as data:
        metadata.update(
            {
                "num_obj_pool": int(data["obj_points"].shape[-2]),
                "num_obj_points": int(train_kwargs["num_obj_points"]),
                "num_hand_points": int(data["hand_points"].shape[1]),
                "num_supervision_edges": int(meta_cfg.num_supervision_edges),
                "contact_radius": float(meta_cfg.contact_radius),
                "coordinate_frame": str(train_loader.dataset.coordinate_frame),
                "val_loader_names": sorted(val_loaders),
            }
        )
    test_loaders = {"test/": test_loader} if test_loader is not None else {}
    metadata["test_loader_names"] = sorted(test_loaders)
    return train_loader, val_loader, test_loader, metadata, val_loaders, test_loaders


def _make_domain_balanced_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    distributed: Any | None,
    domain_specs: list[Any],
) -> tuple[DataLoader, DataLoader | None, dict[str, Any], dict[str, DataLoader]]:
    """Build equal-domain train batches and independent per-domain validation."""
    expected_coordinate_frame = str(getattr(meta_cfg, "coordinate_frame", "hand_root"))
    runtime_resample_object = bool(getattr(meta_cfg, "runtime_resample_object", True))
    use_mano_reconstruction = bool(getattr(meta_cfg, "use_mano_reconstruction", False))
    use_robot_reconstruction = bool(getattr(meta_cfg, "use_robot_reconstruction", False))
    apply_robot_perturb = bool(getattr(meta_cfg, "apply_robot_perturb", False))
    robot_perturb_prob = float(getattr(meta_cfg, "robot_perturb_prob", 1.0))
    train_kwargs = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "num_supervision_edges": int(meta_cfg.num_supervision_edges),
        "contact_supervision_quotas": tuple(getattr(meta_cfg, "contact_supervision_quotas", (16, 16, 16, 16))),
        "contact_supervision_hard_negative_quota": int(
            getattr(meta_cfg, "contact_supervision_hard_negative_quota", 16)
        ),
        "contact_supervision_hard_negative_distance_range": tuple(
            getattr(meta_cfg, "contact_supervision_hard_negative_distance_range", (0.02, 0.03))
        ),
        "contact_radius": float(meta_cfg.contact_radius),
        "base_seed": int(seed),
        "apply_obj_perturb": bool(meta_cfg.apply_obj_perturb),
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", False)),
        "apply_hand_root_perturb": bool(getattr(meta_cfg, "apply_hand_root_perturb", False)),
        "hand_root_rot_std_deg": float(getattr(meta_cfg, "hand_root_rot_std_deg", 10.0)),
        "hand_root_trans_std": float(getattr(meta_cfg, "hand_root_trans_std", 0.01)),
        "hand_root_perturb_prob": float(getattr(meta_cfg, "hand_root_perturb_prob", 1.0)),
        "exclusive_hand_object_perturb": bool(
            getattr(meta_cfg, "exclusive_hand_object_perturb", False)
        ),
        "hand_perturb_prob": float(getattr(meta_cfg, "hand_perturb_prob", 1.0)),
        "obj_rot_std_deg": float(meta_cfg.obj_rot_std_deg),
        "obj_trans_std": float(meta_cfg.obj_trans_std),
        "obj_perturb_prob": float(meta_cfg.obj_perturb_prob),
        "runtime_resample_object": runtime_resample_object,
        "use_mano_reconstruction": use_mano_reconstruction,
        "use_robot_reconstruction": use_robot_reconstruction,
        "apply_robot_perturb": apply_robot_perturb,
        "robot_perturb_prob": robot_perturb_prob,
        "mixed_hand_reconstruction": bool(getattr(meta_cfg, "mixed_hand_reconstruction", False)),
        "robot_perturb_noise_scale_by_domain": _as_plain_mapping(
            getattr(meta_cfg, "robot_perturb_noise_scale_by_domain", {})
        ),
        "exclusive_perturb_mode_probs": tuple(
            getattr(meta_cfg, "exclusive_perturb_mode_probs", (0.4, 0.4, 0.2))
        ),
        "filter_non_interacting_frames": bool(
            getattr(meta_cfg, "filter_non_interacting_frames", False)
        ),
        "interaction_max_distance_m": float(
            getattr(meta_cfg, "interaction_max_distance_m", 0.05)
        ),
        "coordinate_frame": expected_coordinate_frame,
        "transform_to_object_frame": bool(getattr(meta_cfg, "transform_to_object_frame", False)),
    }
    val_clean_kwargs = {
        **train_kwargs,
        "apply_obj_perturb": False,
        "apply_hand_perturb": False,
        "apply_hand_root_perturb": False,
        "exclusive_hand_object_perturb": bool(getattr(meta_cfg, "exclusive_hand_object_perturb", False)) and not bool(getattr(meta_cfg, "apply_hand_root_perturb", False)),
        "apply_robot_perturb": False,
        "obj_perturb_prob": 0.0,
        "eval_sampling_epoch": 0,
    }
    val_perturbed_kwargs = {
        **train_kwargs,
        "apply_obj_perturb": bool(getattr(meta_cfg, "val_apply_obj_perturb", True)),
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", False)),
        "apply_hand_root_perturb": False,
        "exclusive_hand_object_perturb": bool(getattr(meta_cfg, "exclusive_hand_object_perturb", False)) and not bool(getattr(meta_cfg, "apply_hand_root_perturb", False)),
        "apply_robot_perturb": apply_robot_perturb,
        "obj_perturb_prob": float(getattr(meta_cfg, "val_obj_perturb_prob", 1.0)),
        "eval_sampling_epoch": 0,
    }

    domain_ids: list[str] = []
    train_datasets: list[CorrStaticDatasetV2] = []
    val_loaders: dict[str, DataLoader] = {}
    metadata_domains: list[dict[str, Any]] = []
    for raw_spec in domain_specs:
        if isinstance(raw_spec, dict):
            domain_id = str(raw_spec.get("id", raw_spec.get("dataset_id", ""))).strip()
            domain_path = raw_spec.get("path", raw_spec.get("train_path"))
            domain_val_path = raw_spec.get("val_path")
            domain_cache = raw_spec.get("cache_index_path")
            domain_array_cache = raw_spec.get("array_cache_path")
            domain_array_cache_required = bool(raw_spec.get("array_cache_required", False))
        else:
            raise TypeError("data.domain_paths entries must be mappings with id and path.")
        if not domain_id or domain_path in {None, ""}:
            raise ValueError(f"Each data.domain_paths entry needs non-empty id/path, got {raw_spec!r}.")
        if domain_id in domain_ids:
            raise ValueError(f"Duplicate domain id {domain_id!r}.")

        domain_cfg = copy.copy(data_cfg)
        domain_cfg.train_path = str(resolve_data_path(domain_path, root=getattr(data_cfg, "root", None)))
        domain_cfg.val_path = (
            None
            if domain_val_path in {None, ""}
            else str(resolve_data_path(domain_val_path, root=getattr(data_cfg, "root", None)))
        )
        domain_cfg.test_path = None
        domain_cfg.domain_paths = []
        domain_cfg.cache_index_path = (
            domain_cache
            if domain_cache not in {None, ""}
            else getattr(data_cfg, "cache_index_path", None)
        )
        domain_train_kwargs = {
            **train_kwargs,
            "dataset_id": domain_id,
            "cache_index_path": domain_cfg.cache_index_path,
            "array_cache_path": domain_array_cache,
            "array_cache_required": domain_array_cache_required,
        }
        domain_val_kwargs = {
            **val_clean_kwargs,
            "dataset_id": domain_id,
            "cache_index_path": domain_cfg.cache_index_path,
            "array_cache_path": domain_array_cache,
            "array_cache_required": domain_array_cache_required,
        }
        train_loader, val_loader, _, domain_metadata = make_file_split_dataloaders(
            domain_cfg,
            seed,
            dataset_cls=CorrStaticDatasetV2,
            file_pattern="**/*.npz",
            train_dataset_kwargs=domain_train_kwargs,
            val_dataset_kwargs=domain_val_kwargs,
            split_group_fn=_sequence_group_key
            if bool(getattr(data_cfg, "group_val_by_sequence", True))
            else None,
            distributed=distributed,
        )
        train_datasets.append(train_loader.dataset)
        domain_ids.append(domain_id)
        if val_loader is not None:
            val_loader.dataset.set_epoch(0)
            val_loaders[f"val_clean/{domain_id}/"] = val_loader
            val_perturbed_dataset = CorrStaticDatasetV2(
                val_loader.dataset.data_root,
                file_list=val_loader.dataset.file_paths,
                **{
                    **val_perturbed_kwargs,
                    "dataset_id": domain_id,
                    "array_cache_path": domain_array_cache,
                    "array_cache_required": domain_array_cache_required,
                },
            )
            val_perturbed_dataset.set_epoch(0)
            loader_seed = int(seed) + int(
                getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0
            )
            val_loaders[f"val_perturbed/{domain_id}/"] = DataLoader(
                val_perturbed_dataset,
                batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                shuffle=False,
                sampler=make_default_eval_sampler(val_perturbed_dataset, distributed=distributed),
                **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
            )
            if bool(getattr(meta_cfg, "val_apply_hand_root_perturb", False)):
                val_root_dataset = CorrStaticDatasetV2(
                    val_loader.dataset.data_root,
                    file_list=val_loader.dataset.file_paths,
                    **{
                        **val_perturbed_kwargs,
                        "dataset_id": domain_id,
                        "apply_obj_perturb": False,
                        "apply_hand_perturb": False,
                        "apply_hand_root_perturb": True,
                        "obj_perturb_prob": 0.0,
                        "array_cache_path": domain_array_cache,
                        "array_cache_required": domain_array_cache_required,
                    },
                )
                val_root_dataset.set_epoch(0)
                val_loaders[f"val_root_perturbed/{domain_id}/"] = DataLoader(
                    val_root_dataset,
                    batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                    shuffle=False,
                    sampler=make_default_eval_sampler(val_root_dataset, distributed=distributed),
                    **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
                )
        metadata_domains.append(
            {
                "id": domain_id,
                "path": str(domain_cfg.train_path),
                "num_train_samples": len(train_loader.dataset),
                "num_val_samples": 0 if val_loader is None else len(val_loader.dataset),
                **{key: value for key, value in domain_metadata.items() if key.startswith("num_")},
            }
        )

    combined_dataset = DomainConcatDataset(train_datasets, domain_ids)
    rank = int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
    world_size = int(
        getattr(distributed, "world_size", 1) if getattr(distributed, "enabled", False) else 1
    )
    sampler = DomainBalancedSampler(
        combined_dataset,
        batch_size=int(data_cfg.batch_size),
        seed=int(seed),
        rank=rank,
        world_size=world_size,
        drop_last=bool(data_cfg.drop_last),
        sequence_locality=bool(getattr(data_cfg, "sequence_locality_shuffle", True)),
        domain_sampling_weights=tuple(
            getattr(data_cfg, "domain_sampling_weights", ()) or ()
        ),
    )
    loader_seed = int(seed) + rank
    train_loader = DataLoader(
        combined_dataset,
        batch_size=int(data_cfg.batch_size),
        shuffle=False,
        sampler=sampler,
        **make_dataloader_kwargs(data_cfg, loader_seed),
    )
    metadata = {
        "train_path": "domain_paths",
        "val_path": None,
        "test_path": None,
        "num_train_samples": len(combined_dataset),
        "num_val_samples": sum(item["num_val_samples"] for item in metadata_domains),
        "domain_sampling": list(sampler.domain_sampling_weights),
        "domain_ids": domain_ids,
        "domain_steps_per_epoch": sampler.steps_per_epoch,
        "domains": metadata_domains,
        "runtime_resample_object": runtime_resample_object,
        "coordinate_frame": expected_coordinate_frame,
        "val_loader_names": sorted(val_loaders),
        "test_loader_names": [],
    }
    return train_loader, None, metadata, val_loaders
