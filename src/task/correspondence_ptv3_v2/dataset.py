from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.base import make_file_split_dataloaders
from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, shard_sampler_for_distributed
from src.task.correspondence_ptv3_v2.sampling import (
    perturb_object_geometry,
    sample_object_indices,
    sample_random_supervision_edges,
    stable_frame_seed,
)
from src.task.correspondence_ptv3_v2.hand_noise_profiles import (
    infer_stage3_dataset_id,
)


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
        blacklist_path: str | None = None,
        eval_sampling_epoch: int | None = None,
        coordinate_frame: str | None = None,
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
        self.obj_rot_std_deg = float(obj_rot_std_deg)
        self.obj_trans_std = float(obj_trans_std)
        self.obj_perturb_prob = float(obj_perturb_prob)
        self.runtime_resample_object = bool(runtime_resample_object)
        self.use_mano_reconstruction = bool(use_mano_reconstruction)
        self.eval_sampling_epoch = None if eval_sampling_epoch is None else int(eval_sampling_epoch)
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
        # Fix #8 (docs/指导.md): when ``use_mano_reconstruction`` is set, the
        # schema MUST be v2.1+ and every required MANO field must be present.
        # We refuse to silently fall back to the legacy hand_points because
        # that path produces empty gradients on the hand side and corrupts
        # every metric that depends on the MANO reconstruction.
        per_file_frames: set[str] = set()
        per_file_schema_versions: set[str] = set()
        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
                if use_mano_reconstruction:
                    self._validate_mano_schema(path, data)
                    per_file_schema_versions.add(
                        self._scalar_string(data, "schema_version", "0.0.0")
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

        if use_mano_reconstruction and per_file_schema_versions:
            if any(
                self._schema_version_tuple(v) < (2, 1, 0)
                for v in per_file_schema_versions
            ):
                raise ValueError(
                    f"Stage 3 .npz at {self.data_root} has schema_version "
                    f"{sorted(per_file_schema_versions)} but use_mano_reconstruction=True "
                    "requires schema_version >= 2.1.0. Re-run Stage 3 with the current "
                    "process/common/stage3_corr.py on the same Stage 2 root."
                )

        if len(per_file_frames) != 1:
            raise ValueError(
                f"Inconsistent Stage 3 coordinate_frame across {self.data_root}: {sorted(per_file_frames)}"
            )
        self.coordinate_frame = next(iter(per_file_frames))
        if coordinate_frame is not None and self.coordinate_frame != str(coordinate_frame):
            raise ValueError(
                f"Stage 3 .npz at {self.data_root} was generated in "
                f"coordinate_frame={self.coordinate_frame!r} but training expects "
                f"{coordinate_frame!r}. Re-run Stage 3 with "
                f"--coordinate-frame {coordinate_frame} on the same Stage 2 root."
            )

        self._samples: list[tuple[Path, int]] = []
        self.file_sample_ranges: list[tuple[int, int]] = []
        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                num_frames = int(data["raw_frame_id"].shape[0])
            start = len(self._samples)
            self._samples.extend((path, frame_idx) for frame_idx in range(num_frames))
            self.file_sample_ranges.append((start, len(self._samples)))

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
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        self._cached_path = path
        self._cached_data = payload
        return payload

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
        "schema_name",
        "schema_version",
        "mano_global_orient",
        "mano_transl",
        "mano_pose",
        "mano_betas",
        "mano_v_template",
        "mano_use_pca",
        "mano_num_pca_comps",
        "mano_flat_hand_mean",
        "mano_pose_repr",
        "hand_root_pose",
        "hand_to_obj_min_dist",
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

    @staticmethod
    def _schema_version_tuple(value: str) -> tuple[int, ...]:
        parts: list[int] = []
        for chunk in str(value).split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                parts.append(0)
        return tuple(parts) or (0, 0, 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, frame_idx = self._samples[index]
        data = self._load_file(path)
        seq_id = self._scalar_string(data, "seq_id", path.stem)
        side = self._scalar_string(data, "side", "")
        dataset_id = infer_stage3_dataset_id(data)
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
        if has_legacy_candidate_mask:
            candidate_mask = np.asarray(candidate_mask_data[frame_idx], dtype=bool)
        else:
            candidate_mask = np.ones((int(data["obj_points"].shape[1]),), dtype=bool)
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

        hand_points = np.asarray(data["hand_points"][frame_idx], dtype=np.float32)
        hand_normals = np.asarray(data["hand_normals"][frame_idx], dtype=np.float32)
        hand_min_dist = _resolve_hand_to_obj_min_dist(data, frame_idx=frame_idx)
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
            hand_points=hand_points,
            hand_normals=hand_normals,
            seed=aug_seed,
            apply_obj_perturb=self.apply_obj_perturb,
            obj_rot_std_deg=self.obj_rot_std_deg,
            obj_trans_std=self.obj_trans_std,
            obj_perturb_prob=self.obj_perturb_prob,
        )
        # Runtime resampling mode: also perturb the full 4096 pool with
        # the same SE(3) so the runner can pick 384 near + 128 global
        # at prepare_batch time using the (possibly MANO/PCA perturbed)
        # hand as the query. ``full_geometry`` carries the clean
        # (``gt_*``) and the perturbed (``input_*``) versions of the
        # entire object pool; the hand arrays here are unused because
        # hand perturbation is performed by MANO forward in the runner.
        full_geometry = None
        # Keep the batch schema identical across datasets. Some regenerated
        # v2.1 sources (notably ContactPose) still carry the legacy candidate
        # mask while GRAB/ARCTIC may omit it. Runtime resampling must therefore
        # depend only on the training config, not on that optional file field.
        runtime_resample_this = self.runtime_resample_object
        if runtime_resample_this:
            full_geometry = perturb_object_geometry(
                obj_points=full_obj_points,
                obj_normals=full_obj_normals,
                hand_points=hand_points,
                hand_normals=hand_normals,
                seed=aug_seed,
                apply_obj_perturb=self.apply_obj_perturb,
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
            [obj_valid, np.ones((self.num_hand_points,), dtype=bool)],
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
            "contact_seed": torch.tensor(contact_seed, dtype=torch.long),
            "num_obj_points": torch.tensor(self.num_obj_points, dtype=torch.long),
            "num_hand_points": torch.tensor(self.num_hand_points, dtype=torch.long),
        }
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
        if "mano_pose" in data and "mano_global_orient" in data:
            # Fix #1 (docs/指导.md): explicitly tell the runner this sample
            # has MANO parameters, so it actually runs the reconstruction
            # path instead of falling back to the legacy hand_points.
            result["has_mano"] = torch.tensor(True, dtype=torch.bool)
            # Fix #2: per-sample apply_hand_perturb flag so val_clean
            # does not get the same noise injection as train.
            result["apply_hand_perturb"] = torch.tensor(
                bool(self.apply_hand_perturb), dtype=torch.bool
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
            result["__dataset_id__"] = dataset_id
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
        return result


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
    expected_coordinate_frame = str(getattr(meta_cfg, "coordinate_frame", "hand_root"))
    runtime_resample_object = bool(getattr(meta_cfg, "runtime_resample_object", True))
    use_mano_reconstruction = bool(getattr(meta_cfg, "use_mano_reconstruction", False))
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
        "obj_rot_std_deg": float(meta_cfg.obj_rot_std_deg),
        "obj_trans_std": float(meta_cfg.obj_trans_std),
        "obj_perturb_prob": float(meta_cfg.obj_perturb_prob),
        "runtime_resample_object": runtime_resample_object,
        "use_mano_reconstruction": use_mano_reconstruction,
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
        "coordinate_frame": expected_coordinate_frame,
    }
    val_clean_kwargs = {
        **train_kwargs,
        "apply_obj_perturb": False,
        "apply_hand_perturb": False,
        "obj_perturb_prob": 0.0,
        "runtime_resample_object": runtime_resample_object,
        "eval_sampling_epoch": 0,
    }
    val_perturbed_kwargs = {
        **train_kwargs,
        "apply_obj_perturb": True,
        "apply_hand_perturb": bool(getattr(meta_cfg, "apply_hand_perturb", False)),
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

    first_path = train_loader.dataset.file_paths[0]
    with np.load(first_path, allow_pickle=False) as data:
        metadata.update(
            {
                "num_obj_pool": int(data["obj_points"].shape[1]),
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
