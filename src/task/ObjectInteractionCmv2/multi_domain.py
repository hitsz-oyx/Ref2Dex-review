"""V1.4 three-domain reader and source-balanced sampler for Cmv2.

The cache has two different hand contracts.  ``hand_points_world.npy`` is the
old 1538-per-side decoder compatibility stream (3076 bilateral points); the
training input is ``knn_hand_points_world.npy``.  Its size is selected by the
explicit entry variant:

* ``mano``: 2048 points per side, 4096 after bilateral merge;
* ``inspire_f1``: 10135 points per side, 20270 after bilateral merge.

The reader keeps cache arrays in their recorded world/object-pose convention
and converts the current and future frame to the current object frame at sample
time.  A batch collate function pads only the variable hand dimension and
provides ``hand_valid_mask``; padding is never interpreted as a hand point.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


DECODER_POINTS_PER_SIDE = 1538
DECODER_HAND_POINTS = DECODER_POINTS_PER_SIDE * 2
KNN_POINTS_PER_SIDE = {"mano": 2048, "inspire_f1": 10135}
KNN_HAND_POINTS = {name: 2 * count for name, count in KNN_POINTS_PER_SIDE.items()}
DOMAIN_NAMES = ("grab", "arctic", "oakink2")
HAND_VARIANTS = tuple(KNN_POINTS_PER_SIDE)
VARIANT_ALIASES = {
    "mano": frozenset(("mano", "mano_bilateral")),
    "inspire_f1": frozenset(("inspire_f1", "inspire_rl", "inspire", "inspire_geometric")),
}
ARRAYS = (
    "frame_time",
    "source_frame_id",
    "obj_points_pool_world",
    "obj_normals_pool_world",
    "obj_pose_world",
    "knn_hand_points_world",
    "knn_hand_normals_world",
)


def normalize_hand_variant(value: object) -> str:
    raw = str(value or "").strip().lower()
    for canonical, aliases in VARIANT_ALIASES.items():
        if raw in aliases:
            return canonical
    raise ValueError(f"unsupported Cmv2 hand variant: {value!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _world_to_frame(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_world, dtype=np.float32)
    rotation, translation = pose[:3, :3], pose[:3, 3]
    return ((np.asarray(points, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normal_world_to_frame(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose_world, dtype=np.float32)[:3, :3]
    value = np.asarray(normals, dtype=np.float32) @ rotation
    return (value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def _is_se3(pose: np.ndarray) -> bool:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    rotation = pose[:3, :3]
    return bool(
        np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=2e-4)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 2e-4
    )


@lru_cache(maxsize=32)
def _load_arrays(sequence_path: str) -> dict[str, np.ndarray]:
    geometry = Path(sequence_path) / "geometry"
    return {name: np.load(geometry / f"{name}.npy", mmap_mode="r") for name in ARRAYS}


def _manifest_variant(manifest: Mapping[str, Any]) -> str:
    for key in ("hand_variant", "variant", "source"):
        value = manifest.get(key)
        if value:
            return normalize_hand_variant(value)
    raise ValueError("geometry manifest does not declare hand_variant/source")


class InspireSequenceView:
    """Validated lazy view over one geometry cache sequence.

    The class name is retained for compatibility with the first V1.4 draft;
    the view accepts both MANO and Inspire entries.
    """

    def __init__(self, path: str | Path, domain: str, split: str, hand_variant: str | None = None,
                 *, allow_manifest_split_override: bool = False) -> None:
        self.path = Path(path).resolve()
        self.domain = str(domain)
        self.split = str(split)
        if self.domain not in DOMAIN_NAMES:
            raise ValueError(f"unsupported Cmv2 V1.4 domain: {self.domain!r}")
        manifest_path = self.path / "geometry" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        supported_schema = {
            "ref2dex_object_interaction_cm_bilateral_geometry_v1",
            "ref2dex_object_interaction_cm_bilateral_mano_v1_4",
            "ref2dex_object_interaction_cm_oakink2_inspire_v1_4",
        }
        if self.manifest.get("schema_name") not in supported_schema:
            raise ValueError(f"{self.path}: unsupported geometry schema")
        manifest_domain = self.manifest.get("source_dataset", self.manifest.get("dataset"))
        if manifest_domain != self.domain:
            raise ValueError(
                f"{self.path}: index domain {self.domain!r} does not match "
                f"manifest source_dataset={manifest_domain!r}"
            )
        if self.manifest.get("split") != self.split and not allow_manifest_split_override:
            raise ValueError(f"{self.path}: split mismatch")
        self.hand_variant = _manifest_variant(self.manifest)
        if hand_variant is not None and self.hand_variant != normalize_hand_variant(hand_variant):
            raise ValueError(
                f"{self.path}: entry variant {normalize_hand_variant(hand_variant)!r} "
                f"does not match manifest {self.hand_variant!r}"
            )
        expected_knn_points = KNN_HAND_POINTS[self.hand_variant]
        offline_knn = self.manifest.get("offline_knn", {})
        if not isinstance(offline_knn, Mapping):
            offline_knn = {}
        declared_knn = self.manifest.get("knn_hand_points")
        if declared_knn is None:
            declared_knn = offline_knn.get("hand_points")
        expected = {
            "object_pool_points": 4096,
            "decoder_hand_points": DECODER_HAND_POINTS,
            "knn_hand_points": expected_knn_points,
            "knn_k": self.manifest.get("knn_k") or offline_knn.get("k"),
        }
        actual = {
            "object_pool_points": self.manifest.get("object_pool_points"),
            "decoder_hand_points": self.manifest.get("decoder_hand_points", self.manifest.get("hand_points")),
            "knn_hand_points": declared_knn,
            "knn_k": self.manifest.get("knn_k") or offline_knn.get("k"),
        }
        for key, value in expected.items():
            if actual[key] is None or int(actual[key]) != int(value):
                raise ValueError(f"{self.path}: {key} must be {value}, got {actual[key]}")
        points_per_side = self.manifest.get("knn_points_per_side")
        if points_per_side is not None and int(points_per_side) != KNN_POINTS_PER_SIDE[self.hand_variant]:
            raise ValueError(
                f"{self.path}: knn_points_per_side must be "
                f"{KNN_POINTS_PER_SIDE[self.hand_variant]}, got {points_per_side}"
            )
        if self.manifest.get("coordinate_frame") not in ("object_pose_t", "world"):
            raise ValueError(f"{self.path}: expected object_pose_t/world cache")
        if self.manifest.get("hand_side") != "bilateral_merged_left_then_right":
            raise ValueError(f"{self.path}: expected bilateral merged hand stream")
        fps = float(self.manifest.get("effective_fps", 0.0))
        if not np.isfinite(fps) or abs(fps - 30.0) > 1e-3:
            raise ValueError(f"{self.path}: expected 30 Hz geometry, got {fps}")
        self.arrays = _load_arrays(str(self.path))
        frame_count = len(self.arrays["frame_time"])
        self.hand_points = expected_knn_points
        shapes = {
            "frame_time": (frame_count,),
            "source_frame_id": (frame_count,),
            "obj_points_pool_world": (frame_count, 4096, 3),
            "obj_normals_pool_world": (frame_count, 4096, 3),
            "obj_pose_world": (frame_count, 4, 4),
            "knn_hand_points_world": (frame_count, expected_knn_points, 3),
            "knn_hand_normals_world": (frame_count, expected_knn_points, 3),
        }
        for name, shape in shapes.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"{self.path}: {name} expected {shape}, got {self.arrays[name].shape}")
        if frame_count < 2:
            raise ValueError(f"{self.path}: needs at least two frames")
        if not np.isfinite(np.asarray(self.arrays["frame_time"])).all():
            raise ValueError(f"{self.path}: non-finite frame_time")
        source_ids = np.asarray(self.arrays["source_frame_id"])
        if np.any(np.diff(source_ids) <= 0):
            raise ValueError(f"{self.path}: source_frame_id must increase strictly")
        self.frame_count = frame_count
        self.effective_fps = fps
        candidate_path = self.path / "geometry" / "obj_candidate_mask_2cm.npy"
        self.candidate_mask = None
        if candidate_path.is_file():
            self.candidate_mask = np.load(candidate_path, mmap_mode="r")
            if self.candidate_mask.shape != (frame_count, 4096):
                raise ValueError(f"{candidate_path}: invalid candidate mask shape")
            if self.candidate_mask.dtype != np.bool_:
                raise ValueError(f"{candidate_path}: candidate mask must be bool")

    def has_contact(self, frame: int) -> bool:
        if self.candidate_mask is None:
            return True
        return bool(np.asarray(self.candidate_mask[frame], dtype=bool).any())


def _resolve_source_entries(source_specs: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for spec in source_specs:
        domain = str(spec.get("name", ""))
        if domain not in DOMAIN_NAMES:
            raise ValueError(f"V1.4 source name must be one of {DOMAIN_NAMES}, got {domain!r}")
        requested_variant = spec.get("hand_variant", spec.get("variant"))
        if requested_variant is None:
            raise ValueError(f"{domain}: source must explicitly declare hand_variant")
        requested_variant = normalize_hand_variant(requested_variant)
        index_path = Path(str(spec["index"])).resolve()
        manifest_path = Path(str(spec["manifest"])).resolve()
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = cache_manifest.get("validation", {})
        if isinstance(validation, Mapping) and int(validation.get("bad_count", 0) or 0) != 0:
            raise ValueError(f"{manifest_path}: cache validation reports bad sequences")
        declared = cache_manifest.get("knn_hand_points_per_stream", {})
        if isinstance(declared, Mapping):
            declared_count = declared.get(requested_variant)
            if declared_count is not None and int(declared_count) != KNN_HAND_POINTS[requested_variant]:
                raise ValueError(
                    f"{manifest_path}: {requested_variant} stream must have "
                    f"{KNN_HAND_POINTS[requested_variant]} points, got {declared_count}"
                )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index.get("sequences", {}).get(split), list):
            raise ValueError(f"{index_path}: missing {split} sequence list")
        matched = 0
        for item in index["sequences"][split]:
            if not isinstance(item, Mapping) or "path" not in item:
                raise ValueError(f"{index_path}: malformed sequence entry")
            item_domain = str(item.get("dataset", item.get("domain", "")))
            if item_domain != domain:
                continue
            item_variant_value = item.get("hand_variant", item.get("variant", item.get("source")))
            try:
                item_variant = normalize_hand_variant(item_variant_value)
            except ValueError as error:
                raise ValueError(f"{index_path}: {item.get('id')} has no supported hand variant") from error
            if item_variant != requested_variant:
                continue
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = (index_path.parent / path).resolve()
            entries.append({
                "domain": domain,
                "hand_variant": requested_variant,
                "path": str(path),
                "id": str(item.get("id", path.name)),
                "split": split,
                "frame_count": int(item.get("frame_count", 0) or 0),
            })
            matched += 1
        if matched == 0:
            raise ValueError(f"{index_path}: no {domain}/{requested_variant} entries in {split}")
    if not entries:
        raise ValueError(f"no {split} entries in V1.4 sources")
    return entries


class ThreeDomainTransitions(Dataset):
    """Transition samples from three domains with variant-aware hand streams."""

    def __init__(
        self,
        source_specs: Sequence[Mapping[str, Any]],
        split: str,
        *,
        num_obj_points: int = 1024,
        train_stride_values: Mapping[str, Sequence[int]] | None = None,
        fixed_stride: int | None = None,
        active_only: bool = True,
        base_seed: int = 42,
        max_sequences_per_domain: int | None = None,
        allow_manifest_split_override: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.split = split
        self.num_obj_points = int(num_obj_points)
        if not 1 <= self.num_obj_points <= 4096:
            raise ValueError("num_obj_points must lie in [1,4096]")
        self.base_seed = int(base_seed)
        self.active_only = bool(active_only)
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        self.stride_values = {
            str(domain): tuple(int(value) for value in values)
            for domain, values in (train_stride_values or {}).items()
        }
        for domain in DOMAIN_NAMES:
            values = self._stride_values(domain)
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{domain}: stride values must be positive")
        raw_entries = _resolve_source_entries(source_specs, split)
        if max_sequences_per_domain is not None:
            limit = int(max_sequences_per_domain)
            if limit <= 0:
                raise ValueError("max_sequences_per_domain must be positive")
            counts: dict[str, int] = {}
            selected = []
            for entry in raw_entries:
                domain = entry["domain"]
                if counts.get(domain, 0) < limit:
                    selected.append(entry)
                    counts[domain] = counts.get(domain, 0) + 1
            raw_entries = selected
        self.sequences: list[InspireSequenceView] = []
        self.sequence_entries: list[dict[str, Any]] = []
        for entry in raw_entries:
            path = Path(entry["path"])
            if not path.is_dir():
                raise FileNotFoundError(path)
            sequence = InspireSequenceView(
                path, entry["domain"], split, entry["hand_variant"],
                allow_manifest_split_override=allow_manifest_split_override)
            if entry.get("frame_count") and entry["frame_count"] != sequence.frame_count:
                raise ValueError(f"{path}: index frame_count mismatch")
            self.sequences.append(sequence)
            self.sequence_entries.append(entry)
        self.rows: list[tuple[int, int]] = []
        self.dropped_transitions = 0
        for sequence_index, sequence in enumerate(self.sequences):
            max_stride = max(self._stride_values(sequence.domain))
            for frame in range(max(0, sequence.frame_count - max_stride)):
                if self.active_only and not sequence.has_contact(frame):
                    self.dropped_transitions += 1
                    continue
                self.rows.append((sequence_index, frame))
        if not self.rows:
            raise ValueError(f"no valid {split} transitions in V1.4 sources")

    def _stride_values(self, domain: str) -> tuple[int, ...]:
        if self.fixed_stride is not None:
            if self.fixed_stride <= 0:
                raise ValueError("fixed_stride must be positive")
            return (self.fixed_stride,)
        return self.stride_values.get(str(domain), tuple(range(1, 11)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.rows):
            raise IndexError(index)
        sequence_index, current = self.rows[index]
        sequence = self.sequences[sequence_index]
        raw_current = int(np.asarray(sequence.arrays["source_frame_id"])[current])
        seed = _stable_seed(self.base_seed, sequence.path, raw_current, self.split)
        values = self._stride_values(sequence.domain)
        stride = int(values[seed % len(values)])
        future = current + stride
        if future >= sequence.frame_count:
            raise IndexError(f"stride {stride} exceeds {sequence.path} at frame {current}")
        frame_time = np.asarray(sequence.arrays["frame_time"], dtype=np.float64)
        delta_time = float(frame_time[future] - frame_time[current])
        source_ids = np.asarray(sequence.arrays["source_frame_id"])
        if source_ids[future] <= source_ids[current] or not np.isfinite(delta_time) or delta_time <= 0:
            raise ValueError(f"invalid transition timeline: {sequence.path} frame {current}->{future}")
        pose = np.asarray(sequence.arrays["obj_pose_world"][current], dtype=np.float32)
        next_pose = np.asarray(sequence.arrays["obj_pose_world"][future], dtype=np.float32)
        if not _is_se3(pose) or not _is_se3(next_pose):
            raise ValueError(f"invalid rigid pose: {sequence.path} frame {current}")
        rotation, translation = pose[:3, :3], pose[:3, 3]
        next_rotation, next_translation = next_pose[:3, :3], next_pose[:3, 3]
        delta_rotation = rotation.T @ next_rotation
        delta_translation = (next_translation - translation) @ rotation
        point_rng = np.random.default_rng(seed ^ 0xA17)
        selected = point_rng.choice(4096, size=self.num_obj_points, replace=False)
        object_world = np.asarray(sequence.arrays["obj_points_pool_world"][current, selected], dtype=np.float32)
        object_future_world = np.asarray(sequence.arrays["obj_points_pool_world"][future, selected], dtype=np.float32)
        object_points = _world_to_frame(object_world, pose)
        object_future = _world_to_frame(object_future_world, pose)
        object_normals = _normal_world_to_frame(
            np.asarray(sequence.arrays["obj_normals_pool_world"][current, selected], dtype=np.float32), pose)
        hand_world = np.asarray(sequence.arrays["knn_hand_points_world"][current], dtype=np.float32)
        hand_future_world = np.asarray(sequence.arrays["knn_hand_points_world"][future], dtype=np.float32)
        hand_normals_world = np.asarray(sequence.arrays["knn_hand_normals_world"][current], dtype=np.float32)
        hand_points = _world_to_frame(hand_world, pose)
        hand_future = _world_to_frame(hand_future_world, pose)
        hand_normals = _normal_world_to_frame(hand_normals_world, pose)
        obj_flow = object_future - object_points
        tensors = {
            "obj_points": object_points,
            "obj_normals": object_normals,
            "hand_points": hand_points,
            "hand_normals": hand_normals,
            "hand_flow": hand_future - hand_points,
            "obj_flow_gt": obj_flow,
            "delta_translation_gt": delta_translation,
            "delta_rotation_gt": delta_rotation,
        }
        if not all(np.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"non-finite transition: {sequence.path} frame {current}")
        reconstruction = object_points @ delta_rotation.T + delta_translation
        pose_flow_residual = float(np.max(np.linalg.norm(reconstruction - object_future, axis=-1)))
        return {
            **{key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)) for key, value in tensors.items()},
            "delta_time_s": torch.tensor(delta_time, dtype=torch.float32),
            "hand_valid_mask": torch.ones((sequence.hand_points,), dtype=torch.bool),
            "stride": torch.tensor(stride, dtype=torch.int64),
            "source_frame_id": torch.tensor(raw_current, dtype=torch.int64),
            "next_source_frame_id": torch.tensor(int(source_ids[future]), dtype=torch.int64),
            "pose_flow_residual_max_m": torch.tensor(pose_flow_residual, dtype=torch.float32),
            "sequence_id": sequence_entries_id(self.sequence_entries[sequence_index]),
            "source": sequence.domain,
            "hand_variant": sequence.hand_variant,
        }

    def rows_by_source(self) -> dict[str, int]:
        return {
            domain: sum(1 for sequence_index, _ in self.rows if self.sequences[sequence_index].domain == domain)
            for domain in DOMAIN_NAMES
        }

    def rows_by_variant(self) -> dict[str, int]:
        return {
            variant: sum(1 for sequence_index, _ in self.rows if self.sequences[sequence_index].hand_variant == variant)
            for variant in HAND_VARIANTS
        }


def sequence_entries_id(entry: Mapping[str, Any]) -> str:
    return str(entry.get("id") or entry.get("path"))


def collate_three_domain(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack samples and pad variable MANO/Inspire hand streams to batch max H."""
    if not batch:
        raise ValueError("cannot collate an empty V1.4 batch")
    result: dict[str, Any] = {}
    pad_keys = {"hand_points", "hand_normals", "hand_flow", "hand_valid_mask"}
    max_hand = max(int(sample["hand_points"].shape[0]) for sample in batch)
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in pad_keys:
            first = values[0]
            shape = (len(batch), max_hand, *first.shape[1:])
            padded = torch.zeros(shape, dtype=first.dtype)
            for ordinal, value in enumerate(values):
                padded[ordinal, : value.shape[0]] = value
            result[key] = padded
        elif torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


class SourceBalancedSampler(Sampler[int]):
    """Replacement sampler with an exact configured probability per source."""

    def __init__(
        self,
        sources: Sequence[str],
        probabilities: Mapping[str, float],
        *,
        seed: int,
        strict: bool = True,
    ) -> None:
        self.sources = tuple(str(value) for value in sources)
        counts: dict[str, int] = {}
        for source in self.sources:
            counts[source] = counts.get(source, 0) + 1
        requested = {str(key): float(value) for key, value in probabilities.items()}
        if any(not np.isfinite(value) or value <= 0 for value in requested.values()):
            raise ValueError("source probabilities must be finite and positive")
        if strict:
            missing = sorted(set(counts) - set(requested))
            extra = sorted(set(requested) - set(counts))
            if missing or extra:
                raise ValueError(f"strict source probabilities mismatch: missing={missing}, extra={extra}")
        active = {source: value for source, value in requested.items() if source in counts}
        if not active:
            raise ValueError("no source has both rows and a configured probability")
        total = sum(active.values())
        self._weights = torch.tensor(
            [active[source] / total / counts[source] for source in self.sources], dtype=torch.double)
        self.num_samples = len(self.sources)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.multinomial(self._weights, self.num_samples, replacement=True,
                                      generator=generator).tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_source_sampler(dataset: ThreeDomainTransitions, probabilities: Mapping[str, float], seed: int) -> SourceBalancedSampler:
    sources = [dataset.sequences[sequence_index].domain for sequence_index, _ in dataset.rows]
    return SourceBalancedSampler(sources, probabilities, seed=seed, strict=True)
