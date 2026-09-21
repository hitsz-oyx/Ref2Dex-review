"""V1.14 construction and checkpoint contracts; this module does not authorize a run."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml

from .mixed_training import GROUPS
from .part_se3_training import (
    EXPECTED_DATA,
    EXPECTED_LOSS,
    REPO_ROOT,
    build_part_se3_dataset as _build_part_se3_dataset,
    SUPPORTED_PART_SE3_GROUPS,
)
from .part_se3_v114 import PART_SE3_V114_VERSION, PartSE3ObjectInteractionCmv2V114Model


WORK_VERSION = "V1.14"
CONFIG_SCHEMA = "object_interaction_cmv2_shared_start_candidate_v1_14a"
PORTABLE_CONFIG_SCHEMA = "object_interaction_cmv2_portable_bilateral_v1_14b"
CHECKPOINT_SCHEMA = "object_interaction_cmv2_shared_start_candidate_checkpoint_v2"
EXPECTED_HAND_SAMPLING = {
    "contract": "bilateral_fixed_random_2048_per_side_v1",
    "points_per_side": 2048,
    "bilateral_points": 4096,
    "seed": 42,
}
EXPECTED_MODEL = {
    "architecture_version": PART_SE3_V114_VERSION,
    "hidden_width": 128,
    "interaction_dim": 32,
    "num_tokens": 16,
    "knn_k": 32,
    "interaction_radius_m": 0.02,
    "feature_scale_m": 0.02,
    "frame_dt_s": 1 / 30,
}


def _absolute(value: str | Path) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else REPO_ROOT / path).resolve())


def load_v114_config(path: str | Path, *, allow_approved_run: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    schema = config.get("schema_name")
    if (schema not in (CONFIG_SCHEMA, PORTABLE_CONFIG_SCHEMA)
            or config.get("work_version") != WORK_VERSION
            or config.get("initialization") != "random"
            or config.get("run_authorization") not in (
                ("not_approved", "approved") if allow_approved_run else ("not_approved",))):
        raise ValueError("V1.14 implementation config identity mismatch")
    if config.get("model") != EXPECTED_MODEL:
        raise ValueError("V1.14 model contract mismatch")
    if config.get("data") != EXPECTED_DATA or config.get("loss") != EXPECTED_LOSS:
        raise ValueError("V1.14 data or loss contract mismatch")
    if config.get("hand_sampling") != EXPECTED_HAND_SAMPLING:
        raise ValueError("V1.14a hand sampling contract mismatch")
    sources = config.get("sources") or {}
    if schema == CONFIG_SCHEMA:
        active_groups = list(GROUPS)
        if set(sources) != set(active_groups):
            raise ValueError("V1.14a requires the frozen five source groups")
    else:
        active_groups = list(config.get("active_groups") or [])
        if (not active_groups or len(active_groups) != len(set(active_groups))
                or any(group not in SUPPORTED_PART_SE3_GROUPS for group in active_groups)):
            raise ValueError("V1.14b active_groups must be a unique non-empty supported subset")
        if set(sources) != set(active_groups):
            raise ValueError("V1.14b sources must match active_groups exactly")
        weights = config.get("group_weights") or {}
        if set(weights) != set(active_groups) or any(float(value) <= 0 for value in weights.values()):
            raise ValueError("V1.14b group_weights must be positive and match active_groups")
        total = sum(float(value) for value in weights.values())
        config["group_weights"] = {group: float(weights[group]) / total for group in active_groups}
    config["active_groups"] = active_groups
    oakink_required = any(group.startswith("oakink2/") for group in active_groups)
    expected_oakink = {"adapter_root", "selection_index", "annotation_root", "stage3_root"}
    if oakink_required and set(config.get("oakink2_parts") or {}) != expected_oakink:
        raise ValueError("active OakInk2 groups require the validated part adapter inputs")
    for key in ("articulation_metadata", "split_root"):
        config[key] = _absolute(config[key])
    for key, value in (config.get("oakink2_parts") or {}).items():
        config["oakink2_parts"][key] = _absolute(value)
    for source in config["sources"].values():
        for key in ("index", "manifest"):
            source[key] = _absolute(source[key])
    if allow_approved_run:
        training = config.get("training") or {}
        required_training = {"batch_size_per_rank", "world_size", "epochs", "learning_rate", "seed",
                             "checkpoint_interval", "log_interval", "cpu_threads_per_rank"}
        if config.get("run_authorization") != "approved" or set(training) != required_training:
            raise ValueError("V1.14b formal run requires an approved complete training budget")
        resources = config.get("resources") or {}
        physical = [int(value) for value in resources.get("physical_gpus", [])]
        if (not physical or len(physical) != int(training["world_size"])
                or len(physical) != len(set(physical))):
            raise ValueError("V1.14b physical_gpus must uniquely match world_size")
        config["resources"] = {"physical_gpus": physical,
                               "nofile_limit": int(resources.get("nofile_limit", 262144))}
        if config["resources"]["nofile_limit"] < 65536:
            raise ValueError("V1.14b nofile_limit is too small for memmap datasets")
        config["output_root"] = _absolute(config["output_root"])
    backend = config.get("data_backend", "reference")
    if backend == "compact":
        if not config.get("compact_cache_root"):
            raise ValueError("V1.14 compact backend requires an explicit cache root")
        config["compact_cache_root"] = _absolute(config["compact_cache_root"])
    elif backend != "reference":
        raise ValueError(f"unsupported V1.14 data backend: {backend}")
    return config


def normalized_batch_counts(
    active_groups: list[str] | tuple[str, ...],
    weights: Mapping[str, float],
    batch_size: int,
    step: int,
) -> dict[str, int]:
    """Allocate an exact batch with deterministic largest-remainder rounding."""
    groups = tuple(active_groups)
    if batch_size < len(groups) or step < 0 or set(weights) != set(groups):
        raise ValueError("invalid V1.14b batch allocation inputs")
    normalized = {group: float(weights[group]) / sum(float(weights[x]) for x in groups) for group in groups}
    remaining = batch_size - len(groups)
    raw = {group: normalized[group] * remaining for group in groups}
    counts = {group: 1 + int(raw[group]) for group in groups}
    extras = batch_size - sum(counts.values())
    order = sorted(groups, key=lambda group: (-(raw[group] - int(raw[group])),
                                               (groups.index(group) - step) % len(groups)))
    for group in order[:extras]:
        counts[group] += 1
    return counts


def build_part_se3_dataset(config: Mapping[str, Any], group: str, split: str, **kwargs):
    """Build only the V1.14a fixed-4096 reference path."""
    if config.get("hand_sampling") != EXPECTED_HAND_SAMPLING:
        raise ValueError("V1.14a dataset requires the fixed 2048-per-side contract")
    if config.get("data_backend", "reference") != "reference":
        raise ValueError("V1.14a compact-v2 cache has not been materialized")
    return _build_part_se3_dataset(
        config, group, split,
        hand_points_per_side=EXPECTED_HAND_SAMPLING["points_per_side"],
        hand_sampling_seed=EXPECTED_HAND_SAMPLING["seed"],
        **kwargs,
    )


def initialize_random_v114_model(
    config: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> PartSE3ObjectInteractionCmv2V114Model:
    if config.get("initialization") != "random" or config.get("model") != EXPECTED_MODEL:
        raise ValueError("V1.14 only permits the frozen random initialization contract")
    return PartSE3ObjectInteractionCmv2V114Model(
        SimpleNamespace(**dict(config["model"]))).to(device)


def checkpoint_payload_v114(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    best_metric: float,
) -> dict[str, Any]:
    return {
        "schema_name": CHECKPOINT_SCHEMA,
        "work_version": WORK_VERSION,
        "architecture_version": PART_SE3_V114_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": float(best_metric),
    }


def restore_checkpoint_v114(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    identity = (
        payload.get("schema_name"), payload.get("work_version"),
        payload.get("architecture_version"),
    )
    if identity != (CHECKPOINT_SCHEMA, WORK_VERSION, PART_SE3_V114_VERSION):
        raise ValueError("checkpoint is not a V1.14a shared-start checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {key: payload[key] for key in ("epoch", "step", "best_metric")}


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "PORTABLE_CONFIG_SCHEMA",
    "EXPECTED_MODEL",
    "EXPECTED_HAND_SAMPLING",
    "WORK_VERSION",
    "build_part_se3_dataset",
    "checkpoint_payload_v114",
    "initialize_random_v114_model",
    "load_v114_config",
    "normalized_batch_counts",
    "restore_checkpoint_v114",
]
