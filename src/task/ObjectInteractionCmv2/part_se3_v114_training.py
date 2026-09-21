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
)
from .part_se3_v114 import PART_SE3_V114_VERSION, PartSE3ObjectInteractionCmv2V114Model


WORK_VERSION = "V1.14"
CONFIG_SCHEMA = "object_interaction_cmv2_shared_start_candidate_v1_14a"
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


def load_v114_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if (config.get("schema_name") != CONFIG_SCHEMA
            or config.get("work_version") != WORK_VERSION
            or config.get("initialization") != "random"
            or config.get("run_authorization") != "not_approved"):
        raise ValueError("V1.14 implementation config identity mismatch")
    if config.get("model") != EXPECTED_MODEL:
        raise ValueError("V1.14 model contract mismatch")
    if config.get("data") != EXPECTED_DATA or config.get("loss") != EXPECTED_LOSS:
        raise ValueError("V1.14 data or loss contract mismatch")
    if config.get("hand_sampling") != EXPECTED_HAND_SAMPLING:
        raise ValueError("V1.14a hand sampling contract mismatch")
    if set(config.get("sources") or {}) != set(GROUPS):
        raise ValueError("V1.14 requires the frozen five source groups")
    if set(config.get("oakink2_parts") or {}) != {
            "adapter_root", "selection_index", "annotation_root", "stage3_root"}:
        raise ValueError("V1.14 requires the validated OakInk2 part adapter inputs")
    for key in ("articulation_metadata", "split_root"):
        config[key] = _absolute(config[key])
    for key, value in config["oakink2_parts"].items():
        config["oakink2_parts"][key] = _absolute(value)
    for source in config["sources"].values():
        for key in ("index", "manifest"):
            source[key] = _absolute(source[key])
    backend = config.get("data_backend", "reference")
    if backend == "compact":
        if not config.get("compact_cache_root"):
            raise ValueError("V1.14 compact backend requires an explicit cache root")
        config["compact_cache_root"] = _absolute(config["compact_cache_root"])
    elif backend != "reference":
        raise ValueError(f"unsupported V1.14 data backend: {backend}")
    return config


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
    "EXPECTED_MODEL",
    "EXPECTED_HAND_SAMPLING",
    "WORK_VERSION",
    "build_part_se3_dataset",
    "checkpoint_payload_v114",
    "initialize_random_v114_model",
    "load_v114_config",
    "restore_checkpoint_v114",
]
