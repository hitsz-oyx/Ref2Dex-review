"""Frozen V1.12 construction contracts; this module does not authorize a run."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import yaml

from .mixed_training import GROUPS
from .multi_domain import ThreeDomainTransitions
from .part_se3 import PART_SE3_VERSION, PartSE3ObjectInteractionCmv2V112Model
from .part_se3_data import OakInkWholePartTransitions, RigidArticulatedPartTransitions


WORK_VERSION = "V1.12"
COMPACT_WORK_VERSION = "V1.13"
CONFIG_SCHEMA = "object_interaction_cmv2_part_se3_v1_12"
FORMAL_CONFIG_SCHEMA = "object_interaction_cmv2_part_se3_ddp_v1_12"
COMPACT_FORMAL_CONFIG_SCHEMA = "object_interaction_cmv2_part_se3_ddp_v1_13"
CHECKPOINT_SCHEMA = "object_interaction_cmv2_part_se3_checkpoint_v1"
REPO_ROOT = Path(__file__).resolve().parents[3]

EXPECTED_MODEL = {
    "architecture_version": PART_SE3_VERSION,
    "hidden_width": 128,
    "num_tokens": 16,
    "knn_k": 32,
    "interaction_radius_m": 0.02,
    "feature_scale_m": 0.02,
    "frame_dt_s": 1 / 30,
}
EXPECTED_DATA = {
    "num_obj_points": 1024,
    "minimum_points_per_part": 64,
    "train_strides": [1, 2, 3],
    "validation_strides": [1, 2, 3],
    "oakink2_val_fraction": 0.1,
    "endpoint_candidate_k_per_frame": 32,
    "endpoint_final_k": 32,
}
EXPECTED_LOSS = {
    "translation": "smooth_l1_2cm_sample_part_mean",
    "rotation": "so3_geodesic_sample_part_mean",
    "flow": "smooth_l1_2cm_point_mean",
    "weights": {"translation": 1.0, "rotation": 1.0, "flow": 1.0},
}
EXPECTED_TRAINING = {
    "batch_size_per_rank": 64,
    "world_size": 2,
    "epochs": 16,
    "learning_rate": 0.001,
    "seed": 42,
    "checkpoint_interval": 200,
    "log_interval": 20,
    "minimum_free_memory_gib": 20,
    "cpu_threads_per_rank": 8,
}


def _absolute(value: str | Path) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else REPO_ROOT / path).resolve())


def _validate_common(config: dict[str, Any], *, work_version: str = WORK_VERSION) -> dict[str, Any]:
    if config.get("work_version") != work_version or config.get("initialization") != "random":
        raise ValueError(f"{work_version} work_version or initialization mismatch")
    if config.get("model") != EXPECTED_MODEL or config.get("data") != EXPECTED_DATA:
        raise ValueError("V1.12 model or data contract mismatch")
    if config.get("loss") != EXPECTED_LOSS:
        raise ValueError("V1.12 loss contract mismatch")
    if set(config.get("sources") or {}) != set(GROUPS):
        raise ValueError("V1.12 requires the frozen five source groups")
    if set(config.get("oakink2_parts") or {}) != {
            "adapter_root", "selection_index", "annotation_root", "stage3_root"}:
        raise ValueError("V1.12 requires the validated OakInk2 part adapter inputs")
    for key in ("articulation_metadata", "split_root"):
        config[key] = _absolute(config[key])
    for key, value in config["oakink2_parts"].items():
        config["oakink2_parts"][key] = _absolute(value)
    for source in config["sources"].values():
        for key in ("index", "manifest"):
            source[key] = _absolute(source[key])
    return config


def load_part_se3_config(path: str | Path) -> dict[str, Any]:
    """Load only the frozen architecture/data contract, never a run budget."""
    config = yaml.safe_load(Path(path).read_text())
    if config.get("schema_name") != CONFIG_SCHEMA or config.get("run_authorization") != "not_approved":
        raise ValueError("V1.12 implementation config schema or authorization mismatch")
    return _validate_common(config)


def load_part_se3_training_config(path: str | Path) -> dict[str, Any]:
    """Load an explicitly approved V1.12 reference or V1.13 compact run contract."""
    config = yaml.safe_load(Path(path).read_text())
    schema = config.get("schema_name")
    if schema not in (FORMAL_CONFIG_SCHEMA, COMPACT_FORMAL_CONFIG_SCHEMA):
        raise ValueError("part-SE3 formal config schema mismatch")
    if config.get("run_authorization") != "approved":
        raise ValueError("part-SE3 formal run is not approved")
    work_version = WORK_VERSION if schema == FORMAL_CONFIG_SCHEMA else COMPACT_WORK_VERSION
    _validate_common(config, work_version=work_version)
    if schema == FORMAL_CONFIG_SCHEMA:
        if config.get("data_backend", "reference") != "reference":
            raise ValueError("V1.12 formal config requires the reference backend")
    else:
        if config.get("data_backend") != "compact" or not config.get("compact_cache_root"):
            raise ValueError("V1.13 formal config requires an explicit compact cache")
        config["compact_cache_root"] = _absolute(config["compact_cache_root"])
    if config.get("training") != EXPECTED_TRAINING:
        raise ValueError("V1.12 formal training budget mismatch")
    if config.get("resources") != {"physical_gpus": [0, 2], "nofile_limit": 262144}:
        raise ValueError("V1.12 formal resources must be physical GPUs 0 and 2")
    config["output_root"] = _absolute(config["output_root"])
    return config


def initialize_random_model(config: Mapping[str, Any], device: torch.device | str = "cpu"):
    if config.get("initialization") != "random":
        raise ValueError("part-SE3 only permits random initialization")
    model_config = dict(config["model"])
    if model_config != EXPECTED_MODEL:
        raise ValueError("V1.12 model contract mismatch")
    return PartSE3ObjectInteractionCmv2V112Model(SimpleNamespace(**model_config)).to(device)


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, *,
                       epoch: int, step: int, best_metric: float) -> dict[str, Any]:
    return {
        "schema_name": CHECKPOINT_SCHEMA,
        "work_version": WORK_VERSION,
        "architecture_version": PART_SE3_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "best_metric": float(best_metric),
    }


def restore_checkpoint(path: str | Path, model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    identity = (payload.get("schema_name"), payload.get("work_version"),
                payload.get("architecture_version"))
    if identity != (CHECKPOINT_SCHEMA, WORK_VERSION, PART_SE3_VERSION):
        raise ValueError("checkpoint is not a V1.12 direct-part-SE(3) checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return {key: payload[key] for key in ("epoch", "step", "best_metric")}


def build_part_se3_dataset(config: Mapping[str, Any], group: str, split: str, *,
                           fixed_stride: int | None = None,
                           max_sequences: int | None = None,
                           hand_points_per_side: int | None = None,
                           hand_sampling_seed: int = 42):
    """Build the explicit reference or V1.13 compact backend."""
    if group not in GROUPS or split not in ("train", "val", "test"):
        raise ValueError("invalid V1.12 group or split")
    backend = str(config.get("data_backend", "reference"))
    if backend == "compact":
        if max_sequences is not None:
            raise ValueError("compact backend does not support sequence-count truncation")
        from .compact_endpoint import CompactEndpointDataset
        root = config.get("compact_cache_root")
        if not root:
            raise ValueError("compact backend requires compact_cache_root")
        return CompactEndpointDataset(root, group, split, fixed_stride=fixed_stride)
    if backend != "reference":
        raise ValueError(f"unsupported part-SE3 data backend: {backend}")
    domain, variant = group.split("/")
    split_root = Path(config["split_root"])
    strides = (fixed_stride,) if fixed_stride is not None else tuple(
        config["data"]["train_strides" if split == "train" else "validation_strides"])
    if domain == "oakink2":
        base = ThreeDomainTransitions(
            [{"name": domain, "hand_variant": variant,
              "index": str(split_root / "index.json"),
              "manifest": str(split_root / "cache_manifest.json")}],
            split, num_obj_points=1024, train_stride_values={domain: strides},
            fixed_stride=fixed_stride, active_only=False, base_seed=42,
            max_sequences_per_domain=max_sequences, allow_manifest_split_override=True,
            hand_points_per_side=hand_points_per_side,
            hand_sampling_seed=hand_sampling_seed)
        return OakInkWholePartTransitions(base, config["oakink2_parts"]["adapter_root"])
    index = json.loads((split_root / "index.json").read_text())
    articulation = json.loads(Path(config["articulation_metadata"]).read_text())["articulation"]
    rows = [row for row in index["sequences"][split] if row["group"] == group]
    if max_sequences is not None:
        rows = rows[:int(max_sequences)]
    specs = [{"name": domain, "hand_variant": variant, "path": row["path"], "id": row["id"],
              "articulation": articulation if domain == "arctic" else {"num_links": 1, "joints": []}}
             for row in rows]
    return RigidArticulatedPartTransitions(
        specs, split, num_obj_points=1024, stride_values=strides, base_seed=42,
        hand_points_per_side=hand_points_per_side,
        hand_sampling_seed=hand_sampling_seed)
