"""Validated V1.2/V1.3 GRAB/MANO run configuration."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_grab_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("schema_name") not in (
            "object_interaction_cmv2_grab_v1_2", "object_interaction_cmv2_grab_v1_3"):
        raise ValueError("Expected V1.2 or V1.3 GRAB configuration")
    v13 = cfg["schema_name"].endswith("v1_3")
    source = cfg.get("source", {})
    if source.get("name") != "grab" or source.get("hand_source") != "mano" or source.get("stride") != 1:
        raise ValueError("V1.2 permits only GRAB/MANO stride=1")
    if cfg.get("model", {}).get("interaction_radius_m") != 0.02 or cfg.get("model", {}).get("knn_k") != 32:
        raise ValueError("V1.2 requires hard 2 cm swept KNN32")
    if cfg["model"].get("interaction_mode") != "swept":
        raise ValueError("V1.2 requires swept interaction mode")
    if v13:
        model = cfg["model"]
        if (model.get("architecture_version") != "v1_3_rigid_only" or model.get("use_residual") is not False
            or model.get("hidden_width") != 128 or model.get("num_tokens") != 16
            or model.get("feature_scale_m") != 0.02 or model.get("frame_dt_s") != 1 / 30):
            raise ValueError("V1.3 architecture/scale contract mismatch")
        if any("effect" in str(key).lower() for section in (cfg, model, cfg.get("training", {}))
               for key in section):
            raise ValueError("V1.3.2 does not configure an effect branch")
        version = cfg.get("modification_version", "V1.3.2")
        if version not in ("V1.3.2", "V1.3.3"):
            raise ValueError("Unsupported V1.3 run version")
        training = cfg.get("training", {})
        if training.get("mode") == "calibration":
            if (version != "V1.3.3" or training.get("max_sequences") != 3
                or training.get("batch_size") != 16 or training.get("max_steps") != 8):
                raise ValueError("V1.3.3 calibration budget mismatch")
        elif training.get("mode") == "formal":
            if (version != "V1.3.3" or training.get("max_sequences") is not None
                or training.get("batch_size") != 16 or training.get("max_steps") != 20421
                or training.get("max_duration_s") != 14400
                or training.get("checkpoint_interval") != 1000
                or training.get("expected_train_sequences") != 1068
                or training.get("expected_train_pairs") != 326722
                or training.get("seed") != 42
                or training.get("device") != "cuda:1"
                or training.get("learning_rate") != 0.001):
                raise ValueError("V1.3.3 formal GRAB contract mismatch")
    for name in ("index", "manifest"):
        value = os.path.expandvars(source[name])
        if "$" in value:
            raise ValueError(f"Unresolved environment variable in {name}")
        source[name] = str(Path(value).resolve())
        if not Path(source[name]).is_file():
            raise FileNotFoundError(source[name])
    value = os.path.expandvars(cfg["output_root"])
    if "$" in value:
        raise ValueError("Unresolved output_root")
    cfg["output_root"] = str(Path(value).resolve())
    cfg["config_path"] = str(path)
    return cfg
