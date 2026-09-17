"""Validated V1.2 GRAB/MANO run configuration."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_grab_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("schema_name") != "object_interaction_cmv2_grab_v1_2":
        raise ValueError("Expected V1.2 GRAB configuration")
    source = cfg.get("source", {})
    if source.get("name") != "grab" or source.get("hand_source") != "mano" or source.get("stride") != 1:
        raise ValueError("V1.2 permits only GRAB/MANO stride=1")
    if cfg.get("model", {}).get("interaction_radius_m") != 0.02 or cfg.get("model", {}).get("knn_k") != 32:
        raise ValueError("V1.2 requires hard 2 cm swept KNN32")
    if cfg["model"].get("interaction_mode") != "swept":
        raise ValueError("V1.2 requires swept interaction mode")
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
