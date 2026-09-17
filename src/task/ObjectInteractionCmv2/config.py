"""Validated V1.2/V1.3 GRAB and V1.4 three-domain run configurations."""
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
        if version not in ("V1.3.2", "V1.3.3", "V1.3.4"):
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
        elif training.get("mode") in ("ddp_calibration", "ddp_formal"):
            if (version != "V1.3.4" or training.get("device_ids") != [2, 3]
                or training.get("world_size") != 2 or training.get("seed") != 42
                or training.get("learning_rate") != 0.001
                or training.get("expected_train_sequences") != 1068
                or training.get("expected_train_pairs") != 326722):
                raise ValueError("V1.3.4 dual-GPU GRAB contract mismatch")
            if training["mode"] == "ddp_calibration":
                if training.get("batch_candidates") != [32, 64, 96, 128, 160, 192, 224, 256]:
                    raise ValueError("Unapproved DDP calibration candidates")
                if training.get("target_memory_gib") != 24:
                    raise ValueError("Unapproved DDP memory target")
            elif (not isinstance(training.get("batch_size_per_gpu"), int)
                  or training["batch_size_per_gpu"] not in (32, 64, 96, 128, 160, 192, 224, 256)
                  or training.get("epochs") != 4
                  or training.get("max_duration_s") != 28800
                  or training.get("checkpoint_interval") != 200):
                raise ValueError("V1.3.4 formal DDP budget mismatch")
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


def load_three_domain_config(path: str | Path) -> dict:
    """Load and validate the finalized V1.4 Cmv2 three-domain contract."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("schema_name") != "object_interaction_cmv2_three_domain_v1_4":
        raise ValueError("Expected ObjectInteractionCmv2 V1.4 three-domain configuration")
    if cfg.get("modification_version") != "V1.4.2":
        raise ValueError("Unsupported V1.4 three-domain run version")
    model = cfg.get("model", {})
    expected_model = {
        "architecture_version": "v1_3_rigid_only",
        "hidden_width": 128,
        "num_tokens": 16,
        "use_residual": False,
        "knn_k": 32,
        "interaction_radius_m": 0.02,
        "interaction_mode": "swept",
        "feature_scale_m": 0.02,
        "frame_dt_s": 1 / 30,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"V1.4 model contract mismatch for {key}: {model.get(key)!r} != {expected!r}")
    if any("effect" in str(key).lower() for section in (cfg, model, cfg.get("training", {}))
           for key in section):
        raise ValueError("V1.4 does not configure an effect branch")
    sources = cfg.get("sources")
    if not isinstance(sources, list) or [str(item.get("name")) for item in sources] != ["grab", "arctic", "oakink2"]:
        raise ValueError("V1.4 requires sources in grab/arctic/oakink2 order")
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("V1.4 source entries must be mappings")
        variant = str(item.get("hand_variant", ""))
        if variant not in ("mano", "inspire_f1"):
            raise ValueError("V1.4 source entries must declare hand_variant=mano or inspire_f1")
        for key in ("index", "manifest"):
            value = os.path.expandvars(str(item.get(key, "")))
            if not value or "$" in value:
                raise ValueError(f"unresolved V1.4 source {key}: {value!r}")
            item[key] = str(Path(value).resolve())
            if not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
    data = cfg.get("data", {})
    if data.get("num_obj_points") != 1024:
        raise ValueError("V1.4 requires 1024 object points")
    if "hand_points" in data:
        raise ValueError("V1.4 does not accept a fixed hand_points value; use source hand_variant")
    stride_values = data.get("source_stride_values", {})
    if set(stride_values) != {"grab", "arctic", "oakink2"}:
        raise ValueError("V1.4 requires stride values for exactly three domains")
    for domain, values in stride_values.items():
        if tuple(int(value) for value in values) != tuple(range(1, 11)):
            raise ValueError(f"V1.4 {domain} stride must be 1..10")
    if data.get("eval_stride") != 2 or data.get("active_only") is not True:
        raise ValueError("V1.4 requires active train rows and eval stride 2")
    domains = list(data.get("source_domains", []))
    if domains != ["grab", "arctic", "oakink2"]:
        raise ValueError("V1.4 source_domains order must be grab/arctic/oakink2")
    probabilities = {str(key): float(value) for key, value in data.get("source_probabilities", {}).items()}
    if set(probabilities) != set(domains) or any(abs(value - 1 / 3) > 1e-8 for value in probabilities.values()):
        raise ValueError("V1.4 source probabilities must be exactly 1/3 per domain")
    training = cfg.get("training", {})
    if training.get("mode") not in ("smoke", "formal"):
        raise ValueError("V1.4 training mode must be smoke or formal")
    if not isinstance(training.get("batch_size"), int) or training["batch_size"] <= 0:
        raise ValueError("V1.4 requires a positive batch_size")
    if not isinstance(training.get("max_steps"), int) or training["max_steps"] <= 0:
        raise ValueError("V1.4 requires a positive max_steps")
    output_root = os.path.expandvars(str(cfg.get("output_root", "")))
    if not output_root or "$" in output_root:
        raise ValueError("unresolved V1.4 output_root")
    cfg["output_root"] = str(Path(output_root).resolve())
    cfg["config_path"] = str(path)
    return cfg


def load_two_domain_mano_config(path: str | Path) -> dict:
    """Load the approved V1.4.4 GRAB+ARCTIC MANO continuation contract."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or cfg.get("schema_name") != "object_interaction_cmv2_two_domain_mano_v1_4":
        raise ValueError("Expected ObjectInteractionCmv2 V1.4.4 two-domain MANO configuration")
    if cfg.get("modification_version") != "V1.4.4":
        raise ValueError("Unsupported V1.4.4 two-domain MANO run version")
    model = cfg.get("model", {})
    expected_model = {
        "architecture_version": "v1_3_rigid_only", "hidden_width": 128, "num_tokens": 16,
        "use_residual": False, "knn_k": 32, "interaction_radius_m": 0.02,
        "interaction_mode": "swept", "feature_scale_m": 0.02, "frame_dt_s": 1 / 30,
    }
    if any(model.get(key) != expected for key, expected in expected_model.items()):
        raise ValueError("V1.4.4 model contract mismatch")
    sources = cfg.get("sources")
    if not isinstance(sources, list) or [item.get("name") for item in sources] != ["grab", "arctic"]:
        raise ValueError("V1.4.4 requires GRAB then ARCTIC sources")
    for item in sources:
        if item.get("hand_variant") != "mano":
            raise ValueError("V1.4.4 two-domain training permits MANO only")
        for key in ("index", "manifest"):
            value = os.path.expandvars(str(item.get(key, "")))
            if not value or "$" in value:
                raise ValueError(f"unresolved V1.4.4 source {key}")
            item[key] = str(Path(value).resolve())
            if not Path(item[key]).is_file():
                raise FileNotFoundError(item[key])
    data = cfg.get("data", {})
    if data.get("num_obj_points") != 1024 or data.get("active_only") is not True or data.get("eval_stride") != 2:
        raise ValueError("V1.4.4 data contract mismatch")
    probabilities = {str(key): float(value) for key, value in data.get("train_source_probabilities", {}).items()}
    if probabilities != {"grab": 0.5, "arctic": 0.5}:
        raise ValueError("V1.4.4 requires equal GRAB/ARCTIC source probabilities")
    strides = data.get("train_stride_values", {})
    if set(strides) != {"grab", "arctic"} or any(tuple(values) != tuple(range(1, 11)) for values in strides.values()):
        raise ValueError("V1.4.4 requires train stride 1..10 for both domains")
    training = cfg.get("training", {})
    if (training.get("device") != "cuda:0" or training.get("batch_size") != 160
            or training.get("validation_batch_size") != 160 or training.get("max_epochs") != 16
            or training.get("max_duration_s") != 28800 or training.get("checkpoint_interval") != 200
            or training.get("learning_rate") != 0.001 or training.get("seed") != 42
            or training.get("minimum_free_memory_gib") != 35):
        raise ValueError("V1.4.4 training budget mismatch")
    output_root = os.path.expandvars(str(cfg.get("output_root", "")))
    if not output_root or "$" in output_root:
        raise ValueError("unresolved V1.4.4 output_root")
    cfg["output_root"] = str(Path(output_root).resolve())
    cfg["config_path"] = str(path)
    return cfg
