from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.base import TaskConfig, task_config_from_dict
from src.base.base_config import apply_override
from src.base.cli import collect_override_keys


CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG_NAME = "base"
COMPONENT_KEYS = ("edge_sampler", "contact_supervision", "model")


def config_dir() -> Path:
    return CONFIG_DIR


def load_correspondence_config(
    config: str | Path | Mapping[str, Any] | TaskConfig | None = None,
    *,
    overrides: list[str] | None = None,
) -> TaskConfig:
    raw = load_correspondence_raw_config(config, overrides=overrides)
    return runtime_task_config_from_mapping(raw)


def load_correspondence_raw_config(
    config: str | Path | Mapping[str, Any] | TaskConfig | None = None,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(config, TaskConfig):
        return apply_resolved_overrides(config.to_dict(), overrides or [])
    if isinstance(config, Mapping):
        return apply_resolved_overrides(dict(config), overrides or [])
    return _load_raw_config_reference(config, overrides=overrides or [])


def load_correspondence_config_from_args(args: Any) -> TaskConfig:
    config_ref = getattr(args, "config", None) or DEFAULT_CONFIG_NAME
    overrides = list(getattr(args, "set", []) or [])
    override_keys = collect_override_keys(overrides)

    if getattr(args, "data", None) is not None:
        overrides.append(f"data.train_path={json.dumps(str(args.data))}")
        override_keys.add("data.train_path")
    if getattr(args, "output_dir", None) is not None:
        overrides.append(f"train.output_dir={json.dumps(str(args.output_dir))}")
        override_keys.add("train.output_dir")
    if getattr(args, "device", None) is not None:
        overrides.append(f"train.device={json.dumps(str(args.device))}")
        override_keys.add("train.device")
    if bool(getattr(args, "distributed", False)):
        overrides.append("train.distributed.enable=true")
        override_keys.add("train.distributed.enable")

    cfg = load_correspondence_config(config_ref, overrides=overrides)
    setattr(cfg, "_explicit_override_keys", set(override_keys))
    setattr(cfg, "_explicit_name", "name" in override_keys)
    setattr(cfg.wandb, "_explicit_name", "wandb.name" in override_keys)
    setattr(
        cfg.train,
        "_explicit_output_dir",
        bool(getattr(args, "output_dir", None) is not None or "train.output_dir" in override_keys),
    )
    return cfg


def runtime_task_config_from_mapping(raw: Mapping[str, Any]) -> TaskConfig:
    payload = copy.deepcopy(dict(raw))
    component_payloads = {
        key: copy.deepcopy(payload.pop(key))
        for key in COMPONENT_KEYS
        if key in payload
    }
    cfg = task_config_from_dict(payload)
    for key, value in component_payloads.items():
        setattr(cfg, key, value)
    return cfg


def apply_resolved_overrides(
    raw: Mapping[str, Any],
    overrides: list[str],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(raw))
    for override in overrides:
        key, value = _split_override(override)
        if key in COMPONENT_KEYS and "." not in key and value:
            raise ValueError(
                f"Component-group override {key}={value} requires a Hydra recipe source. "
                "Load by config name or by a YAML path under correspondence_ptv3/configs."
            )
        apply_override(payload, override)
    return payload


def _load_raw_config_reference(
    config: str | Path | None,
    *,
    overrides: list[str],
) -> dict[str, Any]:
    if config is None:
        return compose_correspondence_config(DEFAULT_CONFIG_NAME, overrides=overrides)

    path = Path(str(config)).expanduser()
    if path.suffix.lower() == ".json" and path.exists():
        return apply_resolved_overrides(
            json.loads(path.read_text(encoding="utf-8")),
            overrides,
        )

    if path.exists() and path.suffix.lower() in {".yaml", ".yml"}:
        try:
            config_name = _path_to_config_name(path.resolve())
        except ValueError:
            raw = OmegaConf.to_container(
                OmegaConf.load(path),
                resolve=False,
            )
            if not isinstance(raw, dict):
                raise TypeError(f"Expected YAML mapping, got {type(raw).__name__}.")
            if "defaults" in raw:
                raise ValueError(
                    "External YAML with Hydra defaults is not supported. "
                    "Move the recipe under correspondence_ptv3/configs or load it by config name."
                )
            return apply_resolved_overrides(raw, overrides)
        return compose_correspondence_config(config_name, overrides=overrides)

    return compose_correspondence_config(
        _normalize_config_name(str(config)),
        overrides=overrides,
    )


def compose_correspondence_config(
    config_name: str = DEFAULT_CONFIG_NAME,
    *,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        cfg = compose(
            config_name=_normalize_config_name(config_name),
            overrides=list(overrides or []),
        )
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected composed config mapping, got {type(raw).__name__}.")
    return raw


def _normalize_config_name(config_name: str) -> str:
    name = str(config_name or DEFAULT_CONFIG_NAME).strip()
    if not name:
        return DEFAULT_CONFIG_NAME
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"}:
        candidate = path if path.is_absolute() else (Path.cwd() / path).resolve()
        if candidate.exists():
            return _path_to_config_name(candidate)
        name = path.with_suffix("").as_posix()
    return name.lstrip("/")


def _path_to_config_name(path: Path) -> str:
    try:
        relative = path.relative_to(CONFIG_DIR)
    except ValueError as exc:
        raise ValueError(f"{path} is not under {CONFIG_DIR}") from exc
    return relative.with_suffix("").as_posix()


def _split_override(override: str) -> tuple[str, str]:
    item = override.strip()
    if item.startswith("--"):
        item = item[2:]
    if "=" not in item:
        raise ValueError(f"Override must use key=value syntax, got: {override}")
    key, value = item.split("=", 1)
    return key.strip(), value.strip()


__all__ = [
    "DEFAULT_CONFIG_NAME",
    "apply_resolved_overrides",
    "compose_correspondence_config",
    "config_dir",
    "load_correspondence_config",
    "load_correspondence_config_from_args",
    "load_correspondence_raw_config",
    "runtime_task_config_from_mapping",
]
