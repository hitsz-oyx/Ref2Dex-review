from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.task.correspondence_ptv3.sampling.stratified import (
    validate_stratified_edge_sampler_config,
)


DEFAULT_LOGIT_NEAR_RADIUS = 0.02
DEFAULT_LOGIT_FAR_MIN_RADIUS = 0.04


@dataclass(frozen=True)
class EdgeSamplerConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ContactSupervisionConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class CorrespondenceModelConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ResolvedCorrespondenceComponents:
    edge_sampler: EdgeSamplerConfig
    contact_supervision: ContactSupervisionConfig
    model: CorrespondenceModelConfig


def _get_explicit_override_keys(
    owner: Any,
    explicit_override_keys: set[str] | None = None,
) -> set[str]:
    if explicit_override_keys is not None:
        return set(explicit_override_keys)
    return set(getattr(owner, "_explicit_override_keys", set()) or set())


def _selector_conflict(
    *,
    explicit_override_keys: set[str],
    new_selector_key: str,
    new_selector_value: str | None,
    legacy_selector_key: str,
    legacy_selector_value: str | None,
    error_label: str,
) -> None:
    if not new_selector_value or legacy_selector_key not in explicit_override_keys:
        return
    if str(new_selector_value).lower() != str(legacy_selector_value).lower():
        raise ValueError(
            f"Conflicting {error_label} configuration: "
            f"{new_selector_key}={new_selector_value!r} vs "
            f"{legacy_selector_key}={legacy_selector_value!r}."
        )


def resolve_logit_near_radius(
    meta_cfg: Any | None,
    default: float = DEFAULT_LOGIT_NEAR_RADIUS,
) -> float:
    if meta_cfg is None:
        return float(default)
    value = getattr(meta_cfg, "logit_near_radius", None)
    if value is None:
        value = getattr(meta_cfg, "logit_pos_radius", None)
    if value is None:
        value = default
    return float(value)


def resolve_logit_far_min_radius(
    meta_cfg: Any | None,
    default: float = DEFAULT_LOGIT_FAR_MIN_RADIUS,
) -> float:
    if meta_cfg is None:
        return float(default)
    value = getattr(meta_cfg, "logit_far_min_radius", None)
    if value is None:
        value = getattr(meta_cfg, "logit_neg_min_radius", None)
    if value is None:
        value = default
    return float(value)


def _normalize_radius_pair(
    logit_near_radius: Any,
    logit_far_min_radius: Any,
) -> tuple[float, float]:
    near = float(logit_near_radius)
    far = float(logit_far_min_radius)
    if near <= 0.0:
        raise ValueError("logit_near_radius must be positive.")
    if far < near:
        raise ValueError("logit_far_min_radius must be >= logit_near_radius.")
    return near, far


def _value_or_fallback(
    owner: Any,
    key: str,
    fallback: Any,
) -> Any:
    value = getattr(owner, key, None)
    return fallback if value is None else value


def _normalize_weights(
    raw: Any,
    *,
    num_contact_bins: int,
    field_name: str,
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(value) for value in raw)
    if len(values) != int(num_contact_bins):
        raise ValueError(
            f"{field_name} must match num_contact_bins, got "
            f"{len(values)} vs {int(num_contact_bins)}."
        )
    return values


def _resolve_edge_sampler_config(
    meta_cfg: Any,
    *,
    explicit_override_keys: set[str] | None = None,
) -> EdgeSamplerConfig:
    explicit = _get_explicit_override_keys(meta_cfg, explicit_override_keys)
    edge_sampler = getattr(meta_cfg, "edge_sampler", None)
    new_name = getattr(edge_sampler, "name", None) if edge_sampler is not None else None
    legacy_name = str(getattr(meta_cfg, "logit_sampling_mode", "balanced") or "balanced").lower()
    _selector_conflict(
        explicit_override_keys=explicit,
        new_selector_key="meta.edge_sampler.name",
        new_selector_value=str(new_name).lower() if new_name is not None else None,
        legacy_selector_key="meta.logit_sampling_mode",
        legacy_selector_value=legacy_name,
        error_label="edge sampler",
    )

    source = edge_sampler if new_name is not None else meta_cfg
    name = str(new_name).lower() if new_name is not None else legacy_name

    if name == "balanced":
        near, far = _normalize_radius_pair(
            _value_or_fallback(source, "logit_near_radius", resolve_logit_near_radius(meta_cfg)),
            _value_or_fallback(source, "logit_far_min_radius", resolve_logit_far_min_radius(meta_cfg)),
        )
        return EdgeSamplerConfig(
            name=name,
            params={
                "k_near_logit": int(_value_or_fallback(source, "k_near_logit", getattr(meta_cfg, "k_near_logit", 32))),
                "k_far_logit": int(_value_or_fallback(source, "k_far_logit", getattr(meta_cfg, "k_far_logit", 32))),
                "logit_near_radius": near,
                "logit_far_min_radius": far,
            },
        )

    if name == "stratified":
        raw_distance_edges = tuple(
            float(value)
            for value in _value_or_fallback(
                source,
                "distance_edges",
                getattr(meta_cfg, "logit_stratified_distance_edges", (0.005, 0.015, 0.03, 0.06)),
            )
        )
        raw_quotas = tuple(
            int(value)
            for value in _value_or_fallback(
                source,
                "quotas",
                getattr(meta_cfg, "logit_stratified_quotas", (16, 32, 32, 32, 16)),
            )
        )
        validate_stratified_edge_sampler_config(
            raw_distance_edges,
            raw_quotas,
        )
        distance_edges = raw_distance_edges
        quotas = raw_quotas
        near, far = _normalize_radius_pair(
            _value_or_fallback(source, "logit_near_radius", resolve_logit_near_radius(meta_cfg)),
            _value_or_fallback(source, "logit_far_min_radius", resolve_logit_far_min_radius(meta_cfg)),
        )
        return EdgeSamplerConfig(
            name=name,
            params={
                "distance_edges": distance_edges,
                "quotas": quotas,
                "logit_near_radius": near,
                "logit_far_min_radius": far,
            },
        )

    if name == "dense":
        near, far = _normalize_radius_pair(
            _value_or_fallback(source, "logit_near_radius", resolve_logit_near_radius(meta_cfg)),
            _value_or_fallback(source, "logit_far_min_radius", resolve_logit_far_min_radius(meta_cfg)),
        )
        return EdgeSamplerConfig(
            name=name,
            params={
                "logit_near_radius": near,
                "logit_far_min_radius": far,
            },
        )

    if new_name is not None:
        raise ValueError(f"Unsupported edge sampler: {name!r}.")
    raise ValueError(f"logit_sampling_mode must be 'balanced', 'dense', or 'stratified', got {legacy_name!r}.")


def _resolve_contact_supervision_config(
    meta_cfg: Any,
    *,
    explicit_override_keys: set[str] | None = None,
) -> ContactSupervisionConfig:
    explicit = _get_explicit_override_keys(meta_cfg, explicit_override_keys)
    contact_supervision = getattr(meta_cfg, "contact_supervision", None)
    new_name = (
        getattr(contact_supervision, "name", None)
        if contact_supervision is not None
        else None
    )
    legacy_name = str(getattr(meta_cfg, "contact_supervision_mode", "bin") or "bin").lower()
    _selector_conflict(
        explicit_override_keys=explicit,
        new_selector_key="meta.contact_supervision.name",
        new_selector_value=str(new_name).lower() if new_name is not None else None,
        legacy_selector_key="meta.contact_supervision_mode",
        legacy_selector_value=legacy_name,
        error_label="contact supervision",
    )

    source = contact_supervision if new_name is not None else meta_cfg
    name = str(new_name).lower() if new_name is not None else legacy_name

    if name == "soft":
        return ContactSupervisionConfig(name=name, params={})

    if name == "bin":
        num_contact_bins = int(
            _value_or_fallback(source, "num_contact_bins", getattr(meta_cfg, "num_contact_bins", 10))
        )
        if num_contact_bins <= 0:
            raise ValueError("num_contact_bins must be positive.")
        decode_mode = str(
            _value_or_fallback(
                source,
                "contact_bin_decode_mode",
                getattr(meta_cfg, "contact_bin_decode_mode", "expectation"),
            )
        ).lower()
        if decode_mode not in {"expectation", "argmax"}:
            raise ValueError(
                "contact_bin_decode_mode must be 'expectation' or 'argmax', got "
                f"{decode_mode!r}."
            )
        return ContactSupervisionConfig(
            name=name,
            params={
                "num_contact_bins": num_contact_bins,
                "decode_mode": decode_mode,
                "contact_bin_weights": _normalize_weights(
                    _value_or_fallback(
                        source,
                        "contact_bin_weights",
                        getattr(meta_cfg, "contact_bin_weights", None),
                    ),
                    num_contact_bins=num_contact_bins,
                    field_name="contact_bin_weights",
                ),
                "edge_contact_bin_weights": _normalize_weights(
                    _value_or_fallback(
                        source,
                        "edge_contact_bin_weights",
                        getattr(meta_cfg, "edge_contact_bin_weights", None),
                    ),
                    num_contact_bins=num_contact_bins,
                    field_name="edge_contact_bin_weights",
                ),
                "contact_bin_weight_path": (
                    None
                    if _value_or_fallback(
                        source,
                        "contact_bin_weight_path",
                        getattr(meta_cfg, "contact_bin_weight_path", None),
                    )
                    in {None, ""}
                    else str(
                        _value_or_fallback(
                            source,
                            "contact_bin_weight_path",
                            getattr(meta_cfg, "contact_bin_weight_path", None),
                        )
                    )
                ),
            },
        )

    raise ValueError(f"contact_supervision_mode must be 'bin' or 'soft', got {name!r}.")


def _resolve_correspondence_model_config(model_cfg: Any | None) -> CorrespondenceModelConfig:
    if model_cfg is None:
        return CorrespondenceModelConfig(name="ptv3_concat", params={})

    name = getattr(model_cfg, "name", None)
    if name is not None:
        return CorrespondenceModelConfig(name=str(name).lower(), params={})

    class_path = str(getattr(model_cfg, "class_path", "") or "")
    type_name = str(getattr(model_cfg, "type", "") or "").lower()
    if class_path.endswith(".StaticHOCPTv3") or type_name in {"static_hoc_ptv3", "ptv3_concat"}:
        return CorrespondenceModelConfig(name="ptv3_concat", params={})
    return CorrespondenceModelConfig(name="ptv3_concat", params={})


def resolve_correspondence_components(
    cfg: Any,
    *,
    explicit_override_keys: set[str] | None = None,
) -> ResolvedCorrespondenceComponents:
    explicit = _get_explicit_override_keys(cfg, explicit_override_keys)
    meta_cfg = getattr(cfg, "meta", cfg)
    if not explicit and meta_cfg is not cfg:
        explicit = _get_explicit_override_keys(meta_cfg, explicit_override_keys)
    return ResolvedCorrespondenceComponents(
        edge_sampler=_resolve_edge_sampler_config(
            meta_cfg,
            explicit_override_keys=explicit,
        ),
        contact_supervision=_resolve_contact_supervision_config(
            meta_cfg,
            explicit_override_keys=explicit,
        ),
        model=_resolve_correspondence_model_config(
            getattr(cfg, "model", None),
        ),
    )


# Legacy compatibility. New code should use
# resolve_correspondence_components().
def resolve_edge_sampler_config(
    meta_cfg: Any,
    *,
    explicit_override_keys: set[str] | None = None,
) -> EdgeSamplerConfig:
    return _resolve_edge_sampler_config(
        meta_cfg,
        explicit_override_keys=explicit_override_keys,
    )


# Legacy compatibility. New code should use
# resolve_correspondence_components().
def resolve_contact_supervision_config(
    meta_cfg: Any,
    *,
    explicit_override_keys: set[str] | None = None,
) -> ContactSupervisionConfig:
    return _resolve_contact_supervision_config(
        meta_cfg,
        explicit_override_keys=explicit_override_keys,
    )


# Legacy compatibility. New code should use
# resolve_correspondence_components().
def resolve_correspondence_model_config(
    model_cfg: Any | None,
) -> CorrespondenceModelConfig:
    return _resolve_correspondence_model_config(model_cfg)


__all__ = [
    "DEFAULT_LOGIT_FAR_MIN_RADIUS",
    "DEFAULT_LOGIT_NEAR_RADIUS",
    "ContactSupervisionConfig",
    "CorrespondenceModelConfig",
    "EdgeSamplerConfig",
    "ResolvedCorrespondenceComponents",
    "resolve_contact_supervision_config",
    "resolve_correspondence_components",
    "resolve_correspondence_model_config",
    "resolve_edge_sampler_config",
    "resolve_logit_far_min_radius",
    "resolve_logit_near_radius",
]
