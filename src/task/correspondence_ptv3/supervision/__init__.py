from __future__ import annotations

from src.task.correspondence_ptv3.composition import (
    ContactSupervisionConfig,
    resolve_contact_supervision_config,
)

from .base import ContactSupervision
from .bin import BinContactSupervision
from .soft import SoftContactSupervision


def build_contact_supervision(
    config: ContactSupervisionConfig,
) -> ContactSupervision:
    if config.name == "soft":
        return SoftContactSupervision()
    if config.name == "bin":
        return BinContactSupervision(**config.params)
    raise ValueError(f"Unsupported contact supervision: {config.name!r}.")


def build_contact_supervision_from_meta(
    meta_cfg,
    *,
    explicit_override_keys: set[str] | None = None,
) -> ContactSupervision:
    return build_contact_supervision(
        resolve_contact_supervision_config(
            meta_cfg,
            explicit_override_keys=explicit_override_keys,
        )
    )


__all__ = [
    "BinContactSupervision",
    "ContactSupervision",
    "SoftContactSupervision",
    "build_contact_supervision",
    "build_contact_supervision_from_meta",
]
