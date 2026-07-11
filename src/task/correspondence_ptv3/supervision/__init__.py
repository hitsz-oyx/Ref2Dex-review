from __future__ import annotations

from typing import Any

from src.task.correspondence_ptv3.config import resolve_contact_supervision_config

from .base import ContactSupervision
from .bin import BinContactSupervision
from .soft import SoftContactSupervision


def build_contact_supervision(
    meta_cfg: Any,
    *,
    explicit_override_keys: set[str] | None = None,
) -> ContactSupervision:
    config = resolve_contact_supervision_config(
        meta_cfg,
        explicit_override_keys=explicit_override_keys,
    )
    if config.name == "soft":
        return SoftContactSupervision()
    if config.name == "bin":
        return BinContactSupervision(**config.params)
    raise ValueError(f"Unsupported contact supervision: {config.name!r}.")


__all__ = [
    "BinContactSupervision",
    "ContactSupervision",
    "SoftContactSupervision",
    "build_contact_supervision",
]
