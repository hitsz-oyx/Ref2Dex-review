"""V1.14a compact endpoint pilot identity for fixed 2048 points per hand."""
from __future__ import annotations

from typing import Any, Mapping

from .compact_endpoint import (
    CACHE_SCHEMA as V113_CACHE_SCHEMA,
    WORK_VERSION as V113_WORK_VERSION,
    pack_compact_endpoint as _pack_v113,
    unpack_compact_endpoint as _unpack_v113,
)
from .multi_domain import V114A_HAND_POINTS, V114A_HAND_SAMPLING_CONTRACT


WORK_VERSION = "V1.14"
CACHE_SCHEMA = "object_interaction_cmv2_compact_endpoint_v2"


def pack_compact_endpoint_v114a(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Pack an in-memory pilot record; full cache construction is not authorized."""
    if sample["hand_points"].shape != (V114A_HAND_POINTS, 3):
        raise ValueError("V1.14a compact input must contain 2048 points per hand")
    if sample["hand_valid_mask"].shape != (V114A_HAND_POINTS,):
        raise ValueError("V1.14a compact hand_valid_mask must be [4096]")
    record = _pack_v113(sample)
    record.update({
        "schema_name": CACHE_SCHEMA,
        "work_version": WORK_VERSION,
        "hand_sampling_contract": V114A_HAND_SAMPLING_CONTRACT,
        "bilateral_hand_points": V114A_HAND_POINTS,
    })
    return record


def unpack_compact_endpoint_v114a(record: Mapping[str, Any]) -> dict[str, Any]:
    if (record.get("schema_name") != CACHE_SCHEMA
            or record.get("work_version") != WORK_VERSION
            or record.get("hand_sampling_contract") != V114A_HAND_SAMPLING_CONTRACT
            or int(record.get("bilateral_hand_points", -1)) != V114A_HAND_POINTS):
        raise ValueError("unsupported V1.14a compact endpoint record")
    compatible = dict(record)
    compatible["schema_name"] = V113_CACHE_SCHEMA
    compatible["work_version"] = V113_WORK_VERSION
    return _unpack_v113(compatible)


__all__ = [
    "CACHE_SCHEMA",
    "WORK_VERSION",
    "pack_compact_endpoint_v114a",
    "unpack_compact_endpoint_v114a",
]
