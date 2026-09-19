"""Append-only, point-cloud-free transition shards with GPU staging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


SCHEMA = "cmresidual.cm_buffer.v1"
TRANSITION_ONLY_SCHEMA = "cmresidual.cm_buffer.transition_only.v2"
TRANSITION_ONLY_FIELDS = (
    "q_pre", "dq_pre", "object_pose_pre", "residual_action", "reference_target",
    "pd_target", "applied_delta", "authority_limited", "reference_index",
    "reference_start_index", "window_progress", "reference_delta_xi", "q_post",
    "dq_post", "object_pose_post", "actual_delta_xi",
)


class CmTransitionBuffer:
    """Persist only enough state to deterministically rebuild Cmv2 inputs offline."""

    def __init__(self, output_dir: str | Path, *, rank: int, flush_every: int,
                 metadata: Mapping[str, object], schema: str = SCHEMA,
                 expected_fields: tuple[str, ...] | None = None) -> None:
        if flush_every <= 0:
            raise ValueError("CmBuffer flushEvery must be positive")
        if schema not in (SCHEMA, TRANSITION_ONLY_SCHEMA):
            raise ValueError(f"Unsupported CmBuffer schema {schema}")
        if schema == TRANSITION_ONLY_SCHEMA and expected_fields != TRANSITION_ONLY_FIELDS:
            raise ValueError("transition_only CmBuffer requires the fixed V1.16 field set")
        self.root = Path(output_dir).expanduser().resolve() / f"rank_{rank:03d}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.flush_every = int(flush_every)
        self.schema = schema
        self.expected_fields = expected_fields
        self.pending: list[dict[str, torch.Tensor]] = []
        self.pending_shapes: dict[str, tuple[int, ...]] | None = None
        self.count = 0
        self.chunk_index = 0
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema": self.schema,
            "rank": int(rank),
            "flush_every": self.flush_every,
            "sample_count": 0,
            "chunks": [],
            "storage": "compressed_npz_no_point_cloud",
            "transfer": "gpu_staging_batched_cpu_copy",
            "reconstruction": "pinned assets/reference/Cmv2 schema plus stored q/object poses/targets",
            "metadata": dict(metadata),
        }
        self._write_manifest()

    def append(self, record: Mapping[str, torch.Tensor]) -> None:
        if not record:
            raise ValueError("CmBuffer record is empty")
        fields = tuple(record)
        if self.expected_fields is not None and fields != self.expected_fields:
            raise ValueError(f"CmBuffer fields must be exactly {self.expected_fields}, got {fields}")
        tensors: dict[str, torch.Tensor] = {}
        for name, value in record.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError("CmBuffer records must be non-scalar tensors")
            tensors[name] = value.detach().contiguous()
        sizes = {value.shape[0] for value in tensors.values()}
        if len(sizes) != 1:
            raise ValueError("CmBuffer fields must share a batch size")
        shapes = {name: tuple(value.shape[1:]) for name, value in tensors.items()}
        if self.pending_shapes is None:
            self.pending_shapes = shapes
        elif shapes != self.pending_shapes:
            raise ValueError("CmBuffer record schema changed within a shard")
        self.pending.append(tensors)
        self.count += sizes.pop()
        if self.count >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys = tuple(self.pending[0])
        if any(tuple(item) != keys for item in self.pending):
            raise ValueError("CmBuffer record schema changed within a shard")
        staged = {key: torch.cat([item[key] for item in self.pending], dim=0) for key in keys}
        if not all(torch.isfinite(value).all() for value in staged.values()):
            raise FloatingPointError("CmBuffer staged record has non-finite values")
        # One transfer per field per shard, rather than a synchronizing transfer for
        # every environment transition.  Fieldwise arrays preserve original dtypes.
        payload = {key: value.to("cpu").contiguous().numpy() for key, value in staged.items()}
        size = next(iter(payload.values())).shape[0]
        filename = f"chunk_{self.chunk_index:06d}.npz"
        target = self.root / filename
        temporary = self.root / f".{filename}.tmp.npz"
        np.savez_compressed(temporary, **payload)
        temporary.replace(target)
        self.manifest["sample_count"] += int(size)
        self.manifest["chunks"].append({"path": filename, "samples": int(size),
                                        "fields": {name: list(value.shape[1:]) for name, value in payload.items()}})
        self.chunk_index += 1
        self.count = 0
        self.pending.clear()
        self.pending_shapes = None
        self._write_manifest()

    def close(self) -> None:
        """Flush outstanding staged transitions at a controlled lifecycle boundary."""
        self.flush()

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)
