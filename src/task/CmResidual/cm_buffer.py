"""Append-only, point-cloud-free transition shards for frozen Cmv2 diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


SCHEMA = "cmresidual.cm_buffer.v1"


class CmTransitionBuffer:
    """Persist only enough state to deterministically rebuild Cmv2 inputs offline."""

    def __init__(self, output_dir: str | Path, *, rank: int, flush_every: int,
                 metadata: Mapping[str, object]) -> None:
        if flush_every <= 0:
            raise ValueError("CmBuffer flushEvery must be positive")
        self.root = Path(output_dir).expanduser().resolve() / f"rank_{rank:03d}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.flush_every = int(flush_every)
        self.pending: list[dict[str, np.ndarray]] = []
        self.count = 0
        self.chunk_index = 0
        self.manifest_path = self.root / "manifest.json"
        self.manifest = {
            "schema": SCHEMA,
            "rank": int(rank),
            "flush_every": self.flush_every,
            "sample_count": 0,
            "chunks": [],
            "storage": "compressed_npz_no_point_cloud",
            "reconstruction": "pinned assets/reference/Cmv2 schema plus stored q/object poses/targets",
            "metadata": dict(metadata),
        }
        self._write_manifest()

    @staticmethod
    def _cpu_array(value: torch.Tensor) -> np.ndarray:
        if not isinstance(value, torch.Tensor):
            raise TypeError("CmBuffer records must be tensors")
        if not torch.isfinite(value).all():
            raise FloatingPointError("CmBuffer record has non-finite values")
        return value.detach().to("cpu").contiguous().numpy()

    def append(self, record: Mapping[str, torch.Tensor]) -> None:
        arrays = {name: self._cpu_array(value) for name, value in record.items()}
        if not arrays:
            raise ValueError("CmBuffer record is empty")
        sizes = {value.shape[0] for value in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("CmBuffer fields must share a batch size")
        self.pending.append(arrays)
        self.count += sizes.pop()
        if self.count >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        keys = tuple(self.pending[0])
        if any(tuple(item) != keys for item in self.pending):
            raise ValueError("CmBuffer record schema changed within a shard")
        payload = {key: np.concatenate([item[key] for item in self.pending], axis=0) for key in keys}
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
        self._write_manifest()

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)
