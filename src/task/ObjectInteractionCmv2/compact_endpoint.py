"""V1.13 compact endpoint records and fixed-width sharded random access."""
from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .mixed_training import GROUPS, utc_now
from .part_se3 import endpoint_union_topk


WORK_VERSION = "V1.13"
CACHE_SCHEMA = "object_interaction_cmv2_compact_endpoint_v1"
GROUP_TO_ID = {name: index for index, name in enumerate(GROUPS)}
SPLIT_TO_ID = {name: index for index, name in enumerate(("train", "val", "test"))}
INDEX_DTYPE = np.dtype([
    ("group", "u1"), ("split", "u1"), ("stride", "u1"), ("shard", "<u2"),
    ("offset", "<u8"), ("length", "<u4"),
])
REFERENCE_HAND_KEYS = ("hand_points", "hand_normals", "hand_flow", "hand_valid_mask")
COMPACT_EDGE_KEYS = ("edge_hand_points", "edge_hand_normals", "edge_hand_flow",
                     "edge_source_id", "edge_valid_mask")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_nbytes(value: torch.Tensor) -> int:
    return int(value.nelement() * value.element_size())


def pack_compact_endpoint(sample: Mapping[str, Any], *, radius_m: float = 0.02) -> dict[str, Any]:
    """Replace a full hand stream by the unique hand points used by the final 32 edges."""
    required = {
        "obj_points", "obj_normals", "obj_point_id", "obj_part_id", "part_valid_mask",
        "hand_points", "hand_normals", "hand_flow", "hand_valid_mask", "obj_flow_gt",
        "delta_translation_part_gt", "delta_rotation_part_gt", "delta_time_s", "stride",
        "source", "sequence_id", "source_frame_id", "next_source_frame_id",
    }
    missing = sorted(required - sample.keys())
    if missing:
        raise KeyError(f"compact endpoint sample is missing {missing}")
    object_points = sample["obj_points"].to(torch.float32)
    hand_points = sample["hand_points"].to(torch.float32)
    hand_flow = sample["hand_flow"].to(torch.float32)
    hand_valid = sample["hand_valid_mask"].bool()
    distance, source_id, _, _ = endpoint_union_topk(
        object_points[None], hand_points[None], hand_flow[None], hand_valid[None], k=32)
    source_id = source_id[0].cpu()
    valid = (distance[0].cpu() < float(radius_m)) & hand_valid.cpu()[source_id]
    unique_source_id, edge_lookup = torch.unique(
        source_id.reshape(-1), sorted=True, return_inverse=True)
    if int(source_id.max()) >= 2**15 or int(unique_source_id.numel()) >= 2**15:
        raise ValueError("compact endpoint int16 index range exceeded")
    edge_lookup = edge_lookup.reshape_as(source_id).to(torch.int16)
    unique_source_id = unique_source_id.long()

    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in sample.items()
        if torch.is_tensor(value) and key not in REFERENCE_HAND_KEYS
    }
    tensors.update({
        "unique_hand_points": hand_points.cpu()[unique_source_id].contiguous(),
        "unique_hand_normals": sample["hand_normals"].to(torch.float32).cpu()[unique_source_id].contiguous(),
        "unique_hand_flow": hand_flow.cpu()[unique_source_id].contiguous(),
        "edge_lookup": edge_lookup.contiguous(),
        "edge_source_id": source_id.to(torch.int16).contiguous(),
        "edge_valid_mask": valid.contiguous(),
    })
    return {
        "schema_name": CACHE_SCHEMA,
        "work_version": WORK_VERSION,
        "source": str(sample["source"]),
        "sequence_id": str(sample["sequence_id"]),
        "tensors": tensors,
        "source_hand_points": int(hand_points.shape[0]),
        "unique_hand_points": int(unique_source_id.numel()),
        "source_tensor_bytes": sum(
            tensor_nbytes(value) for value in sample.values() if torch.is_tensor(value)),
    }


def unpack_compact_endpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schema_name") != CACHE_SCHEMA or record.get("work_version") != WORK_VERSION:
        raise ValueError("unsupported compact endpoint record")
    tensors = dict(record["tensors"])
    lookup = tensors.pop("edge_lookup").long()
    points = tensors.pop("unique_hand_points")
    normals = tensors.pop("unique_hand_normals")
    flow = tensors.pop("unique_hand_flow")
    if lookup.ndim != 2 or lookup.shape[1] != 32 or lookup.min() < 0 or lookup.max() >= len(points):
        raise ValueError("compact endpoint lookup is outside the unique hand table")
    tensors.update({
        "edge_hand_points": points[lookup].contiguous(),
        "edge_hand_normals": normals[lookup].contiguous(),
        "edge_hand_flow": flow[lookup].contiguous(),
        "source": str(record["source"]),
        "sequence_id": str(record["sequence_id"]),
    })
    if not all(key in tensors for key in COMPACT_EDGE_KEYS):
        raise ValueError("compact endpoint record is incomplete")
    return tensors


def _serialize(record: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(record), buffer)
    return buffer.getvalue()


def _deserialize(payload: bytes) -> dict[str, Any]:
    buffer = io.BytesIO(payload)
    try:
        return torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")


@lru_cache(maxsize=8)
def _validated_manifest(root_value: str) -> dict[str, Any]:
    root = Path(root_value)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("schema_name") != CACHE_SCHEMA
            or manifest.get("work_version") != WORK_VERSION
            or int(manifest.get("validation", {}).get("bad_count", 1)) != 0):
        raise ValueError("compact endpoint cache manifest is not validated V1.13")
    index_path = root / "index.bin"
    if (index_path.stat().st_size != int(manifest["index_bytes"])
            or sha256_file(index_path) != manifest["index_sha256"]):
        raise ValueError("compact endpoint index identity mismatch")
    for relative, identity in manifest.get("shards", {}).items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != int(identity["bytes"]):
            raise ValueError(f"compact endpoint shard size mismatch: {relative}")
    return manifest


class CompactEndpointShardWriter:
    """Write immutable records to large shards plus a fixed-width binary index."""

    def __init__(self, output: str | Path, *, run_id: str, git_commit: str,
                 source_sha256: Mapping[str, str], shard_target_bytes: int = 1 << 30) -> None:
        self.output = Path(output).resolve()
        self.partial = self.output.with_name(self.output.name + ".partial")
        if self.output.exists() or self.partial.exists():
            raise FileExistsError(self.output if self.output.exists() else self.partial)
        if shard_target_bytes < 1 << 20:
            raise ValueError("compact endpoint shards must be at least 1 MiB")
        self.partial.mkdir(parents=True)
        (self.partial / "shards").mkdir()
        self.index_handle = (self.partial / "index.bin").open("wb")
        self.run_id, self.git_commit = str(run_id), str(git_commit)
        self.source_sha256 = dict(source_sha256)
        self.shard_target_bytes = int(shard_target_bytes)
        self.shard_id, self.shard_handle, self.shard_size = -1, None, 0
        self.count, self.group_counts = 0, {}
        self.view_ranges: dict[str, list[list[int]]] = {}
        self.source_bytes = self.record_bytes = self.unique_points = self.source_points = 0
        self.shard_paths: list[Path] = []
        self._write_run_manifest("STARTED")

    def _write_run_manifest(self, status: str, **extra: Any) -> None:
        payload = {
            "task": "ObjectInteractionCmv2", "work_version": WORK_VERSION,
            "run_id": self.run_id, "run_status": status,
            "operation": "compact_endpoint_cache_build", "git_commit": self.git_commit,
            "output": str(self.output), "partial_output": str(self.partial),
            "updated_at": utc_now(), "records_written": self.count,
            "serialized_bytes": self.record_bytes, "conclusion": "N/A", **extra,
        }
        temporary = self.partial / "run_manifest.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(self.partial / "run_manifest.json")

    def _open_shard(self) -> None:
        if self.shard_handle is not None:
            self.shard_handle.close()
        self.shard_id += 1
        path = self.partial / "shards" / f"shard-{self.shard_id:05d}.bin"
        self.shard_paths.append(path)
        self.shard_handle = path.open("wb")
        self.shard_size = 0

    def add(self, sample: Mapping[str, Any], *, group: str, split: str) -> None:
        if group not in GROUP_TO_ID or split not in SPLIT_TO_ID:
            raise ValueError("invalid compact endpoint group or split")
        record = pack_compact_endpoint(sample)
        payload = _serialize(record)
        if self.shard_handle is None or (self.shard_size and
                                         self.shard_size + len(payload) > self.shard_target_bytes):
            self._open_shard()
        offset = self.shard_size
        self.shard_handle.write(payload)
        self.shard_size += len(payload)
        stride = int(sample["stride"])
        if not 0 < stride < 256 or len(payload) >= 2**32:
            raise ValueError("compact endpoint stride or record size is outside index range")
        row = np.array((GROUP_TO_ID[group], SPLIT_TO_ID[split], stride, self.shard_id,
                        offset, len(payload)), dtype=INDEX_DTYPE)
        self.index_handle.write(row.tobytes())
        key = f"{group}/{split}/stride{stride}"
        self.group_counts[key] = self.group_counts.get(key, 0) + 1
        ranges = self.view_ranges.setdefault(key, [])
        if ranges and ranges[-1][0] + ranges[-1][1] == self.count:
            ranges[-1][1] += 1
        else:
            ranges.append([self.count, 1])
        self.count += 1
        self.source_bytes += int(record["source_tensor_bytes"])
        self.record_bytes += len(payload)
        self.unique_points += int(record["unique_hand_points"])
        self.source_points += int(record["source_hand_points"])
        if self.count == 1 or self.count % 100 == 0:
            self._write_run_manifest("RUNNING")
            print(json.dumps({
                "timestamp": utc_now(), "run_id": self.run_id, "run_status": "RUNNING",
                "records_written": self.count, "serialized_bytes": self.record_bytes,
            }), flush=True)

    def finalize(self, *, validation_bad_count: int = 0) -> Path:
        if not self.count:
            raise ValueError("cannot finalize an empty compact endpoint cache")
        self.index_handle.close()
        if self.shard_handle is not None:
            self.shard_handle.close()
        shards = {
            str(path.relative_to(self.partial)): {
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
            for path in self.shard_paths
        }
        index_path = self.partial / "index.bin"
        manifest = {
            "schema_name": CACHE_SCHEMA,
            "work_version": WORK_VERSION,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "created_at": utc_now(),
            "coordinates": "V1.12 current object/root reference; metres; seconds; radians",
            "dtype": "float32 tensors; int16 edge lookup/source IDs; bool edge mask",
            "record_count": self.count,
            "index_dtype": INDEX_DTYPE.descr,
            "index_bytes": index_path.stat().st_size,
            "index_sha256": sha256_file(index_path),
            "group_counts": self.group_counts,
            "view_ranges": self.view_ranges,
            "source_sha256": self.source_sha256,
            "shards": shards,
            "statistics": {
                "source_sample_tensor_bytes": self.source_bytes,
                "serialized_record_bytes": self.record_bytes,
                "serialized_to_source_hand_ratio": self.record_bytes / max(self.source_bytes, 1),
                "unique_to_source_hand_point_ratio": self.unique_points / max(self.source_points, 1),
            },
            "validation": {"bad_count": int(validation_bad_count)},
        }
        (self.partial / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        self._write_run_manifest("COMPLETED", finished_at=utc_now(), manifest="manifest.json")
        self.partial.rename(self.output)
        return self.output

    def close_incomplete(self) -> None:
        if not self.index_handle.closed:
            self.index_handle.close()
        if self.shard_handle is not None and not self.shard_handle.closed:
            self.shard_handle.close()

    def mark_incomplete(self, status: str, **extra: Any) -> None:
        if status not in ("FAILED", "STOPPED"):
            raise ValueError("incomplete compact cache status must be FAILED or STOPPED")
        self.close_incomplete()
        self._write_run_manifest(status, finished_at=utc_now(), **extra)


class CompactEndpointDataset(Dataset):
    """Read one group/split/stride view without opening source geometry arrays."""

    def __init__(self, root: str | Path, group: str, split: str, *,
                 fixed_stride: int | None = None, max_open_shards: int = 8) -> None:
        self.root = Path(root).resolve()
        self.manifest = _validated_manifest(str(self.root))
        if group not in GROUP_TO_ID or split not in SPLIT_TO_ID:
            raise ValueError("invalid compact endpoint dataset selection")
        index_path = self.root / "index.bin"
        index = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
        prefix = f"{group}/{split}/stride"
        ranges = []
        for key, values in self.manifest.get("view_ranges", {}).items():
            if not key.startswith(prefix):
                continue
            if fixed_stride is not None and key != f"{prefix}{int(fixed_stride)}":
                continue
            ranges.extend(values)
        self.rows = np.concatenate([
            np.arange(int(start), int(start) + int(count), dtype=np.int64)
            for start, count in sorted(ranges)
        ]) if ranges else np.empty((0,), dtype=np.int64)
        if not len(self.rows):
            raise ValueError("compact endpoint dataset selection is empty")
        self.index = index
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self._handles: OrderedDict[int, Any] = OrderedDict()
        self.dropped_timeline_transitions = 0

    def __len__(self) -> int:
        return len(self.rows)

    def _handle(self, shard: int):
        handle = self._handles.pop(shard, None)
        if handle is None:
            path = self.root / "shards" / f"shard-{shard:05d}.bin"
            handle = path.open("rb")
        self._handles[shard] = handle
        while len(self._handles) > self.max_open_shards:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = self.index[int(self.rows[item])]
        handle = self._handle(int(row["shard"]))
        handle.seek(int(row["offset"]))
        payload = handle.read(int(row["length"]))
        if len(payload) != int(row["length"]):
            raise IOError("short compact endpoint shard read")
        sample = unpack_compact_endpoint(_deserialize(payload))
        if int(sample["stride"]) != int(row["stride"]):
            raise ValueError("compact endpoint index/record stride mismatch")
        return sample

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
