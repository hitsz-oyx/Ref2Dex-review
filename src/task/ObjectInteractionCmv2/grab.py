"""GRAB-only, MANO-only stride-1 reader for the finalized bilateral cache."""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ARRAYS = ("frame_time", "source_frame_id", "obj_points_pool_world",
          "obj_normals_pool_world", "obj_pose_world", "knn_hand_points_world",
          "knn_hand_normals_world")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=6)
def _arrays(sequence_path: str):
    root = Path(sequence_path) / "geometry"
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in ARRAYS}


class GrabManoTransitions(Dataset):
    def __init__(self, index: str | Path, manifest: str | Path, split: str,
                 max_sequences: int | None = None, object_points: int = 1024):
        self.index_path, self.manifest_path = Path(index).resolve(), Path(manifest).resolve()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.split = split
        run = json.loads(self.manifest_path.read_text())
        if run.get("run_status") != "COMPLETED" or run.get("result") != "SUPPORTED":
            raise ValueError("Input cache must have a completed, supported run manifest")
        catalog = json.loads(self.index_path.read_text())
        if catalog.get("knn_hand_points_per_stream", {}).get("mano") != 4096:
            raise ValueError("Expected bilateral MANO2048 KNN stream")
        if catalog.get("model_object_points") != object_points or catalog.get("object_pool_points") != 4096:
            raise ValueError("Expected fixed 1024/4096 object point contract")
        root = self.index_path.parent
        entries = [entry for entry in catalog["sequences"][split] if entry.get("dataset") == "grab" and entry.get("source") == "mano"]
        if max_sequences is not None:
            entries = entries[:max_sequences]
        self.sequences = []
        self.starts = [0]
        self.dropped_pairs = 0
        surface_hashes = set()
        for entry in entries:
            sequence_path = root / "sequences" / split / "mano" / entry["id"]
            metadata = json.loads((sequence_path / "geometry" / "manifest.json").read_text())
            if (metadata.get("schema_name"), metadata.get("dataset"), metadata.get("source"),
                metadata.get("split"), metadata.get("object_representation"),
                metadata.get("coordinate_frame"), metadata.get("hand_side")) != (
                    "ref2dex_object_interaction_cm_bilateral_mano_v1_4", "grab", "mano",
                    split, "rigid_se3", "object_pose_t", "bilateral_merged_left_then_right"):
                raise ValueError(f"Incompatible GRAB cache: {entry['id']}")
            if (metadata.get("knn_points_per_side") != 2048 or metadata.get("knn_hand_points") != 4096
                or metadata.get("object_pool_points") != 4096 or metadata.get("effective_fps") != 30.0):
                raise ValueError(f"Not MANO2048: {entry['id']}")
            sampling = metadata.get("surface_sampling", {})
            if not sampling.get("cross_frame_fixed"):
                raise ValueError(f"Hand point correspondence not fixed: {entry['id']}")
            surface_hashes.add((sampling.get("left_sha256"), sampling.get("right_sha256")))
            if len(surface_hashes) != 1:
                raise ValueError("MANO surface sampling differs between sequences")
            arrays = _arrays(str(sequence_path))
            frames = int(entry["frame_count"])
            if any(len(arrays[name]) != frames for name in ARRAYS):
                raise ValueError(f"Frame count mismatch: {entry['id']}")
            source_step = np.diff(arrays["source_frame_id"])
            dt = np.diff(arrays["frame_time"].astype(np.float64))
            valid = np.flatnonzero((source_step == 4) & np.isclose(dt, 1 / 30, atol=1e-5, rtol=0))
            self.dropped_pairs += frames - 1 - len(valid)
            self.sequences.append((entry["id"], str(sequence_path), valid))
            self.starts.append(self.starts[-1] + len(valid))
        if not self.sequences or self.starts[-1] == 0:
            raise ValueError("No valid GRAB/MANO stride-1 transitions")
        self.point_indices = np.arange(0, 4096, 4, dtype=np.int64)

    def __len__(self):
        return self.starts[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sequence = bisect_right(self.starts, index) - 1
        sequence_id, path, valid = self.sequences[sequence]
        t = int(valid[index - self.starts[sequence]])
        arrays = _arrays(path)
        selected = self.point_indices
        pose = np.asarray(arrays["obj_pose_world"][t], dtype=np.float32)
        rotation, translation = pose[:3, :3], pose[:3, 3]
        def local(world):
            return (np.asarray(world, dtype=np.float32) - translation) @ rotation
        obj = local(arrays["obj_points_pool_world"][t, selected])
        obj_next = local(arrays["obj_points_pool_world"][t + 1, selected])
        normals = np.asarray(arrays["obj_normals_pool_world"][t, selected]) @ rotation
        hand = local(arrays["knn_hand_points_world"][t])
        hand_next = local(arrays["knn_hand_points_world"][t + 1])
        hand_normals = np.asarray(arrays["knn_hand_normals_world"][t]) @ rotation
        delta_time = float(np.float64(arrays["frame_time"][t + 1]) - np.float64(arrays["frame_time"][t]))
        tensors = {"obj_points": obj, "obj_normals": normals, "hand_points": hand,
                   "hand_normals": hand_normals, "hand_flow": hand_next - hand,
                   "obj_flow_gt": obj_next - obj}
        if not all(np.isfinite(value).all() for value in tensors.values()):
            raise ValueError(f"Nonfinite geometry: {sequence_id} frame {t}")
        return {**{name: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)) for name, value in tensors.items()},
                "delta_time_s": torch.tensor(delta_time, dtype=torch.float32),
                "hand_valid_mask": torch.ones(4096, dtype=torch.bool),
                "sequence_id": sequence_id, "frame_index": t}
