"""Minimal paired dataset for the V1.1.16 FieldRealizer contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import _points_world_to_frame, _rotation_6d
from .kinematics import InspireKinematics, extract_finger_q, relative_pose


class FieldRealizerDataset(Dataset):
    """Pair parent-only F7 goals with actual Inspire supervision.

    The F7 cache is indexed by the same raw frame ids as the paired view. No
    future Inspire pose is read into the source fields; actual q/wrist values
    appear only in the supervision and current-state fields.
    """

    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        field_root: str | Path,
        urdf_path: str | Path,
        window_size: int = 4,
        active_only: bool = True,
    ) -> None:
        self.window_size = int(window_size)
        self.kinematics = InspireKinematics(urdf_path)
        self.sequences: list[dict[str, Any]] = []
        self.rows: list[tuple[int, int]] = []
        field_root = Path(field_root)
        for sequence_index, entry in enumerate(entries):
            cache = field_root / entry["split"] / str(entry["id"]).replace("/", "_")
            manifest = json.loads((cache / "manifest.json").read_text())
            if manifest["field_definition"] != "F7=[r,d,v]; no p/c/contact/E":
                raise ValueError(f"Not a V1.1.16 F7 cache: {cache}")
            item = {
                "id": str(entry["id"]),
                "frame_count": int(entry["frame_count"]),
                "geometry_root": Path(entry["geometry_root"]),
                "q_native": np.load(entry["q_native"], mmap_mode="r"),
                "wrist_pose": np.load(entry["wrist_pose_world"], mmap_mode="r"),
                "object_pose": np.load(Path(entry["geometry_root"]) / "obj_pose_world.npy", mmap_mode="r"),
                "target_hand_points": np.load(
                    Path(entry["geometry_root"]) / "knn_hand_points_world.npy",
                    mmap_mode="r",
                ),
                "active": np.load(Path(entry["geometry_root"]) / "obj_candidate_mask_2cm.npy", mmap_mode="r"),
                "field": np.load(cache / "field_f7.npy", mmap_mode="r"),
                "anchors": np.load(cache / "anchors_object.npy", mmap_mode="r"),
                "anchor_normals": np.load(cache / "anchor_normals_object.npy", mmap_mode="r"),
                "field_raw": np.load(cache / "raw_frame_id.npy", mmap_mode="r"),
            }
            if item["field"].shape != (item["frame_count"] - 1, 128, 7):
                raise ValueError(f"F7 frame/shape mismatch for {item['id']}: {item['field'].shape}")
            self.sequences.append(item)
            for start in range(item["frame_count"] - self.window_size):
                active = np.asarray(item["active"][start:start + self.window_size])
                if not active_only or bool(active.any()):
                    self.rows.append((sequence_index, start))
        if not self.rows:
            raise ValueError("No FieldRealizer windows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence_index, start = self.rows[index]
        item = self.sequences[sequence_index]
        k = self.window_size
        q_native = np.asarray(item["q_native"][start], dtype=np.float64)
        current_finger = extract_finger_q(q_native).astype(np.float32)
        current_wrist = np.asarray(item["wrist_pose"][start], dtype=np.float32).copy()
        object_pose = np.asarray(item["object_pose"][start], dtype=np.float32).copy()
        current_object = np.linalg.inv(object_pose) @ current_wrist
        target_q, target_t, target_r = [], [], []
        target_hand_points = []
        for horizon in range(1, k + 1):
            target_q.append(extract_finger_q(np.asarray(item["q_native"][start + horizon], dtype=np.float64)) - current_finger)
            relative = relative_pose(current_wrist, np.asarray(item["wrist_pose"][start + horizon], dtype=np.float64))
            target_t.append(relative[:3, 3])
            target_r.append(relative[:3, :3])
            target_hand_points.append(
                _points_world_to_frame(item["target_hand_points"][start + horizon], object_pose)
            )
        field = np.asarray(item["field"][start:start + k], dtype=np.float32).copy()
        active = np.asarray(item["active"][start:start + k]).any(axis=1) if np.asarray(item["active"]).ndim == 2 else np.asarray(item["active"][start:start + k])
        anchors = np.asarray(item["anchors"], dtype=np.float32)
        normals = np.asarray(item["anchor_normals"], dtype=np.float32)
        return {
            "f7": torch.from_numpy(field),
            "anchor_pos": torch.from_numpy(np.broadcast_to(anchors, (k, *anchors.shape)).copy()),
            "anchor_normal": torch.from_numpy(np.broadcast_to(normals, (k, *normals.shape)).copy()),
            "current_finger_q": torch.from_numpy(current_finger),
            "current_wrist_pose_world": torch.from_numpy(current_wrist),
            "object_pose_world": torch.from_numpy(object_pose),
            "current_wrist_translation_object": torch.from_numpy(current_object[:3, 3].astype(np.float32)),
            "current_wrist_rotation_6d_object": torch.from_numpy(_rotation_6d(current_object[:3, :3])),
            "current_link_features": torch.from_numpy(self.kinematics.query_features_native(q_native, object_pose)),
            "target_q_delta": torch.from_numpy(np.stack(target_q).astype(np.float32)),
            "target_wrist_translation": torch.from_numpy(np.stack(target_t).astype(np.float32)),
            "target_wrist_rotation": torch.from_numpy(np.stack(target_r).astype(np.float32)),
            "target_hand_points_object": torch.from_numpy(np.stack(target_hand_points).astype(np.float32)),
            "current_hand_points_object": torch.from_numpy(
                _points_world_to_frame(item["target_hand_points"][start], object_pose)
            ),
            "active_mask": torch.from_numpy(active.astype(bool)),
        }
