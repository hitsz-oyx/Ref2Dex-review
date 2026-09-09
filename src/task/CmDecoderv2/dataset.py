"""Windowed Dexplore RL dataset for CmDecoderv2."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, make_default_train_sampler

from .kinematics import InspireKinematics, extract_finger_q, relative_pose, rotation_matrix_to_rotvec, rotvec_to_matrix


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _points_world_to_frame(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    rotation, translation = pose_world[:3, :3], pose_world[:3, 3]
    return ((np.asarray(points, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normals_world_to_frame(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    value = np.asarray(normals, dtype=np.float32) @ pose_world[:3, :3]
    return (value / np.clip(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:3, :2].T.reshape(-1), dtype=np.float32)


class _Sequence:
    def __init__(self, entry: dict[str, Any]) -> None:
        self.entry = dict(entry)
        self.id = str(entry["id"])
        self.geometry_root = Path(entry["geometry_root"])
        self.frame_count = int(entry["frame_count"])
        self.object_points = np.load(self.geometry_root / "obj_points_pool_world.npy", mmap_mode="r")
        self.object_normals = np.load(self.geometry_root / "obj_normals_pool_world.npy", mmap_mode="r")
        self.object_pose = np.load(self.geometry_root / "obj_pose_world.npy", mmap_mode="r")
        self.hand_points = np.load(self.geometry_root / "hand_points_world.npy", mmap_mode="r")
        self.hand_normals = np.load(self.geometry_root / "hand_normals_world.npy", mmap_mode="r")
        self.active = np.load(self.geometry_root / "obj_candidate_mask_5cm.npy", mmap_mode="r")
        self.source_frame = np.load(self.geometry_root / "source_frame_id.npy", mmap_mode="r")
        self.q_native = np.load(entry["q_native"], mmap_mode="r")
        self.wrist_pose = np.load(entry["wrist_pose_world"], mmap_mode="r")
        expected = (self.frame_count,)
        if any(len(array) != self.frame_count for array in (self.object_points, self.object_normals, self.object_pose, self.hand_points, self.hand_normals, self.q_native, self.wrist_pose)):
            raise ValueError(f"Frame mismatch in decoder view {self.id}, expected {expected[0]}")


class CmDecoderV2Dataset(Dataset):
    """K Cm inputs need K+1 geometry frames and predict target states 1..K."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        *,
        urdf_path: str | Path,
        window_size: int = 4,
        num_obj_points: int = 1024,
        num_hand_points: int = 1538,
        seed: int = 42,
        perturb: bool = False,
        active_only: bool = False,
        translation_noise_std_m: float = 0.005,
        translation_noise_clip_m: float = 0.015,
        rotation_noise_std_deg: float = 5.0,
        rotation_noise_clip_deg: float = 15.0,
        finger_q_noise_std_rad: float = 0.05,
        finger_q_noise_clip_rad: float = 0.15,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = int(window_size)
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.seed = int(seed)
        self.perturb = bool(perturb)
        self.active_only = bool(active_only)
        self.translation_noise_std_m = float(translation_noise_std_m)
        self.translation_noise_clip_m = float(translation_noise_clip_m)
        self.rotation_noise_std_rad = math.radians(float(rotation_noise_std_deg))
        self.rotation_noise_clip_rad = math.radians(float(rotation_noise_clip_deg))
        self.finger_q_noise_std_rad = float(finger_q_noise_std_rad)
        self.finger_q_noise_clip_rad = float(finger_q_noise_clip_rad)
        self.kinematics = InspireKinematics(urdf_path)
        self.sequences = [_Sequence(entry) for entry in entries]
        self.rows: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(self.sequences):
            for start in range(max(0, sequence.frame_count - self.window_size)):
                if not self.active_only or bool(np.asarray(sequence.active[start:start + self.window_size]).any()):
                    self.rows.append((sequence_index, start))
        if not self.rows:
            raise ValueError("No complete CmDecoderv2 windows")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _perturb_state(self, finger_q: np.ndarray, wrist: np.ndarray, object_pose: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if not self.perturb:
            return finger_q.copy(), wrist.copy()
        q_noise = np.clip(rng.normal(0.0, self.finger_q_noise_std_rad, size=6), -self.finger_q_noise_clip_rad, self.finger_q_noise_clip_rad)
        finger_q = self.kinematics.clamp_finger_q(finger_q + q_noise)
        translation_object = np.clip(rng.normal(0.0, self.translation_noise_std_m, size=3), -self.translation_noise_clip_m, self.translation_noise_clip_m)
        wrist = wrist.copy()
        wrist[:3, 3] += object_pose[:3, :3] @ translation_object
        rotation_noise = np.clip(
            rng.normal(0.0, self.rotation_noise_std_rad, size=3),
            -self.rotation_noise_clip_rad,
            self.rotation_noise_clip_rad,
        )
        wrist[:3, :3] = wrist[:3, :3] @ rotvec_to_matrix(rotation_noise)
        return finger_q, wrist

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, start = self.rows[index]
        sequence = self.sequences[sequence_index]
        seed = _stable_seed(self.seed, sequence.id, int(sequence.source_frame[start]), self._epoch)
        rng = np.random.default_rng(seed)
        k = self.window_size
        object_points, object_normals = [], []
        hand_points, hand_normals, hand_flow = [], [], []
        for offset in range(k):
            frame = start + offset
            pose = np.asarray(sequence.object_pose[frame], dtype=np.float32)
            selected = np.random.default_rng(seed ^ _stable_seed("object", offset)).choice(sequence.object_points.shape[1], size=self.num_obj_points, replace=False)
            object_points.append(_points_world_to_frame(sequence.object_points[frame, selected], pose))
            object_normals.append(_normals_world_to_frame(sequence.object_normals[frame, selected], pose))
            current_hand = _points_world_to_frame(sequence.hand_points[frame], pose)
            future_hand = _points_world_to_frame(sequence.hand_points[frame + 1], pose)
            current_normals = _normals_world_to_frame(sequence.hand_normals[frame], pose)
            if len(current_hand) != self.num_hand_points:
                raise ValueError(f"Expected {self.num_hand_points} hand points in {sequence.id}")
            hand_points.append(current_hand)
            hand_normals.append(current_normals)
            hand_flow.append(future_hand - current_hand)
        current_native = np.asarray(sequence.q_native[start], dtype=np.float64)
        current_finger = extract_finger_q(current_native)
        current_wrist = np.asarray(sequence.wrist_pose[start], dtype=np.float64)
        object_pose_start = np.asarray(sequence.object_pose[start], dtype=np.float64)
        active_mask = np.asarray(sequence.active[start:start + k], dtype=bool).copy()
        current_finger, current_wrist = self._perturb_state(current_finger, current_wrist, object_pose_start, rng)
        target_q_delta, target_translation, target_rotation, target_hand_points = [], [], [], []
        for horizon in range(1, k + 1):
            future_finger = extract_finger_q(sequence.q_native[start + horizon])
            target_q_delta.append(future_finger - current_finger)
            relative = relative_pose(current_wrist, sequence.wrist_pose[start + horizon])
            target_translation.append(relative[:3, 3])
            target_rotation.append(relative[:3, :3])
            target_hand_points.append(_points_world_to_frame(sequence.hand_points[start + horizon], object_pose_start))
        current_object = np.linalg.inv(object_pose_start) @ current_wrist
        return {
            "obj_points": torch.from_numpy(np.stack(object_points).astype(np.float32)),
            "obj_normals": torch.from_numpy(np.stack(object_normals).astype(np.float32)),
            "obj_valid_mask": torch.ones((k, self.num_obj_points), dtype=torch.bool),
            "hand_points": torch.from_numpy(np.stack(hand_points).astype(np.float32)),
            "hand_normals": torch.from_numpy(np.stack(hand_normals).astype(np.float32)),
            "hand_flow": torch.from_numpy(np.stack(hand_flow).astype(np.float32)),
            "hand_valid_mask": torch.ones((k, self.num_hand_points), dtype=torch.bool),
            "current_finger_q": torch.from_numpy(current_finger.astype(np.float32)),
            "current_wrist_pose_world": torch.from_numpy(current_wrist.astype(np.float32)),
            "object_pose_world": torch.from_numpy(object_pose_start.astype(np.float32)),
            "current_wrist_translation_object": torch.from_numpy(current_object[:3, 3].astype(np.float32)),
            "current_wrist_rotation_6d_object": torch.from_numpy(_rotation_6d(current_object[:3, :3])),
            "current_link_features": torch.from_numpy(self.kinematics.query_features(current_finger, current_wrist, object_pose_start)),
            "target_q_delta": torch.from_numpy(np.stack(target_q_delta).astype(np.float32)),
            "target_wrist_translation": torch.from_numpy(np.stack(target_translation).astype(np.float32)),
            "target_wrist_rotation": torch.from_numpy(np.stack(target_rotation).astype(np.float32)),
            "target_hand_points_object": torch.from_numpy(np.stack(target_hand_points).astype(np.float32)),
            # One mask entry per source frame/Cm horizon.  This is the exact
            # full-object-pool 5 cm candidate mask, before OICM point sampling.
            "active_mask": torch.from_numpy(active_mask),
            "sequence_id": sequence.id,
            "start_frame": torch.tensor(start, dtype=torch.int64),
            "source_frame_id": torch.tensor(int(sequence.source_frame[start]), dtype=torch.int64),
        }


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _entries(index: dict[str, Any], split: str) -> list[dict[str, Any]]:
    values = index.get("sequences", {}).get(split, [])
    expected = "inspire_rl" if split in {"train", "val"} else "mano"
    if not values or any(item.get("variant") != expected for item in values):
        raise ValueError(f"CmDecoderv2 {split} must contain only {expected}")
    return values


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any | None = None):
    view_root = _resolve(str(data_cfg.view_root))
    index_path = view_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_name") != "ref2dex_cm_decoder_v2_dexplore_view_v1":
        raise ValueError(f"Unsupported CmDecoderv2 view: {index.get('schema_name')!r}")
    if index.get("split_contract") != {"train": "inspire_rl", "val": "inspire_rl", "test": "mano_qualitative_only"}:
        raise ValueError("CmDecoderv2 view split contract changed")
    common = dict(
        urdf_path=_resolve(str(data_cfg.urdf_path)), window_size=int(meta_cfg.window_size),
        num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points), seed=int(seed),
        active_only=bool(getattr(data_cfg, "active_only", False)),
        translation_noise_std_m=float(data_cfg.translation_noise_std_m), translation_noise_clip_m=float(data_cfg.translation_noise_clip_m),
        rotation_noise_std_deg=float(data_cfg.rotation_noise_std_deg), rotation_noise_clip_deg=float(data_cfg.rotation_noise_clip_deg),
        finger_q_noise_std_rad=float(data_cfg.finger_q_noise_std_rad), finger_q_noise_clip_rad=float(data_cfg.finger_q_noise_clip_rad),
    )
    train_dataset = CmDecoderV2Dataset(_entries(index, "train"), perturb=bool(data_cfg.perturb_train), **common)
    val_dataset = CmDecoderV2Dataset(_entries(index, "val"), perturb=False, **common)
    train_sampler = make_default_train_sampler(train_dataset, shuffle=True, seed=seed, distributed=distributed, drop_last=False)
    val_sampler = make_default_eval_sampler(val_dataset, distributed=distributed)
    train_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
    train_kwargs.update(batch_size=int(data_cfg.batch_size), shuffle=train_sampler is None, sampler=train_sampler)
    val_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
    val_kwargs.update(batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size), shuffle=False, sampler=val_sampler)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    metadata = {
        "schema_name": index["schema_name"],
        "modification_version": index["modification_version"],
        "coordinate_frame": "object_pose_t",
        "window_size": int(meta_cfg.window_size),
        "effective_fps": 30.0,
        "split_contract": index["split_contract"],
        "view_index": str(index_path),
        "view_cache_manifest": str(view_root / "manifest.json"),
        "view_run_manifest": str(view_root / "run_manifest.json"),
        "source_index_sha256": index["source_index_sha256"],
        "urdf_sha256": index["urdf_sha256"],
        "dataset_split": {"train_sequences": len(index["sequences"]["train"]), "val_sequences": len(index["sequences"]["val"]), "test_sequences": len(index["sequences"]["test"])},
        "training_windows": len(train_dataset),
        "validation_windows": len(val_dataset),
    }
    return train_loader, val_loader, None, metadata, {"val/": val_loader}, {}
