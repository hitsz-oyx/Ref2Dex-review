"""V18 stable-grasp event detection 与 Interaction Y 数据集。"""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.task.InteractionDynamics.dataset_v17 import split_sequences
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.residual_interaction_regression import residual_target
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future
from src.task.InteractionDynamics.uni3d import gather_points, patchify


@dataclass(frozen=True)
class GraspEvent:
    path: Path
    grasp_frame: int
    contact_count: int
    stable_u_cm: float


def all_hand_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(list(root.rglob("left.npz")) + list(root.rglob("right.npz")))


def _rigid_world_to_reference(object_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """由有 correspondence 的刚体 object points 求每帧 world→首帧坐标。"""
    points = object_world[:, ::16]
    reference = points[0]
    source_mean = points.mean(1, keepdim=True)
    target_mean = reference.mean(0, keepdim=True)
    source = points - source_mean
    target = reference[None] - target_mean
    covariance = source.transpose(1, 2) @ target.expand_as(source)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    negative = torch.linalg.det(rotation) < 0
    if negative.any():
        u[negative, :, -1] *= -1
        rotation = u @ vh
    translation = target_mean - source_mean @ rotation
    return rotation, translation


def _transform(points: torch.Tensor, rotation: torch.Tensor,
               translation: torch.Tensor) -> torch.Tensor:
    return points @ rotation + translation


def sequence_interaction(path: str | Path, anchors: int = 128, tau_m: float = .015,
                         device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    """一次构造完整 sequence 的 object-centric Y 与 detector 统计。"""
    path = Path(path); device = torch.device(device)
    with np.load(path.parent / "shared.npz", allow_pickle=False) as shared, \
         np.load(path, allow_pickle=False) as hand:
        object_world = torch.from_numpy(np.asarray(shared["obj_points_world"], np.float32)).to(device)
        normal_world = torch.from_numpy(np.asarray(shared["obj_normals_world"], np.float32)).to(device)
        hand_world = torch.from_numpy(np.asarray(hand["hand_points_world"], np.float32)).to(device)
    rotation, translation = _rigid_world_to_reference(object_world)
    object_reference = object_world[0]
    hand_object = _transform(hand_world, rotation, translation)
    object_object = _transform(object_world, rotation, translation)
    normals_object = normal_world @ rotation
    anchor_points, _, knn = patchify(object_reference[None], anchors, 32)
    anchor_points = anchor_points[0]
    y = build_interaction_y(hand_object, anchor_points, tau_m)
    contact_count = (y["relative_distance"] < .02).sum(1)
    u_rms_cm = 100 * y["relative_motion"].square().mean((1, 2)).sqrt()
    return {**y, "contact_count": contact_count, "u_rms_cm": u_rms_cm,
            "anchors": anchor_points, "object_object": object_object,
            "normals_object": normals_object, "knn": knn[0]}


def detect_stable_grasp(path: str | Path, contact_anchors: int = 4,
                        distance_m: float = .02, max_u_cm: float = .3,
                        stable_frames: int = 6, future_frames: int = 30,
                        motion_cm: float = 1., rotation_deg: float = 5.,
                        device: str | torch.device = "cuda") -> GraspEvent | None:
    interaction = sequence_interaction(path, device=device)
    # sequence_interaction 固定使用与 operational definition 相同的 2 cm contact。
    if distance_m != .02:
        contact = (interaction["relative_distance"] < distance_m).sum(1)
    else:
        contact = interaction["contact_count"]
    u = interaction["u_rms_cm"]
    stable = (contact[:-1] >= contact_anchors) & (u < max_u_cm)
    with np.load(Path(path).parent / "shared.npz", allow_pickle=False) as shared:
        objects = torch.from_numpy(np.asarray(shared["obj_points_world"], np.float32)).to(u.device)[:, ::16]
    limit = min(len(stable) - stable_frames, len(objects) - 1)
    for frame in range(max(0, limit)):
        if not bool(stable[frame:frame + stable_frames].all()):
            continue
        end = min(frame + future_frames, len(objects) - 1)
        translation = 100 * (objects[end].mean(0) - objects[frame].mean(0)).norm()
        source = objects[frame] - objects[frame].mean(0)
        target = objects[end] - objects[end].mean(0)
        u_svd, _, vh = torch.linalg.svd(source.T @ target)
        relative = u_svd @ vh
        trace = relative.trace().clamp(-1, 3)
        angle = torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1)))
        if float(translation) > motion_cm or float(angle) > rotation_deg:
            return GraspEvent(Path(path), frame, int(contact[frame]), float(u[frame]))
    return None


def load_events(path: str | Path) -> list[GraspEvent]:
    rows = json.loads(Path(path).read_text())
    return [GraspEvent(Path(row["path"]), int(row["grasp_frame"]),
                       int(row["contact_count"]), float(row["stable_u_cm"])) for row in rows]


class GraspV18Dataset(Dataset):
    def __init__(self, events: list[GraspEvent], horizon: int = 8,
                 maintenance=(0, 2, 4), tau_m: float = .015,
                 device: str | torch.device = "cuda") -> None:
        self.events, self.horizon, self.tau_m = events, horizon, tau_m
        self.device = device
        self.samples: list[tuple[int, int]] = []
        for event_index, event in enumerate(events):
            with np.load(event.path.parent / "shared.npz", allow_pickle=False) as shared:
                count = len(shared["raw_frame_id"])
            frames = list(range(event.grasp_frame - horizon, event.grasp_frame))
            frames += [event.grasp_frame + offset for offset in maintenance]
            self.samples += [(event_index, frame) for frame in sorted(set(frames))
                             if 0 <= frame and frame + horizon < count]
        self._cache_index = -1; self._cache = None

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        event_index, frame = self.samples[index]
        event = self.events[event_index]
        if event_index != self._cache_index:
            self._cache = sequence_interaction(event.path, device=self.device)
            self._cache_index = event_index
        data = self._cache; assert data is not None
        y = {key: 100 * value[frame:frame + self.horizon + (0 if key == "relative_motion" else 1)].cpu()
             for key, value in data.items() if key in
             {"relative_geometry", "relative_distance", "relative_motion"}}
        state, future = pack_state_future(y, self.horizon)
        residual = residual_target(state, future, self.horizon)
        anchors = data["anchors"].cpu()
        knn = data["knn"].cpu()
        points = data["object_object"][frame].cpu()
        normals = data["normals_object"][frame].cpu()
        patches = torch.cat([100 * (gather_points(points[None], knn[None]) - anchors[None, :, None]),
                             gather_points(normals[None], knn[None])], -1)[0]
        return {"state": state, "future": future, "residual": residual,
                "anchors_cm": 100 * anchors, "object_patches": patches,
                "frame": torch.tensor(frame), "grasp_frame": torch.tensor(event.grasp_frame),
                "frames_to_grasp": torch.tensor(event.grasp_frame - frame)}


class CachedGraspDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.paths = sorted((Path(root) / split).glob("*.pt"))
        self.event_indices = [int(path.stem.rsplit("_", 1)[-1]) for path in self.paths]
        self.shards = [torch.load(path, map_location="cpu") for path in self.paths]
        if not self.shards: raise ValueError(f"No V18 cache for {split}")
        self.offsets = [0]
        for shard in self.shards: self.offsets.append(self.offsets[-1] + len(shard["state"]))
    def __len__(self): return self.offsets[-1]
    def __getitem__(self, index):
        shard = bisect.bisect_right(self.offsets, index) - 1
        local = index - self.offsets[shard]
        item = {key: value[local] for key, value in self.shards[shard].items()}
        item["event_index"] = torch.tensor(self.event_indices[shard])
        return item
