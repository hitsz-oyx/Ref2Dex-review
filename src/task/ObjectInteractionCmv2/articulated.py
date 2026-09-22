"""V1.5 articulated-object path, isolated from the V1.4 rigid runtime.

The first implementation accepts the explicitly confirmed mixed contract: GRAB is
one rigid link with no joint, while ARCTIC is two links with one revolute joint.
It deliberately reads no future object state in ``forward``; future poses and q
are converted to targets by :class:`ArticulatedTransitions` only.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .model import LocalInteractionEncoder, _axis_angle_matrix_stable, _mlp
from .multi_domain import InspireSequenceView, _is_se3, _normal_world_to_frame, _stable_seed, _world_to_frame


ARTICULATED_VERSION = "v1_5_articulated_fk"


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return a stable axis-angle vector for a proper 3x3 rotation."""
    trace = float(np.trace(rotation))
    cosine = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray((rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                       rotation[1, 0] - rotation[0, 1]), dtype=np.float32)
    sine = float(np.linalg.norm(skew) * 0.5)
    if angle < 1e-6:
        return (0.5 * skew).astype(np.float32)
    if sine < 1e-6:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal).astype(np.float32)
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
            axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
        return (angle * axis).astype(np.float32)
    return (angle * skew / (2.0 * sine)).astype(np.float32)


def _validate_joint_spec(value: Mapping[str, Any], num_links: int) -> dict[str, Any]:
    parent, child = int(value["parent"]), int(value["child"])
    if not (0 <= parent < num_links and 0 <= child < num_links and parent != child):
        raise ValueError("joint parent/child must be distinct valid link indices")
    if str(value.get("type", "revolute")) != "revolute":
        raise ValueError("V1.5 first stage supports revolute joints only")
    axis = np.asarray(value["axis_root"], dtype=np.float32)
    origin_value = value.get("origin_root", value.get("origin_root_m"))
    origin = np.asarray(origin_value, dtype=np.float32)
    if axis.shape != (3,) or origin.shape != (3,) or not np.isfinite(axis).all() or not np.isfinite(origin).all():
        raise ValueError("joint axis/origin must be finite 3-vectors")
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-4):
        raise ValueError("joint axis_root must be unit length")
    return {"parent": parent, "child": child, "axis_root": axis, "origin_root": origin}


def _source_kinematics(spec: Mapping[str, Any], domain: str) -> dict[str, Any]:
    source = spec.get("articulation")
    if not isinstance(source, Mapping):
        raise ValueError(f"{domain}: V1.5 source requires explicit articulation metadata")
    links = int(source.get("num_links", -1))
    joints = source.get("joints")
    if not isinstance(joints, list):
        raise ValueError(f"{domain}: articulation.joints must be a list")
    if domain == "grab":
        if links != 1 or joints:
            raise ValueError("GRAB V1.5 fallback must be L=1, J=0")
    elif domain == "arctic":
        if links != 2 or len(joints) != 1:
            raise ValueError("ARCTIC V1.5 first stage must be L=2, J=1")
    else:
        raise ValueError("V1.5 first stage only accepts grab and arctic")
    return {"num_links": links, "joints": [_validate_joint_spec(item, links) for item in joints]}


@lru_cache(maxsize=32)
def _load_articulation_arrays(sequence_path: str, domain: str) -> dict[str, np.ndarray]:
    geometry = Path(sequence_path) / "geometry"
    names = ["obj_root_pose_world" if domain == "arctic" else "obj_pose_world"]
    if domain == "arctic":
        names.extend(("obj_part_id", "obj_articulation"))
    return {name: np.load(geometry / f"{name}.npy", mmap_mode="r") for name in names}


class ArticulatedSequenceView:
    """Validated V1.5 view over a V1.4 geometry sequence and explicit metadata."""

    def __init__(self, path: str | Path, domain: str, split: str, hand_variant: str,
                 articulation: Mapping[str, Any]) -> None:
        self.base = InspireSequenceView(path, domain, split, hand_variant,
                                       allow_manifest_split_override=True)
        self.path, self.domain, self.split = self.base.path, self.base.domain, self.base.split
        self.frame_count, self.hand_points = self.base.frame_count, self.base.hand_points
        self.kinematics = _source_kinematics({"articulation": articulation}, domain)
        self.arrays = _load_articulation_arrays(str(self.path), self.domain)
        self.root_pose_key = "obj_root_pose_world" if self.domain == "arctic" else "obj_pose_world"
        root = self.arrays[self.root_pose_key]
        if root.shape != (self.frame_count, 4, 4):
            raise ValueError(f"{self.path}: {self.root_pose_key} has invalid shape {root.shape}")
        if not all(_is_se3(np.asarray(pose)) for pose in root):
            raise ValueError(f"{self.path}: {self.root_pose_key} is not SE(3)")
        part = self.arrays.get("obj_part_id", np.zeros((4096,), dtype=np.int64))
        if part.shape not in ((4096,), (self.frame_count, 4096)):
            raise ValueError(f"{self.path}: obj_part_id must be [4096] or [T,4096], got {part.shape}")
        part_value = np.asarray(part[0] if part.ndim == 2 else part, dtype=np.int64)
        if np.any(part_value < 0) or np.any(part_value >= self.kinematics["num_links"]):
            raise ValueError(f"{self.path}: obj_part_id is outside declared link range")
        if part.ndim == 2 and not np.array_equal(np.asarray(part), np.broadcast_to(part_value, part.shape)):
            raise ValueError(f"{self.path}: obj_part_id must preserve canonical point identity")
        if self.domain == "arctic":
            q = self.arrays["obj_articulation"]
            if q.shape not in ((self.frame_count,), (self.frame_count, 1)) or not np.isfinite(q).all():
                raise ValueError(f"{self.path}: obj_articulation must be finite [T] or [T,1]")

    def part_ids(self, selected: np.ndarray) -> np.ndarray:
        part = self.arrays.get("obj_part_id")
        if part is None:
            return np.zeros((len(selected),), dtype=np.int64)
        values = np.asarray(part[0] if part.ndim == 2 else part, dtype=np.int64)
        return values[selected]

    def q(self, frame: int) -> np.ndarray:
        if self.domain == "grab":
            return np.zeros((0,), dtype=np.float32)
        return np.asarray(self.arrays["obj_articulation"][frame], dtype=np.float32).reshape(1)


class ArticulatedTransitions(Dataset):
    """Transition dataset that exposes structural V1.5 fields without changing V1.4."""

    def __init__(self, sequence_specs: Sequence[Mapping[str, Any]], split: str, *, num_obj_points: int = 1024,
                 fixed_stride: int | None = 1, stride_values: Sequence[int] | None = None, base_seed: int = 42) -> None:
        values = tuple(int(value) for value in (stride_values if stride_values is not None else (fixed_stride,)))
        if not values or any(value <= 0 for value in values) or not 1 <= num_obj_points <= 4096:
            raise ValueError("V1.5 requires positive stride and 1..4096 object points")
        self.split, self.num_obj_points, self.stride_values, self.base_seed = split, int(num_obj_points), values, int(base_seed)
        self.sequences: list[ArticulatedSequenceView] = []
        self.entries: list[Mapping[str, Any]] = []
        for item in sequence_specs:
            domain = str(item.get("name", item.get("domain", "")))
            if domain not in ("grab", "arctic"):
                raise ValueError("V1.5 accepts only grab and arctic sequence specs")
            view = ArticulatedSequenceView(item["path"], domain, split, str(item["hand_variant"]), item["articulation"])
            self.sequences.append(view); self.entries.append(item)
        self.rows = [(sequence_index, frame) for sequence_index, view in enumerate(self.sequences)
                     for frame in range(view.frame_count - max(self.stride_values))]
        if not self.rows:
            raise ValueError("V1.5 has no valid transitions")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, current = self.rows[index]
        view = self.sequences[sequence_index]
        source_ids = np.asarray(view.base.arrays["source_frame_id"])
        seed = _stable_seed(self.base_seed, view.path, int(source_ids[current]), self.split)
        stride = self.stride_values[seed % len(self.stride_values)]
        future = current + stride
        frame_time = np.asarray(view.base.arrays["frame_time"], dtype=np.float64)
        if source_ids[future] <= source_ids[current] or frame_time[future] <= frame_time[current]:
            raise ValueError(f"{view.path}: invalid timeline")
        root, root_next = (np.asarray(view.arrays[view.root_pose_key][frame], dtype=np.float32)
                           for frame in (current, future))
        rotation, translation = root[:3, :3], root[:3, 3]
        next_rotation, next_translation = root_next[:3, :3], root_next[:3, 3]
        delta_rotation = rotation.T @ next_rotation
        delta_translation = (next_translation - translation) @ rotation
        delta_xi = np.concatenate((delta_translation, _rotation_log(delta_rotation))).astype(np.float32)
        seed = _stable_seed(self.base_seed, view.path, int(source_ids[current]), self.split)
        selected = np.random.default_rng(seed ^ 0xB15).choice(4096, size=self.num_obj_points, replace=False)
        object_world = np.asarray(view.base.arrays["obj_points_pool_world"][current, selected], dtype=np.float32)
        future_world = np.asarray(view.base.arrays["obj_points_pool_world"][future, selected], dtype=np.float32)
        hand_world = np.asarray(view.base.arrays["knn_hand_points_world"][current], dtype=np.float32)
        hand_future_world = np.asarray(view.base.arrays["knn_hand_points_world"][future], dtype=np.float32)
        joint_count, links = len(view.kinematics["joints"]), view.kinematics["num_links"]
        joints = view.kinematics["joints"]
        q, q_next = view.q(current), view.q(future)
        arrays = {
            "obj_points": _world_to_frame(object_world, root),
            "obj_normals": _normal_world_to_frame(np.asarray(view.base.arrays["obj_normals_pool_world"][current, selected]), root),
            "hand_points": _world_to_frame(hand_world, root),
            "hand_normals": _normal_world_to_frame(np.asarray(view.base.arrays["knn_hand_normals_world"][current]), root),
        }
        arrays["hand_flow"] = _world_to_frame(hand_future_world, root) - arrays["hand_points"]
        arrays["obj_flow_gt"] = _world_to_frame(future_world, root) - arrays["obj_points"]
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"{view.path}: non-finite transition")
        return {
            **{key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)) for key, value in arrays.items()},
            "obj_link_id": torch.from_numpy(view.part_ids(selected)),
            "link_valid_mask": torch.ones((links,), dtype=torch.bool),
            "joint_parent": torch.tensor([joint["parent"] for joint in joints], dtype=torch.long),
            "joint_child": torch.tensor([joint["child"] for joint in joints], dtype=torch.long),
            "joint_axis_root": torch.from_numpy(np.asarray([joint["axis_root"] for joint in joints], dtype=np.float32).reshape(joint_count, 3)),
            "joint_origin_root": torch.from_numpy(np.asarray([joint["origin_root"] for joint in joints], dtype=np.float32).reshape(joint_count, 3)),
            "joint_q_t": torch.from_numpy(q.copy()), "joint_valid_mask": torch.ones((joint_count,), dtype=torch.bool),
            "delta_q_gt": torch.from_numpy((q_next - q).copy()), "delta_xi_root_gt": torch.from_numpy(delta_xi),
            "delta_time_s": torch.tensor(float(frame_time[future] - frame_time[current]), dtype=torch.float32),
            "stride": torch.tensor(stride, dtype=torch.int64),
            "hand_valid_mask": torch.ones((view.hand_points,), dtype=torch.bool),
            "source": view.domain, "sequence_id": str(self.entries[sequence_index].get("id", view.path)),
        }


def collate_articulated(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad hand/link/joint dimensions independently for a mixed GRAB/ARCTIC batch."""
    if not batch:
        raise ValueError("cannot collate empty V1.5 batch")
    result: dict[str, Any] = {}
    hand_keys = {"hand_points", "hand_normals", "hand_flow", "hand_valid_mask"}
    link_keys = {"link_valid_mask"}
    joint_keys = {"joint_parent", "joint_child", "joint_axis_root", "joint_origin_root", "joint_q_t", "joint_valid_mask", "delta_q_gt"}
    max_hand = max(int(sample["hand_points"].shape[0]) for sample in batch)
    max_link = max(int(sample["link_valid_mask"].shape[0]) for sample in batch)
    max_joint = max(int(sample["joint_valid_mask"].shape[0]) for sample in batch)
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in hand_keys | link_keys | joint_keys:
            width = max_hand if key in hand_keys else (max_link if key in link_keys else max_joint)
            tail = values[0].shape[1:]
            fill = -1 if key in {"joint_parent", "joint_child"} else 0
            padded = torch.full((len(batch), width, *tail), fill, dtype=values[0].dtype)
            for row, value in enumerate(values):
                padded[row, :value.shape[0]] = value
            result[key] = padded
        elif torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def _segment_mean(values: Tensor, ids: Tensor, valid: Tensor, include: Tensor | None = None) -> Tensor:
    """Masked mean by link id; IDs are structural selectors, never embeddings."""
    batch, _, width = values.shape
    links = valid.shape[1]
    output = values.new_zeros((batch, links, width))
    for link in range(links):
        mask = (ids == link) & valid[:, link, None]
        if include is not None:
            mask = mask & include
        denom = mask.sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        output[:, link] = (values * mask[..., None].to(values.dtype)).sum(1) / denom
    return output


def _revolute_relative(points: Tensor, axis: Tensor, origin: Tensor, delta_q: Tensor) -> Tensor:
    """Rotate row-vector points about an axis/origin by the relative joint angle."""
    rotation = _axis_angle_matrix_stable(axis * delta_q[:, None])
    return torch.bmm(points - origin[:, None], rotation.transpose(1, 2)) + origin[:, None]


def analytic_fk_flow(points: Tensor, link_id: Tensor, link_valid: Tensor, joint_parent: Tensor,
                     joint_child: Tensor, joint_axis: Tensor, joint_origin: Tensor,
                     joint_valid: Tensor, delta_xi_root: Tensor, delta_q: Tensor) -> Tensor:
    """First-stage masked analytic FK: root SE(3) plus independent revolute edges."""
    if points.ndim != 3 or link_id.shape != points.shape[:2]:
        raise ValueError("invalid point/link shapes for analytic FK")
    batch, count, _ = points.shape
    transformed = points.clone()
    for joint in range(joint_valid.shape[1]):
        valid = joint_valid[:, joint]
        if not valid.any():
            continue
        parent, child = joint_parent[:, joint], joint_child[:, joint]
        if torch.any(valid & ((parent < 0) | (child < 0) | (parent >= link_valid.shape[1]) | (child >= link_valid.shape[1]))):
            raise ValueError("valid joint has invalid parent/child")
        rotated = _revolute_relative(transformed, joint_axis[:, joint], joint_origin[:, joint], delta_q[:, joint])
        select = (link_id == child[:, None]) & valid[:, None]
        transformed = torch.where(select[..., None], rotated, transformed)
    rotation = _axis_angle_matrix_stable(delta_xi_root[:, 3:])
    future = torch.bmm(transformed, rotation.transpose(1, 2)) + delta_xi_root[:, None, :3]
    return future - points


class ArticulatedObjectInteractionCmv2V15Model(nn.Module):
    """Local interaction evidence -> link graph -> root/joint motion -> analytic FK."""

    architecture_version = ARTICULATED_VERSION

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        width = int(cfg.hidden_width)
        self.feature_scale_m = float(cfg.feature_scale_m)
        self.geometry_encoder = _mlp(7, width, width)
        self.local_interaction = LocalInteractionEncoder(width, int(cfg.knn_k), float(cfg.interaction_radius_m), "swept",
                                                         feature_scale_m=self.feature_scale_m, frame_dt_s=float(cfg.frame_dt_s))
        self.link_encoder = _mlp(2 * width + 1, width, width)
        self.joint_encoder = _mlp(7, width, width)
        self.message = _mlp(3 * width, width, width)
        self.node_update = _mlp(2 * width, width, width)
        self.edge_update = _mlp(3 * width, width, width)
        self.root_head = nn.Linear(width, 6)
        self.joint_head = nn.Linear(width, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        required = ("obj_points", "obj_normals", "hand_points", "hand_normals", "hand_flow", "hand_valid_mask",
                    "obj_link_id", "link_valid_mask", "joint_parent", "joint_child", "joint_axis_root",
                    "joint_origin_root", "joint_q_t", "joint_valid_mask")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing V1.5 articulated fields: {missing}")
        points, normals = batch["obj_points"], F.normalize(batch["obj_normals"], dim=-1, eps=1e-8)
        link_valid, joint_valid = batch["link_valid_mask"].bool(), batch["joint_valid_mask"].bool()
        if (batch["obj_link_id"] < 0).any() or (batch["obj_link_id"] >= link_valid.shape[1]).any():
            raise ValueError("obj_link_id is outside padded link range")
        centered = points - points.mean(1, keepdim=True)
        scale = centered.square().sum(-1).mean(1).sqrt().clamp_min(1e-8)
        geo = self.geometry_encoder(torch.cat((centered / scale[:, None, None], normals,
                                               torch.log(scale / self.feature_scale_m)[:, None, None].expand(-1, points.shape[1], 1)), -1))
        contact, diagnostics = self.local_interaction(geo, points, normals, batch["hand_points"], batch["hand_normals"],
                                                      batch["hand_flow"], batch["hand_valid_mask"], batch.get("delta_time_s"))
        active = diagnostics["has_interaction"]
        geometry = _segment_mean(geo, batch["obj_link_id"], link_valid)
        interaction = _segment_mean(contact, batch["obj_link_id"], link_valid, active)
        active_fraction = _segment_mean(active[..., None].to(points.dtype), batch["obj_link_id"], link_valid)[..., :1]
        nodes = self.link_encoder(torch.cat((geometry, interaction, active_fraction), -1)) * link_valid[..., None]
        axis, origin, q = batch["joint_axis_root"], batch["joint_origin_root"], batch["joint_q_t"]
        edge = self.joint_encoder(torch.cat((axis, origin / self.feature_scale_m, q[..., None]), -1)) * joint_valid[..., None]
        for _ in range(2):
            incoming = torch.zeros_like(nodes)
            for joint in range(joint_valid.shape[1]):
                valid = joint_valid[:, joint]
                if not valid.any():
                    continue
                parent, child = batch["joint_parent"][:, joint], batch["joint_child"][:, joint]
                if torch.any(valid & ((parent < 0) | (child < 0) | (parent >= nodes.shape[1]) | (child >= nodes.shape[1]))):
                    raise ValueError("joint graph index is invalid")
                p = nodes[torch.arange(nodes.shape[0], device=nodes.device), parent.clamp_min(0)]
                c = nodes[torch.arange(nodes.shape[0], device=nodes.device), child.clamp_min(0)]
                message_pc = self.message(torch.cat((p, c, edge[:, joint]), -1)) * valid[:, None]
                message_cp = self.message(torch.cat((c, p, edge[:, joint]), -1)) * valid[:, None]
                for row in range(nodes.shape[0]):
                    if valid[row]:
                        incoming[row, child[row]] += message_pc[row]
                        incoming[row, parent[row]] += message_cp[row]
                edge[:, joint] = self.edge_update(torch.cat((edge[:, joint], p, c), -1)) * valid[:, None]
            nodes = self.node_update(torch.cat((nodes, incoming), -1)) * link_valid[..., None]
        node_pool = (nodes * link_valid[..., None]).sum(1) / link_valid.sum(1, keepdim=True).clamp_min(1).to(nodes.dtype)
        delta_xi = self.root_head(node_pool)
        delta_q = self.joint_head(edge).squeeze(-1) * joint_valid.to(edge.dtype)
        flow = analytic_fk_flow(points, batch["obj_link_id"], link_valid, batch["joint_parent"], batch["joint_child"], axis,
                                origin, joint_valid, delta_xi, delta_q)
        return {"delta_xi_root": delta_xi, "delta_q": delta_q, "obj_flow_pred": flow,
                "link_nodes": nodes, "joint_nodes": edge, "contact_active": active, **diagnostics}


def articulated_v15_loss(output: Mapping[str, Tensor], batch: Mapping[str, Tensor], scale_m: float = 0.02) -> dict[str, Tensor]:
    """Structured V1.5 smoke loss; joint supervision is masked for rigid samples."""
    flow = F.smooth_l1_loss(output["obj_flow_pred"] / scale_m, batch["obj_flow_gt"] / scale_m)
    translation = F.smooth_l1_loss(output["delta_xi_root"][:, :3] / scale_m, batch["delta_xi_root_gt"][:, :3] / scale_m)
    rotation = F.smooth_l1_loss(output["delta_xi_root"][:, 3:], batch["delta_xi_root_gt"][:, 3:])
    mask = batch["joint_valid_mask"].to(output["delta_q"].dtype)
    joint = ((output["delta_q"] - batch["delta_q_gt"]).abs() * mask).sum() / mask.sum().clamp_min(1)
    return {"total": flow + translation + rotation + joint, "flow": flow, "translation": translation,
            "rotation": rotation, "joint": joint}
