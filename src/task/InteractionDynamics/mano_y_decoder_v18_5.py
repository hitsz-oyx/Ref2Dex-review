"""V18.5 可微 MANO-H→Y structured decoder。"""
from __future__ import annotations

from collections.abc import Callable

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from src.task.InteractionDynamics.inverse_mano import face_centers


def build_interaction_y_batched(hand: torch.Tensor, anchors: torch.Tensor,
                                tau_m: float = .015, anchor_chunk: int = 16
                                ) -> dict[str, torch.Tensor]:
    """与 frozen teacher 等价的 batched/chunked Y；hand[B,T,P,3]。"""
    geometry, distance, motion = [], [], []
    displacement = torch.diff(hand, dim=1)
    for start in range(0, anchors.shape[1], anchor_chunk):
        anchor = anchors[:, None, start:start + anchor_chunk, None, :]
        relative = hand[:, :, None] - anchor
        weights = torch.softmax(-relative.square().sum(-1) / tau_m ** 2, dim=-1)
        geometry.append((weights[..., None] * relative).sum(-2))
        distance.append((weights * relative.norm(dim=-1)).sum(-1))
        motion.append((weights[:, :-1, ..., None] * displacement[:, :, None]).sum(-2))
    return {"relative_geometry": torch.cat(geometry, 2),
            "relative_distance": torch.cat(distance, 2),
            "relative_motion": torch.cat(motion, 2)}


def pack_batched_future(y: dict[str, torch.Tensor]) -> torch.Tensor:
    """把 batched Y 转成 V18 [B,anchors,8*7] future。"""
    transitions = torch.cat([y["relative_motion"], y["relative_geometry"][:, 1:],
                             y["relative_distance"][:, 1:, :, None]], -1)
    return transitions.transpose(1, 2).flatten(2) * 100


def decode_mano_y(delta_h: torch.Tensor, batch: dict, layer_for: Callable,
                  anchor_chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """按 subject/side 分组 MANO forward，scatter 后构造 future Y。"""
    device = delta_h.device; count = len(delta_h)
    surfaces: list[torch.Tensor | None] = [None] * count
    for key in sorted(set(batch["mano_key"])):
        indices = [i for i, value in enumerate(batch["mano_key"]) if value == key]
        index = torch.tensor(indices, device=device)
        layer = layer_for(key, batch["side"][indices[0]], batch["source_raw_file"][indices[0]])
        q = batch["object_rotation"][index]; t = batch["object_translation"][index]
        current_rotation = q[:, :1].transpose(-1, -2) @ axis_angle_to_matrix(
            batch["global_orient"][index, :1])
        relative = axis_angle_to_matrix(delta_h[index, :, 3:6]) @ current_rotation
        world_rotation = q[:, 1:] @ relative
        object_position = (batch["current_h"][index, None, :3] + delta_h[index, :, :3]) / 100
        world_translation = (object_position[..., None, :] - t[:, 1:]) @ q[:, 1:].transpose(-1, -2)
        future_pose = batch["hand_pose"][index, :1] + delta_h[index, :, 6:]
        current_surface = face_centers(layer(
            global_orient=batch["global_orient"][index, 0],
            hand_pose=batch["hand_pose"][index, 0], betas=batch["betas"][index, 0],
            transl=batch["transl"][index, 0]).vertices,
            torch.as_tensor(layer.faces, dtype=torch.long, device=device))
        future_surface = face_centers(layer(
            global_orient=matrix_to_axis_angle(world_rotation).flatten(0, 1),
            hand_pose=future_pose.flatten(0, 1), betas=batch["betas"][index, 1:].flatten(0, 1),
            transl=world_translation[..., 0, :].flatten(0, 1)).vertices,
            torch.as_tensor(layer.faces, dtype=torch.long, device=device)).reshape(len(index), 8, -1, 3)
        world_surface = torch.cat([current_surface[:, None], future_surface], 1)
        object_surface = torch.einsum("btpi,btij->btpj", world_surface, q) + t[:, :, None, 0]
        for local, original in enumerate(indices): surfaces[original] = object_surface[local]
    surface = torch.stack([value for value in surfaces if value is not None])
    y = build_interaction_y_batched(surface, batch["anchors_cm"] / 100, anchor_chunk=anchor_chunk)
    return pack_batched_future(y), surface
