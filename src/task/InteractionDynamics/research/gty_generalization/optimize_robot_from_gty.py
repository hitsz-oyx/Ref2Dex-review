"""用 V20 causal GT-Y 离线优化 URDF robot trajectory。"""
from __future__ import annotations

import torch

from src.task.InteractionDynamics.mano_field_decoder_v20_1 import build_causal_rdv_batched
from src.task.InteractionDynamics.viewer_gty.robot_backend import UrdfHandBackend


def representative_frames(gt_y: torch.Tensor) -> list[int]:
    """从 9 帧窗口选择 pre-contact/first-contact/formation/stable 代表帧。"""
    contacts = (gt_y[..., 3] < 2).sum(0)
    first = int(torch.nonzero(contacts >= 4)[0]) + 1 if bool((contacts >= 4).any()) else 4
    speed = gt_y[..., 4:7].square().mean((0, 2)).sqrt()
    stable = torch.nonzero((contacts >= 4) & (speed < .3))
    stable_frame = int(stable[-1]) + 1 if len(stable) else 8
    selected = {max(0, first - 1), first, min(8, first + 2), stable_frame}
    for candidate in torch.linspace(0, 8, 5).round().int().tolist():
        if len(selected) >= 4:
            break
        selected.add(candidate)
    return sorted(selected)


def optimize_robot_from_gty(backend: UrdfHandBackend, gt_y: torch.Tensor, anchors_cm: torch.Tensor,
                            gt_vertices: torch.Tensor, steps: int = 300, lr: float = 3e-2,
                            sample_points: int = 128) -> dict:
    device = gt_y.device; frames = gt_y.shape[1] + 1
    q_mid = backend.limits.mean(1); q_raw = torch.zeros(frames, backend.dof, device=device, requires_grad=True)
    total_vertices = sum(len(visual.vertices) for visual in backend.visuals)
    indices = torch.linspace(0, total_vertices - 1, min(sample_points, total_vertices),
                             device=device).round().long()
    neutral = backend.vertices(q_mid[None].repeat(frames, 1), torch.zeros(frames, 3, device=device),
                               torch.zeros(frames, 3, device=device), indices)
    root_translation = (gt_vertices.mean(1) - neutral.mean(1)).detach().requires_grad_()
    root_rotation = torch.zeros(frames, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam((q_raw, root_translation, root_rotation), lr=lr)
    start_loss = None
    for _ in range(steps):
        q = backend.limits[:, 0] + torch.sigmoid(q_raw) * (backend.limits[:, 1] - backend.limits[:, 0])
        vertices = backend.vertices(q, root_rotation, root_translation, indices)
        prediction = build_causal_rdv_batched(vertices[None], anchors_cm[None] / 100)[0]
        error = prediction - gt_y
        loss = error[..., :4].square().mean() + .5 * error[..., 4:7].square().mean()
        loss = loss + 1e-3 * (q[1:] - q[:-1]).square().mean()
        if start_loss is None:
            start_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad():
        q = backend.limits[:, 0] + torch.sigmoid(q_raw) * (backend.limits[:, 1] - backend.limits[:, 0])
        sampled = backend.vertices(q, root_rotation, root_translation, indices)
        prediction = build_causal_rdv_batched(sampled[None], anchors_cm[None] / 100)[0]
        vertices = backend.vertices(q, root_rotation, root_translation)
        frame_error = (prediction - gt_y).square().mean((0, 2)).sqrt()
    return {"q": q.detach(), "vertices": vertices.detach(), "prediction": prediction.detach(),
            "frame_error": frame_error.detach(), "joint_margin": backend.joint_margin(q).detach(),
            "start_loss": start_loss, "final_loss": float(loss.detach())}
