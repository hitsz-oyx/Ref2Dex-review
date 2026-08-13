"""用 V20 causal GT-Y 离线优化 URDF robot trajectory。"""
from __future__ import annotations

import math

import torch
from pytorch3d.transforms import axis_angle_to_matrix

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
        if len(selected) >= 4: break
        selected.add(candidate)
    return sorted(selected)


def _loss_terms(prediction: torch.Tensor, target: torch.Tensor,
                q: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    error = prediction - target
    terms = {"r": error[..., :3].square().mean(), "d": error[..., 3].square().mean(),
             "v": error[..., 4:7].square().mean()}
    smooth = (q[1:] - q[:-1]).square().mean()
    return terms, terms["r"] + terms["d"] + .5 * terms["v"] + 1e-3 * smooth


def optimize_robot_from_gty(backend: UrdfHandBackend, gt_y: torch.Tensor, anchors_cm: torch.Tensor,
                            gt_vertices: torch.Tensor, steps: int = 300, lr: float = 3e-2,
                            optimize_mode: str = "full") -> dict:
    if optimize_mode not in {"full", "root", "finger"}:
        raise ValueError(optimize_mode)
    device = gt_y.device; frames = gt_y.shape[1] + 1
    if frames != 9 or len(gt_vertices) != 9:
        raise ValueError(f"GT-Y 优化要求 previous+8 的 9 帧，实际 {frames}/{len(gt_vertices)}")
    q_initial = backend.limits.mean(1)[None].repeat(frames, 1)
    q_raw = torch.zeros(frames, backend.dof, device=device, requires_grad=True)
    zero_rotation = torch.zeros(frames, 3, device=device)
    neutral = backend.surface_points(q_initial, zero_rotation, torch.zeros(frames, 3, device=device))
    translation0 = gt_vertices[0].mean(0) - neutral[0].mean(0)
    root_translation_initial = translation0[None].repeat(frames, 1)
    root_rotation_initial = zero_rotation.clone()
    root_translation = root_translation_initial.clone().requires_grad_()
    root_rotation = root_rotation_initial.clone().requires_grad_()
    parameters = []
    if optimize_mode in {"full", "finger"}: parameters.append(q_raw)
    if optimize_mode in {"full", "root"}: parameters.extend((root_translation, root_rotation))
    optimizer = torch.optim.Adam(parameters, lr=lr); history=[]; gradient_gate=None

    def state():
        q = (backend.limits[:, 0] + torch.sigmoid(q_raw) *
             (backend.limits[:, 1] - backend.limits[:, 0])) if optimize_mode != "root" else q_initial
        translation = root_translation if optimize_mode != "finger" else root_translation_initial
        rotation = root_rotation if optimize_mode != "finger" else root_rotation_initial
        return q, translation, rotation

    def diagnostics(step, terms, loss, q, translation, rotation):
        delta_q = q - q_initial; delta_t = (translation-root_translation_initial).norm(dim=-1)*100
        relative_rotation = (axis_angle_to_matrix(rotation) @
                             axis_angle_to_matrix(root_rotation_initial).transpose(-1, -2))
        cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        delta_r = torch.acos(cosine) * (180 / math.pi)
        return {"step": step, "loss": float(loss.detach()),
            "loss_r": float(terms["r"].detach()), "loss_d": float(terms["d"].detach()),
            "loss_v": float(terms["v"].detach()), "r_rmse": float(terms["r"].sqrt().detach()),
            "d_rmse": float(terms["d"].sqrt().detach()), "v_rmse": float(terms["v"].sqrt().detach()),
            "root_translation_delta_cm_mean": float(delta_t.mean()),
            "root_translation_delta_cm_max": float(delta_t.max()),
            "root_rotation_delta_deg_mean": float(delta_r.mean()),
            "root_rotation_delta_deg_max": float(delta_r.max()),
            "joint_delta_rms_rad": float(delta_q.square().mean().sqrt()),
            "joint_delta_max_rad": float(delta_q.abs().max()),
            "joint_delta_per_joint": {name: float(delta_q[:, i].square().mean().sqrt())
                                      for i, name in enumerate(backend.joint_names)}}

    for step in range(steps + 1):
        q, translation, rotation = state()
        surface = backend.surface_points(q, rotation, translation)
        prediction = build_causal_rdv_batched(surface[None], anchors_cm[None] / 100)[0]
        terms, loss = _loss_terms(prediction, gt_y, q)
        if step % 25 == 0 or step == steps:
            history.append(diagnostics(step, terms, loss, q, translation, rotation))
        if step == steps: break
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if step == 0:
            def grad_norm(value):
                return 0.0 if value.grad is None else float(value.grad.norm())
            gradient_gate = {"q_grad_norm": grad_norm(q_raw),
                "root_translation_grad_norm": grad_norm(root_translation),
                "root_rotation_grad_norm": grad_norm(root_rotation)}
            active_keys = {"full": tuple(gradient_gate), "root": ("root_translation_grad_norm",
                "root_rotation_grad_norm"), "finger": ("q_grad_norm",)}[optimize_mode]
            active = [gradient_gate[key] for key in active_keys]
            if not active or not all(math.isfinite(value) and value > 0 for value in active):
                raise RuntimeError(f"优化梯度 Gate 失败: {gradient_gate}")
        optimizer.step()
    with torch.no_grad():
        q, translation, rotation = state(); surface = backend.surface_points(q, rotation, translation)
        prediction = build_causal_rdv_batched(surface[None], anchors_cm[None] / 100)[0]
        terms, loss = _loss_terms(prediction, gt_y, q); vertices = backend.vertices(q, rotation, translation)
        initial_vertices = backend.vertices(q_initial, root_rotation_initial, root_translation_initial)
        frame_error = (prediction - gt_y).square().mean((0, 2)).sqrt()
    return {"q_initial": q_initial, "q": q.detach(), "vertices": vertices.detach(),
        "initial_vertices": initial_vertices.detach(), "root_translation_initial": root_translation_initial,
        "root_translation": translation.detach(), "root_rotation_initial": root_rotation_initial,
        "root_rotation": rotation.detach(), "prediction": prediction.detach(),
        "frame_error": frame_error.detach(), "joint_margin": backend.joint_margin(q).detach(),
        "start_loss": history[0]["loss"], "final_loss": float(loss),
        "final_metrics": diagnostics(steps, terms, loss, q, translation, rotation),
        "gradient_gate": gradient_gate, "optimization_history": history}
