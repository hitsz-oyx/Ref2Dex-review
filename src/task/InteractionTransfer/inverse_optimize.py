"""V1.0 inverse optimization backend：以 hand action 为变量，冻结 forward model 反解。

目标（第一版，无正则）::

    ΔH* = argmin MSE( ΔO(ΔH), target )        # target_kind = "effect"
    ΔH* = argmin MSE( C_obj(ΔH), C_obj(ΔH_GT) )  # target_kind = "c_obj"

两种参数化（V1.0 指导 §8）：
- ``rigid``：双手各自一个 6 维 twist，共 12 维：左手块由 (t_L, ω_L) 刚体
  变换，右手块由 (t_R, ω_R) 刚体变换，不再把两只手绑成同一 SE(3)；
- ``free``：直接优化 ΔH ∈ R^{1538x3}（上限诊断）。

``optimize_hand_flow`` 返回 initial / optimized 的 flow、object_flow、
object_field、EPE 与 history，供 eval 与 Viewer 共用。
"""
from __future__ import annotations

import torch

from .metrics import epe
from .model import InteractionTransfer


def rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """axis-angle [B,3] -> 旋转矩阵 [B,3,3]，可微。"""
    theta = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)  # [B,1]
    k = omega / theta.clamp_min(1e-8)
    zeros = torch.zeros_like(k[..., 0])
    K = torch.stack([
        torch.stack([zeros, -k[..., 2], k[..., 1]], dim=-1),
        torch.stack([k[..., 2], zeros, -k[..., 0]], dim=-1),
        torch.stack([-k[..., 1], k[..., 0], zeros], dim=-1),
    ], dim=-2)  # [B,3,3]
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(K.shape[0], 3, 3)
    return eye + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * (K @ K)


def fit_rigid_twist(flow: torch.Tensor, hand_points: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """最小二乘把 flow [B,J,3] 拟合为 rigid twist [B,6]。

    每轮用小角度线性解残差并迭代精化，3 轮后误差通常远小于 Adam 步长。
    """
    center = hand_points.mean(dim=1, keepdim=True)
    r = hand_points - center                              # [B,J,3]
    zeros = torch.zeros_like(r[..., 0])
    skew_r = torch.stack([
        torch.stack([zeros, -r[..., 2], r[..., 1]], dim=-1),
        torch.stack([r[..., 2], zeros, -r[..., 0]], dim=-1),
        torch.stack([-r[..., 1], r[..., 0], zeros], dim=-1),
    ], dim=-2)                                            # [B,J,3,3] skew(r)
    A = -skew_r                                           # ω × r = res 中 ω 的系数
    AtA = torch.einsum("bjkl,bjkm->blm", A, A)            # [B,3,3]
    eye = torch.eye(3, device=flow.device, dtype=flow.dtype).expand(AtA.shape)

    twist = torch.zeros(flow.shape[0], 6, device=flow.device, dtype=flow.dtype)
    for _ in range(iters):
        residual = flow - twist_to_flow(twist, hand_points, center)
        rel = residual - residual.mean(dim=1, keepdim=True)
        Atb = torch.einsum("bjkl,bjkm->blm", A, rel.unsqueeze(-1))    # [B,3,1]
        d_omega = torch.linalg.solve(AtA + 1e-8 * eye, Atb).squeeze(-1)
        d_t = (residual - (rodrigues(d_omega) @ r.transpose(1, 2)).transpose(1, 2)).mean(dim=1)
        twist = twist + torch.cat([d_t, d_omega], dim=-1)
    return twist


def twist_to_flow(twist: torch.Tensor, hand_points: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """twist [B,6] -> ΔH [B,J,3]：h' - h = (R(ω)-I)(h-c_H) + t。"""
    R = rodrigues(twist[:, 3:])
    rotated = (R @ (hand_points - center).transpose(1, 2)).transpose(1, 2)
    return rotated - (hand_points - center) + twist[:, :3].unsqueeze(1)


def fit_rigid_twist_pair(flow: torch.Tensor, left_points: torch.Tensor,
                         right_points: torch.Tensor) -> torch.Tensor:
    """左右手各自拟合 rigid twist，返回 [B,12] = [twist_L, twist_R]。"""
    return torch.cat([fit_rigid_twist(flow[:, :left_points.shape[1]], left_points),
                      fit_rigid_twist(flow[:, left_points.shape[1]:], right_points)], dim=-1)


def twist_pair_to_flow(twist: torch.Tensor, left_points: torch.Tensor,
                       right_points: torch.Tensor) -> torch.Tensor:
    """[B,12] 双手 twist -> ΔH [B,J_L+J_R,3]；每只手绕自身质心旋转。"""
    n_left = left_points.shape[1]
    flow_l = twist_to_flow(twist[:, :6], left_points, left_points.mean(dim=1, keepdim=True))
    flow_r = twist_to_flow(twist[:, 6:], right_points, right_points.mean(dim=1, keepdim=True))
    return torch.cat([flow_l, flow_r], dim=1)


def optimize_hand_flow(
    model: InteractionTransfer,
    static: dict,
    hand_points: torch.Tensor,
    target_object_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    init_flow: torch.Tensor,
    dt: torch.Tensor,
    parameterization: str = "rigid",
    target_kind: str = "effect",
    steps: int = 300,
    lr: float = 0.01,
    history_every: int = 25,
) -> dict:
    """冻结 model，优化 hand action 使预测 effect（或 C_obj）逼近 target。

    static 为 ``encode_static`` 输出（已 batch，双手 concat 顺序
    [left, right]）；dt 为该 transition 的 normalized gap；init_flow 为
    初始 action（zero/cross/random/GT 由调用方构造）。
    """
    model.requires_grad_(False)
    device = hand_points.device
    left_count = hand_points.shape[1] // 2
    left_points, right_points = hand_points[:, :left_count], hand_points[:, left_count:]

    if target_kind == "c_obj":
        with torch.no_grad():
            target_field = model.forward_core(**static, hand_flow=gt_flow, dt=dt)["object_field"]

    def forward_out(flow: torch.Tensor) -> dict:
        return model.forward_core(**static, hand_flow=flow, dt=dt)

    with torch.no_grad():
        initial_out = forward_out(init_flow)

    if parameterization == "rigid":
        var = fit_rigid_twist_pair(init_flow, left_points, right_points).detach().clone().requires_grad_(True)

        def current_flow() -> torch.Tensor:
            return twist_pair_to_flow(var, left_points, right_points)
    elif parameterization == "free":
        var = init_flow.detach().clone().requires_grad_(True)

        def current_flow() -> torch.Tensor:
            return var
    else:
        raise ValueError(f"unknown parameterization: {parameterization}")

    optimizer = torch.optim.Adam([var], lr=lr)
    history = []
    for step in range(steps + 1):
        flow = current_flow()
        out = forward_out(flow)
        if target_kind == "effect":
            loss = torch.nn.functional.mse_loss(out["object_flow"], target_object_flow)
        else:
            loss = torch.nn.functional.mse_loss(out["object_field"], target_field)
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % history_every == 0 or step == steps:
            with torch.no_grad():
                history.append({"step": step, "loss": float(loss),
                                "epe_mm": float(epe(out["object_flow"], target_object_flow)) * 1000})

    with torch.no_grad():
        optimized_flow = current_flow().detach()
        optimized_out = forward_out(optimized_flow)
        gt_out = forward_out(gt_flow)

    def summarize(out: dict, flow: torch.Tensor) -> dict:
        return {
            "flow": flow.detach().cpu(),
            "object_flow": out["object_flow"].detach().cpu(),
            "object_field": out["object_field"].detach().cpu(),
            "epe_mm": float(epe(out["object_flow"], target_object_flow)) * 1000,
            "c_obj_l2_to_gt": float(torch.linalg.vector_norm(
                out["object_field"] - gt_out["object_field"], dim=-1).mean()),
        }

    return {
        "initial": summarize(initial_out, init_flow),
        "optimized": summarize(optimized_out, optimized_flow),
        "gt": summarize(gt_out, gt_flow),
        "history": history,
        "meta": {"parameterization": parameterization, "target_kind": target_kind,
                 "steps": steps, "lr": lr, "dt": float(dt.reshape(-1)[0])},
    }
