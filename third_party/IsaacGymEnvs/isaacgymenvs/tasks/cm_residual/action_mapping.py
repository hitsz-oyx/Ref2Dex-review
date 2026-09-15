"""DExplore-compatible target mapping with bounded physical residual authority."""
from __future__ import annotations

import math

import torch

from .contract import INDEPENDENT_NATIVE, MIMIC_NATIVE, MIMIC_SCALE, MIMIC_SOURCE


ACTION_DIM = 18


def _validate(name: str, value: torch.Tensor) -> None:
    if value.shape[-1] != ACTION_DIM:
        raise ValueError(f"{name} must end in {ACTION_DIM}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}")


def apply_mimic(targets: torch.Tensor) -> torch.Tensor:
    """Return targets with the released DExplore Inspire coupling applied."""
    result = targets.clone()
    independent = result[..., list(INDEPENDENT_NATIVE)]
    for mimic, source, scale in zip(MIMIC_NATIVE, MIMIC_SOURCE, MIMIC_SCALE):
        result[..., mimic] = independent[..., source] * scale
    return result


def dexplore_action_to_targets(
    action: torch.Tensor,
    current_native: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Map the released normalized Inspire action to clamped native PD targets."""
    for name, value in (("action", action), ("current_native", current_native),
                        ("lower", lower), ("upper", upper)):
        _validate(name, value)
    normalized = action.clamp(-1.0, 1.0)
    scale = upper - lower
    scale = scale.clone()
    scale[..., :3] = 1.0
    scale[..., 3:6] = math.pi
    offset = torch.zeros_like(lower)
    offset[..., 6:] = lower[..., 6:]
    pd_action = torch.cat((normalized[..., :6], (1.0 + normalized[..., 6:]) / 2.0), dim=-1)
    targets = offset + scale * pd_action
    targets[..., :6] = targets[..., :6] + current_native[..., :6]
    targets = apply_mimic(targets)
    return targets.clamp(lower, upper)


def compose_physical_residual(
    base_action: torch.Tensor,
    residual_action: torch.Tensor,
    current_native: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    *,
    translation_scale_m: float,
    rotation_scale_rad: float,
    finger_scale_rad: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply an 18-D normalized residual after DExplore physical target conversion."""
    _validate("residual_action", residual_action)
    base_targets = dexplore_action_to_targets(base_action, current_native, lower, upper)
    residual = residual_action.clamp(-1.0, 1.0)
    requested = torch.zeros_like(base_targets)
    requested[..., :3] = residual[..., :3] * float(translation_scale_m)
    requested[..., 3:6] = residual[..., 3:6] * float(rotation_scale_rad)
    finger_indices = list(INDEPENDENT_NATIVE)
    requested[..., finger_indices] = residual[..., finger_indices] * float(finger_scale_rad)
    unclamped = apply_mimic(base_targets + requested)
    targets = unclamped.clamp(lower, upper)
    applied = targets - base_targets
    saturation = (targets != unclamped).to(targets.dtype)
    return targets, {
        "base_targets": base_targets,
        "requested_delta": requested,
        "applied_delta": applied,
        "saturation": saturation,
    }
