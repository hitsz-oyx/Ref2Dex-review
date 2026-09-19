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
    """Map the released normalized Inspire action to native PD targets."""
    for name, value in (("action", action), ("current_native", current_native),
                        ("lower", lower), ("upper", upper)):
        _validate(name, value)
    normalized = action.clamp(-1.0, 1.0)
    scale = upper - lower
    scale = scale.clone()
    scale[..., :3] = 1.0
    scale[..., 3:6] = math.pi
    # DExplore deliberately uses a zero offset for every DOF.  Finger actions
    # are moved to [0, 1] before applying the full URDF range.
    offset = torch.zeros_like(lower)
    pd_action = torch.cat((normalized[..., :6], (1.0 + normalized[..., 6:]) / 2.0), dim=-1)
    targets = offset + scale * pd_action
    targets[..., :6] = targets[..., :6] + current_native[..., :6]
    targets = apply_mimic(targets)
    return targets


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
    clamped = unclamped.clamp(lower, upper)
    # The released base policy does not clamp its final target.  Preserve that
    # exact path for strict zero-residual rollouts; the safety clamp remains on
    # every environment that actually requests a residual correction.
    zero_residual = residual.abs().amax(dim=-1, keepdim=True) == 0
    targets = torch.where(zero_residual, base_targets, clamped)
    applied = targets - base_targets
    saturation = ((clamped != unclamped) & ~zero_residual).to(targets.dtype)
    return targets, {
        "base_targets": base_targets,
        "requested_delta": requested,
        "applied_delta": applied,
        "saturation": saturation,
    }


def compose_reference_residual(base_targets: torch.Tensor, residual_action: torch.Tensor,
                               lower: torch.Tensor, upper: torch.Tensor, *,
                               translation_scale_m: float, rotation_scale_rad: float,
                               finger_scale_rad: float,
                               mimic_scales: tuple[float, ...]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add a bounded residual to an absolute retargeted native PD target."""
    for name, value in (("base_targets", base_targets), ("residual_action", residual_action),
                        ("lower", lower), ("upper", upper)):
        _validate(name, value)
    if ((base_targets < lower - 1e-5) | (base_targets > upper + 1e-5)).any():
        raise ValueError("Retargeted base target exceeds native joint limits")
    if len(mimic_scales) != len(MIMIC_NATIVE):
        raise ValueError("Retargeted mimic scale count mismatch")
    def couple(value):
        result = value.clone()
        independent = result[..., list(INDEPENDENT_NATIVE)]
        for mimic, source, scale in zip(MIMIC_NATIVE, MIMIC_SOURCE, mimic_scales):
            result[..., mimic] = independent[..., source] * float(scale)
        return result
    if not torch.allclose(couple(base_targets), base_targets, atol=1e-5, rtol=0):
        raise ValueError("Retargeted base target violates mimic coupling")
    residual = residual_action.clamp(-1.0, 1.0)
    configured_authority = torch.zeros_like(base_targets)
    configured_authority[..., :3] = float(translation_scale_m)
    configured_authority[..., 3:6] = float(rotation_scale_rad)
    configured_authority[..., list(INDEPENDENT_NATIVE)] = float(finger_scale_rad)
    effective_lower, effective_upper = lower.clone(), upper.clone()
    for mimic, source, scale in zip(MIMIC_NATIVE, MIMIC_SOURCE, mimic_scales):
        if float(scale) <= 0:
            raise ValueError("Retargeted mimic scale must be positive")
        source_native = INDEPENDENT_NATIVE[source]
        effective_lower[..., source_native] = torch.maximum(
            effective_lower[..., source_native], lower[..., mimic] / float(scale))
        effective_upper[..., source_native] = torch.minimum(
            effective_upper[..., source_native], upper[..., mimic] / float(scale))
    effective_authority = torch.zeros_like(base_targets)
    requested = torch.zeros_like(base_targets)
    independent = tuple(range(6)) + tuple(INDEPENDENT_NATIVE)
    for index in independent:
        positive = torch.minimum(configured_authority[..., index],
                                 (effective_upper[..., index] - base_targets[..., index]).clamp_min(0))
        negative = torch.minimum(configured_authority[..., index],
                                 (base_targets[..., index] - effective_lower[..., index]).clamp_min(0))
        effective_authority[..., index] = torch.where(residual[..., index] >= 0, positive, negative)
        requested[..., index] = residual[..., index] * effective_authority[..., index]
    # A zero residual retains the exact reference target; it is not an
    # authority reduction event even if that target lies on a joint boundary.
    authority_limited = ((effective_authority < configured_authority - 1e-7) &
                         (residual.abs() > 1e-7))
    for mimic, source, _ in zip(MIMIC_NATIVE, MIMIC_SOURCE, mimic_scales):
        authority_limited[..., mimic] = authority_limited[..., INDEPENDENT_NATIVE[source]]
    unclamped = couple(base_targets + requested)
    targets = unclamped.clamp(lower, upper)
    saturation = (targets.sub(unclamped).abs() > 1e-7).to(targets.dtype)
    return targets, {"base_targets": base_targets, "requested_delta": requested,
                     "applied_delta": targets - base_targets, "saturation": saturation,
                     "configured_authority": configured_authority,
                     "effective_authority": effective_authority,
                     "authority_limited": authority_limited.to(targets.dtype)}
