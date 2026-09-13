"""Tensor-only contract for a frozen base target plus residual policy."""
from __future__ import annotations

import torch

ACTION_DIM = 12  # independent finger residual q6 + wrist translation/rotation residual6
OBSERVATION_DIM = 71  # native q18, native dq18, base q6, wrist pose7, object pose7, five tip offsets15
INDEPENDENT_NATIVE = (6, 8, 10, 12, 14, 15)
MIMIC_NATIVE = (7, 9, 11, 13, 16, 17)
MIMIC_SOURCE = (0, 1, 2, 3, 5, 5)  # indices in independent q6, not native18
MIMIC_SCALE = (1.05, 1.05, 1.05, 1.05, 0.6, 0.8)
NATIVE_TO_URDF = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9)


def residual_action_bounds(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized residual limits used by the first physics wiring smoke."""
    lower = torch.tensor([-1.0] * ACTION_DIM, dtype=torch.float32, device=device)
    upper = torch.tensor([1.0] * ACTION_DIM, dtype=torch.float32, device=device)
    return lower, upper


def expand_native_targets(finger_q: torch.Tensor) -> torch.Tensor:
    """Expand q6 into native18 targets with the DExplore mimic convention.

    The function deliberately does not clamp: the simulator task clamps against
    its loaded URDF limits after composing base target and residual.
    """
    if finger_q.shape[-1] != 6:
        raise ValueError(f"Expected [...,6] independent finger q, got {tuple(finger_q.shape)}")
    result = torch.zeros(*finger_q.shape[:-1], 18, dtype=finger_q.dtype, device=finger_q.device)
    result[..., :6] = 0.0
    result[..., list(INDEPENDENT_NATIVE)] = finger_q
    source = finger_q[..., list(MIMIC_SOURCE)]
    result[..., list(MIMIC_NATIVE)] = source * torch.tensor(MIMIC_SCALE, dtype=finger_q.dtype, device=finger_q.device)
    return result


def native_to_urdf(native_q: torch.Tensor) -> torch.Tensor:
    if native_q.shape[-1] != 18:
        raise ValueError(f"Expected [...,18] native q, got {tuple(native_q.shape)}")
    result = torch.empty_like(native_q)
    result[..., list(NATIVE_TO_URDF)] = native_q
    return result
