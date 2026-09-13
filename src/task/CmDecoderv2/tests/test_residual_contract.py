from __future__ import annotations

import torch

from src.task.CmDecoderv2.rl.residual_contract import (
    expand_native_targets,
    native_to_urdf,
)


def test_residual_target_expansion_matches_rl_mimic_rule() -> None:
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    native = expand_native_targets(q)
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.1, 0.105, 0.2, 0.21, 0.3, 0.315,
                              0.4, 0.42, 0.5, 0.6, 0.36, 0.48]])
    torch.testing.assert_close(native, expected)


def test_native_to_urdf_is_invertible_for_batch() -> None:
    native = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    urdf = native_to_urdf(native)
    expected = torch.tensor([0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 6, 7, 8, 9, 12, 13, 10, 11])
    torch.testing.assert_close(urdf, native[:, expected.long()])
