"""RL contracts shared by the Ref2Dex residual task."""

from .residual_contract import (
    ACTION_DIM,
    OBSERVATION_DIM,
    expand_native_targets,
    residual_action_bounds,
)

__all__ = ["ACTION_DIM", "OBSERVATION_DIM", "expand_native_targets", "residual_action_bounds"]
