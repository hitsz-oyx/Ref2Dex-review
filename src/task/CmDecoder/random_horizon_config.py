"""Random-horizon HRDexDB point-flow pilot.

One pseudo-random target stride in 1..10 is selected per current frame.  Cm is
encoded online from the corresponding hand flow; no horizon-specific Cm sidecar
is used, so this isolates temporal-coverage from cache changes.
"""
from __future__ import annotations

from src.task.CmDecoder.flat_point_config import Config as FlatConfig


class Config(FlatConfig):
    name = "cm_decoder_flat_point_random_horizon"

    class meta(FlatConfig.meta):
        use_cached_cm_tokens = False

    class data(FlatConfig.data):
        random_horizon_max_stride = 10
        require_30hz_pair = False

    class train(FlatConfig.train):
        epochs = 3
        description = "cm_flat_point_flow_random_horizon_1_to_10_pilot"
