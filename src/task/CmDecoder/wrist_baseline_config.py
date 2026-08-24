"""Full object-disjoint 3 Hz q+Cm baseline with relative wrist prediction."""
from __future__ import annotations

from src.task.CmDecoder.point_config import Config as PointConfig


class Config(PointConfig):
    class meta(PointConfig.meta):
        predict_wrist_motion = True
        use_cached_point_bindings = True
        baseline_point_loss_weight = 1.0
        q_loss_weight = 0.0
        wrist_translation_loss_weight = 0.0
        wrist_rotation_loss_weight = 0.0

    class model(PointConfig.model):
        class_path = "src.task.CmDecoder.model.CmDecoderModel"

    class train(PointConfig.train):
        metric_for_best = "val/hand_points/epe_mm"
        description = "cmdecoder_full_object_disjoint_3hz_wrist_baseline_pure_point_loss"
