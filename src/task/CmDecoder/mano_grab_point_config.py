"""GRAB-trained MANO point-flow decoder using the stopped mixed C=64 Cm."""
from __future__ import annotations

from src.task.CmDecoder.point_config import Config as PointConfig


class Config(PointConfig):
    name = "cmdecoder_grab_mano_pointflow_cm64_mixed"

    class meta(PointConfig.meta):
        cm_checkpoint = (
            "outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_cm64_"
            "20260822_190435/checkpoints/best.pt"
        )
        use_cached_cm_tokens = False
        required_hand_flow_frame = None
        point_flow_target_scale = 1.0
        point_flow_loss_beta_m = 0.01

    class data(PointConfig.data):
        root = "data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820"
        split_json_path = (
            "data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/"
            "splits_seed42/splits.json"
        )
        object_v2_filter = "grab"
        cache_manifest = None
        min_stride = 1
        max_stride = 10
        val_stride = 1
        test_stride = 1
        active_only = True
        batch_size = 64
        val_batch_size = 64
        num_workers = 4
        persistent_workers = True

    class train(PointConfig.train):
        epochs = 10
        eval_every_steps = 2000
        eval_every_epochs = None
        metric_for_best = "val/hand_flow/epe_mm"
        lower_is_better = True
        description = "GRAB MANO point-flow decoder, frozen mixed C=64 Cm"

    class wandb(PointConfig.wandb):
        mode = "offline"
        tags = [
            "cm-decoder", "mano-to-mano", "grab-train", "arctic-eval",
            "point-flow", "frozen-cm", "cm64-mixed",
        ]
