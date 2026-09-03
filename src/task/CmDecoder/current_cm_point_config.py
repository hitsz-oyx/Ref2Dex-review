"""CmDecoder configuration aligned with the current Cm object-pose contract."""
from __future__ import annotations

from src.task.CmDecoder.config import Config as BaselineConfig


class Config(BaselineConfig):
    modification_version = "V1.1.0"
    operation_category = ["data", "experiment"]

    class meta(BaselineConfig.meta):
        cm_checkpoint = (
            "outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/"
            "checkpoints/best.pt"
        )
        cache_root = "data/processed_data/cm_decoder/hrdexdb_inspire_f1_object_pose_t_20260901"
        use_cached_cm_tokens = False
        use_cached_point_bindings = False
        required_hand_flow_frame = "object_pose_t"
        coordinate_frame = "object_pose_t"
        point_flow_target_scale = 1.0
        point_flow_loss_beta_m = 0.01

    class model(BaselineConfig.model):
        class_path = "src.task.CmDecoder.point_model.CmPointFlowModel"

    class data(BaselineConfig.data):
        cache_manifest = (
            "data/processed_data/cm_decoder/hrdexdb_inspire_f1_object_pose_t_20260901/"
            "v4/selection_all_object_disjoint_seed42.json"
        )
        random_horizon_max_stride = 20
        random_horizon_stride_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        eval_all_strides = True
        require_30hz_pair = True
        batch_size = 16
        val_batch_size = 16
        num_workers = 4
        persistent_workers = True
        drop_last = True

    class train(BaselineConfig.train):
        epochs = 30
        max_steps = None
        lr = 3.0e-4
        metric_for_best = "val/hand_flow/epe_mm"
        description = "cmdecoder_current_cm_object_pose_t_even_stride_point_flow"

    class wandb(BaselineConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cm-decoder", "inspire-f1", "current-cm", "object-pose-t", "even-stride", "point-flow"]
