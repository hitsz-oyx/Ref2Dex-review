"""V1.2 configuration for Inspire hand-flow reconstruction from ObjectInteractionCm."""
from __future__ import annotations

from src.task.CmDecoder.config import Config as BaselineConfig


class Config(BaselineConfig):
    modification_version = "V1.2.0"
    operation_category = ["architecture", "code", "data", "experiment", "operation"]

    class meta(BaselineConfig.meta):
        cm_checkpoint = (
            "outputs/objectinteractioncm/"
            "object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt"
        )
        cache_root = "data/processed_data/cm_decoder/hrdexdb_inspire_f1_object_pose_t_20260901"
        cache_pair_root = "data/processed_data/cm_decoder/cmdecoder_inspire_f1_object_pose_t_v1_2"
        use_cached_cm_tokens = False
        use_cached_point_bindings = False
        required_hand_flow_frame = "object_pose_t"
        coordinate_frame = "object_pose_t"
        num_hand_points = 1538
        max_hand_points = 3076
        num_obj_points = 1024
        sample_seed = 42
        point_flow_target_scale = 1.0
        point_flow_loss_beta_m = 0.01

    class model(BaselineConfig.model):
        class_path = "src.task.CmDecoder.object_interaction_cm_model.ObjectInteractionCmHandFlowAutoencoder"

    class data(BaselineConfig.data):
        cache_manifest = (
            "data/processed_data/cm_decoder/hrdexdb_inspire_f1_object_pose_t_20260901/"
            "v4/selection_all_object_disjoint_seed42.json"
        )
        random_horizon_max_stride = 20
        random_horizon_stride_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        train_all_strides = True
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
        weight_decay = 1.0e-4
        amp = False
        metric_for_best = "val/hand_flow/epe_mm"
        lower_is_better = True
        description = "CmDecoder V1.2 Inspire-F1 all-point hand-flow conditional autoencoding"

    class wandb(BaselineConfig.wandb):
        enable = False
        project = "ref2dex"
        mode = "offline"
        tags = ["cm-decoder", "inspire-f1", "object-interaction-cm", "v1.2", "all-strides", "all-hand-points"]
