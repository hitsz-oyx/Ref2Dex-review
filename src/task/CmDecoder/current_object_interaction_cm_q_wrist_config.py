"""V1.2.13 direct q+wrist Decoder from frozen ObjectInteractionCm tokens."""
from __future__ import annotations

from src.task.CmDecoder.current_object_interaction_cm_v1_2_config import Config as CmConfig


class Config(CmConfig):
    modification_version = "V1.2.13"
    operation_category = ["architecture", "code", "data", "experiment", "operation"]

    class meta(CmConfig.meta):
        cm_checkpoint = (
            "outputs/objectinteractioncm/"
            "object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt"
        )
        decoder_input = "cm_only"
        prediction_target = "delta_q"
        q_input_scale = 1.0
        q_target_scale = 1.0
        q_loss_beta_rad = 0.02
        q_loss_weight = 1.0
        predict_wrist_motion = True
        wrist_translation_target_scale = 100.0
        wrist_rotation_target_scale = 1.0
        wrist_translation_loss_beta_m = 0.01
        wrist_rotation_loss_beta_rad = 0.05
        wrist_translation_loss_weight = 1.0
        wrist_rotation_loss_weight = 1.0
        baseline_point_loss_weight = 0.0
        use_cached_cm_tokens = False
        use_cached_point_bindings = False
        required_hand_flow_frame = "object_pose_t"
        coordinate_frame = "object_pose_t"
        num_hand_points = 1538
        max_hand_points = 3076
        num_obj_points = 1024
        sample_seed = 42

    class model(CmConfig.model):
        class_path = "src.task.CmDecoder.object_interaction_cm_q_wrist_model.ObjectInteractionCmQWristDecoder"

    class data(CmConfig.data):
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

    class train(CmConfig.train):
        epochs = 3
        max_steps = None
        lr = 3.0e-4
        weight_decay = 1.0e-4
        amp = False
        metric_for_best = "val/loss"
        lower_is_better = True
        description = "CmDecoder V1.2.13 Inspire-F1 direct q+wrist decoding from frozen ObjectInteractionCm"

    class wandb(CmConfig.wandb):
        enable = False
        project = "ref2dex"
        mode = "offline"
        tags = ["cm-decoder", "inspire-f1", "object-interaction-cm", "q-wrist", "v1.2.13"]
