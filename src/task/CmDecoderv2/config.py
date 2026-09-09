"""Configuration contract for CmDecoderv2 V1.1.1."""
from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "cm_decoder_v2"
    runner_class = "src.task.CmDecoderv2.runner.CmDecoderV2Runner"
    modification_version = "V1.1.1"
    operation_category = ["architecture", "code", "data", "experiment", "operation"]

    class meta(TaskConfig.meta):
        coordinate_frame = "object_pose_t"
        window_size = 4
        window_stride = 1
        effective_fps = 30.0
        num_obj_points = 1024
        num_hand_points = 1538
        num_cm_tokens = 16
        cm_dim = 32
        finger_q_dim = 6
        wrist_translation_dim = 3
        wrist_rotation_dim = 3
        query_link_count = 18
        q_smooth_l1_beta_rad = 0.02
        translation_smooth_l1_beta_m = 0.01
        translation_scale = 100.0
        horizon_gamma = 0.8
        loss_q_weight = 1.0
        loss_translation_weight = 1.0
        loss_rotation_weight = 1.0
        point_flow_smooth_l1_beta_m = 0.005
        loss_point_flow_weight = 1.0

    class model(TaskConfig.model):
        class_path = "src.task.CmDecoderv2.model.CmDecoderV2"
        hidden_dim = 128
        num_heads = 4
        num_layers = 2
        feedforward_dim = 512
        dropout = 0.1
        oicm_config = "src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml"
        oicm_checkpoint = "outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt"
        oicm_checkpoint_sha256 = "LOCK_AFTER_OICM_TERMINAL"
        strict_oicm = True
        surface_urdf = "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"

    class data(TaskConfig.data):
        view_root = "data/processed_data/cm_decoder_v2/dexplore_rl_v1_1"
        index_path = "data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json"
        urdf_path = "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"
        batch_size = 8
        val_batch_size = 8
        num_workers = 4
        persistent_workers = True
        pin_memory = True
        # Keep only windows with a 5 cm candidate and gate decoder supervision
        # per horizon by the corresponding source-frame candidate mask.
        active_only = True
        perturb_train = True
        translation_noise_std_m = 0.005
        translation_noise_clip_m = 0.015
        rotation_noise_std_deg = 5.0
        rotation_noise_clip_deg = 15.0
        finger_q_noise_std_rad = 0.05
        finger_q_noise_clip_rad = 0.15

    class train(TaskConfig.train):
        device = "cuda"
        epochs = 50
        max_steps = None
        lr = 3e-4
        weight_decay = 1e-4
        amp = False
        grad_clip_norm = 1.0
        log_every_steps = 100
        eval_every_epochs = 1
        save_every_epochs = 1
        metric_for_best = "val/loss"
        lower_is_better = True
        description = "Temporal-D2: frozen Dexplore OICM Cm window to Inspire q/wrist, supervised by differentiable surface point flow."

    class wandb(TaskConfig.wandb):
        enable = False
        project = "ref2dex"
        mode = "offline"
        tags = ["cm-decoder-v2", "temporal-d2", "point-flow", "dexplore-rl", "inspire", "v1.1.1"]
