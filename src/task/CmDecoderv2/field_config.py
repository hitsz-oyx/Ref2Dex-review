"""Configuration contract for the V1.1.16 B1 FieldRealizer gate."""
from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "cm_decoder_v2_field_realizer_v1_1_16"
    runner_class = "src.task.CmDecoderv2.field_runner.FieldRealizerRunner"
    work_version = "V1.1.16"
    operation_category = ["architecture", "code", "data", "experiment", "operation"]

    class meta(TaskConfig.meta):
        coordinate_frame = "object_pose_t"
        window_size = 4
        effective_fps = 30.0
        num_hand_points = 10135
        query_link_count = 18
        horizon_gamma = 0.8
        point_flow_smooth_l1_beta_m = 0.005
        loss_point_flow_weight = 1.0

    class model(TaskConfig.model):
        class_path = "src.task.CmDecoderv2.field_realizer.FieldRealizerModel"
        hidden_dim = 128
        num_heads = 4
        num_layers = 2
        feedforward_dim = 512
        dropout = 0.1
        surface_urdf = "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"
        surface_sampling = "v1_3_cache"

    class data(TaskConfig.data):
        index_path = "data/processed_data/cm_decoder_v2/mano_actual_finetune_v1_1_14_eligible_20260913/index.json"
        field_root = "data/processed_data/cm_decoder_v2/field_f7_parent_v1_1_16_all"
        urdf_path = "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"
        batch_size = 8
        val_batch_size = 8
        num_workers = 0
        persistent_workers = False
        pin_memory = True
        active_only = True

    class train(TaskConfig.train):
        device = "cuda"
        epochs = 20
        max_steps = None
        lr = 3e-4
        weight_decay = 1e-4
        amp = False
        grad_clip_norm = 1.0
        description = "V1.1.16 B1: parent-only oracle F7 to direct Inspire q/wrist with point-flow supervision."

    class wandb(TaskConfig.wandb):
        enable = False
        mode = "offline"
        tags = ["v1.1.16", "field-realizer", "f7", "b1"]
