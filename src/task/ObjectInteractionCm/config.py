"""V1.2.1 configuration for the independent ObjectInteractionCm task."""
from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "object_interaction_cm"
    runner_class = "src.task.ObjectInteractionCm.runner.ObjectInteractionCmRunner"
    work_version = "V1.2.1"
    operation_category = ["architecture", "code", "data", "operation"]

    class meta(TaskConfig.meta):
        coordinate_frame = "object_pose_t"
        num_obj_pool = 4096
        num_obj_points = 1024
        num_hand_points = 1538
        max_hand_points = 3076
        knn_k = 8
        hand_stream_mode = "decoder"
        interaction_radius_m = 0.05
        frame_filter_distance_m = 0.05
        hand_supervision_radius_m = 0.03
        processing_dim = 128
        cm_dim = 32
        # feature_dim remains as a legacy construction alias for old tests/checkpoints.
        feature_dim = 32
        num_cm_tokens = 16
        slot_iters = 3
        flow_smooth_l1_beta = 0.005
        loss_obj_flow_weight = 1.0
        loss_hand_flow_weight = 1.0
        geometry_scale_m = 0.05
        hand_flow_input_scale = 1.0
        knn_hand_flow_input_scale = 1.0
        object_flow_target_scale = 1.0
        hand_flow_target_scale = 1.0
        scale_manifest_path = "data/processed_data/object_interaction_cm_v1_2_1/scales_train.json"
        sample_loss_mask = True
        dummy_token_policy = "zero"
        sampling_retry_attempts = 0

    class model(TaskConfig.model):
        class_path = "src.task.ObjectInteractionCm.model.ObjectInteractionCmModel"

    class data(TaskConfig.data):
        index_path = "data/processed_data/object_interaction_cm_v1_1/index.json"
        root = ""
        train_path = ""
        val_path = None
        test_path = None
        split_json_path = None
        val_split = 0.1
        test_fraction = 0.1
        min_stride = 1
        max_stride = 20
        eval_stride = 2
        grab_stride_values = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        inspire_stride_values = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
        source_probabilities = {"grab": 0.5, "inspire_f1": 0.5}
        val_strides = (2,)
        test_strides = (2,)
        active_only = True

    class train(TaskConfig.train):
        metric_for_best = "val/obj/flow_epe_mm"
        lower_is_better = True
        amp = False
        device = "cuda"
        epochs = 50
        max_steps = 202300
        lr = 3e-4
        weight_decay = 1e-4
        log_every_steps = 100
        eval_every_epochs = 1
        save_every_epochs = 1
        description = "GRAB + Inspire-F1 mixed ObjectInteractionCm V1.2.1; delayed compression, masked SlotAttn, additive decoders."

    class wandb(TaskConfig.wandb):
        enable = False
        project = "ref2dex"
        mode = "offline"
        tags = ["object-interaction-cm", "slot-attention", "delayed-compression", "v1.2.1"]
