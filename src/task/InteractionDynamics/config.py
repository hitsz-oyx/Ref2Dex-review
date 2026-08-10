"""Configuration contract for InteractionDynamics V1."""
from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "interaction_dynamics"
    runner_class = "src.task.InteractionDynamics.runner.InteractionDynamicsRunner"

    class meta(TaskConfig.meta):
        dense_checkpoint = str(ROOT / "src/task/Cm/densetoken_ckpt/best.pt")
        uni3d_checkpoint = str(ROOT / "src/task/InteractionDynamics/weights/uni3d-s/model.pt")
        chunk_len = 8
        temporal_stride = 1
        effective_fps = 30.0
        num_hand_points = 1538
        num_obj_world_points = 4096
        num_effect_points = 512
        num_hand_patches = 64
        num_obj_patches = 64
        patch_size = 32
        model_dim = 384
        action_temporal_layers = 2
        action_world_layers = 2
        attention_heads = 6
        motion_scale = 100.0
        action_local_gain = 10.0
        use_v7_field = False
        field_dim = 128
        field_sigma_m = 0.05
        field_ablation = "full"
        interaction_reconstruction = False
        action_decomposition_reconstruction = False
        pose_pair_action = False
        dense_edge_knn = 4
        dense_edge_dim = 64
        dense_edge_sigma_m = 0.02

    class model(TaskConfig.model):
        class_path = "src.task.InteractionDynamics.model.InteractionDynamicsModel"

    class data(TaskConfig.data):
        root = str(ROOT / "data/processed_data/interaction_dynamics_v1/data/grab")
        train_path = str(ROOT / "data/processed_data/interaction_dynamics_v1/data/grab")
        dominant_hand_manifest = None
        group_val_by_sequence = True
        max_train_samples = None
        max_val_samples = None
        max_samples_per_sequence = None
        min_object_effect_norm = 0.0
        batch_size = 8
        num_workers = 8
        shuffle = True

    class train(TaskConfig.train):
        seed = 42
        epochs = 30
        optimizer = "adamw"
        lr = 1.0e-4
        uni3d_lr = 1.0e-5
        weight_decay = 0.05
        scheduler = None
        grad_clip_norm = 1.0
        action_loss_weight = 1.0
        patch_effect_loss_weight = 1.0
        relative_loss_weight = 0.0
        relative_edge_radius_cm = 5.0
        relative_target_mode = "cumulative_fixed"
        effect_loss_weight = 1.0
        se3_translation_loss_weight = 0.0
        se3_rotation_loss_weight = 0.0
        dense_relative_loss_weight = 0.0
        interaction_reconstruction_loss_weight = 0.0
        action_articulation_loss_weight = 0.0
        action_root_translation_loss_weight = 0.0
        action_root_rotation_loss_weight = 0.0
        static_pose_reconstruction_loss_weight = 0.0
        amp = True
        amp_dtype = "bfloat16"
        metric_for_best = "val/object/ade_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["interaction-dynamics", "grab", "action-chunk", "object-effect"]


def validate_config(cfg: TaskConfig) -> None:
    """Fail fast when a V1 invariant is changed accidentally."""
    meta = cfg.meta
    expected = {
        "chunk_len": 8, "temporal_stride": 1, "num_hand_points": 1538,
        "num_obj_world_points": 4096, "num_effect_points": 512,
        "num_hand_patches": 64, "num_obj_patches": 64, "patch_size": 32,
        "model_dim": 384, "attention_heads": 6, "motion_scale": 100.0,
        "action_local_gain": 10.0,
    }
    changed = {key: (getattr(meta, key), value) for key, value in expected.items()
               if getattr(meta, key) != value}
    if changed:
        raise ValueError(f"InteractionDynamics V1 fixed parameters changed: {changed}")
    if meta.model_dim % meta.attention_heads:
        raise ValueError("model_dim must be divisible by attention_heads")
