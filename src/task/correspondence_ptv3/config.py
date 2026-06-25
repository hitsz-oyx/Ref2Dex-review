from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "correspondence_ptv3"
    runner_class = "src.task.correspondence_ptv3.runner.CorrespondencePTV3Runner"

    class meta(TaskConfig.meta):
        num_obj_points: int = 512
        num_hand_points: int = 1538
        k_cross: int = 32

        hidden_dim: int = 96
        ptv3_repo_path: str = "/home/oyx/test_ws/PointTransformerV3"
        ptv3_grid_size: float = 0.003
        ptv3_order: tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
        ptv3_stride: tuple[int, ...] = (2, 2, 2, 2)
        ptv3_enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2)
        ptv3_enc_channels: tuple[int, ...] = (96, 192, 384, 384, 384)
        ptv3_enc_num_head: tuple[int, ...] = (6, 12, 24, 24, 24)
        ptv3_enc_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
        ptv3_dec_depths: tuple[int, ...] = (2, 2, 2, 2)
        ptv3_dec_channels: tuple[int, ...] = (96, 192, 384, 384)
        ptv3_dec_num_head: tuple[int, ...] = (6, 12, 24, 24)
        ptv3_dec_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024)
        ptv3_mlp_ratio: float = 4.0
        ptv3_qkv_bias: bool = True
        ptv3_attn_drop: float = 0.0
        ptv3_proj_drop: float = 0.0
        ptv3_drop_path: float = 0.1
        ptv3_pre_norm: bool = True
        ptv3_shuffle_orders: bool = True
        ptv3_enable_rpe: bool = False
        ptv3_enable_flash: bool = True
        ptv3_upcast_attention: bool = False
        ptv3_upcast_softmax: bool = False

        d_pos: float = 0.005
        d_neg: float = 0.03
        gamma: float = 2.0
        corr_contact_label_min: float = 0.1
        hand_rot_std_deg: float = 10.0
        hand_trans_std: float = 0.01
        hand_perturb_prob: float = 1.0
        recompute_input_cross_knn: bool = True
        val_augment: bool = False

        num_fingers: int = 6
        num_regions: int = 6
        use_finger_region_head: bool = False

        loss_contact_weight: float = 1.0
        loss_cano_weight: float = 1.0
        loss_finger_weight: float = 0.25
        loss_region_weight: float = 0.25
        loss_cross_edge_weight: float = 1.0

        augment_rotation: bool = True
        augment_translation: bool = False
        augment_scale: bool = False
        rotation_range: float = 180.0
        translation_range: float = 0.1
        scale_range: tuple[float, float] = (0.9, 1.1)

    class model(TaskConfig.model):
        class_path = "src.task.correspondence_ptv3.model.StaticHOCPTv3"
        type = "static_hoc_ptv3"

    class data(TaskConfig.data):
        train_path: str = ""
        val_path: str = ""
        blacklist_path: str | None = None
        val_split: float = 0.1
        batch_size: int = 8
        val_batch_size: int = 16
        num_workers: int = 0
        shuffle: bool = True
        drop_last: bool = True
        pin_memory: bool = True
        persistent_workers: bool = False
        prefetch_factor: int = 2

    class train(TaskConfig.train):
        output_dir: str = "outputs/train/correspondence_ptv3"
        seed: int = 42
        device: str = "auto"
        epochs: int = 100
        max_steps: int | None = None
        optimizer: str = "adamw"
        lr: float = 1e-4
        weight_decay: float = 1e-4
        scheduler: str = "cosine"
        warmup_steps: int = 1000
        grad_clip_norm: float | None = 1.0
        amp: bool = False
        compile: bool = False
        log_every_steps: int = 50
        eval_every_steps: int | None = None
        eval_every_epochs: int = 1
        save_every_steps: int | None = None
        save_every_epochs: int = 10
        max_to_keep: int = 5
        resume: str | None = None
        early_stopping_patience: int | None = None
        early_stopping_threshold: float = 0.0
        metric_for_best: str = "val/loss"
        lower_is_better: bool = True

    class wandb(TaskConfig.wandb):
        enable: bool = True
        project: str = "ref2dex"
        entity: str | None = None
        group: str | None = None
        name: str = "correspondence_ptv3"
        tags: list = ["correspondence", "ptv3", "static"]
        mode: str = "offline"
        job_type: str = "train"
        log_model: bool = False
