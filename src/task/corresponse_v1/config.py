from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "corresponse_v1"
    runner_class = "src.task.corresponse_v1.runner.CorrResponseRunner"

    class meta(TaskConfig.meta):
        # Point counts
        num_obj_points: int = 512
        num_hand_points: int = 1538

        # KNN graph sizes
        k_obj_local: int = 16
        k_hand_local: int = 16
        k_cross: int = 32

        # Model hyperparameters
        hidden_dim: int = 128
        num_local_layers: int = 3
        num_cross_layers: int = 2
        mlp_ratio: int = 2

        # Embedding dimensions
        type_emb_dim: int = 16
        finger_emb_dim: int = 16
        region_emb_dim: int = 16

        # Contact label parameters
        d_pos: float = 0.005
        d_neg: float = 0.03
        gamma: float = 2.0
        corr_contact_label_min: float = 0.0

        # Number of fingers and regions for classification
        num_fingers: int = 6  # 0=palm, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
        num_regions: int = 6  # Auto-updated from dataset metadata before model build when available.

        # Loss weights
        loss_contact_weight: float = 1.0
        loss_cano_weight: float = 5.0
        loss_finger_weight: float = 0.25
        loss_region_weight: float = 0.25
        loss_cross_edge_weight: float = 1.0

        # Data augmentation
        augment_rotation: bool = True
        augment_translation: bool = True
        augment_scale: bool = False
        rotation_range: float = 180.0  # degrees
        translation_range: float = 0.1  # meters
        scale_range: tuple[float, float] = (0.9, 1.1)

    class model(TaskConfig.model):
        class_path = "src.task.corresponse_v1.model.StaticHOCNet"
        type = "static_hoc_net"

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
        output_dir: str = "outputs/train/corresponse_v1"
        seed: int = 42
        device: str = "auto"
        epochs: int = 100
        max_steps: int | None = None
        optimizer: str = "adamw"
        lr: float = 3e-4
        weight_decay: float = 1e-4
        scheduler: str = "cosine"
        warmup_steps: int = 1000
        grad_clip_norm: float = 1.0
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
        name: str = "corresponse_v1"
        tags: list = ["correspondence", "static"]
        mode: str = "offline"
        job_type: str = "train"
        log_model: bool = False
