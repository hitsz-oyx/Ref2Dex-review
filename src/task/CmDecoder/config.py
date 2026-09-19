from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "cm_decoder"
    runner_class = "src.task.CmDecoder.runner.CmDecoderRunner"
    work_version = "V1.2.12"
    operation_category = ["experiment"]

    class meta(TaskConfig.meta):
        cm_checkpoint: str = str(
            ROOT / "outputs/cm/cm_v121_grab_seed42_20260819_103530/checkpoints/best.pt"
        )
        dataset_root: str = str(ROOT / "dataset" / "HRDexDB" / "v0_nonvideo")
        robot_urdf: str = str(ROOT / "dataset" / "HRDexDB" / "assets" / "robots" / "xarm_inspire_f1_right.urdf")
        num_hand_points: int = 1538
        num_obj_points: int = 512
        sample_seed: int = 42
        cache_root: str = "data/processed_data/cm_decoder/hrdexdb_inspire_f1"
        q_loss_beta_rad: float = 0.02
        q_loss_weight: float = 1.0
        q_input_scale: float = 1.0
        q_target_scale: float = 1.0
        prediction_target: str = "delta_q"
        decoder_input: str = "qt_cm"
        flow_mode: str = "normal"
        use_cached_cm_tokens: bool = False
        use_cached_point_bindings: bool = False
        predict_wrist_motion: bool = False
        wrist_translation_target_scale: float = 100.0
        wrist_rotation_target_scale: float = 1.0
        wrist_translation_loss_beta_m: float = 0.01
        wrist_rotation_loss_beta_rad: float = 0.05
        wrist_translation_loss_weight: float = 1.0
        wrist_rotation_loss_weight: float = 1.0
        baseline_point_loss_weight: float = 0.0
        baseline_point_target_scale: float = 1.0
        baseline_point_loss_beta_m: float = 0.01

    class model(TaskConfig.model):
        class_path = "src.task.CmDecoder.model.CmDecoderModel"

    class data(TaskConfig.data):
        root: str = str(ROOT / "dataset" / "HRDexDB" / "v0_nonvideo")
        cache_manifest: str = "data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_20_seed42.json"
        batch_size: int = 16
        val_batch_size: int = 16
        num_workers: int = 4
        persistent_workers: bool = True
        shuffle: bool = True
        drop_last: bool = True
        val_fraction: float = 0.1
        test_fraction: float = 0.1
        max_episodes = None
        max_frames_per_episode = None
        active_motion_only: bool = False
        active_motion_threshold_deg: float = 0.5
        require_30hz_pair: bool = True
        episode_filter = None

    class train(TaskConfig.train):
        epochs = 30
        amp = False
        metric_for_best = "val/q/mae_deg"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cm-decoder", "hrdexdb", "inspire-f1", "frozen-cm"]
