from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "cm_decoder"
    runner_class = "src.task.CmDecoder.runner.CmDecoderRunner"

    class meta(TaskConfig.meta):
        cm_checkpoint: str = str(
            ROOT / "outputs/cm/cm_v121_grab_seed42_20260819_103530/checkpoints/best.pt"
        )
        dataset_root: str = "/home2/wyy/oyx_ws/HRDexDB/v0"
        robot_urdf: str = "/home2/wyy/oyx_ws/HRDexDB/assets/robots/xarm_inspire_f1_right.urdf"
        num_hand_points: int = 1538
        num_obj_points: int = 512
        sample_seed: int = 42
        cache_root: str = "data/processed_data/cm_decoder/hrdexdb_inspire_f1"
        q_loss_beta_rad: float = 0.02
        q_input_scale: float = 1.0
        q_target_scale: float = 1.0
        prediction_target: str = "delta_q"
        decoder_input: str = "qt_cm"
        flow_mode: str = "normal"
        use_cached_cm_tokens: bool = False

    class model(TaskConfig.model):
        class_path = "src.task.CmDecoder.model.CmDecoderModel"

    class data(TaskConfig.data):
        root: str = "/home2/wyy/oyx_ws/HRDexDB/v0"
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
