from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "action_token"
    runner_class = "src.task.Actiontoken.runner.ActionTokenRunner"

    class meta(TaskConfig.meta):
        mano_path = str(ROOT / "dataset/arctic/data/body_models/mano")
        pose_encoder_checkpoint = str(ROOT / "output/Posetoken/research/pose_encoder_v2_morph.pt")
        side = "right"
        num_hand_points = 1538
        num_patches = 64
        patch_size = 32
        model_dim = 384
        global_dim = 128
        attention_heads = 6
        temporal_layers = 2
        chunk_len = 8
        num_pca_comps = 24
        motion_scale = 100.0

    class model(TaskConfig.model):
        class_path = "src.task.Actiontoken.model.ActionTokenModel"

    class data(TaskConfig.data):
        train_samples = 4096
        val_samples = 512
        test_samples = 512
        val_seed = 43
        test_seed = 44
        generation_batch_size = 128
        velocity_decay = 0.9
        velocity_std = 0.035
        batch_size = 16
        num_workers = 0
        shuffle = True

    class train(TaskConfig.train):
        epochs = 50
        lr = 1.0e-3
        weight_decay = 0.05
        scheduler = "cosine"
        amp = True
        amp_dtype = "bfloat16"
        metric_for_best = "val/action/dense_flow_epe_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["action-token", "synthetic-mano", "articulation"]


def validate_config(cfg: TaskConfig) -> None:
    if int(cfg.meta.chunk_len) != 8 or int(cfg.meta.num_patches) != 64:
        raise ValueError("ActionToken V1 固定使用 8 transitions 和 64 patches")
    if not Path(cfg.meta.pose_encoder_checkpoint).is_file():
        raise FileNotFoundError(f"缺少 PoseEncoder: {cfg.meta.pose_encoder_checkpoint}")
