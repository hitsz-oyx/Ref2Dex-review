"""Synthetic MANO PoseToken V1 配置。"""
from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "pose_token"
    runner_class = "src.task.Posetoken.runner.PoseTokenRunner"

    class meta(TaskConfig.meta):
        mano_path = str(ROOT / "dataset/arctic/data/body_models/mano")
        side = "right"
        num_hand_points = 1538
        num_patches = 64
        patch_size = 32
        model_dim = 384
        global_dim = 128
        attention_heads = 6
        motion_scale = 100.0
        num_pca_comps = 24

    class model(TaskConfig.model):
        class_path = "src.task.Posetoken.model.StaticPoseEncoder"
        pretrained_checkpoint = None

    class data(TaskConfig.data):
        train_samples = 4096
        val_samples = 512
        test_samples = 512
        val_seed = 43
        test_seed = 44
        generation_batch_size = 256
        beta_std = 0.0
        train_pose_group_size = 1
        eval_pose_group_size = 1
        batch_size = 64
        num_workers = 0
        shuffle = True

    class train(TaskConfig.train):
        epochs = 30
        lr = 3.0e-4
        weight_decay = 0.05
        dense_reconstruction_loss_weight = 1.0
        global_reconstruction_loss_weight = 1.0
        amp = True
        amp_dtype = "bfloat16"
        metric_for_best = "val/pose/reconstruction_epe_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["pose-token", "synthetic-mano", "static-pose"]


def validate_config(cfg: TaskConfig) -> None:
    if int(cfg.meta.num_hand_points) != 1538:
        raise ValueError("PoseToken V1 使用 MANO 的 1538 个稳定 face centers")
    if int(cfg.meta.model_dim) % int(cfg.meta.attention_heads):
        raise ValueError("model_dim 必须能被 attention_heads 整除")
    if str(cfg.meta.side) not in {"left", "right"}:
        raise ValueError("meta.side 必须是 left 或 right")
