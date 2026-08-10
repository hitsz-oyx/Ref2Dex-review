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
    if int(cfg.meta.num_patches) != 64 or int(cfg.meta.patch_size) != 32:
        raise ValueError("ActionToken 固定使用 canonical 64×32 atlas")
    if str(cfg.model.class_path).endswith("ActionTokenV2Model"):
        if len(cfg.data.motion_stds) != 3 or len(cfg.data.motion_probs) != 3:
            raise ValueError("V2 motion mixture 必须包含 small/medium/large 三组")
        if abs(sum(float(value) for value in cfg.data.motion_probs) - 1) > 1e-6:
            raise ValueError("V2 motion_probs 之和必须为 1")
        return
    if int(cfg.meta.chunk_len) != 8:
        raise ValueError("ActionToken V1 固定使用 8 transitions")
    if not Path(cfg.meta.pose_encoder_checkpoint).is_file():
        raise FileNotFoundError(f"缺少 PoseEncoder: {cfg.meta.pose_encoder_checkpoint}")
