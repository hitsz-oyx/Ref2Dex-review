"""BaseRunner configuration for temporal CmAction learning."""
from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "cm_action"
    runner_class = "src.task.Cm.runner.CmActionRunner"

    class meta(TaskConfig.meta):
        dense_checkpoint: str = str(ROOT / "src" / "task" / "Cm" / "densetoken_ckpt" / "best.pt")
        coordinate_frame: str = "hand_root_t"
        num_obj_pool: int = 4096
        num_obj_points: int = 512
        num_hand_points: int = 1538
        cm_dim: int = 256
        num_cm_tokens: int = 16
        slot_iters: int = 3
        flow_smooth_l1_beta: float = 0.01
        loss_flow_weight: float = 1.0

    class model(TaskConfig.model):
        class_path = "src.task.Cm.model.CmFlowModel"

    class data(TaskConfig.data):
        group_val_by_sequence: bool = True
        # Stage 4 preserves non-contact motion; the initial frozen-dense-token
        # training path selects only pairs that have at least one valid object
        # candidate.  Set false only for explicit no-contact diagnostics.
        active_only: bool = True
        # Current 5cm contact is the only training-sample filter.  Do not
        # remove static object targets: they teach no-effect interactions.
        min_stride: int = 1
        max_stride: int = 12
        val_strides = tuple(range(1, 13))
        max_train_samples = None
        max_val_samples = None

    class train(TaskConfig.train):
        output_dir = "outputs/train/cm_action"
        amp = False
        metric_for_best = "val/flow_mse"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cm", "temporal", "point-flow", "slot-attention", "frozen-dense-token"]
