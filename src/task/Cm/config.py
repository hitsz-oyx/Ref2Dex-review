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
        loss_object_weight: float = 5.0
        loss_hand_weight: float = 1.0
        loss_wrist_weight: float = 1.0
        loss_articulation_weight: float = 1.0
        loss_global_articulation_weight: float = 0.2
        wrist_rotation_weight_m_per_rad: float = 0.01
        hand_contact_sigma_m: float = 0.005
        hand_articulation_scale_m: float = 0.01
        hand_loss_base_weight: float = 0.2
        hand_loss_contact_weight: float = 1.5
        hand_loss_articulation_weight: float = 0.8
        hand_loss_max_weight: float = 3.0
        # Load an old object-flow Cm checkpoint without restoring its optimizer
        # or requiring the newly added hand decoder tensors to be present.
        warm_start_checkpoint: str | None = None
        freeze_cm_encoder: bool = False
        use_mano_aux: bool = False
        mano_model_dir: str = str(ROOT / "dataset" / "arctic" / "data" / "body_models" / "mano")
        mano_pose_dim: int = 24
        loss_mano_flow_weight: float = 0.5
        loss_mano_consistency_weight: float = 0.0
        loss_mano_pose_weight: float = 0.05

    class model(TaskConfig.model):
        class_path = "src.task.Cm.model.CmFlowModel"

    class data(TaskConfig.data):
        group_val_by_sequence: bool = True
        # Stage 4 preserves non-contact motion; the initial frozen-dense-token
        # training path selects only pairs that have at least one valid object
        # candidate.  Set false only for explicit no-contact diagnostics.
        active_only: bool = True
        min_object_flow_norm: float = 0.0
        max_train_samples = None
        max_val_samples = None

    class train(TaskConfig.train):
        output_dir = "outputs/train/cm_action"
        amp = False
        metric_for_best = "val/hand_epe_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cm", "temporal", "point-flow", "slot-attention", "frozen-dense-token"]
