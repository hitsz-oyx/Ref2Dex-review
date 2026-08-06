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
        # Vector Huber transition in metres; 5 mm is close to the expected
        # point-flow noise scale while retaining EPE-like gradients above it.
        flow_smooth_l1_beta: float = 0.005
        # Calibration values are computed from the train split only by
        # ``python -m src.task.Cm.compute_flow_scale``.  They are stored in
        # config so every checkpoint/inference run can use the same scale.
        flow_target_rms_m: float | None = None
        # Formal experiments require train-root calibration metadata; legacy
        # and explicit scale ablations can opt out deliberately.
        require_flow_calibration: bool = False
        geometry_input_scale: float = 1.0
        hand_flow_input_scale: float = 1.0
        object_flow_target_scale: float = 1.0
        loss_flow_weight: float = 1.0
        loss_slot_count_weight: float = 1.0e-3
        loss_slot_confidence_weight: float = 0.1
        loss_active_overlap_weight: float = 0.0
        # Ablations can route through every Slot Attention output directly.
        # Keep the default enabled so existing hard-gate checkpoints retain
        # their exact behaviour.
        use_slot_gate: bool = True
        # Condition on the physical endpoint interval in seconds.  Disabled
        # by default to preserve existing checkpoints and experiments.
        use_time_condition: bool = False
        # Hard-Concrete's analytic nonzero probability is about 0.83 at a
        # zero log-alpha, so 0.85 makes the fallback path meaningful at init.
        slot_threshold: float = 0.85

    class model(TaskConfig.model):
        class_path = "src.task.Cm.model.CmFlowModel"

    class data(TaskConfig.data):
        group_val_by_sequence: bool = True
        # Stage 4 preserves non-contact motion; the initial frozen-dense-token
        # training path selects only pairs that have at least one valid object
        # candidate.  Set false only for explicit no-contact diagnostics.
        active_only: bool = True
        # Optional offline pseudo-label manifest.  When set, each manifest
        # row fixes (hand side, current frame, stride); the Stage 4 cache stays
        # immutable and the default unfiltered dataset path is unchanged.
        dominant_hand_manifest: str | None = None
        # Current 5cm contact is the only training-sample filter.  Do not
        # remove static object targets: they teach no-effect interactions.
        min_stride: int = 1
        max_stride: int = 10
        val_strides = tuple(range(1, 11))
        max_train_samples = None
        max_val_samples = None
        train_strides = None

    class train(TaskConfig.train):
        amp = False
        metric_for_best = "val/mean_stride_epe_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cm", "temporal", "point-flow", "slot-attention", "frozen-dense-token"]
