"""BaseRunner configuration for temporal CmAction learning."""
from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


# ``config.py`` now lives under ``Cm/src``; the repository root is four
# parents above this file (``repo/src/task/Cm/src/config.py``).
ROOT = Path(__file__).resolve().parents[4]


class Config(TaskConfig):
    name = "cm_action"
    runner_class = "src.task.Cm.src.runner.CmActionRunner"
    modification_version = "V1.2.11"
    operation_category = ["experiment"]

    class meta(TaskConfig.meta):
        # Canonical Cm-owned asset path.  The large checkpoint remains in the
        # historical ``densetoken_ckpt`` directory behind a task-local symlink.
        dense_checkpoint: str = str(
            ROOT / "src" / "task" / "Cm" / "assets" / "checkpoints" / "densetoken" / "best.pt"
        )
        # DenseToken is frozen for the GRAB/ARCTIC base stage and explicitly
        # unfrozen by the HRDexDB fine-tuning config.
        freeze_dense_encoder: bool = True
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
        # ``python -m src.task.Cm.tools.data.compute_flow_scale``.  They are stored in
        # config so every checkpoint/inference run can use the same scale.
        flow_target_rms_m: float | None = None
        # Formal experiments require train-root calibration metadata; legacy
        # and explicit scale ablations can opt out deliberately.
        require_flow_calibration: bool = False
        geometry_input_scale: float = 1.0
        hand_flow_input_scale: float = 1.0
        object_flow_target_scale: float = 1.0
        # Optional geometry-only hand-flow decoder.  Its target normalization
        # is calibrated independently from object flow because the two point
        # fields have different physical magnitude distributions.
        use_hand_flow_decoder: bool = False
        hand_flow_target_rms_m: float | None = None
        hand_flow_target_scale: float = 1.0
        loss_hand_flow_weight: float = 0.0
        # A new hand decoder can be initialized on top of a legacy Cm
        # checkpoint while preserving strict loading for all existing keys.
        allow_missing_hand_decoder_init: bool = False
        loss_flow_weight: float = 1.0
        # Optional candidate-level mixture objective.  When non-zero, the
        # runner applies a soft-min over per-slot candidate flow losses using
        # ``candidate_mixture_temperature``.  This is kept opt-in so legacy
        # aggregate-flow checkpoints retain their exact objective.
        loss_candidate_mixture_weight: float = 0.0
        candidate_mixture_temperature: float = 0.05
        # Additive slot-flow ablation: each slot emits a 3-D contribution and
        # the public flow is their sum.  A small group penalty encourages an
        # effective slot count below ``num_cm_tokens`` without hard gating.
        use_additive_slot_contributions: bool = False
        loss_slot_group_sparsity_weight: float = 0.0
        loss_slot_count_weight: float = 1.0e-3
        loss_slot_confidence_weight: float = 0.1
        loss_active_overlap_weight: float = 0.0
        # Ablations can route through every Slot Attention output directly.
        # Keep the default enabled so existing hard-gate checkpoints retain
        # their exact behaviour.
        use_slot_gate: bool = True
        # Keep the legacy DenseToken z_obj context path by default.  Explicit
        # bottleneck ablations can disable it while retaining raw object
        # points/normals and Cm-relative geometry in the flow decoder.
        use_object_context: bool = True
        # Optional curriculum for hard gating.  During the full warm-up all
        # slots participate in the decoder; the following ramp raises the
        # threshold and L0 count weight to their configured target values.
        gate_warmup_enabled: bool = False
        gate_warmup_full_epochs: int = 5
        gate_warmup_ramp_epochs: int = 5
        # Condition on the physical endpoint interval in seconds.  Disabled
        # by default to preserve existing checkpoints and experiments.
        use_time_condition: bool = False
        # Hard-Concrete's analytic nonzero probability is about 0.83 at a
        # zero log-alpha, so 0.85 makes the fallback path meaningful at init.
        slot_threshold: float = 0.85
        # Decoder-only resume may freeze an already adapted DenseToken while
        # retaining its weights in subsequent Cm checkpoints.
        save_dense_encoder_in_checkpoint: bool = False

    class model(TaskConfig.model):
        class_path = "src.task.Cm.src.model.CmFlowModel"

    class data(TaskConfig.data):
        group_val_by_sequence: bool = True
        # Kept for compatibility with older caches.  New caches expose the
        # complete manipulated-object surface pool, so every transition has
        # valid object points and this flag does not filter by hand distance.
        active_only: bool = True
        # Optional offline pseudo-label manifest.  When set, each manifest
        # row fixes (hand side, current frame, stride); the Stage 4 cache stays
        # immutable and the default unfiltered dataset path is unchanged.
        dominant_hand_manifest: str | None = None
        # New cache sampling is uniform over the object surface; do not remove
        # static object targets because they teach no-effect interactions.
        min_stride: int = 1
        max_stride: int = 10
        val_strides = tuple(range(1, 11))
        max_train_samples = None
        max_val_samples = None
        train_strides = None
        # ---- Scene Cache V1 (see docs/指导/V1.md) ----
        # Dispatch is automatic: a data root whose meta.json declares
        # ``ref2dex_cm_scene_v1_1`` routes to Stage4CmSceneDataset.
        use_mmap: bool = True
        use_sampling_bank: bool = True
        sampling_bank_size: int = 4
        # Validation/test always evaluate a fixed bank for determinism.
        fixed_eval_bank: int = 0
        # Read precomputed frozen DenseToken features from dense_bank/ instead
        # of running the online PTv3 in the training loop.  Toggling false at
        # any time restores the online path for cache-parity validation.
        use_dense_cache: bool = False
        # HRDexDB fine-tune can select one embodiment prefix and omit the
        # GRAB/ARCTIC mixture when a dedicated adaptation run is requested.
        hrdexdb_only: bool = False
        hrdexdb_source_prefix: str | None = None

    class train(TaskConfig.train):
        amp = False
        metric_for_best = "val/mean_stride_epe_mm"
        lower_is_better = True
        # Initialize model weights from a completed base run while starting a
        # fresh optimizer/scheduler/step budget for fine-tuning.
        init_checkpoint: str | None = None
        # When resuming after freezing DenseToken, rebuild a decoder-only
        # optimizer and keep scheduler/global-step progress from the source.
        decoder_only_resume: bool = False

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cm", "temporal", "point-flow", "slot-attention", "frozen-dense-token"]
