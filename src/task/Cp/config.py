from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "cp_human_closure"
    runner_class = "src.task.Cp.runner.CpHumanClosureRunner"

    class meta(TaskConfig.meta):
        stage: str = "cp_effect"  # cp_effect | closed_loop
        num_obj_points: int = 512
        num_hand_points: int = 1538
        dim: int = 256
        cm_dense_token_dim: int = 96
        num_cp_tokens: int = 8
        num_cm_tokens: int = 16
        slot_iters: int = 3
        contact_radius: float = 0.02
        flow_smooth_l1_beta: float = 0.01
        # Stage 5 stores object flow in metres. Normalize it for the Cp
        # reconstruction loss so millimetre-scale effects retain gradients.
        object_flow_scale_m: float = 0.01
        loss_object_flow_weight: float = 1.0
        loss_object_contact_weight: float = 1.0
        cp_effect_checkpoint: str = ""
        # A Cm checkpoint that already includes the trained Cm->Fm decoder.
        cm_hand_checkpoint: str = ""
        # Optional override when the Cm checkpoint config contains an obsolete
        # dense-token checkpoint path.
        cm_dense_checkpoint: str = ""

    class model(TaskConfig.model):
        class_path = "src.task.Cp.model.CpHumanClosureModel"

    class data(TaskConfig.data):
        group_val_by_sequence = True
        active_only = False
        # Optional raw row index inside a single Stage 5 NPZ. This is useful
        # for deterministic one-pair overfit diagnostics.
        selected_pair_index: int | None = None

    class train(TaskConfig.train):
        output_dir = "outputs/train/cp_human_closure"
        amp = False
        metric_for_best = "val/hand_flow_epe_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cp", "human-closure"]
