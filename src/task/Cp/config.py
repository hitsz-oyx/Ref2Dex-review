from __future__ import annotations

from src.base import TaskConfig


class Config(TaskConfig):
    name = "cp_human_closure"
    runner_class = "src.task.Cp.runner.CpHumanClosureRunner"

    class meta(TaskConfig.meta):
        stage: str = "hand_decoder"  # hand_decoder | cp_effect | closed_loop
        num_obj_points: int = 512
        num_hand_points: int = 1538
        dim: int = 256
        num_cp_tokens: int = 8
        num_cm_tokens: int = 16
        slot_iters: int = 3
        contact_radius: float = 0.02
        flow_smooth_l1_beta: float = 0.01
        hand_decoder_checkpoint: str = ""
        cp_effect_checkpoint: str = ""

    class model(TaskConfig.model):
        class_path = "src.task.Cp.model.CpHumanClosureModel"

    class data(TaskConfig.data):
        group_val_by_sequence = True
        active_only = False

    class train(TaskConfig.train):
        output_dir = "outputs/train/cp_human_closure"
        amp = False
        metric_for_best = "val/hand_flow_epe_mm"
        lower_is_better = True

    class wandb(TaskConfig.wandb):
        project = "ref2dex"
        mode = "offline"
        tags = ["cp", "human-closure"]
