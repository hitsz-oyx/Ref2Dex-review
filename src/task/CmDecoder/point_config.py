"""Full object-disjoint 3 Hz configuration for the per-point Cm decoder."""
from __future__ import annotations

from src.task.CmDecoder.config import Config as BaselineConfig


class Config(BaselineConfig):
    class meta(BaselineConfig.meta):
        cm_checkpoint = "outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt"
        cache_root = "data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz"
        use_cached_cm_tokens = True
        required_hand_flow_frame = "current_wrist"
        point_flow_target_scale = 1.0
        point_flow_loss_beta_m = 0.01
        q_fit_steps = 100
        q_fit_lr = 0.05
        q_fit_prior_weight = 1e-4
        wrist_fit_prior_weight = 1e-6

    class model(BaselineConfig.model):
        class_path = "src.task.CmDecoder.point_model.CmPointFlowModel"

    class data(BaselineConfig.data):
        cache_manifest = (
            "data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v2/"
            "selection_576_object_disjoint_seed42.json"
        )
        require_30hz_pair = False

    class train(BaselineConfig.train):
        epochs = 5
        metric_for_best = "val/hand_flow/epe_mm"
        description = "cmdecoder_full_object_disjoint_3hz_point_flow_then_q_fit"
