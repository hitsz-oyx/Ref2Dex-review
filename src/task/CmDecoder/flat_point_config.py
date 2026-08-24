"""30 Hz HRDexDB fast Cm-flat point-flow baseline."""
from __future__ import annotations

from src.task.CmDecoder.config import Config as BaselineConfig


class Config(BaselineConfig):
    name = "cm_decoder_flat_point_30hz"

    class meta(BaselineConfig.meta):
        cm_checkpoint = "outputs/cm/cm_object_v2_grab_arctic_20260819_165241/checkpoints/best.pt"
        cache_root = "data/processed_data/cm_decoder/hrdexdb_inspire_f1"
        use_cached_cm_tokens = True
        # v4's 30 Hz consecutive cache predates the v2 manifest-level frame tag;
        # its hand_flow is nevertheless constructed in the current wrist frame.
        required_hand_flow_frame = None
        flat_point_input_scale = 1.0
        flat_point_hidden_dim = 512
        point_flow_target_scale = 1.0
        point_flow_loss_beta_m = 0.01

    class model(BaselineConfig.model):
        class_path = "src.task.CmDecoder.flat_point_model.CmFlatPointFlowModel"

    class data(BaselineConfig.data):
        cache_manifest = "data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_576_seed42.json"
        require_30hz_pair = True
        batch_size = 64
        val_batch_size = 64
        num_workers = 4
        persistent_workers = True

    class train(BaselineConfig.train):
        epochs = 10
        metric_for_best = "val/hand_flow/epe_mm"
        lower_is_better = True
        description = "cm_flat_point_flow_30hz"
