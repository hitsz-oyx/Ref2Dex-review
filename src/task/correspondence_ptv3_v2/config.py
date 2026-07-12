from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "correspondence_ptv3_v2"
    runner_class = "src.task.correspondence_ptv3_v2.runner.CorrespondencePTV3V2Runner"

    class meta(TaskConfig.meta):
        num_obj_points: int = 512
        num_obj_pool: int = 4096
        num_hand_points: int = 1538
        k_ctx: int = 32
        ctx_radius: float = 0.04
        num_supervision_edges: int = 128
        point_feat_dim: int = 11

        ptv3_repo_path: str = str(ROOT / "third_party" / "PointTransformerV3")
        ptv3_grid_size: float = 0.003
        ptv3_order: tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
        ptv3_stride: tuple[int, ...] = (2, 2, 2, 2)
        ptv3_enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2)
        ptv3_enc_channels: tuple[int, ...] = (96, 192, 384, 384, 384)
        ptv3_enc_num_head: tuple[int, ...] = (6, 12, 24, 24, 24)
        ptv3_enc_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
        ptv3_dec_depths: tuple[int, ...] = (2, 2, 2, 2)
        ptv3_dec_channels: tuple[int, ...] = (96, 192, 384, 384)
        ptv3_dec_num_head: tuple[int, ...] = (6, 12, 24, 24)
        ptv3_dec_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024)
        ptv3_mlp_ratio: float = 4.0
        ptv3_qkv_bias: bool = True
        ptv3_attn_drop: float = 0.0
        ptv3_proj_drop: float = 0.0
        ptv3_drop_path: float = 0.1
        ptv3_pre_norm: bool = True
        ptv3_shuffle_orders: bool = True
        ptv3_enable_rpe: bool = False
        ptv3_enable_flash: bool = True
        ptv3_upcast_attention: bool = False
        ptv3_upcast_softmax: bool = False

        contact_radius: float = 0.01
        quality_focal_beta: float = 2.0

        hand_rot_std_deg: float = 10.0
        hand_trans_std: float = 0.01
        hand_perturb_prob: float = 1.0
        augment: bool = True
        apply_hand_perturb: bool = True
        augment_rotation: bool = True
        augment_translation: bool = False
        augment_scale: bool = False
        rotation_range: float = 180.0
        translation_range: float = 0.1
        scale_range: tuple[float, float] = (0.9, 1.1)
        val_augment: bool = True
        val_hand_perturb_prob: float = 1.0

        loss_obj_contact_weight: float = 1.0
        loss_cross_edge_weight: float = 1.0

    class model(TaskConfig.model):
        class_path = "src.task.correspondence_ptv3_v2.model.StaticHOCPTv3V2"

    class data(TaskConfig.data):
        group_val_by_sequence = True
        sequence_locality_shuffle = True
        blacklist_path = None

    class train(TaskConfig.train):
        output_dir = "outputs/train/correspondence_ptv3_v2"
