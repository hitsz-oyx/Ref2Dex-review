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
        num_supervision_edges: int = 128
        point_feat_dim: int = 11

        # Coordinate frame for points read from Stage 3 .npz:
        #   "object"    - object-root SE(3) frame (legacy)
        #   "hand_root" - MANO wrist at origin, orientation = MANO global_orient
        # The Stage 3 .npz must be regenerated with the matching
        # --coordinate-frame flag.
        coordinate_frame: str = "hand_root"

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

        contact_radius: float = 0.02
        quality_focal_beta: float = 2.0

        # Contact-aware auxiliary supervision. Four target-strength bins
        # (weak / medium / strong / very_strong) share this quota. The
        # sampler does NOT refill from other bins: an object with 100 weak
        # candidates and 0 strong candidates will keep 16 weak + 0 strong.
        contact_supervision_quotas: tuple[int, int, int, int] = (16, 16, 16, 16)
        # Mix in a small set of "near but not touching" negatives so the
        # auxiliary stream learns a contact-vs-near-contact boundary instead
        # of treating every local neighborhood as positive.
        contact_supervision_hard_negative_quota: int = 16
        contact_supervision_hard_negative_distance_range: tuple[float, float] = (0.02, 0.03)

        # Object pose perturbation (applied to the 512 sampled obj points
        # with one shared SE(3); see perturb_object_geometry in sampling.py).
        # In hand-root frame the canonical nuisance variable is the object
        # pose estimation error, not a global scene transform.
        obj_rot_std_deg: float = 10.0
        obj_trans_std: float = 0.01
        obj_perturb_prob: float = 1.0
        apply_obj_perturb: bool = True
        val_obj_perturb_prob: float = 1.0

        # v2.1: hand-side MANO reconstruction on the GPU.  GRAB carries
        # PCA24 pose coefficients; ARCTIC carries axis-angle45.  The
        # dataset pads them for collation and the runner groups/slices by
        # the recorded mano_* descriptor before smplx.MANO forward.
        mano_model_dir: str = str(ROOT / "dataset" / "arctic" / "data" / "body_models" / "mano")
        use_mano_reconstruction: bool = True
        apply_hand_perturb: bool = False
        hand_pca_std: float = 0.5
        runtime_resample_object: bool = True
        num_hand_proxy_points: int = 256
        runtime_near_pool_points: int = 1024
        runtime_near_obj_points: int = 384

        loss_cross_edge_weight: float = 1.0
        # v2.2 hand-root tuning: keep the auxiliary branch as a weak prior so
        # it cannot dominate the random128 objective.
        loss_contact_aux_weight: float = 0.05
        # Dense hand heatmap supervision has many more terms than the sampled
        # edge streams, so it remains a deliberately weak multi-task signal.
        loss_hand_contact_weight: float = 0.005
        # Perturbed-validation-only recovery metrics ignore tiny changes in
        # the pseudo contact target caused by interpolation/numerical noise.
        pseudo_recovery_change_threshold: float = 0.05

    class model(TaskConfig.model):
        class_path = "src.task.correspondence_ptv3_v2.model.StaticHOCPTv3V2"

    class data(TaskConfig.data):
        group_val_by_sequence = True
        sequence_locality_shuffle = True
        blacklist_path = None

    class train(TaskConfig.train):
        amp = False
        diagnostic_every_steps = 20
        # v2.1 ablation: select checkpoints on the unbiased random-edge QFL
        # (the contact auxiliary stream is biased by construction and must
        # NOT be the model-selection metric).
        metric_for_best = "val_clean/cross_edge_random_qfl"
        lower_is_better = True
