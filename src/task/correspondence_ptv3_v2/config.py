from __future__ import annotations

from pathlib import Path

from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    name = "correspondence_ptv3_v2"
    runner_class = "src.task.correspondence_ptv3_v2.runner.CorrespondencePTV3V2Runner"
    modification_version = "V1.2.12"
    operation_category = ["experiment"]

    class meta(TaskConfig.meta):
        num_obj_points: int = 512
        num_obj_pool: int = 4096
        num_hand_points: int = 1538
        num_supervision_edges: int = 128
        point_feat_dim: int = 11

        # Coordinate frame for points read from Stage 3 .npz:
        #   "object"    - object-root SE(3) frame (object-centered protocol)
        #   "hand_root" - MANO wrist at origin, orientation = MANO global_orient
        # The Stage 3 .npz must be regenerated with the matching
        # --coordinate-frame flag.
        coordinate_frame: str = "hand_root"
        # Allow read-only hand-root Stage3 trees to be converted lazily to
        # object frame at dataset load time, avoiding a duplicate NAS tree.
        transform_to_object_frame: bool = False

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
        # Evaluation-only override for the perturbed validation stream. Keep
        # true for the historical object-only protocol; hand-only evaluation
        # sets this false without changing clean validation semantics.
        val_apply_obj_perturb: bool = True

        # v2.1: hand-side MANO reconstruction on the GPU.  GRAB carries
        # PCA24 pose coefficients; ARCTIC carries axis-angle45.  The
        # dataset pads them for collation and the runner groups/slices by
        # the recorded mano_* descriptor before smplx.MANO forward.
        mano_model_dir: str = str(ROOT / "dataset" / "arctic" / "data" / "body_models" / "mano")
        use_mano_reconstruction: bool = True
        apply_hand_perturb: bool = False
        # Object-centered protocol: perturb the hand wrist/root pose directly
        # (in the local hand-root frame), mutually exclusive with MANO/robot
        # pose noise when ``exclusive_hand_object_perturb`` is enabled.
        apply_hand_root_perturb: bool = False
        hand_root_rot_std_deg: float = 10.0
        hand_root_trans_std: float = 0.01
        hand_root_perturb_prob: float = 1.0
        val_apply_hand_root_perturb: bool = False
        # Robot Stage 3 samples use a separate URDF/FK q-space path. Only
        # hand joints are perturbed; the arm/base qpos remains fixed so the
        # stored hand-root frame stays unchanged.
        use_robot_reconstruction: bool = False
        apply_robot_perturb: bool = False
        robot_perturb_prob: float = 1.0
        # A mixed batch may contain MANO human samples and robot-FK samples;
        # the dataset/runner dispatches reconstruction per sample when this
        # flag is enabled.  It is intentionally opt-in so legacy homogeneous
        # configs retain their strict schema validation.
        mixed_hand_reconstruction: bool = False
        # Runtime robot q-space noise multiplier.  Values are calibrated per
        # HRDexDB robot domain so the resulting surface displacement is about
        # the requested geometric target rather than sharing one radian std.
        robot_perturb_noise_scale_by_domain: dict[str, float] = {}
        # Deprecated: q-space scales are the authoritative robot protocol;
        # this scalar is retained only so historical YAMLs remain loadable.
        robot_perturb_target_rms_m: float = 0.01
        # Target protocol for exclusive augmentation: 40% hand-only,
        # 40% object-only, and 20% clean.  The tuple is normalized and
        # validated by the dataset; hand/object perturbations are never
        # compounded in one sample.
        exclusive_perturb_mode_probs: tuple[float, float, float] = (0.4, 0.4, 0.2)
        # When enabled, each sample is assigned to either the hand-noise path
        # or the object-noise path with a stable per-frame gate.  This avoids
        # presenting compounded hand+object errors during training.
        exclusive_hand_object_perturb: bool = False
        # Fix #5 (docs/指导.md): decouple PCA coefficients from axis-angle
        # by giving each parameterisation its own (scale, clip) pair.
        # The PCA path additionally supports per-dim std from the training
        # set via ``hand_pca_train_std_per_dim`` (length 24, GRAB only);
        # when absent we use the scalar hand_pca_std directly and ignore
        # hand_pca_noise_scale. hand_pca_noise_clip always clips the standard
        # normal sample before scaling, so its unit is effective std multiples.
        hand_perturb_prob: float = 0.8
        hand_pca_std: float = 0.5
        hand_pca_noise_scale: float = 0.10
        hand_pca_noise_clip: float = 3.0
        hand_axis_angle_std_rad: float = 0.05
        hand_axis_angle_clip_rad: float = 0.15
        hand_pca_train_std_per_dim: tuple[float, ...] | None = None
        # Dataset/descriptor-specific, geometry-calibrated noise profiles.
        # Keys are normalized dataset IDs (grab/arctic/contactpose), values
        # are JSON files emitted by tools/calibrate_mano_geometry_noise.py.
        hand_geometry_calibration_paths: dict[str, str] = {}

        # Keep only frames whose clean hand/object closest-point distance is
        # within the interaction window.  This is a frame filter, not merely
        # a candidate-object mask; it is applied before object sampling.
        filter_non_interacting_frames: bool = False
        interaction_max_distance_m: float = 0.05
        hand_geometry_noise_scale: float = 1.0
        hand_geometry_noise_required: bool = False
        # Fix #6: number of FPS-sampled hand proxy face indices. The
        # indices are computed once on the canonical MANO face centres
        # and shared by the whole process; the runtime just does an
        # ``index_select`` on the hand input.
        hand_proxy_face_count: int = 256
        # Optional fixed proxy indices into the stored 1538 hand points.
        # This decouples runtime object resampling from MANO reconstruction
        # and MANO model assets for clean-geometry datasets such as OakInk.
        # When use_mano_reconstruction=False and runtime_resample_object=True,
        # this path is required and validated fail-fast by the runner.
        stored_hand_proxy_indices_path: str | None = None
        runtime_resample_object: bool = True
        # Deprecated compatibility fields. Runtime sampling no longer uses
        # near/global quotas or cdist proxies, but old experiment YAMLs may
        # still contain these keys.
        runtime_near_pool_points: int = 1024
        runtime_near_obj_points: int = 384
        runtime_cdist_chunk_size: int = 512

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
        # Optional multi-domain input. The task loader creates one dataset per
        # entry and applies an equal-domain sampler instead of frame-count
        # weighting the concatenated directory.
        domain_paths: list[dict[str, str]] = []
        group_val_by_sequence = True
        sequence_locality_shuffle = True
        blacklist_path = None
        # Optional persistent file/frame-count index. Keep it outside the
        # dataset tree so NAS data stays read-only.
        cache_index_path = None
        # Optional uncompressed NPZ sidecar tree. Source Stage 3 remains the
        # canonical file list; matching relative cache files only replace the
        # array-read path.
        array_cache_path = None
        array_cache_required = False

    class train(TaskConfig.train):
        amp = False
        diagnostic_every_steps = 20
        # v2.1 ablation: select checkpoints on the unbiased random-edge QFL
        # (the contact auxiliary stream is biased by construction and must
        # NOT be the model-selection metric).
        metric_for_best = "val_clean/cross_edge_random_qfl"
        lower_is_better = True
