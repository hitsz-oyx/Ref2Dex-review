"""Inspire residual task with a frozen DExplore teacher and Cm residual."""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.base.vec_task import VecTask
from .contract import (QUERY_LINKS, MIMIC_NATIVE, coupled_finger_bounds, native_sim_indices,
                       native_to_sim, pose_matrix, sim_to_native)
from .action_mapping import compose_physical_residual, compose_reference_residual
from .base_policy import ACTION_DIM, OBSERVATION_DIM, InspireDExplorePolicy
from .dexplore_observation import build_dexplore_observation
from .reference_provider import ReferenceProvider, RetargetedReferenceProvider, validate_reference_usage
from .cm_geometry import SurfaceGeometry
from src.task.CmDecoderv2.kinematics import InspireKinematics
from src.task.CmResidual.cm_v2_adapter import (CONTEXT_DIM as CMV2_CONTEXT_DIM,
                                                MODEL_CONFIG as CMV2_MODEL_CONFIG,
                                                SCHEMA as CMV2_SCHEMA, FrozenCmv2Adapter)
from src.task.CmResidual.cm_v2_action_evaluator import (
    Cmv2ActionEvaluator, build_nominal_hand_sweep, compose_nominal_targets,
    effect_metrics, object_pose_delta_to_xi)
from src.task.CmResidual.cm_buffer import CmTransitionBuffer, SCHEMA as CM_BUFFER_SCHEMA


DEXPLORE_KEY_BODIES = (
    "hand_base_link", "index_proximal", "index_intermediate", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_tip",
    "ring_proximal", "ring_intermediate", "ring_tip",
    "thumb_proximal_base", "thumb_intermediate", "thumb_tip",
)
DEXPLORE_CONTACT_BODIES = (
    "index_intermediate", "middle_intermediate", "pinky_intermediate",
    "ring_intermediate", "thumb_distal",
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CmResidual(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        self.cfg = cfg
        self.retargeted_base = cfg["basePolicy"]["mode"] == "retargeted_reference"
        if cfg["basePolicy"]["mode"] not in ("dexplore", "retargeted_reference"):
            raise ValueError("Unsupported CmResidual base policy mode")
        self.teacher = None if self.retargeted_base else InspireDExplorePolicy(
            cfg["basePolicy"]["checkpoint"], sim_device,
            cfg["basePolicy"].get("checkpointSha256"))
        asset = cfg["env"]["asset"]
        reference_cfg = cfg["reference"]
        object_name = asset.get("objectAssetFileName", "airplane.urdf")
        object_mesh = Path(asset["objectAssetRoot"]) / "objects" / Path(object_name).stem / f"{Path(object_name).stem}.obj"
        provider = RetargetedReferenceProvider if self.retargeted_base else ReferenceProvider
        self.reference = provider(
            reference_cfg["path"], sim_device, reference_cfg.get("sha256"),
            source_tensor=reference_cfg["sourceTensor"],
            source_sha256=reference_cfg["sourceTensorSha256"],
            frame_start=int(reference_cfg["frameStart"]),
            frame_end=int(reference_cfg["frameEnd"]),
            object_mesh=object_mesh,
            **({"urdf_path": Path(asset["assetRoot"]) / asset["assetFileName"]}
               if self.retargeted_base else {}))
        self.reference_usage = validate_reference_usage(
            self.reference.training_eligible,
            reference_cfg.get("allowIneligibleFor", ""))
        self.retargeted_mimic_scales = (tuple(float(self.reference.metadata["mimic_mapping"][str(i)])
                                              for i in MIMIC_NATIVE) if self.retargeted_base else ())
        self.base_observation_dim = 68 if self.retargeted_base else OBSERVATION_DIM
        self.use_oi_cm_context = bool(cfg["basePolicy"].get("useOiCmContext", True))
        self.use_cmv2_context = bool(cfg["basePolicy"].get("useCmv2Context", False))
        self.use_cmv2_action_evaluator = bool(
            cfg["basePolicy"].get("useCmv2ActionEvaluator", False))
        if self.use_oi_cm_context and (self.use_cmv2_context or self.use_cmv2_action_evaluator):
            raise ValueError("Select only one Cm context version")
        if self.use_cmv2_context and self.use_cmv2_action_evaluator:
            raise ValueError("Select either Cmv2 observation context or action evaluator")
        self.cm_dim = int(cfg["basePolicy"].get("cmFeatureDim", 32))
        self.cm_slots = int(cfg["basePolicy"].get("cmNumSlots", 16))
        self.cm_context_dim = (
            self.cm_slots * self.cm_dim + self.cm_slots * 3 + 3
            if self.use_oi_cm_context else CMV2_CONTEXT_DIM if self.use_cmv2_context else 0)
        residual_cfg = cfg["residual"]
        self.residual_translation_scale = float(residual_cfg["translationScaleM"])
        self.residual_rotation_scale = float(residual_cfg["rotationScaleRad"])
        self.residual_finger_scale = float(residual_cfg["fingerScaleRad"])
        self.max_episode_length = int(cfg["env"].get("episodeLength", 2000))
        start_cfg = cfg.get("referenceStart", {})
        self.reference_start_mode = str(start_cfg.get("mode", "fixed_zero"))
        self.reference_window_length = int(start_cfg.get("windowLength", self.max_episode_length))
        reward_cfg = cfg.get("referenceTrackingReward", {})
        self.reward_mode = str(reward_cfg.get("mode", "lift_distance"))
        if self.retargeted_base:
            if self.reference_start_mode not in ("fixed_zero", "random_uniform"):
                raise ValueError("Retargeted referenceStart.mode must be fixed_zero or random_uniform")
            if not 1 <= self.reference_window_length <= self.reference.length - 1:
                raise ValueError("Retargeted referenceStart.windowLength is outside the reference transitions")
            if self.max_episode_length != self.reference_window_length:
                raise ValueError("Retargeted episodeLength must equal referenceStart.windowLength")
        elif self.reference_start_mode != "fixed_zero" or self.reward_mode != "lift_distance":
            raise ValueError("Reference windows and tracking reward require retargeted_reference base")
        if self.reward_mode not in ("lift_distance", "reference_transition"):
            raise ValueError("Unsupported CmResidual reward mode")
        if self.reward_mode == "reference_transition":
            required = ("positionScaleM", "rotationScaleRad", "positionWeight",
                        "rotationWeight", "transitionPositionWeight", "transitionRotationWeight")
            if not all(name in reward_cfg for name in required):
                raise ValueError("Reference tracking reward configuration is incomplete")
            self.reference_tracking_reward = {name: float(reward_cfg[name]) for name in required}
            if (self.reference_tracking_reward["positionScaleM"] <= 0 or
                    self.reference_tracking_reward["rotationScaleRad"] <= 0 or
                    any(self.reference_tracking_reward[name] < 0 for name in required[2:])):
                raise ValueError("Reference tracking reward scales/weights must be non-negative with positive scales")
        else:
            self.reference_tracking_reward = {}
        cfg["env"].update(numObservations=self.base_observation_dim + self.cm_context_dim, numActions=ACTION_DIM, numStates=0,
                          episodeLength=self.max_episode_length)
        self.initial_native = self.reference.robot_q[0].clone()
        self.initial_native_velocity = self.reference.robot_dq[0].clone()
        self.initial_object = self.reference.object_state[0, :7].clone()
        self.initial_object_twist = self.reference.object_state[0, 7:13].clone()
        self.initial_table = self.reference.table_pose.clone()
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless,
                         virtual_screen_capture, force_render)
        self.actor_root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(-1, 13)
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, 18, 2)
        self.rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)
        self.contact_force = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(self.num_envs, self.num_bodies, 3)
        self.dof_pos, self.dof_vel = self.dof_state[..., 0], self.dof_state[..., 1]
        self._refresh_state()
        self.initial_root_states = self.actor_root_state.clone()
        self.hand_indices = torch.as_tensor(self.hand_indices, device=self.device, dtype=torch.int32)
        self.object_indices = torch.as_tensor(self.object_indices, device=self.device, dtype=torch.int32)
        self.table_indices = torch.as_tensor(self.table_indices, device=self.device, dtype=torch.int32)
        self.sim_indices = torch.as_tensor(self.sim_indices, device=self.device, dtype=torch.long)
        self.query_indices = torch.as_tensor(self.query_indices, device=self.device, dtype=torch.long)
        self.tip_indices = torch.as_tensor(self.tip_indices, device=self.device, dtype=torch.long)
        self.key_body_indices = torch.as_tensor(self.key_body_indices, device=self.device, dtype=torch.long)
        self.contact_body_indices = torch.as_tensor(self.contact_body_indices, device=self.device, dtype=torch.long)
        (self.initial_body_pos, self.initial_body_rot,
         self.initial_body_vel, self.initial_body_ang_vel) = self.reference.reset_body_state(
             Path(asset["assetRoot"]) / asset["assetFileName"], self.hand_body_names)
        initial_query_state = torch.cat((self.initial_body_pos, self.initial_body_rot), dim=-1).index_select(
            0, self.query_indices)
        self.initial_links = pose_matrix(initial_query_state)
        self.native_lower = sim_to_native(torch.as_tensor(self.sim_lower, device=self.device), self.sim_indices)
        self.native_upper = sim_to_native(torch.as_tensor(self.sim_upper, device=self.device), self.sim_indices)
        self.q_lower, self.q_upper = coupled_finger_bounds(self.native_lower, self.native_upper)
        self.native_targets = torch.zeros((self.num_envs, 18), device=self.device)
        self.sim_targets = torch.zeros_like(self.native_targets)
        self.base_q = torch.zeros((self.num_envs, 6), device=self.device)
        self.base_wrist = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        self.applied_wrist = self.base_wrist.clone()
        self.base_action = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self.residual_requested_delta = torch.zeros_like(self.native_targets)
        self.residual_applied_delta = torch.zeros_like(self.native_targets)
        self.residual_saturation = torch.zeros_like(self.native_targets)
        self.residual_authority_limited = torch.zeros_like(self.native_targets)
        self.fresh_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.initial_object_z = torch.full((self.num_envs,), self.initial_object[2].item(), device=self.device)
        self.reference_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reference_start_index = torch.zeros_like(self.reference_index)
        self.previous_object_pose = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        self.reference_object_rotation_error_rad = torch.zeros(self.num_envs, device=self.device)
        self.reference_object_transition_translation_error_m = torch.zeros(self.num_envs, device=self.device)
        self.reference_object_transition_rotation_error_rad = torch.zeros(self.num_envs, device=self.device)
        asset = self.cfg["env"]["asset"]
        hand_urdf = Path(asset["assetRoot"]) / asset["assetFileName"]
        object_name = asset.get("objectAssetFileName", "airplane.urdf")
        object_urdf = Path(asset["objectAssetRoot"]) / object_name
        self.cm_context = torch.zeros((self.num_envs, self.cm_context_dim), device=self.device)
        self.current_contact_forces = torch.zeros((self.num_envs, 5, 3), device=self.device)
        self.cm_geometry = None
        self.oi_cm = None
        self.oi_cm_checkpoint = None
        self.oi_cm_checkpoint_sha256 = None
        self.cmv2 = None
        self.cmv2_action_evaluator = None
        cm_buffer_cfg = cfg.get("cmBuffer", {})
        self.cm_buffer_enabled = bool(cm_buffer_cfg.get("enabled", False))
        self.cm_buffer = None
        self.cm_buffer_pending = None
        self.cm_buffer_prediction_translation_error_m = torch.zeros(self.num_envs, device=self.device)
        self.cm_buffer_prediction_rotation_error_rad = torch.zeros(self.num_envs, device=self.device)
        if self.cm_buffer_enabled and (not self.retargeted_base or not self.use_cmv2_action_evaluator):
            raise ValueError("CmBuffer requires retargeted_reference with Cmv2 action evaluator")
        if self.use_oi_cm_context or self.use_cmv2_context or self.use_cmv2_action_evaluator:
            self.cm_geometry = SurfaceGeometry(
                hand_urdf=hand_urdf, object_urdf=object_urdf, query_links=QUERY_LINKS,
                object_count=int(cfg["basePolicy"].get("cmObjectPoints", 1024)),
                hand_count=int(cfg["basePolicy"].get("cmHandPoints", 1538)), device=self.device)
            if self.use_cmv2_context:
                if (cfg["basePolicy"].get("cmv2Schema") != CMV2_SCHEMA or
                        dict(cfg["basePolicy"].get("cmv2ModelConfig", {})) != CMV2_MODEL_CONFIG):
                    raise ValueError("Cmv2 schema/model configuration mismatch")
                if (int(cfg["basePolicy"].get("cmObjectPoints", 1024)),
                        int(cfg["basePolicy"].get("cmHandPoints", 1538))) != (1024, 1538):
                    raise ValueError("Cmv2 requires 1024 object and 1538 hand points")
                self.cmv2 = FrozenCmv2Adapter(
                    cfg["basePolicy"]["cmv2Checkpoint"],
                    cfg["basePolicy"]["cmv2CheckpointSha256"], self.device)
            else:
                if self.use_cmv2_action_evaluator:
                    # The action evaluator loads Cmv2 lazily after the physical task
                    # has been constructed, so a static config can be resolved without
                    # a user checkpoint.  It never appends Cmv2 features to obs.
                    pass
                else:
                    oi_cm_path = Path(str(cfg["basePolicy"].get("oiCmCheckpoint", ""))).expanduser().resolve()
                    self.oi_cm_checkpoint = str(oi_cm_path)
                    self.oi_cm_checkpoint_sha256 = sha256(oi_cm_path) if oi_cm_path.is_file() else ""
                    self.oi_cm = self._load_oi_cm(
                        oi_cm_path, cfg["basePolicy"].get("oiCmCheckpointSha256"),
                        feature_dim=self.cm_dim, num_slots=self.cm_slots,
                        object_points=int(cfg["basePolicy"].get("cmObjectPoints", 1024)),
                        hand_points=int(cfg["basePolicy"].get("cmHandPoints", 1538)))
        if self.cm_buffer_enabled:
            output_dir = str(cm_buffer_cfg.get("outputDir", "")).strip()
            if not output_dir:
                raise ValueError("CmBuffer enabled requires outputDir")
            self.cm_buffer = CmTransitionBuffer(
                output_dir, rank=int(cm_buffer_cfg.get("rank", 0)),
                flush_every=int(cm_buffer_cfg.get("flushEvery", 2048)),
                metadata={
                    "buffer_schema": CM_BUFFER_SCHEMA,
                    "cmv2_evaluator_schema": "cmv2_action_effect_v1",
                    "cmv2_checkpoint": str(cfg["basePolicy"]["cmv2Checkpoint"]),
                    "cmv2_checkpoint_sha256": str(cfg["basePolicy"]["cmv2CheckpointSha256"]),
                    "cmv2_model_config": dict(cfg["basePolicy"]["cmv2ModelConfig"]),
                    "reference_path": str(reference_cfg["path"]),
                    "reference_sha256": str(reference_cfg["sha256"]),
                    "hand_urdf": str(hand_urdf),
                    "hand_urdf_sha256": sha256(hand_urdf),
                    "object_mesh": str(object_mesh),
                    "object_mesh_sha256": sha256(object_mesh),
                    "control_dt_s": float(cfg["sim"]["dt"]) * int(cfg["env"]["controlFrequencyInv"]),
                    "residual_scale": {"translation_m": self.residual_translation_scale,
                                       "rotation_rad": self.residual_rotation_scale,
                                       "finger_rad": self.residual_finger_scale},
                    "sample_semantics": "executed_action_only_pre_to_post_physics_transition",
                })
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_lift = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = 0
        self.successful_episodes = 0
        self._refresh_state()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

    def build_cmv2_action_evaluator(self):
        """Lazily construct the frozen Cmv2 action evaluator after task setup."""
        if not self.use_cmv2_action_evaluator:
            raise RuntimeError("Cmv2 action evaluator is not enabled in this task config")
        if self.cmv2_action_evaluator is None:
            base = self.cfg["basePolicy"]
            if (base.get("cmv2Schema") != CMV2_SCHEMA or
                    dict(base.get("cmv2ModelConfig", {})) != CMV2_MODEL_CONFIG):
                raise ValueError("Cmv2 schema/model configuration mismatch")
            self.cmv2 = FrozenCmv2Adapter(
                base["cmv2Checkpoint"], base["cmv2CheckpointSha256"], self.device)
            self.cmv2_action_evaluator = Cmv2ActionEvaluator(self.cmv2)
        return self.cmv2_action_evaluator

    def evaluate_cmv2_actions(self, residual_actions, *, desired_delta_xi=None,
                              desired_obj_flow=None, candidate_valid_mask=None):
        """Evaluate nominal one-step candidate residuals without using future state."""
        evaluator = self.build_cmv2_action_evaluator()
        residual_actions = residual_actions.to(self.device)
        self._refresh_state()
        native = sim_to_native(self.dof_pos, self.sim_indices)
        if self.retargeted_base:
            targets, target_details = compose_reference_residual(
                self.base_action[:, None, :].expand_as(residual_actions), residual_actions,
                self.native_lower, self.native_upper,
                translation_scale_m=self.residual_translation_scale,
                rotation_scale_rad=self.residual_rotation_scale,
                finger_scale_rad=self.residual_finger_scale,
                mimic_scales=self.retargeted_mimic_scales)
            feasible = ~target_details["saturation"].bool().any(dim=-1)
            candidate_valid_mask = (feasible if candidate_valid_mask is None else
                                    candidate_valid_mask.to(self.device).bool() & feasible)
        else:
            targets, _ = compose_nominal_targets(
                self.base_action, residual_actions, native, self.native_lower, self.native_upper,
                translation_scale_m=self.residual_translation_scale,
                rotation_scale_rad=self.residual_rotation_scale,
                finger_scale_rad=self.residual_finger_scale)
        kinematics = InspireKinematics(self.cfg["env"]["asset"]["assetRoot"] + "/" +
                                       self.cfg["env"]["asset"]["assetFileName"])
        sweep = build_nominal_hand_sweep(self.cm_geometry, kinematics, native, targets)
        object_state = self.actor_root_state[self.object_indices.long()]
        object_pose = pose_matrix(object_state[:, :7])
        object_points, object_normals = self.cm_geometry.object(object_pose)
        return evaluator.evaluate(
            object_points, object_normals, sweep, candidate_actions=residual_actions,
            candidate_valid_mask=candidate_valid_mask,
            desired_delta_xi=desired_delta_xi,
            desired_obj_flow=desired_obj_flow)

    @staticmethod
    def _load_oi_cm(checkpoint, expected_sha256, *, feature_dim, num_slots,
                    object_points, hand_points):
        checkpoint = Path(checkpoint)
        if not str(checkpoint).strip():
            raise ValueError("CmResidual requires basePolicy.oiCmCheckpoint for semantic OI-Cm features")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"OI-Cm checkpoint not found: {checkpoint}")
        digest = sha256(checkpoint)
        if expected_sha256 and digest != str(expected_sha256).lower():
            raise ValueError(f"OI-Cm checkpoint SHA256 mismatch: {digest} != {expected_sha256}")
        from types import SimpleNamespace
        from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_cfg = payload.get("config", {})
        meta_values = checkpoint_cfg.get("meta", {})
        expected = {
            "feature_dim": int(feature_dim),
            "num_cm_tokens": int(num_slots),
            "num_obj_points": int(object_points),
            "num_hand_points": int(hand_points),
        }
        for name, value in expected.items():
            if int(meta_values.get(name, -1)) != value:
                raise ValueError(
                    f"OI-Cm {name} mismatch: {meta_values.get(name)!r} != {value}")
        meta_values = dict(meta_values)
        scale_manifest = Path(str(meta_values.get("scale_manifest_path", ""))).expanduser()
        if not scale_manifest.is_absolute():
            repository_root = Path(__file__).resolve().parents[5]
            scale_manifest = repository_root / scale_manifest
        if not scale_manifest.is_file():
            raise FileNotFoundError(f"OI-Cm scale manifest not found: {scale_manifest}")
        meta_values["scale_manifest_path"] = str(scale_manifest.resolve())
        meta = SimpleNamespace(**meta_values)
        meta.modification_version = checkpoint_cfg.get("modification_version", "V1.3")
        model = ObjectInteractionCmModel(meta).eval()
        model.load_state_dict(payload["model"], strict=True)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return model.to("cpu")

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        plane_cfg = self.cfg["env"]["plane"]
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = float(plane_cfg["staticFriction"])
        plane.dynamic_friction = float(plane_cfg["dynamicFriction"])
        plane.restitution = float(plane_cfg["restitution"])
        self.gym.add_ground(self.sim, plane)
        self._create_envs()

    @staticmethod
    def _gym_pose(pose):
        values = pose.detach().cpu().tolist()
        result = gymapi.Transform()
        result.p = gymapi.Vec3(*values[:3])
        result.r = gymapi.Quat(*values[3:7])
        return result

    def _create_envs(self):
        asset = self.cfg["env"]["asset"]
        object_root = Path(asset["objectAssetRoot"]).resolve()
        for name, expected in self.cfg["env"].get("assetSha256", {}).items():
            if sha256(object_root / name) != expected:
                raise ValueError(f"Scene asset checksum mismatch: {name}")
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        options.disable_gravity = True
        options.collapse_fixed_joints = False
        options.max_angular_velocity = 100.0
        options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        options.angular_damping = 0.01
        hand = self.gym.load_asset(self.sim, str(Path(asset["assetRoot"]).resolve()), asset["assetFileName"], options)
        names = list(self.gym.get_asset_dof_names(hand))
        self.sim_indices = native_sim_indices(names)
        self.hand_body_names = tuple(self.gym.get_asset_rigid_body_names(hand))
        self.hand_num_bodies = self.gym.get_asset_rigid_body_count(hand)
        self.query_indices = [self.gym.find_asset_rigid_body_index(hand, name) for name in QUERY_LINKS]
        self.tip_indices = [self.gym.find_asset_rigid_body_index(hand, name) for name in self.cfg["env"]["tipLinks"]]
        self.key_body_indices = [self.gym.find_asset_rigid_body_index(hand, name) for name in DEXPLORE_KEY_BODIES]
        self.contact_body_indices = [self.gym.find_asset_rigid_body_index(hand, name) for name in DEXPLORE_CONTACT_BODIES]
        self.tracking_root_body_index = self.gym.find_asset_rigid_body_index(hand, "link6")
        self.root_body_index = self.gym.find_asset_rigid_body_index(hand, "hand_base_link")
        required = (self.query_indices + self.tip_indices + self.key_body_indices +
                    self.contact_body_indices + [self.tracking_root_body_index, self.root_body_index])
        if min(required) < 0:
            raise ValueError("Missing Inspire body required by the DExplore observation contract")
        if (self.tracking_root_body_index, self.root_body_index) != (6, 7):
            raise ValueError(
                "Released DExplore observation hard-codes link6/hand_base at body indices 6/7; "
                f"asset returned {self.tracking_root_body_index}/{self.root_body_index}")
        props = self.gym.get_asset_dof_properties(hand)
        props["driveMode"][:] = gymapi.DOF_MODE_POS
        wrist_stiffness = (float(self.cfg["env"].get("wristStiffness", 200.0))
                           if self.retargeted_base else 200.0)
        wrist_damping = (float(self.cfg["env"].get("wristDamping", 20.0))
                         if self.retargeted_base else 20.0)
        if not (np.isfinite(wrist_stiffness) and np.isfinite(wrist_damping)
                and wrist_stiffness > 0 and wrist_damping > 0):
            raise ValueError("Invalid wrist PD gains")
        props["stiffness"][self.sim_indices] = [wrist_stiffness] * 6 + [100.0] * 12
        props["damping"][self.sim_indices] = [wrist_damping] * 6 + [10.0] * 12
        props["velocity"][:] = 7.0
        self.sim_lower, self.sim_upper = props["lower"].copy(), props["upper"].copy()
        initialize_dofs_at_creation = bool(asset.get("initializeDofsAtCreation", False))
        initial_sim_position = np.empty(18, dtype=np.float32)
        initial_sim_velocity = np.empty(18, dtype=np.float32)
        initial_sim_position[self.sim_indices] = self.initial_native.detach().cpu().numpy()
        initial_sim_velocity[self.sim_indices] = self.initial_native_velocity.detach().cpu().numpy()
        obj_options = gymapi.AssetOptions()
        obj_options.density = 20
        obj_options.angular_damping = obj_options.linear_damping = 0.01
        obj_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        obj_options.vhacd_enabled = True
        obj_options.vhacd_params.max_convex_hulls = 20
        obj_options.vhacd_params.max_num_vertices_per_ch = 16
        obj_options.vhacd_params.resolution = 50000
        obj = self.gym.load_asset(self.sim, str(object_root), asset.get("objectAssetFileName", "airplane.urdf"), obj_options)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_options.density = 100000
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table = self.gym.load_asset(self.sim, str(object_root), "table.urdf", table_options)
        self.num_bodies = sum(self.gym.get_asset_rigid_body_count(a) for a in (hand, obj, table))
        self.hand_indices, self.object_indices, self.table_indices, self.envs = [], [], [], []
        spacing = float(self.cfg["env"]["envSpacing"])
        for index in range(self.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), int(np.sqrt(self.num_envs)))
            h = self.gym.create_actor(env, hand, gymapi.Transform(), "inspire", index, 1, 0)
            self.gym.set_actor_dof_properties(env, h, props)
            if initialize_dofs_at_creation:
                dof_state = np.zeros(18, dtype=gymapi.DofState.dtype)
                dof_state["pos"] = initial_sim_position
                dof_state["vel"] = initial_sim_velocity
                self.gym.set_actor_dof_states(env, h, dof_state, gymapi.STATE_ALL)
                self.gym.set_actor_dof_position_targets(env, h, initial_sim_position)
            # Preserve the released Inspire shape-index/body-name lookup
            # exactly; the historical indexing is part of checkpoint physics.
            shapes = self.gym.get_actor_rigid_shape_properties(env, h)
            body_names = self.gym.get_actor_rigid_body_names(env, h)
            for shape_index, shape in enumerate(shapes):
                name = body_names[shape_index]
                shape.filter = 3 if (("thumb" in name and "distal" in name)
                                     or ("thumb" not in name and "intermediate" in name)) else 2
            self.gym.set_actor_rigid_shape_properties(env, h, shapes)
            object_actor_name = str(asset.get("objectActorName", "airplane"))
            o = self.gym.create_actor(env, obj, self._gym_pose(self.initial_object), object_actor_name, index, 0, 1)
            t = self.gym.create_actor(env, table, self._gym_pose(self.initial_table), "table", index, 1, 2)
            self.hand_indices.append(self.gym.get_actor_index(env, h, gymapi.DOMAIN_SIM))
            self.object_indices.append(self.gym.get_actor_index(env, o, gymapi.DOMAIN_SIM))
            self.table_indices.append(self.gym.get_actor_index(env, t, gymapi.DOMAIN_SIM))
            self.envs.append(env)

    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def actual_link_poses(self):
        return pose_matrix(self.rigid_body_state[:, self.query_indices, :7])

    def _sample_reference_start(self, count: int) -> torch.Tensor:
        if not self.retargeted_base or self.reference_start_mode == "fixed_zero":
            return torch.zeros(count, dtype=torch.long, device=self.device)
        max_start = self.reference.length - 1 - self.reference_window_length
        return torch.randint(max_start + 1, (count,), dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device)
        roots = torch.cat([self.hand_indices[env_ids], self.object_indices[env_ids]]).contiguous()
        hand_ids = self.hand_indices[env_ids]
        object_ids = self.object_indices[env_ids]
        self.actor_root_state[hand_ids.long()] = self.initial_root_states[hand_ids.long()]
        start_indices = self._sample_reference_start(len(env_ids))
        initial_native, initial_velocity, initial_object = self.reference.reset_state(len(env_ids), start_indices)
        self.actor_root_state[object_ids.long()] = initial_object
        self.dof_pos[env_ids] = native_to_sim(initial_native, self.sim_indices)
        self.dof_vel[env_ids] = native_to_sim(initial_velocity, self.sim_indices)
        self.native_targets[env_ids] = initial_native
        self.sim_targets[env_ids] = self.dof_pos[env_ids]
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.residual_requested_delta[env_ids] = 0
        self.residual_applied_delta[env_ids] = 0
        self.residual_saturation[env_ids] = 0
        self.residual_authority_limited[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.episode_max_lift[env_ids] = 0
        self.initial_object_z[env_ids] = initial_object[:, 2]
        self.fresh_reset[env_ids] = True
        self.reference_index[env_ids] = start_indices
        self.reference_start_index[env_ids] = start_indices
        self.previous_object_pose[env_ids] = pose_matrix(initial_object[:, :7])
        hand_ids = hand_ids.contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.actor_root_state), gymtorch.unwrap_tensor(roots), len(roots))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_ids), len(hand_ids))
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.sim_targets), gymtorch.unwrap_tensor(hand_ids), len(hand_ids))

    def pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite residual action")
        actions = actions.to(self.device).clamp(-1, 1)
        self.previous_object_pose.copy_(pose_matrix(
            self.actor_root_state[self.object_indices.long(), :7]))
        self.prev_actions.copy_(actions)
        current = sim_to_native(self.dof_pos, self.sim_indices)
        composer = compose_reference_residual if self.retargeted_base else compose_physical_residual
        args = (self.base_action, actions, self.native_lower, self.native_upper) if self.retargeted_base else (
            self.base_action, actions, current, self.native_lower, self.native_upper)
        options = dict(translation_scale_m=self.residual_translation_scale,
                       rotation_scale_rad=self.residual_rotation_scale,
                       finger_scale_rad=self.residual_finger_scale)
        if self.retargeted_base:
            options["mimic_scales"] = self.retargeted_mimic_scales
        targets, details = composer(*args, **options)
        self.residual_requested_delta.copy_(details["requested_delta"])
        self.residual_applied_delta.copy_(details["applied_delta"])
        self.residual_saturation.copy_(details["saturation"])
        self.residual_authority_limited.copy_(details.get("authority_limited", torch.zeros_like(self.residual_saturation)))
        self.native_targets.copy_(targets)
        self.sim_targets.copy_(native_to_sim(targets, self.sim_indices))
        if self.cm_buffer_enabled:
            reference_next = (self.reference_index + 1).clamp_max(self.reference.length - 1)
            reference_delta = object_pose_delta_to_xi(
                pose_matrix(self.reference.object_state.index_select(0, self.reference_index)[:, :7]),
                pose_matrix(self.reference.object_state.index_select(0, reference_next)[:, :7]))
            prediction = self.evaluate_cmv2_actions(
                actions[:, None, :], desired_delta_xi=reference_delta)
            self.cm_buffer_pending = {
                "q_pre": current.clone(),
                "dq_pre": sim_to_native(self.dof_vel, self.sim_indices).clone(),
                "object_pose_pre": self.actor_root_state[self.object_indices.long(), :7].clone(),
                "residual_action": actions.clone(),
                "reference_target": self.base_action.clone(),
                "pd_target": targets.clone(),
                "applied_delta": details["applied_delta"].clone(),
                "authority_limited": details.get("authority_limited", torch.zeros_like(targets)).clone(),
                "reference_index": self.reference_index.clone(),
                "reference_start_index": self.reference_start_index.clone(),
                "window_progress": self.progress_buf.clone(),
                "reference_delta_xi": reference_delta.clone(),
                "predicted_delta_xi": prediction["predicted_delta_xi"][:, 0].clone(),
                "predicted_effect_score": prediction["effect_score"][:, 0].clone(),
                "candidate_valid": prediction["candidate_valid_mask"][:, 0].clone(),
            }
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_targets))

    def _append_cm_buffer_transition(self) -> None:
        """Attach the post-physics label to the action-conditioned pre-state record."""
        if not self.cm_buffer_enabled or self.cm_buffer_pending is None:
            return
        pending = self.cm_buffer_pending
        object_pose_post = self.actor_root_state[self.object_indices.long(), :7].clone()
        actual_delta = object_pose_delta_to_xi(
            pose_matrix(pending["object_pose_pre"]), pose_matrix(object_pose_post))
        predicted = pending["predicted_delta_xi"]
        metrics = effect_metrics(predicted, actual_delta)
        self.cm_buffer_prediction_translation_error_m.copy_(metrics["translation_error_m"])
        self.cm_buffer_prediction_rotation_error_rad.copy_(metrics["rotation_error_rad"])
        record = {
            **pending,
            "q_post": sim_to_native(self.dof_pos, self.sim_indices).clone(),
            "dq_post": sim_to_native(self.dof_vel, self.sim_indices).clone(),
            "object_pose_post": object_pose_post,
            "actual_delta_xi": actual_delta,
        }
        self.cm_buffer.append(record)
        self.cm_buffer_pending = None

    def compute_observations(self):
        self._refresh_state()
        native = sim_to_native(self.dof_pos, self.sim_indices)
        hand_state = self.rigid_body_state[:, :self.hand_num_bodies]
        body_pos = hand_state[..., :3]
        body_rot = hand_state[..., 3:7]
        body_vel = hand_state[..., 7:10]
        body_ang_vel = hand_state[..., 10:13]
        body_contact = self.contact_force[:, :self.hand_num_bodies]
        query_state = torch.cat((body_pos, body_rot), dim=-1).index_select(1, self.query_indices)
        links = pose_matrix(query_state)
        object_state = self.actor_root_state[self.object_indices.long()]
        obj = pose_matrix(object_state[:, :7])
        self.base_q.copy_(native[:, :6])
        self.base_wrist.copy_(links[:, 0])
        contact = body_contact.index_select(1, self.contact_body_indices)
        self.current_contact_forces.copy_(contact)

        idx = self.reference_index
        if self.retargeted_base:
            next_idx = (idx + 1).clamp_max(self.reference.length - 1)
            target = self.reference.robot_q.index_select(0, next_idx)
            ref_object = self.reference.object_state.index_select(0, idx)
            next_object = self.reference.object_state.index_select(0, next_idx)
            base_obs = torch.cat((native, sim_to_native(self.dof_vel, self.sim_indices),
                                  target - native, object_state[:, :7], next_object[:, :7]), dim=-1)
            self.base_action.copy_(target)
            self.reference_root_position_error_m = (
                body_pos[:, self.root_body_index] - self.reference.wrist_pose[idx, :3, 3]).norm(dim=-1)
            self.reference_object_position_error_m = (
                object_state[:, :3] - ref_object[:, :3]).norm(dim=-1)
            object_rotation = pose_matrix(object_state[:, :7])[:, :3, :3]
            reference_rotation = pose_matrix(ref_object[:, :7])[:, :3, :3]
            relative_rotation = object_rotation.transpose(-1, -2) @ reference_rotation
            cosine = (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
            self.reference_object_rotation_error_rad.copy_(torch.acos(cosine.clamp(-1.0, 1.0)))
            ref_contact = self.reference.frame(idx)[:, 168:184][:, [3, 6, 9, 12, 15]]
            self.reference_contact_occupancy = ref_contact.mean(dim=-1)
            tip_names = self.cfg["env"]["tipLinks"]
            tip_ref_idx = [self.reference.link_order.index(name) for name in tip_names]
            reference_tips = self.reference.link_pose[idx][:, tip_ref_idx, :3, 3]
        else:
            ref_now = self.reference.frame(idx)
            obs_1 = build_dexplore_observation(
                body_pos, body_rot, body_vel, body_ang_vel, body_contact,
                object_state, self.reference.frame(idx, 1), self.reference.object_points,
                self.key_body_indices, self.contact_body_indices,
                root_body_id=self.root_body_index,
                tracking_root_body_id=self.tracking_root_body_index)
            obs_16 = build_dexplore_observation(
                body_pos, body_rot, body_vel, body_ang_vel, body_contact,
                object_state, self.reference.frame(idx, 16), self.reference.object_points,
                self.key_body_indices, self.contact_body_indices,
                root_body_id=self.root_body_index,
                tracking_root_body_id=self.tracking_root_body_index)
            base_obs = torch.cat((obs_1, obs_16), dim=-1)
            self.reference_root_position_error_m = (
                body_pos[:, self.tracking_root_body_index] - ref_now[:, 4:7]).norm(dim=-1)
            self.reference_object_position_error_m = (
                object_state[:, :3] - ref_now[:, 106:109]).norm(dim=-1)
            ref_contact = ref_now[:, 168:184][:, [3, 6, 9, 12, 15]]
            self.reference_contact_occupancy = ref_contact.mean(dim=-1)
            reference_tips = ref_now[:, 119:167].view(self.num_envs, 16, 3)[:, [3, 6, 9, 12, 15]]
        actual_tips = body_pos.index_select(1, self.tip_indices)
        self.reference_tip_position_error_m = (actual_tips - reference_tips).norm(dim=-1).mean(dim=-1)
        self.reset_native_error_max = (native - self.initial_native).abs().amax(dim=-1)
        self.reset_wrist_position_error_m = (
            body_pos[:, self.root_body_index] - self.initial_body_pos[self.root_body_index]).norm(dim=-1)
        self.reset_object_position_error_m = (
            object_state[:, :3] - self.initial_object[:3]).norm(dim=-1)
        table_state = self.actor_root_state[self.table_indices.long()]
        self.reset_table_position_error_m = (
            table_state[:, :3] - self.initial_table[:3]).norm(dim=-1)
        if self.use_oi_cm_context or self.use_cmv2_context:
            obj_points, obj_normals = self.cm_geometry.object(obj)
            hand_points, hand_normals = self.cm_geometry.hand(links)
            hand_flow = self.cm_geometry.flow(hand_points, self.fresh_reset)
            if self.use_cmv2_context:
                delta_time_s = float(self.cfg["sim"]["dt"]) * int(self.cfg["env"]["controlFrequencyInv"])
                self.cm_context.copy_(self.cmv2(obj_points, obj_normals, hand_points,
                                                 hand_normals, hand_flow, delta_time_s))
            else:
                with torch.no_grad():
                    cm_out = self.oi_cm({"obj_points": obj_points.cpu(), "obj_normals": obj_normals.cpu(),
                                         "obj_valid_mask": torch.ones(obj_points.shape[:2], dtype=torch.bool),
                                         "hand_points": hand_points.cpu(), "hand_normals": hand_normals.cpu(),
                                         "hand_flow": hand_flow.cpu(),
                                         "hand_valid_mask": torch.ones(hand_points.shape[:2], dtype=torch.bool)})
                tokens = cm_out["cm_tokens"].to(self.device)
                anchors = cm_out["cm_anchor_pos"].to(self.device)
                effect = cm_out["pred_obj_flow"].mean(dim=1).to(self.device)
                self.cm_context.copy_(torch.cat((tokens.flatten(1), anchors.flatten(1), effect), dim=-1))
            obs = torch.cat((base_obs, self.cm_context), dim=-1)
        else:
            obs = base_obs
        if not torch.isfinite(obs).all():
            raise FloatingPointError("Non-finite actual-state observation")
        self.obs_buf.copy_(obs.clamp(-self.clip_obs, self.clip_obs))
        if not self.retargeted_base:
            self.base_action.copy_(self.teacher.act(base_obs))
        return self.obs_buf

    def compute_reward(self):
        env = self.cfg["env"]
        obj = self.actor_root_state[self.object_indices.long()]
        tips = self.rigid_body_state[:, self.tip_indices, :3]
        distance = (tips - obj[:, None, :3]).norm(dim=-1).mean(-1)
        lift = obj[:, 2] - self.initial_object_z
        if self.reward_mode == "reference_transition":
            current_object_pose = pose_matrix(obj[:, :7])
            previous_reference = self.reference.object_state.index_select(
                0, (self.reference_index - 1).clamp_min(0))[:, :7]
            current_reference = self.reference.object_state.index_select(
                0, self.reference_index)[:, :7]
            actual_transition = object_pose_delta_to_xi(self.previous_object_pose, current_object_pose)
            reference_transition = object_pose_delta_to_xi(
                pose_matrix(previous_reference), pose_matrix(current_reference))
            transition = effect_metrics(actual_transition, reference_transition)
            self.reference_object_transition_translation_error_m.copy_(transition["translation_error_m"])
            self.reference_object_transition_rotation_error_rad.copy_(transition["rotation_error_rad"])
            reward = self.reference_tracking_reward
            self.rew_buf[:] = -(
                reward["positionWeight"] * self.reference_object_position_error_m / reward["positionScaleM"] +
                reward["rotationWeight"] * self.reference_object_rotation_error_rad / reward["rotationScaleRad"] +
                reward["transitionPositionWeight"] * transition["translation_error_m"] / reward["positionScaleM"] +
                reward["transitionRotationWeight"] * transition["rotation_error_rad"] / reward["rotationScaleRad"] +
                float(env["actionPenaltyScale"]) * self.prev_actions.square().mean(-1))
        else:
            self.rew_buf[:] = (float(env["liftRewardScale"]) * lift -
                               float(env["distanceRewardScale"]) * distance -
                               float(env["actionPenaltyScale"]) * self.prev_actions.square().mean(-1))
        self.episode_return += self.rew_buf
        self.episode_max_lift = torch.maximum(self.episode_max_lift, lift)
        success = lift > float(env["successLift"])
        done = ((success & bool(env.get("terminateOnSuccess", True))) |
                (self.progress_buf >= self.max_episode_length))
        self.reset_buf.copy_(done.long())
        self.completed_episodes += int(done.sum().item())
        self.successful_episodes += int((success & done).sum().item())
        self._append_cm_buffer_transition()
        if self.cm_buffer_enabled and done.any():
            self.cm_buffer.flush()
        self.extras.pop("episode", None)
        if done.any():
            self.extras["episode"] = {"success": success[done].float(), "return": self.episode_return[done].clone(),
                                      "max_lift_m": self.episode_max_lift[done].clone()}
        self.extras.update(lift_mean=lift.mean(), tip_distance_mean=distance.mean(),
                          residual_rms=self.prev_actions.square().mean().sqrt(),
                          residual_translation_m_rms=self.residual_applied_delta[:, :3].square().mean().sqrt(),
                          residual_rotation_rad_rms=self.residual_applied_delta[:, 3:6].square().mean().sqrt(),
                          residual_finger_rad_rms=self.residual_applied_delta[:, [6, 8, 10, 12, 14, 15]].square().mean().sqrt(),
                          residual_target_delta_max=self.residual_applied_delta.abs().amax(),
                          residual_saturation_ratio=self.residual_saturation.mean(),
                          residual_authority_limited_ratio=self.residual_authority_limited.mean(),
                          contact_occupancy=(self.current_contact_forces.abs().amax(-1) > 0.1).float().mean(),
                          reference_contact_occupancy=self.reference_contact_occupancy.mean(),
                          reference_root_position_error_m=self.reference_root_position_error_m.mean(),
                          reference_object_position_error_m=self.reference_object_position_error_m.mean(),
                          reference_object_rotation_error_rad=self.reference_object_rotation_error_rad.mean(),
                          reference_object_transition_translation_error_m=self.reference_object_transition_translation_error_m.mean(),
                          reference_object_transition_rotation_error_rad=self.reference_object_transition_rotation_error_rad.mean(),
                          cm_buffer_prediction_translation_error_m=self.cm_buffer_prediction_translation_error_m.mean(),
                          cm_buffer_prediction_rotation_error_rad=self.cm_buffer_prediction_rotation_error_rad.mean(),
                          reference_tip_position_error_m=self.reference_tip_position_error_m.mean(),
                          reference_start_index_mean=self.reference_start_index.float().mean(),
                          reference_start_index_min=self.reference_start_index.min(),
                          reference_start_index_max=self.reference_start_index.max(),
                          success_fraction=success.float().mean(),
                          success_rate=self.successful_episodes / max(1, self.completed_episodes))

    def post_physics_step(self):
        self.fresh_reset[:] = False
        self.progress_buf += 1
        self.reference_index = (self.reference_index + 1).clamp_max(self.reference.length - 1)
        self.compute_observations()
        self.compute_reward()

    def step(self, actions):
        obs, reward, done, infos = super().step(actions)
        done = done.clone()
        infos = dict(infos)
        infos["time_outs"] = infos["time_outs"].clone()
        ids = done.nonzero(as_tuple=False).flatten()
        if len(ids):
            infos["terminal_observation"] = obs["obs"].clone()
            infos["terminal_wrist_pose"] = self.actual_link_poses()[:, 0].clone()
            infos["terminal_object_pose"] = self.actor_root_state[self.object_indices.long(), :7].clone()
            self.reset_idx(ids)
            self.compute_observations()
            obs["obs"] = self.obs_buf.clamp(-self.clip_obs, self.clip_obs).to(self.rl_device)
        return obs, reward, done, infos
