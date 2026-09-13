"""Minimal Inspire residual-control task for the Ref2Dex base-target contract.

This first task deliberately separates physics wiring from Cm inference.  The
default ``reference_frozen`` provider reads one coupled Inspire reference
trajectory and holds the base target fixed for each vectorized episode.  The
``decoder`` provider fails closed until a Cm-bank adapter is supplied; it must
not silently replace decoder output with a scripted target.
"""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from isaacgym import gymapi, gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import quat_from_angle_axis, quat_mul, tensor_clamp

from src.task.CmDecoderv2.rl.residual_contract import (
    INDEPENDENT_NATIVE,
    NATIVE_TO_URDF,
    OBSERVATION_DIM,
    expand_native_targets,
    native_to_urdf,
)


def _matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to Isaac Gym's x,y,z,w quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.asarray([x, y, z, w], dtype=np.float32)


class CmResidual(VecTask):
    """Inspire hand + cube task with frozen base target and 12-D residual action."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        env_cfg = cfg["env"]
        self.max_episode_length = int(env_cfg["episodeLength"])
        self.residual_q_scale = float(env_cfg["residualQScale"])
        self.residual_root_translation_scale = float(env_cfg["residualRootTranslationScale"])
        self.residual_root_rotation_scale = float(env_cfg["residualRootRotationScale"])
        self.action_penalty_scale = float(env_cfg["actionPenaltyScale"])
        self.distance_reward_scale = float(env_cfg["distanceRewardScale"])
        self.lift_reward_scale = float(env_cfg["liftRewardScale"])
        self.success_lift = float(env_cfg["successLift"])
        self.base_mode = str(cfg["basePolicy"]["mode"])
        if self.base_mode not in {"reference_frozen", "decoder_bank"}:
            raise ValueError(f"Unknown basePolicy.mode={self.base_mode!r}")

        self.reference_q, self.reference_wrist = self._load_reference(cfg["basePolicy"])
        self.num_reference_frames = int(self.reference_q.shape[0])
        env_cfg["numObservations"] = OBSERVATION_DIM
        env_cfg["numActions"] = 12
        env_cfg["numStates"] = 0
        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.actor_root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(-1, 13)
        self.dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, self.num_dof, 2)
        self.rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]
        self.prev_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.base_q = torch.zeros((self.num_envs, 6), device=self.device)
        self.base_wrist = torch.zeros((self.num_envs, 7), device=self.device)
        self.initial_object_z = torch.zeros(self.num_envs, device=self.device)
        self.hand_q_lower = torch.tensor(self.hand_dof_lower, device=self.device)
        self.hand_q_upper = torch.tensor(self.hand_dof_upper, device=self.device)
        self._refresh_state()

    @staticmethod
    def _load_reference(base_cfg):
        if str(base_cfg.get("mode", "reference_frozen")) == "decoder_bank":
            manifest_path = Path(str(base_cfg["decoderBankManifest"])).expanduser().resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Decoder bank manifest is required: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_name") != "ref2dex_cm_decoder_v2_rl_decoder_bank_v1":
                raise ValueError(f"Unsupported decoder bank schema: {manifest.get('schema_name')!r}")
            checkpoint_path = Path(str(base_cfg["decoderCheckpoint"])).expanduser().resolve()
            expected_sha = str(manifest.get("checkpoint_sha256", "")).lower()
            digest = hashlib.sha256()
            with checkpoint_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha:
                raise ValueError(f"Decoder checkpoint SHA256 mismatch: expected {expected_sha}, got {digest.hexdigest()}")
            q_path = Path(str(manifest["q_path"])).expanduser().resolve()
            wrist_path = Path(str(manifest["wrist_path"])).expanduser().resolve()
        else:
            q_path = Path(str(base_cfg["referenceQPath"])).expanduser().resolve()
            wrist_path = Path(str(base_cfg["referenceWristPath"])).expanduser().resolve()
        if not q_path.is_file() or not wrist_path.is_file():
            raise FileNotFoundError(f"Reference target files are required: {q_path}, {wrist_path}")
        q = np.asarray(np.load(q_path), dtype=np.float32)
        wrist = np.asarray(np.load(wrist_path), dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 6 or wrist.shape != (len(q), 4, 4):
            raise ValueError(f"Expected reference q[T,6] and wrist[T,4,4], got {q.shape}, {wrist.shape}")
        quats = np.stack([_matrix_to_xyzw(row[:3, :3]) for row in wrist], axis=0)
        targets = np.concatenate([wrist[:, :3, 3], quats], axis=1)
        return torch.from_numpy(q), torch.from_numpy(targets)

    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)
        self._create_envs(self.num_envs, float(self.cfg["env"]["envSpacing"]), int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_cfg = self.cfg["env"]["asset"]
        hand_root = os.path.abspath(os.path.expanduser(str(asset_cfg["assetRoot"])))
        hand_file = str(asset_cfg["assetFileName"])
        object_root = os.path.abspath(os.path.expanduser(str(asset_cfg["objectAssetRoot"])))
        object_file = str(asset_cfg["objectAssetFileName"])

        hand_options = gymapi.AssetOptions()
        hand_options.fix_base_link = False
        hand_options.disable_gravity = False
        hand_options.collapse_fixed_joints = False
        hand_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        self.hand_asset = self.gym.load_asset(self.sim, hand_root, hand_file, hand_options)
        if self.gym.get_asset_dof_count(self.hand_asset) != 18:
            raise RuntimeError(f"Expected Inspire native18 DOFs, got {self.gym.get_asset_dof_count(self.hand_asset)}")
        hand_props = self.gym.get_asset_dof_properties(self.hand_asset)
        hand_props["driveMode"][:] = gymapi.DOF_MODE_POS
        hand_props["stiffness"][:] = float(self.cfg["env"]["positionStiffness"])
        hand_props["damping"][:] = float(self.cfg["env"]["positionDamping"])
        self.hand_dof_lower = np.asarray(hand_props["lower"], dtype=np.float32)
        self.hand_dof_upper = np.asarray(hand_props["upper"], dtype=np.float32)
        # Isaac Gym exposes these limits in URDF/asset DOF order, matching the
        # targets produced by native_to_urdf below.

        object_options = gymapi.AssetOptions()
        object_options.fix_base_link = False
        object_asset = self.gym.load_asset(self.sim, object_root, object_file, object_options)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_hand_bodies = self.gym.get_asset_rigid_body_count(self.hand_asset)
        num_bodies_per_env = num_hand_bodies + num_object_bodies
        self.hand_indices, self.object_indices, self.tip_indices, self.envs = [], [], [], []
        tip_names = list(self.cfg["env"]["tipLinks"])
        tip_handles = [self.gym.find_asset_rigid_body_index(self.hand_asset, name) for name in tip_names]
        if any(handle < 0 for handle in tip_handles):
            raise RuntimeError(f"Could not resolve Inspire tip links {tip_names}: {tip_handles}")
        for env_id in range(num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            hand_pose = gymapi.Transform()
            hand_pose.p = gymapi.Vec3(0.0, 0.0, float(self.cfg["env"]["handHeight"]))
            hand = self.gym.create_actor(env, self.hand_asset, hand_pose, "inspire", env_id, 0, 0)
            self.gym.set_actor_dof_properties(env, hand, hand_props)
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0.0, 0.0, float(self.cfg["env"]["objectHeight"]))
            obj = self.gym.create_actor(env, object_asset, obj_pose, "airplane", env_id, 1, 0)
            self.envs.append(env)
            self.hand_indices.append(self.gym.get_actor_index(env, hand, gymapi.DOMAIN_SIM))
            self.object_indices.append(self.gym.get_actor_index(env, obj, gymapi.DOMAIN_SIM))
            self.tip_indices.append(tip_handles)
        self.hand_indices = torch.tensor(self.hand_indices, dtype=torch.int32, device=self.device)
        self.object_indices = torch.tensor(self.object_indices, dtype=torch.int32, device=self.device)
        self.tip_indices = torch.tensor(self.tip_indices, dtype=torch.long, device=self.device)
        self.num_dof = 18
        self.num_bodies = num_bodies_per_env

    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def _reference_targets(self):
        frame = torch.remainder(self.progress_buf, self.num_reference_frames).long()
        self.base_q[:] = self.reference_q.to(self.device)[frame]
        self.base_wrist[:] = self.reference_wrist.to(self.device)[frame]

    def reset_idx(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.prev_actions[env_ids] = 0.0
        self.dof_pos[env_ids] = 0.0
        self.dof_vel[env_ids] = 0.0
        self._reference_targets()
        self.initial_object_z[env_ids] = self.actor_root_state[self.object_indices[env_ids].long(), 2]
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(self.hand_indices[env_ids]), len(env_ids))
        root_ids = torch.cat([self.hand_indices[env_ids], self.object_indices[env_ids]])
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.actor_root_state), gymtorch.unwrap_tensor(root_ids), len(root_ids))

    def pre_physics_step(self, actions):
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids):
            self.reset_idx(reset_ids)
        self._reference_targets()
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        self.prev_actions.copy_(actions)
        q6 = self.base_q + self.residual_q_scale * actions[:, :6]
        native = expand_native_targets(q6)
        urdf = native_to_urdf(native)
        urdf = tensor_clamp(urdf, self.hand_q_lower, self.hand_q_upper)
        targets = torch.zeros((self.num_envs, 18), device=self.device)
        targets[:] = urdf
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

        root = self.actor_root_state[self.hand_indices.long()]
        root[:, :3] = self.base_wrist[:, :3] + self.residual_root_translation_scale * actions[:, 6:9]
        angle = self.residual_root_rotation_scale * torch.linalg.vector_norm(actions[:, 9:12], dim=-1)
        axis = actions[:, 9:12] / torch.linalg.vector_norm(actions[:, 9:12], dim=-1, keepdim=True).clamp_min(1e-6)
        delta = quat_from_angle_axis(angle, axis)
        root[:, 3:7] = quat_mul(delta, self.base_wrist[:, 3:7])
        root[:, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.actor_root_state), gymtorch.unwrap_tensor(self.hand_indices), self.num_envs)

    def compute_observations(self):
        self._refresh_state()
        self._reference_targets()
        urdf_q = self.dof_pos
        native_q = urdf_q[:, torch.argsort(torch.tensor(NATIVE_TO_URDF, device=self.device))]
        urdf_dq = self.dof_vel
        native_dq = urdf_dq[:, torch.argsort(torch.tensor(NATIVE_TO_URDF, device=self.device))]
        hand_root = self.actor_root_state[self.hand_indices.long()]
        obj_root = self.actor_root_state[self.object_indices.long()]
        env_rows = torch.arange(self.num_envs, device=self.device)[:, None]
        tip_pos = self.rigid_body_state[env_rows, self.tip_indices, :3]
        tip_offsets = (tip_pos - obj_root[:, None, :3]).reshape(self.num_envs, -1)
        object_pose = torch.cat([obj_root[:, :3] - hand_root[:, :3], obj_root[:, 3:7]], dim=-1)
        base_wrist_delta = self.base_wrist - hand_root[:, :7]
        self.obs_buf[:] = torch.cat([native_q, native_dq, self.base_q, base_wrist_delta, object_pose, tip_offsets], dim=-1).clamp(-self.cfg["env"]["clipObservations"], self.cfg["env"]["clipObservations"])
        return self.obs_buf

    def compute_reward(self):
        obj_root = self.actor_root_state[self.object_indices.long()]
        hand_root = self.actor_root_state[self.hand_indices.long()]
        env_rows = torch.arange(self.num_envs, device=self.device)[:, None]
        tip_pos = self.rigid_body_state[env_rows, self.tip_indices, :3]
        tip_dist = torch.linalg.vector_norm(tip_pos - obj_root[:, None, :3], dim=-1).mean(dim=-1)
        lift = obj_root[:, 2] - self.initial_object_z
        self.rew_buf[:] = self.lift_reward_scale * lift - self.distance_reward_scale * tip_dist - self.action_penalty_scale * (self.prev_actions.square().mean(dim=-1))
        success = lift > self.success_lift
        self.reset_buf[:] = torch.where(success | (self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.extras["success_rate"] = success.float().mean()
        self.extras["lift_mean"] = lift.mean()

    def post_physics_step(self):
        self.progress_buf += 1
        self.compute_observations()
        self.compute_reward()
