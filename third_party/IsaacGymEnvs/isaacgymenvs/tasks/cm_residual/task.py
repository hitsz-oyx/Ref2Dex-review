"""Inspire residual task with a frozen DExplore teacher and Cm residual."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.base.vec_task import VecTask
from src.task.CmDecoderv2.kinematics import QUERY_LINKS
from src.task.CmDecoderv2.rl.online_base import sha256
from src.task.CmDecoderv2.rl.residual_contract import (
    coupled_finger_bounds, inverse_pose, matrix_pose, native_sim_indices, native_to_sim,
    pose_matrix, sim_to_native,
)
from .base_policy import ACTION_DIM, OBSERVATION_DIM, InspireDExplorePolicy
from .cm_adapter import CmOnlineTarget


class CmResidual(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture=False, force_render=False):
        self.cfg = cfg
        if cfg["basePolicy"]["mode"] != "dexplore":
            raise ValueError("CmResidual now requires the frozen DExplore teacher (mode=dexplore).")
        self.teacher = InspireDExplorePolicy(cfg["basePolicy"]["checkpoint"], sim_device,
                                             cfg["basePolicy"].get("checkpointSha256"))
        self.cm = CmOnlineTarget(OBSERVATION_DIM, int(cfg["basePolicy"].get("cmFeatureDim", 128)),
                                 float(cfg["basePolicy"].get("cmEmaDecay", 0.995)))
        # The package no longer depends on the legacy 12D OnlineCmBase contract.
        self.base = None
        self.max_episode_length = int(cfg["env"].get("episodeLength", 2000))
        cfg["env"].update(numObservations=OBSERVATION_DIM, numActions=ACTION_DIM, numStates=0,
                          episodeLength=self.max_episode_length)
        self.initial_native = torch.zeros(18, device=sim_device)
        self.initial_object = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=sim_device)
        self.initial_table = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=sim_device)
        self.initial_links = torch.eye(4, device=sim_device).repeat(len(QUERY_LINKS), 1, 1)
        self.zero_base_inverse = torch.eye(4, device=sim_device)
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
        self.sim_indices = torch.as_tensor(self.sim_indices, device=self.device, dtype=torch.long)
        self.query_indices = torch.as_tensor(self.query_indices, device=self.device, dtype=torch.long)
        self.tip_indices = torch.as_tensor(self.tip_indices, device=self.device, dtype=torch.long)
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
        self.pd_action_scale = self.native_upper - self.native_lower
        self.pd_action_scale[:3] = 1.0
        self.pd_action_scale[3:6] = np.pi
        self.pd_action_offset = torch.zeros_like(self.native_lower)
        self.pd_action_offset[6:] = self.native_lower[6:]
        self.fresh_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.initial_object_z = torch.full((self.num_envs,), self.initial_object[2].item(), device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_lift = torch.zeros(self.num_envs, device=self.device)
        self.completed_episodes = 0
        self.successful_episodes = 0
        self._refresh_state()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        plane.static_friction = plane.dynamic_friction = 0.9
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
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options.angular_damping = 0.01
        hand = self.gym.load_asset(self.sim, str(Path(asset["assetRoot"]).resolve()), asset["assetFileName"], options)
        names = list(self.gym.get_asset_dof_names(hand))
        self.sim_indices = native_sim_indices(names)
        props = self.gym.get_asset_dof_properties(hand)
        props["driveMode"][:] = gymapi.DOF_MODE_POS
        props["stiffness"][self.sim_indices] = [200.0] * 6 + [100.0] * 12
        props["damping"][self.sim_indices] = [20.0] * 6 + [10.0] * 12
        props["velocity"][:] = 7.0
        self.sim_lower, self.sim_upper = props["lower"].copy(), props["upper"].copy()
        obj_options = gymapi.AssetOptions()
        obj_options.density = 20
        obj_options.angular_damping = obj_options.linear_damping = 0.01
        obj_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        obj_options.vhacd_enabled = True
        obj_options.vhacd_params.max_convex_hulls = 20
        obj_options.vhacd_params.max_num_vertices_per_ch = 16
        obj_options.vhacd_params.resolution = 50000
        obj = self.gym.load_asset(self.sim, str(object_root), "airplane.urdf", obj_options)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table = self.gym.load_asset(self.sim, str(object_root), "table.urdf", table_options)
        self.query_indices = [self.gym.find_asset_rigid_body_index(hand, name) for name in QUERY_LINKS]
        self.tip_indices = [self.gym.find_asset_rigid_body_index(hand, name) for name in self.cfg["env"]["tipLinks"]]
        if min(self.query_indices + self.tip_indices) < 0:
            raise ValueError("Missing Inspire observation link")
        self.num_bodies = sum(self.gym.get_asset_rigid_body_count(a) for a in (hand, obj, table))
        self.hand_indices, self.object_indices, self.envs = [], [], []
        spacing = float(self.cfg["env"]["envSpacing"])
        for index in range(self.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), int(np.sqrt(self.num_envs)))
            h = self.gym.create_actor(env, hand, gymapi.Transform(), "inspire", index, 1, 0)
            self.gym.set_actor_dof_properties(env, h, props)
            # Match Dexplore's filters using body-to-shape ranges, not body IDs.
            shapes = self.gym.get_actor_rigid_shape_properties(env, h)
            body_names = self.gym.get_actor_rigid_body_names(env, h)
            ranges = self.gym.get_actor_rigid_body_shape_indices(env, h)
            for name, span in zip(body_names, ranges):
                value = 3 if ("thumb" in name and "distal" in name) or ("thumb" not in name and "intermediate" in name) else 2
                for shape in shapes[span.start:span.start + span.count]:
                    shape.filter, shape.friction = value, 0.9
            self.gym.set_actor_rigid_shape_properties(env, h, shapes)
            o = self.gym.create_actor(env, obj, self._gym_pose(self.initial_object), "airplane", index, 0, 1)
            t = self.gym.create_actor(env, table, self._gym_pose(self.initial_table), "table", index, 1, 2)
            for actor in (o, t):
                shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
                for shape in shapes:
                    shape.friction = 0.9
                self.gym.set_actor_rigid_shape_properties(env, actor, shapes)
            self.hand_indices.append(self.gym.get_actor_index(env, h, gymapi.DOMAIN_SIM))
            self.object_indices.append(self.gym.get_actor_index(env, o, gymapi.DOMAIN_SIM))
            self.envs.append(env)

    def _refresh_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def actual_link_poses(self):
        poses = pose_matrix(self.rigid_body_state[:, self.query_indices, :7])
        # Gym recomputes rigid-body FK on simulate; use exact reset-state FK only
        # until that first physics step, then exclusively use measured link poses.
        poses[self.fresh_reset] = self.initial_links
        return poses

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        roots = torch.cat([self.hand_indices[env_ids], self.object_indices[env_ids]]).contiguous()
        self.actor_root_state[roots.long()] = self.initial_root_states[roots.long()]
        self.dof_pos[env_ids] = native_to_sim(self.initial_native.expand(len(env_ids), -1), self.sim_indices)
        self.dof_vel[env_ids] = 0
        self.native_targets[env_ids] = self.initial_native
        self.sim_targets[env_ids] = self.dof_pos[env_ids]
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.episode_max_lift[env_ids] = 0
        self.initial_object_z[env_ids] = self.initial_object[2]
        self.fresh_reset[env_ids] = True
        hand_ids = self.hand_indices[env_ids].contiguous()
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.actor_root_state), gymtorch.unwrap_tensor(roots), len(roots))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(hand_ids), len(hand_ids))
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.sim_targets), gymtorch.unwrap_tensor(hand_ids), len(hand_ids))

    def pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Non-finite residual action")
        actions = actions.to(self.device).clamp(-1, 1)
        self.prev_actions.copy_(actions)
        current = sim_to_native(self.dof_pos, self.sim_indices)
        final_action = (self.base_action + actions).clamp(-1, 1)
        # Exact DExplore Inspire mapping: wrist is 0:6, fingers are 6:18.
        finger_action = (1.0 + final_action[:, 6:]) / 2.0
        pd_action = torch.cat((final_action[:, :6], finger_action), dim=-1)
        targets = self.pd_action_offset + self.pd_action_scale * pd_action
        targets[:, :6] = targets[:, :6] + current[:, :6]
        targets[:, 7] = targets[:, 6] * 1.05
        targets[:, 9] = targets[:, 8] * 1.05
        targets[:, 11] = targets[:, 10] * 1.05
        targets[:, 13] = targets[:, 12] * 1.05
        targets[:, 16] = targets[:, 15] * 0.6
        targets[:, 17] = targets[:, 15] * 0.8
        targets = targets.clamp(self.native_lower, self.native_upper)
        self.native_targets.copy_(targets)
        self.sim_targets.copy_(native_to_sim(targets, self.sim_indices))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.sim_targets))

    def compute_observations(self):
        self._refresh_state()
        native = sim_to_native(self.dof_pos, self.sim_indices)
        velocity = sim_to_native(self.dof_vel, self.sim_indices)
        links = self.actual_link_poses()
        obj = pose_matrix(self.actor_root_state[self.object_indices.long(), :7])
        q, wrist = native[:, :6], links[:, 0]
        self.base_q.copy_(q)
        self.base_wrist.copy_(wrist)
        wrist_delta = matrix_pose(inverse_pose(links[:, 0]) @ wrist)
        object_delta = matrix_pose(inverse_pose(links[:, 0]) @ obj)
        tips = self.rigid_body_state[:, self.tip_indices, :3]
        if self.fresh_reset.any():
            tip_queries = [QUERY_LINKS.index(name) for name in self.cfg["env"]["tipLinks"]]
            tips = tips.clone()
            tips[self.fresh_reset] = self.initial_links[tip_queries, :3, 3]
        tip_offsets = (tips - obj[:, None, :3, 3]).reshape(self.num_envs, 15)
        legacy_obs = torch.cat([native, velocity, q, wrist_delta, object_delta, tip_offsets], dim=-1)
        # Preserve the DExplore 1442D tensor contract; unavailable reference fields are explicit zeros.
        obs = torch.zeros((self.num_envs, OBSERVATION_DIM), device=self.device)
        obs[:, :legacy_obs.shape[-1]] = legacy_obs
        if not torch.isfinite(obs).all():
            raise FloatingPointError("Non-finite actual-state observation")
        self.obs_buf.copy_(obs.clamp(-self.clip_obs, self.clip_obs))
        self.base_action.copy_(self.teacher.act(self.obs_buf))
        return self.obs_buf

    def compute_reward(self):
        env = self.cfg["env"]
        obj = self.actor_root_state[self.object_indices.long()]
        tips = self.rigid_body_state[:, self.tip_indices, :3]
        distance = (tips - obj[:, None, :3]).norm(dim=-1).mean(-1)
        lift = obj[:, 2] - self.initial_object_z
        self.rew_buf[:] = float(env["liftRewardScale"]) * lift - float(env["distanceRewardScale"]) * distance - float(env["actionPenaltyScale"]) * self.prev_actions.square().mean(-1)
        self.episode_return += self.rew_buf
        self.episode_max_lift = torch.maximum(self.episode_max_lift, lift)
        success = lift > float(env["successLift"])
        done = success | (self.progress_buf >= self.max_episode_length)
        self.reset_buf.copy_(done.long())
        self.completed_episodes += int(done.sum().item())
        self.successful_episodes += int((success & done).sum().item())
        self.extras.pop("episode", None)
        if done.any():
            self.extras["episode"] = {"success": success[done].float(), "return": self.episode_return[done].clone(),
                                      "max_lift_m": self.episode_max_lift[done].clone()}
        self.extras.update(lift_mean=lift.mean(), tip_distance_mean=distance.mean(),
                          residual_rms=self.prev_actions.square().mean().sqrt(),
                          success_rate=self.successful_episodes / max(1, self.completed_episodes))

    def post_physics_step(self):
        self.fresh_reset[:] = False
        self.progress_buf += 1
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
