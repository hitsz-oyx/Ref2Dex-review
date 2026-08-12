"""V16.6 active + next meaningful object-effect 条件扩散。"""
from __future__ import annotations

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle
from torch import nn

from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule, sinusoidal_embedding
from src.task.InteractionDynamics.state_interaction_diffusion import AdaLNBlock


def object_increments(object_poses: torch.Tensor) -> torch.Tensor:
    """从 frame-to-world pose 得到相邻局部 SE(3) 的 6D cm/rad 表示。"""
    relative = torch.linalg.inv(object_poses[:-1]) @ object_poses[1:]
    return torch.cat([100 * relative[:, :3, 3], matrix_to_axis_angle(relative[:, :3, :3])], -1)


def meaningful_motion_mask(object_poses: torch.Tensor, window: int = 3,
                           translation_cm: float = .2,
                           rotation_deg: float = 1.) -> torch.Tensor:
    if window < 1 or len(object_poses) <= window:
        raise ValueError("Object trajectory is too short for motion check window")
    relative = torch.linalg.inv(object_poses[:-window]) @ object_poses[window:]
    translation = 100 * relative[:, :3, 3].norm(dim=-1)
    rotation = torch.rad2deg(matrix_to_axis_angle(relative[:, :3, :3]).norm(dim=-1))
    return (translation > translation_cm) | (rotation > rotation_deg)


def extract_goal(object_poses: torch.Tensor, current: int, active: bool,
                 segment: int = 4, window: int = 3,
                 translation_cm: float = .2, rotation_deg: float = 1.) -> tuple[torch.Tensor, int]:
    """Goal motion 只读取 object pose；inactive 时严格返回全零。"""
    goal = torch.zeros(1 + 6 * segment, dtype=object_poses.dtype, device=object_poses.device)
    if not active:
        return goal, -1
    goal[0] = 1
    meaningful = meaningful_motion_mask(
        object_poses, window, translation_cm, rotation_deg)
    candidates = torch.nonzero(meaningful[current:], as_tuple=False).flatten()
    if not len(candidates):
        return goal, -1
    onset = current + int(candidates[0])
    increments = object_increments(object_poses)
    available = increments[onset:min(onset + segment, len(increments))]
    goal[1:1 + available.numel()] = available.flatten()
    return goal, onset


def estimate_active_interval(data: dict[str, np.ndarray], meaningful: np.ndarray,
                             near_cm: float = 10.) -> tuple[int, int, np.ndarray]:
    """以 demonstration 几何生成 active label；该 label 不编码进 motion goal。"""
    hand = np.asarray(data["hand_points_world"], np.float32)[:, ::20]
    obj = np.asarray(data["obj_points_world"], np.float32)[:, ::50]
    distance = np.linalg.norm(hand[:, :, None] - obj[:, None], axis=-1).min((1, 2)) * 100
    motion_frames = np.flatnonzero(meaningful)
    if not len(motion_frames):
        raise ValueError("Sequence has no meaningful object motion")
    near = np.flatnonzero(distance <= near_cm)
    first_near = int(near[0]) if len(near) else int(motion_frames[0])
    # 从首次近场向前回溯持续接近段，避免把尚未接触的 approach 标为 inactive。
    smooth = np.convolve(distance, np.ones(5, np.float32) / 5, mode="same")
    approach = np.flatnonzero(smooth[:-5] - smooth[5:] > .5)
    before = approach[approach < first_near]
    start = int(before[-1]) if len(before) else first_near
    while start > 0 and start + 4 < len(smooth) and smooth[start - 1] - smooth[start + 4] > .2:
        start -= 1
    after = near[near >= motion_frames[-1]]
    last_near = int(after[-1]) if len(after) else int(motion_frames[-1])
    # release 后的持续远离仍属于 manipulation interval，直到 withdraw 基本结束。
    withdrawing = np.flatnonzero(smooth[5:] - smooth[:-5] > .5) + 5
    tail = withdrawing[withdrawing >= last_near]
    end = int(tail[-1]) if len(tail) else last_near
    return start, end, distance


def stratified_frames(frame_count: int, active_start: int, active_end: int,
                      horizon: int, count: int = 32) -> list[int]:
    """尽量按 inactive-before / active / inactive-after = 8/16/8 采样。"""
    valid_end = frame_count - horizon - 1
    groups = [np.arange(0, min(active_start, valid_end + 1)),
              np.arange(max(0, active_start), min(active_end, valid_end) + 1),
              np.arange(max(active_end + 1, 0), valid_end + 1)]
    quotas = [count // 4, count // 2, count - 3 * count // 4]
    result: list[int] = []
    for values, quota in zip(groups, quotas):
        if len(values):
            selected = np.rint(np.linspace(0, len(values) - 1, min(quota, len(values)))).astype(int)
            result.extend(values[selected].tolist())
    if len(result) < count:
        remaining = sorted(set(range(valid_end + 1)) - set(result))
        selected = np.rint(np.linspace(0, len(remaining) - 1, count - len(result))).astype(int)
        result.extend(remaining[index] for index in selected)
    return sorted(result[:count])


class GoalInteractionDiffusion(nn.Module):
    def __init__(self, horizon: int = 4, goal_segment: int = 4, dim: int = 256,
                 heads: int = 8, layers: int = 6) -> None:
        super().__init__()
        self.horizon, self.goal_segment = horizon, goal_segment
        self.future_dim = 7 * horizon
        self.dim = dim
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.future = nn.Sequential(nn.Linear(self.future_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.goal = nn.Sequential(nn.Linear(1 + 6 * goal_segment, 128), nn.GELU(), nn.Linear(128, dim))
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.condition = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.output_norm, self.output = nn.LayerNorm(dim), nn.Linear(dim, self.future_dim)

    def forward(self, noisy_future: torch.Tensor, current_state: torch.Tensor,
                anchors_cm: torch.Tensor, object_patches: torch.Tensor,
                goal: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        tokens = (self.future(noisy_future) + self.state(current_state) + self.anchor(anchors_cm)
                  + self.object_point(object_patches).amax(2))
        condition = self.condition(torch.cat([
            self.goal(goal), self.time(sinusoidal_embedding(timestep, self.dim))], -1))
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.output(self.output_norm(tokens))


@torch.no_grad()
def sample_goal_ddim(model: GoalInteractionDiffusion, state: torch.Tensor,
                     anchors_cm: torch.Tensor, patches: torch.Tensor, goal: torch.Tensor,
                     steps: int, initial_noise: torch.Tensor | None = None,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    _, alpha_bar = cosine_schedule(steps, anchors_cm.device)
    value = (torch.randn((*anchors_cm.shape[:2], model.future_dim), device=anchors_cm.device,
                         generator=generator) if initial_noise is None else initial_noise.clone())
    for index in reversed(range(steps)):
        timestep = torch.full((len(value),), index, device=value.device, dtype=torch.long)
        epsilon = model(value, state, anchors_cm, patches, goal, timestep)
        clean = ((value - (1 - alpha_bar[index]).sqrt() * epsilon)
                 / alpha_bar[index].sqrt()).clamp(-5, 5)
        value = (alpha_bar[index - 1].sqrt() * clean
                 + (1 - alpha_bar[index - 1]).sqrt() * epsilon) if index else clean
    return value
