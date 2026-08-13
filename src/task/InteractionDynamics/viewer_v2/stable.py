"""V19.2 解析 stable-grasp latch 状态机。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GraspObservation:
    d_cm: np.ndarray
    u_cm: np.ndarray
    p_mm: np.ndarray | None = None


@dataclass(frozen=True)
class StableState:
    contact_count: int
    u_rms_cm: float
    penetration_max_mm: float | None
    stable_now: bool
    stable_count: int
    latched: bool


class StableDetector:
    def __init__(self, contact_anchors: int = 4, distance_cm: float = 2.,
                 max_u_cm: float = .3, consecutive_frames: int = 3) -> None:
        self.contact_anchors = contact_anchors; self.distance_cm = distance_cm
        self.max_u_cm = max_u_cm; self.consecutive_frames = consecutive_frames
        self.stable_count = 0; self.latched = False

    def update(self, observation: GraspObservation) -> StableState:
        contact = int(np.count_nonzero(np.asarray(observation.d_cm) < self.distance_cm))
        u_rms = float(np.sqrt(np.mean(np.square(observation.u_cm))))
        stable = contact >= self.contact_anchors and u_rms < self.max_u_cm
        self.stable_count = self.stable_count + 1 if stable else 0
        self.latched |= self.stable_count >= self.consecutive_frames
        p_max = None if observation.p_mm is None else float(np.max(observation.p_mm, initial=0))
        return StableState(contact, u_rms, p_max, stable, self.stable_count, self.latched)


def detect_latch(distance_cm: np.ndarray, motion_cm: np.ndarray,
                 penetration_mm: list[np.ndarray | None] | None = None,
                 consecutive_frames: int = 3) -> tuple[int | None, list[StableState]]:
    detector = StableDetector(consecutive_frames=consecutive_frames); states = []
    for frame in range(len(motion_cm)):
        p = None if penetration_mm is None else penetration_mm[frame]
        state = detector.update(GraspObservation(distance_cm[frame], motion_cm[frame], p))
        states.append(state)
        if state.latched: return frame, states
    return None, states
