"""Simulator-independent V1.21c collection contract.

The official DExplore runtime remains an external adapter.  This accumulator owns
only the deterministic stop/uniqueness/quota rules, so a collector cannot silently
back-fill one phase with another or duplicate a state when the runtime is noisy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.task.CmResidual.v121c_ranking import (
    H_REF,
    PHASE_NAMES,
    PHASE_QUOTAS,
    CollectionInsufficient,
)


@dataclass(frozen=True)
class CollectionConfig:
    collection_batch_id: str
    target_unique: int = 512
    max_collection_episodes: int = 64
    first_episode_seed: int = 5909
    reference_horizon: int = H_REF
    phase_quotas: Mapping[int, int] = field(default_factory=lambda: dict(PHASE_QUOTAS))

    def __post_init__(self) -> None:
        if not self.collection_batch_id:
            raise ValueError("collection_batch_id is required")
        if self.target_unique != sum(int(value) for value in self.phase_quotas.values()):
            raise ValueError("target_unique must equal the sum of fixed phase quotas")
        if self.target_unique <= 0 or self.max_collection_episodes <= 0:
            raise ValueError("collection targets must be positive")
        if self.reference_horizon != H_REF:
            raise ValueError("V1.21c fixes reference_horizon=6")

    def episode_seed(self, episode_id: int) -> int:
        if episode_id < 0 or episode_id >= self.max_collection_episodes:
            raise ValueError("episode_id exceeds the collection episode budget")
        return self.first_episode_seed + int(episode_id)


class CollectionAccumulator:
    """Collect selected state records without modifying their contents."""

    def __init__(self, config: CollectionConfig):
        self.config = config
        self._records: list[Mapping[str, Any]] = []
        self._state_ids: set[str] = set()
        self._episode_ids: set[int] = set()
        self._phase_counts = {int(key): 0 for key in config.phase_quotas}

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def phase_counts(self) -> dict[str, int]:
        return {PHASE_NAMES.get(phase, str(phase)): self._phase_counts[phase]
                for phase in self._phase_counts}

    @property
    def complete(self) -> bool:
        return self.count == self.config.target_unique and all(
            self._phase_counts[phase] >= int(quota)
            for phase, quota in self.config.phase_quotas.items())

    def add(
        self,
        *,
        state_id: str,
        episode_id: int,
        seed: int,
        frame_id: int,
        sequence_length: int,
        active: bool,
        phase_id: int,
        record: Mapping[str, Any],
    ) -> bool:
        """Try to add one state; return false for deterministic non-selected states."""
        if self.complete:
            return False
        if episode_id < 0 or episode_id >= self.config.max_collection_episodes:
            raise ValueError("episode_id exceeds the collection episode budget")
        if seed != self.config.episode_seed(episode_id):
            raise ValueError("episode seed must equal first_episode_seed + episode_id")
        if frame_id < 0 or frame_id > sequence_length - 1 - self.config.reference_horizon:
            return False
        if not active or phase_id not in self.config.phase_quotas:
            return False
        if state_id in self._state_ids:
            return False
        if self._phase_counts[phase_id] >= int(self.config.phase_quotas[phase_id]):
            return False
        if not isinstance(record, Mapping):
            raise TypeError("record must be a mapping")
        if str(record.get("state_id", "")) != str(state_id):
            raise ValueError("record.state_id must match the collection state_id")
        self._records.append(record)
        self._state_ids.add(state_id)
        self._episode_ids.add(int(episode_id))
        self._phase_counts[phase_id] += 1
        return True

    def finalize(self) -> list[Mapping[str, Any]]:
        """Return canonical state-id order or fail with the collection stop reason."""
        if not self.complete:
            missing = {PHASE_NAMES.get(phase, str(phase)): int(quota) - self._phase_counts[phase]
                       for phase, quota in self.config.phase_quotas.items()
                       if self._phase_counts[phase] < int(quota)}
            raise CollectionInsufficient(
                f"collection insufficient after {len(self._episode_ids)} episodes: {missing}")
        return sorted(self._records, key=lambda record: str(record["state_id"]))
