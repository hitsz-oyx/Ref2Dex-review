from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import RandomSampler, Sampler, SequentialSampler


@dataclass(frozen=True)
class DistributedState:
    enabled: bool = False
    backend: str | None = None
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


class DistributedSamplerAdapter(Sampler[int]):
    """Shard an arbitrary sampler across ranks.

    `pad=True` keeps every rank at the same length, which is required for DDP
    training steps. `pad=False` avoids duplicate samples and is suitable for
    evaluation-only loaders.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        *,
        num_replicas: int,
        rank: int,
        drop_last: bool,
        pad: bool,
    ) -> None:
        self.sampler = sampler
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.pad = bool(pad)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}.")

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        indices = list(iter(self.sampler))
        total_size = len(indices)
        if self.drop_last:
            total_size = (total_size // self.num_replicas) * self.num_replicas
            indices = indices[:total_size]
        elif self.pad:
            total_size = math.ceil(total_size / self.num_replicas) * self.num_replicas
            if total_size > len(indices):
                padding = total_size - len(indices)
                indices.extend(indices[:padding])
        return iter(indices[self.rank:total_size:self.num_replicas])

    def __len__(self) -> int:
        total_size = len(self.sampler)
        if self.drop_last:
            return total_size // self.num_replicas
        if self.pad:
            return math.ceil(total_size / self.num_replicas)
        remaining = max(0, total_size - self.rank)
        return math.ceil(remaining / self.num_replicas)


def distributed_enabled(train_cfg: Any) -> bool:
    dist_cfg = getattr(train_cfg, "distributed", None)
    return bool(getattr(dist_cfg, "enable", False))


def init_distributed(train_cfg: Any) -> DistributedState:
    requested = distributed_enabled(train_cfg)
    world_size = _get_env_int("WORLD_SIZE", 1)
    rank = _get_env_int("RANK", 0)
    local_rank = _get_env_int("LOCAL_RANK", 0)

    if world_size > 1 and not requested:
        raise ValueError(
            "WORLD_SIZE > 1 detected but train.distributed.enable is false. "
            "Pass --distributed or set train.distributed.enable=true."
        )
    if not requested or world_size <= 1:
        return DistributedState(
            enabled=False,
            backend=_resolve_backend(train_cfg),
            world_size=1,
            rank=0,
            local_rank=0,
        )

    backend = _resolve_backend(train_cfg)
    if torch.cuda.is_available() and backend == "nccl":
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_minutes = int(getattr(getattr(train_cfg, "distributed", None), "timeout_minutes", 30))
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=max(1, timeout_minutes)),
        )
    return DistributedState(
        enabled=True,
        backend=backend,
        world_size=dist.get_world_size(),
        rank=dist.get_rank(),
        local_rank=local_rank,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_dict(
    values: dict[str, float | int | torch.Tensor],
    *,
    device: torch.device,
    average: bool = False,
) -> dict[str, float]:
    reduced = {key: _scalar_value(value) for key, value in values.items()}
    if not (dist.is_available() and dist.is_initialized()):
        return reduced
    gathered_keys: list[list[str]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_keys, sorted(reduced))
    keys = sorted({key for key_group in gathered_keys for key in key_group})
    payload = torch.tensor([reduced.get(key, 0.0) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    if average:
        payload /= float(dist.get_world_size())
    return {key: float(payload[idx].item()) for idx, key in enumerate(keys)}


def wrap_model_for_distributed(
    model: torch.nn.Module,
    *,
    device: torch.device,
    train_cfg: Any,
    state: DistributedState,
) -> torch.nn.Module:
    if not state.enabled:
        return model
    dist_cfg = getattr(train_cfg, "distributed", None)
    device_ids = [device.index] if device.type == "cuda" else None
    output_device = device.index if device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=bool(getattr(dist_cfg, "broadcast_buffers", False)),
        find_unused_parameters=bool(getattr(dist_cfg, "find_unused_parameters", False)),
    )


def make_default_train_sampler(
    dataset: Any,
    *,
    shuffle: bool,
    seed: int,
    distributed: DistributedState,
    drop_last: bool,
) -> Sampler[int] | None:
    if not distributed.enabled:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    base_sampler: Sampler[int]
    if shuffle:
        base_sampler = RandomSampler(dataset, generator=generator)
    else:
        base_sampler = SequentialSampler(dataset)
    return DistributedSamplerAdapter(
        base_sampler,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        drop_last=drop_last,
        pad=True,
    )


def make_default_eval_sampler(
    dataset: Any,
    *,
    distributed: DistributedState,
) -> Sampler[int] | None:
    if not distributed.enabled:
        return None
    return DistributedSamplerAdapter(
        SequentialSampler(dataset),
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        drop_last=False,
        pad=False,
    )


def shard_sampler_for_distributed(
    sampler: Sampler[int],
    *,
    distributed: DistributedState,
    drop_last: bool,
    pad: bool,
) -> Sampler[int]:
    if not distributed.enabled:
        return sampler
    return DistributedSamplerAdapter(
        sampler,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        drop_last=drop_last,
        pad=pad,
    )


def _resolve_backend(train_cfg: Any) -> str:
    backend = str(getattr(getattr(train_cfg, "distributed", None), "backend", "auto")).lower()
    if backend == "auto":
        return "nccl" if torch.cuda.is_available() else "gloo"
    return backend


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def _scalar_value(value: float | int | torch.Tensor) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("reduce_dict only supports scalar tensors.")
        return float(value.detach().cpu().item())
    return float(value)
