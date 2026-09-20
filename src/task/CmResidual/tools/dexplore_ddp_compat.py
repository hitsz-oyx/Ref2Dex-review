"""Task-local torch.distributed facade for DExplore's legacy Horovod calls.

This module deliberately does not import Horovod or modify external DExplore /
rl_games files.  ``install_horovod_facade`` only exposes a process-local module
to their legacy imports after a torchrun process group has been initialized.
"""
from __future__ import annotations

from contextlib import nullcontext
import os
import sys
import types
from typing import Any, Iterable, Optional, Tuple


_TORCH = None
_DIST = None


def _torch_and_dist():
    """Import Torch lazily so Isaac Gym can own the first Torch import."""
    global _TORCH, _DIST
    if _TORCH is None:
        import torch
        import torch.distributed as dist

        _TORCH, _DIST = torch, dist
    return _TORCH, _DIST


def is_initialized() -> bool:
    _, dist = _torch_and_dist()
    return dist.is_available() and dist.is_initialized()


def initialize_from_env(backend: str = "nccl", timeout_seconds: int = 1800) -> Tuple[int, int, int]:
    """Initialize one torchrun rank and bind its logical CUDA device."""
    torch, dist = _torch_and_dist()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL facade requires CUDA")
        torch.cuda.set_device(local_rank)
    if not is_initialized():
        from datetime import timedelta

        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_seconds))
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("torch.distributed environment disagrees with initialized process group")
    return rank, local_rank, world_size


def cleanup() -> None:
    _, dist = _torch_and_dist()
    if is_initialized():
        dist.destroy_process_group()


def _require_initialized() -> Tuple[Any, Any]:
    torch, dist = _torch_and_dist()
    if not is_initialized():
        raise RuntimeError("DDP facade has not been initialized; launch with torchrun bootstrap")
    return torch, dist


def _rank() -> int:
    _, dist = _require_initialized()
    return dist.get_rank()


def _size() -> int:
    _, dist = _require_initialized()
    return dist.get_world_size()


def _local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def _allreduce(value, name: Optional[str] = None, average: bool = True):
    del name
    torch, dist = _require_initialized()
    result = value.detach().clone()
    copy_back = result.device.type == "cpu" and dist.get_backend() == "nccl"
    if copy_back:
        result = result.to(torch.device("cuda", _local_rank()))
    dist.all_reduce(result)
    if average:
        result.div_(_size())
    if copy_back:
        result = result.cpu()
    return result


def _broadcast_parameters(params: dict, root_rank: int = 0) -> None:
    torch, dist = _require_initialized()
    for value in params.values():
        if not hasattr(value, "data"):
            raise TypeError("broadcast_parameters expects tensors")
        tensor = value.data
        copy_back = tensor.device.type == "cpu" and dist.get_backend() == "nccl"
        if copy_back:
            tensor = tensor.to(torch.device("cuda", _local_rank()))
        dist.broadcast(tensor, src=root_rank)
        if copy_back:
            value.data.copy_(tensor.cpu())


def _broadcast_optimizer_state(optimizer, root_rank: int = 0) -> None:
    torch, dist = _require_initialized()
    base = getattr(optimizer, "optimizer", optimizer)
    payload = [base.state_dict() if _rank() == root_rank else None]
    device = torch.device("cuda", _local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    dist.broadcast_object_list(payload, src=root_rank, device=device)
    if _rank() != root_rank:
        base.load_state_dict(payload[0])


class DistributedOptimizer:
    """Minimal Horovod DistributedOptimizer-compatible wrapper backed by NCCL."""

    def __init__(self, optimizer, named_parameters: Optional[Iterable] = None,
                 backward_passes_per_step: int = 1, **_: Any):
        if backward_passes_per_step < 1:
            raise ValueError("backward_passes_per_step must be positive")
        self.optimizer = optimizer
        self.named_parameters = list(named_parameters) if named_parameters is not None else None
        self.backward_passes_per_step = backward_passes_per_step
        self._synchronized = False

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, *args, **kwargs):
        self._synchronized = False
        return self.optimizer.zero_grad(*args, **kwargs)

    def synchronize(self) -> None:
        _, dist = _require_initialized()
        if self._synchronized:
            return
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                dist.all_reduce(parameter.grad)
                parameter.grad.div_(_size())
        self._synchronized = True

    def skip_synchronize(self):
        return nullcontext()

    def step(self, *args, **kwargs):
        return self.optimizer.step(*args, **kwargs)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self.optimizer.load_state_dict(state_dict)

    def __getattr__(self, name: str):
        return getattr(self.optimizer, name)


def install_horovod_facade() -> types.ModuleType:
    """Install a non-packaged, process-local legacy import facade.

    DExplore and rl_games import ``horovod.torch`` by name.  We provide only
    the APIs they call, backed by this module, and never import a Horovod wheel.
    """
    existing = sys.modules.get("horovod.torch")
    if existing is not None:
        if getattr(existing, "__ref2dex_ddp_facade__", False):
            return existing
        raise RuntimeError("refusing to replace an already imported Horovod module")

    package = types.ModuleType("horovod")
    module = types.ModuleType("horovod.torch")
    module.__ref2dex_ddp_facade__ = True
    module.init = lambda: initialize_from_env()
    module.rank = _rank
    module.size = _size
    module.local_rank = _local_rank
    module.allreduce = _allreduce
    module.broadcast_parameters = _broadcast_parameters
    module.broadcast_optimizer_state = _broadcast_optimizer_state
    module.DistributedOptimizer = DistributedOptimizer
    package.torch = module
    sys.modules["horovod"] = package
    sys.modules["horovod.torch"] = module
    return module
