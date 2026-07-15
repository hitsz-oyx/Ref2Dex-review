from __future__ import annotations

import importlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MetricStat:
    """A metric represented by a reducible numerator and valid count."""

    total: float
    count: float
    expose_validity: bool = False

    @classmethod
    def from_value(
        cls,
        value: float,
        *,
        count: float = 1.0,
        valid: bool = True,
        expose_validity: bool = False,
    ) -> "MetricStat":
        if not valid or count <= 0:
            return cls.invalid(expose_validity=expose_validity)
        count = float(count)
        return cls(
            total=float(value) * count,
            count=count,
            expose_validity=expose_validity,
        )

    @classmethod
    def invalid(cls, *, expose_validity: bool = False) -> "MetricStat":
        return cls(total=0.0, count=0.0, expose_validity=expose_validity)


def set_seed(seed: int) -> int:
    if seed == -1:
        seed = int(np.random.randint(0, 10000))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def resolve_device(device: str, *, local_rank: int | None = None) -> torch.device:
    if device == "auto":
        if local_rank is not None and torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if local_rank is not None and resolved.type == "cuda":
        return torch.device("cuda", local_rank)
    return resolved


def import_from_path(dotted_path: str) -> Any:
    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"Cannot import '{attr}' from '{module_path}'") from exc


def resolve_activation(name: str | None) -> nn.Module:
    if name is None:
        return nn.Identity()
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "celu": nn.CELU,
        "lrelu": nn.LeakyReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
        "identity": nn.Identity,
        "linear": nn.Identity,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Invalid activation '{name}'. Valid activations: {sorted(activations)}")
    return activations[key]()


def resolve_optimizer(name: str) -> type[torch.optim.Optimizer]:
    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    key = name.lower()
    if key not in optimizers:
        raise ValueError(f"Invalid optimizer '{name}'. Valid optimizers: {sorted(optimizers)}")
    return optimizers[key]


def flatten_features(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(-1)
    return x.flatten(start_dim=1)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class AverageMeter:
    def __init__(self) -> None:
        self.total: float | torch.Tensor = 0.0
        self.count = 0

    def update(self, value: float | torch.Tensor, n: int = 1) -> None:
        if torch.is_tensor(value):
            # Keep CUDA scalar metrics on-device until the logging or epoch
            # reduction boundary. Detaching avoids retaining each step graph.
            contribution = value.detach() * n
            self.total = contribution + self.total
        else:
            self.total = (
                self.total + float(value) * n
                if torch.is_tensor(self.total)
                else self.total + float(value) * n
            )
        self.count += n

    def update_total_count(self, total: float, count: float) -> None:
        self.total = float(self.total) + float(total)
        self.count += float(count)

    @property
    def avg(self) -> float:
        total = self.total.detach().cpu() if torch.is_tensor(self.total) else self.total
        return float(total) / self.count


class MetricAverager:
    def __init__(self) -> None:
        self.meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
        self.expose_validity: dict[str, bool] = {}

    def update(self, metrics: dict[str, float | torch.Tensor | MetricStat], n: int = 1) -> None:
        for key, value in metrics.items():
            if isinstance(value, MetricStat):
                self.meters[key].update_total_count(value.total, value.count)
                self.expose_validity[key] = (
                    self.expose_validity.get(key, False) or value.expose_validity
                )
            else:
                self.meters[key].update(value, n)

    def compute(self, prefix: str = "") -> dict[str, float]:
        result: dict[str, float] = {}
        for key, meter in self.meters.items():
            if meter.count > 0:
                result[f"{prefix}{key}"] = meter.avg
            if self.expose_validity.get(key, False):
                result[f"{prefix}{key}_valid_count"] = float(meter.count)
                result[f"{prefix}{key}_valid"] = float(meter.count > 0)
        return result


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("time", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
