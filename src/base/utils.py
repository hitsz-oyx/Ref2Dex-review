from __future__ import annotations

import importlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


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


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


class MetricAverager:
    def __init__(self) -> None:
        self.meters: dict[str, AverageMeter] = defaultdict(AverageMeter)

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.meters[key].update(float(value), n)

    def compute(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{key}": meter.avg for key, meter in self.meters.items()}


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
