from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .utils import flatten_features, resolve_activation


class MLP(nn.Module):
    """A small MLP inspired by RoboGym's learning.modules.MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        activation: str = "elu",
        output_activation: str | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *[input_dim if dim == -1 else int(dim) for dim in hidden_dims], output_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if is_last:
                if output_activation is not None:
                    layers.append(resolve_activation(output_activation))
            else:
                layers.append(resolve_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def init_weights(self, method: str = "orthogonal") -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == "orthogonal":
                    nn.init.orthogonal_(module.weight, gain=1.0)
                elif method == "xavier_uniform":
                    nn.init.xavier_uniform_(module.weight)
                elif method == "kaiming_uniform":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                elif method in {"none", "default"}:
                    pass
                else:
                    raise ValueError(f"Unknown init method: {method}")
                if module.bias is not None and method not in {"none", "default"}:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_shape: tuple[int, ...],
        cfg: Any,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_shape = output_shape
        self.backbone = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
            output_activation=cfg.output_activation,
            dropout=cfg.dropout,
        )
        self.backbone.init_weights(cfg.init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = flatten_features(x.float())
        y = self.backbone(flat)
        return y.reshape(y.shape[0], *self.output_shape)


def build_model(cfg: Any, metadata: dict[str, Any]) -> nn.Module:
    input_shape = tuple(int(v) for v in metadata["input_shape"])
    target_shape = tuple(int(v) for v in metadata["target_shape"])
    input_dim = int(getattr(cfg, "input_dim", None) or math.prod(input_shape))
    model_type = str(getattr(cfg, "type", "mlp")).lower()

    if model_type not in {"mlp", "policy_mlp", "linear"}:
        raise ValueError(f"Unknown model type '{getattr(cfg, 'type', None)}'. Currently supported: mlp, linear.")

    model_cfg = SimpleNamespace(
        hidden_dims=[] if model_type == "linear" else list(getattr(cfg, "hidden_dims", [])),
        activation=getattr(cfg, "activation", "elu"),
        output_activation=getattr(cfg, "output_activation", None),
        dropout=float(getattr(cfg, "dropout", 0.0)),
        init=getattr(cfg, "init", "orthogonal"),
    )

    output_dim_config = getattr(cfg, "output_dim", None)
    output_shape = target_shape if output_dim_config is None else (int(output_dim_config),)
    output_dim = int(math.prod(output_shape)) if output_shape else 1

    return MLPModel(input_dim=input_dim, output_dim=output_dim, output_shape=output_shape, cfg=model_cfg)
