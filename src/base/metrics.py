from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


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


class AverageMeter:
    def __init__(self) -> None:
        self.total: float | torch.Tensor = 0.0
        self.count = 0

    def update(self, value: float | torch.Tensor, n: int = 1) -> None:
        if torch.is_tensor(value):
            # Keep scalar metrics on-device until the aggregation boundary;
            # detaching avoids retaining an autograd graph per training step.
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
