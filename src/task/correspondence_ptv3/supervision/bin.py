from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from src.task.correspondence_ptv3.contracts import ContactLossResult
from src.utils.correspondence import contact_prob_to_bins, decode_contact_logits

from .common import reduce_loss_map_per_object


@dataclass(frozen=True)
class BinContactSupervision:
    num_contact_bins: int
    decode_mode: str
    contact_bin_weights: list[float] | None
    edge_contact_bin_weights: list[float] | None
    contact_bin_weight_path: str | None
    name: str = "bin"

    @property
    def output_dim(self) -> int:
        return int(self.num_contact_bins)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return decode_contact_logits(
            logits,
            supervision_mode="bin",
            mode=self.decode_mode,
        )

    def export_compat(
        self,
        logits: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        del logits
        return self.safe_logit_from_prob(probability)

    def legacy_output_aliases(
        self,
        *,
        pred_obj_contact_logits: torch.Tensor,
        pred_cross_contact_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "pred_obj_contact_bin": pred_obj_contact_logits,
            "pred_cross_contact_bin": pred_cross_contact_logits,
        }

    def compute_object_loss(
        self,
        *,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> ContactLossResult:
        target_bin = contact_prob_to_bins(
            target_probability,
            num_bins=self.num_contact_bins,
        )
        class_weight = self._get_contact_bin_weights(
            logits.device,
            logits.dtype,
            weights=self.contact_bin_weights,
        )
        num_classes = int(logits.shape[-1])
        loss_map = F.cross_entropy(
            logits.reshape(-1, num_classes),
            target_bin.reshape(-1),
            reduction="none",
            weight=class_weight,
        ).view_as(target_bin).float()
        ce = (loss_map * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
        return ContactLossResult(loss=ce, metrics={"ce": ce.detach()})

    def compute_edge_loss(
        self,
        *,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        edge_weight: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> ContactLossResult:
        target_bin = contact_prob_to_bins(
            target_probability,
            num_bins=self.num_contact_bins,
        )
        class_weight = self._get_contact_bin_weights(
            logits.device,
            logits.dtype,
            weights=self.edge_contact_bin_weights,
            fallback_weights=self.contact_bin_weights,
        )
        num_classes = int(logits.shape[-1])
        loss_map = F.cross_entropy(
            logits.reshape(-1, num_classes),
            target_bin.reshape(-1),
            reduction="none",
            weight=class_weight,
        ).view_as(target_bin).float()
        ce = reduce_loss_map_per_object(loss_map, edge_weight, obj_valid_mask)
        return ContactLossResult(loss=ce, metrics={"ce": ce.detach()})

    def _get_contact_bin_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        weights: list[float] | None,
        fallback_weights: list[float] | None = None,
    ) -> torch.Tensor | None:
        raw = weights
        if raw is None:
            raw = fallback_weights
        if raw is None:
            raw = self._load_default_contact_bin_weights()
        if raw is None:
            return None
        weight = torch.as_tensor(raw, device=device, dtype=dtype)
        if weight.numel() != int(self.num_contact_bins):
            raise ValueError(
                "contact_bin_weights must match num_contact_bins, got "
                f"{weight.numel()} vs {int(self.num_contact_bins)}."
            )
        return weight

    def _load_default_contact_bin_weights(self) -> list[float] | None:
        if not self.contact_bin_weight_path:
            return None
        path = Path(str(self.contact_bin_weight_path)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"contact_bin_weight_path does not exist: {path}")
        return [
            float(item)
            for item in path.read_text(encoding="utf-8").split()
            if item.strip()
        ]

    @staticmethod
    def safe_logit_from_prob(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        prob = prob.clamp(min=eps, max=1.0 - eps)
        return torch.log(prob) - torch.log1p(-prob)
