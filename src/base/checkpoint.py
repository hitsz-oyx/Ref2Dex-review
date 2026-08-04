from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch


EPOCH_PREFIX = "epoch_"
CHECKPOINT_SUFFIX = ".pt"
BEST_FILE = "best.pt"
LATEST_FILE = "latest.pt"


class CheckpointManager:
    def __init__(self, output_dir: str | Path, max_to_keep: int = 5) -> None:
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    def save(
        self,
        *,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: Any | None,
        best_metric: float | None,
        metadata: dict[str, Any],
        config: dict[str, Any],
        runner_state: dict[str, Any] | None = None,
        is_best: bool = False,
    ) -> Path:
        raw_model = unwrap_model(model)
        ckpt_path = self.root / format_step_checkpoint_file(step, epoch)
        payload = {
            "step": step,
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "metadata": metadata,
            "config": config,
            "runner_state": runner_state or {},
        }
        # Write atomically so aliases to an older checkpoint inode remain valid
        # when the same epoch filename is saved again later in the epoch.
        temp_path = ckpt_path.with_name(f".{ckpt_path.name}.tmp-{os.getpid()}")
        torch.save(payload, temp_path)
        os.replace(temp_path, ckpt_path)
        self._copy_checkpoint(ckpt_path, self.root / LATEST_FILE)
        if is_best:
            self._copy_checkpoint(ckpt_path, self.root / BEST_FILE)
        self._prune()
        return ckpt_path

    def latest_checkpoint(self) -> Path | None:
        latest = self.root / LATEST_FILE
        if latest.exists():
            return latest
        checkpoints = self._checkpoint_files()
        return checkpoints[-1] if checkpoints else None

    def _checkpoint_files(self) -> list[Path]:
        return sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_file() and checkpoint_sort_key(path) is not None
            ),
            key=lambda path: checkpoint_sort_key(path),
        )

    def _copy_checkpoint(self, source: Path, target: Path) -> None:
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        # Checkpoints can exceed 1 GB. ``latest`` and ``best`` are aliases, not
        # independent artifacts, so prefer a hard link on the same filesystem.
        # Fall back to a real copy for filesystems that do not support links.
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _prune(self) -> None:
        if self.max_to_keep <= 0:
            return
        checkpoints = self._checkpoint_files()
        while len(checkpoints) > self.max_to_keep:
            removable = checkpoints[0]
            removable.unlink(missing_ok=True)
            checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint != removable]


def resolve_checkpoint_file(path: str | Path) -> Path:
    """Resolve a checkpoint written by the current shared checkpoint format.

    Task-specific legacy layouts intentionally belong to their task package;
    this base layer only understands files plus the current ``latest.pt``,
    ``best.pt``, and ``epoch_XXXXXX.pt`` aliases.
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for alias in (LATEST_FILE, BEST_FILE):
            candidate = path / alias
            if candidate.exists():
                return candidate
        checkpoints = sorted(
            (
                candidate
                for candidate in path.iterdir()
                if candidate.is_file() and checkpoint_sort_key(candidate) is not None
            ),
            key=lambda candidate: checkpoint_sort_key(candidate),
        )
        if checkpoints:
            return checkpoints[-1]
    return path


def format_epoch_checkpoint_dir(epoch: int) -> str:
    return f"{EPOCH_PREFIX}{int(epoch):06d}"


def format_epoch_checkpoint_file(epoch: int) -> str:
    return f"{format_epoch_checkpoint_dir(epoch)}{CHECKPOINT_SUFFIX}"


def format_step_checkpoint_file(step: int, epoch: int) -> str:
    return f"step_{int(step):09d}_epoch_{int(epoch):06d}{CHECKPOINT_SUFFIX}"


def checkpoint_sort_key(path: Path) -> tuple[int, int] | None:
    stem = path.stem
    if stem.startswith("step_") and "_epoch_" in stem:
        step_text, epoch_text = stem[len("step_"):].split("_epoch_", 1)
        if step_text.isdigit() and epoch_text.isdigit():
            return (1, int(step_text))
    if stem.startswith(EPOCH_PREFIX):
        suffix = stem[len(EPOCH_PREFIX) :]
        if suffix.isdigit():
            return (0, int(suffix))
    return None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt_file = resolve_checkpoint_file(path)
    if not ckpt_file.exists():
        raise FileNotFoundError(ckpt_file)
    return torch.load(ckpt_file, map_location=map_location)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    while True:
        next_model = getattr(current, "module", None)
        if next_model is not None:
            current = next_model
            continue
        next_model = getattr(current, "_orig_mod", None)
        if next_model is not None:
            current = next_model
            continue
        return current
