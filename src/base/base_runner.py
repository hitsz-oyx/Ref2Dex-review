from __future__ import annotations

import copy
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .base_config import (
    TaskConfig,
    load_config,
    save_config,
    task_config_from_dict,
)
from .checkpoint import CheckpointManager, load_checkpoint, unwrap_model
from .utils import (
    JsonlLogger,
    MetricAverager,
    format_seconds,
    import_from_path,
    resolve_device,
    resolve_optimizer,
    set_seed,
    to_jsonable,
)


@dataclass
class RunnerOutput:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    batch_size: int | None = None


class BaseRunner:
    """Runner with train/eval control flow plus overridable data, model, loss, and inference hooks."""

    def __init__(
        self,
        cfg: TaskConfig,
        mode: str = "train",
        checkpoint: str | Path | None = None,
        device: str | None = None,
        build_data: bool = True,
    ) -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(f"Runner mode must be 'train' or 'eval', got {mode!r}.")
        self.cfg = cfg
        self.mode = mode
        if device is not None:
            self.cfg.train.device = device

        self.output_dir = Path(cfg.train.output_dir)
        self.seed = set_seed(cfg.train.seed)
        self.device = resolve_device(cfg.train.device)
        self.metadata: dict[str, Any] = {}
        self.train_loader = None
        self.val_loader = None
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.cuda.amp.GradScaler | None = None
        self.checkpoints: CheckpointManager | None = None
        self.jsonl: JsonlLogger | None = None
        self.wandb_run: Any | None = None
        self.global_step = 0
        self.start_epoch = 0
        self.best_metric: float | None = None
        self.train_dataset: Any | None = None
        self.epochs_without_improvement = 0
        self.evals_without_improvement = 0
        self._early_stopping_triggered = False

        if build_data:
            if mode == "train":
                self._setup_train()
                if cfg.train.resume:
                    self.load(cfg.train.resume, load_optimizer=True)
                if cfg.train.compile and self.model is not None:
                    self.model = torch.compile(self.model)
            else:
                if checkpoint is None:
                    raise ValueError("eval mode requires a checkpoint.")
                self._setup_eval(checkpoint)

    def run(self) -> dict[str, float]:
        if self.mode == "train":
            return self.learn()
        metrics = self.evaluate(prefix="val/")
        for key, value in metrics.items():
            print(f"{key}: {value:.6g}")
        return metrics

    def learn(self) -> dict[str, float]:
        self._require_train_ready()
        start_time = time.time()
        last_metrics: dict[str, float] = {}
        final_epoch = self.start_epoch
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            final_epoch = epoch + 1
            train_metrics = self.train_epoch(epoch)
            last_metrics.update(train_metrics)

            if (
                self.val_loader is not None
                and self.cfg.train.eval_every_steps is None
                and self._epoch_due(getattr(self.cfg.train, "eval_every_epochs", 1), epoch + 1)
            ):
                val_metrics = self.evaluate(prefix="val/")
                last_metrics.update(val_metrics)
                self._record_metrics(last_metrics, epoch + 1)
                self._save_if_best(val_metrics, epoch + 1)
                if self._check_early_stopping(val_metrics, epoch + 1):
                    break

            if (epoch + 1) % self.cfg.train.save_every_epochs == 0:
                self.save(epoch=epoch + 1, is_best=False)

            if self.global_step >= self.total_steps:
                break

        if self.val_loader is not None:
            val_metrics = self.evaluate(prefix="val/")
            last_metrics.update(val_metrics)
            self._save_if_best(val_metrics, final_epoch)
        self.save(epoch=final_epoch, is_best=False)
        elapsed = format_seconds(time.time() - start_time)
        if self._early_stopping_triggered:
            print(f"Training early stopped at step {self.global_step} in {elapsed}.")
        else:
            print(f"Training finished at step {self.global_step} in {elapsed}.")
        if self.wandb_run is not None:
            self.wandb_run.finish()
        return last_metrics

    fit = learn

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self._require_train_ready()
        self.train_mode()
        averager = MetricAverager()
        for batch in self.train_loader:
            if self.global_step >= self.total_steps:
                break
            batch = self.prepare_batch(batch)
            metrics = self.train_step(batch)
            averager.update(metrics, n=self.batch_size(batch))
            self.global_step += 1

            if self.global_step % self.cfg.train.log_every_steps == 0:
                logged = {f"train/{key}": value for key, value in metrics.items()}
                self._record_metrics(logged, epoch + 1)

            if self.val_loader is not None and self._step_due(self.cfg.train.eval_every_steps):
                val_metrics = self.evaluate(prefix="val/")
                self._record_metrics(val_metrics, epoch + 1)
                self._save_if_best(val_metrics, epoch + 1)

            if self._step_due(self.cfg.train.save_every_steps):
                self.save(epoch=epoch + 1, is_best=False)

        epoch_metrics = averager.compute(prefix="train/")
        self._record_metrics(epoch_metrics, epoch + 1)
        return epoch_metrics

    def train_step(self, batch: Any) -> dict[str, float]:
        self._require_train_ready()
        self.optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, enabled=True)
            if self.cfg.train.amp and self.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            output = self.step(self.model, batch, mode="train")
            loss = output.loss

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_clip_norm = self.cfg.train.grad_clip_norm
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            if grad_clip_norm is not None
            else compute_grad_norm(self.model.parameters())
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

        metrics = dict(output.metrics)
        metrics.setdefault("loss", float(loss.detach().cpu()))
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        grad_norm_value = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
        metrics["grad_norm"] = grad_norm_value
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
            metrics["grad_clip_threshold"] = grad_clip_norm
            metrics["grad_clip_ratio"] = grad_norm_value / max(grad_clip_norm, 1e-12)
            metrics["grad_clipped"] = float(grad_norm_value > grad_clip_norm)
        return metrics

    def evaluate(self, prefix: str = "val/") -> dict[str, float]:
        self._require_eval_ready()
        if self.val_loader is None:
            raise ValueError("No validation dataset is available. Set data.val_path or data.val_split > 0.")
        was_training = self.model.training
        self.eval_mode()
        averager = MetricAverager()
        for batch in self.val_loader:
            batch = self.prepare_batch(batch)
            with self.eval_context():
                output = self.step(self.model, batch, mode="eval")
            metrics = dict(output.metrics)
            metrics.setdefault("loss", float(output.loss.detach().cpu()))
            averager.update(metrics, n=output.batch_size or self.batch_size(batch))
        if was_training:
            self.train_mode()
        return averager.compute(prefix=prefix)

    def make_dataloaders(self, data_cfg, seed: int):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement make_dataloaders(). "
            "BaseRunner does not assume task-specific sample keys or dataset formats."
        )

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        self.metadata = metadata
        self.train_dataset = train_dataset

    def build_model(self, model_cfg) -> torch.nn.Module:
        from .models import build_model

        class_path = getattr(model_cfg, "class_path", None)
        if class_path:
            return self.build_model_from_config(model_cfg)
        return build_model(model_cfg, self.metadata)

    def build_model_from_config(self, model_cfg, **kwargs: Any) -> torch.nn.Module:
        model_cls = import_from_path(model_cfg.class_path)
        model_cfg = copy.copy(model_cfg)
        model_cfg.meta = self.cfg.meta
        return model_cls(model_cfg, **kwargs)

    def eval_context(self):
        return torch.enable_grad() if self.eval_requires_grad() else torch.no_grad()

    def eval_requires_grad(self) -> bool:
        return False

    def prepare_batch(self, batch: Any) -> Any:
        return move_to_device(batch, self.device)

    def step(self, model: torch.nn.Module, batch: Any, mode: str = "train") -> RunnerOutput:
        raise NotImplementedError

    def inference(self, model: torch.nn.Module, inputs: Any) -> Any:
        inputs = self.prepare_batch(inputs)
        return model(inputs)

    def batch_size(self, batch: Any) -> int:
        if isinstance(batch, dict):
            for value in batch.values():
                if torch.is_tensor(value):
                    return int(value.shape[0])
        if torch.is_tensor(batch):
            return int(batch.shape[0])
        raise ValueError("Cannot infer batch size for this runner. Override BaseRunner.batch_size().")

    def state_dict(self) -> dict[str, Any]:
        return {
            "epochs_without_improvement": self.epochs_without_improvement,
            "evals_without_improvement": self.evals_without_improvement,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict:
            self.epochs_without_improvement = int(state_dict.get("epochs_without_improvement", 0))
            self.evals_without_improvement = int(state_dict.get("evals_without_improvement", 0))

    def train_mode(self) -> None:
        if self.model is not None:
            self.model.train()

    def eval_mode(self) -> None:
        if self.model is not None:
            self.model.eval()

    def save(self, epoch: int, is_best: bool = False) -> Path:
        self._require_train_ready()
        path = self.checkpoints.save(
            step=self.global_step,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            best_metric=self.best_metric,
            metadata=self.metadata,
            config=self.cfg.to_dict(),
            runner_state=self.state_dict(),
            is_best=is_best,
        )
        if self.wandb_run is not None and self.cfg.wandb.log_model and is_best:
            self._log_checkpoint_artifact(path)
        return path

    def load(
        self,
        path: str | Path,
        load_optimizer: bool = True,
        map_location: str | torch.device | None = None,
    ) -> dict[str, Any]:
        ckpt_path = self._resolve_resume_path(path)
        checkpoint = load_checkpoint(ckpt_path, map_location=map_location or self.device)
        self._load_checkpoint_payload(checkpoint, load_optimizer=load_optimizer)
        print(f"Loaded checkpoint from {ckpt_path} at step {self.global_step}.")
        return checkpoint

    resume = load

    def setup_inference(self, checkpoint: str | Path) -> dict[str, Any]:
        checkpoint_data = load_checkpoint(checkpoint, map_location="cpu")
        self.metadata = checkpoint_data["metadata"]
        self.configure_data(self.metadata, None)
        self.model = self.build_model(self.cfg.model).to(self.device)
        self._load_checkpoint_payload(checkpoint_data, load_optimizer=False)
        self.eval_mode()
        return checkpoint_data

    def _setup_train(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config(self.cfg, self.output_dir / "config.json")
        self.train_loader, self.val_loader, self.metadata = self.make_dataloaders(self.cfg.data, seed=self.seed)
        self.configure_data(self.metadata, self.train_loader.dataset)
        self._write_metadata()
        self.model = self.build_model(self.cfg.model).to(self.device)
        self.optimizer = self._build_optimizer()
        self.total_steps = self._resolve_total_steps()
        self.scheduler = self._build_scheduler(self.total_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.train.amp and self.device.type == "cuda")
        self.checkpoints = CheckpointManager(self.output_dir, max_to_keep=self.cfg.train.max_to_keep)
        self.jsonl = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.wandb_run = self._build_wandb_run()

    def _setup_eval(self, checkpoint: str | Path) -> None:
        self.train_loader, self.val_loader, self.metadata = self.make_dataloaders(self.cfg.data, seed=self.seed)
        if self.val_loader is None:
            raise ValueError("No validation dataset is available. Set data.val_path or data.val_split > 0.")
        self.configure_data(self.metadata, self.train_loader.dataset)
        self.model = self.build_model(self.cfg.model).to(self.device)
        self.load(checkpoint, load_optimizer=False, map_location=self.device)
        self.eval_mode()

    def _load_checkpoint_payload(self, checkpoint: dict[str, Any], load_optimizer: bool) -> None:
        self._require_model_ready()
        unwrap_model(self.model).load_state_dict(checkpoint["model"])
        if load_optimizer and self.optimizer is not None and checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if load_optimizer and self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if load_optimizer and self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint.get("runner_state")
        if state is not None:
            self.load_state_dict(state)
        self.global_step = int(checkpoint.get("step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_metric = checkpoint.get("best_metric")
        if state is None:
            self.epochs_without_improvement = 0
            self.evals_without_improvement = 0
        self._early_stopping_triggered = False

    def _resolve_resume_path(self, path: str | Path) -> Path:
        if path == "auto":
            if self.checkpoints is None:
                raise ValueError("resume='auto' is only available after train setup.")
            ckpt_path = self.checkpoints.latest_checkpoint()
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoint found in {self.checkpoints.root}")
            return ckpt_path
        return Path(path)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cls = resolve_optimizer(self.cfg.train.optimizer)
        return optimizer_cls(self.model.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        name = self.cfg.train.scheduler
        if name is None or str(name).lower() in {"none", "null"}:
            return None
        name = str(name).lower()
        warmup = max(0, self.cfg.train.warmup_steps)
        if name == "cosine":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: cosine_schedule(step, total_steps=total_steps, warmup_steps=warmup),
            )
        if name == "linear":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: linear_schedule(step, total_steps=total_steps, warmup_steps=warmup),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(1, total_steps // 3), gamma=0.1)
        raise ValueError(f"Unknown scheduler '{self.cfg.train.scheduler}'. Use cosine, linear, step, or null.")

    def _resolve_total_steps(self) -> int:
        if self.cfg.train.max_steps is not None:
            return int(self.cfg.train.max_steps)
        return max(1, self.cfg.train.epochs * len(self.train_loader))

    def _step_due(self, interval: int | None) -> bool:
        return interval is not None and interval > 0 and self.global_step % interval == 0

    def _epoch_due(self, interval: int | None, epoch: int) -> bool:
        return interval is not None and interval > 0 and epoch % interval == 0

    def _save_if_best(self, metrics: dict[str, float], epoch: int) -> None:
        key = self.cfg.train.metric_for_best
        if key not in metrics:
            return
        value = float(metrics[key])
        if self.best_metric is None:
            improved = True
        elif self.cfg.train.lower_is_better:
            improved = value < self.best_metric
        else:
            improved = value > self.best_metric
        if improved:
            self.best_metric = value
            self.save(epoch=epoch, is_best=True)

    def _check_early_stopping(self, metrics: dict[str, float], epoch: int) -> bool:
        patience = self.cfg.train.early_stopping_patience
        if patience is None or patience <= 0:
            return False

        key = self.cfg.train.metric_for_best
        if key not in metrics:
            return False

        value = float(metrics[key])
        if self.best_metric is None:
            self.epochs_without_improvement = 0
            self.evals_without_improvement = 0
            return False

        threshold = float(self.cfg.train.early_stopping_threshold)
        if self.cfg.train.lower_is_better:
            improved = value < (self.best_metric - threshold)
        else:
            improved = value > (self.best_metric + threshold)

        if improved:
            self.epochs_without_improvement = 0
            self.evals_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            self.evals_without_improvement += 1

        if self.epochs_without_improvement >= patience or self.evals_without_improvement >= patience:
            self._early_stopping_triggered = True
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"({self.epochs_without_improvement} epochs, {self.evals_without_improvement} evals without improvement)."
            )
            return True
        return False

    def _record_metrics(self, metrics: dict[str, float], epoch: int) -> None:
        payload = {"step": self.global_step, "epoch": epoch, **metrics}
        self.jsonl.write(payload)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=self.global_step)
        message = " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
        print(f"step={self.global_step:06d} epoch={epoch:03d} {message}")

    def _build_wandb_run(self) -> Any | None:
        if not self.cfg.wandb.enable:
            return None
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("W&B logging is enabled, but wandb is not installed. Install with `pip install wandb`.") from exc

        init_kwargs: dict[str, Any] = {
            "project": self.cfg.wandb.project,
            "entity": self.cfg.wandb.entity,
            "group": self.cfg.wandb.group,
            "name": self.cfg.wandb.name or self.cfg.name,
            "tags": self.cfg.wandb.tags,
            "job_type": self.cfg.wandb.job_type,
            "dir": str(self.output_dir),
            "config": to_jsonable(self.cfg.to_dict()),
        }
        if self.cfg.wandb.mode is not None:
            init_kwargs["mode"] = self.cfg.wandb.mode
        return wandb.init(**init_kwargs)

    def _log_checkpoint_artifact(self, checkpoint_path: Path) -> None:
        try:
            import wandb
        except ImportError:
            return
        artifact = wandb.Artifact(f"{self.cfg.name}-best", type="model")
        if checkpoint_path.is_dir():
            artifact.add_dir(str(checkpoint_path))
        else:
            artifact.add_file(str(checkpoint_path))
        self.wandb_run.log_artifact(artifact)

    def _write_metadata(self) -> None:
        path = self.output_dir / "metadata.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(self.metadata), f, indent=2, ensure_ascii=False)

    def _require_train_ready(self) -> None:
        self._require_model_ready()
        if self.train_loader is None or self.optimizer is None or self.scaler is None or self.checkpoints is None:
            raise RuntimeError("Runner is not initialized for training.")

    def _require_eval_ready(self) -> None:
        self._require_model_ready()
        if self.val_loader is None:
            raise RuntimeError("Runner is not initialized for evaluation.")

    def _require_model_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("Runner model is not initialized.")


def build_runner(
    cfg: TaskConfig,
    mode: str = "train",
    checkpoint: str | Path | None = None,
    device: str | None = None,
    build_data: bool = True,
) -> BaseRunner:
    runner_cls = resolve_runner_class(cfg)
    return runner_cls(cfg=cfg, mode=mode, checkpoint=checkpoint, device=device, build_data=build_data)


def build_runner_from_checkpoint(
    checkpoint: str | Path,
    config: str | Path | dict[str, Any] | TaskConfig | type[Any] | None = None,
    mode: str = "eval",
    device: str = "auto",
    build_data: bool = True,
) -> BaseRunner:
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    cfg = load_config(config) if config is not None else task_config_from_dict(ckpt["config"])
    cfg.train.device = device
    return build_runner(cfg, mode=mode, checkpoint=checkpoint, device=device, build_data=build_data)


def resolve_runner_class(cfg: TaskConfig) -> type[BaseRunner]:
    runner_path = getattr(cfg, "runner_class", cfg.name)

    if "." not in runner_path:
        raise ValueError(
            f"Runner path '{runner_path}' must be a full dotted Python path "
            "(e.g., 'wm.task.wm.runner.WMRunner')."
        )

    runner_cls = import_from_path(runner_path)
    if not issubclass(runner_cls, BaseRunner):
        raise TypeError(f"Runner class must inherit from base.base_runner.BaseRunner, got {runner_cls}.")
    return runner_cls


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def compute_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    grads = [param.grad.detach() for param in parameters if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    device = grads[0].device
    if norm_type == float("inf"):
        return torch.stack([grad.abs().max().to(device) for grad in grads]).max()
    norms = torch.stack([torch.linalg.vector_norm(grad, ord=norm_type).to(device) for grad in grads])
    return torch.linalg.vector_norm(norms, ord=norm_type)


def cosine_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_schedule(step: int, total_steps: int, warmup_steps: int = 0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 1.0 - progress)
