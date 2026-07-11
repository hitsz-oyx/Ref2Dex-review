from __future__ import annotations

import copy
import json
import math
import re
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
    set_config_default_if_not_explicit,
    task_config_from_dict,
)
from .checkpoint import CheckpointManager, load_checkpoint, unwrap_model
from .distributed import (
    DistributedState,
    barrier,
    init_distributed,
    reduce_dict,
    wrap_model_for_distributed,
)
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

    @classmethod
    def configure_overfit_mode(
        cls,
        cfg: TaskConfig,
        explicit_override_keys: set[str],
    ) -> None:
        """Apply generic overfit-diagnosis defaults without overriding explicit CLI values."""
        set_config_default_if_not_explicit(
            cfg,
            key="data.shuffle",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="data.drop_last",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="data.num_workers",
            value=0,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="data.persistent_workers",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.weight_decay",
            value=0.0,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.scheduler",
            value=None,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.warmup_ratio",
            value=0.0,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.warmup_steps",
            value=0,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.grad_clip_norm",
            value=None,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.amp",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.compile",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="train.early_stopping_patience",
            value=None,
            explicit_override_keys=explicit_override_keys,
        )

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
        self.explicit_override_keys = set(
            getattr(self.cfg, "_explicit_override_keys", set()) or set()
        )
        if self.mode == "train" and bool(getattr(self.cfg.train, "overfit_mode", False)):
            type(self).configure_overfit_mode(
                self.cfg,
                self.explicit_override_keys,
            )

        self.distributed: DistributedState = init_distributed(self.cfg.train)
        self.is_primary = self.distributed.is_primary
        self.run_name, self.output_dir = self._resolve_run_identity()
        self.cfg.train.output_dir = str(self.output_dir)
        self.seed = int(cfg.train.seed)
        self.process_seed = set_seed(self.seed + self.distributed.rank)
        self.device = resolve_device(
            cfg.train.device,
            local_rank=self.distributed.local_rank if self.distributed.enabled else None,
        )
        self.metadata: dict[str, Any] = {}
        self.train_loader = None
        self.val_loader = None
        self.val_loaders: dict[str, Any] = {}
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.cuda.amp.GradScaler | None = None
        self.checkpoints: CheckpointManager | None = None
        self.jsonl: JsonlLogger | None = None
        self.wandb_run: Any | None = None
        self.global_step = 0
        self.start_epoch = 0
        self.total_steps = 0
        self.resolved_warmup_steps = 0
        self.resolved_warmup_ratio = 0.0
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
            else:
                if checkpoint is None:
                    raise ValueError("eval mode requires a checkpoint.")
                self._setup_eval(checkpoint)

    def run(self) -> dict[str, float]:
        if self.mode == "train":
            return self.learn()
        metrics = self.evaluate_all()
        if self.is_primary:
            for key, value in metrics.items():
                print(f"{key}: {value:.6g}")
        return metrics

    def learn(self) -> dict[str, float]:
        self._require_train_ready()
        start_time = time.time()
        last_metrics: dict[str, float] = {}
        final_epoch = self.start_epoch
        last_eval_epoch: int | None = None
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            final_epoch = epoch + 1
            train_metrics = self.train_epoch(epoch)
            last_metrics.update(train_metrics)

            if (
                self.val_loaders
                and self.cfg.train.eval_every_steps is None
                and self._epoch_due(getattr(self.cfg.train, "eval_every_epochs", 1), epoch + 1)
            ):
                val_metrics = self.evaluate_all()
                last_metrics.update(val_metrics)
                self._record_metrics(val_metrics, epoch + 1)
                self._save_if_best(val_metrics, epoch + 1)
                last_eval_epoch = epoch + 1
                if self._check_early_stopping(val_metrics, epoch + 1):
                    break

            if (epoch + 1) % self.cfg.train.save_every_epochs == 0:
                self.save(epoch=epoch + 1, is_best=False)

            if self.global_step >= self.total_steps:
                break

        if self.val_loaders and last_eval_epoch != final_epoch:
            val_metrics = self.evaluate_all()
            last_metrics.update(val_metrics)
            self._record_metrics(val_metrics, final_epoch)
            self._save_if_best(val_metrics, final_epoch)
        self.save(epoch=final_epoch, is_best=False)
        elapsed = format_seconds(time.time() - start_time)
        if self._early_stopping_triggered:
            if self.is_primary:
                print(f"Training early stopped at step {self.global_step} in {elapsed}.")
        else:
            if self.is_primary:
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
                logged = {
                    f"train_step/{key}": value
                    for key, value in self._reduce_step_metrics(metrics).items()
                }
                self._record_metrics(logged, epoch + 1)

            if self.val_loaders and self._step_due(self.cfg.train.eval_every_steps):
                val_metrics = self.evaluate_all()
                self._record_metrics(val_metrics, epoch + 1)
                self._save_if_best(val_metrics, epoch + 1)

            if self._step_due(self.cfg.train.save_every_steps):
                self.save(epoch=epoch + 1, is_best=False)

        epoch_metrics = self._compute_averager_metrics(averager, prefix="train_epoch/")
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
        val_loaders = self._resolve_val_loaders()
        if prefix not in val_loaders:
            raise ValueError(
                f"Validation loader for prefix {prefix!r} is not available. "
                f"Available prefixes: {sorted(val_loaders)}"
            )
        return self.evaluate_loader(val_loaders[prefix], prefix=prefix)

    def evaluate_loader(self, loader: Any, *, prefix: str) -> dict[str, float]:
        if loader is None:
            raise ValueError("Validation loader is None.")
        was_training = self.model.training
        self.eval_mode()
        averager = MetricAverager()
        for batch in loader:
            batch = self.prepare_batch(batch)
            with self.eval_context():
                output = self.step(self.model, batch, mode="eval")
            metrics = dict(output.metrics)
            metrics.setdefault("loss", float(output.loss.detach().cpu()))
            averager.update(metrics, n=output.batch_size or self.batch_size(batch))
        if was_training:
            self.train_mode()
        return self._compute_averager_metrics(averager, prefix=prefix)

    def evaluate_all(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for prefix, loader in self._resolve_val_loaders().items():
            metrics.update(self.evaluate_loader(loader, prefix=prefix))
        return metrics

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
        if not self.is_primary:
            barrier()
            return self.checkpoints.root / "latest.pt"
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
        barrier()
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
        if self.is_primary:
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
        if self.is_primary:
            save_config(self.cfg, self.output_dir / "config.json")
        barrier()
        dataloader_bundle = self.make_dataloaders(self.cfg.data, seed=self.seed)
        self.train_loader, self.val_loader, self.metadata, self.val_loaders = (
            self._unpack_dataloader_bundle(dataloader_bundle)
        )
        self.configure_data(self.metadata, self.train_loader.dataset)
        self.total_steps = self._resolve_total_steps()
        self.resolved_warmup_steps = self._resolve_warmup_steps(self.total_steps)
        self.resolved_warmup_ratio = (
            float(self.resolved_warmup_steps) / float(self.total_steps)
            if self.total_steps > 0
            else 0.0
        )
        self.metadata.setdefault("run_name", self.run_name)
        self.metadata.update(
            {
                "total_steps": int(self.total_steps),
                "warmup_steps": int(self.resolved_warmup_steps),
                "warmup_ratio": float(self.resolved_warmup_ratio),
            }
        )
        self._write_metadata()
        self.model = self.build_model(self.cfg.model).to(self.device)
        if self.cfg.train.compile:
            self.model = torch.compile(self.model)
        self.model = wrap_model_for_distributed(
            self.model,
            device=self.device,
            train_cfg=self.cfg.train,
            state=self.distributed,
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler(self.total_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.train.amp and self.device.type == "cuda")
        self.checkpoints = CheckpointManager(self.output_dir, max_to_keep=self.cfg.train.max_to_keep)
        self.jsonl = JsonlLogger(self.output_dir / "metrics.jsonl") if self.is_primary else None
        self.wandb_run = self._build_wandb_run()
        self._log_train_setup()

    def _setup_eval(self, checkpoint: str | Path) -> None:
        dataloader_bundle = self.make_dataloaders(self.cfg.data, seed=self.seed)
        self.train_loader, self.val_loader, self.metadata, self.val_loaders = (
            self._unpack_dataloader_bundle(dataloader_bundle)
        )
        if not self.val_loaders:
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
        warmup = self._resolve_warmup_steps(total_steps)
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

    def _resolve_warmup_steps(self, total_steps: int) -> int:
        ratio = getattr(self.cfg.train, "warmup_ratio", None)
        if ratio is not None:
            ratio = float(ratio)
            if not 0.0 <= ratio < 1.0:
                raise ValueError(f"train.warmup_ratio must be in [0, 1), got {ratio}")
            warmup_steps = int(round(total_steps * ratio))
            if ratio > 0.0:
                warmup_steps = max(1, warmup_steps)
            return min(warmup_steps, max(0, total_steps - 1))
        return min(max(0, int(self.cfg.train.warmup_steps)), max(0, total_steps - 1))

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
            if self.is_primary:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"({self.epochs_without_improvement} epochs, {self.evals_without_improvement} evals without improvement)."
                )
            return True
        return False

    def _record_metrics(self, metrics: dict[str, float], epoch: int) -> None:
        payload = {"step": self.global_step, "epoch": epoch, **metrics}
        if self.jsonl is not None:
            self.jsonl.write(payload)
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=self.global_step)
        if self.is_primary:
            message = " ".join(f"{key}={value:.6g}" for key, value in metrics.items())
            print(f"step={self.global_step:06d} epoch={epoch:03d} {message}")

    def _build_wandb_run(self) -> Any | None:
        if not self.cfg.wandb.enable or not self.is_primary:
            return None
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("W&B logging is enabled, but wandb is not installed. Install with `pip install wandb`.") from exc

        init_kwargs: dict[str, Any] = {
            "project": self.cfg.wandb.project,
            "entity": self.cfg.wandb.entity,
            "group": self.cfg.wandb.group,
            "name": self.cfg.wandb.name or self.run_name,
            "tags": self.cfg.wandb.tags,
            "job_type": self.cfg.wandb.job_type,
            "dir": str(self.output_dir),
            "config": to_jsonable(self.cfg.to_dict()),
            "settings": wandb.Settings(init_timeout=120),
        }
        if self.cfg.wandb.mode is not None:
            init_kwargs["mode"] = self.cfg.wandb.mode
        return wandb.init(**init_kwargs)

    def _log_checkpoint_artifact(self, checkpoint_path: Path) -> None:
        try:
            import wandb
        except ImportError:
            return
        artifact = wandb.Artifact(f"{self.run_name}-best", type="model")
        if checkpoint_path.is_dir():
            artifact.add_dir(str(checkpoint_path))
        else:
            artifact.add_file(str(checkpoint_path))
        self.wandb_run.log_artifact(artifact)

    def _write_metadata(self) -> None:
        if not self.is_primary:
            barrier()
            return
        path = self.output_dir / "metadata.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(self.metadata), f, indent=2, ensure_ascii=False)
        barrier()

    def _reduce_step_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return reduce_dict(metrics, device=self.device, average=True)

    def _compute_averager_metrics(self, averager: MetricAverager, prefix: str = "") -> dict[str, float]:
        totals = {key: meter.total for key, meter in averager.meters.items()}
        counts = {key: meter.count for key, meter in averager.meters.items()}
        totals = reduce_dict(totals, device=self.device, average=False)
        counts = reduce_dict(counts, device=self.device, average=False)
        return {
            f"{prefix}{key}": totals[key] / max(1.0, counts.get(key, 0.0))
            for key in sorted(totals)
        }

    def _log_train_setup(self) -> None:
        if not self.is_primary:
            return
        per_device_batch = int(self.cfg.data.batch_size)
        global_batch = per_device_batch * self.distributed.world_size
        val_batch = int(getattr(self.cfg.data, "val_batch_size", None) or per_device_batch)
        print(
            "train_setup "
            f"run_name={self.run_name} "
            f"device={self.device} world_size={self.distributed.world_size} "
            f"per_device_batch={per_device_batch} global_batch={global_batch} "
            f"val_batch={val_batch} num_workers={int(self.cfg.data.num_workers)} "
            f"amp={bool(self.cfg.train.amp)} total_steps={int(self.total_steps)} "
            f"warmup_steps={int(self.resolved_warmup_steps)} "
            f"warmup_ratio={self.resolved_warmup_ratio:.6f} output_dir={self.output_dir}"
        )

    def _resolve_run_identity(self) -> tuple[str, Path]:
        if self.mode != "train":
            run_name = self._slugify(
                str(getattr(self.cfg.wandb, "name", None) or getattr(self.cfg, "name", "task") or "task")
            )
            return run_name, Path(self.cfg.train.output_dir)

        explicit_output_dir = bool(getattr(self.cfg.train, "_explicit_output_dir", False))
        explicit_cfg_name = bool(getattr(self.cfg, "_explicit_name", False))
        explicit_wandb_name = bool(getattr(self.cfg.wandb, "_explicit_name", False))

        task_slug = self._task_slug()
        if explicit_wandb_name and str(getattr(self.cfg.wandb, "name", "")).strip():
            run_name = self._slugify(str(self.cfg.wandb.name))
        elif explicit_cfg_name and str(getattr(self.cfg, "name", "")).strip():
            run_name = self._slugify(str(self.cfg.name))
        else:
            run_name = f"{task_slug}_{self._shared_timestamp()}"

        output_value = str(getattr(self.cfg.train, "output_dir", "") or "").strip()
        output_root = Path(output_value) if output_value else Path("outputs/train")
        if not explicit_output_dir and output_root.name == task_slug:
            output_root = output_root.parent
        if getattr(self.cfg.train, "resume", None) is not None and not explicit_output_dir:
            output_dir = Path(output_value) if output_value else output_root / run_name
        else:
            output_dir = output_root if explicit_output_dir else output_root / run_name

        if not explicit_wandb_name:
            self.cfg.wandb.name = run_name
        return run_name, output_dir

    def _task_slug(self) -> str:
        runner_class = str(getattr(self.cfg, "runner_class", "") or "").strip()
        parts = runner_class.split(".")
        if len(parts) >= 3 and parts[0] == "src" and parts[1] == "task":
            return self._slugify(parts[2])
        return self._slugify(str(getattr(self.cfg, "name", "task") or "task"))

    def _shared_timestamp(self) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) if self.is_primary else None
        if self.distributed.enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
            payload = [stamp]
            torch.distributed.broadcast_object_list(payload, src=0)
            stamp = payload[0]
        return str(stamp)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        slug = slug.strip("._-")
        return slug or "task"

    def _unpack_dataloader_bundle(self, bundle: Any) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        if not isinstance(bundle, tuple):
            raise TypeError(
                "make_dataloaders() must return a tuple of "
                "(train_loader, val_loader, metadata) or "
                "(train_loader, val_loader, metadata, val_loaders)."
            )
        if len(bundle) == 3:
            train_loader, val_loader, metadata = bundle
            val_loaders = {"val/": val_loader} if val_loader is not None else {}
            return train_loader, val_loader, metadata, val_loaders
        if len(bundle) == 4:
            train_loader, val_loader, metadata, val_loaders = bundle
            resolved_val_loaders = {
                str(prefix): loader
                for prefix, loader in dict(val_loaders or {}).items()
                if loader is not None
            }
            if not resolved_val_loaders and val_loader is not None:
                resolved_val_loaders = {"val/": val_loader}
            return train_loader, val_loader, metadata, resolved_val_loaders
        raise ValueError(
            "make_dataloaders() returned an unexpected tuple length. "
            f"Expected 3 or 4 values, got {len(bundle)}."
        )

    def _resolve_val_loaders(self) -> dict[str, Any]:
        if self.val_loaders:
            return self.val_loaders
        if self.val_loader is not None:
            return {"val/": self.val_loader}
        return {}

    def _require_train_ready(self) -> None:
        self._require_model_ready()
        if self.train_loader is None or self.optimizer is None or self.scaler is None or self.checkpoints is None:
            raise RuntimeError("Runner is not initialized for training.")

    def _require_eval_ready(self) -> None:
        self._require_model_ready()
        if not self._resolve_val_loaders():
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
