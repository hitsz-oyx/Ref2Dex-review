"""Stable shared interfaces for task configuration, training, and evaluation."""

from .base_config import (
    BaseConfig,
    TaskConfig,
    load_config,
    save_config,
    set_config_default_if_not_explicit,
    task_config_from_dict,
)
from .checkpoint import (
    CheckpointManager,
    load_checkpoint,
    resolve_checkpoint_file,
    unwrap_model,
)
from .data import (
    make_file_split_dataloaders,
    make_dataloader_kwargs,
    make_worker_init_fn,
    split_items,
    resolve_data_path,
)
from .distributed import (
    DistributedSamplerAdapter,
    DistributedState,
    barrier,
    cleanup_distributed,
    distributed_enabled,
    init_distributed,
    make_default_eval_sampler,
    make_default_train_sampler,
    reduce_dict,
    shard_sampler_for_distributed,
    wrap_model_for_distributed,
)
from .metrics import MetricAverager, MetricStat
from .performance import PerformanceMonitor
from .run_manifest import build_run_manifest, write_run_manifest, write_run_summary
from .utils import (
    JsonlLogger,
    format_seconds,
    import_from_path,
    resolve_device,
    resolve_optimizer,
    set_seed,
    to_jsonable,
)
from .base_runner import (
    BaseRunner,
    RunnerOutput,
    build_runner,
    build_runner_from_checkpoint,
    move_to_device,
    resolve_runner_class,
)
from .component import Component, ComponentSpec, ManifestError, PortSpec, TrainableComponent, compatible_ports, load_manifest, resolve_entrypoint
from .registry import ComponentRegistry, RegistryError, check_task_config
from .artifact import Artifact, ArtifactRef
from .context import ExecutionContext
from .pipeline import PipelineEdge, PipelineError, PipelineNode, PipelineSpec
from .contract import Contract

__all__ = [
    "BaseConfig",
    "TaskConfig",
    "task_config_from_dict",
    "load_config",
    "save_config",
    "set_config_default_if_not_explicit",
    "CheckpointManager",
    "load_checkpoint",
    "resolve_checkpoint_file",
    "unwrap_model",
    "make_file_split_dataloaders",
    "make_dataloader_kwargs",
    "make_worker_init_fn",
    "split_items",
    "resolve_data_path",
    "DistributedSamplerAdapter",
    "DistributedState",
    "barrier",
    "cleanup_distributed",
    "distributed_enabled",
    "init_distributed",
    "make_default_eval_sampler",
    "make_default_train_sampler",
    "reduce_dict",
    "shard_sampler_for_distributed",
    "wrap_model_for_distributed",
    "JsonlLogger",
    "MetricAverager",
    "MetricStat",
    "PerformanceMonitor",
    "build_run_manifest",
    "write_run_manifest",
    "write_run_summary",
    "format_seconds",
    "import_from_path",
    "resolve_device",
    "resolve_optimizer",
    "set_seed",
    "to_jsonable",
    "BaseRunner",
    "RunnerOutput",
    "build_runner",
    "build_runner_from_checkpoint",
    "move_to_device",
    "resolve_runner_class",
    "ComponentSpec",
    "Component",
    "TrainableComponent",
    "ManifestError",
    "PortSpec",
    "compatible_ports",
    "load_manifest",
    "resolve_entrypoint",
    "ComponentRegistry",
    "RegistryError",
    "check_task_config",
    "Artifact",
    "ArtifactRef",
    "ExecutionContext",
    "PipelineEdge",
    "PipelineError",
    "PipelineNode",
    "PipelineSpec",
    "Contract",
]
