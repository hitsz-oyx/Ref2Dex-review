from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import yaml


class BaseConfig:
    """RoboGym 风格嵌套类配置的基类。

    自动将内部嵌套类实例化为成员对象，支持深度嵌套配置结构。
    """

    def __init__(self) -> None:
        self.init_member_classes(self)

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @staticmethod
    def init_member_classes(obj: Any) -> None:
        """遍历对象的所有属性：
        - 如果是 class，则实例化并递归初始化其成员
        - 如果是可序列化的值，则深拷贝一份，避免多个实例共享同一配置对象
        """
        for key in dir(obj):
            if key.startswith("_") or key == "__class__":
                continue
            value = getattr(obj, key)
            if inspect.isclass(value):
                member = value()
                setattr(obj, key, member)
                if not isinstance(member, BaseConfig):
                    BaseConfig.init_member_classes(member)
            elif _is_config_value(value):
                setattr(obj, key, copy.deepcopy(value))


class TaskConfig(BaseConfig):
    """任务配置模板。具体任务应继承此类并显式覆盖字段。"""

    name = "task"

    class meta(BaseConfig):
        pass

    class model(BaseConfig):
        pass

    class data(BaseConfig):
        train_path = ""
        val_path = None
        # ``test_path`` is intentionally explicit: test data must never be
        # sampled from training data or used by the training loop.
        test_path = None
        # Optional split-file interface.  When any split is set, ``root`` is
        # the one physical data directory and each text file lists paths
        # relative to it; files are never copied into train/val/test folders.
        root = ""
        # Canonical split interface.  The JSON file names the train/val/test
        # text lists, keeping one auditable split source per experiment.
        split_json_path = None
        train_split = None
        val_split_path = None
        test_split = None
        val_split = 0.1
        batch_size = 256
        val_batch_size = None
        test_batch_size = None
        num_workers = 0
        shuffle = True
        drop_last = False
        pin_memory = True
        persistent_workers = False
        prefetch_factor = 2

    class train(BaseConfig):
        # ========== 路径和随机种子 ==========
        # Run directories are always resolved as
        # ``outputs/<task>/<task>_<timestamp>/`` by BaseRunner.  This legacy
        # field is retained only so old configs can still be parsed.
        output_dir = ""
        # Human-readable context belongs in metadata, never in the path.
        description = ""
        seed = 42
        overfit_mode = False

        # ========== 设备和训练长度 ==========
        device = "auto"
        epochs = 50
        max_steps = None

        # ========== 优化器 ==========
        optimizer = "adamw"
        lr = 3e-4
        weight_decay = 1e-4

        # ========== 学习率调度 ==========
        scheduler = "cosine"
        warmup_ratio = None
        warmup_steps = 0
        # ``cosine_restart`` starts a fresh fine-tuning cosine phase after a
        # checkpoint from another scheduler. ``max_steps`` stays absolute.
        finetune_steps = None
        min_lr = None

        # ========== 梯度裁剪和混合精度 ==========
        grad_clip_norm = 1.0
        amp = False
        amp_dtype = "float16"
        compile = False

        # ========== 日志和评估 ==========
        # 默认 100 步一记，与 PerformanceMonitor 在每窗口只同步一次 CUDA 的策略匹配；
        # 设为 1 会让 light 模式每步都强制 ``cuda.synchronize``，破坏异步执行。
        # 任务 yaml 可显式覆盖（小样本 overfit 调试时可设为 1）。
        log_every_steps = 100
        eval_every_steps = None
        eval_every_epochs = 1

        # ========== 检查点保存 ==========
        save_every_steps = None
        save_every_epochs = 10
        max_to_keep = 5
        resume = None

        # ========== 早停 ==========
        # 早停耐心值，连续N个epoch或N次验证没有改善则停止训练（None=禁用）
        early_stopping_patience = None
        # 改善阈值，只有超过此阈值才认为真正改善
        early_stopping_threshold = 0.0

        # ========== 最佳模型选择 ==========
        metric_for_best = "val/loss"
        lower_is_better = True

        class distributed(BaseConfig):
            enable = False
            backend = "auto"
            timeout_minutes = 30
            broadcast_buffers = False
            find_unused_parameters = False

    class performance(BaseConfig):
        # ``off``：完全关闭，不记录 perf/* 也不启动 profiler。
        # ``light``：长期轻量统计，记录 step_ms / data_wait_ms / data_wait_ratio
        #           / samples_per_s / eta_hours。
        # ``profile``：在 light 之上叠加 ``torch.profiler`` 短期采集并写
        #              TensorBoard trace 到 ``<output_dir>/profiler/``。
        mode = "light"

        # 前 N 步不参与 throughput / ETA 估计，避免把 CUDA 初始化、worker
        # 启动和文件冷缓存算进正常吞吐。
        warmup_steps = 20

        # torch.profiler 的 schedule(wait / warmup / active) 参数。
        # 仅 ``mode == "profile"`` 时生效。
        profile_wait_steps = 5
        profile_warmup_steps = 5
        profile_active_steps = 10

    class wandb(BaseConfig):
        enable = True
        project = "sl-framework"
        entity = None
        group = None
        name = None
        tags = []
        mode = "offline"
        job_type = "train"
        log_model = False
        init_timeout = 600


class ConfigNode(BaseConfig):
    """通用配置节点，用于动态构建嵌套配置结构。"""
    pass


def load_config(config: str | Path | dict[str, Any] | TaskConfig | type[Any], overrides: list[str] | None = None) -> TaskConfig:
    """加载配置，支持路径字符串、文件路径、字典、配置对象或配置类。

    Args:
        config: 配置来源，支持多种格式（文件路径、字典、配置对象等）
        overrides: 命令行覆盖参数列表，格式为 ["key=value", ...]

    Returns:
        解析后的 TaskConfig 对象
    """
    raw = config_to_dict(config)
    if overrides:
        raw = copy.deepcopy(raw)
        for override in overrides:
            apply_override(raw, override)
    return task_config_from_dict(raw)


def save_config(cfg: TaskConfig, path: str | Path) -> None:
    """将配置保存为 JSON 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)


def task_config_from_dict(raw: dict[str, Any]) -> TaskConfig:
    """从字典构建 TaskConfig 对象。"""
    cfg = TaskConfig()
    _apply_mapping(cfg, raw)
    return cfg


def config_to_dict(config: str | Path | dict[str, Any] | BaseConfig | type[Any] | Any) -> dict[str, Any]:
    """将多种格式的配置统一转换为字典。

    支持：字典、路径/字符串（文件引用）、BaseConfig 对象/类，
    以及定义了 to_config() 或 to_dict() 方法的任意对象。
    """
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, Path)):
        return _load_config_reference(str(config))
    if isinstance(config, BaseConfig):
        return _object_to_dict(config)
    if isinstance(config, type) and issubclass(config, BaseConfig):
        return config().to_dict()
    if hasattr(config, "to_config") and callable(config.to_config):
        return config_to_dict(config.to_config())
    if hasattr(config, "to_dict") and callable(config.to_dict):
        value = config.to_dict()
        if not isinstance(value, dict):
            raise TypeError("config.to_dict() must return a dict.")
        return value
    return class_to_dict(config)


def class_to_dict(obj: Any) -> dict[str, Any] | Any:
    """将任意对象递归转换为可序列化的字典。

    处理 None、基本类型、Path、dict、list/tuple、BaseConfig 等类型，
    对于普通对象则提取所有非私有属性。
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(value) for value in obj]
    if isinstance(obj, BaseConfig):
        return obj.to_dict()
    if isinstance(obj, type) and issubclass(obj, BaseConfig):
        return obj().to_dict()

    result: dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        value = getattr(obj, key)
        if not _is_config_value(value) and not inspect.isclass(value):
            continue
        result[key] = class_to_dict(value)
    return result


def apply_override(raw: dict[str, Any], override: str) -> None:
    """将命令行覆盖参数（如 'data.batch_size=128'）应用到配置字典。

    支持点号分隔的嵌套路径，自动创建中间字典节点。
    """
    item = override.strip()
    if item.startswith("--"):
        item = item[2:]
    if "=" not in item:
        raise ValueError(f"Override must use key=value syntax, got: {override}")
    key, value_text = item.split("=", 1)
    value = parse_override_value(value_text)

    cursor: dict[str, Any] = raw
    parts = key.split(".")
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested override through non-dict key: {part}")
        cursor = next_value
    cursor[parts[-1]] = value


def parse_override_value(value_text: str) -> Any:
    """解析命令行覆盖参数的值字符串为 Python 对象。

    支持：bool（true/false）、None（none/null）、JSON 类型（数字、列表、字典等），
    以及普通字符串。
    """
    lowered = value_text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(value_text)
    except json.JSONDecodeError:
        return value_text


def set_config_default_if_not_explicit(
    cfg: Any,
    *,
    key: str,
    value: Any,
    explicit_override_keys: set[str] | None,
) -> bool:
    """Set a dotted config key only when the user did not explicitly override it.

    Returns:
        True if the default was applied, False if it was skipped due to an
        explicit override.
    """
    explicit_override_keys = set(explicit_override_keys or set())
    if key in explicit_override_keys:
        return False
    _set_dotted_attr(cfg, key, value)
    return True


def _apply_mapping(obj: Any, values: dict[str, Any]) -> None:
    """将字典中的键值对递归应用到配置对象上。

    对于字典类型的值，会递归创建/更新嵌套的 ConfigNode 或 BaseConfig 子对象。
    """
    for key, value in values.items():
        if isinstance(value, dict):
            child = getattr(obj, key, None)
            if isinstance(child, dict):
                setattr(obj, key, copy.deepcopy(value))
                continue
            if not isinstance(child, BaseConfig):
                child = ConfigNode()
            _apply_mapping(child, value)
            setattr(obj, key, child)
        else:
            setattr(obj, key, copy.deepcopy(value))


def _set_dotted_attr(obj: Any, key: str, value: Any) -> None:
    cursor = obj
    parts = key.split(".")
    for part in parts[:-1]:
        child = getattr(cursor, part, None)
        if child is None:
            child = ConfigNode()
            setattr(cursor, part, child)
        cursor = child
    setattr(cursor, parts[-1], copy.deepcopy(value))


def _object_to_dict(obj: Any) -> dict[str, Any]:
    """将 BaseConfig 对象的实例属性转换为字典。"""
    return {key: class_to_dict(value) for key, value in vars(obj).items() if not key.startswith("_")}


def _is_config_value(value: Any) -> bool:
    """判断一个值是否为可序列化的配置值（非模块、非函数、非可调用对象）。"""
    if inspect.ismodule(value) or inspect.isfunction(value) or inspect.ismethod(value):
        return False
    return not callable(value)


def _load_mapping(path: Path) -> dict[str, Any]:
    """从 JSON 或 YAML 文件加载配置字典。

    支持通过 _base_ 字段继承另一个配置文件，自动合并。
    """
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported config format '{path.suffix}'. Use a Python config class, .json, or .yaml."
        )
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping, got {type(data).__name__}")
    base_ref = data.pop("_base_", None)
    if base_ref is not None:
        if not isinstance(base_ref, str):
            raise TypeError("Config '_base_' must be a string reference.")
        base_target = base_ref
        base_path = Path(base_ref)
        if (
            not base_path.is_absolute()
            and (base_path.suffix or "/" in base_ref or "\\" in base_ref)
        ):
            candidate = (path.parent / base_path).resolve()
            if candidate.exists():
                base_target = str(candidate)
        base_data = _load_config_reference(base_target)
        data = _deep_merge_dict(base_data, data)
    return data


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个字典：override 中的键会覆盖 base 中的对应键，
    对于嵌套字典则递归合并。
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_reference(reference: str) -> dict[str, Any]:
    """根据引用字符串加载配置。

    支持三种格式：
    1. JSON/YAML 文件路径：直接加载
    2. .py 文件路径或存在的文件：加载为 Python 模块并提取配置属性
    3. Python 模块引用（如 'package.module:attr'）：导入模块并提取属性
    """
    target, attr = _split_reference(reference)
    path = Path(target)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        if attr is not None:
            raise ValueError("File config references cannot include an attribute.")
        return _load_mapping(path)
    if path.suffix.lower() == ".py" or path.exists():
        module = _load_python_module_from_path(path)
        value = _select_config_attr(module, attr)
        return config_to_dict(value)

    module, value = _load_python_module_reference(target, attr)
    return config_to_dict(value if value is not None else module)


def _split_reference(reference: str) -> tuple[str, str | None]:
    """将 'module:attr' 格式的引用拆分为模块路径和属性名。"""
    if ":" in reference:
        target, attr = reference.rsplit(":", 1)
        return target, attr
    return reference, None


def _load_python_module_from_path(path: Path) -> Any:
    """从文件路径动态加载 Python 模块。"""
    if not path.exists():
        raise FileNotFoundError(path)
    module_name = f"_sl_config_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_python_module_reference(target: str, attr: str | None) -> tuple[Any, Any | None]:
    """通过模块导入语法加载 Python 配置。

    如果指定了 attr，直接获取该属性；否则尝试自动推断配置属性名。
    支持 'package.module.ClassName' 格式自动推断 attr。
    """
    if attr is not None:
        module = importlib.import_module(target)
        return module, _resolve_attr(module, attr)

    try:
        module = importlib.import_module(target)
        return module, _select_config_attr(module, None)
    except ModuleNotFoundError:
        module_path, _, inferred_attr = target.rpartition(".")
        if not module_path:
            raise
        module = importlib.import_module(module_path)
        return module, _resolve_attr(module, inferred_attr)


def _select_config_attr(module: Any, attr: str | None) -> Any:
    """从模块中选择配置属性。

    如果指定了 attr，直接按路径解析；否则按优先级尝试
    'cfg' -> 'config' -> 'Config'。
    """
    if attr is not None:
        return _resolve_attr(module, attr)
    for candidate in ("cfg", "config", "Config"):
        if hasattr(module, candidate):
            return getattr(module, candidate)
    raise AttributeError(
        f"Config module '{getattr(module, '__name__', module)}' must define cfg, config, Config, or use ':attr'."
    )


def _resolve_attr(obj: Any, attr_path: str) -> Any:
    """按点号分隔的路径解析对象的嵌套属性，如 'ModelConfig.data.batch_size'。"""
    value = obj
    for part in attr_path.split("."):
        value = getattr(value, part)
    return value
