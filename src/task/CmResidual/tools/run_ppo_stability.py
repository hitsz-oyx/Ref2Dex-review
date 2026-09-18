"""Run one approved V1.8, V1.9, or V1.10 residual PPO stability stage."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

from eval_zero_residual import (
    DEFAULT_DEXPLORE,
    DEFAULT_DEXPLORE_SOURCE,
    DEFAULT_OI_CM,
    REPOSITORY_ROOT,
    VENDOR_ROOT,
    _git,
    _input,
    _now,
    _write_json,
)
from eval_residual_stability import DEFAULT_REFERENCE, _resolve_modification_version
from run_ppo_formal import _event_metrics, _tensor_tree_finite


NUM_ENVS = 64
HORIZON_LENGTH = 32
MINIBATCH_SIZE = 2048
LOG_SIGMA = math.log(0.1)
V19_VARIANTS = {
    "control": {
        "train_config": "CmResidualSafeCriticControlPPO",
        "actor_input_dim": 1442,
        "critic_input_dim": 1442,
        "critic_candidate_input_dims": [1442, 2005],
    },
    "critic_cm": {
        "train_config": "CmResidualSafeCriticCmPPO",
        "actor_input_dim": 1442,
        "critic_input_dim": 2005,
        "critic_candidate_input_dims": [1442, 2005],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--activity-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--max-epochs", type=int, choices=(2, 10, 20), required=True)
    parser.add_argument(
        "--variant", choices=("v18_no_cm", *V19_VARIANTS), default="v18_no_cm")
    parser.add_argument(
        "--modification-version", default="",
        help="Explicit run provenance version; V1.10 experiments must pass V1.10.1")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--dexplore-checkpoint", type=Path, default=DEFAULT_DEXPLORE)
    parser.add_argument("--oi-cm-checkpoint", type=Path, default=DEFAULT_OI_CM)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--dexplore-source", type=Path, default=DEFAULT_DEXPLORE_SOURCE)
    parser.add_argument(
        "--isaac-gym-python", type=Path,
        default=Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")))
    return parser.parse_args()


def _resolved_config(overrides: list[str]) -> dict:
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    import isaacgymenvs  # noqa: F401 - registers OmegaConf resolvers
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    excluded = ("+train.params.config.train_dir=", "+full_experiment_name=", "hydra.")
    with initialize_config_dir(
            version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
        cfg = compose(config_name="config", overrides=[
            value for value in overrides if not value.startswith(excluded)])
    return OmegaConf.to_container(cfg, resolve=True)


def _build_model(params: dict, observation_dim: int):
    import torch
    from isaacgymenvs.learning.cm_models import ModelCmContinuous, ModelCmEffectContinuous
    from isaacgymenvs.learning.cm_network_builder import CmBuilder, CmEffectBuilder

    builder_name = params["network"].get("name", "cm_actor_critic")
    builders = {"cm_actor_critic": CmBuilder, "cm_effect_actor_critic": CmEffectBuilder}
    if builder_name not in builders:
        raise ValueError(f"Unsupported CmResidual network builder {builder_name}")
    builder = builders[builder_name]()
    builder.load(params["network"])
    model_name = params["model"].get("name", "cm_continuous")
    models = {"cm_continuous": ModelCmContinuous,
              "cm_effect_continuous": ModelCmEffectContinuous}
    if model_name not in models:
        raise ValueError(f"Unsupported CmResidual model {model_name}")
    model = models[model_name](builder).build({
        "input_shape": (observation_dim,), "actions_num": 18, "num_seqs": 1, "value_size": 1,
        "normalize_input": bool(params["config"]["normalize_input"]),
        "normalize_value": bool(params["config"]["normalize_value"]),
    })
    with torch.no_grad():
        mean = model({"obs": torch.zeros(2, observation_dim), "is_train": False})["mus"]
    return model, mean


def _safe_contract(params: dict, observation_dim: int) -> dict:
    import torch

    model, mean = _build_model(params, observation_dim)
    network = model.a2c_network
    sigma = network.sigma.detach().exp()
    result = {
        "observation_dim": observation_dim,
        "actor_input_dim": int(getattr(network, "actor_input_dim", observation_dim)),
        "critic_input_dim": int(getattr(network, "critic_input_dim", observation_dim)),
        "critic_candidate_input_dims": list(
            getattr(network, "critic_candidate_input_dims", ())),
        "action_dim": 18,
        "initial_mean_abs_max": float(mean.abs().amax().item()),
        "sigma_min": float(sigma.min().item()),
        "sigma_max": float(sigma.max().item()),
        "sigma_requires_grad": bool(network.sigma.requires_grad),
    }
    passed = (
        result["initial_mean_abs_max"] == 0.0
        and torch.allclose(sigma, torch.full_like(sigma, 0.1), rtol=0.0, atol=1e-7)
        and not result["sigma_requires_grad"])
    result["passed"] = bool(passed)
    return result


def _load_checkpoint(checkpoint: Path, params: dict, observation_dim: int) -> tuple[dict, dict]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _ = _build_model(params, observation_dim)
    state = {
        key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key: value
        for key, value in payload["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        fixed = torch.linspace(-1, 1, observation_dim).repeat(2, 1)
        first = model({"obs": fixed,
                       "is_train": False})["mus"]
        second = model({"obs": fixed,
                        "is_train": False})["mus"]
    sigma = model.a2c_network.sigma.detach().exp()
    validation = {
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_frame": int(payload.get("frame", -1)),
        "model_finite": _tensor_tree_finite(payload.get("model", {})),
        "optimizer_finite": _tensor_tree_finite(payload.get("optimizer", {})),
        "action_finite": _tensor_tree_finite((first, second)),
        "deterministic_reload_action_max_abs_diff": float((first - second).abs().amax().item()),
        "sigma_min": float(sigma.min().item()),
        "sigma_max": float(sigma.max().item()),
    }
    return payload, validation


def _matched_actor_initialization_contract(control_params: dict, critic_cm_params: dict,
                                           seed: int) -> dict:
    import torch

    models = []
    post_build_rng_states = []
    for params in (control_params, critic_cm_params):
        torch.manual_seed(seed)
        models.append(_build_model(params, 2005)[0])
        post_build_rng_states.append(torch.get_rng_state().clone())
    states = [model.a2c_network.state_dict() for model in models]
    actor_keys = sorted(
        key for key in states[0]
        if key.startswith("actor_mlp.") or key.startswith("mu.") or key == "sigma")
    mismatches = [
        key for key in actor_keys if not torch.equal(states[0][key], states[1][key])]
    prefix = torch.linspace(-1.0, 1.0, 1442).repeat(2, 1)
    suffix_a = torch.zeros(2, 563)
    suffix_b = torch.ones(2, 563)
    with torch.no_grad():
        actor_results = []
        suffix_diffs = []
        for model in models:
            model.eval()
            first = model({"obs": torch.cat((prefix, suffix_a), dim=-1),
                           "is_train": False})
            second = model({"obs": torch.cat((prefix, suffix_b), dim=-1),
                            "is_train": False})
            actor_results.append((first["mus"], first["sigmas"]))
            suffix_diffs.append(float((first["mus"] - second["mus"]).abs().amax().item()))
        stochastic_actions = []
        for model, rng_state in zip(models, post_build_rng_states):
            torch.set_rng_state(rng_state)
            stochastic_actions.append(model({
                "obs": torch.cat((prefix, suffix_a), dim=-1),
                "is_train": False,
            })["actions"])
    cross_mean_diff = float(
        (actor_results[0][0] - actor_results[1][0]).abs().amax().item())
    cross_sigma_diff = float(
        (actor_results[0][1] - actor_results[1][1]).abs().amax().item())
    post_build_cpu_rng_equal = bool(torch.equal(*post_build_rng_states))
    stochastic_action_diff = float(
        (stochastic_actions[0] - stochastic_actions[1]).abs().amax().item())
    result = {
        "seed": seed,
        "actor_state_keys": actor_keys,
        "actor_state_mismatches": mismatches,
        "suffix_mean_max_abs_diff": suffix_diffs,
        "cross_variant_mean_max_abs_diff": cross_mean_diff,
        "cross_variant_sigma_max_abs_diff": cross_sigma_diff,
        "post_build_cpu_rng_equal": post_build_cpu_rng_equal,
        "stochastic_action_max_abs_diff": stochastic_action_diff,
    }
    result["passed"] = bool(
        actor_keys and not mismatches and max(suffix_diffs) == 0.0
        and cross_mean_diff == 0.0 and cross_sigma_diff == 0.0
        and post_build_cpu_rng_equal and stochastic_action_diff == 0.0)
    return result


def main() -> int:
    args = parse_args()
    is_v19 = args.variant in V19_VARIANTS
    modification_version = _resolve_modification_version(
        args.variant, args.modification_version)
    variant_contract = V19_VARIANTS.get(args.variant)
    observation_dim = 2005 if is_v19 else 1442
    train_config = (
        variant_contract["train_config"] if variant_contract else "CmResidualSafePPO")
    use_oi_cm_context = bool(is_v19)
    if args.gpu != 5 or args.seed != 42 or args.num_envs != NUM_ENVS:
        raise ValueError(
            f"{modification_version} stability stages are locked to GPU5, seed42, and 64 envs")
    if args.max_epochs in (2, 10) and args.initial_checkpoint is not None:
        raise ValueError(f"T{args.max_epochs} must start without a residual checkpoint")
    if args.max_epochs == 20 and args.initial_checkpoint is None:
        raise ValueError("T20 requires the explicit passing T10 checkpoint")

    run_label = "v18" if not is_v19 else f"v19_{args.variant}"
    run_id = args.run_id or f"cmresidual_{run_label}_t{args.max_epochs}_{datetime.now():%Y%m%d_%H%M%S}"
    output = (args.output or REPOSITORY_ROOT / "outputs/CmResidual" / run_id).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stability stage: {output}")
    output.mkdir(parents=True)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "train.log"
    validation_path = output / "checkpoint_validation.json"
    experiment_name = (
        f"CmResidualSafeT{args.max_epochs}" if not is_v19
        else f"CmResidualV19{args.variant.title().replace('_', '')}T{args.max_epochs}")

    dexplore = _input(args.dexplore_checkpoint, "dexplore_checkpoint")
    oi_cm = _input(args.oi_cm_checkpoint, "oi_cm_checkpoint") if is_v19 else None
    reference = _input(args.reference, "reference")
    source = _input(args.dexplore_source, "dexplore_source_tensor")
    initial = _input(args.initial_checkpoint, "initial_residual_checkpoint") if args.initial_checkpoint else None
    initial_epoch = 0
    if initial:
        import torch
        initial_payload = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        initial_epoch = int(initial_payload.get("epoch", -1))
        if initial_epoch != 10:
            raise ValueError(f"T20 requires an epoch-10 checkpoint, got epoch {initial_epoch}")
    reference_manifest_path = Path(reference["path"]).with_name("manifest.json")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))

    overrides = [
        "task=CmResidual", f"train={train_config}", "headless=True",
        f"task.basePolicy.useOiCmContext={'true' if use_oi_cm_context else 'false'}",
        f"task.reference.path={args.reference.resolve()}",
        f"num_envs={args.num_envs}",
        f"train.params.config.max_epochs={args.max_epochs}",
        f"train.params.config.minibatch_size={MINIBATCH_SIZE}",
        "pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0",
        "graphics_device_id=0", "force_render=False",
        "task.sim.physx.max_gpu_contact_pairs=8388608",
        "task.sim.physx.default_buffer_size_multiplier=5.0",
        f"seed={args.seed}",
        f"+train.params.config.train_dir={output}",
        f"+full_experiment_name={experiment_name}",
        f"hydra.run.dir={output / 'hydra'}", "hydra.job.chdir=True",
    ]
    if args.initial_checkpoint:
        overrides.append(f"checkpoint={args.initial_checkpoint.resolve()}")
    if args.max_epochs == 2:
        overrides.append("train.params.config.save_frequency=1")
    resolved = _resolved_config(overrides)
    if bool(resolved["task"]["basePolicy"]["useOiCmContext"]) != use_oi_cm_context:
        raise RuntimeError(f"{modification_version} OI-Cm context contract drift")
    resolved_max_epochs = int(resolved["train"]["params"]["config"]["max_epochs"])
    if resolved_max_epochs != args.max_epochs:
        raise RuntimeError(
            f"{modification_version} stage budget drift: requested {args.max_epochs}, resolved {resolved_max_epochs}")
    contract = _safe_contract(resolved["train"]["params"], observation_dim)
    if variant_contract:
        for name in (
                "actor_input_dim", "critic_input_dim",
                "critic_candidate_input_dims"):
            if contract[name] != variant_contract[name]:
                raise RuntimeError(
                    f"{args.variant} {name} drift: {contract[name]} != {variant_contract[name]}")
    if not contract["passed"]:
        raise RuntimeError(f"{modification_version} safe policy contract failed: {contract}")
    if args.initial_checkpoint:
        _, initial_validation = _load_checkpoint(
            args.initial_checkpoint.resolve(), resolved["train"]["params"], observation_dim)
        if initial_validation["checkpoint_epoch"] != 10:
            raise ValueError(
                f"T20 requires a schema-compatible epoch-10 checkpoint, got "
                f"epoch {initial_validation['checkpoint_epoch']}")
    matched_actor_contract = None
    if is_v19:
        paired_params = {}
        for paired_variant, paired_contract in V19_VARIANTS.items():
            paired_overrides = [
                value if not value.startswith("train=")
                else f"train={paired_contract['train_config']}"
                for value in overrides
            ]
            paired_params[paired_variant] = _resolved_config(
                paired_overrides)["train"]["params"]
        matched_actor_contract = _matched_actor_initialization_contract(
            paired_params["control"], paired_params["critic_cm"], args.seed)
        if not matched_actor_contract["passed"]:
            raise RuntimeError(
                f"V1.9 matched actor initialization failed: {matched_actor_contract}")
    _write_json(config_path, {
        "run_id": run_id, "physical_gpu": args.gpu, "resolved": resolved,
        "runtime_overrides": overrides, "variant": args.variant,
        "safe_policy_contract": contract,
        "matched_actor_initialization_contract": matched_actor_contract,
    })

    train_entrypoint = VENDOR_ROOT / "isaacgymenvs/train.py"
    compat_bootstrap = (
        "import runpy, numpy as np; "
        "np.float = float; "
        f"runpy.run_path({str(train_entrypoint)!r}, run_name='__main__')")
    command_values = [sys.executable, "-c", compat_bootstrap, *overrides]
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "created_at": _now(),
        "mode": f"{run_label}_stability_t{args.max_epochs}", "task": "CmResidual",
        "run_id": run_id,
        "activity_id": args.activity_id or (
            f"ACT-CMRESIDUAL-{modification_version.replace('.', '')}-"
            f"{args.variant.upper()}-T{args.max_epochs}"),
        "run_status": "STARTED", "modification_version": modification_version,
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output),
        "command": " ".join(shlex.quote(value) for value in command_values),
        "seed": args.seed, "base_commit": _git("rev-parse", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "config_snapshot": str(config_path),
        "metadata_snapshot": str(reference_manifest_path.resolve()),
        "input_references": [dexplore] + ([oi_cm] if oi_cm else []) + [reference, source]
                            + ([initial] if initial else []),
        "reference_training_eligible": bool(reference_manifest.get("training_eligible", False)),
        "reference_allow_ineligible_for": "", "initial_checkpoint": initial,
        "checkpoint": None, "metrics": str(metrics_path), "log": str(log_path),
        "physical_gpu": args.gpu,
        "budget": {"num_envs": args.num_envs, "initial_epoch": initial_epoch,
                   "max_epochs": args.max_epochs, "horizon_length": HORIZON_LENGTH,
                   "minibatch_size": MINIBATCH_SIZE},
        "safe_policy_contract": contract, "conclusion": "INCONCLUSIVE",
        "variant": args.variant,
        "actor_input_dim": contract["actor_input_dim"],
        "critic_input_dim": contract["critic_input_dim"],
        "matched_actor_initialization_contract": matched_actor_contract,
    }
    _write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["DEXPLORE_CHECKPOINT"] = dexplore["path"]
    if oi_cm:
        environment["OI_CM_CHECKPOINT"] = oi_cm["path"]
    environment["CMRESIDUAL_REFERENCE"] = reference["path"]
    environment["CMRESIDUAL_DEXPLORE_SOURCE"] = source["path"]
    python_paths = [str(args.isaac_gym_python.resolve()), str(REPOSITORY_ROOT), str(VENDOR_ROOT)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)

    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            result = subprocess.run(command_values, cwd=REPOSITORY_ROOT, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training subprocess exited with code {result.returncode}")
        experiment_dir = output / experiment_name
        checkpoints = sorted((experiment_dir / "nn").glob("*.pth"),
                             key=lambda path: path.stat().st_mtime_ns)
        if not checkpoints:
            raise RuntimeError(f"No checkpoint under {experiment_dir / 'nn'}")
        last_checkpoint = checkpoints[-1]
        expected_events = args.max_epochs - initial_epoch
        rows, event_tags = _event_metrics(experiment_dir / "summaries", expected_events)
        with metrics_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        _, validation = _load_checkpoint(
            last_checkpoint, resolved["train"]["params"], observation_dim)
        validation["checkpoint"] = str(last_checkpoint)
        validation["checkpoint_sha256"] = _input(last_checkpoint, "ppo_checkpoint")["sha256"]
        validation["event_scalar_tags"] = event_tags
        _write_json(validation_path, validation)
        passed = (
            validation["checkpoint_epoch"] >= args.max_epochs
            and validation["model_finite"] and validation["optimizer_finite"]
            and validation["action_finite"]
            and validation["deterministic_reload_action_max_abs_diff"] <= 1e-6
            and abs(validation["sigma_min"] - 0.1) <= 1e-7
            and abs(validation["sigma_max"] - 0.1) <= 1e-7
            and len(rows) == expected_events)
        manifest.update({
            "run_status": "COMPLETED" if passed else "FAILED", "completed_at": _now(),
            "last_step": validation["checkpoint_frame"],
            "last_epoch": validation["checkpoint_epoch"],
            "best_metric": {"name": "checkpoint_reload_action_max_abs_diff",
                            "value": validation["deterministic_reload_action_max_abs_diff"]},
            "checkpoint": str(last_checkpoint),
            "checkpoint_validation": str(validation_path),
            "exit_reason": f"Reached {modification_version} {args.variant} T{args.max_epochs} training budget",
            "gate_passed": passed,
            "conclusion": "SUPPORTED" if passed else "INVALID_IMPLEMENTATION",
            "scientific_conclusion": "INCONCLUSIVE",
            "summary": {"epochs_in_stage": len(rows), "final_metrics": rows[-1]},
        })
        _write_json(manifest_path, manifest)
        if not passed:
            raise RuntimeError(f"{modification_version} training contract failed: {validation}")
    except BaseException as error:
        traceback.print_exc()
        if not metrics_path.exists():
            metrics_path.write_text(json.dumps({"status": "FAILED", "error": str(error)}) + "\n")
        if manifest.get("run_status") != "COMPLETED":
            manifest.update({
                "run_status": "FAILED", "completed_at": _now(),
                "exit_reason": f"{type(error).__name__}: {error}",
                "conclusion": "INVALID_IMPLEMENTATION", "scientific_conclusion": "INCONCLUSIVE",
            })
            _write_json(manifest_path, manifest)
        raise
    print(json.dumps({"run_id": run_id, "run_status": "COMPLETED",
                      "last_epoch": manifest["last_epoch"], "checkpoint": manifest["checkpoint"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
