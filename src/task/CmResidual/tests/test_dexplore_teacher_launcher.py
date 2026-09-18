"""Contract checks for the V1.17 external DExplore teacher smoke launcher."""
from __future__ import annotations

import importlib.util
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools/run_dexplore_grab_teacher.py"
SPEC = importlib.util.spec_from_file_location("dexplore_teacher_launcher", TOOL)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_v117_smoke_command_uses_the_fixed_on_policy_batch_contract():
    command = LAUNCHER._command(Path("/tmp/dexplore-v117-smoke"), num_envs=4,
                                max_iterations=LAUNCHER.SMOKE_ITERATIONS)

    def value(option: str) -> str:
        index = command.index(option)
        return command[index + 1]

    assert command[:2] == [LAUNCHER.sys.executable, "dexplore/run.py"]
    assert "--output" not in command
    assert "--headless" in command
    assert value("--motion_file") == str(Path("/tmp/dexplore-v117-smoke") / "motion_input")
    assert LAUNCHER.SEQUENCE == "s1_airplane_lift"
    assert value("--num_envs") == "4"
    assert value("--horizon_length") == "64"
    assert value("--minibatch_size") == "256"
    assert value("--max_iterations") == "1"
    assert value("--seed") == "42"
    assert value("--sim_device") == "cuda:0"
    assert value("--rl_device") == "cuda:0"


def test_v117_formal_contract_is_fixed_to_2048_envs_and_five_million_step_checkpoints():
    settings = LAUNCHER._run_settings("formal")
    command = LAUNCHER._command(Path("/tmp/dexplore-v117-formal"), num_envs=2048,
                                max_iterations=settings["max_iterations"],
                                train_config="/tmp/dexplore-v117-formal/train_config.yaml")

    def value(option: str) -> str:
        index = command.index(option)
        return command[index + 1]

    assert settings == {"max_iterations": 152, "save_frequency": 38, "target_env_steps": 20_054_016}
    assert value("--num_envs") == "2048"
    assert value("--max_iterations") == "152"
    assert value("--cfg_train") == "/tmp/dexplore-v117-formal/train_config.yaml"


def test_v117_horovod_command_uses_two_synchronized_ranks_and_rank_local_env_counts():
    settings = LAUNCHER._run_settings("formal", num_envs=2048, world_size=LAUNCHER.HOROVOD_WORLD_SIZE)
    command = LAUNCHER._horovod_command(
        Path("/tmp/dexplore-v117-horovod"), num_envs=2048,
        max_iterations=settings["max_iterations"],
    )

    assert command[:5] == [
        str(Path(LAUNCHER.sys.executable).with_name("horovodrun")),
        "-np", "2", "-H", "localhost:2",
    ]
    assert command[5:7] == [LAUNCHER.sys.executable, "dexplore/run.py"]
    assert command.count("--horovod") == 1
    assert command[command.index("--num_envs") + 1] == "2048"
    assert settings["target_env_steps"] == 40_108_032
    assert LAUNCHER.HOROVOD_PHYSICAL_GPUS == (0, 3)


def test_v117_horovod_smoke_budget_counts_all_rank_local_environments():
    settings = LAUNCHER._run_settings("smoke", num_envs=2048, world_size=LAUNCHER.HOROVOD_WORLD_SIZE)
    assert settings == {"max_iterations": 1, "save_frequency": None, "target_env_steps": 262_144}


def test_v117_formal_train_config_only_changes_checkpoint_cadence(tmp_path):
    import yaml

    path = LAUNCHER._write_formal_train_config(tmp_path, save_frequency=38)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    params = config["params"]["config"]
    assert params["save_frequency"] == 38
    assert params["save_best_after"] == 38
    assert params["horizon_length"] == 64
    assert params["minibatch_size"] == 16384
