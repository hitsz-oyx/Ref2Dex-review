"""Contract checks for the V1.17 external DExplore teacher smoke launcher."""
from __future__ import annotations

import importlib.util
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools/run_dexplore_grab_teacher.py"
SPEC = importlib.util.spec_from_file_location("dexplore_teacher_launcher", TOOL)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_v117_smoke_command_uses_the_fixed_on_policy_batch_contract():
    command = LAUNCHER._command(Path("/tmp/dexplore-v117-smoke"), num_envs=4)

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
