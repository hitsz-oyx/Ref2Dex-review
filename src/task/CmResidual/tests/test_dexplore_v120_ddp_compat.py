from __future__ import annotations

import importlib.util
from pathlib import Path
import socket

import pytest
import torch
import torch.multiprocessing as mp


COMPAT_PATH = Path(__file__).resolve().parents[1] / "tools/dexplore_ddp_compat.py"
LAUNCHER_PATH = Path(__file__).resolve().parents[1] / "tools/run_dexplore_v120_ddp.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _two_rank_update(rank: int, port: int, result_dir: str) -> None:
    import os

    os.environ.update({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                       "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "2"})
    compat = _load_module(f"dexplore_ddp_compat_worker_{rank}", COMPAT_PATH)
    compat.initialize_from_env(backend="gloo")
    facade = compat.install_horovod_facade()
    assert facade.__ref2dex_ddp_facade__
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = facade.DistributedOptimizer(torch.optim.SGD([parameter], lr=0.1), named_parameters=[("p", parameter)])
    loss = (parameter * float(rank + 1)).square().mean()
    loss.backward()
    optimizer.synchronize()
    optimizer.step()
    values = [torch.zeros_like(parameter) for _ in range(2)]
    torch.distributed.all_gather(values, parameter.detach())
    assert values[0].equal(values[1])
    torch.save(parameter.detach().cpu(), Path(result_dir) / f"rank{rank}.pt")
    compat.cleanup()


def test_two_rank_gradient_average_matches_concatenated_single_process(tmp_path):
    mp.spawn(_two_rank_update, args=(_free_port(), str(tmp_path)), nprocs=2, join=True)
    actual = torch.load(tmp_path / "rank0.pt", weights_only=True)
    reference = torch.tensor([1.0])
    # mean([d(w*x)^2/dw for x in {1,2}]) = 5 at w=1; SGD(lr=.1) => .5.
    assert torch.allclose(actual, torch.tensor([0.5]), atol=1e-7, rtol=0.0)


def test_launcher_requires_explicit_multiple_gpus_and_uses_torchrun(tmp_path):
    launcher = _load_module("dexplore_v120_ddp_launcher", LAUNCHER_PATH)
    with pytest.raises(ValueError, match="at least two"):
        launcher.parse_gpus("3")
    bootstrap = tmp_path / "bootstrap.py"
    dexplore = tmp_path / "run.py"
    bootstrap.write_text("", encoding="utf-8")
    dexplore.write_text("", encoding="utf-8")
    original = launcher.BOOTSTRAP
    launcher.BOOTSTRAP = bootstrap
    try:
        command = launcher.torchrun_command(gpus=(2, 5), dexplore_run=dexplore, dexplore_args=["--horovod"])
    finally:
        launcher.BOOTSTRAP = original
    assert command[:5] == [launcher.sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    assert command[-1] == "--horovod"


def test_launcher_strips_remainder_separator(monkeypatch, capsys):
    launcher = _load_module("dexplore_v120_ddp_launcher_main", LAUNCHER_PATH)
    monkeypatch.setattr(launcher, "torchrun_command", lambda **kwargs: list(kwargs["dexplore_args"]))
    launcher.main(["--gpus", "2,5", "--dry-run", "--", "--horovod"])
    assert "--\n" not in capsys.readouterr().out


def test_smoke_arguments_fix_the_engineering_contract(tmp_path):
    launcher = _load_module("dexplore_v120_ddp_launcher_args", LAUNCHER_PATH)
    arguments = launcher.smoke_dexplore_args(motion_root=tmp_path, output=tmp_path / "out",
                                             num_envs=64, horizon_length=64, minibatch_size=256,
                                             max_iterations=1, seed=42)
    assert arguments[arguments.index("--minibatch_size") + 1] == "256"
    assert arguments[arguments.index("--horizon_length") + 1] == "64"
    assert arguments[-1] == "--horovod"


def test_launcher_accepts_explicit_formal_training_batch_contract(tmp_path):
    launcher = _load_module("dexplore_v120_ddp_launcher_formal_args", LAUNCHER_PATH)
    arguments = launcher.smoke_dexplore_args(motion_root=tmp_path, output=tmp_path / "out",
                                             num_envs=2048, horizon_length=64, minibatch_size=16384,
                                             max_iterations=4999, seed=42)
    assert arguments[arguments.index("--num_envs") + 1] == "2048"
    assert arguments[arguments.index("--horizon_length") + 1] == "64"
    assert arguments[arguments.index("--minibatch_size") + 1] == "16384"
    assert arguments[arguments.index("--max_iterations") + 1] == "4999"


def test_launcher_default_source_is_repo_local_vendor_snapshot():
    launcher = _load_module("dexplore_v120_ddp_launcher_vendor", LAUNCHER_PATH)
    assert launcher.DEFAULT_DEXPLORE_RUN == launcher.REPOSITORY_ROOT / "third_party/DExplore/dexplore/run.py"
    assert launcher.DEFAULT_DEXPLORE_RUN.is_file()


def test_runtime_assets_are_fixed_raw_grab_meshes():
    launcher = _load_module("dexplore_v120_ddp_launcher_assets", LAUNCHER_PATH)
    assert [(source.relative_to(launcher.REPOSITORY_ROOT), target, digest)
            for source, target, digest in launcher.RUNTIME_ASSETS] == [
        (Path("data/raw_data/GRAB/objects/airplane/mesh.obj"),
         Path("dexplore/data/assets/mjcf/objects/airplane/airplane.obj"),
         "dcbb1cce38e65b3ee608e20f0846cbf93aa3b7bbf863a67582d9944f9a0d64f0"),
        (Path("data/raw_data/GRAB/objects/table/mesh.obj"),
         Path("dexplore/data/assets/mjcf/objects/table/table.obj"),
         "25c6fb8b774a04f5314a13a538b9c26886716d196d46a68979e817755cf0e383"),
    ]
