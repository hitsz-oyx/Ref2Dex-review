from pathlib import Path

import pytest
import yaml

from src.task.ObjectInteractionCmv2.config import load_grab_config
from src.task.ObjectInteractionCmv2.train_grab_ddp import shard_epoch_indices


def test_two_rank_epoch_has_no_padding_or_repeated_pairs():
    count = 1000
    left = shard_epoch_indices(count, 42, 0, 0, 2).tolist()
    right = shard_epoch_indices(count, 42, 0, 1, 2).tolist()
    assert len(left) == len(right) == count // 2
    assert sorted(left + right) == list(range(count))
    assert left == shard_epoch_indices(count, 42, 0, 0, 2).tolist()
    assert left != shard_epoch_indices(count, 42, 1, 0, 2).tolist()


def test_ddp_config_rejects_wrong_physical_gpus(tmp_path):
    source = Path(
        "src/task/ObjectInteractionCmv2/configs/active/"
        "grab_mano_v1_3_ddp_calibration.yaml")
    config = yaml.safe_load(source.read_text())
    for name in ("index.json", "run_manifest.json"):
        (tmp_path / name).write_text("{}")
    config["source"]["index"] = str(tmp_path / "index.json")
    config["source"]["manifest"] = str(tmp_path / "run_manifest.json")
    config["output_root"] = str(tmp_path / "output")
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_grab_config(path)["training"]["world_size"] == 2
    config["training"]["device_ids"] = [1, 3]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="dual-GPU"):
        load_grab_config(path)
