"""验证 Cm 破坏性迁移后的唯一规范入口。"""

from pathlib import Path

from src.base import load_config
from src.task.Cm.dataset import Stage4CmDataset
from src.task.Cm.dataset.object_v2 import CmObjectV2Dataset
from src.task.Cm.src import CmActionRunner, CmFlowModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "src" / "task" / "Cm" / "configs"


def test_cm_new_imports_and_old_paths_are_absent() -> None:
    active_path = CONFIG_ROOT / "active" / "object_v2_grab_arctic_subject_template_20260820.yaml"
    assert not (CONFIG_ROOT / active_path.name).exists()
    active_cfg = load_config(str(active_path))

    assert isinstance(active_cfg.name, str) and active_cfg.name.startswith("cm_")
    assert CmActionRunner.__name__ == "CmActionRunner"
    assert CmFlowModel.__name__ == "CmFlowModel"
    assert Stage4CmDataset.__name__ == "Stage4CmDataset"
    assert CmObjectV2Dataset.__name__ == "CmObjectV2Dataset"


def test_cm_config_lifecycle_directories_exist() -> None:
    assert (CONFIG_ROOT / "active").is_dir()
    assert (CONFIG_ROOT / "archive").is_dir()
    assert (CONFIG_ROOT / "active" / "README.md").is_file()
    assert (CONFIG_ROOT / "archive" / "README.md").is_file()
