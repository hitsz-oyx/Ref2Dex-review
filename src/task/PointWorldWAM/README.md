# PointWorldWAM

本 Task 按 `docs/指导/V0.md` 先验证 F0：

```text
GT MANO/hand point tracks -> rigid object point tracks
```

当前不包含 inverse MANO branch。PointWorld 保持上游源码不改，以 git submodule 固定：

- repository: <https://github.com/NVlabs/PointWorld>
- commit: `05484826dfef74cbe278a3974179a5a16705d35d`
- checkpoint: `nvidia/PointWorld_models/small-droid/model-best.pt`（本地文件，不提交）

## 本地依赖

官方 PointWorld 预期 Python 3.9+。当前机器使用 Python 3.8，因此适配层仅在导入
`DynamicsPredictor` 时隔离不会被它使用的 `norm_stats/losses/metrics`；未修改上游代码。

本次验证环境为 PyTorch 2.4.1 + CUDA 12.1，并额外安装：

```text
spconv-cu120==2.3.6
cumm-cu120==0.4.11
torch-scatter==2.1.2+pt24cu121
timm==1.0.19
flash-attn==2.6.3+cu123torch2.4cxx11abiFALSE
```

下载 checkpoint：

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "nvidia/PointWorld_models",
    "small-droid/model-best.pt",
    local_dir="third_party/PointWorld/pretrained_checkpoints",
)
PY
```

## 运行

```bash
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.data_sanity.diag_grab_window

PYTHONPATH=. python3 -m src.task.PointWorldWAM.train \
  --config src/task/PointWorldWAM/configs/grab_forward_overfit.yaml

PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.forward_smoke.diag_forward

PYTHONPATH=. python3 -m src.task.PointWorldWAM.visualize
```

训练与诊断产物分别写入 `output/exp/pointworld_wam_forward_overfit/` 和
`output/research/pointworld_wam/`，均不提交。
