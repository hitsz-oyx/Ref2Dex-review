# PointWorldWAM

本 Task 按 `docs/指导/V0.md`–`V0.2.md` 验证 forward，按 `docs/指导/V1.md` 验证
双手 inverse Flow Matching，按 `docs/指导/V1.1.md` 验证 world/action chunk 联合生成，并按
`docs/指导/V1.2.md` 验证修正后架构的 deterministic upper bound：

```text
GT MANO/hand point tracks -> rigid object point tracks
```

PointWorld 保持上游源码不改，以 git submodule 固定：

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

# 四卡 long-overfit 与未收敛实验续训
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.long_overfit.run_matrix --gpus 0,1,2,3
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.long_overfit.continue_matrix --gpus 0,1,2

# best checkpoint 统一复算与 held-out sequence 诊断
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.long_overfit.diag_matrix
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.long_overfit.diag_heldout

# V0.1 geometry-enhanced 5000-step overfit 与统一复算
PYTHONPATH=. python3 -m src.task.PointWorldWAM.train \
  --config src/task/PointWorldWAM/configs/grab_forward_geometry_v1.yaml
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.geometry_v1.diag_geometry_v1

# V0.2A one-step capacity test；未达 gate 时不运行 V0.2B
PYTHONPATH=. python3 -m src.task.PointWorldWAM.train_one_step \
  --config src/task/PointWorldWAM/configs/grab_one_step_v02a.yaml
PYTHONPATH=. python3 -m src.task.PointWorldWAM.research.one_step_v02a.diag_one_step

# V1 需要同时包含 smplx 与 PointWorld 依赖的 graspenv
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_bimanual_forward \
  --config src/task/PointWorldWAM/configs/grab_bimanual_forward.yaml
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_wam_v1 \
  --config src/task/PointWorldWAM/configs/grab_wam_v1.yaml
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.eval_wam_v1

# V1.1 10-frame Chunk Joint WAM；debug gate 未通过时不扩全量
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_wam_v2 \
  --config src/task/PointWorldWAM/configs/grab_wam_v2_chunk.yaml
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.eval_wam_v2 --batch-size 4
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.research.wam_v2.diag_chunk

# V1.2 normalization sanity 与两个独立 deterministic regression
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.research.wam_v12.diag_normalization
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_wam_v2_regression --mode inverse --device cuda:0
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_wam_v2_regression --mode forward --device cuda:1
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.eval_wam_v2_regression --mode inverse --batch-size 4
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.eval_wam_v2_regression --mode forward --batch-size 4

# deterministic 双门禁通过后，分别训练与复算单任务 FM
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_wam_v2 --config src/task/PointWorldWAM/configs/grab_wam_v12_fm.yaml \
  --mode inverse --device cuda:0 --steps 6000
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.train_wam_v2 --config src/task/PointWorldWAM/configs/grab_wam_v12_fm.yaml \
  --mode forward --device cuda:1 --steps 6000
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.eval_wam_v12_fm --mode inverse --device cuda:0 --batch-size 4
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.PointWorldWAM.eval_wam_v12_fm --mode forward --device cuda:1 --batch-size 4
```

训练产物写入 `output/exp/pointworld_wam_*`，诊断产物写入
`output/research/pointworld_wam/`，均不提交。
