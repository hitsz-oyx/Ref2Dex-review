# 数据集审计补充说明

## 1. 用途

本目录为 OakInk2 新增本地、可共享的数据质量审计工具。这不是 OakInk2 官方源码。工具代码位于 `dataset_audit/`，生成结果统一写入 `outputs/dataset_audit/`。

## 2. Docker / Miniconda 使用方式

不要在 macOS host 上直接安装依赖或运行数据集脚本。先进入当前仓库的 Docker/Miniconda 环境：

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate oakink2-dev
```

使用 MeshCat 时，将容器端口 `7000` 映射到 Mac 的 `7011`：

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name oakink2-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu 原生使用方式

在 Ubuntu 22.04/24.04 上，可以从 OakInk2 仓库根目录直接使用 conda 环境运行审计工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n oakink2-dev python=3.10 -y
conda activate oakink2-dev
python -m pip install --upgrade pip
python -m pip install numpy scipy trimesh meshcat rtree
python -m pip install -e .
```

只有需要完整官方 preview stack 时才运行 `python -m pip install -r req_preview.txt`。该文件固定了 CUDA/PyTorch wheel，并会引用 `thirdparty/` 下的第三方源码目录；在当前 vendored、无 submodule 布局中，安装前需要先把这些目录作为普通文件补齐。

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

下载后的 OakInk2 根目录可通过 `OAKINK2_DIR` 或 `OAKINK2_DATASET_PREFIX` 指定，也可以放在本地 `OakInk-v2-hub/` 或 `data/`。如果 Ubuntu 是远程机器，优先用 `--headless-check` 验证；需要浏览器可视化时再转发 MeshCat 端口。

## 4. 数据可用性

审计工具不会下载 OakInk2 tarball、MANO 或 SMPL-X 资产。manifest 会检查 `OakInk-v2-hub`、`data` 以及相关环境变量。缺失数据只会被报告，不会自动下载。

OakInk2 官方数据托管在 [Hugging Face OakInk-v2](https://huggingface.co/datasets/kelvin34501/OakInk-v2)，官网 [OakInk2](https://oakink.net/v2/) 的 Download 区也指向该页面。官方 README 提醒：2024-12 之后部分 annotation 文件重新上传过，使用数据时应把 Hugging Face snapshot 更新到最新 commit；2025-04 起 `object_preview.tar` 作为兼容文件恢复。官方说明见 `README_OAKINK2.md` 和仓库内 `script/download.py`。

推荐从 OakInk2 仓库根目录下载 snapshot，至少预留 Hugging Face cache 和本地目录空间：

```bash
python -m pip install -U huggingface_hub
python script/download.py
```

等价的手动命令是：

```bash
huggingface-cli download kelvin34501/OakInk-v2 \
  --repo-type dataset \
  --local-dir ./OakInk-v2-hub \
  --cache-dir ./hub
```

至少需要一组 sequence 的 `data/` tarball 和对应 `anno_preview/` annotation，再加 `object_raw.tar`、`object_repair.tar`、`object_affordance.tar`、`program.tar`；`object_preview.tar` 只为兼容旧路径。下载后按官方 layout 解压或整理到同一个 dataset prefix：

```text
data/
├── data/
│   └── scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS/
├── anno_preview/
│   └── scene_0x__y00z++00000000000000000000__YYYY-mm-dd-HH-MM-SS.pkl
├── object_preview/        # deprecated，仅兼容旧路径
├── object_raw/
├── object_repair/
├── object_affordance/
└── program/
```

如果直接使用 `OakInk-v2-hub/` 作为数据根，设置：

```bash
export OAKINK2_DIR=$PWD/OakInk-v2-hub
export OAKINK2_DATASET_PREFIX=$PWD/OakInk-v2-hub
```

如果把 tarball 解压到当前仓库 `data/`，则可以不设置环境变量；manifest 会自动检查本地 `data/`。官方 preview tool 还需要从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/download.php) 下载 v1.1 并放到 `asset/smplx_v1_1/`；可选的 segment preview 需要从 [MANO 官网](https://mano.is.tue.mpg.de/) 下载 v1.2 并放到 `asset/mano_v1_2/`。这些模型和下载缓存都不要提交到 Git。

## 5. 可视化

```bash
python dataset_audit/scripts/dataset_manifest.py --out outputs/dataset_audit/manifest.csv
python dataset_audit/scripts/prepare_sequence_cache.py --item <sample_item>
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/cache/<sample>.npz --fps 3 --stride 20 --loop
```

无数据 smoke test：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py --toy --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz --headless-check
```

## 6. 穿模 / 负距离统计

```bash
python dataset_audit/scripts/check_point_sdf_penetration.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/penetration.json --out-csv outputs/dataset_audit/penetration.csv

python dataset_audit/scripts/sample_penetration_statistics.py \
  --manifest outputs/dataset_audit/manifest.csv --sample-size 5 --frames-per-seq 100 \
  --out-json outputs/dataset_audit/sample_penetration_statistics.json \
  --out-csv outputs/dataset_audit/sample_penetration_statistics.csv
```

跨数据集统一 signed distance 约定：负值表示 query hand/body point 位于 object 或 proxy 内部，`frame_max_negative_depth_mm` 是该帧最大非负穿透深度。

## 7. 表面相交检测

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

surface collision 是 overlap/intersection 线索，不是 penetration-depth 指标。

## 8. 接触信息

静态审计没有识别到直接内置的 penetration CLI/API。OakInk2 在数据存在时包含 affordance 和 primitive task metadata。affordance 描述 object part function 或 task semantics，不等同于逐帧几何 contact。如果没有直接 contact 字段，应通过 hand-object distance threshold 派生 contact。

## 9. 限制

point-SDF 的符号只有在 watertight mesh 或 proxy 下可靠。non-watertight visual mesh 只能作为 heuristic。surface collision 可能统计到边界接触或 mesh fitting noise。OakInk2 preview/toolkit 的 mesh 重建需要逐 sequence 验证后，才能信任 aggregate statistics。
