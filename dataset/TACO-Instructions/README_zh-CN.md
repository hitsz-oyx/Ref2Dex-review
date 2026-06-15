# 数据集审计补充说明

## 1. 用途

本目录为 TACO-Instructions 新增本地、可共享的数据质量审计工具。这不是 TACO-Instructions 官方源码。工具代码位于 `dataset_audit/`，生成结果统一写入 `outputs/dataset_audit/`。

## 2. Docker / Miniconda 使用方式

不要在 macOS host 上直接安装依赖或运行数据集脚本。先进入当前仓库的 Docker/Miniconda 环境：

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate taco-instructions-dev
```

使用 MeshCat 时，将容器端口 `7000` 映射到 Mac 的 `7011`：

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name taco-instructions-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu 原生使用方式

在 Ubuntu 22.04/24.04 上，可以从 TACO-Instructions 仓库根目录直接使用 conda 环境运行审计工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n taco-instructions-dev python=3.9 -y
conda activate taco-instructions-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install meshcat rtree
```

使用真实 TACO-Instructions 资产前，先跑无数据 smoke test：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

下载后的数据根目录可通过 `TACO_DATASET_ROOT` 或 `TACO_INSTRUCTIONS_DATASET_ROOT` 指定；如果 object model 不在数据根目录内，再设置 `TACO_OBJECT_MODEL_ROOT`。如果 Ubuntu 是远程机器，优先用 `--headless-check` 验证；需要浏览器可视化时再转发 MeshCat 端口。

## 4. 数据可用性

审计工具不会下载 TACO-Instructions 数据集、object model 或 MANO 文件。manifest 会检查 data-list 文件以及本地/环境变量指定的数据根目录，并明确报告缺失资产。

官方项目页和 README 提供两个数据入口：pre-release 版本在 OneDrive，完整 **Whole Dataset Version 1** 在 Dropbox，并提供一个 Dropbox 备份链接；pre-release 还提供百度网盘备份。入口见 [TACO 官网](https://taco2024.github.io/) 和 `README_TACO.md`。

下载建议：

1. 如果只是快速验证数据结构，下载 pre-release 版本即可；它包含 244 条高质量 motion sequence、206 个高分辨率 object model、hand-object pose/mesh annotation、egocentric RGB-D video 和 8-view allocentric RGB video。
2. 如果要做完整审计或复现实验，下载 Whole Dataset Version 1；它包含 2317 条 motion sequence、206 个 object model、hand-object pose/mesh annotation、egocentric RGB-D video、12-view allocentric RGB video、camera parameters、automatic hand-object 2D segmentation 和 marker-removed allocentric RGB video。
3. 如果使用百度网盘 pre-release 备份，部分视频压缩包会被拆分，需要先合并：

```bash
cat Allocentric_RGB_Videos_split.* > Allocentric_RGB_Videos.zip
cat Egocentric_Depth_Videos_split.* > Egocentric_Depth_Videos.zip
```

解压后，dataset root 应包含官方 README 中的目录：

```text
TACO_Dataset_V1/
├── Allocentric_RGB_Videos/
├── Egocentric_Depth_Videos/
├── Egocentric_RGB_Videos/
├── Hand_Poses/
├── Object_Poses/
├── Object_Models/
└── Marker_Removed_Allocentric_RGB_Videos/
```

将下载后的目录暴露给审计工具：

```bash
export TACO_DATASET_ROOT=/path/to/TACO_Dataset_V1
export TACO_INSTRUCTIONS_DATASET_ROOT=$TACO_DATASET_ROOT
export TACO_OBJECT_MODEL_ROOT=$TACO_DATASET_ROOT/Object_Models
```

官方 visualization 还需要从 [MANO 官网](https://mano.is.tue.mpg.de/) 下载 `MANO_LEFT.pkl` 和 `MANO_RIGHT.pkl`，放到：

```text
dataset_utils/manopth/mano/models/
├── MANO_LEFT.pkl
└── MANO_RIGHT.pkl
```

`data_lists/` 中的列表只说明哪些 sequence 存在或属于哪个 split，不能替代真实下载的数据、object model 和 MANO 文件。

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

`frame_min_signed_distance_mm` 是被检查帧里的最负 signed distance。`frame_max_negative_depth_mm` 是把负 signed distance 转成非负 penetration value 后的最大深度。sequence 和 aggregate summary 会报告均值、方差、最大值和帧比例。

## 7. 表面相交检测

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

surface collision 报告 intersection cue 和 triangle ratio，不报告稳定 penetration depth。

## 8. 接触信息

静态审计没有识别到官方 contact 或 penetration 字段。TACO-Instructions 提供 hand-object pose/mesh annotation 和 visualization workflow。如果下载后的 annotation 中没有直接 contact 字段，contact 应通过 hand-object geometric distance threshold 派生。

## 9. 限制

visual mesh 不一定是 collision mesh。non-watertight object 上的 point-SDF 是 heuristic。surface collision 可能统计到接触或拟合误差。真实 TACO-Instructions sequence 中的 MANO/object 坐标对齐必须结合 MeshCat 复查。
