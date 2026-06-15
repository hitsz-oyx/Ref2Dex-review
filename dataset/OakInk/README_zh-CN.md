# 数据集审计补充说明

## 1. 用途

本目录为 OakInk 新增本地、可共享的数据质量审计工具。这不是 OakInk 官方源码。工具代码位于 `dataset_audit/`，生成结果统一写入 `outputs/dataset_audit/`。

## 2. Docker / Miniconda 使用方式

不要在 macOS host 上直接安装依赖或运行数据集脚本。先进入当前仓库的 Docker/Miniconda 环境：

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate oakink-dev
```

使用 MeshCat 时，将容器端口 `7000` 映射到 Mac 的 `7011`：

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name oakink-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu 原生使用方式

在 Ubuntu 22.04/24.04 上，可以从 OakInk 仓库根目录直接使用 conda 环境运行审计工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n oakink-dev python=3.9 -y
conda activate oakink-dev
python -m pip install --upgrade pip
python -m pip install numpy scipy trimesh meshcat rtree
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果 PyTorch、PyTorch3D 或 Open3D wheel 与 Ubuntu CUDA 栈不匹配，先安装匹配的 wheel，再继续安装剩余 requirements。只跑审计 smoke test 时，轻量的 `numpy scipy trimesh meshcat rtree` 依赖已经足够。

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

下载后的 OakInk 数据根目录可通过 `OAKINK_DIR` 指定，也可以放在本地 `data/`、`OakInk/` 或 `oakink/`。如果 Ubuntu 是远程机器，优先用 `--headless-check` 验证；需要浏览器可视化时再转发 MeshCat 端口。

## 4. 数据可用性

审计工具不会下载 OakInk 数据或 MANO/object 资产。manifest 会检查 `OAKINK_DIR` 以及本地 `data/`、`OakInk/`、`oakink/` 路径。如果数据缺失，manifest 仍会正常生成，并标记 `data_available=false`。

按官方说明，OakInk 包含 **OakBase**、**OakInk-Image** 和 **OakInk-Shape** 三部分。当前官网的下载入口指向 [Hugging Face OakInk-v1](https://huggingface.co/datasets/oakink/OakInk-v1)，国内用户也可按官网说明使用百度云镜像；`anno_v2.1.zip` 需要填写官网链接的 Google Form 后获取。官方数据说明见 `README_OAKINK.md`、`docs/datasets.md` 和 [OakInk 官网](https://oakink.net/)；本地旧文档里偶见 `anno_v2_1.zip` 写法，整理文件时以官网和 `docs/checksum.json` 的 `anno_v2.1.zip` 为准。

```bash
export OAKINK_DIR=/storage/data/OakInk
mkdir -p "$OAKINK_DIR/zipped"

python -m pip install -U huggingface_hub
huggingface-cli download oakink/OakInk-v1 \
  --repo-type dataset \
  --local-dir "$OAKINK_DIR/zipped"
```

下载完成后，按官方 layout 整理压缩包；如果 Hugging Face 或镜像下载后的目录层级不同，以这个结构为准手动移动：

```text
$OAKINK_DIR/zipped/
├── OakBase.zip
├── image/
│   ├── anno_v2.1.zip        # Google Form 获取
│   ├── obj.zip
│   └── stream_zipped/
│       ├── oakink_image_v2.z01
│       ├── ...
│       └── oakink_image_v2.zip
└── shape/
    ├── metaV2.zip
    ├── OakInkObjectsV2.zip
    ├── oakink_shape_v2.zip
    └── OakInkVirtualObjectsV2.zip
```

官方解压脚本依赖 7zip：

```bash
sudo apt-get install -y p7zip-full
python scripts/verify_checksum.py
python scripts/unzip_all.py
```

解压完成后，`$OAKINK_DIR` 下应出现 `OakBase/`、`image/anno/`、`image/obj/`、`image/stream_release_v2/`、`shape/metaV2/`、`shape/OakInkObjectsV2/`、`shape/oakink_shape_v2/` 和 `shape/OakInkVirtualObjectsV2/`。如果想让本仓库自动探测，也可以把解压后的 OakInk root 放到当前仓库的 `data/`、`OakInk/` 或 `oakink/`。

需要官方 toolkit 重建 MANO hand 时，还要从 [MANO 官网](https://mano.is.tue.mpg.de/) 下载 `mano_v1_2.zip`，解压后按官方 `docs/install.md` 放入 `assets/`。这些模型文件不要提交到 Git。

## 5. 可视化

```bash
python dataset_audit/scripts/dataset_manifest.py --out outputs/dataset_audit/manifest.csv
python dataset_audit/scripts/prepare_sequence_cache.py --item <sample_item>
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/cache/<sample>.npz --fps 3 --stride 20 --loop
```

如果没有真实 OakInk 数据，可以生成并检查 toy cache：

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

`frame_min_signed_distance_mm < 0` 表示 hand 查询点位于 object 或 proxy 内部。`frame_max_negative_depth_mm` 是一帧中的最大非负穿透深度。sequence summary 会包含均值、方差、最大深度、inside ratio，以及 minor/significant penetration frame ratio。

## 7. 表面相交检测

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

surface collision 与 point-SDF depth 是不同指标。它输出 overlap/collision ratio，不产生稳定的 penetration depth。

## 8. 接触信息

静态审计没有识别到 OakInk 的直接逐帧 contact annotation 或内置 penetration checker。OakInk 提供 data loading、split、visualization，并且在数据和资产齐全时可以重建 MANO hand 与 object geometry。如果本地数据没有官方 contact 字段，应通过 hand-object distance threshold 派生 contact，并明确记录为 derived contact。

## 9. 限制

point-SDF 需要 watertight object mesh 或 proxy 才能可靠判断 inside/outside。non-watertight visual mesh 只能作为 heuristic。surface collision 可能把边界接触或 mesh 拟合误差判为相交。使用 aggregate statistics 前，必须用真实 OakInk sequence 验证坐标系、MANO 重建和 object 重建。
