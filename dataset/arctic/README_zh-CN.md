# 数据集审计补充说明

## 1. 用途

本目录为 ARCTIC 新增本地、可共享的数据质量审计工具。这不是 ARCTIC 官方源码。工具代码位于 `dataset_audit/`，生成结果统一写入 `outputs/dataset_audit/`。

## 2. Docker / Miniconda 使用方式

不要在 macOS host 上直接安装依赖或运行数据集脚本。先进入当前仓库的 Docker/Miniconda 环境：

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate arctic-dev
```

使用 MeshCat 时，将容器端口 `7000` 映射到 Mac 的 `7011`：

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name arctic-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu 原生使用方式

在 Ubuntu 22.04/24.04 上，可以从 ARCTIC 仓库根目录直接使用 conda 环境运行审计工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n arctic-dev python=3.9 -y
conda activate arctic-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install meshcat rtree
```

使用真实 ARCTIC 数据前，先跑无数据 smoke test：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

真实数据建议放在 `data/arctic_data/`，body model 放在 `data/body_models/`，object template 放在 `data/arctic_data/data/meta/object_vtemplates/`。如果 Ubuntu 是远程机器，优先用 `--headless-check` 验证；需要浏览器可视化时再转发 MeshCat 端口。

## 4. 数据可用性

审计工具不会下载 ARCTIC 数据、MANO/SMPL-X 文件或物体资产。它可以复用本地已有 cache，例如 `outputs/meshcat_cache/*_world_verts.npz`；当 `data/arctic_data/data/meta/object_vtemplates/` 可用时，也可以为 cache 补充 object faces。

## 5. 可视化

```bash
python dataset_audit/scripts/dataset_manifest.py --out outputs/dataset_audit/manifest.csv
python dataset_audit/scripts/prepare_sequence_cache.py --item s01/capsulemachine_use_01
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/cache/arctic_s01_capsulemachine_use_01_mesh_sequence.npz --fps 3 --stride 20 --loop
```

如果只想检查 cache，不打开浏览器：

```bash
python dataset_audit/scripts/view_meshcat.py --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz --headless-check
```

## 6. 穿模 / 负距离统计

```bash
python dataset_audit/scripts/check_point_sdf_penetration.py \
  --cache outputs/dataset_audit/cache/arctic_s01_capsulemachine_use_01_mesh_sequence.npz \
  --out-json outputs/dataset_audit/arctic_capsulemachine_penetration.json \
  --out-csv outputs/dataset_audit/arctic_capsulemachine_penetration.csv \
  --surface-mode vertices --stride 50

python dataset_audit/scripts/sample_penetration_statistics.py \
  --manifest outputs/dataset_audit/manifest.csv \
  --sample-size 5 --frames-per-seq 100 \
  --out-json outputs/dataset_audit/sample_penetration_statistics.json \
  --out-csv outputs/dataset_audit/sample_penetration_statistics.csv
```

`frame_min_signed_distance_mm < 0` 表示 hand/body 查询点位于 object 或 collision proxy 内部。`frame_max_negative_depth_mm` 是对应的非负穿透深度。sequence 级字段包括 `mean_frame_max_negative_depth_mm`、`var_frame_max_negative_depth_mm`、`max_frame_max_negative_depth_mm`、`total_inside_ratio`、`raw_negative_frame_ratio`、`minor_penetration_frame_ratio` 和 `significant_penetration_frame_ratio`。

## 7. 表面相交检测

```bash
python dataset_audit/scripts/check_surface_collision.py \
  --cache outputs/dataset_audit/cache/arctic_s01_capsulemachine_use_01_mesh_sequence.npz \
  --out-json outputs/dataset_audit/arctic_capsulemachine_surface_collision.json \
  --out-csv outputs/dataset_audit/arctic_capsulemachine_surface_collision.csv \
  --stride 100
```

surface collision 会检查 triangle center 是否进入 object/proxy，并输出 collision ratio。它不输出稳定的 penetration depth。

## 8. 接触信息

ARCTIC 不提供直接人工标注的 force/contact label。它的 InterField 路径提供最近距离字段，例如 `dist.ro`、`dist.lo`、`dist.or` 和 `dist.ol`；contact 通常由 `dist < contact_bnd` 这样的阈值派生。这是几何距离派生的 contact，不是力接触，也不是穿模/碰撞检测。

## 9. 限制

point-SDF 的可靠性依赖 watertight object mesh 或 collision proxy。non-watertight visual mesh 会被标记为 `heuristic`。surface collision 可能把边界接触、共面接触或 mesh 拟合误差判为相交，而且不会输出 penetration depth。把统计结果当作数据集标签前，需要用 MeshCat 复查可疑帧。
