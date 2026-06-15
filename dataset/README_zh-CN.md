# Ref2Dex 数据集工作区

本目录集中放置 Ref2Dex 风格 hand/body-object interaction 审计所需的数据集源码目录：

```text
dataset/
  arctic/
  OakInk/
  OakInk2/
  TACO-Instructions/
  GRAB/
```

每个子目录都作为普通源码文件夹 vendored 到 Ref2Dex 中。上游 `.git` 元数据已经移除，因此 GitHub 会直接展示这些文件，而不是展示 submodule 链接。

## 1. 获取方式和上传模型

正常克隆本仓库即可，不需要任何 submodule 命令：

```bash
git clone <ref2dex-repo-url>
cd Ref2Dex
```

`dataset/` 目录应从 Ref2Dex 父仓库里像普通文件一样 commit。除非明确想重新引入 submodule 或独立 upstream checkout，否则不要恢复子目录里的 `.git`。

## 2. 环境配置

### macOS / Docker 路径

不要在 macOS host 上直接安装依赖或运行数据集处理脚本。请进入每个仓库自己的 Docker/Miniconda 环境：

```bash
cd /Users/key_z/Documents/github/dex/Ref2Dex/dataset/<repo>
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate <repo-dev-env>
```

环境名如下：

```text
arctic              arctic-dev
OakInk              oakink-dev
OakInk2             oakink2-dev
TACO-Instructions   taco-instructions-dev
GRAB                grab-dev
```

### Ubuntu 原生路径

在 Ubuntu 22.04/24.04 workstation 或 server 上，可以用原生 conda 环境运行审计脚本和轻量数据检查：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

cd dataset/<repo>
conda create -n <repo-dev-env> python=<python-version> -y
conda activate <repo-dev-env>
python -m pip install --upgrade pip
python -m pip install numpy scipy trimesh meshcat rtree
```

各数据集环境配置如下：

| 数据集 | 环境名 | Python | 官方/toolkit 依赖 | 审计工具读取的数据根变量 |
| --- | --- | --- | --- | --- |
| ARCTIC | `arctic-dev` | `3.9` | `python -m pip install -r requirements.txt` | 本地 `data/arctic_data/` 和 `outputs/meshcat_cache/` |
| OakInk | `oakink-dev` | `3.9` | `python -m pip install -r requirements.txt`；按 Ubuntu CUDA 栈调整 PyTorch/PyTorch3D | `OAKINK_DIR` 或本地 `data/`、`OakInk/`、`oakink/` |
| OakInk2 | `oakink2-dev` | `3.10` | `python -m pip install -e .`；完整 preview stack 再用 `req_preview.txt` | `OAKINK2_DIR`、`OAKINK2_DATASET_PREFIX` 或本地 `OakInk-v2-hub/`、`data/` |
| TACO-Instructions | `taco-instructions-dev` | `3.9` | `python -m pip install -r requirements.txt` | `TACO_DATASET_ROOT`、`TACO_INSTRUCTIONS_DATASET_ROOT`、`TACO_OBJECT_MODEL_ROOT` |
| GRAB | `grab-dev` | `3.9` | `python -m pip install -r requirements.txt` | `GRAB_DATASET_PATH`、`GRAB_PATH`、`GRAB_OBJECT_ROOT` |

OakInk2 的完整 preview stack 会引用 `thirdparty/` 下的第三方源码目录。在当前无 submodule 布局中，这些目录需要作为普通文件复制或 vendored 进来后，才能安装 `req_preview.txt`。轻量审计 smoke test 不依赖这些 thirdparty 目录。

轻量审计依赖安装完成后，每个数据集仓库都应先通过无数据 smoke test：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

这些工具不会自动下载数据集、MANO/SMPL-X 模型或模型权重。数据集授权、账号登录和模型资产下载必须按每个官方数据集要求手动完成。

## 3. 数据下载和资产放置

授权与下载链接请以原官方文档为准。本工作区中，如果存在原始 README 备份，会保存在各仓库内，例如 `README_ARCTIC.md`、`README_OAKINK.md`、`README_OAKINK2.md`、`README_TACO.md` 和 `README_GRAB.md`。

建议本地数据路径：

| 数据集 | 预期本地输入 |
| --- | --- |
| ARCTIC | `data/arctic_data/`、`data/body_models/`、`data/arctic_data/data/meta/object_vtemplates/` 下的 object template |
| OakInk | `OAKINK_DIR` 或本地 `data/` 中的 OakInk image/shape 数据 |
| OakInk2 | `OakInk-v2-hub/` 或 `data/`，以及用于 preview/toolkit 重建的 MANO/SMPL-X 资产 |
| TACO-Instructions | dataset root、object model root，以及官方 visualization 路径下的 MANO 文件 |
| GRAB | `GRAB_DATASET_PATH` / `GRAB_PATH`、SMPL-X/MANO 模型和 object 资产 |

生成文件和本地数据已经通过 `.gitignore` 忽略：`outputs/`、`.codex/`、下载压缩包、模型文件、数据根目录、Python/macOS 缓存都不应上传 GitHub。

## 4. Manifest 和 Cache 流程

进入某个数据集仓库后：

```bash
python dataset_audit/scripts/dataset_manifest.py \
  --out outputs/dataset_audit/manifest.csv \
  --limit 5
```

真实 sequence cache：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py --item <sample_item>
```

轻量 smoke test：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
```

统一 cache 是 `.npz` mesh sequence，字段尽量包括：

```text
verts_right / verts_left / verts_hand / verts_body
verts_object
faces_right / faces_left / faces_hand / faces_body
faces_object
object_name
subject
sequence
dataset_name
frame_indices
source_paths
contact_available
contact_source
```

每个 adapter 负责把原数据集文件转换为这个统一格式。

## 5. MeshCat 可视化

在数据集容器里运行 MeshCat，并将容器端口 `7000` 映射到 Mac 的 `7011`：

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name <repo>-meshcat \
  -p 7011:7000 \
  dev bash
```

然后运行：

```bash
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/cache/<sample>.npz \
  --fps 3 \
  --stride 20 \
  --loop
```

无浏览器检查：

```bash
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

## 6. 穿模和负距离统计

统一约定如下：

```text
signed_distance_mm < 0  query point 在 object/proxy 内部
signed_distance_mm = 0  接触边界
signed_distance_mm > 0  外部
negative_distance_mm = min(signed_distance_mm, 0)
penetration_depth_mm = max(0, -signed_distance_mm)
```

运行 point-SDF / negative-distance 审计：

```bash
python dataset_audit/scripts/check_point_sdf_penetration.py \
  --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/penetration.json \
  --out-csv outputs/dataset_audit/penetration.csv \
  --surface-mode vertices \
  --stride 20
```

多 sequence 聚合：

```bash
python dataset_audit/scripts/sample_penetration_statistics.py \
  --manifest outputs/dataset_audit/manifest.csv \
  --sample-size 5 \
  --frames-per-seq 100 \
  --out-json outputs/dataset_audit/sample_penetration_statistics.json \
  --out-csv outputs/dataset_audit/sample_penetration_statistics.csv
```

关键字段：

| 字段 | 含义 |
| --- | --- |
| `frame_min_signed_distance_mm` | 当前帧最负 signed distance |
| `frame_max_negative_depth_mm` | 当前帧最大非负 penetration depth |
| `mean_frame_max_negative_depth_mm` | 每条 sequence 中逐帧最大深度的均值 |
| `var_frame_max_negative_depth_mm` | 每条 sequence 中逐帧最大深度的方差 |
| `max_frame_max_negative_depth_mm` | 每条 sequence 的最大深度 |
| `total_inside_ratio` | inside query points / all query points |
| `raw_negative_frame_ratio` | 至少一个 inside point 的帧比例 |
| `minor_penetration_frame_ratio` | 超过 minor depth/ratio 阈值的帧比例 |
| `significant_penetration_frame_ratio` | 超过 significant depth/ratio/count 阈值的帧比例 |

## 7. Surface Collision

```bash
python dataset_audit/scripts/check_surface_collision.py \
  --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json \
  --out-csv outputs/dataset_audit/surface_collision.csv \
  --stride 50
```

surface collision 不是 penetration depth。它输出 surface overlap/intersection 线索和 triangle ratio。它可能把边界接触、共面接触或 visual mesh 拟合误差也计入，因此可疑帧需要用 MeshCat 复查。

## 8. Mesh Topology

```bash
python dataset_audit/scripts/audit_object_mesh_topology.py \
  --limit 20 \
  --out-csv outputs/dataset_audit/object_mesh_topology.csv \
  --out-json outputs/dataset_audit/object_mesh_topology_summary.json
```

该命令输出 watertight、boundary edges、non-manifold edges、connected components、Euler number、volume、area 和 bbox。non-watertight visual mesh 会导致 point-SDF 符号只能作为 heuristic。

## 9. Contact 信息

```bash
python dataset_audit/scripts/inspect_contact_info.py \
  --out-json outputs/dataset_audit/contact_info_summary.json
```

数据集级总结：

| 数据集 | Contact 信息 |
| --- | --- |
| ARCTIC | InterField 最近距离字段可用阈值派生 contact；不是 force contact，也不是 penetration |
| OakInk | 静态审计未识别直接 contact annotation；需要时从 hand-object distance 派生 |
| OakInk2 | affordance/primitive metadata 是语义/任务信息，不是逐帧几何 contact |
| TACO-Instructions | 静态审计未识别直接 contact/penetration 字段；需要时从距离派生 |
| GRAB | 官方 body-object binary contact map 会按 body/hand part 标注 object vertices；不是 penetration depth |

## 10. 实现逻辑

审计层刻意与官方数据集代码分离：

1. `adapters/*_adapter.py` 发现本地数据、读取已有 cache，并记录各数据集的 contact 说明。
2. `common/mesh_io.py` 负责统一 `.npz` cache 和 synthetic toy cache 的读写。
3. `common/topology.py` 在不跑全量数据处理的前提下计算 mesh topology。
4. `common/point_sdf.py` 将 hand/body surface points 查询到 object mesh 上；如果有 `trimesh`，使用 `trimesh.proximity.signed_distance` 并转换为统一符号；否则用 triangle distance + ray-cast heuristic 降级。
5. `common/surface_collision.py` 提供轻量 surface-overlap proxy，用于 smoke test 和人工复查路由。
6. `scripts/*.py` 是稳定 CLI 入口，覆盖 manifest、cache、visualization、penetration statistics、topology audit 和 contact inspection。

实际判断数据质量时，应结合 point-SDF 统计、topology reliability、surface collision 和 MeshCat 人工复查。任何单一指标都不应直接当作 ground-truth contact 或 collision。
