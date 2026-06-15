# 数据集审计补充说明

## 1. 用途

本目录为 GRAB 新增本地、可共享的数据质量审计工具。这不是 GRAB 官方源码。工具代码位于 `dataset_audit/`，生成结果统一写入 `outputs/dataset_audit/`。

## 2. Docker / Miniconda 使用方式

不要在 macOS host 上直接安装依赖或运行数据集脚本。先进入当前仓库的 Docker/Miniconda 环境：

```bash
./.codex/enter.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate grab-dev
```

使用 MeshCat 时，将容器端口 `7000` 映射到 Mac 的 `7011`：

```bash
docker compose -f .codex/compose.yaml run --rm \
  --name grab-meshcat \
  -p 7011:7000 \
  dev bash
```

## 3. Ubuntu 原生使用方式

在 Ubuntu 22.04/24.04 上，可以从 GRAB 仓库根目录直接使用 conda 环境运行审计工具：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential cmake pkg-config ffmpeg \
  libgl1 libglib2.0-0 libosmesa6-dev libegl1 libgl1-mesa-dev

conda create -n grab-dev python=3.9 -y
conda activate grab-dev
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install meshcat rtree
```

使用真实 GRAB 资产前，先跑无数据 smoke test：

```bash
python dataset_audit/scripts/prepare_sequence_cache.py \
  --toy \
  --out-cache outputs/dataset_audit/toy_cache/toy_hand_object.npz
python dataset_audit/scripts/view_meshcat.py \
  --cache outputs/dataset_audit/toy_cache/toy_hand_object.npz \
  --headless-check
```

下载后的 GRAB motion data root 可通过 `GRAB_DATASET_PATH` 或 `GRAB_PATH` 指定；如果 object 资产不在数据根目录内，再设置 `GRAB_OBJECT_ROOT`。如果 Ubuntu 是远程机器，优先用 `--headless-check` 验证；需要浏览器可视化时再转发 MeshCat 端口。

## 4. 数据可用性

审计工具不会下载 GRAB 数据、SMPL-X、MANO 或 object 资产。manifest 会检查 `GRAB_DATASET_PATH`、`GRAB_PATH` 和本地数据目录。缺失数据会被明确报告。

按官方说明，先在 [GRAB 官网](https://grab.is.tue.mpg.de/) 注册、登录并接受 license；Download 区只有登录后可访问。官网还要求按下载页说明获取 `object_meshes.zip`。官方 README 备份见 `README_GRAB.md`。

下载流程：

1. 从官网 Download 区下载 GRAB dataset 的所有 ZIP 文件，不要手动提前解压。
2. 确认已经按官网说明拿到 `object_meshes.zip`，否则后续 mesh 可视化和审计 cache 无法完整重建。
3. 把所有 GRAB ZIP 放到同一个目录，例如 `/storage/downloads/grab_zips/`。
4. 用官方脚本统一解压：

```bash
python grab/unzip_grab.py \
  --grab-path /storage/downloads/grab_zips \
  --extract-path /storage/data/GRAB
```

解压后应得到类似结构：

```text
GRAB/
├── grab/
│   ├── s1/
│   ├── ...
│   └── s10/
├── tools/
│   ├── object_meshes/
│   ├── object_settings/
│   ├── subject_meshes/
│   ├── subject_settings/
│   └── smplx_correspondence/
└── mocap/                 # 可选
```

设置本仓库审计工具会读取的路径：

```bash
export GRAB_DATASET_PATH=/storage/data/GRAB
export GRAB_PATH=$GRAB_DATASET_PATH
export GRAB_OBJECT_ROOT=$GRAB_DATASET_PATH/tools/object_meshes
```

官方 preprocessing、vertex extraction 和 visualization 还需要从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/) 按说明下载 SMPL-X 和 MANO models，并把模型目录传给 `--model-path`，例如：

```bash
export SMPLX_MODEL_FOLDER=/storage/models/smplx
python grab/grab_preprocessing.py \
  --grab-path "$GRAB_DATASET_PATH" \
  --model-path "$SMPLX_MODEL_FOLDER" \
  --out-path outputs/grab_preprocessed
```

GRAB license 对受试者数据和用途有明确限制；下载数据、object mesh、SMPL-X/MANO 模型和生成 cache 都不要提交到 Git。

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

统一约定是 `signed_distance_mm < 0` 表示 body/hand 查询点位于 object 或 proxy 内部。`frame_max_negative_depth_mm` 是该帧最大非负深度。aggregate summary 会对每条 sequence 的均值、方差、最大深度和帧比例再做平均。

## 7. 表面相交检测

```bash
python dataset_audit/scripts/check_surface_collision.py --cache outputs/dataset_audit/cache/<sample>.npz \
  --out-json outputs/dataset_audit/surface_collision.json --out-csv outputs/dataset_audit/surface_collision.csv
```

surface collision 与 point-SDF penetration depth 是不同指标。它可以标记 surface overlap/touch，但不输出稳定深度。

## 8. 接触信息

GRAB 有官方 body-object binary contact map。每个 motion frame 中，object vertices 会被标为 no contact 或与某个 body/hand part 接触。这是 proximity/contact label，不是 signed-distance penetration 或 triangle-collision test。它应与几何负距离统计分开记录。

## 9. 限制

GRAB contact map 和几何穿模检查回答的是不同问题。visual mesh 不一定是 collision mesh。point-SDF 需要 watertight mesh 或 proxy 才能可靠判断符号。surface collision 可能标记 touch 和 fitting artifact。需要验证 SMPL-X/MANO/object 对齐，并用 MeshCat 复查可疑帧。
