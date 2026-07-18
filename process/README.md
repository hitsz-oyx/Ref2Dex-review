# 数据处理目录

当前流水线只有 Stage 2 和 Stage 3，不生成独立 Stage 1 数据。

```text
raw dataset
  -> process/<DATASET>/optimize.py
  -> processed_data/generated/stage2/<variant>/*.pkl
  -> process/stage3/prepare_corr_static.py
  -> processed_data/generated/stage3/<variant>/*.npz
  -> correspondence_ptv3
```

## 目录职责

```text
process/
├── GRAB/
│   ├── optimize.py       # GRAB raw -> 公共 Stage 2
│   ├── raw.py            # GRAB 原始字段解析、MANO 和几何构造
│   ├── build_subset.py   # 确定性构建 GRAB 子集 manifest
│   └── geneoh.py         # 可选 GeneOH 物体轨迹去噪实验
├── ARCTIC/
│   ├── optimize.py       # ARCTIC raw -> 公共 Stage 2
│   └── raw.py            # ARCTIC 原始字段解析、MANO 和几何构造
├── common/
│   └── stage2.py         # 公共 Stage 2 schema、过滤和保存
└── stage3/
    └── prepare_corr_static.py
```

`raw.py` 是数据集内部适配器，不是落盘 Stage1 数据的入口。数据集差异留在
各自目录内；跨数据集共享的落盘字段只由 `process/common/stage2.py` 定义。

独立网格检查和修复脚本位于 `tools/mesh_repair/`。

## 环境

```bash
conda activate graspenv
cd /home/oyx/test_ws/Ref2Dex
export DISPLAY=localhost:10.0
```

以下命令均从仓库根目录运行，并使用模块入口。

## GRAB

构建固定的 100 序列 manifest：

```bash
python -m process.GRAB.build_subset \
  --sample-size 100 \
  --out-csv tmp/manifests/grab_subset_100.csv \
  --out-json tmp/manifests/grab_subset_100_summary.json
```

生成公共 Stage 2：

```bash
python -m process.GRAB.optimize \
  --manifest tmp/manifests/grab_subset_100.csv \
  --side both \
  --num-obj-points 4096 \
  --frame-keep-threshold 0.05 \
  --ds-rate 4 \
  --device cuda \
  --output-root processed_data/generated/stage2/grab_subset100_initonly_4096_ds4
```

单序列 smoke test：

```bash
python -m process.GRAB.optimize \
  --seq s1/bowl_pass_1 \
  --side right \
  --max-frames 8 \
  --device cuda \
  --output-root processed_data/generated/stage2/grab_smoke
```

`process.GRAB.geneoh` 是可选实验入口，不属于默认数据生成链。

## ARCTIC

生成公共 Stage 2：

```bash
python -m process.ARCTIC.optimize \
  --side both \
  --num-obj-points 4096 \
  --frame-keep-threshold 0.05 \
  --device cuda \
  --output-root processed_data/generated/stage2/arctic_initonly_4096
```

可用 `--seq s05/box_grab_01`、`--subject s05` 或 `--raw-file <path>`
限制输入范围。

## Stage 3

```bash
python -m process.stage3.prepare_corr_static \
  --stage2-root processed_data/generated/stage2/grab_subset100_initonly_4096_ds4_handrootsrc \
  --output-root processed_data/generated/stage3/grab_subset100_initonly_4096_ds4_hand_root_v2 \
  --num-obj-pool 4096 \
  --num-obj-train 512 \
  --candidate-threshold 0.05 \
  --coordinate-frame hand_root \
  --device cuda
```

Stage 3 v2 保存完整 4096 点物体池和法向、1538 个手点和法向、5cm 候选掩码，以及
hand-to-full-object 最短距离（dense hand heatmap GT）。训练时根据 `(frame, epoch)`
的稳定种子从候选池采样 512 点；候选不足时直接 padding。它不再写入 KNN、point id、
canonical hand 或 finger/region 字段，因此只能供 correspondence_ptv3_v2 使用。

## 可视化与检查

```bash
DISPLAY=localhost:10.0 python -m render.stage2_visualize \
  --input processed_data/generated/stage2/<variant>/<subject>/<seq>_<side>.pkl

# Legacy only: does not accept the minimal Stage 3 v2 schema.
DISPLAY=localhost:10.0 python -m render.stage3_visualize \
  --input processed_data/generated/stage3/<variant>/<subject>/<seq>_<side>.npz

# Legacy only: does not accept the minimal Stage 3 v2 schema.
python -m tools.stage3_npz_to_ply \
  --input processed_data/generated/stage3/<variant>/<subject>/<seq>_<side>.npz \
  --object-view sampled \
  --epoch 0
```
