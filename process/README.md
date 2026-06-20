# Process Layout

`process/` 存放 Ref2Dex 的数据整理、优化和资产修复脚本。

当前目录约定：

```text
process/
├── preprocess/
│   ├── arctic_preprocess.py
│   └── grab_preprocess.py
├── opti/
│   └── mano_smplx_fit.py
└── repair/
    ├── check_manifold.py
    ├── filter_manifold_objs.py
    └── repair_non_manifold_with_blender*.py
```

相关输入/输出根目录：

- 原始数据：`dataset/arctic`，`dataset/GRAB`
- 共享资产：`assets/shared/mano`
- 数据集对象资产：`assets/arctic/objects`，`assets/grab/objects`
- 预处理输出：`processed_data/arctic`，`processed_data/grab`
- 优化输出：`outputs/mano_fit/*`
- 可视化：`render/processed_data_visualize.py`

## Environment

默认使用：

```bash
conda activate graspenv
cd /home/oyx/test_ws/Ref2Dex
```

如果要显示 Open3D 窗口：

```bash
export DISPLAY=localhost:10.0
```

## ARCTIC 预处理

单条序列：

```bash
python process/preprocess/arctic_preprocess.py \
  --seq s01/box_grab_01 \
  --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
```

单个被试：

```bash
python process/preprocess/arctic_preprocess.py \
  --subject s01 \
  --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
```

全量处理：

```bash
python process/preprocess/arctic_preprocess.py \
  --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
```

批量按 subject 跑：

```bash
for sid in s01 s02 s03 s04 s05 s06 s07 s08 s09 s10; do
  python process/preprocess/arctic_preprocess.py \
    --subject "$sid" \
    --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
done
```

## GRAB 预处理

单条动作名过滤：

```bash
python process/preprocess/grab_preprocess.py \
  --seq airplane_fly_1 \
  --ds-rate 4 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
```

限制帧数做 smoke test：

```bash
python process/preprocess/grab_preprocess.py \
  --seq airplane_fly_1 \
  --ds-rate 4 \
  --max-frames 8 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
```

全量处理：

```bash
python process/preprocess/grab_preprocess.py \
  --ds-rate 4 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
```

批量跑一组动作：

```bash
for seq in airplane_fly_1 airplane_lift airplane_pass_1; do
  python process/preprocess/grab_preprocess.py \
    --seq "$seq" \
    --ds-rate 4 \
    --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
done
```

## MANO/SMPLX 优化

ARCTIC 右手：

```bash
python process/opti/mano_smplx_fit.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/arctic \
  --seq-id s01/box_grab_01 \
  --side right \
  --frame-step 1 \
  --n-iter 20 \
  --lr 0.01 \
  --joint-target-weight 1000 \
  --vertex-target-weight 1000 \
  --lambda-contact-loss 10 \
  --lambda-repulsion-loss 0.05 \
  --respect-hand-valid \
  --device cuda
```

GRAB 右手：

```bash
python process/opti/mano_smplx_fit.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/grab \
  --seq-id s1/airplane_fly_1 \
  --side right \
  --frame-step 1 \
  --n-iter 20 \
  --lr 0.01 \
  --joint-target-weight 1000 \
  --vertex-target-weight 1000 \
  --lambda-contact-loss 10 \
  --lambda-repulsion-loss 0.05 \
  --respect-hand-valid \
  --device cuda
```

批量跑一个数据根下的所有序列右手：

```bash
find /home/oyx/test_ws/Ref2Dex/processed_data/grab -name '*.npz' | sort | while read -r npz; do
  seq="${npz#/home/oyx/test_ws/Ref2Dex/processed_data/grab/}"
  seq="${seq%.npz}"
  python process/opti/mano_smplx_fit.py \
    --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/grab \
    --seq-id "$seq" \
    --side right \
    --frame-step 1 \
    --n-iter 20 \
    --lr 0.01 \
    --joint-target-weight 1000 \
    --vertex-target-weight 1000 \
    --lambda-contact-loss 10 \
    --lambda-repulsion-loss 0.05 \
    --respect-hand-valid \
    --device cuda
done
```

如果左手也要跑，把 `--side right` 改成 `--side left`。

## 可视化

查看 ARCTIC：

```bash
python render/processed_data_visualize.py \
  --seq s01/box_grab_01 \
  --data_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic \
  --show-opti-hand
```

查看 GRAB：

```bash
python render/processed_data_visualize.py \
  --seq s1/airplane_fly_1 \
  --data_root /home/oyx/test_ws/Ref2Dex/processed_data/grab \
  --arctic_root "" \
  --show-opti-hand
```

## Repair

`process/repair/` 下是网格流形性检查和修复工具，详细命令见：

- `process/repair/README.md`
- `process/repair/README_zh-CN.md`

## Notes

- `process/preprocess/*.py` 和 `process/opti/*.py` 现在都按仓库根目录解析依赖，不再假设自己位于旧的 `preprocess/` 目录。
- `meta.json` 是下游读取配置的统一入口；可视化和优化优先从 `processed_data/<dataset>/meta.json` 解析 `dataset_name`、`mano_model_dir`、`object_asset_root`。
