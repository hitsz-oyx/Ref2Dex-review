# Process Layout

`process/` 存放 Ref2Dex 的数据整理、优化和资产修复脚本。

当前目录约定：

```text
process/
├── check/
│   └── summarize_penetration.py
├── preprocess/
│   ├── arctic_preprocess.py
│   └── grab_preprocess.py
├── opti/
│   └── mano_smplx_fit.py
├── train/
│   └── prepare_corr_static.py
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
- 优化输出：`processed_data/generated/mano_fit/*`
- 优化结果检查：`process/check/summarize_penetration.py`
- 训练前处理输出：`processed_data/generated/train_corr_static/*`
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
  --device cuda \
  --mano-batch-size 1024 \
  --nn-batch-size 16 \
  --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
```

单个被试：

```bash
python process/preprocess/arctic_preprocess.py \
  --subject s01 \
  --device cuda \
  --mano-batch-size 1024 \
  --nn-batch-size 16 \
  --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
```

全量处理：

```bash
python process/preprocess/arctic_preprocess.py \
  --device cuda \
  --mano-batch-size 1024 \
  --nn-batch-size 16 \
  --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
```

批量按 subject 跑：

```bash
for sid in s01 s02 s03 s04 s05 s06 s07 s08 s09 s10; do
  python process/preprocess/arctic_preprocess.py \
    --subject "$sid" \
    --device cuda \
    --mano-batch-size 1024 \
    --nn-batch-size 16 \
    --output_root /home/oyx/test_ws/Ref2Dex/processed_data/arctic
done
```

## GRAB 预处理

先生成一个固定的 GRAB 子集 manifest（例如 100 条）：

```bash
python process/preprocess/build_grab_subset_manifest.py \
  --sample-size 100 \
  --out-csv /home/oyx/test_ws/Ref2Dex/tmp/manifests/grab_subset_100.csv \
  --out-json /home/oyx/test_ws/Ref2Dex/tmp/manifests/grab_subset_100_summary.json
```

按 manifest 精确预处理：

```bash
python process/preprocess/grab_preprocess.py \
  --manifest /home/oyx/test_ws/Ref2Dex/tmp/manifests/grab_subset_100.csv \
  --ds-rate 4 \
  --device cuda \
  --nn-batch-size 16 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab_subset100_ds4
```

单条动作名过滤：

```bash
python process/preprocess/grab_preprocess.py \
  --seq airplane_fly_1 \
  --ds-rate 4 \
  --device cuda \
  --nn-batch-size 16 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
```

限制帧数做 smoke test：

```bash
python process/preprocess/grab_preprocess.py \
  --seq airplane_fly_1 \
  --ds-rate 4 \
  --max-frames 8 \
  --device cuda \
  --nn-batch-size 16 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
```

全量处理：

```bash
python process/preprocess/grab_preprocess.py \
  --ds-rate 4 \
  --device cuda \
  --nn-batch-size 16 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
```

批量跑一组动作：

```bash
for seq in airplane_fly_1 airplane_lift airplane_pass_1; do
  python process/preprocess/grab_preprocess.py \
    --seq "$seq" \
    --ds-rate 4 \
    --device cuda \
    --nn-batch-size 16 \
    --output-root /home/oyx/test_ws/Ref2Dex/processed_data/grab
done
```

## MANO/SMPLX 优化

先从预处理好的双手 `.npz` 里生成单手 job 清单：

```bash
python process/opti/build_hand_job_manifest.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/grab_subset100_ds4 \
  --out-csv /home/oyx/test_ws/Ref2Dex/tmp/manifests/grab_subset100_hand_jobs.csv \
  --out-json /home/oyx/test_ws/Ref2Dex/tmp/manifests/grab_subset100_hand_jobs_summary.json
```

按单手 manifest 批量跑 Stage 2：

```bash
python process/opti/run_mano_fit_batch.py \
  --manifest /home/oyx/test_ws/Ref2Dex/tmp/manifests/grab_subset100_hand_jobs.csv \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/grab_subset100_ds4 \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit/grab_subset100_rep3e-2 \
  --summary-json /home/oyx/test_ws/Ref2Dex/tmp/logs/mano_fit_grab_subset100_rep3e-2_batch_summary.json \
  --device cuda \
  --repulsion-mode sdf_grid \
  --penetration-tol-mm 2.0 \
  --lambda-repulsion-loss 0.03 \
  --lambda-contact-loss 10.0 \
  --frame-batch-size 8 \
  --n-iter 100 \
  --lr 0.01
```

ARCTIC 右手：

```bash
python process/opti/mano_smplx_fit.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/arctic \
  --seq-id s01/box_grab_01 \
  --side right \
  --frame-step 1 \
  --frame-batch-size 8 \
  --n-iter 20 \
  --lr 0.01 \
  --joint-target-weight 1000 \
  --vertex-target-weight 1000 \
  --lambda-contact-loss 10 \
  --lambda-repulsion-loss 0.05 \
  --device cuda
```

GRAB 右手：

```bash
python process/opti/mano_smplx_fit.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/grab \
  --seq-id s1/airplane_fly_1 \
  --side right \
  --frame-step 1 \
  --frame-batch-size 8 \
  --n-iter 20 \
  --lr 0.01 \
  --joint-target-weight 1000 \
  --vertex-target-weight 1000 \
  --lambda-contact-loss 10 \
  --lambda-repulsion-loss 0.05 \
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
    --frame-batch-size 8 \
    --n-iter 20 \
    --lr 0.01 \
    --joint-target-weight 1000 \
    --vertex-target-weight 1000 \
    --lambda-contact-loss 10 \
    --lambda-repulsion-loss 0.05 \
    --device cuda
done
```

如果左手也要跑，把 `--side right` 改成 `--side left`。

当前 `mano_smplx_fit.py` 默认会按 Stage 1 的 `frame_keep_3cm` / `hand_valid` 只导出保留帧。
如果只是为了 debug，想把非接触帧也一并保留到 `.pkl`，加：

```bash
python process/opti/mano_smplx_fit.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/arctic \
  --seq-id s01/box_grab_01 \
  --side right \
  --keep-all-frames \
  --device cuda
```

如果第一次开 `contact_loss`，脚本会自动在 `processed_data/<dataset>/_cpf_contact_cache_v1/` 下生成 `cpf_contact_data` 缓存；后续重复优化会直接命中缓存。想强制重建时加：

```bash
python process/opti/mano_smplx_fit.py \
  --processed-root /home/oyx/test_ws/Ref2Dex/processed_data/arctic \
  --seq-id s01/box_grab_01 \
  --side right \
  --frame-batch-size 8 \
  --lambda-contact-loss 10 \
  --device cuda \
  --rebuild-cpf-cache
```

## 优化结果穿模检查

直接汇总优化 `.pkl` 里的 `penetration_depth` 字段，默认按 `2 / 5 / 10 mm` 三档统计。如果旧结果里没有这个字段，也可以加 `--recompute` 从 `opt_hand_verts_world + object_trajectory + object mesh` 重新计算。

```bash
python process/check/summarize_penetration.py \
  --input-pkl /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit_sdfgrid_tol2mm/grab_smplx_cpf/s1/airplane_fly_1_right.pkl \
  --depth-threshold-mm 2 5 10 \
  --out-json /home/oyx/test_ws/Ref2Dex/tmp/check/grab_airplane_fly_1_right_penetration.json \
  --out-csv /home/oyx/test_ws/Ref2Dex/tmp/check/grab_airplane_fly_1_right_penetration.csv
```

如果想强制重算 penetration depth：

```bash
python process/check/summarize_penetration.py \
  --input-pkl /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit_sdfgrid_tol2mm/grab_smplx_cpf/s1/airplane_fly_1_right.pkl \
  --depth-threshold-mm 2 5 10 \
  --recompute
```

## 训练前处理

单条 Stage 2 结果转成 Stage 3 训练样本：

```bash
python process/train/prepare_corr_static.py \
  --mano-opt-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit/grab_smplx_cpf \
  --seq-id s1/airplane_fly_1 \
  --side right \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/train_corr_static
```

批量处理一个数据根下的全部单手 `.pkl`：

```bash
python process/train/prepare_corr_static.py \
  --mano-opt-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit/grab_smplx_cpf \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/train_corr_static
```

指定只处理左手或右手：

```bash
python process/train/prepare_corr_static.py \
  --mano-opt-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit/arctic_smplx_cpf \
  --side left \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/train_corr_static
```

如果想压缩 `.npz`，加：

```bash
python process/train/prepare_corr_static.py \
  --mano-opt-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/mano_fit/grab_smplx_cpf \
  --output-root /home/oyx/test_ws/Ref2Dex/processed_data/generated/train_corr_static \
  --save-compressed
```

Stage 3 默认配置：

- `num_obj_points = 512`
- `num_hand_points = 1538`
- `num_near_points = 256`
- `K_obj_local = 16`
- `K_hand_local = 16`
- `K_cross = 32`
- `obj_contact_label` 使用 `d_pos=5mm, d_neg=30mm, gamma=2.0` 的 soft distance decay

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
- `process/train/prepare_corr_static.py` 读取的是 Stage 2 单手 `.pkl`，输出按序列打包的 Stage 3 `.npz`；每个 `.npz` 内部的 sample unit 是 `single_frame_single_hand`。
- `meta.json` 是下游读取配置的统一入口；可视化和优化优先从 `processed_data/<dataset>/meta.json` 解析 `dataset_name`、`mano_model_dir`、`object_asset_root`。
- `mano_smplx_fit.py` 现在会在 `processed_data/generated/mano_fit/<dataset>_smplx_cpf/meta.json` 写出 Stage 2 schema 信息；`prepare_corr_static.py` 会在 `processed_data/generated/train_corr_static/<dataset>/meta.json` 写出 Stage 3 schema 信息。
- 预处理默认改为 `np.savez` 非压缩保存，速度更快；只有在确实要省磁盘时再加 `--save-compressed`。
- 预处理最近邻现在支持 GPU：`--device cuda --nn-batch-size 16` 是比较稳妥的起点；显存够的话可以继续增大。
- MANO/SMPLX 优化现在支持多帧 batch：`--frame-batch-size 8` 起步即可；如果只看 `jt_loss/vert_loss`，把 `--lambda-contact-loss 0 --lambda-repulsion-loss 0` 会明显更快。
- CPF contact 准备现在默认走 torch 路径，并带落盘缓存；第一次会慢一些，但后续同一 `processed-root/seq/side` 的优化基本不再重复做 contact assignment。
