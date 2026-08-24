# CmAction Pipeline

## 数据与划分

Stage 4 缓存每条 sequence 的世界坐标轨迹：

```text
<split>/<subject>/<sequence>/shared.npz
<split>/<subject>/<sequence>/left.npz
<split>/<subject>/<sequence>/right.npz
```

`shared.npz` 存储 `obj_points_world [T,4096,3]`、法向、稳定点 ID、帧号与采样率；
side 文件存储 hand points/normals、`hand_root_pose_world` 和
`obj_candidate_mask_5cm`。Dataset 在 current frame 的 hand-root 坐标系中构造
`obj_points`、`hand_points`、`obj_flow_gt` 和 `hand_flow`。

训练、验证与测试必须是 sequence 级互斥目录。正式配置为
`sequence_hard_gate_train_val_test_short.yaml`，使用 `train/val/test` 三个显式路径；
test 不参与 checkpoint 选择。

## 校准 object flow target scale

只对 train 目录运行：

```bash
conda activate graspenv
python -m src.task.Cm.tools.data.compute_flow_scale \
  --train-path data/processed_data/stage4/cm_sequence_hard_gate_val20_split/train \
  --min-stride 1 --max-stride 10 --num-obj-points 512
```

脚本复现 Dataset 的 current-frame 范围 `range(T - max_stride)`，每个 current
枚举所有 stride，并始终只统计 5cm candidate。每个 pair 的权重是
`min(candidate_count, num_obj_points)`，所以候选区域大不会被过度加权。结果写入
`train/metadata.json`，并将打印的 `flow_target_rms_m` 与 `flow_target_scale`
同步写入 YAML。不要用 val/test 重新校准。

## 训练与评估

```bash
conda activate graspenv
PYTHONPATH=. python -m src.task.Cm.train \
  --config src/task/Cm/configs/sequence_hard_gate_train_val_test_short.yaml
```

启动时会检查 schema、hand-root 坐标系、object/hand 点数和 calibration metadata。
正式配置的 `meta.require_flow_calibration: true` 要求 metadata 完整存在；若
`object_flow_target_scale`、stride 范围、`active_only` 或 512 点采样规则不一致，
训练会失败而不是静默使用错误 scale。只做 scale 消融时，才应显式关闭该字段。

训练 step 只记录 EPE；训练 epoch 在全 epoch 点加权聚合后再计算 relative EPE、
norm ratio 与 zero-flow improvement。验证会完整计算每个固定 stride 的同类指标。
JSONL 中保留 `val/stride_*/*` 与 `test/stride_*/*`；W&B 对两者都记录汇总及
stride 1/5/10 EPE。最佳模型由 `val/mean_stride_epe_mm` 决定。overfit 配置没有验证集，故
`metric_for_best: null`，应查看 latest checkpoint 与训练曲线。

评估或测试：

```bash
PYTHONPATH=. python -m src.task.Cm.eval \
  --checkpoint outputs/train/<run>/checkpoints/best.pt \
  --split val \
  --device cuda
```

Use `--split test` only for the final held-out evaluation; the evaluator never
selects the test split implicitly.

## 当前模型接口

模型输入仅包含当前 object/hand 几何、当前 normals、hand flow、valid mask；
输出的核心字段是：

```text
pred_obj_flow        # [B, N_obj, 3] metres
cm_tokens            # [B, K, cm_dim]
cm_anchor_pos        # [B, K, 3] metres
cm_anchor_normal     # [B, K, 3]
cm_assignment        # [B, K, N_hand], sum over K = 1
cm_slot_weights      # [B, K, N_hand], sum over N_hand = 1
slot_nonzero_prob    # [B, K]
slot_hard_mask       # [B, K], fallback guarantees at least one true slot
decoder_slot_usage   # [B, K], sum over K = 1
```

没有 `wrist_delta`、`cm_hand_flow`、null expert、activity head 或 Top-K anchor
接口。旧 Cm head checkpoint 与当前 hard-gate 架构不兼容；冻结 DenseToken checkpoint
仍可复用。
