# Cm 当前状态

- scope: task:Cm
- last_updated: 2026-08-29
- last_verified: 2026-08-28
- related: [架构记录](architecture_log.md)、[接手记忆](repo_memory.md)、[实验记录](experiment_log.md)、[训练结果对比](../../results/object_v2_training_comparison.md)

## 当前状态

- 当前阶段 / 指导: V1.2.1 object-only Cm；正在从受平均 MANO template 影响的旧 GRAB ObjectV2 切换到 subject-template 修复版数据。
- 当前进行中: 原 CmDecoder 与 Inspire-F1 decoder-only Cm 均已无活跃训练进程，checkpoint 原样保留。新的 GRAB/Inspire-F1 手流重建混合实验运行于 GPU 0/1/2：global batch=`96`、50 epoch、DenseToken 冻结、source=`0.5/0.5`，输出 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324`；已完成 epoch 21 validation（step `90090`），当前在 epoch 22（step `94380`）训练中，约 `320--335 samples/s`，无 OOM/NaN。
- 最近可靠结论: mixed C=256 在 epoch 45--48 的 val mean stride EPE 约 `11.81 mm`，mixed C=64 在 epoch 35--40 约 `12.53--12.64 mm`，均无继续下降趋势；停止时分别保留最近完整 epoch 48 与 epoch 40 checkpoint。修复版 DexYCB C=256 评测平均 `14.92 mm`、相对 zero-flow 改善 `68.27%`。
- 阻塞 / 风险: 当前 best object `val/mean_stride_epe_mm=8.4257 mm`（epoch 21），best hand `val/mean_stride_hand_epe_mm=2.8225 mm`；GRAB object/hand=`13.2589/4.3258 mm`，Inspire-F1 object/hand=`3.5925/1.3193 mm`。两源均未出现明显 hand loss 失衡，但 object 曲线仍有小幅波动，尚不能宣称收敛。GPU 0 另有轻量进程，本 run 每卡约 `2676 MiB`，当前无显存压力。
- 下一步: 继续跑满 50 epoch；按当前每 epoch 约 43--44 分钟估计，剩余约 29 个 epoch，预计 8 月 30 日中午前后结束。继续关注后续 source×stride object/hand 指标和 slot usage；已生成的 slot-wise t-SNE 产物保留在 `output/research/cm_tsne_inspire_f1_latest_stride5/`。
- HRDexDB 微调入口: `configs/active/hrdexdb_finetune_cm64.yaml`；全量非视频原始数据已迁移到 `dataset/HRDexDB/v0_nonvideo` 并由根 `.gitignore` 排除。共享 geometry cache 允许保留，但 HRDexDB Cm 微调强制 `data.use_dense_cache=false`，DenseToken 输出不得预缓存，必须在线参与反向传播。已完成 loader/builder 框架修正：HRDexDB geometry 需要 4096 稳定物体池、5cm candidate mask，Cm 在线采样 512 点；验证/测试按 source×stride=1/5/10 分开。全量 cache 已导出并生成正式 object-disjoint manifest。
- 证据与相关文档: [EXP-010](experiment_log.md#exp-010--dexycb-subject-10-修复版正式评估)、[EXP-009](experiment_log.md#exp-009--dexycb-subject-10-首次评估无效性诊断)、[EXP-008](experiment_log.md#exp-008--subject-template-修复版-mixed-c64-长训)、[EXP-007](experiment_log.md#exp-007--subject-template-修复版-mixed-正式长训)。

## 新版训练入口

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> \
  /home2/wyy/miniconda3/envs/graspenv/bin/torchrun \
  --standalone --nproc_per_node=<num_gpus> \
  -m src.task.Cm.src.train \
  --config src/task/Cm/configs/active/object_v2_grab_arctic_subject_template_20260820.yaml \
  --distributed
```

当前正式 run 使用 `CUDA_VISIBLE_DEVICES=2,3`、`--set data.batch_size=72 --set data.val_batch_size=72 --set train.resume=<latest.pt>` 续训，输出目录仍为 `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835`。

C=64 严格容量对照使用 `src/task/Cm/configs/active/object_v2_grab_gate_cm64_geometry_only_no_time.yaml`，在 GPU 1/2 以 DDP 启动，per-device batch 32、global batch 64，输出目录为 `outputs/cm/cm_object_v2_grab_gate_cm64_geometry_only_no_time_20260824_155338`。
