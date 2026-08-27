# Cm 当前状态

- scope: task:Cm
- last_updated: 2026-08-27
- last_verified: 2026-08-27
- related: [架构记录](architecture_log.md)、[接手记忆](repo_memory.md)、[实验记录](experiment_log.md)、[训练结果对比](../../results/object_v2_training_comparison.md)

## 当前状态

- 当前阶段 / 指导: V1.2.1 object-only Cm；正在从受平均 MANO template 影响的旧 GRAB ObjectV2 切换到 subject-template 修复版数据。
- 当前进行中: 严格 C=64 additive 已跑满 50 epoch / 202300 steps并已退出；严格 C=32 additive 已从 epoch 27 / step `109242` checkpoint 恢复，迁移到 GPU 0/2，配置和 global batch 不变，当前进入 epoch 29；迁移后 epoch 28 validation mean stride EPE=`13.920 mm`。C=32 hard-gate objective-only 与 C=16 additive capacity match 在 GPU 2/3 运行至 epoch 12。Inspire-F1-only C=64 additive 全量 DenseToken 阶段已停止于 epoch 8 / step `19160`，最新 checkpoint 已用于 decoder-only continuation：DenseToken 冻结、GPU 1/3、当前 step `34500` / epoch 15 训练中，最近一次完整 validation 为 epoch 14 / step `33530`，输出 `outputs/cm/cm_hrdexdb_inspire_f1_decoder_only_resume_20260827_011451`；原始全量微调输出保留不变。所有当前 run 无 OOM/NaN。
- 最近可靠结论: mixed C=256 在 epoch 45--48 的 val mean stride EPE 约 `11.81 mm`，mixed C=64 在 epoch 35--40 约 `12.53--12.64 mm`，均无继续下降趋势；停止时分别保留最近完整 epoch 48 与 epoch 40 checkpoint。修复版 DexYCB C=256 评测平均 `14.92 mm`、相对 zero-flow 改善 `68.27%`。
- 阻塞 / 风险: 同 epoch 5，C=32 additive=`16.834 mm`，优于严格 C=32 hard-gate 的 `18.497 mm`；但 hard-gate 尚处 full-open warm-up，需观察 epoch 6--10 ramp。C=16 additive epoch 5=`18.215 mm`，较 C=32 additive差 `1.381 mm`，且 contribution usage 更集中（effective branch `8.88` vs `13.71`），尚未单-slot collapse。Inspire-F1 decoder-only continuation 最近 6 次 validation 的 mean stride EPE 在 `3.656--3.873 mm` 间波动，尚未形成严格收敛平台；GPU 3 与其他 run 共用导致吞吐约 `70 samples/s`（日志早期受竞争影响更低），需要继续观察后续 validation。
- 下一步: 继续观察 C=32 hard-gate epoch 6--10 gate ramp；跟踪 C=16 additive 容量平台；检查 Inspire-F1 微调首个 validation 后的 source/stride 指标，并在必要时将该 run 迁移到空闲 GPU。
- HRDexDB 微调入口: `configs/hrdexdb_finetune_cm64.yaml`；全量非视频原始数据已迁移到 `dataset/HRDexDB/v0_nonvideo` 并由根 `.gitignore` 排除。共享 geometry cache 允许保留，但 HRDexDB Cm 微调强制 `data.use_dense_cache=false`，DenseToken 输出不得预缓存，必须在线参与反向传播。已完成 loader/builder 框架修正：HRDexDB geometry 需要 4096 稳定物体池、5cm candidate mask，Cm 在线采样 512 点；验证/测试按 source×stride=1/5/10 分开。全量 cache 已导出并生成正式 object-disjoint manifest。
- 证据与相关文档: [EXP-010](experiment_log.md#exp-010--dexycb-subject-10-修复版正式评估)、[EXP-009](experiment_log.md#exp-009--dexycb-subject-10-首次评估无效性诊断)、[EXP-008](experiment_log.md#exp-008--subject-template-修复版-mixed-c64-长训)、[EXP-007](experiment_log.md#exp-007--subject-template-修复版-mixed-正式长训)。

## 新版训练入口

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> \
  /home2/wyy/miniconda3/envs/graspenv/bin/torchrun \
  --standalone --nproc_per_node=<num_gpus> \
  -m src.task.Cm.train \
  --config src/task/Cm/configs/object_v2_grab_arctic_subject_template_20260820.yaml \
  --distributed
```

当前正式 run 使用 `CUDA_VISIBLE_DEVICES=2,3`、`--set data.batch_size=72 --set data.val_batch_size=72 --set train.resume=<latest.pt>` 续训，输出目录仍为 `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_20260822_125835`。

C=64 严格容量对照使用 `src/task/Cm/configs/object_v2_grab_gate_cm64_geometry_only_no_time.yaml`，在 GPU 1/2 以 DDP 启动，per-device batch 32、global batch 64，输出目录为 `outputs/cm/cm_object_v2_grab_gate_cm64_geometry_only_no_time_20260824_155338`。
