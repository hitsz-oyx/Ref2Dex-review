# Cm 当前状态

- scope: task:Cm
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [架构记录](architecture_log.md)、[接手记忆](repo_memory.md)、[实验记录](experiment_log.md)、[训练结果对比](../../results/object_v2_training_comparison.md)

## 当前状态

- 当前阶段 / 指导: V1.2.1 object-only Cm；正在从受平均 MANO template 影响的旧 GRAB ObjectV2 切换到 subject-template 修复版数据。
- 当前进行中: 新版 mixed C=256 在 GPU 2/3、C=64 在 GPU 1/6 续训，global batch 均为 144。GRAB object-only 的 C=32 geometry-only/no-time hard-gate ablation 已从头启动于 GPU 7（session 22800，单卡 batch 48，50 epoch，W&B offline），run 为 `outputs/cm/cm_object_v2_grab_gate_cm32_geometry_only_no_time_20260823_235356`。HRDexDB 全量 geometry cache 同时在 CPU 后台构建，日志 `output/research/hrdexdb_cache/build_all_v1.log`。Cm 独立数据工具已集中到 `src/task/Cm/tools/data/`，根级薄 wrapper 已删除。
- 最近可靠结论: C=256 当前 best 为 step 118080 / epoch 32 / `12.066 mm`；C=64 当前 best 为 step 84870 / epoch 23 / `13.024 mm`。两版迁移后首步均正常，world size/global batch、optimizer 和 scheduler 恢复通过。修复版 DexYCB C=256 评测平均 `14.92 mm`、相对 zero-flow 改善 `68.27%`。
- 阻塞 / 风险: 迁移使用最近完整 checkpoint，C=256 step 118080 后约 1 epoch、C=64 step 84870 后约 1.5 epoch 的未落盘进度被舍弃；2 GPU 吞吐首步约为 C=256 `200 samples/s`、C=64 `243 samples/s`，低于原 3 GPU。DexYCB 当前只覆盖 subject-10/right，不能外推到全部 subject/side。
- 下一步: 保持新版 C=256/C=64 两条 run 继续；后续若扩展 DexYCB，沿用修复版 adapter 和独立 split，增加其他 subject 后再报告跨主体方差。
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

C=64 对照使用 `src/task/Cm/configs/object_v2_grab_arctic_subject_template_20260820_cm64.yaml` 和相同 batch/resume overrides，运行于 GPU 1/6，输出目录为 `outputs/cm/cm_object_v2_grab_arctic_subject_template_20260820_cm64_20260822_190435`。
