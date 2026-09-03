# ObjectInteractionCm 实验记录

- scope: task:ObjectInteractionCm
- related: [任务入口](../README.md)、[V1.1 执行计划](../plan/V1.1.md)、[V1.1 架构](../architecture/V1.1.md)、[活动记录](activity_log.md)

## 2026-09-03 — ObjectInteractionCm best 的 GRAB/Inspire-F1 t-SNE

- run_id: `objectinteractioncm_tsne_best_20260903`
- checkpoint: `outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt`（epoch 17 / step 193953，`object_pose_t`）
- split: test；GRAB stride `1..10`，Inspire-F1 偶数 stride `2..20`；各 source 每 stride 等量采样，最终 520/source、1040 条 pooled 样本。
- 产物：[output/research/objectinteractioncm_tsne_best_20260903/](../../../../../output/research/objectinteractioncm_tsne_best_20260903/)
- pooled 原空间 dataset silhouette=`0.1835287`；stride 着色在主体区域内明显交叠，而 dataset 着色仍可见 source-specific 区域。该结果是探索性表征诊断，不能替代下游泛化评估；结论状态 `SUPPORTED`（运行和产物有效）。

## 2026-09-03 — V1.1.3 GRAB + Inspire-F1 available-hand union 全量训练

- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- modification_version: V1.1.3
- operation_category: experiment / operation
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- seed: 42
- initial_checkpoint: null
- conclusion: INCONCLUSIVE

**假设与变量**

在 V1.1 冻结合同下，单一 `16×32` object-centric Cm 可以从所有真实可用 hand points 的运动中稳定学习 object/hand flow。GRAB 与 Inspire-F1 按 `0.5/0.5` source probability 混训；GRAB stride 为 `1..10`，Inspire-F1 stride 为偶数 `2..20`；object pool 4096 点、运行时采样 1024 点；不输入 source/side ID 或 GT object flow。

**运行入口**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/)
- [配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)
- [运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)
- [终态摘要](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json)
- [逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl)
- [最佳 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt)

**结果**

训练在 18 个 epoch 后达到 202300 steps 并正常结束，无 NaN/Inf 或未处理异常。按两 source 等权 object-flow EPE 选择的最佳 checkpoint 位于 epoch 17 / step 193953：

| 指标 | best epoch 17 | final epoch 18 |
| --- | ---: | ---: |
| equal-source object EPE (mm) | 4.089469 | 4.097444 |
| GRAB object EPE (mm) | 5.823295 | 5.851712 |
| Inspire-F1 object EPE (mm) | 2.355643 | 2.343177 |
| GRAB hand EPE, 3 cm mask (mm) | 2.729673 | 2.728593 |
| Inspire-F1 hand EPE, 3 cm mask (mm) | 0.777010 | 0.774676 |

**结论边界**

该运行支持“实现能够稳定完成混合训练并生成可用 checkpoint”这一工程判断。由于尚未运行 held-out test、旧 Cm/OI-Cm 对照或下游任务验证，不能判断 V1.1 表征是否优于基线或具有预期泛化能力，科研结论标记为 `INCONCLUSIVE`。
