# ObjectInteractionCm 实验记录

- scope: task:ObjectInteractionCm
- related: [任务入口](../README.md)、[V1.1 执行计划](../plan/V1.1.md)、[V1.1 架构](../architecture/V1.1.md)、[活动记录](activity_log.md)

## 2026-09-10 — V1.3 unique-KNN-hand 全量训练（进行中）

- run_id: `object_interaction_cm_dexplore_rl_v1_3_20260910_020856`
- run_status: `RUNNING`
- modification_version: `V1.3`
- operation_category: `experiment / operation`
- base_commit: `158e0f34068e260d7077a94b4457799b7fca2f31`
- seed: `42`
- initial_checkpoint: `null`（from scratch）
- data index: [V1.3 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json)
- scale manifest: [V1.3 unique-KNN scale](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3_unique_knn.json)
- split: train `509`（MANO `254` / Inspire-F1 `255`）、val `58`（`28` / `30`）、test `63` MANO-only；沿用 V1.2.5 assignment。
- training: 右手、`object_pose_t`、30 Hz、object pool/sample `4096/1024`、KNN `K=32`、interaction/hand supervision radius `2 cm`、MANO/Inspire KNN hand `2048/10135`、unique valid-edge hand supervision、batch-max dynamic padding、每卡 batch `32`、global batch `96`、目标 `202300` steps、物理 GPU `0,2,3`。

**假设与变量**

在不改变 split、GT、坐标系和 object sampling 语义的前提下，使用采样 object 点的有效 KNN 边构造
interaction；将这些边中的 hand ID 去重后，每个 hand 点只进入一次 hand decoder/loss。动态 padding
只到当前 batch 的最大 unique hand count，不将完整 Inspire `10135` 点固定送入每个 batch。

**运行入口**

- [V1.3 指导](../指导/V1.3.md)
- [V1.3 执行计划](../plan/V1.3.md)
- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/)
- [配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/config.json)
- [运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/run_manifest.json)
- [逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/metrics.jsonl)
- [训练日志](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/train.log)
- [最佳 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt) 与 [最近 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/latest.pt) — 已生成；epoch `1` 至 `4` 阶段 checkpoint 已生成。

**当前状态与结论边界**

- 已完成 `step=2944`、`epoch=4`；训练进程仍为 `RUNNING`，最新吞吐约 `645 samples/s`，ETA 约 `8.2 h`。
- 当前仅支持工程启动和 forward/backward 运行正常；正式科研结论保持 `INCONCLUSIVE`，待完整训练、
  validation 和 checkpoint 结果后再更新。

## 2026-09-07 — V1.2.5 修正 DExplore 实际物体轨迹、KNN=16 的 Cm 重训（已完成）

- run_id: `object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606`
- run_status: `COMPLETED`
- modification_version: `V1.2.5`
- operation_category: `experiment / operation / data`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- seed: `42`
- initial_checkpoint: `null`（from scratch）
- data index: `data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json`
- scale manifest: `data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/scales_train_v1_2_5.json`
- split: train `509`（MANO 254 / RL-Inspire 255）、val `58`（28 / 30）、test `63` MANO-only；630 个 parent sequence 全部互斥。
- training: 实际 simulated object trajectory、右手 1538 点、object pool/sample `4096/1024`、`D=128,C=32,S=16,K=16`，global batch 96，GPU 0/1/2，最大 202300 steps。

**假设与变量**

修正后的 RL-Inspire hand/object 联合轨迹能够为 ObjectInteractionCm 提供一致的 object-flow GT；将局部
KNN 从 8 改为 16 后，从随机初始化学习新的 object-centric Cm。其余 stride、source probability、loss、
优化器和 checkpoint 选择规则沿用 V1.2.3。旧错误轨迹 checkpoint 不参与初始化或对照选择。

**运行入口**

- [V1.2.5 执行计划](../plan/V1.2.5.md)
- [修正 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)、[index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json) 与 [KNN=16 scale](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/scales_train_v1_2_5.json)
- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/)、[配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/config.json)、[运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/run_manifest.json)、[逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/metrics.jsonl) 与 [训练日志](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/train.log)
- [当前 best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/latest.pt)

**结果与结论边界**

- 三卡 2-step smoke 正常完成；正式 run 在 262 个 epoch 后达到 `202300/202300` steps，自然结束，耗时
  `06:30:21`。全程 loss/gradient finite，未发现 traceback、NaN、OOM 或 NCCL failure。
- 按两 source 等权 `val/obj/flow_epe_mm` 选择的 best 位于 epoch 143 / step 110682：总指标
  `6.422504 mm`，MANO `7.243497 mm`，RL-Inspire `5.601512 mm`；对应 source hand-flow EPE 为
  MANO `2.809946 mm`、RL-Inspire `1.776661 mm`。
- 最终 epoch 262 / step 202300 的等权 object EPE 为 `6.706175 mm`（MANO `7.733837 mm`、RL-Inspire
  `5.678513 mm`），略差于 best；后续消费必须显式使用 `best.pt`，不能把 `latest.pt` 当作最佳模型。
- 结论 `SUPPORTED`：修正实际物体轨迹、KNN=16 的 Cm 训练链路稳定完成并生成有效 checkpoint。
  该结论不证明 CmDecoderV2 或 rollout 已改善；下游效果仍为 `INCONCLUSIVE`，需使用本轮 best checkpoint
  重新训练/评估 decoder。

## 2026-09-05 — V1.2.3 Dexplore RL/MANO right-hand Cm mixed training（已停止）

- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `STOPPED`
- modification_version: `V1.2.3`
- operation_category: `experiment / operation / data`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- seed: `42`
- initial_checkpoint: `null`（from scratch）
- data index: `data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json`
- scale manifest: `data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json`
- split: train `1004`（MANO 501 / RL-Inspire 503）、val `126`（64 / 62）、test `125` MANO-only；同一 parent sequence 不跨 variant。
- stride: 两种 source 均为 30 Hz、stride `1..10`；val/test 固定 stride 2。
- training: 右手 1538 点，object pool 4096 / sample 1024，`D=128,C=32,S=16,K=8`，source probability `0.5/0.5`，每卡 batch 32、global batch 96、最大 202300 steps，GPU 0/1/2。

**假设与边界**

新 Dexplore RL-Inspire 与 MANO variant 在不共享 parent sequence 的前提下混合训练，能够学习不依赖 source/手形态 ID 的 object-centric Cm。当前只证明训练工程链路；CmDecoderV2 和 test 可视化不在本 run 内，科研结论保持 `INCONCLUSIVE` 直到正式训练终态及后续 decoder 验证。

**运行入口**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)
- [配置快照](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)
- [运行清单](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)
- [逐步指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)
- [训练日志](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [最佳 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt) — epoch 46 / step 81420，equal-source object EPE `9.470077 mm`。
- [最近 checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt) — epoch 65 / step 115050，包含 optimizer/scheduler/scaler，可恢复。

**结果与终态**

- run 在 step 115900 / epoch 66 非正常停止，只完成 202300 计划 steps 的约 57.3%；最后完整 checkpoint/validation 为 step 115050 / epoch 65。
- 最佳 validation 位于 epoch 46 / step 81420：equal-source object EPE `9.470077 mm`，MANO `6.301521 mm`，RL-Inspire `12.638633 mm`；对应 hand EPE 为 MANO `4.170636 mm`、RL-Inspire `3.261057 mm`。
- 最后完整 validation 的 equal-source object EPE 为 `9.574271 mm`；训练到中断前 loss/gradient finite，未发现 NaN、Python traceback、CUDA OOM 或 kernel OOM 证据。
- 进程在日志无终止标记的情况下消失，准确外部信号未知；因此 `run_status=STOPPED`，科研结论保持 `INCONCLUSIVE`。该中途 checkpoint 可用于恢复，但不能当作完成的 202300-step 正式结果。

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
