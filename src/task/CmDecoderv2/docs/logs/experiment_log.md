# CmDecoderv2 实验记录

## 2026-09-10 — V1.3 OICM + Inspire 全点监督 decoder 正式训练

- experiment_id: `cmdecoderv2-temporal-d2-oicm-v1.3-full10135-v1.1.7`
- activity_id: [`cmdecoderv2-v1.1.7-formal-20260910-163524`](activity_log.md)
- run_id: `cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812`
- run_status: `COMPLETED`
- modification_version: `V1.1.7`
- operation_category: `architecture / code / data / experiment / operation`
- importance: `primary`
- pinned: `true`

**假设与固定合同**

验证最终 OICM V1.3 的 `K=32`、2 cm unique-KNN edge stream 能否在不回退到 1538 点 hand stream 的情况下，接入 Temporal-D2 decoder，并使用完整 Inspire-F1 surface `10135` 点进行 point-flow supervision。2 cm 只用于 KNN edge validity 和 `active_only` frame gating；point-flow loss 对全部 `10135` 点计算，不使用旧的 point-level 2 cm mask。

- frozen OICM: [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)，SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
- data view: [dexplore_rl_v1_3_full10135](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/)，train/val/test=`255/30/63` sequences，train/val windows=`63988/7807`。
- supervision: target `knn_hand_points_world.npy`，`10135` points，all points=`true`，point mask=`none`；hand stream=`unique_knn_edges`，`K=32`，radius=`0.02 m`，distance runtime recompute，batch-max dynamic padding。
- split: `inspire_rl` train/val only；MANO test 保持 qualitative-only，不参与训练或正式定量 test。
- decoder: Temporal-D2，`K=4`，30 Hz，active-only，state perturbation，50 epochs，从头训练；frozen OICM 保持 eval/no-grad。

**运行**

- command: `CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --distributed`
- device: physical GPU `0,2,3`；world size `3`；per-device batch `8`；global batch `24`；seed `42`。
- total: `72400` steps / `50` epochs / approximately `03:52:52`。

**结果**

- 首个 validation：`val/loss=0.0093770677`，point-flow EPE=`26.4114 mm`。
- checkpoint selection best：epoch `5` / step `7240`，`val/loss=0.0039939559`，point-flow EPE=`12.8782 mm`，h1=`7.1633 mm`。
- total point-flow EPE 最好：epoch `27` / step `39096`，`12.6071 mm`；这不是当前 best checkpoint 的选择指标。
- final epoch `50` / step `72400`：`val/loss=0.0041478906`，point-flow EPE=`12.7156 mm`，h1=`9.4626 mm`，wrist translation=`11.8189 mm`，wrist rotation=`6.1306 deg`。
- train final epoch：loss=`0.0019386694`，point-flow EPE=`7.6044 mm`。

**结论边界**

- `SUPPORTED`（工程协议）：V1.3 frozen OICM、unique KNN edge input、dynamic padding、完整 `10135` 点 point-flow target、frozen OICM no-grad 和 50-epoch train/val 均已正常完成；`metrics.jsonl` 共记录 50 个 train epoch 和 50 个 val epoch，全部数值 finite。
- `INCONCLUSIVE`（科研效果）：validation 在 epoch 5 取得 checkpoint selection 的最低 loss，后续在约 `0.00415` 附近平台化；该 run 只包含 Inspire RL train/val，没有 MANO→Inspire 定量 test，因此不能仅凭这次训练宣称跨 embodiment decoder 效果成立。

**证据**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/)
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/run_manifest.json)
- [config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/config.json)
- [metadata.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metadata.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/latest.pt)
- [最终计划](../plan/v1.1.md)
## 2026-09-08 — teacher-forced effect 误差归因诊断

- experiment_id: `cmdecoderv2-teacherforced-effect-attribution-v1.1.4`
- activity_id: [`cmdecoderv2-teacherforced-effect-attribution-20260908-202626`](activity_log.md)
- run_id: `cmdecoderv2-teacherforced-effect-attribution-20260908-202626`
- run_status: `COMPLETED`
- modification_version: `V1.1.4`
- operation_category: `diagnostic / operation`
- sequence/checkpoint: `s1/mouse_lift`，修正数据 decoder best + 冻结 OICM V1.2.5；逐帧只读复核，无新持久化输出。

**问题定义**

viewer 的 `教师强制` 是“GT 当前 Inspire state → decoder 预测下一 state → 预测 hand flow → OICM”，并非 GT hand flow 直通 OICM。需要用 GT hand flow→OICM 对照拆分 decoder 与 OICM 误差。

**结果**

- 近物体且 OICM 有效的 `275` 帧中，GT object effect `<=0.1 mm` 的 `126` 帧：teacher-forced 预测 effect RMS 均值/中位数/最大值 `7.317/7.293/10.120 mm`，GT effect RMS 均值 `0.048 mm`，prediction-GT EPE `7.286 mm`。
- 这 `126` 帧的 teacher-forced decoder hand-flow EPE 均值 `3.850 mm`，GT hand-flow RMS 均值 `1.179 mm`。
- 同一 `126` 帧将真实 GT hand flow 直接送入同一 OICM：预测 effect RMS 均值/中位数/最大值 `0.706/0.688/1.786 mm`，prediction-GT EPE `0.660 mm`。

**结论边界**

- `SUPPORTED`：6–7 mm 现象主要由 decoder 单步 hand-state/hand-flow 误差引起；GT hand flow 下 OICM 没有同量级固有偏置。
- `INCONCLUSIVE`：这不能单独证明 decoder 的所有 hand-state 误差来源，也不能外推到所有序列；teacher-forced 只排除了递归漂移。

**证据入口**

- [活动记录](activity_log.md)
- [完整 effect 运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)

## 2026-09-08 — 修正数据 best.pt 的中文 Inspire effect viewer（完整重跑）

- experiment_id: `cmdecoderv2-inspire-rollout-effect-viewer-v1.2.5`
- activity_id: [`cmdecoderv2-inspire-effect-v1.2.5-best-20260908-194900`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-20260908-195042`
- run_status: `RUNNING`（368 帧 effect 已计算；Viser `8104` 保持运行）
- modification_version: `V1.1.4`
- operation_category: `code / diagnostic / experiment / operation / documentation`
- decoder/OICM: 修正数据 decoder best epoch 24 / step 36120 与冻结 OICM V1.2.5 best；checkpoint SHA256 见 activity。
- sequence: `s1/mouse_lift`，368 个递归 rollout transition；GT object flow 只在当前帧独立计算，display-only。
- controls: 中文右侧控件支持任意合法帧 handoff、`教师强制`、`递归Rollout`、播放/停止、跳帧、`预测+GT`/`仅预测`/`仅GT` 和 raw/effective prediction 切换。

**结果**

- 远于 `50 mm`：12 帧全部 `sample_valid=false`，effective effect RMS `0 mm`。
- 近于或等于 `50 mm`：356 帧全部有效，effective effect RMS 均值 `4.528 mm`，中位数 `1.647 mm`，最大 `52.341 mm`。
- 全局 raw/effective effect RMS 均值 `4.998/4.380 mm`；GT effect RMS 均值 `2.693 mm`；预测-GT EPE 均值 `2.512 mm`；OICM valid ratio `0.9674`。

**结论边界**

- `SUPPORTED` 子结论：该 held-out 序列远距离 hand flow 仍被 OICM validity gate 屏蔽为零有效 object effect。
- `INCONCLUSIVE` 总结：单序列可视化和 display-only EPE 不能证明 decoder 整体动作质量或 MANO→Inspire 泛化。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect_summary.json)

## 2026-09-08 — 修正数据 best.pt 的 Inspire rollout self-effect 诊断

- experiment_id: `cmdecoderv2-inspire-rollout-effect-oicm-v1.2.5-best`
- activity_id: [`cmdecoderv2-inspire-effect-v1.2.5-best-20260908-125300`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-20260908-151014`
- run_status: `STOPPED`（368-step effect 已完成；8104 Viser 后被会话超时终止）
- modification_version: `V1.1.4`（复用既有诊断协议）
- operation_category: `diagnostic / experiment / operation`
- decoder: 修正数据正式 run 的 epoch 24 / step 36120 `best.pt`，SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`
- OICM: V1.2.5 `best.pt`，SHA256 `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`
- sequence: `s1/mouse_lift`，368 个纯 Inspire recursive rollout transition
- contract: 每步 rollout hand point flow 重新 forward OICM，保存 raw/effective `pred_obj_flow`；GT object flow 仅 display-only，不进入模型

**结果**

- 距离 `>50 mm`：12/12 帧无效，effective object-flow RMS 均值和最大值均为 `0 mm`。
- 距离 `<=50 mm`：356/356 帧有效，effective object-flow RMS 均值 `4.528 mm`、中位数 `1.647 mm`、最大值 `52.341 mm`。
- 全局 OICM valid ratio `0.9674`，effective effect RMS 均值 `4.380 mm`。

**结论边界**

- `SUPPORTED`：本 held-out 序列中，修正数据 decoder rollout 的远距离 Inspire motion 仍被 OICM 5 cm validity gate 正确屏蔽为零有效 object effect。
- `INCONCLUSIVE`：单序列诊断不能证明 decoder 整体动作或 MANO→Inspire 泛化已经正确；无效帧的 raw dummy head 输出也没有物理语义。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect_summary.json)

## 2026-09-08 — 修正 DExplore RL 数据上的 Temporal-D2 正式训练

- experiment_id: `cmdecoderv2-temporal-d2-oicm-v1.2.5`
- activity_id: [`cmdecoderv2-decoder-v1.2.5-20260908-101402`](activity_log.md)
- run_id: `cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402`
- run_status: `COMPLETED`
- modification_version: `V1.1.6`
- operation_category: `data / experiment / operation`
- input: 修正 DExplore RL 物体轨迹 decoder view + OICM V1.2.5 `best.pt`；decoder 从头训练

**结果**

- 50 epochs / `75250` steps，耗时 `02:32:18`。
- best epoch 24 / step 36120：`val/loss=0.0040070958`、`val/hand/point_flow_epe_mm=12.4907`、h1=`9.0049 mm`。
- final epoch 50：`val/loss=0.0041402271`、`val/hand/point_flow_epe_mm=12.5936`。

**结论边界**

- `SUPPORTED`：冻结 OICM V1.2.5 的正式 decoder 训练、验证和 best checkpoint 选择完整完成。
- `INCONCLUSIVE`：RL-Inspire validation 收敛不等于 MANO→Inspire recursive rollout 的跨 embodiment 效果成立，需单独可视化/诊断。

**证据**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt)

## 2026-09-10 — V1.3 中间 best.pt MANO/Inspire Cm 来源分类诊断

- experiment_id: `cmdecoderv2-cm-source-classifier-v13-v1.1.6`
- activity_id: [`cmdecoderv2-cm-source-classifier-v13-20260910-110919`](activity_log.md)
- run_id: `cmdecoderv2-cm-source-classifier-20260910-111037`
- run_status: `COMPLETED`
- modification_version: `V1.1.6`（分类诊断 Task）；上游冻结 OICM 为 `V1.3`
- operation_category: `diagnostic / experiment / operation`
- hypothesis: 若 V1.3 Cm 保留明显的手来源/embodiment 信息，只用 `cm_tokens` pooling 的小分类器也能在未见 sequence 上区分 MANO 与 Inspire。
- source: 冻结 OICM V1.3 训练过程中的中间 `best.pt`；标签 `mano=0`、`inspire_rl=1`。
- split: 使用 V1.3 现有 train sequence 训练分类器，val sequence 作为 `val-held-out`；正式 test 不使用；不做 frame-level train/test mixing。
- feature: 主输入为 V1.3 `unique_knn_edges` 路径输出的 `cm_tokens` permutation-invariant mean/max/std pooling；anchor pooling 为 object-side control。
- input: V1.3 cache 的 `K=32`、2 cm 半径、MANO `2048` 点、Inspire-F1 `10135` 点；有效 KNN 边去重后按 batch max 动态 padding。

**结果**

- 共同 object 类别 `11` 个；train `60 MANO + 60 Inspire` sequences，val-held-out `11 MANO + 11 Inspire` sequences；每条 sequence 抽取 `8` 个 transition，共 `1136` 个样本。
- 全部样本（train `960`，val-held-out `176`，两类各 `88`）：Cm val-held-out accuracy/balanced accuracy `57.95%`，F1 `0.4714`，AUROC `0.6475`；anchor control accuracy `57.39%`，AUROC `0.6058`。
- `sample_valid=true` 子集（train 可用 `443`、val-held-out 可用 `90`；平衡后训练 `208+208`、held-out `44+44`）：Cm val-held-out accuracy/balanced accuracy `73.86%`，F1 `0.7473`，AUROC `0.7531`；confusion matrix `[[31, 13], [10, 34]]`。
- 同一有效子集的 anchor control accuracy `62.50%`、AUROC `0.6689`，Cm 高于 anchor control。

**结论边界**

- `SUPPORTED`：在当前中间 OICM V1.3 checkpoint 和 `val-held-out` 口径下，Cm 保留了弱到中等的可识别 MANO/Inspire source/embodiment 信号。
- `INCONCLUSIVE`：该结果弱于旧 V1.2.1 分类实验，且不能单独证明是纯静态手型信息；运动幅度、接触状态和 source-domain 差异仍可能造成可分性。

**证据**

- [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/)
- [run manifest](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/run_manifest.json)
- [features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/features.npz)
- [metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/metrics.json)
- 中间 OICM checkpoint SHA256：`3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。

## 2026-09-07 — V1.1.6 MANO/Inspire Cm 来源分类诊断

- experiment_id: `cmdecoderv2-cm-source-classifier-v1.1.6`
- activity_id: [`cmdecoderv2-cm-source-classifier-final-20260907-214000`](activity_log.md)
- run_id: `cmdecoderv2-cm-source-classifier-20260907-213748`
- run_status: `COMPLETED`
- modification_version: `V1.1.6`
- operation_category: `diagnostic / experiment / operation`
- hypothesis: 若 Cm 保留明显的手来源/embodiment 信息，只用 Cm token pooling 的小分类器也能在 held-out sequence 上区分 MANO 与 Inspire。
- source: 冻结 ObjectInteractionCm V1.2.1 best checkpoint；标签 `mano=0`、`inspire_rl=1`。
- split: 使用现有 ObjectInteractionCm train/val sequence split；每个 split 按共同 object 类别平衡两种 source 的 sequence 数；不做 frame-level train/test mixing。
- feature: 主输入为 `cm_tokens` 的 permutation-invariant mean/max/std pooling；anchor pooling 为 object-side control。

**结果**

- 全部样本：Cm test accuracy `74.48%`、balanced accuracy `74.48%`、AUROC `0.8467`；anchor control AUROC `0.6263`。
- 只保留 `sample_valid=true` 的有效 Cm：Cm test accuracy `97.52%`、balanced accuracy `97.52%`、F1 `0.9752`、AUROC `0.9958`；anchor control accuracy `62.81%`、AUROC `0.6702`。
- 有效 Cm 测试 confusion matrix 为 `[[118, 3], [3, 118]]`，两类各 121 个平衡样本。

**结论边界**

- `SUPPORTED`：当前 Cm 包含明显可识别的 MANO/Inspire source/embodiment signal，且该信号在 object anchor control 之上仍然很强。
- `INCONCLUSIVE`：该实验不能单独证明 Cm 编码的是纯静态手型；运动幅度、接触状态和 source-domain 统计也可能贡献分类结果。

**证据**

- [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/)
- [run_manifest.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/run_manifest.json)
- [features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/features.npz)
- [metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/metrics.json)

## 2026-09-07 — V1.1.5 MANO rollout self-effect 对照诊断

- experiment_id: `cmdecoderv2-mano-rollout-effect-v1.1.5`
- activity_id: [`cmdecoderv2-mano-effect-full-20260907-203300`](activity_log.md)
- run_id: `cmdecoderv2-mano-effect-20260907-203231`
- run_status: `RUNNING`（348 步 effect 计算已完成；8103 Viser 仍运行）
- modification_version: `V1.1.5`
- operation_category: `diagnostic / experiment / operation`
- checkpoint: CmDecoderv2 best epoch 3 / step 10296；冻结当前锁定的 OICM `best.pt`
- sequence: `s1/camera_takepicture_3_Retake`；正式 MANO qualitative-only source，352 帧，348 个 rollout transition
- rollout_contract: handoff frame 0 使用 `q=0 + MANO wrist` 初始化 Inspire，之后递归反馈 decoder 预测 Inspire state
- input_contract: 当前 GRAB/MANO object geometry/pose、当前 MANO-source Cm window、当前 rollout Inspire geometry；不读取 paired Inspire GT、未来 object pose/flow 或 GT object flow
- output_contract: 保存 raw `pred_obj_flow` 与 `sample_valid` 屏蔽后的 effective effect；无效 sample 的 raw 输出仅作 dummy computational path 审计

**结果**

- 距离 `>50 mm`：258 帧全部 `sample_valid=false`，effective effect 为 0；raw effect RMS 均值 `12.535 mm`。
- 距离 `<=50 mm`：90 帧中 89 帧有效；effective effect RMS 均值 `9.194 mm`，最大 `41.302 mm`。
- 唯一无效的近距离帧为 frame `258`，距离约 `47.42 mm`、sampled active count `0`，符合随机采样未命中 candidate 的情况。

**结论边界**

- `SUPPORTED`：MANO-source rollout 也呈现和纯 Inspire rollout 一致的 OICM 5 cm validity gate；远距离 rollout hand flow 没有合同有效的 object effect。
- `INCONCLUSIVE`：MANO→Inspire decoder 的整体动作是否正确、以及远处动作是否由 decoder 产生了错误状态，仍需通过手部 rollout 几何和其它指标单独判断。

**证据**

- [运行目录](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/)
- [run_manifest.json](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/run_manifest.json)
- [effect.npz](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect.npz)
- [effect_summary.json](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect_summary.json)

## 2026-09-07 — V1.1.4 rollout self-effect OICM 旁路诊断

- experiment_id: `cmdecoderv2-inspire-rollout-effect-v1.1.4`
- activity_id: [`cmdecoderv2-inspire-effect-full-20260907-171315`](activity_log.md)
- run_id: `cmdecoderv2-inspire-effect-20260907-171349`
- run_status: `RUNNING`（348 步 effect 计算已完成；8102 Viser 仍运行）
- modification_version: `V1.1.4`
- operation_category: `diagnostic / experiment / operation`
- checkpoint: CmDecoderv2 best epoch 3 / step 10296；冻结当前锁定的 OICM `best.pt`
- sequence: `s1/camera_takepicture_3_Retake`；352 帧，348 个 `state_t -> state_{t+1}` transition
- input_contract: 当前 object geometry/pose、当前 rollout Inspire geometry、当前 rollout hand point flow；不读取未来 object pose/flow，不使用 GT object flow
- output_contract: 保存 raw `pred_obj_flow` 与 `sample_valid` 屏蔽后的 effective effect；无效 sample 的 raw 输出仅作 dummy computational path 审计

**结果**

- 距离 `>50 mm`：122 帧全部 `sample_valid=false`，effective effect 为 0；raw effect RMS 均值 `12.533 mm` 不具物理语义。
- 距离 `<=50 mm`：226 帧中 225 帧有效；effective effect RMS 均值 `7.959 mm`，最大 `39.277 mm`。
- 唯一无效的近距离帧为 frame `127`，最近距离 `48.29 mm`、sampled active count `0`，符合 1024 点随机采样未命中 candidate 的情况。

**结论边界**

- `SUPPORTED`：本序列中 OICM 的 5 cm validity gate 能把远离物体的 rollout hand flow 屏蔽为无有效 object effect。
- `INCONCLUSIVE`：该旁路诊断不能单独判断 decoder 是否已经学会正确的远距离动作，或 MANO→Inspire rollout 的整体动作质量。

**证据**

- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/)
- [run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect_summary.json)

## 2026-09-06 — V1.1.1 Temporal-D2 工程 smoke

- experiment_id: `cmdecoderv2-temporal-d2-smoke-v1.1.1`
- activity_id: [`cmdecoderv2-v1.1.1-20260906-102433-implementation`](activity_log.md)
- hypothesis: 冻结的 Dexplore OICM 能以 `K=4` window 接入 Temporal-D2，并只对新 decoder 反向传播。
- dataset: pilot view；RL-Inspire train/val 各 1 sequence，MANO test 1 sequence 不参与训练或定量评估。
- run: `cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619`（最终 smoke；此前 smoke
  `..._102937` / `..._104051` / `..._104215` / `..._104811` 用于逐项补齐 OICM SHA256、cache manifest、
  perturbation provenance 和 translation loss 语义，均不作为
  provenance，均不作为
  最终证据入口）
- result: 两个 optimizer step 与完整 pilot val 完成；OICM 保持 `eval/no-grad`，decoder head/core 可反向；
  `cm_sample_valid` 和 per-horizon q/wrist 指标正常写入。
- conclusion: `SUPPORTED`（只支持工程 wiring；两步 smoke 不支持收敛、泛化或跨 embodiment 效果结论）。
- evidence: [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/)、
  [run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/metrics.jsonl)、
  [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/train.log)、
  [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/best.pt)。
