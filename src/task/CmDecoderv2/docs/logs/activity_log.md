# CmDecoderv2 活动记录

- scope: `src/task/CmDecoderv2/`
- last_updated: 2026-09-10
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)

## 2026-09-10 17:48:24 +0800 — V1.1.7 纯 Inspire rollout/effect 诊断接入 V1.3 KNN

- activity_id: `cmdecoderv2-inspire-effect-v13-impl-20260910-174824`
- timestamp: `2026-09-10 17:48:24 +0800`
- modification_version: `V1.1.7`
- type: `code / diagnostic / documentation / operation`
- change_level: `L2`（Task-local 诊断输入合同从旧 full-hand stream 适配到 V1.3 `unique_knn_edges`、10135 点、K=32 和 2 cm validity；不改变正式 cache/schema、split、模型或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认按 `s1/mouse_lift`、V1.3 decoder `best.pt`、冻结 OICM V1.3、逐步 `t -> t+1` display-only object flow 对照和临时 Inspire KNN 执行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `59f1d809a910041f888e68565f6e3c6bc9dac229`
- worktree_dirty: `true`（本条记录随实现一同提交前）
- run_id: `cmdecoderv2-inspire-effect-v13-smoke-20260910-174418`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅支持 V1.3 诊断 wiring 和 2-step smoke；完整 368-step 科研结果待后续独立 run）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/`、对应 Task test 和 [V1.1 plan](../plan/v1.1.md) 第 15 节；正式 V1.3 cache、split、训练配置、decoder/OICM checkpoint 和 `src/base/` 均未修改。

**文件**

- [plan/v1.1.md](../plan/v1.1.md) — 追加用户批准的 V1.3 纯 Inspire rollout/effect 诊断边界：单序列 `s1/mouse_lift`、368 个逐步 transition、临时 source KNN、真实 object flow 只作 display-only。
- [research/inspire_rollout_effect/run.py](../../research/inspire_rollout_effect/run.py) — 对 V1.3 `unique_knn_edges` 添加临时 GT Inspire source KNN、V1.3 surface sampling、compact unique hand stream、per-step predicted-hand KNN effect path、20 mm 动态分桶和 rollout hand/flow EPE。
- [research/inspire_rollout_effect/README.md](../../research/inspire_rollout_effect/README.md) — 更新 V1.3 运行命令、10135/K=32/2 cm 合同和临时 KNN 产物说明。
- [research/inspire_rollout_effect/experiment.yaml](../../research/inspire_rollout_effect/experiment.yaml) — 将实验元数据更新到 V1.1.7/V1.3 诊断口径。
- [tests/test_inspire_effect.py](../../tests/test_inspire_effect.py) — 增加 20 mm summary 分桶和 compact KNN edge stream 回归。

**原因**

V1.3 decoder/OICM 已改为 `unique_knn_edges`，旧诊断脚本会用 full hand stream 或误用 MANO test KNN，
不能无歧义评估纯 Inspire `10135` 点 rollout effect。需要为诊断单独生成临时 Inspire KNN，并保持真实
object flow 只作为 display-only 对照。

**验证**

- py_compile: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/research/inspire_rollout_effect/run.py`，通过。
- Task tests: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`，`18 passed`。
- `git diff --check`：通过。
- smoke command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v13-smoke-20260910-174418 --run-id cmdecoderv2-inspire-effect-v13-smoke-20260910-174418 --max-steps 2 --knn-batch-size 4`。
- smoke evidence: [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/)、[run_manifest.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/run_manifest.json)、[effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/effect.npz)、[effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/effect_summary.json)、[source_knn_indices.npy](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-v13-smoke-20260910-174418/source_knn_indices.npy)。
- smoke result: 2 个 transition 完成；`distance_threshold_mm=20.0`，2/2 为远距离无效 sample，effective object-flow RMS 为 `0 mm`；真实 object flow 为 display-only，`overall_pred_gt_effect_epe_mm=1.58898`。

**保护边界与回滚**

- 未复用 MANO test cache 的 `obj_knn_indices` 作为纯 Inspire KNN；未写入 `data/processed_data/`，未修改正式 V1.3 decoder view、OICM cache/checkpoint、训练配置、正式 split 或旧输出。
- 回滚入口：移除本次 Task-local 诊断代码/文档改动，隔离 smoke 输出目录；不删除原始数据、正式 cache 或 checkpoint。

## 2026-09-10 16:35:24 +0800 — V1.1.7 Inspire 全点监督 decoder 正式训练完成

- activity_id: `cmdecoderv2-v1.1.7-formal-20260910-163524`
- timestamp: `2026-09-10 16:35:24 +0800`
- modification_version: `V1.1.7`
- type: `experiment / operation`
- change_level: `L3 + L2`（按已定稿 [V1.1 plan](../plan/v1.1.md) 执行 V1.3 OICM、KNN hand stream、GT/point-flow 点数和三卡正式长训练）
- approval: `user-approved`
- approval_basis: 用户确认使用最终 OICM V1.3 `best.pt`，Inspire 使用完整 `10135` 点监督，并回复“是的，就按你的来”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `cb4161cab47678a46b69ec440d3dd06334e5645f`
- worktree_dirty: `false`（正式 run 启动和完成时；本条为终态补记）
- run_id: `cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812`
- run_status: `COMPLETED`
- last_step: `72400`
- last_epoch: `50`
- best_metric: `val/loss=0.003993955866854461`（epoch 5 / step 7240）
- conclusion: `INCONCLUSIVE`（工程训练协议 `SUPPORTED`；validation 在早期达到最好，且该 run 不包含 MANO→Inspire 定量 test，因此不单独宣称跨 embodiment 效果成立）
- scope: `src/task/CmDecoderv2/` 的 V1.3 decoder view、冻结 OICM 接入、unique KNN edge stream、动态 hand padding、Inspire `10135` 点 point-flow supervision 和正式训练输出；不改旧 cache、checkpoint、split 或 `src/base/`

**固定输入与运行**

- frozen OICM checkpoint: [V1.3 best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)，SHA256 `3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`。
- decoder view: [dexplore_rl_v1_3_full10135](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/)，`K=32`、2 cm 仅用于 edge validity/active-only、运行时重算 32 条边距离、全局 hand id 去重、batch-max 动态 padding、point-flow target 全部 `10135` 点。
- view evidence: [view manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/manifest.json)、[view run manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_3_full10135/run_manifest.json)；train/val/test sequence=`255/30/63`，view train/val windows=`63988/7807`，启用 `active_only` 后实际 train/val windows=`34746/4254`，test 仍为 MANO qualitative-only。
- command: `CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_3_full10135.yaml --distributed`
- device: physical GPU `0,2,3`；per-device batch `8`；global batch `24`；seed `42`；decoder 从头训练；50 epochs / `72400` steps；耗时约 `03:52:52`。

**结果与证据**

- validation 首个 epoch：`val/loss=0.0093770677`，point-flow EPE=`26.4114 mm`。
- best `val/loss`：epoch `5` / step `7240`，`0.0039939559`；point-flow EPE=`12.8782 mm`，h1=`7.1633 mm`；[best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt)。
- best total point-flow EPE：epoch `27` / step `39096`，`12.6071 mm`；该指标不是当前 checkpoint 选择指标。
- final epoch `50` / step `72400`：`val/loss=0.0041478906`，point-flow EPE=`12.7156 mm`，h1=`9.4626 mm`，wrist translation=`11.8189 mm`，wrist rotation=`6.1306 deg`；[latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/latest.pt)。
- [正式运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/)、[config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/config.json)、[metadata.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metadata.json)、[run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/run_manifest.json)、[metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/metrics.jsonl)、[train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/train.log)。

**原因**

用户已批准在最终 OICM V1.3 上重新从头训练 decoder；本次终态核对用于确认完整 `10135` 点 point-flow supervision、V1.3 unique-KNN 输入和三卡长任务均按最终 plan 执行，并确定可供后续诊断使用的 best checkpoint。

**验证**

- `py_compile`、`pytest -q src/task/CmDecoderv2/tests`：实现前已通过，Task tests=`15 passed`。
- 正式 `metrics.jsonl`：`824` 条记录，train/val epoch 各 `50`；最终 step/epoch=`72400/50`；所有数值 finite，未发现 `Traceback`、`NaN`、`Inf`、`OOM` 或数据加载错误。
- `torch.load` 核对：`best.pt` 对应 epoch `5` / step `7240`，`latest.pt` 对应 epoch `50` / step `72400`；best metric 与 metrics 一致。
- 训练终止后 torchrun 和 3 个 worker 均已退出；其他 GPU 上的既有任务未停止。
- [V1.1.7 最终计划](../plan/v1.1.md) 与当前配置、view manifest、metadata 的 KNN/10135-point contract 一致。

**保护边界与回滚**

- 未修改既有 V1.2.5 decoder view、V1.3 OICM cache/checkpoint、原始数据、正式 split、旧 decoder output、`src/base/` 或其他 GPU 任务。
- 回滚入口：隔离本次 V1.1.7 的 Task-local 代码/config、`dexplore_rl_v1_3_full10135` view 和本 run output；不删除旧 cache、checkpoint 或运行。

**规范反馈**

- 本次没有审批或目录阻碍。当前 BaseRunner 的 `run_manifest.json` 记录启动时 provenance，训练终态按 Skill 规范登记在本条 activity；manifest 中通用路径扫描器会把 OICM SHA256 字符串额外显示为一个 `missing` 路径，但实际 checkpoint 路径、大小和 SHA256 均已在 metadata/activity 中核验，不影响本次 run 可复现性。

## 2026-09-10 11:09:19 +0800 — V1.3 中间 best.pt 的 MANO/Inspire Cm 来源分类诊断完成

- activity_id: `cmdecoderv2-cm-source-classifier-v13-20260910-110919`
- timestamp: `2026-09-10 11:09:19 +0800`
- modification_version: `V1.1.6`（分类诊断 Task）；上游冻结 OICM 为 `V1.3`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local 分类诊断输入适配；不改变 OICM、cache、正式 split 或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户确认使用 V1.3 训练中的中间 `best.pt`，按 train 训练、val-held-out 评估，并只修改分类诊断脚本输入适配。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d5711026a6bd6687a8e7835785108034f21389a9`
- worktree_dirty: `true`（本条启动记录尚未提交）
- run_id: `cmdecoderv2-cm-source-classifier-20260910-111037`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（Cm 来源信号子结论为 `SUPPORTED`；纯静态手型归因仍不充分）
- scope: `src/task/CmDecoderv2/research/cm_hand_source_classifier/run.py`；使用 V1.3 unique-KNN cache 和当前中间 OICM `best.pt`，保持旧分类协议

**文件**

- `src/task/CmDecoderv2/research/cm_hand_source_classifier/run.py` — 接受 V1.3 index/cache，按有效 `K=32` 边去重构造动态 hand stream，并将 held-out 结果明确标记为 `val_held_out`。
- `src/task/CmDecoderv2/docs/logs/experiment_log.md` — 记录本次 V1.3 中间 checkpoint 分类结果、证据和结论边界。

**原因**

复现实验“只用冻结 Cm 区分 MANO/Inspire”，检验 V1.3 Cm 是否仍包含可识别的手来源/embodiment 信息；主特征为 `cm_tokens` mean/max/std pooling，anchor pooling 继续作为 object-side control。

**运行**

- command: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_hand_source_classifier.run --index data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json --oicm-config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml --oicm-checkpoint outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt --device cuda:0 --activity-id cmdecoderv2-cm-source-classifier-v13-20260910-110919 --frames-per-sequence 8 --batch-size 8 --epochs 40`
- device: physical GPU `1`（进程内 `cuda:0`）；当前 OICM V1.3 训练继续使用物理 GPU `0,2,3`
- split: ObjectInteractionCm `train` 训练分类器，`val` 作为 `val-held-out`；正式 `test` 不使用
- output: [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/)
- completed_at: `2026-09-10 11:10:39 +0800`

**验证**

- 代码提交前已通过 `py_compile`、`git diff --check` 和单序列 CUDA smoke；完整运行 `COMPLETED`。
- 共同 object 类别 `11` 个；train `60 MANO + 60 Inspire`，val-held-out `11 MANO + 11 Inspire`；抽取 `1136` 个 transition。
- 全部样本 Cm：val-held-out accuracy/balanced accuracy `0.5795`，F1 `0.4714`，AUROC `0.6475`；anchor control AUROC `0.6058`。
- `sample_valid=true`：Cm val-held-out accuracy/balanced accuracy `0.7386`，F1 `0.7473`，AUROC `0.7531`；anchor control accuracy `0.6250`，AUROC `0.6689`。
- 证据：[metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/metrics.json)、[features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/features.npz)、[run_manifest.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260910-111037/run_manifest.json)。

**保护边界与解释限制**

- 使用的中间 OICM `best.pt` SHA256：`3a3d6c0f88565b9e41f257e4f8b87a3ca5731fd356a98f4514091754036a7283`；未使用随后完成训练产生的最终状态重新运行。
- 未修改 V1.3 cache、OICM 训练配置/模型、正式 train/val/test split 或 checkpoint；只提交了分类诊断脚本适配。
- 结果支持 Cm 仍有弱到中等的可识别 MANO/Inspire source/embodiment 信号，但不能单独证明 Cm 编码纯静态手型；运动、接触状态和 source-domain 统计仍可能贡献分类结果。

## 2026-09-08 20:26:26 +0800 — V1.1.4 teacher-forced effect 误差归因诊断

- activity_id: `cmdecoderv2-teacherforced-effect-attribution-20260908-202626`
- timestamp: `2026-09-08 20:26:26 +0800`
- modification_version: `V1.1.4`
- type: `diagnostic / operation`
- change_level: `L0`（只读逐帧诊断；不修改代码、checkpoint、cache、split 或 viewer 运行）
- approval: `user-approved`
- approval_basis: 用户询问 teacher-forced 下 GT effect 近零而预测 effect 约 6 mm 是否合理。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动）
- run_id: `cmdecoderv2-teacherforced-effect-attribution-20260908-202626`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（6 mm 现象主要由 teacher-forced decoder 单步手流误差解释；OICM GT-hand-flow 对照不支持 OICM 固有 6 mm 偏置）
- scope: `s1/mouse_lift`、修正数据 decoder best、冻结 OICM；无持久化新输出目录，证据记录如下。

**原因**

确认 viewer 的“教师强制”语义：当前 Inspire state 使用 GT，但下一状态仍由 decoder 预测；它不是把 GT hand flow 原样送入 OICM。因此需要把 decoder teacher-forced 路径与 GT hand-flow→OICM 路径分开比较。

**文件**

- `src/task/CmDecoderv2/docs/logs/activity_log.md` — 记录 teacher-forced 与 GT hand-flow→OICM 归因证据。
- `src/task/CmDecoderv2/docs/logs/experiment_log.md` — 追加本次诊断结果和结论边界。

**运行与证据**

- 使用 `EffectDiagnostic.teacher_record` 对 `s1/mouse_lift` 的 `367` 个 transition 逐帧 forward；另用同一 decoder checkpoint 中冻结的 OICM，将真实 GT Inspire hand flow 直接 forward，未写入文件。
- teacher-forced 近物体（距离 `<=50 mm`）`275` 帧全部有效；其中 GT object effect `<=0.1 mm` 的 `126` 帧：GT effect RMS 均值 `0.048 mm`，预测 effect RMS 均值/中位数/最大值 `7.317/7.293/10.120 mm`，prediction-GT EPE 均值 `7.286 mm`。
- 同一 `126` 帧中，teacher-forced decoder hand-flow EPE 均值 `3.850 mm`；GT hand-flow RMS 均值 `1.179 mm`。
- 严格 GT hand-flow→OICM 对照的同一 `126` 帧：预测 object effect RMS 均值/中位数/最大值 `0.706/0.688/1.786 mm`，prediction-GT EPE 均值 `0.660 mm`。
- 远距离帧的 raw 输出仍是无效 sample 的 dummy head path；不用于物理解释。
- [完整 viewer 运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)，其中的 GT/pred effect 字段仍仅作显示和诊断。

**验证**

- 结论：GT object effect 为零而 teacher-forced prediction 约 `6–7 mm` 在当前实现和当前 checkpoint 下是可复现的，但不是“正确的物理预测”；它是 decoder 单步 hand-state/hand-flow 误差经接触敏感 OICM 放大的结果。
- “教师强制”并未消除单步 decoder 误差；它只消除了前序 rollout 状态漂移。若要单独测 OICM，应使用 GT hand-flow→OICM 对照。
- 现有测试仍为 `13 passed`，8104 viewer 保持 HTTP `200`；本次未改变任何运行代码或研究变量。

**保护边界**

- 未修改 decoder/OICM checkpoint、数据、cache、split、配置、模型代码或运行中的 8104/8102/8103 viewer。

## 2026-09-08 19:53:57 +0800 — V1.1.4 中文 Inspire effect viewer 与修正数据完整重跑

- activity_id: `cmdecoderv2-inspire-effect-v1.2.5-best-20260908-194900`
- timestamp: `2026-09-08 19:53:57 +0800`
- modification_version: `V1.1.4`
- type: `code / diagnostic / experiment / operation / documentation`
- change_level: `L1`（Task-local viewer 交互和 display-only GT 诊断字段；不改变 decoder/OICM、训练变量或正式 split）
- approval: `user-approved`
- approval_basis: 用户要求右侧注释中文化、允许任意帧启动递归 rollout、可切换 teacher-forcing/rollout，并同时显示预测 effect 与 GT effect。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动；未覆盖其他 Task 或旧 viewer）
- run_id: `cmdecoderv2-inspire-effect-20260908-195042`
- run_status: `RUNNING`（effect 计算完成；中文 Viser 8104 保持运行）
- conclusion: `INCONCLUSIVE`（支持 OICM 远距离 validity gate 子结论；单序列和跨 embodiment rollout 整体质量仍不能由此判定）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/` 及其 Task 测试/说明

**原因**

将上次会话未完成的交互需求收口：右侧说明中文化，允许从任意合法帧重新 handoff，支持教师强制与递归
Rollout 随时切换，并把真实 object flow 作为不进入模型的 GT effect 对照显示。

**文件**

- `src/task/CmDecoderv2/research/inspire_rollout_effect/run.py` — 增加中文交互 viewer、任意帧 handoff、教师强制/递归 Rollout 切换、预测/GT effect 显示；GT flow 仅 display-only；Ctrl-C 收口 manifest。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/README.md` — 更新中文控件、运行入口和 GT display-only 合同。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/experiment.yaml` — 明确 future object flow 仅允许 display-only，禁止作为模型输入。
- `src/task/CmDecoderv2/tests/test_inspire_effect.py` — 增加可选 GT 摘要字段回归。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/run_manifest.json` — 将被超时终止的旧 viewer 如实标为 `STOPPED`。

**运行与产物**

- full command: `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v1.2.5-best-20260908-194900 --port 8104 --fps 8 --serve`
- smoke command: 同一 checkpoint/config 加 `--max-steps 2`，run `cmdecoderv2-inspire-effect-20260908-194949`，`COMPLETED`。
- decoder checkpoint: epoch `24` / step `36120`；SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`。
- frozen OICM checkpoint SHA256: `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。
- sequence: `s1/mouse_lift`，368 个 recursive rollout transition；GT object flow 仅用于显示和离线 EPE，不进入模型。
- [完整运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/)
- [run manifest](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-195042/effect_summary.json)
- Viser: `http://localhost:8104`，HTTP `200`；进程保持运行。

**验证**

- `py_compile`（run.py、visualize_inspire_test.py）：通过；`pytest -q src/task/CmDecoderv2/tests`：`13 passed`。
- 2-step smoke 已生成 `gt_obj_flow_object/world`、`gt_effect_rms_mm` 和 `pred_gt_effect_epe_mm`，验证 display-only 字段合同。
- 完整结果：远于 `50 mm` 的 12 帧全部 `sample_valid=false`，effective effect RMS 均值/最大值 `0 mm`；近距离 356 帧全部有效，effective effect RMS 均值 `4.528 mm`、中位数 `1.647 mm`、最大 `52.341 mm`。
- 全局 raw/effective effect RMS 均值 `4.998/4.380 mm`，GT effect RMS 均值 `2.693 mm`，预测-GT EPE 均值 `2.512 mm`，OICM valid ratio `0.9674`。
- 子结论 `SUPPORTED`：该 held-out 序列中远距离 hand flow 仍由 OICM validity gate 屏蔽；总体 decoder rollout/跨 embodiment 结论保持 `INCONCLUSIVE`。

**保护边界与旧运行收口**

- 未读取 GT flow 作为 decoder/OICM 输入，未改变 checkpoint、cache、split、训练配置或旧 8102/8103 viewer。
- 旧 run `cmdecoderv2-inspire-effect-20260908-151014` 的 effect 已完成但 viewer 被执行会话超时终止，manifest 已更正为 `STOPPED`；旧产物保留可审计。

## 2026-09-08 15:12:08 +0800 — V1.1.4 当前修正数据 best.pt 的 Inspire rollout self-effect 诊断

- activity_id: `cmdecoderv2-inspire-effect-v1.2.5-best-20260908-125300`
- timestamp: `2026-09-08 15:12:08 +0800`
- modification_version: `V1.1.4`（复用既有 rollout self-effect 诊断协议；输入 decoder/OICM 为本次修正数据终态）
- type: `diagnostic / experiment / operation`
- change_level: `L0`（按既有诊断协议运行，不修改模型、数据、split、checkpoint 或指标合同）
- approval: `user-approved`
- approval_basis: 用户要求用当前 `best.pt`，将 Inspire rollout 每步产生的 Cm 送入 OICM 预测 object point flow。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动；本次仅产生独立诊断输出并追加记录）
- run_id: `cmdecoderv2-inspire-effect-20260908-151014`
- run_status: `STOPPED`（368-step effect 已计算完成；端口 8104 的 Viser 后续被执行会话超时终止，产物保留）
- conclusion: `INCONCLUSIVE`（远距离 validity gate 子结论成立；单序列 rollout 整体质量仍未判定）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/`

**原因**

需要使用修正数据训练后的当前 decoder `best.pt` 重做既有纯 Inspire self-effect 诊断，观察 decoder 递归
rollout 产生的 Inspire hand flow 经冻结 OICM 后，对物体点流的预测是否仍遵守 5 cm interaction gate；
不与 GT object flow 比较。

**文件**

- [实验记录](experiment_log.md) — 追加当前终态 `best.pt` 的 Inspire rollout effect 证据及正式训练终态。
- [活动记录](activity_log.md) — 登记本次运行状态、命令、输入 checkpoint 和可视化入口。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt --device cuda:0 --sequence s1/mouse_lift --rl-root data/processed_data/inspire_rl_object_dexplore --activity-id cmdecoderv2-inspire-effect-v1.2.5-best-20260908-125300 --port 8104 --fps 8 --serve`
- decoder checkpoint: epoch `24` / step `36120`；SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`。
- frozen OICM checkpoint SHA256: `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。
- sequence: `s1/mouse_lift`，修正后 MANO-only test parent 中具备 DExplore Inspire tensor 的 held-out 序列；368 个 rollout transition。
- Viser: `http://localhost:8104` 曾返回 HTTP `200`；进程已停止。
- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/)
- [run manifest](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260908-151014/effect_summary.json)

**验证**

- 距离 `>50 mm`：12 帧全部 `sample_valid=false`，effective effect RMS 均值/最大值均为 `0 mm`；raw dummy head RMS 均值 `18.949 mm`，不作物理解释。
- 距离 `<=50 mm`：356 帧全部有效，effective effect RMS 均值 `4.528 mm`、中位数 `1.647 mm`、最大值 `52.341 mm`。
- 全局 effective effect RMS 均值 `4.380 mm`，OICM valid ratio `0.9674`；旧 manifest 未包含 display-only GT 字段，已由新完整重跑补齐。
- 初次尝试的历史序列 `s1/camera_takepicture_3_Retake` 不属于修正后的 63 条 test split，loader 在生成输出前按预期拒绝；随后选择同一新 test split 内的 `s1/mouse_lift`，未绕过 split 闸门。

**保护边界**

- 未读取未来 object pose/flow 或 GT object flow；未修改正式 decoder/OICM checkpoint、数据 cache、split、训练配置或已有 viewer。

## 2026-09-08 10:14:02 +0800 — V1.1.6 修正 DExplore RL 数据上的 CmDecoderv2 正式训练完成

- activity_id: `cmdecoderv2-decoder-v1.2.5-20260908-101402`
- timestamp: `2026-09-08 10:14:02 +0800`
- modification_version: `V1.1.6`
- type: `data / experiment / operation`
- change_level: `L2`（切换到修正后的 RL 物体轨迹、OICM V1.2.5 终态 checkpoint 和对应 decoder view；模型结构、loss、split 合同保持不变）
- approval: `user-approved`
- approval_basis: 用户确认按修正后的 DExplore RL 数据和 OICM V1.2.5 best checkpoint 从头训练 decoder。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有工作区改动；本次只新增独立 decoder config/view/output 并追加记录）
- run_id: `cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（正式训练和 checkpoint 选择按冻结合同完成；该结论不等于 MANO→Inspire rollout 质量成立）
- scope: `src/task/CmDecoderv2/`、`data/processed_data/cm_decoder_v2/dexplore_rl_v1_2_5/`、`outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/`

**原因**

旧 decoder view 和 checkpoint 绑定旧版 OICM 及旧物体轨迹；本次需要在修正后的 DExplore RL 物体轨迹和
OICM V1.2.5 终态 `best.pt` 上重新从头训练，以隔离数据修正对 decoder 的影响。

**验证**

- decoder view builder：`cmdecoderv2-view-full-20260908-100512`，train/val/test=`255/30/63`，RL
  sidecars=`285`，windows=`63988/7807`，split contract 为 Inspire RL train/val + MANO qualitative-only test。
- `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5_smoke.yaml`：`last_step=2`，完成 train/val，OICM strict-load、point-flow loss 和反向传播通过。
- `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`12 passed`。
- 2026-09-08 11:09:55 +0800 检查：正式 run 已完成约 `26600/75250` steps、进入 epoch 18，进程和三卡 worker 正常；当前最近验证为 step `25585` 的 `val/loss=0.0043540`、`val/hand/point_flow_epe_mm=13.3516`，历史最佳 `val/loss=0.0041447`（epoch 5 / step 7525）。这些仍是中途训练指标，不作为终态科研结论。
- 2026-09-08 11:56:43 +0800 检查：正式 run 已完成约 `50200/75250` steps、进入 epoch 34，进程和三卡 worker 仍正常；最近验证为 step `49665` 的 `val/loss=0.00412665`、`val/hand/point_flow_epe_mm=12.6080`，历史最佳已改善为 `val/loss=0.00400710`（epoch 24 / step 36120）。预计剩余约 45–55 分钟；仍不作终态科研结论。
- 终态：50 epochs / `75250` steps 正常完成，用时 `02:32:18`；best 为 epoch `24` / step `36120`，`val/loss=0.0040070958`、`val/hand/point_flow_epe_mm=12.4907`、h1=`9.0049 mm`；final epoch 50 的 `val/loss=0.0041402271`。

**文件/输入**

- [正式配置](../../configs/active/dexplore_rl_v1_2_5.yaml) — 指向 OICM V1.2.5 `best.pt` 和修正数据 view；从头训练。
- [smoke 配置](../../configs/active/dexplore_rl_v1_2_5_smoke.yaml) — 同一输入合同，关闭扰动并限制 2 steps。
- [decoder view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_2_5/) — 基于修正 OICM index，train/val Inspire RL，test MANO qualitative-only。
- OICM checkpoint SHA256: `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。

**运行**

- command: `CUDA_VISIBLE_DEVICES=4,5,6 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_2_5.yaml --distributed`
- device: `cuda:0,1,2` within `CUDA_VISIBLE_DEVICES=4,5,6`；world size `3`；per-device batch `8`；global batch `24`。
- total steps: `75250`；last epoch: `50`；best metric: `val/loss=0.0040070958`。
- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/)（`COMPLETED`）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/run_manifest.json)
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/config.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/best.pt) — epoch `24` / step `36120`，SHA256 `596ae0ceba947115a3b35c65735b5537f7e5452e1159d644ac4c89f9a7b101dc`。
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_2_5_20260908_101402/checkpoints/latest.pt) — epoch `50` / step `75250`，SHA256 `23c33e1022e970b659d0ae651df04620f9ec1894af98a1b8ae39881f0dfa6ac2`。

**保护边界**

- 旧 decoder view/checkpoint、旧 OICM cache/checkpoint、原始数据、MANO-only test 口径和已有可视化进程不修改。

## 2026-09-07 21:39:30 +0800 — V1.1.6 MANO/Inspire Cm 来源分类诊断

- activity_id: `cmdecoderv2-cm-source-classifier-final-20260907-214000`
- timestamp: `2026-09-07 21:39:30 +0800`
- modification_version: `V1.1.6`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local 表征诊断脚本和小分类器；冻结 OICM、数据 cache、正式 split、decoder checkpoint 和训练变量）
- approval: `user-approved`
- approval_basis: 用户确认训练小分类器检验当前 Cm 是否包含可识别的 MANO/Inspire 手来源信息。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留前序 CmDecoderv2/ObjectInteractionCm 和 rollout 诊断工作区改动）
- run_id: `cmdecoderv2-cm-source-classifier-20260907-213748`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（冻结 OICM 的 `cm_tokens` 含有明显可识别的 MANO/Inspire source/embodiment signal；纯静态手型归因仍需额外控制）
- scope: `src/task/CmDecoderv2/research/cm_hand_source_classifier/`；不改变正式训练或数据合同

**文件**

- `src/task/CmDecoderv2/research/cm_hand_source_classifier/run.py` — 从 MANO/Inspire geometry cache 生成冻结 OICM `cm_tokens`，按 sequence split 训练小型 source classifier，并报告 anchor control。
- `src/task/CmDecoderv2/research/cm_hand_source_classifier/README.md`、`experiment.yaml` — 记录标签、特征、共同 object 类别、split 和输出合同。
- `src/task/CmDecoderv2/docs/plan/v1.1.md` — 记录用户批准的 Cm 来源分类诊断边界。
- `src/task/CmDecoderv2/docs/README.md`、`docs/current_versions.yaml` — 增加分类诊断入口并更新 CmDecoderv2 指针为 `V1.1.6`。
- `src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md` — 其他工作区差异保留，未由本实验覆盖。

**原因**

需要判断当前 Cm 是否保留了足够的 MANO/Inspire 手来源信息，使一个小型分类器能够在未见过的 sequence 上区分两类 source。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cm_hand_source_classifier.run --activity-id cmdecoderv2-cm-source-classifier-final-20260907-214000 --device cuda:0 --frames-per-sequence 8 --batch-size 8 --epochs 40`
- OICM config: `src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml`；checkpoint 为冻结 `best.pt`，SHA256 `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- split: train `226 MANO + 226 Inspire` sequences；val/test `36 MANO + 36 Inspire` sequences；仅使用两种 source 在 train/val 都共同出现的 23 个 object 类别；每条 sequence 抽取 8 个 transition。
- [运行目录](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/)
- [run manifest](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/run_manifest.json)
- [features.npz](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/features.npz)
- [metrics.json](../../research/cm_hand_source_classifier/output/cmdecoderv2-cm-source-classifier-20260907-213748/metrics.json)

**验证**

- 主特征是 `cm_tokens` 的 mean/max/std pooling（96D），不输入 hand flow、object points、anchor 或 decoder 输出；anchor mean/max/std（9D）只作为 control。
- 全部样本：Cm test accuracy `0.7448`、balanced accuracy `0.7448`、AUROC `0.8467`；anchor control test AUROC `0.6263`。
- `sample_valid=true` 子集：Cm test accuracy `0.9752`、balanced accuracy `0.9752`、F1 `0.9752`、AUROC `0.9958`；anchor control accuracy `0.6281`、AUROC `0.6702`。
- 特征提取和分类器训练共生成 `4192` 个样本；`PYTHONPATH=. pytest -q src/task/CmDecoderv2/tests`：`12 passed`；脚本 `py_compile` 和 `git diff --check` 通过。
- activity 链接审计待本条写入后执行。

**保护边界与解释限制**

- 这是 source/embodiment 可分性证据，不是“纯静态手型”因果证明；Cm 可能同时编码 hand geometry、motion statistics 和 source-domain 差异。
- 旧 smoke/中间输出保留但不作为最终证据；最终 run 使用小 manifest 和上述 metrics 作为入口。
- 未修改 OICM/decoder checkpoint、正式 train/val/test split、geometry cache、训练配置或共享代码。

## 2026-09-07 20:34:30 +0800 — V1.1.5 MANO rollout self-effect 348-step 对照诊断

- activity_id: `cmdecoderv2-mano-effect-full-20260907-203300`
- timestamp: `2026-09-07 20:34:30 +0800`
- modification_version: `V1.1.5`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local MANO rollout 旁路诊断与可视化；冻结 decoder/OICM、正式 MANO test source、split/cache 和既有 viewer）
- approval: `user-approved`
- approval_basis: 用户要求使用 MANO rollout 做同一 self-effect 诊断并可视化比较。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 和前序诊断工作区改动）
- run_id: `cmdecoderv2-mano-effect-20260907-203231`
- run_status: `RUNNING`（348 步 effect 计算完成；8103 Viser 仍运行）
- conclusion: `SUPPORTED`（本序列中 MANO-source rollout 的远距离 hand flow 同样被 OICM 5 cm validity gate 屏蔽；decoder 整体跨 embodiment 动作质量仍不由此单独判定）
- scope: `src/task/CmDecoderv2/research/mano_rollout_effect/`；不改变正式 test 或训练合同

**文件**

- `src/task/CmDecoderv2/research/mano_rollout_effect/run.py` — 使用正式 MANO source 递归 rollout，并将每步 Inspire hand flow 旁路 forward 冻结 OICM。
- `src/task/CmDecoderv2/research/mano_rollout_effect/README.md`、`experiment.yaml` — 记录 MANO 对照实验合同和入口。
- `src/task/CmDecoderv2/docs/plan/v1.1.md` — 记录用户批准的 MANO rollout self-effect 对照边界。
- `src/task/CmDecoderv2/docs/README.md`、`docs/current_versions.yaml` — 增加诊断导航并更新 CmDecoderv2 指针为 `V1.1.5`。
- `src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md` — 其他工作区差异保留，未由本运行覆盖。

**原因**

需要判断同一 OICM effect 诊断在正式 MANO-source rollout 下是否也把远离物体的 decoder 产生运动识别为无有效 object effect，并和纯 Inspire rollout 结果对照。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.mano_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt --device cuda:0 --sequence s1/camera_takepicture_3_Retake --activity-id cmdecoderv2-mano-effect-full-20260907-203300 --port 8103 --fps 8 --serve`
- tmux session: `cmdecoderv2_mano_effect_20260907_2033`；Viser：`http://localhost:8103`。
- [运行目录](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/)
- [run manifest](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/run_manifest.json)
- [effect.npz](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect.npz)
- [effect_summary.json](../../research/mano_rollout_effect/output/cmdecoderv2-mano-effect-20260907-203231/effect_summary.json)

**验证**

- 2-step smoke：完成，MANO handoff 初始距离约 `1.27–1.31 m`，OICM 无效且 effective effect 为 `0`。
- `PYTHONPATH=. pytest -q src/task/CmDecoderv2/tests`：`12 passed`；新脚本 `py_compile` 通过。
- 348 个 MANO-source rollout transition 完成；输入使用当前 GRAB/MANO object geometry/pose 和当前 rollout Inspire geometry，不读取 paired Inspire GT、未来 object pose/flow 或 object-flow GT。
- 距离 `>50 mm` 的 258 帧：`sample_valid=0/258`，raw effect RMS 均值 `12.535 mm`，effective effect RMS 均值 `0 mm`。
- 距离 `<=50 mm` 的 90 帧：`89/90` 有效，effective effect RMS 均值 `9.194 mm`，最大 `41.302 mm`；唯一无效近距离帧为 frame `258`，距离约 `47.42 mm` 且 sampled active count 为 `0`。
- 8103 HTTP `200`；`git diff --check` 通过；activity 链接审计待本条写入后执行。

**保护边界**

- raw effect 仅作为无效 OICM sample 的 dummy path 审计量；Viser 默认显示 effective effect，可切换 raw。
- 未修改正式 MANO test index、split/cache、decoder/OICM checkpoint、训练配置、paired Inspire 数据或现有 8101/8102 viewer。

## 2026-09-07 17:31:59 +0800 — V1.1.4 rollout self-effect 348-step OICM 旁路诊断

- activity_id: `cmdecoderv2-inspire-effect-full-20260907-171315`
- timestamp: `2026-09-07 17:31:59 +0800`
- modification_version: `V1.1.4`
- type: `code / diagnostic / experiment / operation`
- change_level: `L1`（Task-local 诊断脚本、实验索引和可视化；冻结 decoder/OICM、数据 split、cache 和既有 viewer 不变）
- approval: `user-approved`
- approval_basis: 用户确认在 rollout 每一步用自身 Inspire point flow 重新 forward OICM effect，不与 GT object flow 比较。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2 point-flow/active-only 及其他 Task 工作区改动）
- run_id: `cmdecoderv2-inspire-effect-20260907-171349`
- run_status: `RUNNING`（effect 已完成；8102 Viser 仍运行）
- conclusion: `SUPPORTED`（本序列中 OICM 的 5 cm validity gate 对 rollout hand flow 生效；decoder 的动作质量仍不由本诊断单独判定）
- scope: `src/task/CmDecoderv2/research/inspire_rollout_effect/`；不改变正式训练或已有 viewer

**文件**

- `src/task/CmDecoderv2/research/inspire_rollout_effect/run.py` — 生成 rollout hand flow，冻结 OICM forward，并提供 raw/effective effect viewer。
- `src/task/CmDecoderv2/research/inspire_rollout_effect/README.md`、`experiment.yaml` — 记录实验合同和入口。
- `src/task/CmDecoderv2/tests/test_inspire_effect.py` — raw/effective mask、分桶摘要和颜色映射测试。
- `src/task/CmDecoderv2/docs/plan/v1.1.md` — 记录用户批准的 rollout self-effect 诊断边界。
- `docs/current_versions.yaml` — CmDecoderv2 指针更新为 `V1.1.4`。
- `src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md` — 本次之前已存在的工作区差异，仅作保护范围，未由本运行修改。

**原因**

需要判断“远离物体时的 rollout Inspire 运动是否被 Cm/OICM 识别为无 object effect”，同时避免把 GT object flow 或未来 object pose 引入诊断。

**运行与产物**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.inspire_rollout_effect.run --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt --device cuda:0 --sequence s1/camera_takepicture_3_Retake --activity-id cmdecoderv2-inspire-effect-full-20260907-171315 --port 8102 --fps 8 --serve`
- tmux session: `cmdecoderv2_inspire_effect_20260907_1713`；Viser：`http://localhost:8102`。
- [运行目录](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/)
- [run manifest](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/run_manifest.json)
- [effect.npz](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect.npz)
- [effect_summary.json](../../research/inspire_rollout_effect/output/cmdecoderv2-inspire-effect-20260907-171349/effect_summary.json)

**验证**

- `py_compile` 和 `PYTHONPATH=. pytest -q src/task/CmDecoderv2/tests`：`12 passed`；包含本诊断新增的 3 个合同测试。
- 348 个 rollout transition 完成；只使用当前 object geometry/pose 作为 `object_pose_t` 输入，没有读取未来 object pose/flow。
- 距离 `>50 mm` 的 122 帧：`sample_valid=0/122`，raw effect RMS 均值 `12.533 mm`，但合同有效 effect 为 `0 mm`。
- 距离 `<=50 mm` 的 226 帧：`225/226` 有效，effective effect RMS 均值 `7.959 mm`，最大 `39.277 mm`；唯一无效帧为 frame `127`，距离 `48.29 mm` 且 sampled active count 为 0，符合 1024 点采样未命中 candidate 的语义。
- 8102 HTTP `200`；`git diff --check` 通过；`audit_diff.py --check-links` 通过（13 个变更路径、4 个本地链接）。

**保护边界**

- raw effect 仅作为 dummy computational path 审计量；不能解释为无效 Cm 的物理 effect。Viser 默认显示 effective effect，可切换 raw 观察该差异。
- 未停止 8101 纯 Inspire viewer；未修改 decoder/OICM checkpoint、正式 split/cache、训练配置或既有输出。

## 2026-09-07 15:45:00 +0800 — V1.1.1 best.pt 纯 Inspire 自回归 viewer 与跳帧控件运行核验

- activity_id: `cmdecoderv2-inspire-best-jump-20260907_103836`
- timestamp: `2026-09-07 15:45:00 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / operation`
- change_level: `L0`（核验既有 viewer、trajectory 和运行清单；本次未修改模型、数据、配置或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户要求继续查看 point-flow `best.pt` 的 Inspire 自回归 rollout，并增加跳帧观察漂移；viewer 实现和启动范围已在前一轮确认。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留用户已有 CmDecoderv2 point-flow/active-only 改动及其他 Task 改动）
- run_id: `cmdecoderv2-inspire-test-vis-20260907-103857`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（纯 Inspire viewer 工程运行正常；尚未据此形成 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`

**运行与控件**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize_inspire_test --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt --device cuda:0 --sequence s1/camera_takepicture_3_Retake --port 8100 --fps 8 --activity-id cmdecoderv2-inspire-best-jump-20260907_103836`
- tmux session: `cmdecoderv2_inspire_best_20260907_103836`；GPU3 进程仍在运行。
- 实际 Viser 地址：`http://localhost:8101`。请求端口 `8100` 被 Viser 自动递增处理，命令行摘要和 manifest 中的请求端口仍为 8100；本次以监听端口 8101 为准。
- 序列：`s1/camera_takepicture_3_Retake`，352 帧；使用 point-flow 正式训练的 `best.pt`（epoch 3 / step 10296）。
- viewer 默认 `rollout`；支持 `teacherforced` / `rollout`、`GT` / `pred` / `both`、点大小滑块、`Jump size (frames)`（1–30）、`Previous jump` / `Next jump`、播放/停止和当前帧重启 rollout。

**产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260907-103857/)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260907-103857/run_manifest.json)
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260907-103857/trajectory.npz)
- [viewer 实现](../../visualize_inspire_test.py)

**原因**

本次继续的目的只是把上次会话中已经启动的 viewer 运行状态、实际监听端口和跳帧控件证据补齐；不重新训练，也不改变纯 Inspire 诊断与正式 MANO-only test 的边界。

**验证**

- `ps` / `tmux capture-pane`：viewer 主进程存活，Viser 已报告监听 8101。
- `urllib.request http://127.0.0.1:8101/`：HTTP `200`；8097–8100 当前无监听。
- trajectory 文件已存在且可读；运行清单记录 checkpoint、sequence、source contract 和 `run_status=RUNNING`。

**保护边界**

- 未停止或重启正式训练；未修改 best/latest checkpoint、数据 view/cache、split、配置或旧 viewer 输出。
- 下列工作区差异在本次继续前已存在，本次未修改，仅作为交接保护范围：`src/task/CmDecoderv2/`、`src/task/ObjectInteractionCm/docs/logs/activity_log.md`。
- 本次继续操作不改变 MANO→Inspire 科研结论；仍需人工通过 viewer 比较 teacherforced 与 rollout 的漂移。

## 2026-09-07 10:21:16 +0800 — V1.1.3 point-flow 三卡正式训练终态与收敛诊断

- activity_id: `cmdecoderv2-pointflow-train-final-20260907-102116`
- timestamp: `2026-09-07 10:21:16 +0800`
- modification_version: `V1.1.3`
- type: `experiment / diagnostic / operation`
- change_level: `L0`（只读解析既有训练产物并补记终态，不修改模型、数据或 checkpoint）
- approval: `user-approved`
- approval_basis: 用户要求核对当前训练结果；原正式训练由用户明确批准。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 Task-local point-flow/active-only 实现和用户工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046`
- run_status: `COMPLETED`
- last_step: `171600`
- last_epoch: `50`
- best_metric: `val/loss=0.003370641680534728`（epoch `3`，step `10296`）
- conclusion: `INCONCLUSIVE`（正式训练工程上完整结束且 point-flow validation 明显优于 epoch 1；但验证最优过早、后续回退，且 q/rotation 诊断劣于 identity baseline，跨 embodiment 效果需用 best.pt 可视化确认）
- scope: `src/task/CmDecoderv2/`

**终态产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/)
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/run_manifest.json)
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/config.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/train.log)
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt) — epoch `3` / step `10296`。
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/latest.pt) — epoch `50` / step `171600`。

**原因**

训练进程和 tmux session 均已退出，需要区分“训练完整结束”“优化收敛”和“跨 embodiment 方案成立”，并核对 best.pt 是否确实对应最低 validation loss。

**验证**

- 50 epochs / `171600` steps 正常完成，用时 `07:28:30`；`train.log` 末行明确报告结束，无残留 CmDecoderv2 train/torchrun 进程。
- train loss 从 epoch 1 的 `0.0116015` 持续降到 epoch 50 的 `0.00296866`；val loss 从 `0.00757946` 降至 epoch 3 最佳 `0.00337064`，之后未再刷新，epoch 50 为 `0.00382680`。
- best epoch 3：validation 全 horizon point-flow EPE `11.3794 mm`，h1 `6.8813 mm`；final epoch 50 为 `12.1087 / 9.1764 mm`。因此选模必须使用 best.pt，不应使用 latest.pt。
- best epoch 3 的 h1 诊断：q MAE `0.04289 rad`、wrist translation `5.935 mm`、wrist rotation `2.570°`；对应 identity baseline 为 `0.01529 rad / 11.046 mm / 2.010°`。模型明显改善 wrist translation，但 q 与 wrist rotation 尚未超过 identity baseline。
- validation `cm/supervision_frame_ratio=0.966668`，5 cm active-only 与 OICM valid gate 正常生效。

**保护边界与下一证据**

- 未修改或覆盖 best/latest checkpoint、数据 view/cache、split、旧 run 或可视化产物。
- 下一步应固定使用 best.pt 做 teacher-forced 与 rollout 可视化；仅凭本次 loss 曲线不能把 MANO→Inspire 可行性写为 `SUPPORTED`。

## 2026-09-06 20:31:18 +0800 — V1.1.3 启动 CmDecoderv2 point-flow 三卡正式训练

- activity_id: `cmdecoderv2-pointflow-train-20260906-203118`
- timestamp: `2026-09-06 20:31:18 +0800`
- modification_version: `V1.1.3`
- type: `operation / experiment`
- change_level: `L3 + L2`（按已定稿 v1.1 plan 启动长任务；point-flow、active-only 和 RL-only 数据合同已由用户批准）
- approval: `user-approved`
- approval_basis: 用户明确要求“现在开始训练”，沿用此前 GPU 0/1/2 三卡正式训练范围。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（沿用已批准的 Task-local point-flow/active-only 改动；不覆盖旧 run、OICM checkpoint 或 viewer）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（已确认训练 wiring 正常；尚无收敛或 MANO→Inspire 效果结论）
- scope: `src/task/CmDecoderv2/`

**命令与运行状态**

- command: `CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --distributed`
- tmux session: `cmdecoderv2_pointflow_20260906_2030`
- launcher/rank 已启动，world size `3`，global batch `24`，per-device batch `8`，训练上限 `171600` steps。
- step `2000` 已写入 point-flow loss `0.0086414`，h1 点流 EPE `16.6742 mm`，active/supervision ratio `1.0`；最近 20 个记录点的 loss 均值约 `0.01177`、标准差约 `0.00206`，属于 batch-level 波动，不能据此判断收敛。

**产物**

- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/)（`RUNNING`）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/run_manifest.json)
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/config.json)
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/metrics.jsonl)
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/train.log)
- best/latest checkpoint：`outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/`（尚未生成，`PENDING`）

**原因**

在 point-flow loss、可微 Inspire FK、5 cm active-only gate 和旧 OICM best checkpoint strict-load smoke 均通过后，开始新的独立三卡正式 run。旧 decoder run `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701` 保持停止并保留。

**验证**

- OICM frozen best checkpoint SHA256 与配置一致：`fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- 训练启动后已生成 run manifest/config/metadata/metrics/train.log，3 个 rank 正常运行；GPU 0/1/2 当前显存约 `5.6/3.1/3.1 GB`（总显存 24 GB），利用率随 rank 负载波动。
- 早期 step 已产生 finite point-flow loss 和诊断指标；正式科研结论等待完整 validation 与最终 checkpoint。

**保护边界与停止入口**

- 未覆盖旧 checkpoint、旧输出、decoder view/cache、split 或 8098/8099 viewer。
- 如需停止，使用 tmux session `cmdecoderv2_pointflow_20260906_2030` 发送安全中断，并在本条补记终态、last step/epoch、best metric 和 checkpoint。

## 2026-09-06 20:22:37 +0800 — V1.1.3 将 CmDecoderv2 loss 改为可微 Inspire 点流监督

- activity_id: `cmdecoderv2-pointflow-20260906-202237`
- timestamp: `2026-09-06 20:22:37 +0800`
- modification_version: `V1.1.3`
- type: `architecture / code / experiment`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认纯 point-flow loss、q/wrist 输出保持不变、固定 1538 点/seed 2024、`object_pose_t` 坐标、可微 FK 和 active-only mask 全部按默认方案执行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（新增 Task-local point-flow FK/loss/测试/计划；不覆盖旧 checkpoint、split 或 cache）
- run_id: `cmdecoderv2-pointflow-smoke-20260906-202237`
- run_status: `COMPLETED`（仅完成静态、CPU smoke 和真实 checkpoint forward/loss smoke；未启动长训练）
- conclusion: `SUPPORTED`（工程 point-flow wiring、FK parity、梯度和 active-only gate 通过；科研效果尚未重新训练，仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；旧 q/wrist-loss decoder run、旧 checkpoint 和 8098/8099 viewer 保持不变。

**原因**

用户要求保留 q/wrist 作为 decoder 输出，但将训练监督改为由 q/wrist 经 Inspire FK 生成的对应表面点流，避免直接 q/wrist loss 与最终几何目标不一致。

**修改**

- [pointflow.py](../../pointflow.py) — 使用同一 URDF、area-weighted surface sampling、seed `2024` 构造固定 1538 点，并实现可微 Inspire FK、SE(3) delta 和 `object_pose_t` 变换；surface buffers 不进入 checkpoint state dict。
- [model.py](../../model.py) — 保留 q/wrist 输出，训练 batch 额外生成预测点和预测点流；可视化 batch 缺少 pose 字段时保持旧输出合同。
- [dataset.py](../../dataset.py) — 提供扰动当前 wrist/object pose 及未来 GT hand points，GT 点流与扰动后的当前状态对齐。
- [runner.py](../../runner.py)、[config.py](../../config.py) — 纯 point-flow Smooth-L1（beta `0.005 m`），q/wrist 只保留诊断指标；5 cm active mask 与 `cm_sample_valid` 继续 gating。
- [test_pointflow.py](../../tests/test_pointflow.py) — 增加 cache parity、可微梯度和 horizon 点坐标变换测试。
- [plan/v1.1.md](../plan/v1.1.md) — 记录 point-flow supervision 合同。

**验证**

- `py_compile`：通过。
- `CUDA_VISIBLE_DEVICES='' ... pytest -q src/task/CmDecoderv2/tests`：`9 passed`。
- differentiable FK 与现有 1538-point cache parity：最大误差约 `0.00014 mm`，梯度 finite。
- 旧 decoder `best.pt` strict load：通过；固定 surface buffers 不破坏旧 checkpoint 兼容。
- 真实 OICM checkpoint + pilot batch（1024 object points）forward/loss smoke：预测点流 `[1,4,1538,3]`、loss finite，active-only supervision ratio 正常。
- 未启动新的长训练，因此没有新的 `metrics.jsonl`、`train.log` 或正式 checkpoint。

**保护边界与回滚**

- 未修改 decoder view/cache、正式 train/val/test split、OICM checkpoint、旧 decoder checkpoint 或 viewer 进程。
- 回滚入口为恢复旧 runner q/wrist loss、移除 point-flow module/model 输出字段，并隔离本次未启动的新训练目录。

## 2026-09-06 19:49:47 +0800 — V1.1.2 停止旧 decoder 训练并启用 5 cm active-only supervision

- activity_id: `cmdecoderv2-active-only-20260906-194947`
- timestamp: `2026-09-06 19:49:47 +0800`
- modification_version: `V1.1.2`
- type: `operation / code / experiment`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户要求停止训练并加入 `active_only`，确保进入 5 cm 范围后才开始 q/wrist 监督；该条沿用此前 5 cm diagnostic 的明确修正方向。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（Task-local active-only 代码/配置/测试/计划增补；不覆盖旧 checkpoint 或正式 split/cache）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `STOPPED`（进程核查时已不存在；未向训练进程发送 kill；旧 run 输出保留）
- last_step: `255900`；last_epoch: `40`
- best_metric: `val/loss=1.3800124928725293`（step `83213`，epoch `13`）
- best_checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- latest_checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt`
- conclusion: `INCONCLUSIVE`（工程 mask smoke 通过；active-only 版本尚未重新长训，不能据此下科研效果结论）
- scope: `src/task/CmDecoderv2/`；8098/8099 viewer 保持运行，正式 train/val/test split、decoder view/cache、OICM checkpoint 和旧 decoder checkpoint 保持不变。

**原因**

旧 decoder 训练的 `data.active_only=false`，且 q/wrist loss 未使用 `cm_sample_valid`；历史指标中约一半窗口帧可能来自无效 Cm。用户要求停止旧训练并只在 5 cm active frame 监督。

**修改**

- [config.py](../../config.py)、[dexplore_rl_v1_1_1.yaml](../../configs/active/dexplore_rl_v1_1_1.yaml)、[dexplore_rl_v1_1_1_smoke.yaml](../../configs/active/dexplore_rl_v1_1_1_smoke.yaml) — `data.active_only=true`。
- [dataset.py](../../dataset.py) — 返回与 `C_t...C_{t+K-1}` 对齐的完整 object-pool 5 cm `active_mask`。
- [runner.py](../../runner.py) — 将 `active_mask` 与冻结 OICM `cm_sample_valid` 相交，q/wrist 三项 loss 和 metrics 只使用有效 horizon，并对剩余 horizon 重新归一化。
- [test_dataset.py](../../tests/test_dataset.py)、[test_model.py](../../tests/test_model.py) — 增加 active mask 和全无效 loss 的回归测试。
- [plan/v1.1.md](../plan/v1.1.md) — 记录本次用户追加批准的 5 cm active-only 训练合同。

**验证**

- `py_compile`：通过。
- `CUDA_VISIBLE_DEVICES='' ... pytest -q src/task/CmDecoderv2/tests`：`7 passed`。
- CPU active-only dataset smoke：pilot windows `428 -> 359`，样本 `active_mask` 与窗口形状一致。
- CPU runner smoke：active mask 与 `cm_sample_valid` 相交、全无效 batch 返回有限零 loss，反向梯度 finite。
- [旧 metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)、[旧 train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log) — 旧无 active-only run 的证据保留。

**保护边界与回滚**

- 未删除或覆盖旧 decoder/OICM checkpoint，未修改 formal split、view/cache、原始数据和可视化进程。
- 回滚入口为恢复 `active_only=false`、移除 `active_mask` loss gate，并隔离本次未启动的新训练 run；旧 run 仍可作为 baseline 使用。

## 2026-09-06 19:31:01 +0800 — V1.1.1 修复 Inspire rollout restart 回调死锁并重启 viewer

- activity_id: `cmdecoderv2-inspire-test-camera-20260906-192900`
- timestamp: `2026-09-06 19:31:01 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户报告 rollout 卡死；诊断确认 restart 回调中的同步 Viser property update 与普通锁重入造成 futex deadlock，修复后重启同一 camera test viewer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（新增 Task-local UI 修复和运行记录；不改正式 split/cache/checkpoint）
- run_id: `cmdecoderv2-inspire-test-vis-20260906-193101`
- run_status: `RUNNING`
- conclusion: `SUPPORTED`（已定位并修复工程死锁；纯 Inspire teacher/rollout 科研效果仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；正式 MANO-only test index、训练和 checkpoint 选择保持不变。

**文件**

- [visualize_inspire_test.py](../../visualize_inspire_test.py) — 将 Viser 回调锁改为 `threading.RLock`，并避免 restart 回调重复触发 rollout 初始化。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-193101/run_manifest.json) — 修复后 viewer 运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-193101/trajectory.npz) — 修复后 camera test 轨迹输出。

**原因**

旧进程停在 `futex_wait_queue_me`：`on_restart` 持有普通 `threading.Lock` 时设置 `mode.value`，Viser 同步触发 `on_mode_change`，后者再次获取同一把锁。该死锁与 decoder 推理或数据无关。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_inspire_test.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 修复后 viewer 已在 `http://localhost:8099` 重新启动，序列仍为 `s1/camera_takepicture_3_Retake`，Viser HTTP 200。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧死锁运行 `cmdecoderv2-inspire-test-vis-20260906-192300` 输出保留；未修改 raw data、正式 index、cache、checkpoint 或训练配置。
- 回滚入口为停止 8099 viewer、隔离本次输出，或单独回退 RLock/restart 回调改动。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 19:23:00 +0800 — V1.1.1 切换纯 Inspire viewer 到右手接触轨迹并修复 teacher 尾帧边界

- activity_id: `cmdecoderv2-inspire-test-camera-20260906-191200`
- timestamp: `2026-09-06 19:23:00 +0800`
- modification_version: `V1.1.1`
- type: `code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户要求换一条轨迹可视化；根据右手几何接触诊断切换到 `s1/camera_takepicture_3_Retake`，并修复播放到不完整 Cm window 尾帧时的 teacher 边界错误。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（本次新增 viewer 边界修复和运行记录；不改正式 split/cache/checkpoint）
- run_id: `cmdecoderv2-inspire-test-vis-20260906-192300`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（仅为右手接触轨迹的定性 viewer；不作 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`；正式 MANO-only test index、训练和 checkpoint 选择保持不变。

**文件**

- [visualize_inspire_test.py](../../visualize_inspire_test.py) — teacher 模式在最后不足 `K+1` 帧时停止预测、保留 GT playback，避免尾帧 `IndexError`。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-192300/run_manifest.json) — 切换后的 camera test viewer 清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-192300/trajectory.npz) — `s1/camera_takepicture_3_Retake` 的 Inspire GT/object 和 teacher/rollout 槽位。

**原因**

`s1/alarmclock_offhand_1` 的右手 Inspire 几何距离中位数约 `217.7 mm`，仅 `33/271` 帧小于 `2 cm`，不适合评估右手抓取。`s1/camera_takepicture_3_Retake` 有 `253/352` 帧小于 `2 cm`（约 `71.9%`），因此作为本次可视化轨迹。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_inspire_test.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 新 viewer 已在 `http://localhost:8099` 启动，序列 `s1/camera_takepicture_3_Retake`、352 帧；Viser HTTP 200。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧 offhand run 输出保留，不覆盖；未修改正式 index、raw data、cache、checkpoint 或训练配置。
- 回滚入口为停止当前 8099 viewer、恢复旧 run 目录；代码边界修复可单独回退。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 19:04:23 +0800 — V1.1.1 新增纯 Inspire held-out test teacher/rollout 诊断 viewer

- activity_id: `cmdecoderv2-inspire-test-vis-20260906-implementation`
- timestamp: `2026-09-06 19:04:23 +0800`
- modification_version: `V1.1.1`
- type: `architecture / code / data / experiment / operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认使用原 test parent 的 Dexplore RL-Inspire hand/object/q/wrist 完整轨迹，teacher 每帧 GT state，rollout 只在 handoff 使用一次 GT state 后递归；该分支仅作纯 Inspire 诊断。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810`
- worktree_dirty: `true`（新增诊断 viewer 和 plan/activity 修改；不改正式 split/cache/checkpoint）
- run_id: `cmdecoderv2-inspire-test-vis-20260906-190423`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（工程 wiring smoke 通过；该 viewer 不产生正式 test 科研结论）
- scope: `src/task/CmDecoderv2/`；原正式 `dexplore_rl_v1_1/index.json` 的 `test=mano` 合同、训练和 checkpoint 选择均保持不变。

**文件**

- [visualize_inspire_test.py](../../visualize_inspire_test.py) — 新增独立 viewer，按需从 raw `interaction_hand_inspire.pt`、对应 Dexplore object pose、原 parent object surface 和 Inspire URDF FK 构造 held-out Inspire test source；支持 teacherforced/rollout、GT/pred/both 和点大小滑块。
- [plan/v1.1.md](../plan/v1.1.md) — 记录用户追加批准的 Inspire-test diagnostic 分支及其不改变正式 split 的边界。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-190423/run_manifest.json) — 当前 GPU3 诊断运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-inspire-test-vis-20260906-190423/trajectory.npz) — 当前序列的 Inspire GT、Dexplore object 和 teacher/rollout 槽位。

**原因**

用户希望区分“纯 Inspire decoder 是否能学好”和 MANO→Inspire 跨 embodiment 效果。正式 decoder view 的 125 条 test entry 只标记 MANO，但每个 held-out parent 都存在对应 raw Dexplore RL-Inspire tensor；新 viewer 使用这些 raw test trajectories 在线构造 source，不把它们写回正式 index，也不用于训练或选 checkpoint。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_inspire_test.py`：通过。
- GPU3 real-checkpoint smoke：`s1/alarmclock_offhand_1` 的 q `[271,18]`、hand `[271,1538,3]`、object `[271,4096,3]`、wrist `[271,4,4]` 全部构造成功；teacher h=1 和 rollout h=1 均输出 finite Inspire points `[1538,3]`。
- viewer 已在 `http://localhost:8099` 启动，HTTP/Viser server 正常；formal MANO viewer 8098 未改动。
- `git diff --check`：通过；activity link audit：通过（5 个本地链接可导航）。

**保护边界与回滚**

- 未生成或修改新的全量 OICM/decoder cache；未修改训练数据 index、模型配置、checkpoint、原始 Dexplore/GRAB 数据。
- 回滚入口为移除 [visualize_inspire_test.py](../../visualize_inspire_test.py)、诊断 plan 增补和本次输出目录；不影响正式 MANO viewer 和 decoder 训练。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 17:28:42 +0800 — V1.1.1 修复 handoff 帧 Viser 回调崩溃并重启 GPU3 viewer

- activity_id: `cmdecoderv2-mano-grab-vis-20260906-172800`
- timestamp: `2026-09-06 17:28:42 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic / code / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户报告 viewer 中断；根据 traceback 修复 Task-local UI 对 handoff 初始状态的错误处理并重新启动同一 viewer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 CmDecoder、ObjectInteractionCm、CmDecoderv2 及训练改动）
- run_id: `cmdecoderv2-mano-grab-vis-20260906-172842`
- run_status: `RUNNING`
- conclusion: `SUPPORTED`（已定位并修复 handoff UI 的工程错误；MANO→Inspire 科研效果仍为 `INCONCLUSIVE`）
- scope: `src/task/CmDecoderv2/`；未修改正式 decoder 训练、GPU0/1/2 训练进程、数据 cache、split 或 checkpoint。

**文件**

- [visualize_mano.py](../../visualize_mano.py) — handoff 帧的 Inspire 初始状态 `cm_valid=None` 时显示 initialized 文本，不再执行 `len(None)`。
- `docs/current_versions.yaml` — 保留既有 `CmDecoderv2: V1.1.1` 指针；本次不改变版本值。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172842/run_manifest.json) — 修复后 GPU3 viewer 的运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172842/trajectory.npz) — 修复后运行的 GT 和 rollout 槽位。

**原因**

旧运行在切换 rollout 后推进 frame 时崩溃。traceback 显示 handoff 初始状态的 `cm_valid=None` 被 UI 当成数组执行 `len(valid)`；这是状态语义正常、显示分支错误，不是模型或 parent cache 数据错误。

**验证**

- 原始 traceback 已保存于旧 Viser session：`TypeError: len() of unsized object`，位置为 `visualize_mano.py` 的 Cm-valid 文本分支。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_mano.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- 修复后 GPU3 viewer 已在 `http://localhost:8098` 启动，HTTP 200；新的 run manifest 已生成，formal decoder training GPU0/1/2 未被中断。
- `git diff --check` 与 activity link audit：通过。

**保护边界与回滚**

- 旧崩溃运行 `cmdecoderv2-mano-grab-vis-20260906-172219` 的输出保留作诊断证据；修复后运行使用新的 run_id，不覆盖旧输出。
- 回滚入口为移除本次 handoff 分支改动或隔离新 viewer 输出；不删除原始 GRAB parent cache、decoder checkpoint 或正式训练输出。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 17:22:19 +0800 — V1.1.1 新增 MANO/GRAB-source Viser 定性 viewer 并启动 GPU3

- activity_id: `cmdecoderv2-mano-grab-vis-20260906-172000`
- timestamp: `2026-09-06 17:22:19 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确确认最终可视化必须使用原始 GRAB MANO 手和物体轨迹；Dexplore RL 只用于 decoder 训练；允许从任意帧以 q=0 + MANO wrist 切换到 rollout。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 CmDecoder、ObjectInteractionCm、CmDecoderv2 及训练改动）
- run_id: `cmdecoderv2-mano-grab-vis-20260906-172219`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（仅确认数据来源、坐标和 viewer wiring；没有 paired Inspire test GT，不作定量或科学效果结论）
- scope: `src/task/CmDecoderv2/`；未修改正式 decoder 训练、GPU0/1/2 训练进程、Dexplore/GRAB 原始数据、已有 RL-Inspire 诊断 viewer。

**文件**

- [visualize_mano.py](../../visualize_mano.py) — 新增最终定性 viewer：GT playback 渲染原始 GRAB parent cache 的 MANO hand/object；rollout 在当前帧以零 finger q 和 MANO `hand_root_pose_world` 初始化，使用 MANO source Cm window 递归预测 Inspire；GT/pred/both 和点大小滑块均可切换。
- `docs/current_versions.yaml` — 保留既有 `CmDecoderv2: V1.1.1` 指针；本次不改变版本值。
- [run_manifest.json](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172219/run_manifest.json) — 当前 GPU3 Viser 运行清单。
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-grab-vis-20260906-172219/trajectory.npz) — 原始 MANO/GRAB GT、物体轨迹和当前已生成的 rollout 槽位。

**原因**

之前的 `visualize_teacherforce.py` 使用 Dexplore RL-Inspire validation GT，因此不能作为最终 MANO→Inspire 测试。新 viewer 将 Dexplore RL 限定为 decoder 训练来源，最终可视化只使用原始 GRAB parent cache；没有把对应 Inspire 轨迹作为测试标签，避免把不同抓取方式混在比较中。

**验证**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize_mano --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence s1/alarmclock_offhand_1 --port 8098 --fps 8 --activity-id cmdecoderv2-mano-grab-vis-20260906-172000`
- checkpoint SHA256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`。
- viewer: `http://localhost:8098`；GPU3 进程正在运行，formal decoder training 的 GPU0/1/2 进程未停止。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmDecoderv2/visualize_mano.py`：通过。
- `CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`。
- GPU3 real-checkpoint smoke：加载 `s1/alarmclock_offhand_1` 原始 parent cache，GT shapes 为 hand `[271,1538,3]` / object `[271,4096,3]`；rollout handoff frame 0 后递归到 frame 1，预测 Inspire points `[1538,3]`、q `[6]`、wrist `[4,4]`，输出 finite。
- `curl http://localhost:8098/`：HTTP 200；run manifest 与 trajectory 已生成。
- `git diff --check`：通过。

**保护边界与回滚**

- 未修改旧 RL-Inspire teacher-forcing viewer；未修改任何数据 cache、split、checkpoint 或正式训练配置。
- 回滚入口为删除/隔离新增 [visualize_mano.py](../../visualize_mano.py) 和本次 viewer 输出目录；不删除原始 GRAB parent cache 或 decoder checkpoint。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。

## 2026-09-06 16:42:31 +0800 — V1.1.1 object pose 非接触运动因果诊断

- activity_id: `ACT-20260906-164231-CMDECODERV2-OBJECT-MOTION-CAUSALITY-DIAGNOSTIC`
- timestamp: `2026-09-06 16:42:31 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户追问“没有接触为什么物体会动”；只读核对 Dexplore native tensor 字段定义、cache builder 和 frame 200 轨迹，不修改数据或训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（确认数据没有“无接触冻结物体”的物理约束；是否按右手接触重构/裁剪物体轨迹仍待研究决定）
- scope: `src/task/CmDecoderv2/`、ObjectInteractionCm cache builder 和 Dexplore export schema 的只读诊断；未修改研究变量。

**原因**

确认 frame 200 无右手几何接触时 object pose 仍变化的直接数据来源，并区分 object trajectory、contact label 与 viewer 渲染行为。

**关键证据**

- Dexplore native tensor 的 `198:201` 是 object position、`201:205` 是 object quaternion、`205` 是 object contact flag；cache builder 分别读取 pose 和 flag，没有用 contact flag 对 pose 做冻结或门控。
- 因此 `contact=0` 只表示该帧的 object contact label 为 0，不代表 `T_object[t+1] == T_object[t]`；数据管线不承诺无接触时物体静止。
- `s1/banana_lift` frame 200 的 native contact flag 为 `0`，右手到物体最近距离约 `32.3 mm`，但 object frame 200→201 仍有约 `1.2 mm` translation 和 `2.57°` rotation；这是原始 object pose trajectory 中的轻微抖动/轨迹变化。
- Dexplore 文档明确指出 object motion 可能由未被当前 Inspire 右手显示的其他手/身体接触驱动；当前 tensor 的 object flag 是 object-level contact，不是本 viewer 右手的独立因果门控。
- 更大的 object motion 出现在 frame 220–230，此前 frame 220–226 仍存在 `<2 cm` 接触，随后 object z 约从 `1.129 m` 上升到 `1.335 m`，说明该段是接触后的真实 object trajectory，而非 Viser 自动生成。

**证据入口**

- [Dexplore export schema](../../../../../../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md)
- [cache builder](../../tools/data/build_dexplore_view.py)
- [ObjectInteractionCm RL cache builder](../../../ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读核对 native tensor 字段、builder 字段映射、contact flag、world-frame object pose 和 frame 200 hand-object 距离；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:38:21 +0800 — V1.1.1 Viser 世界坐标与 frame 200 物体运动诊断

- activity_id: `ACT-20260906-163821-CMDECODERV2-VISER-COORD-FRAME200-DIAGNOSTIC`
- timestamp: `2026-09-06 16:38:21 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 Viser 使用的坐标系，并指出 frame 200 GT 未接触但物体仍在动；只读核对 viewer 数组索引、world/object pose 和 frame 200 的 hand-object 距离。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（当前 world-frame 物体运动与数组一致；是否要改成 object-centric viewer 仍待用户决定）
- scope: `src/task/CmDecoderv2/` 与 validation geometry 的只读诊断；未修改 viewer、数据或训练。

**原因**

需要确认 frame 200 的物体运动是否来自 Viser 坐标/索引错误，还是来自 Dexplore geometry 中记录的 object pose trajectory。

**关键证据**

- 当前 Viser 使用 `obj_points_world.npy`、`hand_points_world.npy` 和 `wrist_pose_world.npy` 的 Dexplore world frame（manifest 标记为 `dexplore_native_object_pose_world`），没有把所有帧变换到固定 object-centric frame。
- teacherforced frame `t` 显示：object 使用 frame `t+1`，GT current 使用 frame `t`，GT next 使用 frame `t+1`；因此 frame 200 的场景物体是 object frame 201，绿色 GT next 也是 hand frame 201。
- frame 200 的 hand→object 最近距离约 `32.3 mm`，`<20 mm` 接触比例为 `0%`，所以此帧确实没有 2 cm 级物理接触；但此前 frame 175–195 有多段 `<2 cm` 接触，frame 200 是一次短暂分离后的状态。
- frame 200→201 物体质心移动约 `1.4 mm`，object pose translation 约 `1.2 mm`，旋转约 `2.57°`；因此 world-frame 播放时物体确实会轻微移动，主要是该帧的 object pose/rotation trajectory，不是 viewer 凭空移动。
- 更明显的 object motion 出现在 frame 220–230：此区间 hand 在 frame 220–226 仍有 `<2 cm` 接触，之后 object pose z 从约 `1.129 m` 上升到 `1.335 m`；这与前一段接触后的物体运动一致。

**证据入口**

- [teacherforce viewer](../../visualize_teacherforce.py)
- [geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读比较 world-frame object/hand 数组、object pose 帧间增量和 frame 200 的近邻距离；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:29:24 +0800 — V1.1.1 validation GT 预接触运动诊断

- activity_id: `ACT-20260906-162924-CMDECODERV2-GT-PRECONTACT-DIAGNOSTIC`
- timestamp: `2026-09-06 16:29:24 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户反馈 validation Viser 中 GT 在接触物体前已经运动；只读检查 RL q、wrist、hand/object geometry、5 cm candidate mask 和 source contact flag，不修改 viewer、数据或训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有 viewer、训练和数据改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser 仍在 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（数值上 GT 与 cache/FK 自洽，观察到的是动作的预接触阶段；尚未决定是否新增 grasp-only crop）
- scope: `src/task/CmDecoderv2/` 与 `data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/sequences/val/s1_banana_lift/` 的只读诊断；未改变研究变量。

**原因**

需要确认“接触前运动”是数据包含 reach/approach 动作，还是 q/wrist、hand geometry、object pose 或 frame index 错位。

**关键证据**

- `s1/banana_lift` 为 30 Hz、578 帧；物体基本静止，手从 frame 0 就在接近，hand-object 最近距离从约 `1316 mm` 降到 frame 50 的 `72 mm`。
- 按生成的 5 cm candidate mask，首个 active frame 为 `53`；按实际最近距离，首个 `<20 mm` 接触帧为 `54`；Dexplore source contact flag 首个为 `56`。因此前约 `1.8 s` 是正常 reach/approach 段，不是 hand 静止等待接触。
- frame 0→1 的 GT 手指 q 变化约 `24.18°`（6 维 q 的 L2），腕部平移约 `40.76 mm`、旋转约 `3.39°`；这与“动作一开始就运动”一致。
- `geometry/manifest.json` 标明 RL hand geometry 由 Dexplore native q 经 Inspire URDF FK 生成，decoder view 的 wrist pose 也由同一 q/FK 生成；当前证据未发现 GT frame index 或 coordinate-frame 错位。
- source flag 与几何 5 cm mask 不完全同帧：source flag 从 frame 56 开始，几何 mask 从 frame 53 开始；这是 5 cm candidate 与 Dexplore contact label 的语义差异，不等于 GT 错位。

**证据入口**

- [geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)
- [decoder view manifest](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/sequences/val/s1_banana_lift/manifest.json)
- [teacherforce trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)

**验证**

- 只读计算 hand/object 最近距离、候选 active、source contact flag、GT q/wrist 帧间增量和 object/hand centroid 轨迹；未停止 GPU0/1/2 正式训练，也未修改当前 Viser。

## 2026-09-06 16:20:49 +0800 — V1.1.1 Viser 点云显示与 teacherforced/rollout 切换

- activity_id: `ACT-20260906-162100-CMDECODERV2-TEACHERFORCE-TOGGLE`
- timestamp: `2026-09-06 16:20:49 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户确认暂不使用 mesh，要求点云大小滑块、GT/pred/both 显示切换以及 teacherforced/rollout 可随时切换；采用已说明的 GT handoff 语义。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、旧 rollout 和用户改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-161956`
- run_status: `RUNNING`（Viser server 使用 GPU3 / port 8097）
- conclusion: `INCONCLUSIVE`（交互可视化 wiring 已通过 smoke；不构成 decoder 科研效果结论）
- scope: `src/task/CmDecoderv2/`（更新 `visualize_teacherforce.py`）与独立 teacher-forcing Viser 输出目录；未修改训练变量、checkpoint 或正式训练。

**原因**

需要在同一 validation sequence 中直接比较 teacher forcing 与 state-feedback rollout，并能按帧选择 GT、预测或两者，同时调整点云可见性。

**实现与运行**

- Viser 新增 `Execution mode`：`teacherforced` / `rollout`；切到 rollout 时以当前帧 GT state 初始化，之后按预测 state 递归，跳帧会从该帧 GT state 重新 handoff；切回 teacherforced 时自动回到 GT state。
- 新增 `Hand display`：`GT` / `pred` / `both`；新增 `Point size (m)` 滑块（`0.001–0.020 m`）。当前仍只渲染点云，不加载 mesh。
- [teacherforce.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/teacherforce.npz)、[run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-161956/run_manifest.json)。Viser 地址：`http://localhost:8097`。

**验证**

- 8-frame Viser smoke 和无服务器 rollout handoff smoke 均通过；完整 validation sequence `s1/banana_lift` 574 帧预计算完成。
- `python3 -m py_compile src/task/CmDecoderv2/visualize_teacherforce.py`、`git diff --check` 和 Viser HTTP 探活通过。
- 正式 decoder 训练 GPU0/1/2 未停止；本次可视化只使用 GPU3。

## 2026-09-06 15:58:41 +0800 — V1.1.1 RL-Inspire validation teacher-forcing Viser 可视化

- activity_id: `ACT-20260906-161100-CMDECODERV2-TEACHERFORCE-VISER`
- timestamp: `2026-09-06 15:58:41 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户要求先不看 closed-loop rollout，改用 teacher forcing 检查 decoder 表现并使用 Viser 可视化；默认使用 RL-Inspire validation，因为 MANO test 没有对应 Inspire GT。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、rollout 输出和用户改动）
- run_id: `cmdecoderv2-teacherforce-val-20260906-155735`
- run_status: `RUNNING`（Viser server 仍在 GPU3 / port 8097 提供交互式查看）
- conclusion: `INCONCLUSIVE`（teacher-forced wiring 和定性/诊断指标已生成；不等同于最终科研结论）
- scope: `src/task/CmDecoderv2/`（新增 `visualize_teacherforce.py`）与独立 validation teacher-forcing 输出目录；不修改训练变量、checkpoint 或 MANO rollout。

**原因**

需要隔离 closed-loop state feedback drift：每一帧使用 RL-Inspire validation 的真实当前 q/wrist 和真实 Cm window，只预测 h=1，不把预测状态反馈到下一帧；Viser 同时显示 GT 当前手、GT 下一帧手和 decoder 预测。

**实现与运行**

- 新增 [visualize_teacherforce.py](../../visualize_teacherforce.py)，只接受 `variant=inspire_rl` 的 validation entry，使用 frozen OICM + CmDecoderv2 checkpoint，输出 teacher-forced h=1 轨迹并启动 Viser。
- 使用 GPU3（进程内 `cuda:0`），序列 `s1/banana_lift`，574 个完整 window/frame；Viser 地址为 `http://localhost:8097`。
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-155735/run_manifest.json)、[teacherforce.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-teacherforce-val-20260906-155735/teacherforce.npz)。

**诊断指标**

- 全部帧：q MAE `1.367°`，wrist translation `13.98 mm`，wrist rotation `3.38°`，hand nearest-neighbor EPE `8.40 mm`。
- 当前 Cm 有效帧（379/574，`66.0%`）：q MAE `1.289°`，wrist translation `5.15 mm`，wrist rotation `2.47°`，hand EPE `5.50 mm`。
- 当前 Cm 无效帧：wrist translation `31.14 mm`，hand EPE `14.03 mm`；该差异是 teacher-forcing 诊断中必须单独报告的 mask 分层结果。

**验证**

- 8-frame smoke 已完成；checkpoint load、GT state/window 对齐、forward 和 Viser HTTP 均正常。
- `python3 -m py_compile src/task/CmDecoderv2/visualize_teacherforce.py`、`git diff --check` 通过。
- 正式 decoder 训练 GPU0/1/2 未停止；本次可视化仅使用 GPU3。当前 run 仍保持 RUNNING，未覆盖历史 rollout 或 checkpoint。

## 2026-09-06 15:50:00 +0800 — V1.1.1 核对 5 cm 帧屏蔽与 decoder loss mask

- activity_id: `ACT-20260906-155000-CMDECODERV2-5CM-MASK-DIAGNOSTIC`
- timestamp: `2026-09-06 15:50:00 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问训练是否按 5 cm 屏蔽帧；本次只读核对数据集 active mask、OICM 交互半径和 CmDecoderv2 loss，不修改代码、配置或运行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、CmDecoderv2 实现和用户改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（已确认 5 cm 机制的作用层级；当前 decoder 未按该 mask 重新训练）
- scope: `src/task/CmDecoderv2/` 与其引用的 ObjectInteractionCm 配置/代码；未改变研究变量

**原因**

需要区分 OICM 提取 Cm 时的 5 cm 几何筛选、OICM loss 的 sample mask，以及 CmDecoderv2 自身是否按 5 cm 有效帧筛选和屏蔽 decoder loss。

**核对结果**

- ObjectInteractionCm 配置的 `interaction_radius_m=0.05`、`active_only=true`：数据集会根据预计算的 `obj_candidate_mask_5cm` 丢弃当前帧完全没有 5 cm 候选的行；运行时局部交互边也要求距离不超过 5 cm，`sample_valid` 再表示采样到的 object pool 是否仍有有效交互。OICM runner 会用 `sample_valid` 屏蔽自身的 object/hand flow loss。
- CmDecoderv2 配置的 `data.active_only=false`：decoder window 不会因 5 cm candidate mask 在数据集层被丢弃。
- CmDecoderv2 runner 只记录 `cm/valid_frame_ratio`，q、wrist translation、wrist rotation 三项 loss 没有用 `cm_sample_valid` 做 mask。因此当前 decoder 训练并不是“按 5 cm 有效帧训练”；无效 Cm 仍进入 loss。

**证据入口**

- [CmDecoderv2 config](../../config.py)、[CmDecoderv2 dataset](../../dataset.py)、[CmDecoderv2 runner](../../runner.py)
- [ObjectInteractionCm RL config](../../../../../src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml)、[ObjectInteractionCm dataset](../../../../../src/task/ObjectInteractionCm/dataset.py)、[ObjectInteractionCm runner](../../../../../src/task/ObjectInteractionCm/runner.py)
- [formal training metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)

**验证**

- 只读检查配置、数据集行筛选、5 cm 交互半径、`sample_valid` 传播和 decoder loss 计算；未停止 GPU0/1/2 正式训练。

## 2026-09-06 15:33:31 +0800 — V1.1.1 完整 rollout 的“乱动”原因诊断

- activity_id: `ACT-20260906-153331-CMDECODERV2-ROLLOUT-DIAGNOSTIC`
- timestamp: `2026-09-06 15:33:31 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求分析完整 rollout 中 Inspire 手持续乱动的原因；本次仅做只读轨迹、代码、坐标和训练指标核对，不修改训练、checkpoint、数据或可视化代码。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、CmDecoderv2 实现和用户改动）
- run_id: `cmdecoderv2-mano-viz-20260906-151601`；关联正式训练 run：`cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `COMPLETED`（诊断）；正式训练保持运行状态，未被本次检查中断
- conclusion: `INCONCLUSIVE`（已定位实现与坐标问题，但尚未按修复方案重新训练/复测，不能据此下科研效果结论）
- scope: `src/task/CmDecoderv2/` 与独立 rollout/训练产物；未修改研究变量

**原因**

用户反馈完整 rollout 中 Inspire 手持续乱动，需要区分模型本身、无效 Cm 处理、初始化坐标和视频完整性四类因素；本次只做证据核对，不改变正式训练运行。

**关键证据**

- 可视化 rollout 共 267 steps，MP4 完整；`cm_sample_valid` 为 `[267,4]`，仅 504/1068 个 window-frame 位置有效。当前帧在约 0–90、220–266 均无效，但 visualizer 仍无条件执行 `finger += q_delta`、`wrist = wrist @ delta`，因此 dummy Cm 产生的非零预测会持续累积漂移。前 0–90 帧的每步 q 增量中位数约 `0.00965 rad`，腕部平移中位数约 `5.61 mm`；到第 219 帧 q 范数约 `1.548`、腕部相对初始位移约 `0.944 m`。
- CmDecoderv2 runner 当前只记录 `cm/valid_frame_ratio`，没有用 `cm_sample_valid` mask q/translation/rotation loss。正式训练已完成 epoch 的 valid-frame ratio 约 `0.51816`（train）/`0.52366`（val），约 48% 的 frame supervision 来自无效/dummy Cm；因此当前 best checkpoint 不能视作已按有效交互帧严格训练的最终模型。
- 本次 `mano_wrist` 初始化直接读取 parent GRAB cache 的 `hand_root_pose_world.npy`，但 decoder 测试 geometry view 的 object pose 与 parent cache 的 object pose 存在共同平移偏差，frame 0 约为 `[-39.7, -583.3, +17.6] mm`（旋转基本一致）。直接把 parent-world MANO wrist 当作 view-world Inspire base 会造成初始空间错位；应通过 object pose 关系先变换到 view world。
- 完整 MP4 [rollout](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/rollout.mp4) 已由 `ffprobe` 验证为 267 帧、30 FPS、8.9 秒；此前提前结束的版本是显式 `--max-steps 120`，不是当前编码或接触阶段截断。

**证据入口**

- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/trajectory.npz)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/run_manifest.json)
- [visualize.py](../../visualize.py)、[runner.py](../../runner.py)、[model.py](../../model.py)
- [formal training metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)

**建议的修复顺序（本次未执行）**

1. 用 `T_view_obj @ inv(T_parent_obj) @ T_mano_wrist` 把 MANO wrist 初始化转换到 decoder view world，并确认 MANO root 与 Inspire `hand_base_link` 的固定坐标约定。
2. rollout 只在当前帧 `cm_sample_valid[0,0]` 为真时应用 h=1 action；无效帧明确保持状态，不能用 window 内任意未来有效帧替代当前帧判断。
3. 在 CmDecoderv2 训练中对无效当前帧 mask/drop q、translation、rotation supervision，再重新训练并复测；当前正式 run 和 best.pt 仅作为诊断基线保留。

**验证**

- 只读检查 `trajectory.npz` 的有效帧掩码、状态累积和模型输出范数；对 `visualize.py`、`runner.py`、数据 schema 和 object pose 做源码/数值对齐。
- 本条记录写入后运行 `audit_diff.py --check-links` 与 `git diff --check`；本次未停止正式 GPU0/1/2 训练，也未覆盖旧输出。

## 2026-09-06 15:19:06 +0800 — V1.1.1 修正 MANO wrist 初始化并完成完整 sequence 可视化

- activity_id: `ACT-20260906-151000-CMDECODERV2-MANO-VIZ-MANO-WRIST`
- timestamp: `2026-09-06 15:19:06 +0800`
- modification_version: `V1.1.1`
- type: `code / experiment / operation`
- change_level: `L2 + L1`
- approval: `user-approved`
- approval_basis: 用户确认“用 MANO wrist 初始化、Inspire 手指 q 保持 0”，并要求完整跑完接触过程；本次不改变训练变量。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留既有训练、数据 view、旧可视化和用户改动）
- run_id: `cmdecoderv2-mano-viz-20260906-151601`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（仅为 MANO-only qualitative rollout；无 paired Inspire test GT，不构成定量或科研效果结论）
- scope: `src/task/CmDecoderv2/visualize.py` 与独立 `outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/` 产物；正式训练 GPU0/1/2 未停止
- checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- checkpoint_sha256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`

**文件**

- `src/task/CmDecoderv2/` — 本 Task 现有实现、配置、数据工具、测试和文档均保留；本次实际代码改动集中在 `visualize.py`，其余路径仅由 activity scope 归档。
- [visualize.py](../../visualize.py) — 默认新增 `mano_wrist` 初始化模式；从 MANO parent cache 读取右手 `hand_root_pose_world.npy`，初始化 Inspire wrist；保留 `manual` 模式和 q=0 手指初始化；MP4 改为 ffmpeg 编码并校验文件大小。
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/run_manifest.json)、[MP4](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/rollout.mp4)、[trajectory](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/trajectory.npz)、[PNG frames](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-151601/frames/)

**原因**

旧 rollout 显式限制为 `120` steps，而 `s1/alarmclock_offhand_1` 的完整可用长度为 `frame_count-K=267`，因此视频在接触阶段提前结束；旧默认 wrist 也没有对齐 MANO source。按用户确认，新的 rollout 使用第 0 帧 MANO wrist 世界位姿，手指 q 保持全 0，并跑完整 267 steps。

**验证**

- 命令：`CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence-index 0 --max-steps 10000 --initial-state mano_wrist --render-every 1 --write-mp4 --output-root outputs/cmdecoderv2 --activity-id ACT-20260906-151000-CMDECODERV2-MANO-VIZ-MANO-WRIST`
- `py_compile src/task/CmDecoderv2/visualize.py`、`git diff --check`：通过。
- sequence `s1/alarmclock_offhand_1`：267 steps；MP4 经 `ffprobe` 验证为 H.264、267 帧、30 FPS、8.9 秒；checkpoint SHA 在运行前后保持一致。
- 旧正式 decoder 训练 torchrun 仍在 GPU0/1/2 运行；本次只使用 GPU3。
- 首帧、133 帧和末帧已抽查；画面仅作人工观察，MANO→Inspire 的定性质量仍需用户结合完整视频评估。

## 2026-09-06 15:06:22 +0800 — V1.1.1 使用当前 best.pt 完成 GPU3 MANO 定性 rollout

- activity_id: `ACT-20260906-150300-CMDECODERV2-MANO-VIZ`
- timestamp: `2026-09-06 15:06:22 +0800`
- modification_version: `V1.1.1`
- type: `experiment / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户要求“拿 GPU3 用目前的 best.pt 来可视化”；使用独立输出目录，不触碰 0/1/2 正式训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cmdecoderv2-mano-viz-20260906-150326`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（PNG/MP4 仅供人工定性观察，不含 paired Inspire test GT，不作定量或科研效果结论）
- scope: `src/task/CmDecoderv2/` 的可视化入口与独立 `outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/` 产物
- device: `CUDA_VISIBLE_DEVICES=3`（进程内 `cuda:0`）；正式 decoder 训练 GPU 0/1/2 保持运行
- checkpoint: `outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt`
- checkpoint_sha256: `263f19252799b992518bb5afd100211b498edf61c5eb68d2c67fd62b3aaec615`

**原因**

在不停止正式三卡训练的情况下，用当前 decoder best checkpoint 对 MANO-only test sequence 做 receding-horizon `h=1` 定性 rollout，供人工检查 MANO source 与 Inspire predicted 的空间关系。

**命令与产物**

- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.visualize --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --checkpoint outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt --device cuda:0 --sequence-index 0 --max-steps 120 --render-every 1 --write-mp4 --output-root outputs/cmdecoderv2 --activity-id ACT-20260906-150300-CMDECODERV2-MANO-VIZ`
- sequence: `s1/alarmclock_offhand_1`；120 steps；初始 Inspire state 由脚本默认参数显式记录。
- [运行目录](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/)
- [run manifest](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/run_manifest.json)
- [MP4 rollout](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/rollout.mp4)
- [PNG frames](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/frames/)
- [trajectory.npz](../../../../../outputs/cmdecoderv2/cmdecoderv2-mano-viz-20260906-150326/trajectory.npz)

**验证**

- 可视化进程在 GPU3 正常完成；manifest 记录 `paired_inspire_test_gt=false`，未读取配对 Inspire GT。
- 120 张 PNG 已生成；最初的 imageio MP4 写出因 `write() got an unexpected keyword argument 'fps'` 失败，随后使用 ffmpeg 将同一批 PNG 转换为有效 H.264 MP4（120 帧、30 FPS、4 秒），并通过 `ffprobe` 验证。
- `best.pt` 在可视化前后 SHA256 保持一致；正式训练进程仍在 GPU0/1/2 运行，未被中断。
- 首帧/中段/末帧已做肉眼抽查；这是人工观察入口，不把当前画面直接解释为 decoder 科研效果成立。

## 2026-09-06 15:01:07 +0800 — V1.1.1 训练中期 loss/validation 状态诊断

- activity_id: `ACT-20260906-150107-CMDECODERV2-LOSS-STATUS-DIAGNOSTIC`
- timestamp: `2026-09-06 15:01:07 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练状态；只读检查进程、metrics、validation 和 checkpoint，不中断或调整训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- last_step: `156100`（epoch 25 进行中）
- last_completed_epoch: `24`
- best_metric: `val/loss=1.3800124928725293`（epoch 13 / step 83213）
- conclusion: `INCONCLUSIVE`（train loss 仍下降，但 validation 在 epoch 13 后平台化并波动，训练尚未结束）
- scope: `src/task/CmDecoderv2/` 与正式 run 的只读 loss/validation/进程/checkpoint 诊断

**原因**

需要区分优化是否仍在下降与泛化 validation 是否继续改善，避免把持续下降的 train loss 误判为模型已收敛。

**验证**

- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)：已完成 epoch 1–24；train loss 从 `2.059890` 降到 `1.159078`，仍在下降。
- validation 最佳为 epoch 13 的 `1.380012`；epoch 14–24 在约 `1.3867–1.4349` 间波动，epoch 24 为 `1.399368`，尚未恢复 epoch 13 最佳。
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt) 为 epoch 13 / step 83213；[latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt) 为 epoch 24 / step 153624。
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)：torchrun 与 3 个 rank 仍在运行；GPU 0/1/2 利用率约 `68%/57%/71%`。
- 当前判断：训练过程没有异常退出，但 validation 已出现早期平台/波动；是否最终收敛需继续观察后续 epochs，当前科研结论仍为 `INCONCLUSIVE`。

## 2026-09-06 12:54:47 +0800 — V1.1.1 正式训练早期 loss 趋势诊断

- activity_id: `ACT-20260906-125447-CMDECODERV2-LOSS-TREND-DIAGNOSTIC`
- timestamp: `2026-09-06 12:54:47 +0800`
- modification_version: `V1.1.1`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练状态及 loss 是否持续下降；只读检查进程、metrics 和 checkpoint，不中断或调整训练。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- last_step: `49300`（查询时 epoch 8 进行中）
- last_completed_epoch: `7`
- best_metric: `val/loss=1.424412033150659`（epoch 7 / step 44807）
- conclusion: `INCONCLUSIVE`（前 7 个 epoch 的 train/val loss 持续下降，但训练仅完成约 15%，尚不能判断最终收敛）
- scope: `src/task/CmDecoderv2/` 与正式 run 的只读 loss/进程/checkpoint 诊断

**原因**

step loss 受 batch 组成与扰动影响明显，需使用完整 epoch 的 train aggregate 和 validation 指标判断趋势，并拆分 q、wrist translation、wrist rotation 确认总 loss 的下降来源。

**验证**

- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)：epoch 1→7 的 train loss `2.059890→1.507454`，val loss `1.785547→1.424412`，两者在每个已完成 epoch 均下降。
- 总 loss 的主要下降来自 wrist translation：train `1.794571→1.249410`，val `1.623184→1.263725`；q 与 wrist rotation 分项目前主要呈平台/小幅改善。
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt) 均为 epoch 7 / step 44807，`best_metric=1.424412033150659`。
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)：torchrun 与 3 个 rank 仍在运行；GPU 0/1/2 查询时利用率约 `69%/57%/69%`，无退出迹象。
- 当前 step loss 在约 `0.87–2.63` 间抖动是 batch-level 波动，不能据此要求逐 step 单调下降；科研结论需等待更多 validation 和最终 MANO→Inspire 可视化。

## 2026-09-06 11:57:01 +0800 — V1.1.1 使用冻结 OICM best.pt 启动 CmDecoderv2 三卡正式训练

- activity_id: `ACT-20260906-115701-CMDECODERV2-FORMAL-TRAIN-STARTED`
- timestamp: `2026-09-06 11:57:01 +0800`
- modification_version: `V1.1.1`
- type: `operation / experiment`
- change_level: `L3 + L2`（按已定稿 v1.1 plan 运行；锁定 OICM checkpoint gate 并启动长任务）
- approval: `user-approved`
- approval_basis: 用户明确要求“先停止训练吧，然后用 bestpt 直接开始训练 CmDecoderv2。照样用三卡”。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留旧 CmDecoder、ObjectInteractionCm 及既有工作区改动）
- run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（正式训练已启动；终态前不作收敛或跨 embodiment 效果结论）
- scope: 固定 Dexplore RL decoder view、Temporal-D2 `K=4`、q/wrist 输出和正式 train/val；GPU 0/1/2，DDP，per-device batch 8，global batch 24，50 epochs（320050 total steps）。
- frozen_oicm_checkpoint: `outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt`
- frozen_oicm_checkpoint_sha256: `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`

**命令与产物（运行中）**

- command: `CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml --distributed`
- [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/)（`RUNNING`, PENDING）
- [配置快照](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/config.json)（`RUNNING`, PENDING）
- [run manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/run_manifest.json)（`RUNNING`, PENDING）
- [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/metrics.jsonl)（`RUNNING`, PENDING）
- [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/train.log)（`RUNNING`, PENDING）
- [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt)（`RUNNING`, PENDING）
- [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/latest.pt)（`RUNNING`, PENDING）

**启动证据**

- launcher 与 3 个 rank 已正常启动；`train_setup` 报告 `device=cuda:0`、`world_size=3`、`global_batch=24`、`total_steps=320050`。
- 本条目仅记录运行启动；正式训练终态、last step/epoch、best metric 和实际 checkpoint 路径待运行结束或用户要求停止后补记。

**文件**

- `src/task/CmDecoderv2/` — 本 Task 的实现、配置、数据工具、测试和文档；本次只更新正式配置中的 OICM checkpoint SHA gate，未改动模型代码或数据 view。

**原因**

OICM 续训已按用户要求停止，必须把静态 decoder 配置绑定到已冻结的 `best.pt`，再使用 approved Temporal-D2、RL-only train/val 和三卡 DDP 进行正式训练，避免运行中动态追随 checkpoint。

**验证**

- `sha256sum outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt` 与配置 gate 一致：`fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。
- 启动命令已创建 run manifest/config/metadata/metrics/train.log；3 个 rank 正常运行，GPU 0/1/2 利用率已上升。
- 前 800 steps 已写入 metrics，loss 与 q/wrist 子损失均为 finite；这是工程运行证据，不代表 decoder 已收敛或 MANO→Inspire 定性效果成立。

## 2026-09-06 10:56:47 +0800 — V1.1.1 Temporal-D2 实现、完整 decoder view 与 smoke

- activity_id: `cmdecoderv2-v1.1.1-20260906-102433-implementation`
- timestamp: `2026-09-06 10:56:47 +0800`
- modification_version: `V1.1.1`
- type: `architecture / code / data / experiment / operation / documentation`
- change_level: `L3 + L2`
- approval: `user-approved`
- approval_basis: 用户确认 v1.1 默认方案、要求首版直接使用 Temporal-D2，并明确回复“可以，你直接开始实现吧”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（旧 CmDecoder、ObjectInteractionCm 和既有实验改动均保留且未纳入本 Task）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅工程实现、数据合同和 smoke；正式训练尚未开始，不构成收敛或跨 embodiment 效果结论）
- scope: 新建独立 CmDecoderv2；实现 Dexplore RL decoder view、`K=4` Temporal-D2、q/wrist loss、
  BaseRunner 训练入口与 MANO-only 定性 rollout。未修改 `src/base/`、旧 `src/task/CmDecoder/`、
  ObjectInteractionCm 代码/cache/checkpoint、原始 GRAB/Dexplore 数据、既有 split 或运行进程。

**设计与文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 新增 `CmDecoderv2: V1.1.1` Task 指针。
- [Task package](../../) 与 [config.py](../../config.py) — 建立独立 Python Task 包和默认静态合同；不从旧
  CmDecoder 导入兼容实现。
- [指导 v1.1](../指导/v1.1.md) 与 [最终计划 v1.1](../plan/v1.1.md) — 冻结研究目标、数据语义、Temporal-D2、
  q/wrist、扰动和分阶段闸门。
- [model.py](../../model.py) — 冻结并 strict-load OICM；把 `K x 16` Cm/anchor/time token set 作为 memory，
  当前 global/link/future-step 作为 query；不添加 slot-ID embedding，输出四步 q residual 与 wrist 相对 `SE(3)`。
- [kinematics.py](../../kinematics.py) — 固化 Dexplore native→URDF、6 个独立 q、mimic、`hand_base_link`
  wrist FK 和 18 个 object-frame link query。
- [dataset.py](../../dataset.py) 与 [decoder view builder](../../tools/data/build_dexplore_view.py) — 每个
  `K=4` Cm window 使用 `K+1=5` 个连续 30 Hz cache frame；RL train/val 提供 q/wrist GT，MANO test 不含 GT。
- [Task-local tools](../../tools/) — 包含独立 tools/data Python 包与 decoder view 入口。
- [runner.py](../../runner.py)、[train.py](../../train.py) 与 [正式配置](../../configs/active/dexplore_rl_v1_1_1.yaml) —
  实现 q Smooth-L1、translation x100 Smooth-L1、SO(3) geodesic、`0.8^(h-1)` horizon 权重和 per-horizon/identity 指标。
- [smoke 配置](../../configs/active/dexplore_rl_v1_1_1_smoke.yaml) 与 [Task tests](../../tests/) — smoke
  关闭 perturbation；正式配置使用已批准的 q/translation/rotation Gaussian perturbation。
- [visualize.py](../../visualize.py) — MANO source Cm 的 receding-horizon rollout 只执行 `h=1`，显式记录
  初始 Inspire state，导出 q/wrist/点云、PNG，并可选 MP4；不读取 paired Inspire test GT。
- [Task 文档入口](../README.md)、[架构记录](architecture_log.md)、[实验记录](experiment_log.md) 与
  [可迁移事实](repo_memory.md) — 记录新 Task 的事实和证据导航。
- [资产合同](../../assets/README.md) — Task-local Inspire 右手 URDF 由本机软链接提供，URDF SHA256 写入 view。

**原因**

旧 CmDecoder 的 HRDexDB/point-flow 兼容路径与本研究的 Dexplore RL q/wrist 语义不同。独立 Task 可以固定
MANO-source Cm → Inspire target 的数据边界，并用未来 Cm window + 当前 Inspire state 直接检验
Temporal-D2 与 receding-horizon 假设，同时避免把同一 parent 的 MANO/Inspire pair 泄露进 decoder 监督。

**数据运行**

- run_id: `cmdecoderv2-view-pilot-20260906-105616`
- run_status: `COMPLETED`
- command: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.tools.data.build_dexplore_view --mode pilot --output data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot --window-size 4 --resume --activity-id cmdecoderv2-v1.1.1-20260906-102433-implementation`
- evidence: [pilot view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/)、
  [manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/manifest.json)、
  [index.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/index.json)、
  [run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot/run_manifest.json)。
- result: RL train/val 各 1 sequence、MANO-only test 1 sequence，供静态检查和 smoke。

- run_id: `cmdecoderv2-view-full-20260906-105602`
- run_status: `COMPLETED`
- command: `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.tools.data.build_dexplore_view --mode full --output data/processed_data/cm_decoder_v2/dexplore_rl_v1_1 --window-size 4 --resume --activity-id cmdecoderv2-v1.1.1-20260906-102433-implementation`
- evidence: [完整 view](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/)、
  [manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/manifest.json)、
  [index.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/index.json)、
  [run_manifest.json](../../../../../data/processed_data/cm_decoder_v2/dexplore_rl_v1_1/run_manifest.json)。
- result: RL train `503` sequence / `153602` window，RL val `62` / `19730` window，MANO-only test
  `125` sequence；565 个 RL sidecar 全部存在且 shape 正确，test entry 中 Inspire GT 字段计数为 `0`。

**训练 smoke**

- final run_id: `cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=3 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.train --config src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1_smoke.yaml`
- last_step: `2`; last_epoch: `1`; best_metric: `val/loss=1.4703673253`
- evidence: [运行目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/)、
  [config.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/config.json)、
  [run_manifest.json](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/metrics.jsonl)、
  [train.log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/train.log)、
  [best checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/best.pt)、
  [latest checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_105619/checkpoints/latest.pt)。
- result: 两步 optimizer 与完整 pilot val 完成；真实 Cm shape 为 `[B,4,16,32]`，OICM 保持
  `eval/no-grad`，decoder 可反向；metadata 记录 OICM SHA256
  `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`。

实现过程中另产生四个终态为 `COMPLETED` 的预备 smoke，分别用于发现 provenance 缺项和验证修正；不作为
最终证据，但保留可审计产物：

- `..._102937`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_102937/checkpoints/best.pt)。
- `..._104051`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104051/checkpoints/best.pt)。
- `..._104215`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104215/checkpoints/best.pt)。
- `..._104811`：[目录](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/)、
  [config](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/config.json)、
  [manifest](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/run_manifest.json)、
  [metrics](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/metrics.jsonl)、
  [log](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/train.log)、
  [checkpoint](../../../../../outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_smoke_20260906_104811/checkpoints/best.pt)。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile $(find src/task/CmDecoderv2 -name '*.py' -type f | sort)`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmDecoderv2/tests`：`6 passed`；覆盖
  mimic/FK、`K+1` window、输出 shape、slot permutation invariance 与 identity rotation finite gradient。
- 单卡真实 checkpoint forward/backward：`pred_q_delta [1,4,6]`、translation/rotvec `[1,4,3]`；
  `oicm_training=False`、`oicm_grad_count=0`，有效 contact window 的四帧 `cm_sample_valid=True`。
- 完整 view 审计：所有 RL window arrays 为 `[T-4,5]`，所有 sidecar 链接存在，MANO test 不含 q/wrist GT。
- 正式数据加载：503 个 train sequence / 153602 windows 可构造；perturbation 同 epoch 可复现、跨 epoch
  变化，浮点张量全部 finite。batch=8（OICM flatten batch=32）单卡 backward 峰值 allocated `1.39 GiB`、
  reserved `2.57 GiB`。
- 正式 checkpoint 闸门负向测试：`LOCK_AFTER_OICM_TERMINAL` 未替换时按预期拒绝建模，避免误用动态 best。
- `git diff --check`：通过；`audit_diff.py --worktree --scope-prefix src/task/CmDecoderv2
  --scope-prefix docs/current_versions.yaml --check-links`：通过，覆盖 `24` 个变更路径和 `60` 个本地链接。

**未启动与回滚**

- 正式 CmDecoderv2 训练未启动：OICM 续训仍在 0/1/2 卡运行，当前 `best.pt` 只能作为 smoke 输入；必须等
  OICM 终态后冻结 checkpoint SHA256，再进入 Gate C。MANO rollout 代码已实现但遵循计划，需在正式 decoder
  训练后才执行 Gate D。
- 回滚入口为隔离/移走本 Task 目录、`data/processed_data/cm_decoder_v2/` 新 view 和
  `outputs/cmdecoderv2/` 新 smoke；无需也不得改动或删除旧 CmDecoder、OICM cache/checkpoint、原始数据或运行。

**规范反馈**

- 本次未遇到需要修改 AGENTS、Skill 或公共合同的阻碍。Task 级测试不在根 `pytest.ini` 的默认
  `testpaths` 中，因此按目录规范显式传入 `src/task/CmDecoderv2/tests`；无需改变共享 pytest 配置。
