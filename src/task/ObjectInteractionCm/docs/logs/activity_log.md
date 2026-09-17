# ObjectInteractionCm 活动记录

- scope: task:ObjectInteractionCm
- last_updated: 2026-09-12
- current_pointer: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- related: [任务入口](../README.md)、[执行计划](../plan/V1.1.md)、[架构快照](../architecture/V1.1.md)、[指导](../指导/V1.1.md)

## 2026-09-13 09:01:27 +0000 — ObjectInteractionCm 扩大训练数据可行性只读诊断

- activity_id: ACT-20260913-090127-OICM-DIAG
- timestamp: 2026-09-13 09:01:27 +0000
- modification_version: V1.3.2
- type: diagnostic
- operation_category: [diagnostic, documentation]
- change_level: L0
- approval: auto
- approval_basis: 用户澄清目标为 ObjectInteractionCm，并请求判断扩大 Cm 训练数据的可行性与必要性；本次仅阅读与只读查询
- skills_used: research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留本次会话已有的 Cm 活动记录改动）
- scope: ObjectInteractionCm 近期 Git 提交、V1.3 训练记录、数据 index/split/采样配置和跨源诊断；不修改模型、runner、dataset、GT、坐标、split、cache、checkpoint 或运行状态
- conclusion: SUPPORTED（可行性与已有证据复核）；INCONCLUSIVE（扩大现有帧数对效果的必要性）

**文件**
- [`../../../../../docs/logs/activity_log.md`](../../../../../docs/logs/activity_log.md) — 读取根级近期合并与版本指针记录。
- [`experiment_log.md`](experiment_log.md) — 读取 V1.3 全量训练、跨源 effect 和机制诊断结果。
- [`activity_log.md`](activity_log.md) — 读取 ObjectInteractionCm 近期训练/诊断活动。
- [`../configs/active/dexplore_rl_v1_3.yaml`](../../configs/active/dexplore_rl_v1_3.yaml) — 读取当前 V1.3 数据、采样、split 和训练预算。
- [`../plan/V1.3.md`](../plan/V1.3.md) — 读取数据合同与不变量。

**原因**
区分增加独立交互序列、增加同序列帧/stride 曝光和延长训练预算三种“扩容”，并依据当前 V1.3 已有样本覆盖、平台期与误差归因判断优先级。

**验证**
- `git log --date=iso --format='%h %ad %s' -n 30 -- src/task/ObjectInteractionCm`：确认最近变更集中于 V1.3 cache/训练终态、跨源 effect 诊断和机制诊断。
- 只读扫描 V1.3 记录与配置：确认 train `509`（MANO/Inspire-F1 `254/255`）、val `58`、test `63`，目标 `202300` steps，global batch `96`，source probability `0.5/0.5`。
- 既有 V1.3 结果：best equal-source object EPE `6.516671 mm`（step `132480`），最近 `6.714598 mm`（step `143520`），连续 15 次验证未刷新 best，训练尚未完成且已出现平台期迹象。
- 既有机制诊断：val `7670` 有效样本；低接触样本仅占总 EPE `12.49%/13.89%`，不能将主要瓶颈归因于简单缺样本。
- `git status --short --branch`：当前工作区仅保留会话内活动记录改动；数据路径为外部 NAS 软链接，未对实体 cache 做本机数量复核。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links`：本条写入后执行，链接与日志审计通过。

**回滚与规范反馈**

- 仅移除本 activity 条目即可回滚本次诊断记录；未产生代码、配置、数据或实验产物变更。
- 本次未遇到目录、版本或审批阻碍；由于目标从旧 Cm 更正为 ObjectInteractionCm，结论仅适用于本 Task。

## 2026-09-13 09:37:13 +0000 — ARCTIC 扩容与 CmDecoderv2 重定向链路可行性诊断

- activity_id: ACT-20260913-093713-OICM-ARCTIC-DECODER-DIAG
- timestamp: 2026-09-13 09:37:13 +0000
- modification_version: V1.3.2
- type: diagnostic
- operation_category: [diagnostic, documentation]
- change_level: L0
- approval: auto
- approval_basis: 用户询问加入 NAS 上 ARCTIC 等数据集的必要性，以及 ObjectInteractionCm 是否支持 CmDecoderv2 重定向；本次仅只读核对接口和既有结果
- skills_used: research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留会话内已有活动记录改动）
- scope: ObjectInteractionCm V1.3 cache/训练合同、CmDecoderv2 OICM 接口、Inspire point-flow decoder 训练与 rollout 诊断；不修改模型、数据、split、cache、checkpoint 或运行状态
- conclusion: SUPPORTED（接口与数据接入方向可行）；INCONCLUSIVE（“加入 ARCTIC 后得到较好重定向结果”）

**文件**
- [`experiment_log.md`](experiment_log.md) — 读取 ObjectInteractionCm V1.3 训练平台期、跨源 effect 与机制诊断。
- [`../../../CmDecoderv2/docs/logs/experiment_log.md`](../../../CmDecoderv2/docs/logs/experiment_log.md) — 读取 CmDecoderv2 V1.3 full10135 训练、teacher-forced 与递归 rollout 结果。
- [`../configs/active/dexplore_rl_v1_3.yaml`](../../configs/active/dexplore_rl_v1_3.yaml) — 读取 OI V1.3 的 4096/1024 object、KNN=32、2 cm、右手与 source split 合同。
- [`../../../CmDecoderv2/model.py`](../../../CmDecoderv2/model.py) — 读取 frozen OICM 加载、`cm_tokens`/anchor 窗口接口与 CmDecoderv2 调用路径。
- [`../指导/V1.3.md`](../指导/V1.3.md) — 读取 V1.3 固定 630 sequence、159476 frame、split/GT/坐标/hand-side 不变量。

**原因**
区分“ObjectInteractionCm 能否作为 CmDecoderv2 的编码器”与“整条链路能否在递归重定向中产生稳定效果”，并评估 ARCTIC 对交互覆盖、articulated object 和跨 embodiment 泛化的实际价值。

**验证**
- OI V1.3 模型输出 `cm_tokens [B,16,32]`、`cm_anchor_pos [B,16,3]`、`cm_anchor_normal [B,16,3]`；CmDecoderv2 将窗口展平后恢复为 `[B,4,16,32]`，接口形状和坐标合同一致。
- CmDecoderv2 已用冻结 OI V1.3 完成 50 epoch Inspire point-flow 训练；best point-flow EPE `12.8782 mm`，最终 `12.7156 mm`，工程链路 `SUPPORTED`，但只有 Inspire train/val，跨 embodiment 效果仍 `INCONCLUSIVE`。
- teacher-forced 诊断中，GT hand flow 直接进入 OI 时 effect EPE `0.660 mm`；说明 OI effect 映射具备有效信号，主要误差来自 decoder hand-state/hand-flow 与递归漂移。
- 接触起点递归诊断中，接触段 effect EPE `3.545 mm`，但 hand position EPE `87.909 mm`，后段 38/323 帧离开有效区；从 frame 0 的另一 run 中 OI valid ratio 为 `0`、hand EPE 均值约 `640.851 mm`。因此不能把当前重定向瓶颈归因于 OI 样本量不足。
- ARCTIC 不能直接塞入当前 V1.3：当前指导固定右手、630 sequence、159476 frame、object_pose_t 和现有 split；接入 ARCTIC 需生成同合同的 articulated-object cache、KNN/半径字段、source split 与 train-only scale，并同步 decoder 的目标手/状态数据视图。
- 本机 `data/processed_data` 仍是未挂载 NAS 软链接，未对 ARCTIC 实体数量做本机复核。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links`：本条写入后执行，链接与日志审计通过。

**回滚与规范反馈**

- 仅移除本 activity 条目即可回滚本次诊断记录；未产生代码、配置、数据或实验产物变更。
- 本次未遇到格式、目录、版本或审批阻碍；若要接入 ARCTIC 或改变 OI/CmDecoderv2 训练合同，需要新 plan 和用户确认。

## 2026-09-13 09:52:23 +0000 — RL 基础策略前的 rollout 偏移影响诊断

- activity_id: ACT-20260913-095223-OICM-RL-OFFSET-DIAG
- timestamp: 2026-09-13 09:52:23 +0000
- modification_version: V1.3.2
- type: diagnostic
- operation_category: [diagnostic, documentation]
- change_level: L0
- approval: auto
- approval_basis: 用户询问进入强化学习前 rollout 偏移的影响与当前优先问题；本次只读核对 RL 任务、base-policy 配置和既有 rollout 证据
- skills_used: research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留会话内已有活动记录改动）
- scope: CmDecoderv2 rollout、ObjectInteractionCm→CmDecoderv2 接口、IsaacGymEnvs CmResidual 任务与 RL decoder-bank 配置；不修改模型、数据、训练配置、checkpoint 或运行状态
- conclusion: SUPPORTED（偏移对当前 2 cm 有效区和 residual base 的影响判断）；INCONCLUSIVE（正式 RL 成功率与物理抓取效果）

**文件**
- [`../../../CmDecoderv2/docs/logs/experiment_log.md`](../../../CmDecoderv2/docs/logs/experiment_log.md) — 读取单步、短递归和接触起点 rollout 结果。
- [`../../../CmDecoderv2/model.py`](../../../CmDecoderv2/model.py) — 读取 OI frozen encoder、Cm window 和 decoder state 接口。
- [`../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py`](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py) — 读取当前 residual base、观测、动作和奖励实现。
- [`../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDecoderBank.yaml`](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDecoderBank.yaml) — 读取 decoder-bank base 配置。
- [`../../../CmDecoderv2/tools/rl/export_decoder_bank.py`](../../../CmDecoderv2/tools/rl/export_decoder_bank.py) — 读取 bank 的 teacher-forced、非在线闭环语义。

**原因**
判断 decoder 的时序偏移是否会让 residual PPO 在错误的基础轨迹上学习，并区分当前真正需要解决的状态预测、接触有效性、物理任务和 RL 工程问题。

**验证**
- 当前 `CmResidual` 只提供 `reference_frozen` 与 `decoder_bank` 两种 base mode；`decoder_bank` 由预先导出的 q/wrist 轨迹按 `progress_buf` 索引，bank manifest 标记 `online_closed_loop=false`，不是仿真内逐步调用 CmDecoderv2 的闭环 base。
- residual action 范围为独立 finger `±0.08`、wrist translation `±15 mm`、wrist rotation `±0.20 rad`；当前 OICM 交互有效半径为 `2 cm`。因此几十毫米级 hand-state 偏移可能直接把 OICM 置于无效区，超过 residual 一步可补偿范围。
- CmDecoderv2 既有 16 步短递归中，正确 Cm 的 hand EPE 从 `8.5667 mm` 增至 `53.1931 mm`；接触起点单序列中 hand position EPE `87.909 mm`，后段 `38/323` 帧离开有效区；从 frame 0 的另一 run 中 OICM valid ratio `0`、hand EPE 均值 `640.851 mm`。
- teacher-forced GT hand flow 进入同一 OICM 时 effect EPE `0.660 mm`，表明当前主要优先级是 decoder/state rollout 与物理闭环对齐，而不是先扩大 OI 数据或重写 Cm effect head。
- 当前 RL vendor smoke 只验证 action `(12,)`、observation `(71,)` 和单 PPO epoch；`rew=-inf` 是未完成 episode 的 smoke 统计伪影，尚无正式 RL 成功率证据。
- 当前配置的 reference 路径为 `s1_airplane_lift`，仿真 object asset 为 cube；正式 PPO 前必须确认 reference/object/reward 三者语义一致。
- 本机数据仍通过未挂载 NAS 软链接访问；本诊断不重新扫描数据或启动 rollout。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links`：本条写入后执行，链接与日志审计通过。

**回滚与规范反馈**

- 仅移除本 activity 条目即可回滚本次诊断记录；未产生代码、配置、数据或实验产物变更。
- 本次未遇到格式、目录、版本或审批阻碍；正式 RL 训练、在线 decoder 接入或改变 ARCTIC 数据合同都需另行形成 final plan 并确认。

## 2026-09-12 13:30:43 +0800 — Cm 后续路线的仓库复核与定向文献调研

- activity_id: ACT-20260912-133043-OICM-RESEARCH-REVIEW
- timestamp: 2026-09-12 13:30:43 +0800
- modification_version: V1.3.2
- task_mode: read-only/diagnostic
- type: diagnostic
- operation_category: [diagnostic, documentation]
- change_level: L0
- approval: auto
- approval_basis: 用户要求浏览仓库并调研/探索 Cm 效果和数据瓶颈；本次只读复核已有产物及文献，不扩展训练或核心实现授权。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: e7df6b46e9a3009a5b6e07c41bad5216d032a607
- worktree_dirty: true（保留进入本次请求前的全部差异；仅追加本条记录）
- scope: ObjectInteractionCm V1.3 既有结果重聚合、模型输入审阅、CmDecoderv2 历史证据与研究建议。
- run_id: repository_review_20260912_133043
- run_status: COMPLETED
- conclusion: SUPPORTED（描述性统计复核）；INCONCLUSIVE（压缩/条件信息/数据规模/迁移机制）。
- last_step: N/A；last_epoch: N/A；best_metric: N/A；checkpoint: N/A（无新训练或推理）。
- exit_reason: 正常完成只读调查；训练与科学变量变更仍需最终计划。

**文件**

- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 只追加本次诊断事件，不修改历史条目。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/repository_review_20260912_133043](../../research/cross_source_effect/output/repository_review_20260912_133043) — 新诊断目录，已忽略。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/repository_review_20260912_133043/report.md](../../research/cross_source_effect/output/repository_review_20260912_133043/report.md) — 证据、文献链接、建议及其适用边界。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/repository_review_20260912_133043/observations.json](../../research/cross_source_effect/output/repository_review_20260912_133043/observations.json) — 原指标/index 重聚合与关键帧入口。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/repository_review_20260912_133043/run_manifest.json](../../research/cross_source_effect/output/repository_review_20260912_133043/run_manifest.json) — 本次追溯入口。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/repository_review_20260912_133043/config.json](../../research/cross_source_effect/output/repository_review_20260912_133043/config.json) 与 [src/task/ObjectInteractionCm/research/cross_source_effect/output/repository_review_20260912_133043/metadata.json](../../research/cross_source_effect/output/repository_review_20260912_133043/metadata.json) — 操作范围和输入摘要。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl) 与 [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json) — 只读源证据。

**原因**

将效果平台期拆分为模型对可用运动信息的利用、缺失状态条件、训练覆盖与递归控制误差；提出能够区分这些解释的下一项实验，避免盲目延长训练或扩数据。

**验证**

- Python 标准库只读读取原 index 和 7670 行 metrics，按 source/sequence_id 聚合；5 项源内均值与原 diagnosis.json 差小于 1e-9 mm，primary_valid 全 true。
- train/val/test parent_seq_id 两两交集为 0；val/test object_name 相对 train 的差集为空。只读计数不改变 split 或其语义。
- MANO 两条最高 MSE 序列贡献约 66.2% 平方误差；flashlight_lift frame 327 的 Cm/rigid-hand EPE 为 270.54/4.51 mm。均为后验局部定位，不证明全局病因或物理 GT 正确。
- 验证命令：python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links
- 验证命令：git diff --check -- src/task/ObjectInteractionCm/docs/logs/activity_log.md；另核对诊断 JSON、报告本地链接和进入写入前已有差异的 SHA256。
- 本次无模型 smoke 或正式训练效果结果；未修改 experiment_log 科研结论或版本指针。
- 交接复核通过：最新 activity 的 9 个本地链接、报告 12 个本地链接和 4 个 JSON 均有效；15 个既有文件 SHA256 不变，移除本条后历史 activity 字节摘要与写入前完全一致。审计脚本按合同排除 activity 自身，因本次仅增加日志及 ignored 产物，其代码变更路径计数为 0；未将该计数解释为全工作树 diff 审计。

**回滚**

仅移除本 activity_id 条目与上述新输出目录；其余既有差异、数据、配置、checkpoint、指导、架构和运行无需恢复。

**规范反馈**

本次未遇到格式、目录、版本或审批阻碍。dirty 工作树的审计仅限定本次 activity 路径，并单独核对既有文件摘要，避免把其他工作纳入本次审计范围；未修改治理规则。


## 2026-09-12 13:22:38 +0800 — V1.3.2 机制诊断完成，修正低接触主因判断

- activity_id: `ACT-20260912-132238-OICM-MECHANISM-END`
- timestamp: `2026-09-12 13:22:38 +0800`
- modification_version: `V1.3.2`
- type: `experiment_run / code_change`
- operation_category: `[code, diagnostic, experiment, operation, documentation]`
- change_level: `L3`（局部诊断定义 L2，运行保守 L3）
- approval: `user-approved`
- approval_basis: 用户授权自主探索并要求继续；在现有 final 范围追加无学习机制诊断，不启动网络训练。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`；保留 V1.3.1 全部未提交差异，无 Git 提交。
- scope: 新增诊断脚本/定义/测试，维护 README、现有计划范围、Task 指针和日志。主模型/runner/dataset、GT、单位/坐标、cache、split、训练配置、checkpoint、指导、架构和旧输出均未修改。
- run_id: `mechanism_v1_3_2_val_20260912_131550`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（相对平均手位移有有限增益、数据刚性/工程检查）；`REFUTED`（低接触贡献大部分总 EPE、effective_count 可证明 slot 利用率的解释）；`INCONCLUSIVE`（Cm 压缩必要性及跨手型表征）。
- last_step: `N/A`；last_epoch: `N/A`；best_metric: `N/A`（没有训练或新 checkpoint）。
- exit_code: `0`；exit_reason: `正常完成`；elapsed: `49.1 s`；outputs: `12103557 bytes`。
- command: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.diagnose --run-id mechanism_v1_3_2_val_20260912_131550`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针从 V1.3.1 到 V1.3.2。
- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) — 沿用现有 final，追加第 8 节最小诊断边界；没有新建 plan。
- [src/task/ObjectInteractionCm/research/cross_source_effect/diagnose.py](../../research/cross_source_effect/diagnose.py) — 无学习运动基线、刚性、slot 谱、误差份额和 bootstrap。
- [src/task/ObjectInteractionCm/research/cross_source_effect/diagnostic.yaml](../../research/cross_source_effect/diagnostic.yaml) — exploratory 诊断定义。
- [src/task/ObjectInteractionCm/tests/test_effect_diagnostics.py](../../tests/test_effect_diagnostics.py) — 合成/退化/反射、独立 SVD、配对 bootstrap、slot 指标反例和误差份额。
- [src/task/ObjectInteractionCm/research/cross_source_effect/README.md](../../research/cross_source_effect/README.md) — 新增诊断入口、符号与 oracle 限制。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本次启动/终态。
- [src/task/ObjectInteractionCm/docs/logs/experiment_log.md](experiment_log.md) — 完整数值、结论及对上一轮解释的明确修正。
- [src/task/ObjectInteractionCm/research/cross_source_effect/run.py](../../research/cross_source_effect/run.py)、[src/task/ObjectInteractionCm/research/cross_source_effect/verify.py](../../research/cross_source_effect/verify.py)、[src/task/ObjectInteractionCm/research/cross_source_effect/experiment.yaml](../../research/cross_source_effect/experiment.yaml)、[src/task/ObjectInteractionCm/tests/test_cross_source_effect.py](../../tests/test_cross_source_effect.py) — 仅保留 V1.3.1 未提交工作区差异，本轮未改。

**原因**

先用可计算的误差份额检验低接触假设，再比较弱零流之外的无学习基线。避免因均值/分层相关性直接扩充
null 或刚体结构，也避免把 source-heldout 失败归因于未经定位的 Cm 瓶颈。

**验证**

- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCm/tests -q`：最终 `18 passed`。
- smoke `mechanism_v1_3_2_smoke_20260912_131600` 的 6 行与 full 对应行完全一致；独立 NumPy SVD
  对两源 12 个真实样本复核刚体拟合，EPE 最大差 `2.79e-13 mm`；独立重算全部分源均值一致。
- 58 条 val sequence、7670 有效样本（MANO 3610 / Inspire RL 4060），raw/future frame ID、stride=2、
  抽样 geometry/GT、hand-valid 点数均与原运行严格对应；所有值有限；原输入摘要与几何 stat 不变。
- 无学习平均手位移 EPE `8.4420 / 7.1632 mm`，完整 Cm 为 `7.2907 / 5.7426 mm`。改善 `1.1513 / 1.4206 mm`，
  paired sequence CI `[0.5037,1.8441] / [0.8449,2.1415]`。平均手位移已达到原 zero-flow gain 的约 `94.9% / 92.5%`。
- 无学习刚体手拟合 EPE `7.7724 / 7.9028 mm`。MANO 上 Cm 的改善 CI 跨零，不能证明优于或等价。
- active<0.1 仅占总 EPE 的 `12.49% / 13.89%`；置零只使全体 EPE 改善 `0.1023 / 0.1768 mm`。
  明确修正 `ACT-20260912-122139-OICM-CROSS-SOURCE-SPARSITY-DIAG` 中“主要来自低接触、优先 null”的判断。
- pred 刚体投影仅改善 `0.0065 / 0.0100 mm`；GT 刚体拟合残差约 `6.3e-5 mm`。这只核对刚性，未证明物理 GT 全部正确。
- `runner.py` 的 slot-weight 均值归一化使 `effective_count` 几乎恒等于 16，不能判断 slot 利用率；只定位未修改。
  当前 tokens 的跨 slot 标准差/整体 RMS 约 `0.37`，不属于所有 tokens 数值相同；谱秩不等同语义 slot 数。
- train 最后 EPE=8.1995 mm、val=6.6839 mm，但 stride=1..10 vs 2，不能直接判断训练/泛化差距。
- 本轮没有训练 geometry-only、null 或 no-Cm 模型。下一学习对照应优先使用相同输入与局部交互 encoder 的
  无 Cm 压缩 effect predictor，统一训练/验证评估 horizon，再定位瓶颈还是输入/监督；这是后续工作，不冒充已验证结果。
- `git diff --check` 通过；`python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --check-links` 通过，11 个受审差异路径一致、20 个本地链接可导航；staged diff 为空。

**产物**

- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run_manifest.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/config.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/config.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metadata.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metadata.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run.log](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run.log)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/diagnosis.json)
- [outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt) — 原冻结预测来源，本轮只校验摘要，无新加载推理/训练。

**回滚与规范反馈**

- 可独立撤销本次 3 个新增诊断/测试文件以及 README、第 8 节、版本/日志增量；保留原 V1.3.1 差异和旧产物。
  新 output 可单独移除，主模型/数据/checkpoint 无写入，无需恢复。
- 目录、链接和审批无未解决阻碍；审计要求将“验证与结果”标题写成“验证”，已局部修正，无科研影响。
  按用户“不必新写计划”的要求沿用现有 final 范围，不新建 plan，未改治理规则。

## 2026-09-12 13:15:50 +0800 — V1.3.2 无学习机制对照启动

- activity_id: `ACT-20260912-131550-OICM-MECHANISM-START`
- timestamp: `2026-09-12 13:15:50 +0800`
- modification_version: `V1.3.2`
- type: `experiment_run`
- operation_category: `[code, diagnostic, experiment, operation, documentation]`
- change_level: `L3`（局部诊断定义 L2，运行保守 L3）
- approval: `user-approved`
- approval_basis: 用户授权自主探索并要求继续；本次先开展无学习基线与机制诊断，不启动网络训练，不改科研目标或保护边界。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`（保留 V1.3.1 未提交差异）
- scope: 原冻结结果与相同 Dataset 逐样本重放；比较平均手位移、手刚体拟合、pred 刚体投影，检查 GT 刚性、误差归因与 slot 指标。
- run_id: `mechanism_v1_3_2_val_20260912_131550`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`
- command: `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.diagnose --run-id mechanism_v1_3_2_val_20260912_131550`

**文件**

- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) — 现有计划第 8 节记录最小执行范围，不另建 plan。
- [src/task/ObjectInteractionCm/research/cross_source_effect/diagnose.py](../../research/cross_source_effect/diagnose.py)
- [src/task/ObjectInteractionCm/research/cross_source_effect/diagnostic.yaml](../../research/cross_source_effect/diagnostic.yaml)
- [src/task/ObjectInteractionCm/tests/test_effect_diagnostics.py](../../tests/test_effect_diagnostics.py)
- [src/task/ObjectInteractionCm/research/cross_source_effect/README.md](../../research/cross_source_effect/README.md)
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run_manifest.json](../../research/cross_source_effect/output/mechanism_v1_3_2_val_20260912_131550/run_manifest.json)

**原因**

上一轮从低接触区间误差高直接跳到 null 分支，未量化它能改善多少总体误差；本次先做误差份额和无学习强基线。
同时代码检查发现 `slot/effective_count` 由各 slot 已归一化权重的均值计算，几乎恒为 S，不能用作 slot 利用率证据。

**验证**

- `CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCm/tests -q`：17 passed。
- smoke_run_id: `mechanism_v1_3_2_smoke_20260912_131600`；smoke_run_status: `COMPLETED`；命令同上诊断入口，替换 run-id 并加 `--smoke`，6 样本、退出码 0、约 1 秒，sample/geometry/GT 对齐和输入摘要通过。仅工程 SUPPORTED。
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/](../../research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/run_manifest.json](../../research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/metrics.jsonl](../../research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/run.log](../../research/cross_source_effect/output/mechanism_v1_3_2_smoke_20260912_131600/run.log)
- CPU、最多 15 分钟/100 MiB；无 optimizer，无新 checkpoint，原模型/缓存/GT/split/旧输出不写入。

## 2026-09-12 12:21:39 +0800 — V1.3.1 结果的接触稀疏性后验诊断

- activity_id: `ACT-20260912-122139-OICM-CROSS-SOURCE-SPARSITY-DIAG`
- timestamp: `2026-09-12 12:21:39 +0800`
- modification_version: `V1.3.1`
- type: `diagnostic`
- operation_category: `[diagnostic]`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户授权自主探索；本次只读分析已有冻结运行产物，不改代码、配置、数据、cache、split 或 checkpoint。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`（沿用已记录的本次实验差异）
- scope: 分析 `cross_source_effect_v1_3_1_val_20260912_112900` 的每样本 EPE、zero-flow、active fraction、flow RMS 和空间置换结果；不产生新运行目录。
- source_run_id: `cross_source_effect_v1_3_1_val_20260912_112900`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（后验诊断，不改变冻结实验的主要结论）

**文件**

- [冻结运行逐样本指标](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metrics.jsonl)
- [冻结运行统计](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_summary.json)
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md)
- [src/task/ObjectInteractionCm/research/cross_source_effect/README.md](../../research/cross_source_effect/README.md)
- [src/task/ObjectInteractionCm/research/cross_source_effect/experiment.yaml](../../research/cross_source_effect/experiment.yaml)
- [src/task/ObjectInteractionCm/research/cross_source_effect/run.py](../../research/cross_source_effect/run.py)
- [src/task/ObjectInteractionCm/research/cross_source_effect/verify.py](../../research/cross_source_effect/verify.py)
- [src/task/ObjectInteractionCm/tests/test_cross_source_effect.py](../../tests/test_cross_source_effect.py)
- [实验记录](experiment_log.md)
- [活动记录](activity_log.md)

**原因**

判断当前“效果差”主要来自 Cm 本身，还是来自低接触/低运动样本的失效；为 source-heldout 之前确定真正的下一项实验。

**验证**

- 只读命令按 `active_fraction` 分箱重算 full Cm、zero-flow、预测优于 zero-flow 的比例和后验 gate；未修改任何输入。
- MANO：active `<0.1` 的 240/3610 样本 EPE `10.39–14.84 mm`，zero-flow `7.22–15.59 mm`，平均反而更差；active `0.1–0.5` 时 EPE `7.84/7.03 mm`，zero-flow `28.83/32.33 mm`。
- Inspire RL：active `<0.1` 的 200/4060 样本 EPE `14.17–20.34 mm`，zero-flow `11.02–17.03 mm`，平均反而更差；active `0.25–0.5` 与 `>0.5` 时 EPE `5.63/3.57 mm`，zero-flow `26.75/25.30 mm`。
- 中位 EPE 仅 MANO `4.01 mm`、Inspire `2.49 mm`，但 p99 为 `69.62/59.46 mm`；均值主要受少量失败样本和接触稀疏区间影响。
- EPE 与 hand-flow RMS 相关系数 MANO/Inspire 为 `0.502/0.328`；active fraction 相关系数为 `-0.147/-0.243`。这支持先处理稀疏接触和长尾，而不是直接把差距解释为跨手型差异。
- 后验仅在 val 上观察，不能作为已训练 gate 的效果或正式泛化结论。

**判断**

当前最有信息量的下一项是训练一个输入分布内的 **null/low-contact-aware control**，并同时做 Full vs geometry-only 归因；source-heldout 暂缓。配对同-effect 实验暂不开展。

## 2026-09-12 11:35:10 +0800 — V1.3.1 冻结跨源 effect 诊断完成与独立复核

- activity_id: `ACT-20260912-113510-OICM-CROSS-SOURCE-END`
- timestamp: `2026-09-12 11:35:10 +0800`
- modification_version: `V1.3.1`
- type: `experiment_run / code_change`
- operation_category: `[code, diagnostic, experiment, operation, documentation]`
- change_level: `L3`（研究对照/指标 L2，运行保守按 L3）
- approval: `user-approved`
- approval_basis: 用户批准第一阶段冻结跨源 effect 实验，并明确“可以，你直接开始吧”；未授权第二阶段重训。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`（9 个本次相关代码/文档路径，未提交；运行产物已忽略）
- scope: 独立 cross_source_effect 诊断及测试、Task 计划/活动/实验和 Task 版本指针；未修改核心 model/dataset、src/base、GT、坐标/单位、cache、split、训练配置、checkpoint、指导或架构。
- final_plan: [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) 第 7 节。
- run_id: `cross_source_effect_v1_3_1_val_20260912_112900`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（跨手型等价/未见手型泛化）；`SUPPORTED`（本验证集两源平均重建优于零流，以及工程检查）。
- last_step: `N/A`（冻结诊断，无训练）；last_epoch: `N/A`；best_metric: `N/A`（没有优化或新 checkpoint）。
- loaded_checkpoint_step: `132480`；loaded_checkpoint_epoch: `180`；loaded_best_metric: `6.5166713276 mm`。
- exit_code: `0`；exit_reason: `正常完成`；推理/统计/绘图约 `149.3 s`，产物约 `870.15 MiB`。
- command: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.run --run-id cross_source_effect_v1_3_1_val_20260912_112900`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 仅 ObjectInteractionCm 指针更新为 V1.3.1。
- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) — 新增已批准 final 的冻结诊断节；文档更正 Dataset fixed-stride 产生 T-2 行，不改变实现。
- [src/task/ObjectInteractionCm/research/cross_source_effect/run.py](../../research/cross_source_effect/run.py) — 同 checkpoint 冻结推理、三种对照、固定难度匹配、sequence cluster bootstrap、原始张量归档。
- [src/task/ObjectInteractionCm/research/cross_source_effect/verify.py](../../research/cross_source_effect/verify.py) — 独立 NumPy 重算全部归档预测的误差、有效性、匹配与输入/代码 SHA256。
- [src/task/ObjectInteractionCm/research/cross_source_effect/experiment.yaml](../../research/cross_source_effect/experiment.yaml) — primary/pinned 实验定义。
- [src/task/ObjectInteractionCm/research/cross_source_effect/README.md](../../research/cross_source_effect/README.md) — 入口、指标、产物与结论边界。
- [src/task/ObjectInteractionCm/tests/test_cross_source_effect.py](../../tests/test_cross_source_effect.py) — 单位/掩码、置换、匹配、cluster bootstrap 和 decoder 重放测试。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本次开始、smoke、首轮终态、复跑和终态。
- [src/task/ObjectInteractionCm/docs/logs/experiment_log.md](experiment_log.md) — 定量证据、CI、失败序列及结论边界。

**原因**

从已有 mixed-source shared decoder 的分源验证进一步检查对照、运动信息和源差距。完整 Cm 包含
tokens 与 anchors，decoder 还读取物体几何；验证“共享解码有效”不能替代“跨手型表征等价”。
首次运行发现 `config:split -> val` 的通用 manifest 路径误判，保留原输出，只在本实验中将标签
改名为 `evaluation_partition` 后复跑；不改共享合同、研究变量或原始结果。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCm/tests -q`：`9 passed`，包括 5 个本次定向测试与 4 个 V1.3 KNN 测试。
- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.verify src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900`：58 NPZ / 14622 行 / 916 matched 全部通过，NumPy/Torch EPE 最大差 `0.00002693 mm`，manifest 引用存在、SHA256 不变。
- 全部 58 条 val sequence、seed=42、epoch=0、stride=2；7670 有效（MANO 3610 / Inspire 4060），matched 458+458、11 类、115 strata、14+12 序列。无 train/test 指标混入。
- 原 decoder 重放最大绝对差 `0 m`；GT 不进入模型；finite、模型 state/梯度、checkpoint/index/scale 摘要和数据文件 stat 检查通过。
- 完整 Cm EPE `7.290740 / 5.742603 mm`；零流 `29.962009 / 24.785472 mm`；空间置换 `8.882639 / 6.232415 mm`；零-token `47.044641 / 43.269641 mm`。equal-source `6.5166713434 mm` 复现上游 best metric。
- matched EPE `5.735400 / 6.817376 mm`，源差 CI `[-4.7626, 2.3963] mm`；未设置等价容差，也非同 effect 配对，因此等价结论保持 INCONCLUSIVE。
- 主有效集仍有 MANO 1/28、Inspire 6/30 条序列不优于零流；零-token 为 OOD/保留 anchors 消融，不宣称它证明 token-only 充分性。
- 与首轮相比 `metrics.jsonl`、`matched_selection.json` 逐字节一致，primary/matched/coverage/checks JSON 子树完全一致。
- `git diff --check` 通过；`python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --check-links` 通过，8 个受审变更路径与最新 activity 一致，20 个本地链接可导航。`git diff --cached --stat` 为空，HEAD 未变化。

**产物**

- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/config.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/config.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metadata.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metadata.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metrics.jsonl](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run.log](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run.log)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_summary.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_summary.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/matched_selection.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/matched_selection.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_comparison.png](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/effect_comparison.png)
- [outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt) — 仅只读载入，无新 checkpoint。

**回滚与规范反馈**

- 仅需撤销上述 9 个本次相关代码/文档差异并按需独立移除新 output；现有数据、训练输出和 checkpoint 无写入，无需回滚。未提交 Git。
- 阻碍：通用 manifest 以字段名启发式识别路径，将 `split: val` 误当输入文件；workaround 为局部改名 `evaluation_partition`，保留首次输出并复跑，额外约 150 秒与 870 MiB。
- 风险：标签与路径混淆导致错误缺失告警和重复运行。建议今后为共享 manifest 增加显式路径引用接口或约定分区标签字段，保留旧路径启发式兼容；本次未改 `src/base`、AGENTS、Skill 或公共合同，实施建议需用户另行确认。

## 2026-09-12 11:29:01 +0800 — V1.3.1 manifest 标签修复后复跑

- activity_id: `ACT-20260912-112901-OICM-CROSS-SOURCE-RERUN`
- timestamp: `2026-09-12 11:29:01 +0800`
- modification_version: `V1.3.1`
- type: `experiment_run`
- operation_category: `[code, diagnostic, experiment, operation, documentation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 已批准第一阶段范围内的本地 provenance 修正，不改科研变量或共享 manifest 合同。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`
- scope: 本实验 config 将 `split` 标签改名 `evaluation_partition`，避免通用 manifest 路径启发式误判；新增独立 NPZ 验证入口；修正文档对现有 fixed-stride 行范围的描述，采样代码不变。
- run_id: `cross_source_effect_v1_3_1_val_20260912_112900`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`
- command: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.run --run-id cross_source_effect_v1_3_1_val_20260912_112900`

**文件**

- [src/task/ObjectInteractionCm/research/cross_source_effect/run.py](../../research/cross_source_effect/run.py)
- [src/task/ObjectInteractionCm/research/cross_source_effect/verify.py](../../research/cross_source_effect/verify.py)
- [src/task/ObjectInteractionCm/research/cross_source_effect/README.md](../../research/cross_source_effect/README.md)
- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md)
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112900/run_manifest.json)

**原因**

首次完整运行数值有效，但 manifest 把字符串 `val` 误报为不存在的输入文件。保留首轮产物并使用新目录复跑，不覆盖旧 evidence。

**验证**

- 首轮已精确复现 best checkpoint 两源验证 EPE，9 个 Task 测试通过。
- 新运行额外核对所有 manifest 输入引用存在、逐 NPZ 重算误差/掩码/匹配和完整 T-2 行覆盖。
- 30 分钟/3 GiB 单 run 预算保持不变；保护所有已有训练进程与数据。

## 2026-09-12 11:29:00 +0800 — V1.3.1 首轮全验证集诊断完成，待修正 manifest 标签

- activity_id: `ACT-20260912-112900-OICM-CROSS-SOURCE-FIRST-END`
- timestamp: `2026-09-12 11:29:00 +0800`
- modification_version: `V1.3.1`
- type: `operation`
- operation_category: `[diagnostic, experiment, operation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 已批准冻结诊断。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`
- scope: 58 条 val sequence 的冻结推理与预定匹配。
- run_id: `cross_source_effect_v1_3_1_val_20260912_112331`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（跨手型等价未证实；数值检查通过，manifest 标签需修正）
- last_step: `N/A`；last_epoch: `N/A`；best_metric: `N/A`（冻结诊断无训练/新 checkpoint）。
- command: 见 `ACT-20260912-112331-OICM-CROSS-SOURCE-START` 的完整命令。

**文件**

- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/metrics.jsonl](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/run.log](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/run.log)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/effect_summary.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/effect_summary.json)

**原因**

记录首轮实际终态，并明确 provenance 瑕疵与模型效果是不同问题。

**验证**

- 退出码 0，约 150 秒，输出 870.15 MiB；14622 行、7670 有效，matched 916 行。
- 两源 EPE `7.290740 / 5.742603 mm`，equal-source `6.516671 mm`，复现 checkpoint best metric。
- 所有 finite、state/input unchanged 通过；decoder replay 最大绝对误差 `0 m`。
- manifest 中仅 `config:split -> val` 被误识别为缺失路径；已决定只修正本实验字段名称并复跑，不修改 `src/base`。

## 2026-09-12 11:23:31 +0800 — V1.3.1 全验证集冻结诊断启动

- activity_id: `ACT-20260912-112331-OICM-CROSS-SOURCE-START`
- timestamp: `2026-09-12 11:23:31 +0800`
- modification_version: `V1.3.1`
- type: `experiment_run`
- operation_category: `[diagnostic, experiment, operation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户批准第一阶段冻结评估；双源 smoke 已通过，按 final plan 执行全部 val。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`
- scope: 58 条 val sequence，单物理 GPU 2；不重训、不改 cache/GT/split/权重。
- run_id: `cross_source_effect_v1_3_1_val_20260912_112331`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`
- command: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.run --run-id cross_source_effect_v1_3_1_val_20260912_112331`

**文件**

- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md)
- [src/task/ObjectInteractionCm/research/cross_source_effect/run.py](../../research/cross_source_effect/run.py)
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_val_20260912_112331/run_manifest.json)

**原因**

在全部验证序列上建立源差距、零流/空间打乱/零-token 对照和预定匹配分析证据。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ObjectInteractionCm/tests/test_cross_source_effect.py -q`：5 passed。
- 双源 smoke 16 行、14 有效，decoder 重放/finite/state/input checks 通过；工程证据不能证明效果等价。
- 预算单 GPU、30 分钟/3 GiB；超预算或异常终止并保留独立产物。

## 2026-09-12 11:23:30 +0800 — V1.3.1 双源 smoke 完成

- activity_id: `ACT-20260912-112330-OICM-CROSS-SOURCE-SMOKE-END`
- timestamp: `2026-09-12 11:23:30 +0800`
- modification_version: `V1.3.1`
- type: `operation`
- operation_category: `[diagnostic, operation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 已批准的冻结诊断之双源 wiring 验证。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`
- scope: 两源各一条序列；不用于科研效果判断。
- run_id: `cross_source_effect_v1_3_1_smoke_20260912_112231`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅工程 smoke）
- last_step: `N/A`；last_epoch: `N/A`；best_metric: `N/A`（无优化，无新 checkpoint）。
- command: 与 `ACT-20260912-112231-OICM-CROSS-SOURCE-SMOKE` 相同。

**文件**

- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/run_manifest.json)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/metrics.jsonl](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/run.log](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/run.log)
- [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/effect_summary.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/effect_summary.json)

**原因**

正式运行前核对冻结权重、decoder replay、空交互排除和对照接线。

**验证**

- 退出码 0；16 行中 14 有效，全部有限；decoder replay、state 与输入保护检查通过；耗时约 2.7 秒。
- 无 matched 样本（smoke 两序列 object 不同），属于预期，不扩大为科研结论。

## 2026-09-12 11:22:31 +0800 — V1.3.1 冻结跨源 effect 诊断 smoke 启动

- activity_id: `ACT-20260912-112231-OICM-CROSS-SOURCE-SMOKE`
- timestamp: `2026-09-12 11:22:31 +0800`
- modification_version: `V1.3.1`
- type: `experiment_run`
- operation_category: `[code, diagnostic, experiment, operation, documentation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认第一阶段冻结诊断并明确“可以，你直接开始吧”；不授权 source-heldout 重训。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e7df6b46e9a3009a5b6e07c41bad5216d032a607`
- worktree_dirty: `true`（仅本次独立实验与文档）
- scope: Task-local 冻结 checkpoint 双源 effect 诊断；保护核心 model/dataset、src/base、GT、split、cache、权重、指导和架构。
- run_id: `cross_source_effect_v1_3_1_smoke_20260912_112231`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`（smoke 尚未完成，且不作为科研效果证据）
- command: `CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.research.cross_source_effect.run --run-id cross_source_effect_v1_3_1_smoke_20260912_112231 --smoke`

**文件**

- [src/task/ObjectInteractionCm/docs/plan/V1.3.md](../plan/V1.3.md) — 第 7 节已批准 final 方案。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — Task 指针 V1.3.1。
- [src/task/ObjectInteractionCm/research/cross_source_effect/run.py](../../research/cross_source_effect/run.py) — 冻结评估、对照、匹配和 sequence bootstrap。
- [src/task/ObjectInteractionCm/research/cross_source_effect/experiment.yaml](../../research/cross_source_effect/experiment.yaml)
- [src/task/ObjectInteractionCm/research/cross_source_effect/README.md](../../research/cross_source_effect/README.md)
- [src/task/ObjectInteractionCm/tests/test_cross_source_effect.py](../../tests/test_cross_source_effect.py)
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/)
- PENDING [src/task/ObjectInteractionCm/research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/run_manifest.json](../../research/cross_source_effect/output/cross_source_effect_v1_3_1_smoke_20260912_112231/run_manifest.json)

**原因**

补齐共享 decoder 的显式双源对照与难度匹配诊断。原训练已有共享 decoder 分源验证，但不能据此证明跨手型等价。

**验证**

- 已完成静态 `git diff --check`；定向 pytest 执行中。
- smoke 使用物理 GPU 2、batch=16、workers=2，两个 source 各一条 sequence 的最多 8 个样本。
- 回滚入口为本次新增 research/tests、计划第 7 节、版本指针和日志差异；不动已有训练/数据产物。

## 2026-09-10 08:34:42 +0800 — V1.3 全量训练收敛状态检查

- activity_id: `ACT-20260910-083442-OBJECTINTERACTIONCM-V13-FULL-TRAIN-CONVERGENCE`
- timestamp: `2026-09-10 08:34:42 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[operation, diagnostic]`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 V1.3 全量训练是否收敛；本次只读检查训练进程、metrics、验证曲线、checkpoint 和日志，未修改代码、配置、数据、cache 或 checkpoint
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `caf3f4fb70a710047c0bb1a418e49f23ca8646a0`
- worktree: `false`
- run_id: `object_interaction_cm_dexplore_rl_v1_3_20260910_020856`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（工程上已进入验证平台期，但训练尚未完成，不能据此宣称最终收敛或科研假设成立）
- scope: 判断 V1.3 full training 的 object validation plateau、GRAB/Inspire-F1 分源趋势、unique-KNN hand validation 和剩余训练进度。

**文件**

- [活动记录](activity_log.md)
- [实验记录](experiment_log.md)
- [正式训练输出目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/)
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/metrics.jsonl)
- [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/train.log)
- [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)
- [latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/latest.pt)

**原因**

区分“验证指标已经基本不再持续下降”和“正式训练已达到终态”。当前训练计划仍为
`202300 steps`，需要结合最近验证窗口和分源指标判断是否已经进入平台期。

**验证**

- 检查时间：`2026-09-10 08:34:42 +0800`；主 `torchrun` PID `1771994` 和三个 rank
  `1772101/1772102/1772103` 仍在运行。
- 最新训练进度：`step=144000`、`epoch=196`，约完成 `71.2%`；最新 ETA 约 `2.38 h`，吞吐约
  `653.63 samples/s`。
- equal-source object validation 最佳为 `6.516671 mm`，位于 `epoch=180 / step=132480`；
  最近验证为 `6.714598 mm`，位于 `epoch=195 / step=143520`。
- 最佳之后的 15 次验证中，equal-source 指标范围为 `6.574918–6.805946 mm`，均未超过
  `6.516671 mm` 的历史最佳；最近 15 次均值 `6.700502 mm`、标准差 `0.063896 mm`，
  表明 object 指标已在约 `0.1 mm` 量级内波动，属于明显平台期。
- 分源最近/最佳：GRAB 最近 `7.658820 mm`、最佳 `7.239350 mm`（epoch 146）；Inspire-F1
  最近 `5.770376 mm`、最佳 `5.701175 mm`（epoch 179）。两源都没有在最近阶段形成持续下降。
- unique-KNN hand validation 最近为 `1.921042 mm`，为当前全程最佳，说明 hand 支路仍有局部改善，
  但 object 主指标尚未继续刷新 best。
- 训练进程、显存和日志检查未发现 `traceback`、OOM、NaN、NCCL 或其他错误。

**判断**

当前应标记为“object 指标基本收敛/进入平台期，正式 run 尚未完成”，继续保留训练到计划的
`202300 steps`；最终是否采用 `best.pt` 仍以完整训练后的最终验证结果和下游评估为准。

## 2026-09-10 08:30:16 +0800 — V1.3 全量训练运行中状态检查

- activity_id: `ACT-20260910-083016-OBJECTINTERACTIONCM-V13-FULL-TRAIN-STATUS`
- timestamp: `2026-09-10 08:30:16 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[operation, diagnostic]`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 V1.3 全量训练当前状态；本次只读检查进程、GPU、日志、metrics 和 checkpoint，未修改训练配置、代码、数据、cache 或 checkpoint
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e6998f31ad37587f56bdc6bcf9863d4001ee4c46`
- worktree_dirty: `false`
- run_id: `object_interaction_cm_dexplore_rl_v1_3_20260910_020856`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（运行尚未完成；当前只确认进程和中间指标）
- scope: 查询 V1.3 full training 的当前 step、epoch、ETA、validation metric、checkpoint 和错误关键词；训练仍使用 launch 时记录的代码基线 `158e0f34068e260d7077a94b4457799b7fca2f31`。

**文件**

- [活动记录](activity_log.md) — 追加本次状态检查记录。
- [正式训练输出目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/)
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/metrics.jsonl)
- [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt)
- [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/latest.pt)

**原因**

用户需要确认全量训练是否仍在正常推进、是否出现错误，以及当前距离完成还有多久。

**验证**

- 时间：`2026-09-10 08:30:16 +0800`；主 `torchrun` PID `1771994` 和三个 rank PID `1772101/1772102/1772103` 均仍在运行，已运行约 `06:21:15`。
- 最新 metrics：`step=142200`、`epoch=194`，约完成 `70.3%`；最新 ETA `2.45 h`，吞吐 `653.37 samples/s`。
- 最新 validation：`step=141312`、`epoch=192`，equal-source `val/obj/flow_epe_mm=6.714225`，GRAB `7.671749`，Inspire-F1 `5.756701`，`val/hand/flow_epe_knn_unique_mm=1.942620`。
- 当前 best validation：`step=132480`、`epoch=180`，equal-source `val/obj/flow_epe_mm=6.516671`，GRAB `7.290740`，Inspire-F1 `5.742603`，`val/hand/flow_epe_knn_unique_mm=2.024513`。
- `train.log` 未匹配 `traceback/out of memory/cuda error/nan/inf/nccl/failed/error`。
- GPU 状态：物理 GPU `0/2/3` 仍在高利用率运行；显存约 `23.4 GiB / 17.6 GiB / 17.7 GiB`。

## 2026-09-10 02:11:08 +0800 — V1.3 全量训练启动（物理 GPU 0/2/3）

- activity_id: `ACT-20260910-021108-OBJECTINTERACTIONCM-V13-FULL-TRAIN-START`
- timestamp: `2026-09-10 02:11:08 +0800`
- modification_version: `V1.3`
- type: `experiment_run`
- operation_category: `[experiment, operation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户要求实现完成后立即使用物理 GPU `0,2,3` 启动 V1.3 全量训练；V1.3 final plan 已定稿。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `158e0f34068e260d7077a94b4457799b7fca2f31`
- worktree_dirty_at_launch: `false`
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- run_id: `object_interaction_cm_dexplore_rl_v1_3_20260910_020856`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（运行尚未完成；当前证据只用于工程启动确认）
- scope: 使用 V1.3 full config、全量 V1.3 cache、`hand_stream_mode=unique_knn_edges`、KNN `K=32`、两个半径均为 2 cm、MANO 2048/Inspire 10135 高分辨率 hand stream；有效 KNN 边中的 hand ID 去重后监督，每卡 batch=32、global batch=96、目标 202300 steps。

**文件**

- [V1.3 指导](../指导/V1.3.md)
- [V1.3 执行计划](../plan/V1.3.md)
- [实验记录](experiment_log.md)
- [V1.3 full config](../../configs/active/dexplore_rl_v1_3.yaml)
- [V1.3 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json)
- [V1.3 unique-KNN scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3_unique_knn.json)
- [正式训练输出目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/)
- [run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/run_manifest.json)
- [config snapshot](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/config.json)
- [metadata snapshot](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/metadata.json)
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/metrics.jsonl)
- [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/train.log)
- [checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt) 与 [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/latest.pt) — 已生成；另有 epoch `1` 至 `4` 的阶段 checkpoint。

**命令**

```text
CUDA_VISIBLE_DEVICES=0,2,3 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun
  --standalone --nproc_per_node=3
  src/task/ObjectInteractionCm/train.py
  --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml
  --distributed
```

**原因**

通过 V1.3 unique-KNN-hand 的静态检查、专项测试、legacy 回归测试、真实 cache mixed-batch smoke 和
每卡 batch=32 smoke 后，开始用户批准的正式全量训练。训练只读取已生成 cache 和 train-only scale，
不改写 cache、GT、split 或旧 output。

**验证**

- `run_manifest.json`：`modification_version=V1.3`、`base_commit=158e0f3`、`dirty=false`、
  `world_size=3`、`num_obj_points=1024`、`schema_name=ref2dex_object_interaction_cm_v1_3`。
- `step=2944`、`epoch=4` 已完成；当前日志无 OOM、NaN、NCCL failure 或 traceback。
- 当前吞吐约 `645 samples/s`，最新 ETA 约 `8.2 h`；这是运行状态证据，不是科研效果结论。

**终态待补**

- `run_status`、`last_step`/`last_epoch`、`best_metric`、最佳/最近 checkpoint、完整
  `metrics.jsonl`/`train.log` 入口和终态原因。

## 2026-09-10 01:15:19 +0800 — 核对 V1.3 高分辨率手点与 mask 语义

- activity_id: `ACT-20260910-011519-OBJECTINTERACTIONCM-V13-CACHE-MASK-DIAGNOSTIC`
- timestamp: `2026-09-10 01:15:19 +0800`
- modification_version: `V1.3`
- type: `diagnostic`
- operation_category: `[diagnostic]`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前 V1.3 cache 是否已将高分辨率手点裁剪为接触子集；本次只读核对 cache 文件 shape、manifest、Dataset 和 producer，未修改代码、配置、数据或 cache
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d970a7b9c042f12b1f21c669070656083b69a035`
- worktree_dirty: `false`
- run_id: `oicm-v1-3-cache-structure-diagnostic-20260910-011519`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（确认工程数据语义，不代表模型效果或科研假设）
- scope: V1.3 cache 中 1538 点 decoder stream、MANO 2048/Inspire 10135 高分辨率 hand stream、2 cm object candidate mask、1538 点 hand supervision mask、`uint16` KNN index，以及当前 Dataset 的 padding 行为

**文件**

- [V1.3 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json) — 记录 `decoder_hand_points_per_stream=1538` 和 source-specific `knn_hand_points_per_stream`。
- [MANO geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/sequences/train/mano/s1_airplane_fly_1/geometry/manifest.json) — 记录 `[T,1538,3]`、`[T,2048,3]`、`[T,4096,32]` 和两个 mask shape。
- [Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json) — 记录 `[T,1538,3]`、`[T,10135,3]`、`[T,4096,32]` 和两个 mask shape。
- [ObjectInteractionCm Dataset](../../dataset.py) — 当前分别加载旧 1538 hand stream 和高分辨率 KNN stream；KNN stream 按 Dataset 全局 `max_knn_hand_points=10135` 补零并提供 valid mask。
- [V1.3 cache producer](../../tools/data/build_dexplore_rl_v1_3_cache.py) — 当前只生成 `[T,1538]` hand supervision mask，高分辨率 hand stream 保持全量点数组。

**原因**

需要先确认“高分辨率 hand 点是否已按 2 cm 接触区域裁剪”以及“现有 mask 是否可以直接作为高分辨率 hand supervision”；
这决定后续是采用 batch 内动态最大长度，还是需要新增高分辨率监督 mask/压缩点集。

**验证**

- MANO representative sequence：`hand_points_world [279,1538,3]`、`knn_hand_points_world [279,2048,3]`、`obj_knn_indices [279,4096,32] uint16`、`obj_candidate_mask_2cm [279,4096] bool`、`hand_supervision_mask_2cm [279,1538] bool`。
- Inspire representative sequence：`hand_points_world [432,1538,3]`、`knn_hand_points_world [432,10135,3]`、`obj_knn_indices [432,4096,32] uint16`、`obj_candidate_mask_2cm [432,4096] bool`、`hand_supervision_mask_2cm [432,1538] bool`。
- `obj_candidate_mask_2cm` 是完整 4096 个 object pool 点的候选 mask；`hand_supervision_mask_2cm` 是独立的旧 1538 点 hand mask；二者都不会从 `.npy` 点数组中删除点。
- `obj_knn_indices` 是每个 object pool 点对应的 32 个高分辨率 hand 点 ID，指向完整的 2048/10135 点数组，不是全局接触 hand 点列表。
- 现有 cache 没有 `[T,2048]` 或 `[T,10135]` 的高分辨率 hand supervision mask；若切换到高分辨率 hand loss，需要新增该 mask 或重新计算。
- 本次未修改代码、配置、GT、split、cache、checkpoint 或训练输出；未运行训练和评估。

## 2026-09-10 01:59:31 +0800 — V1.3 unique-KNN-hand 训练语义实现完成

- activity_id: `ACT-20260910-015931-OBJECTINTERACTIONCM-V13-UNIQUE-KNN-HAND-IMPLEMENTATION`
- timestamp: `2026-09-10 01:59:31 +0800`
- modification_version: `V1.3.1`
- type: `code_change`
- operation_category: `[architecture, code, data, documentation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确要求只使用采样 object 点的有效 KNN 边、对有效边中的 hand ID 去重后每个 hand 点只监督一次，并要求随后启动全量训练
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d970a7b9c042f12b1f21c669070656083b69a035`
- worktree_dirty: `true`
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- scope: 将 V1.3 hand stream 从固定 1538 点改为有效 KNN 边中的 unique hand ID；Dataset 运行时重算采样 1024 个 object 点对应的 32 条边、生成局部边索引和边 mask；自定义 collate 按 batch 最大 unique hand count 动态 padding；model 保持离线 index gather 和 32 距离重算；runner 对 unique hand point 做一次 hand-flow loss；保留旧 1538 stream 兼容路径

**文件**

- [V1.3 指导](../指导/V1.3.md) — 更新高分辨率 hand 点、有效边和去重监督合同。
- [V1.3 执行计划](../plan/V1.3.md) — 更新 Dataset/model/config/scale/训练执行与验证范围。
- [Dataset](../../dataset.py) — 新增 unique KNN hand selection、局部 edge index、edge mask 和 batch-max dynamic collate。
- [Model](../../model.py) — 接收 edge mask，防止 padding edge 参与 interaction。
- [Runner](../../runner.py) — hand loss/metric 改为 unique KNN hand point 语义。
- [Config](../../config.py)、[V1.3 full config](../../configs/active/dexplore_rl_v1_3.yaml)、[V1.3 smoke config](../../configs/active/dexplore_rl_v1_3_smoke.yaml) — 增加 `hand_stream_mode=unique_knn_edges` 和新的 scale manifest 入口。
- [Scale calibrator](../../tools/data/calibrate_v1_3_scales.py) — 按训练 object sampling 统计有效 edge flow 与 unique hand flow。
- [V1.3 tests](../../tests/test_v1_3_offline_knn.py) — 增加 unique hand 去重、edge mask 和动态 padding 验证。

**原因**

当前 full cache 已经包含完整的 MANO 2048/Inspire 10135 hand points 和
`obj_knn_indices [T,4096,32] uint16`，无需新增 high-resolution supervision mask。训练时将
采样 object 点对应的有效 KNN edge 映射到局部 unique hand stream，使同一 hand ID 可在 interaction
中被多条边引用，但 hand decoder/loss 只保留一个该 hand 点。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_v1_3_offline_knn.py`：`4 passed`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`，仅有已有 NumPy legacy warning。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile ...`：Dataset、Model、Runner、scale calibrator 和测试通过。
- 真实 full cache 混合样本检查：MANO/Inspire 均能输出 unique hand IDs；batch hand 维度按 batch 最大值动态补齐。
- [V1.3 2-step smoke](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_smoke_20260910_015521/)：`COMPLETED`，forward/backward 和 unique hand loss 通过。
- [V1.3 batch=32 smoke](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_015641/)：`COMPLETED`，单卡未出现 OOM。
- 工程结论：`SUPPORTED`；尚未代表正式训练效果或科研假设成立。

**回滚入口**

恢复本条涉及的 Git 文件到 `d970a7b9c042f12b1f21c669070656083b69a035`，删除
[unique-KNN scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3_unique_knn.json)
和两个 smoke output；不删除或修改 V1.2.5 cache、V1.3 full cache、旧 checkpoint 或旧训练输出。

## 2026-09-10 01:56:53 +0800 — V1.3 batch=32 dynamic hand smoke 完成

- activity_id: `ACT-20260910-015653-OBJECTINTERACTIONCM-V13-BATCH32-SMOKE-COMPLETED`
- timestamp: `2026-09-10 01:56:53 +0800`
- modification_version: `V1.3.1`
- type: `operation`
- operation_category: `[diagnostic, operation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: V1.3 unique-KNN-hand 语义已确认；正式训练前验证每卡 batch=32 的显存和反向路径
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d970a7b9c042f12b1f21c669070656083b69a035`
- worktree_dirty: `true`
- run_id: `oicm-v1-3-batch32-smoke-20260910-015641`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`
- scope: 使用正式 V1.3 配置、真实 full cache、单卡 batch=32、单 step、跳过 eval，验证动态 hand batch、forward/backward 和显存

**文件**

- [batch=32 smoke output](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_015641/) — `PENDING` 已改为终态可读目录。
- [batch=32 smoke run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_015641/run_manifest.json) — 输入 cache/config 和运行 provenance。
- [batch=32 smoke train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_20260910_015641/train.log) — 单步训练日志。

**原因**

确认按 batch 最大 unique hand count padding 后，正式 per-device batch=32 能够执行，不会因为
Inspire 全量 10135 点被固定带入每个 batch 而触发 OOM。

**验证**

```text
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -u
src/task/ObjectInteractionCm/train.py --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml
--set train.max_steps=1 --set train.epochs=1 --set train.skip_eval=true
--set data.batch_size=32 --set data.val_batch_size=32 --set data.num_workers=0
--set data.persistent_workers=false --set performance.mode=off
```

- `step=1` 完成，`hand_valid_points=532.156`，`hand/loss_knn_unique=0.0751868`，未出现 OOM/NaN/traceback。
- 工程结论：`SUPPORTED`；这是 wiring/capacity smoke，不是科研效果结论。

## 2026-09-10 01:55:36 +0800 — V1.3 unique-KNN-hand 2-step smoke 完成

- activity_id: `ACT-20260910-015536-OBJECTINTERACTIONCM-V13-UNIQUE-KNN-SMOKE-COMPLETED`
- timestamp: `2026-09-10 01:55:36 +0800`
- modification_version: `V1.3.1`
- type: `operation`
- operation_category: `[diagnostic, operation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: V1.3 unique-KNN-hand 语义实现后的真实 cache wiring 验证
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d970a7b9c042f12b1f21c669070656083b69a035`
- worktree_dirty: `true`
- run_id: `oicm-v1-3-unique-knn-smoke-20260910-015521`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`
- scope: 真实 V1.3 full cache 的混合 MANO/Inspire loader、unique hand selection、局部 edge mask、model forward/backward

**文件**

- [unique-KNN 2-step smoke output](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_smoke_20260910_015521/) — `PENDING` 已改为终态可读目录。
- [unique-KNN smoke run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_smoke_20260910_015521/run_manifest.json) — 输入 cache/config 和运行 provenance。
- [unique-KNN smoke train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_3_smoke_20260910_015521/train.log) — 两步训练日志。

**原因**

确认 interaction 使用有效 KNN 边，hand decoder/loss 使用 valid edge hand ID 的 unique 子集，且
动态 collate 与 edge mask 在真实混合 cache 上可运行。

**验证**

- 两步训练完成；`hand/loss_knn_unique`、`hand/flow_epe_knn_unique_mm` 和
  `data/hand_valid_points` 正常记录。
- 真实 mixed batch 的 hand dimension 为 batch 内最大 unique hand count，而不是 10135。
- 工程结论：`SUPPORTED`；不代表科研效果或最终训练结论。

## 2026-09-10 01:54:48 +0800 — V1.3 unique-KNN-hand scale 校准完成

- activity_id: `ACT-20260910-015448-OBJECTINTERACTIONCM-V13-UNIQUE-KNN-SCALE-COMPLETED`
- timestamp: `2026-09-10 01:54:48 +0800`
- modification_version: `V1.3.1`
- type: `operation`
- operation_category: `[data, operation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认 unique valid-KNN-hand supervision 语义；训练前需要匹配新 hand stream 的 train-only scale
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d970a7b9c042f12b1f21c669070656083b69a035`
- worktree_dirty: `true`
- run_id: `oicm-v1-3-unique-knn-scale-20260910-015448`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`
- scope: 读取 V1.3 train split，按 1024 object sampling、2 cm valid KNN edge 和 unique hand ID 统计四项 scale

**文件**

- [unique-KNN scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3_unique_knn.json) — `s_geo=0.0140784203`、`s_hand_flow=0.1165185915`、`s_knn_hand_flow=0.1166939775`、`s_obj_flow=0.0919974885`。
- [V1.3 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json) — train-only scale input。

**原因**

旧 scale 的 `s_hand_flow` 基于固定 1538 点 decoder mask，不再匹配新的 unique KNN hand supervision；
新 manifest 将 `s_hand_flow` 改为有效 KNN 边去重 hand flow，`s_knn_hand_flow` 保持有效边 flow。

**验证**

- 命令退出码 `0`，20 个 source/stride groups 均完成，四项 scale 均为正。
- 未修改 full cache geometry、GT、split 或旧 scale manifest。
- 工程结论：`SUPPORTED`；scale 统计有效不代表科研效果成立。

## 2026-09-10 00:32:20 +0800 — V1.3 全量离线 KNN cache 启动

- activity_id: `ACT-20260910-003220-OBJECTINTERACTIONCM-V13-FULL-CACHE-START`
- timestamp: `2026-09-10 00:32:20 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[data, operation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认生成全部 630 条 sequence、159476 帧的 GPU-batched V1.3 cache，并指定物理 GPU `2,4,5,6`；V1.3 final plan 已定稿。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e3af6758444304151c3cd30d9dfeda5d7a3923a3`
- worktree_dirty: `true`（activity log 在启动前已写入本次 run 记录）
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- run_id: `oicm-v1-3-cache-20260910-003220`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`
- scope: 按 V1.2.5 source index 的确定性顺序分为 4 个 shard，在物理 GPU `2,4,5,6` 各启动一个 worker；输出新目录 [data/processed_data/object_interaction_cm_dexplore_rl_v1_3/](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/) — `PENDING`。保存固定 1538 点 decoder stream、MANO 2048/Inspire 10135 KNN stream、`uint16 [T,4096,32]` index、2 cm masks 和 validation manifest；不覆盖旧 cache。

**文件**

- [V1.3 执行计划](../plan/V1.3.md) — 长任务范围、资源、schema、验证和回滚入口。
- [V1.3 cache producer](../../tools/data/build_dexplore_rl_v1_3_cache.py) — worker/finalize 实现。
- [V1.2.5 source index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json) — 630 条 sequence 的输入索引。
- [V1.3 cache output](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/) — `PENDING`，待 worker/finalize 完成。

**原因**

为 V1.3 训练准备全量离线 KNN 资产。worker 只对当前 sequence 写入独立目录，最后由 finalize 校验
覆盖、split/source/frame count 和 cache schema，避免并发写共享 index。

**命令**

```text
CUDA_VISIBLE_DEVICES=2 ... --mode worker --shard-index 0 --num-shards 4 --device cuda:0
CUDA_VISIBLE_DEVICES=4 ... --mode worker --shard-index 1 --num-shards 4 --device cuda:0
CUDA_VISIBLE_DEVICES=5 ... --mode worker --shard-index 2 --num-shards 4 --device cuda:0
CUDA_VISIBLE_DEVICES=6 ... --mode worker --shard-index 3 --num-shards 4 --device cuda:0
```

**验证**

- 启动前确认目标 V1.3 输出目录不存在、无同名 producer 进程、V1.2.5 source index 可读。
- 启动前 GPU 状态：物理 GPU `2/4/5/6` 空闲；避开已有任务所在的 `1/3/7`。
- 启动前 Git 基线：`e3af6758444304151c3cd30d9dfeda5d7a3923a3`，工作树干净。
- 终态待补：四个 worker report、finalize validation、run manifest、总 sequence/frame/source 计数和失败原因（如有）。

**回滚入口**

停止本 run 的四个 worker，保留失败现场并记录状态；确认无进程后删除仅由本 run 创建的
`data/processed_data/object_interaction_cm_dexplore_rl_v1_3/`，V1.2.5 cache、配置、checkpoint 和输出不变。

## 2026-09-10 00:46:30 +0800 — V1.3 全量离线 KNN cache 完成

- activity_id: `ACT-20260910-004630-OBJECTINTERACTIONCM-V13-FULL-CACHE-COMPLETED`
- timestamp: `2026-09-10 00:46:30 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[data, operation]`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续 V1.3 final plan 和用户批准的四 GPU 全量 cache 方案。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e3af6758444304151c3cd30d9dfeda5d7a3923a3`
- worktree_dirty: `true`（仅 activity log 有本次运行记录变更）
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- run_id: `oicm-v1-3-cache-20260910-003220`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`
- scope: 四个 GPU worker 完成全部 V1.2.5 source entries；finalize 完成新 cache 的全量 shape、dtype、index range、finite geometry 和覆盖校验。未修改 V1.2.5 cache、配置、checkpoint、GT、split 或训练输出。

**文件**

- [V1.3 cache output](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/) — 全量 cache 目录，约 `81G`。
- [V1.3 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json) — 新 index schema 与完整 sequence 列表。
- [V1.3 assignment](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/assignment.json) — source/split assignment。
- [V1.3 validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/validation_summary.json) — 630 条 sequence 的校验结果。
- [V1.3 run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/run_manifest.json) — finalize 运行清单。
- [V1.3 worker reports](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/workers/) — shard `00/01/02/03` 完成报告。

**原因**

为后续 V1.3 训练提供全量的固定 decoder/loss 1538 点、高分辨率 KNN hand stream、
`uint16 [T,4096,32]` 邻居索引和 2 cm masks。

**验证**

- 四个 worker 均 `COMPLETED`：shard sequence 数为 `158/158/157/157`。
- finalize 输出：`630` 条 sequence、`159476` 帧；train `254 grab + 255 inspire_f1`、val `28 + 30`、test `63 + 0`，与 V1.2.5 完全一致。
- 每条 sequence 校验 `obj [T,4096,3]`、decoder hand `[T,1538,3]`、source-specific KNN hand `[T,2048/10135,3]`、index `[T,4096,32] uint16`、2 cm mask 和 finite geometry。
- 工程结论：`SUPPORTED`；这只证明 cache/schema 生成完整，不代表模型效果或科研假设成立。

**回滚入口**

停止并确认无 V1.3 cache 相关进程后，删除 [V1.3 cache output](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/)；
V1.2.5 cache、配置、checkpoint 和训练输出不受影响。

## 2026-09-10 00:48:10 +0800 — V1.3 train-only scale 校准启动

- activity_id: `ACT-20260910-004810-OBJECTINTERACTIONCM-V13-SCALE-CALIBRATION-START`
- timestamp: `2026-09-10 00:48:10 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[data, operation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: V1.3 final plan 已明确 train-only scale，用户已确认新 cache 的 KNN/半径合同。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e3af6758444304151c3cd30d9dfeda5d7a3923a3`
- worktree_dirty: `true`
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- run_id: `oicm-v1-3-scale-20260910-004810`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`
- scope: 读取 V1.3 train split，按 source/stride 等权统计 `s_geo`、固定 1538 点 decoder `s_hand_flow`、KNN hand-flow `s_knn_hand_flow` 和 `s_obj_flow`；输出 [V1.3 scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3.json) — `PENDING`。不改写 cache geometry。

**文件**

- [V1.3 scale calibrator](../../tools/data/calibrate_v1_3_scales.py) — train-only 统计实现。
- [V1.3 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json) — 校准输入。
- [V1.3 scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3.json) — `PENDING`。

**原因**

为 KNN interaction 流和固定 decoder/loss hand 流分别提供归一化尺度，避免用 1538 点 decoder
flow 统计替代 2048/10135 点 KNN flow 统计。

**命令**

```text
/home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.calibrate_v1_3_scales
  --index data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json
  --output data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3.json
  --max-sequences-per-source 32 --frames-per-sequence-stride 4 --radius-m 0.02 --knn-k 32 --seed 42
  --grab-strides 1 2 3 4 5 6 7 8 9 10 --inspire-strides 1 2 3 4 5 6 7 8 9 10
```

**验证**

- 启动前确认 V1.3 index 已完成且 `knn_k=32`；scale 输出不存在。
- 终态待补：scale manifest、四项数值、命令退出状态和 smoke 引用。

**回滚入口**

停止本 run 后删除仅由校准命令创建的 `scales_train_v1_3.json`；cache geometry 和 V1.2.5 产物不变。

## 2026-09-10 00:51:11 +0800 — V1.3 train-only scale 校准完成

- activity_id: `ACT-20260910-005111-OBJECTINTERACTIONCM-V13-SCALE-CALIBRATION-COMPLETED`
- timestamp: `2026-09-10 00:51:11 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[data, operation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 延续 V1.3 final plan 和已批准的 train-only scale 统计方案。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e3af6758444304151c3cd30d9dfeda5d7a3923a3`
- worktree_dirty: `true`（仅 activity log 有本次运行记录变更）
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- run_id: `oicm-v1-3-scale-20260910-004810`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`
- scope: 读取 V1.3 train split，按 20 个 source/stride group 统计固定 1538 点 decoder hand-flow、KNN hand-flow、交互几何和 object-flow scale；未修改 cache geometry、split 或 GT。

**文件**

- [V1.3 scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3.json) — train-only 四项尺度和 group 统计。
- [V1.3 scale calibrator](../../tools/data/calibrate_v1_3_scales.py) — 可复现命令入口。
- [V1.3 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json) — 输入 index。

**原因**

为 KNN interaction 流与固定 decoder/loss hand 流分别提供归一化尺度，保持高分辨率 KNN 只影响邻域
交互，不改变 1538 点 decoder/loss 合同。

**验证**

- 命令退出码为 `0`；`groups=20`。
- `s_geo=0.0140709252`、`s_hand_flow=0.1166113303`、`s_knn_hand_flow=0.1167068417`、
  `s_obj_flow=0.0919873854`，四项均为正且已写入 manifest。
- 工程结论：`SUPPORTED`；scale 可读性已确认，不代表模型效果或科研假设成立。

**回滚入口**

删除 [V1.3 scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3.json)；
cache geometry、V1.2.5 产物和代码不受影响。

## 2026-09-10 00:29:04 +0800 — V1.3 离线 KNN cache 代码合同与配置完成

- activity_id: `ACT-20260910-002904-OBJECTINTERACTIONCM-V13-OFFLINE-KNN-IMPLEMENTATION`
- timestamp: `2026-09-10 00:29:04 +0800`
- modification_version: `V1.3`
- type: `code_change`
- operation_category: `[code, data, documentation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认 K=32、interaction 与 hand supervision 半径均为 2 cm、MANO KNN=2048、Inspire KNN=10135、固定 decoder/loss hand=1538、全量 cache、uint16 离线索引且运行时只重算 32 个距离，并批准 GPU batched 方案。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `2e52ff22c3fb81fcaf820f31d1d4d25a7ed712a8`
- worktree_dirty: `true`
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- scope: 新增 V1.3 离线 KNN cache producer、train-only scale calibrator、配置和 Task 测试；扩展 Dataset/Model/Runner 读取新 schema。保留 V1.2.5 cache、配置、checkpoint、GT、split、旧 decoder/loss 1538 点合同和 `src/task/CmDecoderv2/` 不变。

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 将 ObjectInteractionCm 当前指针推进到 `V1.3`。
- [src/task/ObjectInteractionCm/config.py](../../config.py) — 增加 KNN hand-flow scale 的兼容默认值。
- [src/task/ObjectInteractionCm/dataset.py](../../dataset.py) — 接入 V1.3 index、KNN geometry/index、2 cm mask 和固定 decoder/KNN 双流。
- [src/task/ObjectInteractionCm/model.py](../../model.py) — 离线 index gather + 32 距离重算；保留 runtime fallback；区分 decoder/KNN hand-flow scale。
- [src/task/ObjectInteractionCm/runner.py](../../runner.py) — 校验 `max_knn_hand_points` 数据合同。
- [src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_v1_3_cache.py](../../tools/data/build_dexplore_rl_v1_3_cache.py) — 新增 worker/finalize/pilot GPU batched cache producer。
- [src/task/ObjectInteractionCm/tools/data/calibrate_v1_3_scales.py](../../tools/data/calibrate_v1_3_scales.py) — 新增 train-only 四项 scale 校准。
- [src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3.yaml](../../configs/active/dexplore_rl_v1_3.yaml) — V1.3 正式配置。
- [src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_3_smoke.yaml](../../configs/active/dexplore_rl_v1_3_smoke.yaml) — V1.3 工程 smoke 配置。
- [src/task/ObjectInteractionCm/docs/README.md](../README.md)、[V1.3 指导](../指导/V1.3.md)、[V1.3 执行计划](../plan/V1.3.md) — 补充版本导航和 cache schema 记录。
- [src/task/ObjectInteractionCm/tests/test_v1_3_offline_knn.py](../../tests/test_v1_3_offline_knn.py) — 新增 index/loader、无 `cdist` 离线路径和 scale 分离测试。

**原因**

将高分辨率 MANO/Inspire hand stream 限定在 KNN，同时保持 1538 点 decoder/loss 输入，消除训练时对
4096 object pool 与全部高分辨率 hand 点的重复邻居搜索；通过独立 index schema 和配置避免污染 V1.2.5。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile ...`：V1.3 producer、calibrator、Dataset、Model、Runner、config 和测试均通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_v1_3_offline_knn.py`：`3 passed`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`，3 个 NumPy legacy warning。
- V1.3 smoke 配置加载通过；使用 [MANO pilot manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3_pilot_mano/run_manifest.json) 和 [Inspire pilot manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3_pilot_inspire/run_manifest.json) 构造混合 batch，decoder 输出 `(2,1538,3)`、离线 edge index `(2,1024,32)`，forward/backward 通过。
- 已有 pilot 的 exact GPU parity：MANO/Inspire 随机抽帧的邻居集合、2 cm candidate mask 和 decoder supervision mask 均无不一致；距离重算最大误差分别为 `1.19e-7 m`、`2.38e-7 m`。
- 工程结论：`SUPPORTED`；尚未执行全量 cache、正式训练或效果评估，不构成科研效果结论。

**回滚入口**

删除本条 V1.3 新增代码、配置、测试和文档，恢复 `docs/current_versions.yaml` 中 ObjectInteractionCm 指针；
V1.2.5 cache、配置、checkpoint、训练输出和已有 pilot 不受影响。

## 2026-09-10 00:53:36 +0800 — V1.3 全量 cache mixed forward/backward smoke 完成

- activity_id: `ACT-20260910-005336-OBJECTINTERACTIONCM-V13-MIXED-SMOKE-COMPLETED`
- timestamp: `2026-09-10 00:53:36 +0800`
- modification_version: `V1.3`
- type: `operation`
- operation_category: `[code, data, operation]`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: V1.3 final plan 要求在 full cache 完成后验证混合 MANO/Inspire loader 与 model forward/backward；用户已确认该配置边界。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e3af6758444304151c3cd30d9dfeda5d7a3923a3`
- worktree_dirty: `true`（仅 activity log 有运行记录变更）
- final_plan: [V1.3 执行计划](../plan/V1.3.md)
- run_id: `oicm-v1-3-mixed-smoke-20260910-005336`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`
- scope: 从 full V1.3 train cache 取一条 GRAB/MANO 和一条 Inspire-RL sequence，构造混合 batch，运行 V1.3 model forward、2 cm hand loss 和 backward；不启动正式训练，不修改 cache、GT、split 或 checkpoint。

**文件**

- [V1.3 smoke config](../../configs/active/dexplore_rl_v1_3_smoke.yaml) — 运行配置。
- [V1.3 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/index.json) — 输入 index。
- [V1.3 scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3.json) — model scale 输入。
- [V1.3 cache output](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_3/) — full cache。

**原因**

验证高分辨率 KNN stream 只进入局部 interaction，固定 1538 点 hand stream 仍承担 decoder/loss，
并确认混合 source batch 的 padding 和离线 index gather 合同可执行。

**验证**

- `CUDA_VISIBLE_DEVICES=2 ...` full-cache mixed forward/backward 命令退出码为 `0`。
- batch KNN valid points 为 `[2048, 10135]`；decoder hand output 为 `(2,1538,3)`；
  offline edge index 为 `(2,1024,32)`。
- monkeypatch `torch.cdist` 后 forward 仍通过，确认离线 KNN 路径没有构造完整距离矩阵。
- loss backward 通过，loss 为 `0.0163753554`；工程结论：`SUPPORTED`。
- 该 smoke 仅证明工程 wiring 和梯度路径，不代表模型效果或科研假设成立。

**回滚入口**

删除本次 smoke 的临时运行记录即可；V1.3 cache、scale、V1.2.5 产物和 checkpoint 不受影响。

## 2026-09-09 22:34:07 +0800 — V1.2.18 估算离线 KNN 构建耗时

- activity_id: `ACT-20260909-223407-OBJECTINTERACTIONCM-OFFLINE-KNN-RUNTIME-DIAGNOSTIC`
- timestamp: `2026-09-09 22:34:07 +0800`
- modification_version: `V1.2.18`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问离线对每帧完整 4096 物体点计算 K=16 最近手点是否会很久；本次只读 benchmark 当前正式/预览 cache，不生成离线索引文件。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_id: `object_interaction_cm_offline_knn_runtime_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（耗时估算有效；正式离线索引生成仍需新的 cache/schema 变体和用户确认）
- scope: 使用当前 V1.2.5 正式 1538 点 Inspire cache 与 V1.2.15 10135 点预览 cache，评估 `4096 object × K=16 hand` 离线 KNN 的 CPU/GPU 量级、active-only 节省和 IO 写入边界；未修改模型、配置、cache、GT、split 或 checkpoint。

**结果**

- CPU `scipy.spatial.cKDTree` 单进程实测：1538 点约 `13.6–14.0 ms/frame`；10135 点约 `17.9–20.8 ms/frame`。按当前 Inspire-RL `72935` 帧线性外推，10135 点约 `21.7–25.2 min`；按全量 `159476` 帧约 `47.5–55.2 min`。
- GPU PyTorch3D batched 实测（RTX 3090，含 NumPy→GPU transfer，不含磁盘写索引）：1538 点约 `0.093 ms/frame`；10135 点约 `0.348 ms/frame`。纯计算外推 Inspire-RL 约 `0.42 min`，全量约 `0.93 min`。
- 实际端到端 builder 还要读 geometry、转 `uint16`、写 `.npy` 和写 manifest。对于 10135 点 Inspire，若只写 `[T,4096,16] uint16` 索引，输出约 `8.90 GiB`；IO 通常会把端到端时间拉到数分钟级。
- 当前 V1.2.5 `active_only=true`。Inspire-RL 共 `72935` 帧，其中 active `39554` 帧（`54.23%`）；GRAB 共 `86541` 帧，active `52278` 帧（`60.41%`）。若索引只为训练实际使用的 active 帧计算/保存，Inspire 输出约从 `8.90 GiB` 降到 `4.83 GiB`，计算时间也约降到 `54%`。
- 因此粗略判断：GPU batched builder 做 Inspire-only active 帧，应是几分钟量级；CPU 单进程是十几分钟量级，CPU 多进程可压到几分钟到十分钟。全量 all-source 索引也不像训练那样要数小时，更可能是十几分钟到一小时内，取决于 IO 和并行策略。

**实现建议**

- 若只是给 10135 Inspire 加速，优先做 Inspire-only active-frame 索引 cache；MANO 1538 点 KNN 本身较小，可先不做。
- builder 应按 sequence 分文件保存，便于 resume，不要一次写全局巨型数组；每个 geometry manifest 记录 `knn_index_file`、`knn_k`、`hand_point_count`、`object_pool_points`、`active_only` 和点集指纹。
- 训练读取时必须保留当前 object 随机抽样语义：Dataset 从 4096 中抽 1024 个点后，再从离线 `[4096,16]` 中 gather 对应行。
- 若后续想进一步提速，3 cm `hand_supervision_mask` 也应离线保存；否则 10135 点会把 Dataset 侧 CPU KDTree 成本放大。

**原因**

需要判断离线 KNN 是否会变成长时间预处理，并区分 compute、IO 写入和 active-only 策略的成本。该判断会影响后续是否值得为 10135 点 Inspire cache 新增 KNN 索引字段。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)：更新 ObjectInteractionCm 当前指针为 `V1.2.18`。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)：记录本次离线 KNN 构建耗时诊断。
- [V1.2.5 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)
- [面积比例手点轨迹预览 index](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/index.json)
- [Dataset](../../dataset.py)
- [ObjectInteractionCm model](../../model.py)

**验证**

- CPU benchmark：对正式 1538 点 Inspire sequence 与预览 10135 点 Inspire sequence 各取最多 48 帧，执行 `cKDTree(hand).query(object_pool, k=16)`。
- GPU benchmark：对同两类 sequence 各取最多 96 帧，用 PyTorch3D `knn_points` batched 计算 `4096×K16`，chunk 分别为 16 和 8。
- active frame 统计：只读扫描 V1.2.5 index 下所有 sequence 的 `obj_candidate_mask_5cm.npy`。
- 未生成离线索引、未运行训练或评估；`git diff --check` 与本条 activity 的本地链接审计在交接前执行。

**回滚入口**

- 删除本条活动记录并将 [当前版本指针](../../../../../docs/current_versions.yaml) 恢复为 `V1.2.17`；正式 cache、模型、训练配置和既有预览不受影响。

## 2026-09-09 22:15:21 +0800 — V1.2.17 评估离线 KNN 索引加速方案

- activity_id: `ACT-20260909-221521-OBJECTINTERACTIONCM-OFFLINE-KNN-DIAGNOSTIC`
- timestamp: `2026-09-09 22:15:21 +0800`
- modification_version: `V1.2.17`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 KNN 是否可以加速以及是否可以离线预计算；本次只读检查当前 Dataset/模型语义、估算索引存储并在空闲 GPU 上进行小型 KNN/gather 基准，未修改代码、cache、配置或运行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_id: `object_interaction_cm_offline_knn_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（离线保存完整物体池的 KNN 索引在当前刚体坐标语义下可行；实际整步训练收益仍需 pilot parity/throughput 实验确认，不构成科研效果结论）
- scope: 当前 V1.2.5 ObjectInteractionCm 的 `4096` 物体池、运行时随机抽取 `1024` 点、`K=16` KNN、`5 cm` edge mask、`3 cm` hand supervision mask、变长 Inspire 手点的离线索引设计和 GPU 微基准；正式 cache、模型、训练配置、GT、split 和 checkpoint 未修改。

**结论与推荐方案**

- 可以离线预计算。为保持当前每个 epoch 从 `4096` 个物体点随机抽取 `1024` 个的语义，应为每帧完整 `4096` 点物体池保存 `K=16` 个最近手点的索引，而不是只保存一次运行时抽到的 `1024` 点结果。
- 推荐 cache 保存 `knn_indices`，形状 `[T,4096,16]`、`uint16`；`10135 < 65535`，索引类型足够。Dataset 在随机抽样后 gather 对应的 `[1024,16]` 索引，模型只对这 16 个邻居重新计算距离并施加 `distance <= 0.05 m`，因此不必保存距离或逐 edge mask。
- 当前 object/hand 都经过同一个 `object_pose_t` 刚体变换；刚体变换不改变邻居索引和距离，所以索引可在 world 坐标或 object frame 离线计算。它只依赖当前帧几何，不依赖未来 stride 或 hand flow。
- 若将 Inspire 改为 `10135` 点，必须用新手点集合重建索引；若改变 `K` 或 object/hand 点池，也必须重建。仅改变半径时，保留 top-16 索引并在运行时重算 16 个距离即可继续使用。

**存储量**

- 每帧完整索引为 `4096×16×2 = 131072 B`，约 `128 KiB`。
- 当前 `72935` 个 Inspire-RL 帧需要约 `8.90 GiB`；当前全部 `159476` 帧都保存索引需要约 `19.47 GiB`。
- 仅保存索引比同时保存 `float16` 距离或逐 edge bool 更省；若再保存每物体点的有效邻居数，额外只需约 `0.28 GiB`（Inspire）或 `0.61 GiB`（全量）。
- 结合上一条 V1.2.16 估算，10135 Inspire 手点使当前 cache 约为 `34.11 GiB`；再加 Inspire-only KNN 索引约 `43.01 GiB`，全量 source 索引约 `53.58 GiB`。当前文件系统可用空间约 `815 GiB`，容量不是立即阻塞，但 `/home2` 已使用约 `97%`，应避免产生多份全量副本。

**运行时与实现边界**

- GPU 微基准（RTX 3090，`B=8`、`N_obj=1024`、`K=16`）中，PyTorch3D KNN 从 `1538` 手点的 `0.584 ms/batch` 增至 `10135` 手点的 `2.116 ms/batch`，约 `3.63x`。
- 使用离线索引后，gather 16 个邻居并重算 16 个距离约 `0.068 ms/batch`；相对 `10135` 点 KNN，KNN 选择阶段约 `31.2x` 降低。该基准只覆盖邻居选择，不代表整个训练 step 同比例加速。
- 离线 KNN 不会消除 `SharedHandFlowDecoder` 对全部手点和 `16` 个 slot 的计算；若手点变为 `10135`，该部分仍约按点数线性增大。若 hand decoder 只用于 `3 cm` mask 下的辅助 loss，可进一步考虑只在监督点上计算，但这属于额外的模型/损失实现变更，需要单独确认。
- 变长 MANO/Inspire 混合 batch 仍需处理。当前不同 valid point count 会绕过 prefix fast path，退回 `cdist+topk`；应采用 source/point-count homogeneous batch，或实现按有效点数分组的 KNN 路径。
- 当前 Dataset 每个样本还会用 CPU `cKDTree` 计算 `3 cm` hand supervision mask。若使用 `10135` 点，建议同时离线保存逐手点 bool mask；Inspire train/val 约增加 `0.69 GiB`，可移除该 CPU 重复计算。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)：更新 ObjectInteractionCm 当前指针为 `V1.2.17`。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)：记录本次离线 KNN 方案诊断。
- [V1.2.5 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)
- [Dataset](../../dataset.py)
- [ObjectInteractionCm model](../../model.py)
- [ObjectInteractionCm decoder](../../decoder.py)
- [面积比例手点预览 index](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/index.json)

**原因**

需要区分“离线缓存能否消除全量手点搜索”和“10135 点仍会带来的 decoder/padding 成本”。完整物体池索引可以保留当前 object 随机采样，而只把每次前向的 KNN 搜索替换为固定邻居 gather。

**验证**

- 只读检查 [Dataset](../../dataset.py) 的随机 object sampling、刚体坐标转换、hand supervision mask 和 [model](../../model.py) 的 PyTorch3D/fallback KNN 分支。
- 扫描 V1.2.5 index 得到当前 Inspire-RL `285` 条序列、`72935` 帧和全量 `159476` 帧，并计算 `uint16` 索引及可选 mask/count/distance 存储量。
- 使用 `CUDA_VISIBLE_DEVICES=1` 在 NVIDIA GeForce RTX 3090 上完成 `B=8,N_obj=1024,K=16` 的 1538/10135 KNN 与 16-neighbor gather 微基准；未接触正在使用的 GPU 7 任务。
- 使用 3 条 10135 点 Inspire 预览序列做稀疏性抽查：active object fraction 为 `0.4286–0.8160`，平均有效邻居为 `6.571–13.018`；样本量不足以据此替代全量存储设计。
- `git diff --check` 与本条 activity 的本地链接审计在交接前执行。

**回滚入口**

- 删除本条活动记录并将 [当前版本指针](../../../../../docs/current_versions.yaml) 恢复为 `V1.2.16`；正式 cache、模型、训练配置和既有预览不受影响。

## 2026-09-09 21:58:14 +0800 — V1.2.16 评估 Inspire 10135 手点 cache 存储开销

- activity_id: `ACT-20260909-215814-OBJECTINTERACTIONCM-CACHE-STORAGE-10135-DIAGNOSTIC`
- timestamp: `2026-09-09 21:58:14 +0800`
- modification_version: `V1.2.16`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问将 Inspire 手点从 1538 增加到 10135 后的 cache 存储开销；本次只读统计现有 V1.2.5 cache 并估算线性存储/邻域搜索规模，不重建或覆盖 cache。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_id: `object_interaction_cm_cache_storage_10135_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（存储和张量规模估算有效；是否值得接入训练仍需单独实验，不构成效果结论）
- scope: 当前 `object_interaction_cm_dexplore_rl_v1_2_5` cache 的 Inspire-RL train/val geometry、1538→10135 手点存储估算、混合 batch padding 和 `LocalHandInteraction` KNN 输入规模；正式 cache、训练配置、模型、GT、split 和 checkpoint 未修改。

**结果**

- 当前 Inspire-RL 为 `285` 条序列、`72935` 帧；完整 cache 约 `9.19 GiB`，其中 `hand_points_world.npy` 与 `hand_normals_world.npy` 合计约 `2.51 GiB`。
- 每帧手部两份 `float32 [N,3]` 数组从 `36912 B`（1538 点）增至 `243240 B`（10135 点），手部部分增加 `6.59x`，每帧多约 `201.5 KiB`。
- 仅替换 Inspire-RL 的手点数组，预计新增约 `14.02 GiB`；Inspire-RL 子 cache 约从 `9.19 GiB` 增至 `23.21 GiB`，整个当前 `20.09 GiB` cache 约增至 `34.11 GiB`，总量约 `1.70x`。
- 若物体点池仍为 `4096` 点，单帧 object+hand 几何从约 `132.0 KiB` 增至 `333.5 KiB`，约 `2.53x`；这还未计入 loader 临时张量。
- 当前 Dataset/模型使用统一的 `num_hand_points=1538` 和 `max_hand_points=3076`。若 Inspire 直接使用 `10135` 点，混合 GRAB/Inspire batch 需要扩大统一 padding 或实现变长 batch；否则会触发 shape/schema 不兼容。
- 若前向也使用全部 `10135` 点，物体 `1024` 点对手点的邻域搜索候选量约为原来的 `6.59x`。混合不同有效点数时，当前前缀 mask 快路径还可能退回完整 `cdist` 路径，运行时成本会高于单纯的磁盘增长。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)：更新 ObjectInteractionCm 当前指针为 `V1.2.16`。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)：记录本次只读存储/运行时开销诊断。
- [V1.2.5 cache index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)
- [示例 Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s5_mouse_pass_1/geometry/manifest.json)
- [Dataset](../../dataset.py)
- [ObjectInteractionCm model](../../model.py)

**原因**

需要将“cache 多占多少磁盘”和“训练前向是否变重”分开判断。10135 点主要改变手点数组、统一 batch padding 和 KNN 搜索规模；它不会自动改变物体池 `4096` 点或运行时物体抽样 `1024` 点。

**验证**

- 只读扫描 `index.json` 中所有 train/val Inspire-RL sequence，并按实际 `.npy` 文件大小汇总当前 cache。
- 按 `float32`、两份 `[T,N,3]` 手点/法线数组和 `1538→10135` 的点数比例估算新存储；当前 `.npy` header 对总量影响可忽略。
- 对照 [Dataset](../../dataset.py) 的统一点数/padding 合同和 [model](../../model.py) 的 prefix fast path、KNN 输入；未运行训练、评估、数据重建或前向 benchmark。
- `git diff --check` 与本条 activity 的本地链接审计在交接前执行。

**回滚入口**

- 删除本条活动记录并将 [当前版本指针](../../../../../docs/current_versions.yaml) 恢复为 `V1.2.15`；正式 cache、训练配置、模型和既有预览不受影响。

## 2026-09-09 20:57:40 +0800 — V1.2.15 面积比例手点轨迹预览与 viewer 兼容

- activity_id: `ACT-20260909-205740-OBJECTINTERACTIONCM-VARIABLE-HAND-TRAJECTORY-PREVIEW`
- timestamp: `2026-09-09 20:57:40 +0800`
- modification_version: `V1.2.15`
- type: `code / diagnostic / data / operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认按最新完整 mesh 面积比例方案继续：MANO 2048 点、Inspire 10135 点，生成若干条数据集轨迹并使用 `visualize_grab.py` 可视化。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)（用户确认后追加轨迹预览扩展）
- run_id: `variable_hand_trajectory_preview_20260909`
- run_status: `COMPLETED`; viewer_run_id: `visualize_grab_8100_20260909`，viewer `run_status: RUNNING`，监听 `0.0.0.0:8100`
- conclusion: `SUPPORTED`（工程预览合同、点数/法线/帧对齐和 viewer 读取均通过；是否解决局部视觉空带仍需用户在交互界面中判断，不构成训练效果结论）
- scope: 从正式 V1.2.5 index 的 train split 确定性选择 MANO 和 Inspire-RL 各 3 条序列；沿用 4096 点物体池、物体姿态、帧号和 provenance，仅按 canonical triangle+barycentric 对应重建 MANO 2048 / Inspire 10135 手点，并逐帧重算 flat normal 与 5 cm activity mask。viewer 读取端允许 manifest 声明的变长手点数。
- protection: 未修改正式 `data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/` cache、训练配置、GT、split 定义、坐标系、模型或 checkpoint。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)
- [轨迹预览构建脚本](../../research/hand_region_sampling/build_trajectory_preview.py)
- [viewer 兼容修改](../../visualize_grab.py)
- [预览 index](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/index.json)
- [assignment](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/assignment.json)
- [run manifest](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/run_manifest.json)
- [预览输出目录](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/)

**选中的序列**

- MANO：`s1/airplane_fly_1`（279 帧，2048 点，223 帧 5 cm active）、`s1/airplane_pass_1`（219 帧，2048 点，130 active）、`s1/alarmclock_lift`（510 帧，2048 点，450 active）。
- Inspire-RL：`s1/airplane_lift`（432 帧，10135 点，358 active）、`s1/alarmclock_see_1`（254 帧，10135 点，135 active）、`s1/apple_pass_1`（177 帧，10135 点，109 active）。

**原因**

最新 PLY 采样合同只给出了 canonical 静态点云，无法直接观察手点在真实物体轨迹中的覆盖。此次用同一 face/barycentric 对应跨帧重建手点，并让 viewer 读取 manifest 中的手点数，从而在不重建正式 cache 的前提下检查 2048/10135 点采样的动态空间分布。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/build_trajectory_preview.py src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- 6 条序列数组检查：物体池均为 `[T,4096,3]`；MANO 手点为 `[T,2048,3]`；Inspire 手点为 `[T,10135,3]`；全部 finite；法线最大范数误差 MANO `1.79e-7`、Inspire `5.96e-8`。
- viewer `--check-only`：MANO `s1/airplane_fly_1` 与 Inspire `s1/airplane_lift` 均通过，KNN 1/4/8/16/32/64 和 mesh provenance 可读。
- 交互 viewer 已启动：`http://localhost:8100`（服务监听 `0.0.0.0:8100`，当前初始序列为 MANO `s1/airplane_fly_1`，界面可切换 6 条轨迹）。
- 首次生成中发现 Inspire 极小三角面的法线下限设置问题，已修正为 `1e-12` 并用同一 seed 完整重跑；错误中间目录已移至 `/tmp/variable_hand_trajectory_preview_20260909_bad_normals`，未进入最终预览目录。
- `git diff --check`：通过；针对本次 3 个变更路径执行 `audit_diff.py --worktree --check-links`：通过（10 个本地链接可导航）。完整 Task worktree 审计仍会包含此前用户遗留的其它未提交文件，故未将其误归入本条 activity；未运行训练或评估。

**回滚入口**

删除 [轨迹预览输出目录](../../research/hand_region_sampling/output/variable_hand_trajectory_preview_20260909/)、[轨迹预览构建脚本](../../research/hand_region_sampling/build_trajectory_preview.py) 和 viewer 对变长手点的兼容改动，并将当前版本指针恢复到 `V1.2.14`；正式 cache 与既有 PLY 预览不受影响。

## 2026-09-09 16:27:23 +0800 — V1.2.14 核对 cache 点存储与 KNN16 计算位置

- activity_id: `ACT-20260909-162723-OBJECTINTERACTIONCM-CACHE-KNN-RUNTIME-DIAGNOSTIC`
- timestamp: `2026-09-09 16:27:23 +0800`
- modification_version: `V1.2.14`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前 cache 是保存点后前向计算 KNN16，还是 cache 预计算 KNN16/5 cm 阈值；本次只读检查 cache 文件、Dataset、模型和 V1.2.5 manifest。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（cache 保存每帧完整的 4096 点物体池、1538 点手表面及其法线；5 cm 只预计算了帧级 candidate active 标记，KNN16 与逐物体点的 5 cm edge mask 在前向动态计算）
- scope: 只读核对 V1.2.5 cache 的 geometry 文件、cache builder、ObjectInteractionCm Dataset 和 LocalHandInteraction；未修改代码、配置、cache、split、GT、checkpoint 或训练产物。

**文件与证据**

- `docs/current_versions.yaml`：更新 ObjectInteractionCm 当前版本指针为 `V1.2.14`。
- [V1.2.5 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)：记录 `surface_points=1538`、`surface_seed=2024` 和已完成 cache conversion。
- [示例 Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s5_mouse_pass_1/geometry/manifest.json)：记录 `object_pool_points=4096`、`hand_points=1538`、`candidate_threshold_m=0.05`、`candidate_mask_shape=[167]`。
- [ObjectInteractionCm Dataset](../../dataset.py)：读取 `obj_points_pool_world.npy [T,4096,3]`、`hand_points_world.npy [T,1538,3]`，在 `__getitem__` 中动态随机抽取 1024 个物体点，并由当前/未来帧点计算 flow；同时用 full 4096 物体池动态计算 3 cm hand supervision mask。
- [ObjectInteractionCm model](../../model.py)：按配置的 `knn_k=16` 对运行时 1024 个物体点和全部有效手点做 KNN，再对 KNN 结果施加 `interaction_radius_m=0.05` 的逐 edge 阈值。
- [V1.2.5 active config](../../configs/active/dexplore_rl_v1_2_5.yaml)：明确 `num_obj_pool=4096`、`num_obj_points=1024`、`num_hand_points=1538`、`knn_k=16`、`interaction_radius_m=0.05`。

**原因**

当前 cache 的目标是保存可复用的几何和轨迹基础数据，不绑定某一次运行时的物体子采样或 KNN 配置。因而 KNN16 没有写入 cache；训练前向可以根据配置改变 K 值，而不必重建几何 cache。

**验证**

- V1.2.5 示例 cache 文件形状核对：`obj_points_pool_world.npy=(T,4096,3)`、`obj_normals_pool_world.npy=(T,4096,3)`、`hand_points_world.npy=(T,1538,3)`、`hand_normals_world.npy=(T,1538,3)`、`obj_candidate_mask_5cm.npy=(T,)`。
- Dataset `active_only=True` 只用一维 candidate 标记筛掉整帧；该标记不携带每个 object point 的 KNN 索引、距离或 16 邻居 mask。
- Model V1.2.5 前向顺序是：运行时选 1024 object points → KNN `k=16` → `distance <= 0.05 m` edge mask → attention/Cm；因此不是 cache 预存 KNN16。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未运行训练、评估或数据重建。

## 2026-09-09 13:24:57 +0800 — V1.2.13 按 mesh 表面积比例生成 MANO/Inspire 预览

- activity_id: `ACT-20260909-132457-OBJECTINTERACTIONCM-AREA-RATIO-UNIFORM-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 13:24:57 +0800`
- modification_version: `V1.2.13`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认使用按 mesh 表面积换算的点数进行均匀采样；MANO 固定 2048 点，Inspire 使用 10135 点。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `uniform_surface_ratio_2048_10135_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（MANO 2048 与 Inspire 10135 的完整 mesh 表面积均匀采样 PLY 已生成，实测点密度比为 `1.000055`；视觉适用性仍需用户检查，不构成训练效果结论）
- scope: 新增 `uniform_surface_ratio` 预览 profile；MANO/Inspire 均无 Region/link quota，分别按完整 mesh 面积比例采样；不覆盖既有 2048/2048 预览。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.13`。
- [预览脚本](../../research/hand_region_sampling/run.py)：支持 MANO 2048、Inspire 10135 的独立均匀采样点数和 manifest 记录。
- [实验 README](../../research/hand_region_sampling/README.md) 与 [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记 `uniform_surface_ratio` profile。
- [area-ratio 输出目录](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/)。
- [MANO 2048 PLY](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/mano_uniform_2048.ply) 与 [Inspire 10135 PLY](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/inspire_uniform_10135.ply)。
- [采样统计](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/run_manifest.json)。

**原因**

前一版两者都使用 2048 点，但完整 mesh 表面积相差约 4.94846 倍。为保持单位表面积点密度一致，本次固定 MANO 2048 点，并按 `2048 × A_Inspire/A_MANO = 10134.44` 向上取整为 Inspire 10135 点。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile uniform_surface_ratio --modification-version V1.2.13 --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909`：完成。
- MANO PLY 为 2048 点、Inspire PLY 为 10135 点；两者均 finite，统一灰色，最大单位法线误差分别约 `4.0e-8` 和 `4.4e-8`。
- 实测点密度：MANO `50250.20 points/m²`，Inspire `50252.98 points/m²`，密度比 `1.000055`。
- 同 seed 重跑后两份 PLY SHA-256 完全一致；未修改既有 2048/2048 PLY、正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未运行训练和评估。
- 回滚入口：删除 `uniform_surface_ratio` profile 的代码/文档改动、[area-ratio 输出目录](../../research/hand_region_sampling/output/uniform_surface_ratio_2048_10135_preview_20260909/) 及版本指针更新；既有预览不需回滚。

## 2026-09-09 13:19:26 +0800 — V1.2.12 统计 MANO/Inspire mesh 表面积与等密度点数

- activity_id: `ACT-20260909-131926-OBJECTINTERACTIONCM-MESH-AREA-RATIO-DIAGNOSTIC`
- timestamp: `2026-09-09 13:19:26 +0800`
- modification_version: `V1.2.12`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问两份 canonical mesh 的表面积比例，以及 MANO 2048 点对应的 Inspire 等表面密度点数；本次只读统计已有 uniform-surface sampling summary。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（按当前 canonical 坐标解释，两份 mesh 面积比约 4.94846；相同表面点密度下 MANO 2048 点对应 Inspire 约 10134 点，取不低于该密度可用 10135 点）
- scope: 只读汇总 `uniform_surface_2048_preview_20260909/sampling_summary.json` 中 MANO/Inspire 的完整 mesh 三角面面积；未修改采样、PLY、cache 或训练数据。

**文件与证据**

- [uniform surface sampling summary](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/sampling_summary.json)：提供两种 canonical mesh 的总三角面面积和资产指纹。
- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.12`。

**原因**

用户希望按 mesh 表面积比例保持 MANO/Inspire 的点密度一致。计算使用 `N_inspire = 2048 × A_inspire / A_mano`，不改变已有 2048 点 PLY。

**验证**

- MANO 总表面积 `0.0407560583 m² = 407.5606 cm²`。
- Inspire 总表面积 `0.2016795812 m² = 2016.7958 cm²`。
- 面积比 `A_inspire/A_mano = 4.94845649`；换算点数 `2048 × ratio = 10134.4389`，最近整数为 `10134`，向上取整为 `10135`。
- 结果属于当前 canonical mesh/坐标口径的几何密度换算，不代表两种手的训练效果或跨形态泛化结论。

## 2026-09-09 13:10:34 +0800 — V1.2.11 MANO/Inspire 完整 mesh 均匀表面采样预览

- activity_id: `ACT-20260909-131034-OBJECTINTERACTIONCM-UNIFORM-SURFACE-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 13:10:34 +0800`
- modification_version: `V1.2.11`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求重新在 MANO/Inspire 各自完整 mesh 上均匀采样 2048 点并生成 PLY；本次将“均匀”固定为单位表面积采样概率相同，不使用 Region、link、segment 或 tip quota。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `uniform_surface_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（两份完整 mesh 表面积均匀采样 PLY 已生成，点数、finite、法线、无配额和确定性检查通过；视觉适用性仍由用户检查，不构成正式训练效果结论）
- scope: 在现有预览入口新增 `uniform_surface` profile；两种手各自全局按三角形面积选面并在面内作均匀 barycentric 采样，总数 2048，统一灰色显示；不覆盖旧预览。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.11`。
- [预览脚本](../../research/hand_region_sampling/run.py)：新增完整 mesh 表面积均匀采样、灰色 PLY 和 sampling method manifest 字段。
- [实验 README](../../research/hand_region_sampling/README.md) 与 [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记 `uniform_surface` profile 及无配额合同。
- [uniform surface 输出目录](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/)。
- [MANO uniform PLY](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/mano_uniform_2048.ply) 与 [Inspire uniform PLY](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/inspire_uniform_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/run_manifest.json)。

**原因**

前序 Region、tip-heavy 和 per-link 预览都显式改变了局部点密度，无法直接观察完整 mesh 在统一面积密度下的自然分布。本次取消所有语义配额，MANO 与 Inspire 各自在完整 canonical mesh 上独立生成 2048 点表面均匀基线。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile uniform_surface --modification-version V1.2.11 --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/uniform_surface_2048_preview_20260909`：完成。
- 两份 PLY 均为 2048 点、全字段 finite、RGB 全部为 `(190,190,190)`，最大单位法线误差小于 `4.4e-8`；summary 中 `region_quotas=null`、`stratum_quotas=null`。
- Inspire 按 source visual ID 的自然计数为 `1491,14,68,44,37,61,34,57,52,67,40,56,27`；`hand_base_link` 占 1491 点来自其约 73.8% 的总表面积，是全局表面积均匀采样的预期结果，不是额外权重。
- 同 seed 重跑后 MANO/Inspire PLY SHA-256 完全一致；balanced profile 重跑也与 V1.2.6 基线 PLY 哈希一致。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未修改正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint，未运行训练和评估。
- 回滚入口：删除 `uniform_surface` profile 的代码/文档改动、[uniform surface 输出目录](../../research/hand_region_sampling/output/uniform_surface_2048_preview_20260909/) 及版本指针更新；既有 balanced/tip-heavy/link-stratified 输出不需回滚。

## 2026-09-09 11:50:27 +0800 — V1.2.10 核对 MANO mesh 与空带来源

- activity_id: `ACT-20260909-115027-OBJECTINTERACTIONCM-MANO-MESH-STRUCTURE-DIAGNOSTIC`
- timestamp: `2026-09-09 11:50:27 +0800`
- modification_version: `V1.2.10`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 MANO 是否存在对应 link mesh，以及 MANO PLY 空带是否由形态缺少 mesh 造成；本次只读检查模型资产字段和现有采样代码。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（MANO 每个形态由同一 778 顶点/1538 三角面拓扑经 `shapedirs` 和蒙皮权重变形得到整体 mesh；不存在 Inspire 式独立 link mesh。空带属于采样/分区覆盖问题，不是形态没有 mesh）
- scope: 只读核对 `MANO_RIGHT.pkl` 的 `v_template`、`shapedirs`、`J_regressor`、`weights`、`f` 字段及 GRAB/MANO 当前 face-center 与预览采样路径；未修改代码、配置、cache 或产物。

**文件与证据**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.10`。
- [MANO 资产](../../../../../dataset/arctic/data/body_models/mano/MANO_RIGHT.pkl)：包含 `v_template (778,3)`、`shapedirs (778,3,10)`、`posedirs (778,3,135)`、`J_regressor (16,778)`、`weights (778,16)` 和 `f (1538,3)`；形状和姿态共享同一 mesh topology。
- [GRAB MANO 预处理](../../../../../process/GRAB/raw.py)：正式手点以每帧 MANO 变形顶点和固定 `faces` 的 face center 生成，保留跨帧 face 对应。
- [MANO 预览脚本](../../research/hand_region_sampling/run.py)：预览从同一 MANO faces 构造三角面池；`link_stratified` 对应的是 joint-segment 语义，不是 URDF link mesh。

**原因**

MANO 的 `β` 只改变同一模板 mesh 的顶点位置，不能提供 `thumb_distal`、`index_tip` 之类的独立刚体网格。当前 MANO 空带更可能来自面中心/有放回采样、按最近关节和 PCA 末端划分造成的空间覆盖不足；若要修复，应基于 MANO 关节位置、顶点 skinning weights 或 geodesic 邻域定义 joint-centered 区域，而不是寻找不存在的 link mesh。

**验证**

- 使用 NumPy 兼容 shim 成功读取 `MANO_RIGHT.pkl` 字段和形状；未发现独立 link mesh 字段。
- 只读检查 `process/GRAB/raw.py` 与预览脚本的 face/triangle 采样路径；未运行训练、评估或数据重建。
- 结论不代表新的采样方案已经验证，仅确认 MANO 几何表示和空带问题的来源类别。

## 2026-09-09 11:43:35 +0800 — V1.2.9 严格 per-link mesh 采样预览

- activity_id: `ACT-20260909-114335-OBJECTINTERACTIONCM-LINK-STRATIFIED-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 11:43:35 +0800`
- modification_version: `V1.2.9`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认改为每个有 mesh 的 Inspire link 独立采样，并为 MANO 使用对应 joint-segment 分层；总点数保持 2048，不覆盖既有预览和正式 cache。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `link_stratified_v2_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（已验证 Inspire 每个非空 visual link 获得独立 quota 且点来自该 link 自己的 mesh；但按 URDF `*_tip` frame 统计，小指仍无 15 mm 内点，局部 tip 空带是否消失需用户查看 PLY，不能仅据此宣称已解决）
- scope: 在 V1.2.6 预览入口新增严格 `link_stratified` profile；Inspire 按 13 个非空 visual link 分层，MANO 按 11 个 joint-segment 分层；不改变 balanced/tip_heavy 基线行为。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.9`。
- [预览脚本](../../research/hand_region_sampling/run.py)：新增 link/segment strata、独立面积加权采样和严格 link quota 映射。
- [实验 README](../../research/hand_region_sampling/README.md)：记录 `link_stratified` profile 及其保护边界。
- [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记 link-stratified quota profile。
- [link-stratified 输出目录](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/)。
- [MANO link-stratified PLY](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/mano_weighted_2048.ply) 与 [Inspire link-stratified PLY](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/inspire_weighted_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/run_manifest.json)。

**原因**

之前的 Region/tip-heavy 方案虽然读取了 link mesh，但同一根手指的多个 link 仍共享 Region quota，不能保证每个 link 都有点。本次将 hand base 固定为 512 点；五个末端父 link（`thumb_distal`、`index_intermediate`、`middle_intermediate`、`ring_intermediate`、`pinky_intermediate`）合计 1024 点；其余七个 finger link 合计 512 点。每个 link 内仍按自身三角面面积加权并作 barycentric surface sampling。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile link_stratified --modification-version V1.2.9 --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909`：完成。
- MANO/Inspire PLY 均为 2048 点，点和法线 finite，最大单位法线误差约 `4.0e-8`。
- Inspire `source_visual_id` 计数严格为 `512,74,73,73,205,73,205,73,205,73,205,73,204`，对应 13 个非空 visual link；所有点均由对应 link 的 mesh 三角面抽取。
- MANO segment Region 计数为 `512,103,205,103,205,102,205,102,205,102,204`。
- 基线 `balanced` profile 重新生成后 MANO/Inspire PLY SHA-256 与 V1.2.6 基线完全一致；未改变既有 profile。
- Inspire tip frame 15 mm 内点数为 thumb `56`、index `73`、middle `17`、ring `68`、pinky `0`；说明 per-link quota 不等于 tip-frame 邻域覆盖，后续若仍有局部空带需另做 joint-centered/collar 规则。
- `git diff --check` 与 `audit_diff.py --check-links`：通过；未修改正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint，未运行训练和评估。
- 回滚入口：删除 `link_stratified` 代码/文档改动、[link-stratified 输出目录](../../research/hand_region_sampling/output/link_stratified_v2_2048_preview_20260909/) 及版本指针更新，即可恢复既有 V1.2.6 预览行为；balanced/tip-heavy 输出不需回滚。

## 2026-09-09 11:12:46 +0800 — V1.2.7 tip-heavy 2048 点预览对比

- activity_id: `ACT-20260909-111246-OBJECTINTERACTIONCM-TIP-HEAVY-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 11:12:46 +0800`
- modification_version: `V1.2.7`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认保持总点数 2048，只增加 tip 配额以检查指尖关节处空带是否由比例不足造成；不覆盖基线、不接入正式 cache。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的其它用户改动）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `tip_heavy_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（tip 配额翻倍后拇指/食指/中指/无名指 tip 关节附近覆盖明显增加，但小指因 `pinky_tip` frame 与末端 mesh 不对齐仍无 15 mm 内点；仅增加比例不能完全解决该问题）
- scope: 在现有 V1.2.6 预览脚本中新增 tip-heavy 配额 profile；总点数、三角面采样、Region 几何划分、seed、姿态和正式 cache/训练合同保持不变。

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml)：将 ObjectInteractionCm 指针更新为 `V1.2.7`。
- [预览脚本](../../research/hand_region_sampling/run.py)：新增 `balanced`/`tip_heavy` quota profile，并将 quota 显式传入采样与统计函数。
- [实验 README](../../research/hand_region_sampling/README.md)：记录 tip-heavy 对照的变量和命令。
- [实验定义](../../research/hand_region_sampling/experiment.yaml)：登记默认 profile 与 tip-heavy profile。
- [tip-heavy 输出目录](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/)。
- [MANO tip-heavy PLY](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/mano_weighted_2048.ply) 与 [Inspire tip-heavy PLY](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/inspire_weighted_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/run_manifest.json)。

**原因**

当前 balanced profile 每根手指约 102 个 tip 点，且这些点分散在末端 20% 表面。为隔离“tip 比例是否不足”这一变量，本次保持所有几何和采样规则不变，只将 `palm/body/tip` 从 `512/1024/512` 调整为 `512/512/1024`，即每根手指 tip 约 205 点。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --quota-profile tip_heavy --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/tip_heavy_2048_preview_20260909`：完成。
- MANO/Inspire PLY 均为 2048 点，Region 计数为 `512,103,205,103,205,102,205,102,205,102,204`，点和法线 finite，最大单位法线误差约 `4.2e-8`。
- 同 seed 重跑后两份 PLY 的 SHA-256 完全一致；统计 JSON 仅因输出目录绝对路径不同而不同。
- Inspire tip frame 15 mm 邻域计数（balanced → tip-heavy）：thumb `55→119`、index `95→183`、middle `50→83`、ring `93→179`、pinky `0→0`；说明增加配额对前四指有效，但不能修复小指 frame/mesh 对齐问题。
- `git diff --check`：通过；未修改正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint，未运行训练和评估。
- 回滚入口：删除新增 profile、文档改动和 [tip-heavy 输出目录](../../research/hand_region_sampling/output/tip_heavy_2048_preview_20260909/)，即可恢复 V1.2.6 预览行为；基线输出不受影响。

## 2026-09-09 10:26:31 +0800 — V1.2.6 完成 MANO/Inspire 2048 点 Region 加权采样预览

- activity_id: `ACT-20260909-102631-OBJECTINTERACTIONCM-REGION-SAMPLING-PREVIEW`
- timestamp: `2026-09-09 10:26:31 +0800`
- modification_version: `V1.2.6`
- type: `code / diagnostic / data`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认按默认方案执行：MANO/Inspire 共用 11 Region、总点数 2048、`palm/body/tip=25%/50%/25%`，Region 内保留面积加权。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留工作树中已有的 ObjectInteractionCm 转换器/配置/查看器、CmDecoderv2 改动及其它用户修改）
- final_plan: [V1.2.6 Region 加权手点预览计划](../plan/V1.2.6.md)
- run_id: `region_weighted_2048_preview_20260909`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（预览脚本、两份 2048 点 PLY、Region 配额、法线和确定性检查通过；PLY 的视觉适用性等待用户人工确认，正式训练效果尚未评估）
- scope: 新增 Task-local 研究预览脚本和定义，生成 MANO/Inspire canonical PLY；不修改 V1.2.5 正式 cache、训练配置、模型、GT、split、坐标系或 checkpoint。

**文件与产物**

- [V1.2.6 计划](../plan/V1.2.6.md)：锁定 11 Region、2048 配额、20% 末端 tip、Region 内面积加权和预览保护边界。
- [Task 文档入口](../README.md) 与 [当前版本指针](../../../../../docs/current_versions.yaml)：加入/指向 V1.2.6 预览计划；未改其它 Task 版本。
- [预览脚本](../../research/hand_region_sampling/run.py)：实现 MANO/Inspire canonical triangle pool、Region 划分、固定配额采样和 PLY/manifest 输出。
- [实验定义](../../research/hand_region_sampling/experiment.yaml) 与 [实验 README](../../research/hand_region_sampling/README.md)：声明 exploratory、只输出预览、不接入正式 cache。
- [MANO 2048 点 PLY](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/mano_weighted_2048.ply)。
- [Inspire 2048 点 PLY](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/inspire_weighted_2048.ply)。
- [采样统计](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/sampling_summary.json) 与 [run manifest](../../research/hand_region_sampling/output/region_weighted_2048_preview_20260909/run_manifest.json)。

**原因**

现有 V1.2.5 的全局面积加权 1538 点使 Inspire 指尖覆盖不足。本次先用独立预览验证“固定 2048 点、显式 palm/body/tip 配额、Region 内仍按面积加权”的视觉分布，再决定是否改变正式数据语义。

**验证**

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/hand_region_sampling/run.py`：通过。
- 预览命令：`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/hand_region_sampling/run.py --output src/task/ObjectInteractionCm/research/hand_region_sampling/output/region_weighted_2048_preview_20260909`：完成，生成 MANO/Inspire 两份 PLY、统计和 run manifest。
- PLY 读取校验：两份均为 2048 vertices、11 Region 计数严格符合预设配额、点和法线 finite，最大单位法线误差小于 `4e-8`。
- 采样确定性校验：同一 seed=2024 下 MANO/Inspire 两次生成的点数组逐元素一致。
- 初次运行发现的 `smplx/chumpy` 与新 NumPy 别名兼容问题已通过预览入口本地 shim 解决；未改动生产模块。
- `git diff --check` 与 `audit_diff.py --staged --check-links`：通过；未运行训练、评估或全量数据处理。
- 回滚入口：删除新增的 [预览脚本](../../research/hand_region_sampling/run.py)、[实验定义](../../research/hand_region_sampling/experiment.yaml)、[计划](../plan/V1.2.6.md)、README 和对应 `output/region_weighted_2048_preview_20260909/`；V1.2.5 cache/config/checkpoint 不需回滚。

## 2026-09-09 00:08:25 +0800 — V1.2.5 核对 Inspire 三角面中心点数量

- activity_id: `ACT-20260909-000825-OBJECTINTERACTIONCM-INSPIRE-TRIANGLE-CENTER-COUNT`
- timestamp: `2026-09-09 00:08:25 +0800`
- modification_version: `V1.2.5`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户追问 1538 点总预算对指尖稀疏的影响，以及每个三角面中心作为点时的数量；本次只读复现正式 URDF mesh 的有效三角面计数。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅追加本诊断记录；保留工作树中已有改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（确认当前 1538 总预算显著压缩了三角面候选；“每面中心”会产生约 24.7 万点，但其三角剖分密度并不等于均匀几何采样）
- scope: 只读统计当前 Inspire URDF 的 visual mesh 三角面、退化面和每 link 分布；未修改代码、配置、cache、split、GT、训练或评估产物。

**文件与证据**

- [cache builder](../../tools/data/build_dexplore_rl_cache.py)：验证当前实现将所有有效三角面合并后按面积有放回抽取固定 1538 点。
- [V1.2.5 正式 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)：确认正式 cache 使用 `surface_points=1538`、`surface_seed=2024`。
- [本活动记录](activity_log.md)：登记本次只读诊断和保护边界。

**原因**

13 个 visual mesh 共 247,412 个三角面，其中 8 个退化面被当前代码过滤，得到 247,404 个有效三角面。因此保留每个三角面中心会产生 `247,404×3` 的点坐标数组，约为当前 1538 点的 160.9 倍。全量中心点还会继承 STL 三角剖分密度，不能直接视为均匀表面采样。

**验证**

- 有效三角面按大区统计：`hand_base=68,832`、`thumb=38,624`、`index=35,248`、`middle=34,006`、`ring=35,248`、`pinky=35,446`。
- 若保留所有面中心，五个末端 mesh 的点数分别为 `15,718/14,846/13,604/14,846/15,044`（拇/食/中/无名/小指），几何最末 10 mm 分别约有 `7,328/5,053/5,039/5,085/5,803` 个面中心；这说明全量中心点会显著改善末端覆盖，但也会带来约 2.97 MB 的单帧 float32 坐标，仅计算量和存储量就不再等同于当前训练合同。
- 未运行训练、评估或数据处理，未生成或修改 cache；回滚入口仅为删除本条 activity。

## 2026-09-08 23:58:38 +0800 — V1.2.5 为 Viser 查看器增加未来帧叠加

- activity_id: `ACT-20260908-235838-OBJECTINTERACTIONCM-VISER-FUTURE-OVERLAY`
- timestamp: `2026-09-08 23:58:38 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确确认未来步长 `delta` 范围为 0–10，0 表示不显示未来，其余按已协商方案叠加 `t+delta` 的手和物体；改动仅限 Task-local 只读可视化。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的鼠标 GUI、未来点云/mesh 场景节点及只读检查输出；不改变 cache、schema、GT、split、坐标系、采样、KNN、距离定义、训练或评估实现。
- conclusion: `SUPPORTED`（未来帧索引、末帧夹取及 MANO/Inspire-RL 服务端渲染通过工程验证；未进行浏览器端人工点击与主观观感回归，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：在“播放”区域加入 `未来 Δ（cache 帧）` 鼠标滑块，范围为 0–10；0 隐藏未来层，1–10 叠加 `min(t+Δ,T-1)` 的手和物体。未来物体使用绿色、未来手使用紫色，不绘制连线；未来点云和 mesh 分别服从既有 `点云显示`、`Mesh 显示` 控件，当前帧的累计距离阈值、最近距离和物体→手 KNN 语义保持不变；新增 `--future-delta` 启动/检查参数。
- [本活动记录](activity_log.md)：登记审批、边界、验证和回滚入口。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

需要在同一视图对比当前状态和指定未来 cache 帧的整体手物运动。实现复用训练 index 中同一条轨迹的逐帧点云与 mesh pose，不插值、不连接轨迹、不计算未来层的距离/KNN；在序列尾部显式夹到最后一帧并在状态区提示，避免越界或循环到序列开头。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py` 与 `_future_frame`/parser 断言：通过；确认 `Δ=0/1/6/10`、末帧夹取及参数范围。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --frame 0 --future-delta 3`：通过；MANO 当前 cache 帧 0 映射未来 cache 帧 3、原始帧 12，未夹取。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split val --sequence s1/torussmall_lift --frame 488 --future-delta 10`：通过；Inspire-RL 的 489 帧序列在末帧正确夹到 cache 帧 488、原始帧 1952，并报告 `future_clamped=true`。
- `timeout 12s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --frame 0 --future-delta 3 --point-display both --mesh-display both --host 127.0.0.1 --port 8134 --fps 1`：Viser 使用当前 V1.2.5 index 完成 MANO 当前/未来点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `timeout 15s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split val --sequence s1/torussmall_lift --frame 0 --future-delta 10 --point-display both --mesh-display both --host 127.0.0.1 --port 8135 --fps 1`：Viser 完成 Inspire-RL 当前/未来点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `timeout 7s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --future-delta 0 --point-display both --mesh-display both --host 127.0.0.1 --port 8136 --fps 1`：`Δ=0` 路径完成初始渲染，未来点云不可见且未来 mesh 不加载；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 7 个变更路径一致，4 个本地链接可导航。
- 回滚入口：仅恢复 [查看器](../../visualize_grab.py) 中未来帧 helper、GUI、四个未来 scene group 和 `--future-delta` 参数，并删除本条 activity；数据/cache 无需回滚。

## 2026-09-08 23:56:22 +0800 — V1.2.5 诊断 Inspire 指尖表面点稀疏

- activity_id: `ACT-20260908-235622-OBJECTINTERACTIONCM-INSPIRE-TIP-SAMPLING-DIAGNOSTIC`
- timestamp: `2026-09-08 23:56:22 +0800`
- modification_version: `V1.2.5`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前训练数据中 Inspire 手点的采样方式，并反馈可视化中指尖手点很少；本次仅只读核对 cache builder、URDF visual mesh、正式 cache manifest，并复现固定采样的区域计数。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅追加本诊断记录；保留工作树中已有的 ObjectInteractionCm 转换器、配置、查看器、训练记录及 CmDecoderv2 改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（确认当前 Inspire 固定表面采样确实造成指尖低覆盖；尚未做重采样或性能消融，不能据此断言它对指标的因果影响）
- scope: 只读检查 V1.2.5 Inspire 手点的 URDF visual surface sampling、固定点随 FK 变换的实现和 seed=2024 的逐 link/末端区域计数；未修改代码、配置、cache、split、GT、训练或评估产物。

**文件与证据**

- [cache builder](../../tools/data/build_dexplore_rl_cache.py)：从 13 个 URDF visual mesh 合并全部有效三角形，按三角形面积全局有放回抽取 1538 个三角形，再作重心采样；固定样本只在每帧随各 link FK 变换。
- [V1.2.5 正式 cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json) 与 [示例 Inspire geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)：确认 `surface_seed=2024`、`surface_points=1538`、`method=area_weighted_triangle_barycentric`。
- [V1.2.5 训练配置](../../configs/active/dexplore_rl_v1_2_5.yaml)：确认运行时手点数 1538、物体→手 KNN=16、交互半径 5 cm。
- [本活动记录](activity_log.md)：登记本次只读诊断、定量结果和保护边界。

**原因**

当前策略没有 per-link 最小配额、指尖加权或全局 FPS。`hand_base_link` visual mesh 的面积为 `1488.759 cm²`，占全部 visual surface 的 `73.818%`，因此固定 seed 实际将 1538 点中的 1109 点分配给掌/腕 base，五个含指尖表面的末端 mesh 合计只有 139 点。用户看到的指尖稀疏不是单帧渲染偶发，而是全部 Inspire 序列共享的固定采样布局。

**验证**

- 使用正式 builder 的 `InspireUrdfModel.surface_samples(1538, 2024)` 复现采样：掌/腕 base 1109 点；拇指各 link 合计 126 点（末端 28），食指 74（末端 27），中指 86（末端 40），无名指 78（末端 26），小指 65（末端 18），总计 1538。
- 沿各末端 mesh 轴向统计几何最末 10 mm：拇/食/中/无名/小指分别仅 `4/5/9/5/4` 点；该计数支持“指尖覆盖稀疏”，但不是模型性能消融。
- URDF 的五个 `*_tip` link 均为空 link、没有 visual mesh，采样器不会为 tip link 单独留点；另发现 `pinky_tip` fixed frame 与末端 mesh 的局部轴向明显不一致，但该 frame 未参与当前 surface sampling，因而不是本次稀疏的直接原因。
- 未运行训练、评估或数据处理，未生成或修改 cache；回滚入口仅为删除本条 activity。

## 2026-09-08 22:13:29 +0800 — V1.2.5 缩小点云 sprite 并改用 float32 圆点渲染

- activity_id: `ACT-20260908-221329-OBJECTINTERACTIONCM-VISER-FINE-POINT-RENDERING`
- timestamp: `2026-09-08 22:13:29 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 在只读诊断确认 Inspire 小物体的 4096 点密度、1 mm 最小 sprite 和 Viser 默认 float16 量化后，Agent 提议将滑块下限降至 0.1 mm 并使用 `float32/circle/flat`，用户明确回复“可以，你继续吧”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的 Viser 点云显示参数；不改变点云数组、采样数量、坐标系、KNN、距离、mesh、训练或评估合同。
- conclusion: `SUPPORTED`（静态检查和 Inspire-RL 小物体服务端 smoke；未进行浏览器端主观观感回归，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：手点和物体点滑块由最小 `0.001 m`/步长 `0.001 m` 改为最小 `0.0001 m`/步长 `0.0001 m`；两类点云均显式使用 `precision="float32"`、`point_shape="circle"`、`point_shading="flat"`，避免世界坐标约 1 m 时的 float16 毫米级量化和方形渐变 sprite 造成视觉膨胀。
- [本活动记录](activity_log.md)：登记审批、验证和回滚入口。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

V1.2.5 的 Inspire 小物体可小至约 40 mm，而 4096 点的典型间距约 0.44 mm；旧的最小 1 mm sprite 会明显重叠。Viser 1.0.30 默认还会把点坐标转换为 float16，在当前约 1 m 的世界坐标处量化粒度接近 1 mm。此次只收细渲染参数，不通过缩放坐标或改写采样来掩盖显示问题。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过；静态核对两个 slider 均为 `min=0.0001, step=0.0001`，两个 `add_point_cloud` 均传入 `float32/circle/flat`。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split val --sequence s1/torussmall_lift --point-size 0.0001 --point-display both --mesh-display off --host 127.0.0.1 --port 8133 --fps 1`：当前 V1.2.5 index 的 Inspire-RL 小物体轨迹以 0.1 mm 点大小完成 Viser 初始渲染；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；未生成或修改数据/cache/运行产物。
- 回滚入口：仅恢复 [查看器](../../visualize_grab.py) 的两个 slider 范围和两处 `add_point_cloud` 渲染参数；数据/cache 无需回滚。

## 2026-09-08 22:02:49 +0800 — V1.2.5 诊断 Inspire 点云看起来偏大的原因

- activity_id: `ACT-20260908-220249-OBJECTINTERACTIONCM-VISER-POINT-SIZE-DIAGNOSTIC`
- timestamp: `2026-09-08 22:02:49 +0800`
- modification_version: `V1.2.5`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问查看器物体点是否为采样点，并反馈 Inspire 视图在最小点大小下仍显得偏大；本条仅做只读数组、尺度和 Viser 渲染语义核对。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（只追加本诊断记录；保留工作树已有的训练、转换器、配置和查看器改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 只读检查 [近期 V1.2.5 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)、indexed geometry、缓存构建器和本地 Viser 1.0.30 的 `point_size` 实现；未修改数据、代码或渲染参数。
- conclusion: `SUPPORTED`（采样来源和单位诊断得到工程证据；尚未实施点大小修正，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：只读核对其加载完整 `obj_points_pool_world.npy`、点数和 `point_size` 传递。
- [V1.2.5 训练 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json) 与 [示例 geometry](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)：核对采样点池 shape/dtype、来源 manifest 和坐标单位。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

需要区分“数据点本身过大”和“渲染点 sprite/点密度相对物体尺寸显得过大”。本次检查同时对照 cache builder 的输入输出和 Viser 的本地渲染实现，避免未经证据就修改坐标或采样语义。

**验证**

- `obj_points_pool_world.npy` 在 MANO 和 Inspire-RL 序列均为 `[T,4096,3]`、`float32`，示例帧 4096 个点全部唯一；不是直接使用 mesh 顶点。近期 V1.2.5 index 的运行时模型点数为 1024，但查看器故意显示完整 4096 点池。
- [cache builder](../../tools/data/build_dexplore_rl_cache.py) 先读取父 cache 的 `obj_points_world.npy [T,4096,3]`，再按当前物体姿态写出 `obj_points_pool_world.npy`；Inspire 手点则由 1538 个固定面积加权表面样本生成。两者均为采样点云，不是原始 mesh 顶点。
- 对 V1.2.5 Inspire 示例核对：普通 `airplane` 物体尺寸约 `138×155×44 mm`；`cubesmall`/`torussmall` 等小物体约 `40 mm` 尺寸，4096 点的典型最近邻间距约 `0.44 mm`（torussmall）。因此查看器最小 `point_size=0.001 m` 仍是 1 mm，在小物体上会发生明显 sprite 重叠。
- 本地 Viser 1.0.30 的 `add_point_cloud` 文档和客户端 shader 均按场景单位解释 `point_size`，查看器当前物体/手点 slider 下限为 `0.001 m`，且默认 `point_shape="square"`、`point_shading="gradient"`；这会使密集 Inspire 点云看起来比 MANO 大，但不表示坐标单位错误。
- 未修改 `visualize_grab.py` 或任何 cache；若要改善观感，后续可在用户确认后把最小点大小降到 `0.0001 m`、改用 `circle/flat`，并可选择显示 1024 点可视化子集而保持 KNN 在完整池上计算。

## 2026-09-08 21:37:48 +0800 — V1.2.5 为点云与 Mesh 增加独立鼠标显示切换

- activity_id: `ACT-20260908-213748-OBJECTINTERACTIONCM-VISER-GEOMETRY-TOGGLES`
- timestamp: `2026-09-08 21:37:48 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认采用两个独立鼠标下拉控件，分别切换点云和 Mesh 的关闭、仅物体、仅手、物体+手显示状态。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的显示 GUI 与点云 handle 可见性；不改变数据、KNN 定义、距离计算、mesh 几何、训练或评估实现。
- conclusion: `SUPPORTED`（工程 smoke 证据；未进行浏览器端人工点击回归，不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：在同一“点云与 Mesh 显示”区域加入独立的 `点云显示` 和 `Mesh 显示` 下拉框；二者均支持关闭、仅物体、仅手和物体+手。点云切换只更新既有 point-cloud handle 的 `visible`，Mesh 仍按需加载，不重建点云数据或改变着色结果；同时新增可选的 `--point-display off|object|hand|both` 启动参数。
- [本活动记录](activity_log.md)：登记显示组合、验证和保护边界。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

此前点云始终显示，只有 Mesh 可以切换，无法快速在纯点云、纯 Mesh 和叠加视图之间比较。两个独立鼠标控件让物体和手的点云/mesh 可自由组合，同时保持轨迹、播放、距离阈值和 KNN 控件不变。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过；`--help` 显示 `--point-display {off,object,hand,both}` 与既有 mesh 参数。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split train --sequence s1/airplane_fly_1 --point-display off --mesh-display both --host 127.0.0.1 --port 8130 --fps 1`：Viser 启动并完成“关闭点云、显示物体+手 Mesh”的 MANO 初始渲染；timeout 主动退出，退出码 124。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split val --sequence s1/banana_lift --point-display hand --mesh-display off --host 127.0.0.1 --port 8131 --fps 1`：Viser 启动并完成“仅手点、关闭 Mesh”的 Inspire-RL 初始渲染；timeout 主动退出，退出码 124。
- `timeout 8s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json --split train --sequence s1/airplane_fly_1 --point-display off --mesh-display both --host 127.0.0.1 --port 8132 --fps 1`：使用当前 V1.2.5 正式训练 index（630 条序列）再次通过纯 Mesh 初始渲染；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；未生成或修改数据/cache/运行产物。

## 2026-09-08 20:36:03 +0800 — V1.2.5 为 Viser 查看器增加物体→手 KNN 高亮

- activity_id: `ACT-20260908-203603-OBJECTINTERACTIONCM-VISER-KNN-HIGHLIGHT`
- timestamp: `2026-09-08 20:36:03 +0800`
- modification_version: `V1.2.5`
- type: `code / diagnostic`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求增加 KNN 显示，提供 `1/4/8/16/32/64` 档位；该改动只增加 task-local 可视化，不改变训练、GT、cache、split 或指标合同。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有的训练、CmDecoderv2 及其他未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py` 的鼠标 GUI 和只读 KNN 诊断；不绘制连线、不写入数据/cache/output。
- conclusion: `SUPPORTED`（工程与数据语义 smoke 证据；不构成科研效果结论）

**文件**

- [Viser 查看器](../../visualize_grab.py)：新增 `物体→手 KNN` 下拉控件，档位为关闭、`n=1`、`n=4`、`n=8`、`n=16`、`n=32`、`n=64`；每帧对完整 4096 点物体池查询最近 n 个右手点，并将被任一物体点选中的手点取并集后以黄色高亮。距离阈值仍为红色，KNN 黄色在重叠点上优先显示；不创建线段。
- [本活动记录](activity_log.md)：登记 KNN 方向、颜色语义、验证和保护边界。
- 工作树中此前已有、此次未修改但需保护的 Task 路径：`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml`、`src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5_smoke.yaml`、`src/task/ObjectInteractionCm/docs/README.md`、`src/task/ObjectInteractionCm/docs/logs/experiment_log.md`、`src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py`。

**原因**

用户需要快速查看“物体附近的整体手点区域”，而不是逐条显示 KNN 连线。实现与当前 Cm 模型的 KNN 方向保持一致：物体点作为 query、手点作为候选邻居，对各物体点的最近 n 个手点取并集并着色；因此 n=1 表示每个物体点的第一近手点的并集，而非强行只保留一个手点。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split train --sequence s1/airplane_fly_1`：通过；KNN 并集手点数依次为 `n=1:1`、`4:6`、`8:10`、`16:22`、`32:37`、`64:73`，随 n 单调增加。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split val --sequence s1/banana_lift`：通过；KNN 并集手点数依次为 `n=1:1`、`4:4`、`8:9`、`16:18`、`32:33`、`64:70`，确认 Inspire-RL 轨迹同样使用物体→手方向。
- `timeout 12s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split train --sequence s1/airplane_fly_1 --host 127.0.0.1 --port 8129 --fps 1`：Viser HTTP/WebSocket 启动并完成初始点云渲染；timeout 主动退出，退出码 124。KNN 控件通过同一 GUI callback 接入，未绘制连线。
- `git diff --check -- src/task/ObjectInteractionCm` 与 `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；未生成或修改数据/cache/运行产物。

## 2026-09-08 09:46:19 +0800 — V1.2.5 修正轨迹、KNN=16 的 Cm 正式训练完成

- activity_id: `ACT-20260908-094619-OICM-CORRECTED-KNN16-TRAIN-COMPLETED`
- timestamp: `2026-09-08 09:46:19 +0800`
- modification_version: `V1.2.5`
- type: `experiment / operation`
- change_level: `L0`（既有已批准正式 run 的终态核对与记录，不改变科研变量或运行产物）
- approval: `user-approved`
- approval_basis: 用户已批准 V1.2.5 修正轨迹、KNN=16 的正式训练；本次用户询问当前状态，只读核对已结束的既有 run 并收口终态记录。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 查看器和 V1.2.5 未提交改动）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 只读核对 V1.2.5 正式 run 的进程、metrics、train log 与 checkpoint 元数据，并更新 Task activity/experiment 终态；不修改任何运行产物。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_5.yaml --distributed`
- last_step: `202300`
- last_epoch: `262`
- duration: `06:30:21`
- best_metric: `val/obj/flow_epe_mm=6.4225044410`（epoch 143 / step 110682，MANO `7.2434972881 mm`，RL-Inspire `5.6015115938 mm`）
- final_metric: `val/obj/flow_epe_mm=6.7061753591`（epoch 262 / step 202300，MANO `7.7338374220 mm`，RL-Inspire `5.6785132961 mm`）
- conclusion: `SUPPORTED`（修正数据与 KNN=16 的正式训练稳定完成）；下游 CmDecoderV2/rollout 效果仍为 `INCONCLUSIVE`

**产物与证据**

- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/config.json)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/metrics.jsonl) 与 [train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/best.pt)：epoch 143 / step 110682，SHA256 `a73b7dbf93cf4ca3b6de21e70c74acd22ba58d69ec8003d1c9fdae3f76180493`。
- [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/latest.pt)：epoch 262 / step 202300，SHA256 `f8bcd466021eebf3b5ee9f10e0e36091eb38d60a4a469feb5b381f1ab1db1899`。
- [实验记录](experiment_log.md) 与 [执行计划](../plan/V1.2.5.md)

**原因**

用户询问正式训练进展。进程已自然退出，需要区分 best 与 final validation，并将启动条目的运行中状态收口为唯一终态证据，避免后续 decoder 错误选择最后一轮 checkpoint。

**验证**

- `ps` 未发现该 run 的 torchrun、训练 rank 或 DataLoader 进程；GPU 0/1/2 利用率均为 0%。
- `train.log` 末尾为 `Training finished at step 202300 in 06:30:21.`；未检出 traceback、exception、NaN、OOM、NCCL failure 或 killed 记录。
- 解析 `metrics.jsonl` 共 262 次 validation：最小等权 source object EPE 位于 epoch 143 / step 110682；最终 validation 位于 epoch 262 / step 202300。
- 直接读取 best/latest checkpoint 元数据，确认 step、epoch 和共同保存的 `best_metric=6.422504440981543`；SHA256 已记录。
- 未修改 checkpoint、cache、配置或训练输出；旧错误轨迹 run 和所有既有用户改动均保留。

## 2026-09-07 23:58:32 +0800 — V1.2.5 full cache、KNN=16 smoke 与正式重训启动

- activity_id: `ACT-20260907-235832-OICM-CORRECTED-KNN16-FULL-TRAIN-START`
- timestamp: `2026-09-07 23:58:32 +0800`
- modification_version: `V1.2.5`
- type: `data / code / experiment / operation / documentation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认使用 630 条修正相交集重新训练 Cm，并追加要求 `KNN=16`；此前已说明独立 cache、scale、smoke、正式三卡训练和旧产物保护边界。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 查看器及旧 cache/output/checkpoint）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 新建修正轨迹 cache/index/scale、V1.2.5 配置和新训练 run；转换器只补充真实 RL root/version provenance；不修改旧训练变量之外的模型/loss、公共 `src/base`、旧数据或旧 checkpoint。
- cache_run_id: `oicm-dexplore-rl-full-20260907-231239`
- cache_run_status: `COMPLETED`
- smoke_run_id: `object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511`
- smoke_run_status: `COMPLETED`
- run_id: `object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606`
- run_status: `RUNNING`
- process: torchrun PID `2889418`，GPU `0,1,2`，DDP world size 3
- last_step: `4100`（人工检查于 2026-09-08 00:04:23 +0800；最近完整 checkpoint 为 step 3870 / epoch 5）
- last_epoch: `6`（运行中）
- best_metric: `val/obj/flow_epe_mm=7.9856659836`（epoch 4 / step 3096）
- conclusion: `INCONCLUSIVE`（正式训练运行中；cache 和 smoke 工程证据为 `SUPPORTED`）

**文件与产物**

- [当前版本指针](../../../../../docs/current_versions.yaml) 与 [Task 入口](../README.md)
- [执行计划](../plan/V1.2.5.md)、[正式配置](../../configs/active/dexplore_rl_v1_2_5.yaml)、[smoke 配置](../../configs/active/dexplore_rl_v1_2_5_smoke.yaml) 与 [cache 转换器](../../tools/data/build_dexplore_rl_cache.py)
- [full cache 目录](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/)、[cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/run_manifest.json)、[index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json)、[validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/validation_summary.json) 与 [KNN=16 scale](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/scales_train_v1_2_5.json)
- [smoke 运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/)、[smoke run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/run_manifest.json)、[smoke metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/metrics.jsonl)、[smoke train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/train.log) 与 [smoke latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_smoke_20260907_235511/checkpoints/latest.pt)
- [正式运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/config.json)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/train.log)、[best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/best.pt) 与 [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_5_20260907_235606/checkpoints/latest.pt)
- [实验记录](experiment_log.md)

**原因**

旧 OICM cache 使用 GRAB/reference object trajectory，不能监督 DExplore rollout 的实际物体运动。此次从修正后的 hand/object 联合轨迹重建全部 geometry 和 flow GT；`KNN=16` 同时进入 scale calibration 与模型局部 interaction，避免模型/统计口径不一致。

**验证**

- full cache `COMPLETED`：630 条互斥 parent sequence，train/val/test=`509/58/63`，train MANO/Inspire=`254/255`、val=`28/30`、test=`63/0`；index 与 run manifest 的 RL root 均为 `inspire_rl_object_dexplore`。
- scale calibration：train-only、seed 42、两 source stride 1..10、`knn_k=16`；`s_geo=0.0296379011 m`、`s_hand_flow=0.1148334428 m`、`s_obj_flow=0.0911242272 m`。
- Dataset probe：train/val/test frame samples=`74246/8232/9334`，batch object `[2,1024,3]`、hand `[2,1538,3]`，metadata/config 均为 KNN=16。
- 三卡 2-step smoke `COMPLETED`：global batch 6，loss/gradient finite，sample valid ratio 1，产生 config、manifest、metrics、train log 和 checkpoint；属于工程 `SUPPORTED`。
- 正式训练从随机初始化启动：global batch 96、202300 steps；截至 epoch 4 / step 3096，best equal-source object EPE `7.985666 mm`（MANO `8.277207`、RL-Inspire `7.694125`），best/latest checkpoint 已生成；运行未终止，科研结论保持 `INCONCLUSIVE`。
- `PYTHONPATH=. CUDA_VISIBLE_DEVICES='' ... pytest -q tests/test_object_interaction_cm.py`：`3 passed`；属于 Task 现有 legacy 定向回归。
- `git diff --check -- docs/current_versions.yaml src/task/ObjectInteractionCm`：通过。
- `audit_diff.py --worktree --scope-prefix <本次 8 个受控路径> --check-links`：通过；审计 7 个相对 HEAD 的变更路径及最新 activity 的 25 个现有本地链接。
- 回滚入口：停止该新 run 并移除 V1.2.5 新 cache/config/output；旧 cache、旧 checkpoint 和旧运行均未覆盖、可继续只读复核。

## 2026-09-07 23:03:24 +0800 — V1.2.5 修正物体轨迹、KNN=16 方案与 pilot

- activity_id: `ACT-20260907-230324-OICM-CORRECTED-TRAJECTORY-KNN16-PILOT`
- timestamp: `2026-09-07 23:03:24 +0800`
- modification_version: `V1.2.5`
- type: `data / code / experiment`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认使用 630 条修正数据相交集重新训练 Cm，并明确要求 `KNN=16`。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留既有 CmDecoderv2、ObjectInteractionCm 查看器、旧 cache、旧 output 和旧 checkpoint）
- final_plan: [V1.2.5 修正物体轨迹与 KNN=16 重训计划](../plan/V1.2.5.md)
- scope: 新增 V1.2.5 cache/config/plan；转换器只增加实际 `rl_root` provenance 和 modification-version 参数；不修改旧 cache、旧配置、旧 checkpoint 或公共运行时。
- run_id: `oicm-dexplore-rl-pilot-20260907-230232`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（pilot 数据合同和修正轨迹接入通过；full cache、scale 和正式训练尚未完成）

**文件与产物**

- [执行计划](../plan/V1.2.5.md)
- [full 配置](../../configs/active/dexplore_rl_v1_2_5.yaml) 与 [smoke 配置](../../configs/active/dexplore_rl_v1_2_5_smoke.yaml)：新训练使用 `knn_k=16` 和新 index/scale 路径。
- [cache 转换器](../../tools/data/build_dexplore_rl_cache.py)：index 的 `source_roots.inspire_rl` 改为记录实际 `--rl-root`；run manifest 的版本号由命令行锁定。
- [pilot run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5_pilot/run_manifest.json)、[pilot contact check](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5_pilot/pilot_contact_check.json) 和 [pilot geometry manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5_pilot/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)。

**原因与保护边界**

旧 cache 使用旧参考物体轨迹；修正根的 actual object pose 必须重新进入 object geometry、flow GT 和后续 Cm。`KNN=16` 是本轮用户确认的新模型变量，因此同步用于模型与 train-only scale calibration。旧数据、旧运行和依赖旧 checkpoint 的结果均保留，可直接回滚到旧入口。

**验证**

- `py_compile`：转换器与 calibration 脚本通过。
- pilot command：`build_dexplore_rl_cache.py --mode pilot --sequence s1/airplane_lift --variant inspire_rl --rl-root data/processed_data/inspire_rl_object_dexplore --refresh-rl-candidates`，通过；parent intersection `630`，pilot frames `432`，object pool `4096`，hand `1538`。
- pilot contact check：export 与 geometry active frames 均 `358`，disagreement `0`。
- pilot 数值核对：新 cache translation 与 corrected tensor 最大误差 `0`；与旧 tensor 最大序列平移差 `0.125691 m`。
- 尚未启动 full cache 或训练；当前 Viser 进程和 GPU 上其他进程未停止。

## 2026-09-07 22:37:40 +0800 — 核对修正后 DExplore RL 物体轨迹与旧 OICM cache provenance

- activity_id: `ACT-20260907-223740-OICM-CORRECTED-OBJECT-TRAJECTORY-DIAGNOSTIC`
- timestamp: `2026-09-07 22:37:40 +0800`
- modification_version: `V1.2.4`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求只读浏览已更新的 DExplore RL 数据并确认重新训练 Cm 的语义；本事件不修改数据、代码、配置、split、GT、checkpoint 或运行状态。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留用户已有 CmDecoderv2、ObjectInteractionCm 查看器和日志改动）
- scope: 修正后 `inspire_rl_object_dexplore`、旧 `inspire_rl`、现有 `object_interaction_cm_dexplore_rl_v1` cache/index 及转换脚本的只读 provenance 与数值核对。
- conclusion: `SUPPORTED`（确认旧 OICM cache 使用旧参考物体轨迹，未使用修正后的实际模拟物体轨迹；尚未执行新 cache 构建或重新训练）

**原因**

用户说明旧 DExplore RL 导出遗漏实际模拟物体轨迹，并要求在重新训练 Cm 前核对更新数据。由于物体轨迹直接决定 object flow GT、cache 几何和 checkpoint 解释，先进行只读 provenance 与数值检查，避免从旧 cache 恢复或覆盖旧证据。

**证据与发现**

- [修正后 RL 数据 manifest](../../../../../data/processed_data/inspire_rl_object_dexplore/manifest.json) 记录 660 条 DExplore-compatible 序列；`object_pose[198:205]` 与 `inspire_qpos[373:391]` 均来自确定性策略的实际模拟状态，660 条序列的物体位姿均相对 geometric reference 发生变化。
- [旧 OICM cache run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) 和 [旧 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json) 明确绑定 `data/processed_data/inspire_rl`，生成时间早于修正数据；旧转换脚本实际从 tensor `198:205` 构造 `obj_pose_world`。
- 对旧 index 中同时存在于修正根的 273 条 Inspire-RL 序列逐条核对：旧 cache 的平移与旧 `inspire_rl[:,198:201]` 最大误差为 `0`；旧/新每序列最大平移差的 median / p95 / max 为 `0.099516 / 1.378110 / 3.827840 m`。因此旧训练输入不是修正后的实际物体轨迹。
- 修正根共 660 条，其中与旧 parent index、GRAB geometry 和 DExplore geometry 同时相交 630 条；沿用当前 seed=42、parent split 和互斥 variant 分配算法时，预计 train/val/test 为 `509/58/63`，train MANO/Inspire=`254/255`、val=`28/30`、test=`63/0`。这与旧 `1004/126/125` split 数量不同，属于后续重建前必须确认的数据语义变化。
- 旧 cache、旧 OICM checkpoint 以及依赖该 checkpoint 的 CmDecoderv2/rollout/classifier 结果均保留，仅能解释旧错误数据链路；本诊断没有删除或覆盖这些可回滚证据。

**验证**

- 只读检查两个 RL 根的 tensor shape、序列集合、有限值及新 manifest；旧根 1335 条，修正后的 DExplore-compatible 根 660 条，tensor 均为 `[T,598]`。
- 只读运行现有 `_intersection` 与 `_assign_variants` 得到 630 条相交序列及上述预计 split/variant 计数；没有写出 assignment、cache 或 manifest。
- `ps` 检查未发现 OICM 训练或 cache builder 进程；未启动训练。现有 ObjectInteractionCm Viser 进程保持不变。

## 2026-09-07 20:35:48 +0800 — V1.2.4 将 Viser 查看器对齐近期训练索引与两类手 mesh

- activity_id: `ACT-20260907-203548-OBJECTINTERACTIONCM-TRAINING-VISER-PROVENANCE`
- timestamp: `2026-09-07 20:35:48 +0800`
- modification_version: `V1.2.4`
- type: `code`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户质疑查看器是否使用近期训练数据；Agent 说明旧实现默认读取旧 `cm_object_v2/grab` cache，并提出改为近期训练 `index.json`、按 index variant 加载 MANO/Inspire-RL mesh 的方案，用户回复“可以”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（仅修改 ObjectInteractionCm 查看器和本日志；保留用户已有 CmDecoderv2 改动及未跟踪文件）
- final_plan: [V1.2.3 Dexplore RL 混合训练计划](../plan/V1.2.3.md)
- scope: `src/task/ObjectInteractionCm/visualize_grab.py`；只读消费近期训练 index、逐序列 geometry manifest/数组、GRAB canonical object mesh、MANO parent mesh 和 Inspire RL qpos/URDF，不写入或迁移数据。
- conclusion: `SUPPORTED`（工程静态检查、几何 provenance 检查和服务端 smoke；未进行浏览器端人工点击回归，不构成科研效果结论）

**文件与数据入口**

- [Viser 查看器](../../visualize_grab.py)：默认入口改为 [近期训练 index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json)，轨迹严格按 index 的 split/source/variant 枚举；点云直接 mmap 每条 indexed sequence 的训练 geometry。
- [近期训练配置](../../configs/active/dexplore_rl_v1_2_3.yaml)：其 `data.index` 与查看器新默认值一致。
- [MANO 检查序列 manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/train/mano/s1_airplane_fly_1/geometry/manifest.json) 与 [Inspire-RL 检查序列 manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/sequences/val/inspire_rl/s1_banana_lift/geometry/manifest.json)：记录本次 smoke 的 split、source、variant、坐标系、parent cache 或 RL qpos 来源。
- [本活动记录](activity_log.md)：登记语义修正、审批、验证和回滚入口。

**原因**

旧查看器默认读取历史 `data/processed_data/cm_object_v2/grab` 双手点云，与近期混合训练使用的 indexed MANO/Inspire-RL 单变体数据不是同一输入合同。此次修正让可视化证据与实际训练 index、逐序列 manifest 和坐标系一致，并在界面中显式暴露 provenance。

**实现与保护边界**

- GUI 仍全部使用鼠标，保留轨迹前后切换、split/轨迹下拉、播放/暂停、逐帧跳转、1–5 cm 累计阈值着色、最近手物点距离、点大小/FPS 和物体/手 mesh 选择控件。
- 查看器显示每帧完整 4096 点物体 pool 与 1538 点右手；状态栏明确提示训练运行时从 4096 pool 确定性采样 1024 点。最近距离和阈值着色基于所显示的完整训练 cache 点云。
- 物体 mesh 使用 GRAB canonical mesh 和该 indexed sequence 的 `obj_pose_world`。MANO 手 mesh 从 manifest 的 `input_parent_cache` 读取，并复现 cache builder 的旧物体系到新 Dexplore 物体姿态变换；Inspire-RL 手 mesh 从 manifest 的 `rl_q.tensor` 读取 native qpos，并复用 cache builder 的 `InspireUrdfModel`、joint reorder 与 FK。
- mesh 依赖缺失时只在状态栏降级提示，点云查看不受影响。未修改 index、geometry 数组、cache/schema、split、GT、训练配置、checkpoint、output、公共 `src/base` 或 `src/task/CmDecoderv2/`。
- 回滚入口：恢复本条目前的 `visualize_grab.py` 旧 root-based loader；所有消费的数据和 mesh 资产均为只读，无数据产物需要回滚。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split train --sequence s1/airplane_fly_1`：通过；确认 `train/grab/mano`、279 帧、4096/1024 物体点合同、1538 右手点；物体 mesh 与点云 bbox 中心误差 `0.1184 mm`，MANO mesh 与点云误差 `0.8879 mm`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --split val --sequence s1/banana_lift`：通过；确认 `val/inspire_f1/inspire_rl`、578 帧、4096/1024 物体点合同、1538 右手点；物体 mesh 与点云 bbox 中心误差 `0.0978 mm`，13 个 Inspire visual mesh（742,236 顶点）与 cache 手点云误差 `2.0421 mm`。
- `timeout 12s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split train --sequence s1/airplane_fly_1 --mesh-display both --host 127.0.0.1 --port 8127 --fps 1`：Viser HTTP/WebSocket 启动并完成 MANO 点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `timeout 15s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --split val --sequence s1/banana_lift --mesh-display both --host 127.0.0.1 --port 8128 --fps 1`：Viser HTTP/WebSocket 启动并完成 Inspire-RL 点云及双 mesh 初始渲染；timeout 主动退出，退出码 124。
- `git diff --check -- src/task/ObjectInteractionCm`：通过；未生成或修改数据/cache/运行产物。

## 2026-09-07 10:28:50 +0800 — V1.2.3 为 GRAB Viser 查看器增加物体/手 mesh 控件

- activity_id: `ACT-20260907-102850-OBJECTINTERACTIONCM-GRAB-VISER-MESH`
- timestamp: `2026-09-07 10:28:50 +0800`
- modification_version: `V1.2.3`
- type: `code`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求继续为现有查看器增加 mesh 控件，可选择加载物体或手 mesh；用户在确认仓库已有 mesh 来源后多次回复“继续”。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（只修改 ObjectInteractionCm 查看器及本日志，保留用户已有 CmDecoderv2 改动）
- scope: `src/task/ObjectInteractionCm/visualize_grab.py`；只读对齐现有 sampled-point cache、mesh/object-pose cache 与 GRAB canonical object mesh，不改变坐标系、cache/schema、GT、split、训练或评估实现。
- conclusion: `SUPPORTED`（工程 smoke 证据；不构成科研效果结论）

**文件**

- [GRAB Viser 查看器](../../visualize_grab.py)：新增 `Mesh 显示` 鼠标控件（关闭/仅物体/仅手/物体+手）、透明度控件、raw frame 对齐、缺失 mesh 降级提示，以及 `--mesh-root`、`--object-mesh-root`、`--mesh-display`、`--mesh-opacity` 参数。
- [本活动记录](activity_log.md)：登记 mesh 来源、坐标处理、验证和保护边界。

**原因**

现有点云 cache 不保存 mesh；仓库已有按序列保存的 [GRAB mesh/object-pose cache](../../../../../data/processed_data/cm_object_v2_mesh_object_pose_20260830/) 和 [GRAB canonical object meshes](../../../../../data/raw_data/GRAB/tools/object_meshes/contact_meshes/)。查看器按序列相对路径和 `raw_frame_id` 对齐两类 cache：手 mesh 直接读取世界系逐帧顶点，物体 canonical mesh 通过对应帧 `obj_pose_world` 放置到世界系。mesh cache 缺失时不阻止点云查看，只在状态栏报告不可用原因。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift`：通过；物体 banana mesh 为 48,370 顶点/96,736 面，左右手各 778 顶点/1,538 面，物体 mesh 与 sampled cloud 的首帧 bbox 中心误差为 `0.0977 mm`。
- `timeout 10s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift --mesh-display both --host 127.0.0.1 --port 8127 --fps 1`：Viser HTTP/WebSocket 启动并完成物体+双手 mesh 初始渲染，timeout 主动退出（退出码 124）。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --sequence s1/doorknob_use_1`：无对应 mesh cache 时返回 `mesh.available=false` 和明确错误，点云数据仍正常加载。
- Viser handle 定向 smoke 通过 mesh `vertices`、`opacity`、`visible`、`wxyz`、`position` 更新；`git diff --check` 通过。未进行浏览器端人工点击回归，未修改任何数据/cache 或生成运行产物。

## 2026-09-06 20:29:23 +0800 — V1.2.3 新增 GRAB sampled point-cloud Viser 查看器

- activity_id: `ACT-20260906-202923-OBJECTINTERACTIONCM-GRAB-VISER-VIEWER`
- timestamp: `2026-09-06 20:29:23 +0800`
- modification_version: `V1.2.3`
- type: `code`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求脚本放入 ObjectInteractionCm、使用 Viser、全部通过鼠标 GUI 操作、采用累计距离阈值着色，并支持轨迹/帧播放控制。
- skills_used: `research-change-control`
- branch: `oyx`
- base_commit: `d550810d9c91aa7738ee5f2cc8f20c5503798189`
- worktree_dirty: `true`（保留用户已有的 CmDecoderv2 修改和未跟踪文件，未触碰）
- scope: `src/task/ObjectInteractionCm/visualize_grab.py`；只读消费现有 `data/processed_data/cm_object_v2/grab` 点云 cache，不改变 cache、schema、GT、split、坐标语义或训练代码。
- conclusion: `SUPPORTED`（工程 smoke 证据；不构成科研效果结论）

**文件**

- [GRAB sampled point-cloud Viser 查看器](../../visualize_grab.py)：新增鼠标 GUI 轨迹切换、播放/暂停、逐帧前后跳转、左右手选择、1–5 cm 累计阈值高亮、点大小/FPS 控件和每帧左右手最近距离显示；提供 `--check-only` 只读检查入口。
- [本活动记录](activity_log.md)：登记实现范围、审批、验证与保护边界。

**原因**

为便于检查 GRAB 双手与物体的采样点云关系，需要一个不依赖键盘快捷键的浏览器界面。训练用 `_SequenceView` 会强制要求 object pose，而当前 GRAB 点云 cache 已有世界系物体/手点云但没有 pose 文件；查看器因此使用只读轻量 loader，仅读取点云、raw frame 和 metadata，不扩展或改写 cache schema。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py` 通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift` 通过：578 帧、物体 4096 点、左右手各 1538 点；首帧最近距离左 `1370.14 mm`、右 `1359.92 mm`。
- `timeout 5s /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --root data/processed_data/cm_object_v2/grab --sequence s1/banana_lift --host 127.0.0.1 --port 8124 --fps 1` 通过启动 Viser HTTP/WebSocket 服务并完成初始渲染（timeout 主动退出，退出码 124）；未进行浏览器端人工交互回归。
- `git diff --check` 通过；未修改或生成数据/cache、checkpoint、训练输出和公共 `src/base` 文件。

## 2026-09-06 12:01:31 +0800 — V1.2.3 核对 best.pt 的模型选择指标

- activity_id: `ACT-20260906-120131-OBJECTINTERACTIONCM-BEST-METRIC-DIAGNOSTIC`
- timestamp: `2026-09-06 12:01:31 +0800`
- modification_version: `V1.2.3`
- type: `diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问 ObjectInteractionCm 的 best checkpoint 是否按 loss 选择；只读检查配置、runner、BaseRunner 和冻结 checkpoint。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: `true`（保留用户已有改动，未修改代码、配置或 checkpoint）
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（确认 checkpoint 选择规则；不构成模型效果结论）
- scope: `src/task/ObjectInteractionCm/`、`src/base/base_runner.py` 及冻结 OICM run 的 config/checkpoint 元数据

**原因**

需要确认当前送入 CmDecoderv2 的 OICM `best.pt` 是按总 loss 还是按 object-flow 验证指标保存，避免误解 checkpoint 的优化目标。

**验证**

- [OICM config](../../configs/active/dexplore_rl_v1_2_3.yaml) 与 [run config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json) 均显示 `metric_for_best: val/obj/flow_epe_mm`、`lower_is_better: true`。
- [ObjectInteractionCm runner](../../runner.py) 的 `evaluate_all()` 将 `val/grab/obj/flow_epe_mm` 与 `val/inspire_f1/obj/flow_epe_mm` 做等权算术平均，写入 `val/obj/flow_epe_mm`；不按样本/帧数加权。
- [BaseRunner best 保存逻辑](../../../../../src/base/base_runner.py) 在每次 validation 后按该指标取更小值保存 `best.pt`；`val/loss` 只记录，不作为当前 run 的 best 选择指标。
- 冻结 best.pt 记录的 `best_metric=9.422408395136284`，对应 `val/obj/flow_epe_mm`。

## 2026-09-06 11:55:07 +0800 — V1.2.3 按用户要求停止 OICM 续训并冻结 best.pt

- activity_id: ACT-20260906-115507-OBJECTINTERACTIONCM-STOP-BEST-FREEZE
- timestamp: 2026-09-06 11:55:07 +0800
- modification_version: V1.2.3
- type: operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求停止当前 OICM 续训，并使用 best.pt 启动 CmDecoderv2；仅停止该 run，不删除或覆盖已有产物
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 27316ef8e9552b7b335e53400453902d745b1ebc
- worktree_dirty: true（保留用户已有工作区改动）
- run_id: `object-interaction-cm-dexplore-rl-v1-2-3-20260905-234051`
- run_status: STOPPED
- scope: `src/task/ObjectInteractionCm/`；仅停止既有 OICM run 并冻结其 best checkpoint，不改变上游代码、数据 cache 或旧运行产物
- last_step: 185850
- last_epoch: 105
- best_metric: `val/obj/flow_epe_mm=9.422408395136284`（best.pt step 122130 / epoch 69）
- stop_reason: 用户要求切换到 CmDecoderv2 训练；向 torchrun 主进程 481539 发送 SIGTERM，并确认 OICM 训练进程及其子进程全部退出
- frozen_checkpoint_sha256: `fde9984a79caff801ea06b566b1ee0f387944b4662909e4d5986c2e464b45b26`

**原因**

用户要求结束当前 OICM 续训并将 `best.pt` 直接作为 CmDecoderv2 的冻结上游；继续让 OICM 运行会改变 decoder 的输入版本，破坏 checkpoint provenance。

**产物与证据**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)
- [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)
- [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)

**验证**

- `src/task/ObjectInteractionCm/` — 本 Task 的既有配置、缓存构建工具、文档和实验记录均保留；本次仅追加停止记录。
- `ps` 复核 torchrun 主进程、3 个 rank 和 DataLoader 子进程均已退出；`sha256sum` 复核 best.pt 与 decoder 配置 gate 相同。
- latest.pt 保留在停止时的 step 185850 / epoch 105，可作为后续恢复入口；此停止条目不宣称 OICM 已完全收敛。

**验证与边界**

- 进程核对：OICM torchrun、3 个 worker 及 DataLoader 子进程均已退出；未触碰 GPU 4–7 上的其他任务。
- `sha256sum` 复核 best.pt 与配置 gate 一致；best.pt 未被续训覆盖（mtime 2026-09-06 09:52:45 +0800）。
- `nvidia-smi`：GPU 0/1/2 仅有残留显存约 2.5/5/6 MiB，利用率为 0%，可用于后续三卡 decoder run。
- 该条目记录停止操作和 checkpoint 冻结，不代表 OICM 或 decoder 的科研效果结论；旧 checkpoint、cache、数据和运行目录均保留，可从 latest.pt 恢复。

## 2026-09-06 09:38:15 +0800 — V1.2.3 从 latest checkpoint 续训启动

- activity_id: `ACT-20260906-093815-OICM-DEXPLORE-RL-FULL-RESUME-STARTED`
- timestamp: 2026-09-06 09:38:15 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1（沿用既有 final plan 和配置恢复正式训练；不改变科研变量）
- approval: user-approved
- approval_basis: 用户明确要求“先续训吧”。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 同一 V1.2.3 run、同一 index/config/seed/batch/stride，GPU 0/1/2 从 `latest.pt` step 115050 恢复至绝对 stop step 202300。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `RUNNING`
- resume_from: `outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt`（epoch 65 / step 115050）
- conclusion: INCONCLUSIVE（续训刚启动；终态前不作科研效果结论）

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [resume config snapshot](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config_resume_20260906_093911.json)、[resume metadata](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metadata_resume_20260906_093911.json)、[resume run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest_resume_20260906_093911.json)
- [resume checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)、[best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

原正式 run 在 step 115900 无终止标记地停止；latest checkpoint 含 model/optimizer/scheduler/scaler，用户要求先从该点继续完成既定 202300-step 预算。

**验证**

- `torch.load(latest.pt, map_location=cpu)` 通过，确认 step `115050`、epoch `65`、best metric `9.470077`，optimizer/scheduler/scaler 均存在。
- 启动前 `ps` 无旧训练进程；GPU 0/1/2 分别约 2.55/0.005/0.006 GiB，占用可用。
- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml --set train.resume=outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt --distributed`
- 恢复后 launcher/rank 正常运行并推进到 step `115600`；GPU 0/1/2 utilization 约 `86%/77%/82%`，恢复后的 `metrics.jsonl`、`train.log` 持续更新。

## 2026-09-06 09:25:13 +0800 — V1.2.3 收敛性诊断

- activity_id: `ACT-20260906-092513-OICM-DEXPLORE-RL-CONVERGENCE-DIAGNOSTIC`
- timestamp: 2026-09-06 09:25:13 +0800
- modification_version: V1.2.3
- type: diagnostic
- change_level: L0（只读分析既有 metrics/checkpoint，并记录证据；不启动续训）
- approval: user-approved
- approval_basis: 用户询问“目前收敛了吗”。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 分析 V1.2.3 的 epoch-level train/validation 曲线、最佳/最后 checkpoint、source-specific EPE 和停止状态。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `STOPPED`（诊断不改变既有终态）
- conclusion: INCONCLUSIVE（validation 已平台化，但训练未达到计划终点，不能宣称完全收敛或最终效果成立）

**文件**

- [metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)、[best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

需要区分“validation 是否达到平台”和“训练是否完成/优化是否仍在变化”，避免把中断在 57.3% steps 的 run 误报为收敛。

**验证**

- 使用 Python 读取 `metrics.jsonl` 的 65 个 validation epoch，统计最佳点、最近窗口均值/斜率和 source-specific 曲线；未改动 metrics 或 checkpoint。
- 最佳 epoch 46 / step 81420：equal-source object EPE `9.470077 mm`；最后完整 validation epoch 65 / step 115050：`9.574271 mm`。
- 最后 10 个 validation 的 equal-source均值 `9.601506 mm`、首末 `9.631856→9.574271 mm`；最后 20 个均值 `9.644924 mm`，在 `9.47–9.89 mm` 间震荡。
- train object EPE 从 epoch 1 的 `30.27025 mm` 降至 epoch 65 的 `18.32059 mm`，train loss 从 `0.03751` 降至 `0.02047`；说明优化曲线仍在下降，不能称完全收敛。
- 最后 20 个 validation 的 MANO object EPE 均值 `6.466792 mm`、RL-Inspire `12.823055 mm`，差距约 `6.36 mm`，source/domain 差异仍稳定存在。

## 2026-09-06 08:09:59 +0800 — V1.2.3 全量训练非正常停止终态核对

- activity_id: `ACT-20260906-080959-OICM-DEXPLORE-RL-FULL-STOPPED`
- timestamp: 2026-09-06 08:09:59 +0800
- modification_version: V1.2.3
- type: operation / diagnostic
- change_level: L0（只读核对并记录已发生的运行终态；未恢复或改变训练）
- approval: user-approved
- approval_basis: 用户询问“现在怎么样了”；终态记录继承 V1.2.3 final plan。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 核对 V1.2.3 正式 run 的进程、GPU、日志、metrics、checkpoint 和系统 OOM 线索；不自行续跑。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `STOPPED`
- last_step: 115900 / 202300
- last_epoch: 66（最后完整 validation/checkpoint 为 epoch 65）
- best_metric: `val/obj/flow_epe_mm=9.470077`，epoch 46 / step 81420
- best_checkpoint: `checkpoints/best.pt`
- latest_checkpoint: `checkpoints/latest.pt`，epoch 65 / step 115050
- exit_reason: 进程与三个 rank 均已消失，日志于 2026-09-06 03:16:50 +0800 无终止标记地停止；未发现 Python traceback、CUDA OOM 或对应时段 kernel/journal OOM，准确外部信号未知。
- conclusion: INCONCLUSIVE（完成约 57.3% 计划 steps，存在可恢复 checkpoint，但正式训练未完成）

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

旧 activity 仍标记 `RUNNING`，但用户查询时 GPU 已空闲且进程不存在；必须以实际日志和 checkpoint 更新唯一活动时间线，避免把中断 run 误报为完成。

**验证**

- `ps` 未找到 launcher/rank/DataLoader 进程；GPU 0/1/2 utilization 均为 0%，显存回到旧可视化进程占用约 2.55/0.005/0.006 GiB。
- `train.log`/`metrics.jsonl` 最后更新时间均为 03:16:50，最后记录 step 115900 / epoch 66；没有 `Training finished`、`Training failed`、`Traceback` 或 OOM 标记。
- 65 次完整 validation 中，最佳 equal-source object EPE 为 `9.470077 mm`（MANO `6.301521 mm`、RL-Inspire `12.638633 mm`）；最后完整 validation 为 `9.574271 mm`。
- 本地 `best.pt` 和 `latest.pt` 均可读取且包含 model/optimizer/scheduler/scaler；latest 为 step 115050 / epoch 65，可作为后续同配置恢复入口。
- 未恢复训练、未修改 config、数据、GT、split 或 checkpoint。

## 2026-09-05 23:54:31 +0800 — V1.2.3 全量训练人工状态检查

- activity_id: `ACT-20260905-235431-OICM-DEXPLORE-RL-FULL-PROGRESS`
- timestamp: 2026-09-05 23:54:31 +0800
- modification_version: V1.2.3
- type: operation / diagnostic
- change_level: L0（只读检查进程、日志、指标和 checkpoint，并更新运行记录）
- approval: user-approved
- approval_basis: 用户询问“现在还在训练吗”。
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: 只读核对 V1.2.3 正式 run 的进程、GPU、实时日志、validation 指标和 checkpoint 状态；不改变训练。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `RUNNING`
- last_step: 7100 / 202300
- last_epoch: 5（epoch 4 validation 已完成）
- best_metric: `val/obj/flow_epe_mm=10.283183`，epoch 4 / step 7080
- conclusion: INCONCLUSIVE（训练仍在早期运行；当前 validation 只作进度证据）

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/)、[run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- [best checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/best.pt)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/checkpoints/latest.pt)
- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)、[experiment log](experiment_log.md)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)

**原因**

核对用户询问时训练是否真实存活，而不是只依据旧 activity 的 `RUNNING` 文本判断。

**验证**

- launcher PID 26636 和三个 rank PID 26747/26748/26749 均存活；rank 进程状态为 `Rsl`。
- GPU 0/1/2 utilization 为 89%/85%/88%，显存约 8.98/6.43/6.43 GiB。
- `metrics.jsonl` 与 `train.log` 在 23:54:21 继续更新；未发现 `Traceback`、OOM 或 training failed。
- 最近 validation：MANO object EPE `7.151705 mm`，RL-Inspire object EPE `13.414660 mm`，equal-source `10.283183 mm`；sample valid ratio `0.997809`。
- 当前吞吐约 `1026 samples/s`，runner ETA 约 `5.04 h`。

## 2026-09-05 23:36:08 +0800 — V1.2.3 Dexplore RL/MANO scale 校准启动

- activity_id: `ACT-20260905-233608-OICM-DEXPLORE-RL-SCALE-STARTED`
- timestamp: 2026-09-05 23:36:08 +0800
- modification_version: V1.2.3
- type: data / operation
- change_level: L1（只生成 train-only scale JSON，并修正 stride policy metadata；不改变 geometry、GT 或 split）
- approval: user-approved
- approval_basis: 用户确认按 V1.2.3 默认训练方案执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有 CmDecoder 与 ObjectInteractionCm 用户改动）
- scope: 新 index 的两 source 统一 30 Hz stride `[1..10]`；从 train split 重新校准 `s_geo/s_hand_flow/s_obj_flow`。
- run_id: `oicm-dexplore-rl-scale-v1_2_3-20260905-233608`
- run_status: `COMPLETED`
- conclusion: SUPPORTED（train-only scale 统计和 20 个 source/stride group 产出通过；不构成 Cm 效果结论）

**命令与 PENDING 产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.tools.data.calibrate_scales --index data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json --output data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json --max-sequences-per-source 32 --frames-per-sequence-stride 4 --radius-m 0.05 --knn-k 8 --seed 42 --grab-strides 1 2 3 4 5 6 7 8 9 10 --inspire-strides 1 2 3 4 5 6 7 8 9 10`
- [scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json) — `s_geo=0.03182924`、`s_hand_flow=0.09928792`、`s_obj_flow=0.07794207`，20 个 group。
- 验证：index `stride_policy` 与 scale sampling 均为两 source `[1..10]`；输出 JSON 可解析，输入仅为 train split。

## 2026-09-05 23:38:44 +0800 — V1.2.3 三卡 smoke 启动

- activity_id: `ACT-20260905-233844-OICM-DEXPLORE-RL-SMOKE-STARTED`
- timestamp: 2026-09-05 23:38:44 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1（Task-local 训练配置与运行；不修改模型科研语义或旧输出）
- approval: user-approved
- approval_basis: 用户确认按 V1.2.3 final plan 执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: Dexplore RL/MANO right-hand mixed Cm；三卡 GPU 0/1/2；smoke 每卡 batch 2、global batch 6、2 steps；from scratch。
- run_id: `oicm-dexplore-rl-smoke-20260905-233844`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（smoke 运行中；不构成科研效果结论）

**命令与 PENDING 产物**

- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3_smoke.yaml --distributed`
- [smoke output](../../../../../outputs/objectinteractioncm/) — PENDING，具体 timestamp run directory 待 runner 创建。

## 2026-09-05 23:39:47 +0800 — V1.2.3 smoke 完成

- activity_id: `ACT-20260905-233947-OICM-DEXPLORE-RL-SMOKE-COMPLETED`
- timestamp: 2026-09-05 23:39:47 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1
- approval: user-approved
- approval_basis: V1.2.3 final plan。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（三卡工程链路通过；仅为 smoke，不代表 Cm 科研效果）

**验证与产物**

- 三卡命令使用 `CUDA_VISIBLE_DEVICES=0,1,2`、world size 3、每卡 batch 2、global batch 6；完成 2 steps，无 NaN/Inf。
- `sample/valid_ratio=1`、`sample/sampling_miss_ratio=0`、slot effective count=16；loss、梯度和 metrics 均写出。
- [smoke output](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/) — 含配置、metadata、run manifest、metrics、train log。
- [smoke metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/metrics.jsonl)
- [smoke train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/train.log)
- [smoke checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_smoke_20260905_233936/checkpoints/latest.pt)

## 2026-09-05 23:40:15 +0800 — V1.2.3 全量 Cm 训练启动

- activity_id: `ACT-20260905-234015-OICM-DEXPLORE-RL-FULL-STARTED`
- timestamp: 2026-09-05 23:40:15 +0800
- modification_version: V1.2.3
- type: experiment / operation
- change_level: L1（Task-local 正式训练）
- approval: user-approved
- approval_basis: smoke 已通过，用户确认按 V1.2.3 final plan 执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有用户改动）
- scope: Dexplore RL/MANO mixed right-hand Cm；from scratch；三卡 GPU 0/1/2；每卡 batch 32、global batch 96、202300 steps。
- run_id: `object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051`
- run_status: `RUNNING`
- conclusion: INCONCLUSIVE（正式训练进行中；终态前不作科研效果结论）

**命令与 PENDING 产物**

- 命令：`CUDA_VISIBLE_DEVICES=0,1,2 /home2/wyy/miniconda3/envs/graspenv/bin/torchrun --standalone --nproc_per_node=3 -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/dexplore_rl_v1_2_3.yaml --distributed`
- [full output](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/) — 当前运行目录。
- [full config](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/config.json)
- [full run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/run_manifest.json)
- [full metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/metrics.jsonl)
- [full train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_dexplore_rl_v1_2_3_20260905_234051/train.log)
- 当前证据：step 500/202300，约 1091 samples/s，ETA 约 4.9 h；GPU 0/1/2 均约 87% utilization，显存分别约 8.98/6.43/6.43 GiB。

**文件**

- [V1.2.3 final plan](../plan/V1.2.3.md)、[formal config](../../configs/active/dexplore_rl_v1_2_3.yaml)、[smoke config](../../configs/active/dexplore_rl_v1_2_3_smoke.yaml)
- [scale calibrator](../../tools/data/calibrate_scales.py)、[Dexplore cache producer](../../tools/data/build_dexplore_rl_cache.py)、[Dexplore V1.2.2 plan](../plan/V1.2.2.md)
- [experiment log](experiment_log.md)
- [scale manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/scales_train_v1_2_3.json)

**原因**

新 cache 的 MANO 与 RL-Inspire 都是 30 Hz；训练前需要把旧 Inspire 偶数 stride 口径替换为两 source 统一 `[1..10]`，并用 train-only 统计重新校准 scale，避免把旧数据跨度或统计量带入正式实验。

**验证**

- 配置加载、`py_compile`、index JSON 解析和 `git diff --check` 已通过。
- 两 step 三卡 smoke 已 `SUPPORTED`；正式 run 当前 `RUNNING`，终态前不作科研效果结论。

## 2026-09-05 23:38:59 +0800 — V1.2.3 smoke 首次启动失败（配置语法）

- activity_id: `ACT-20260905-233859-OICM-DEXPLORE-RL-SMOKE-FAILED`
- timestamp: 2026-09-05 23:38:59 +0800
- modification_version: V1.2.3
- type: operation
- change_level: L1
- approval: user-approved
- approval_basis: 继承 V1.2.3 final plan。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- run_id: `oicm-dexplore-rl-smoke-20260905-233844`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（YAML 未加引号的 `off` 被解析为布尔 `false`，runner 在 PerformanceMonitor 初始化阶段拒绝；未进入 forward/backward，无 checkpoint 或科研证据生成）
- 保护边界：三 rank 已正常拉起并退出；旧 output、cache、checkpoint 和 GPU 0 可视化进程未修改。
- 修复：smoke 配置改为 `performance.mode: "off"`，随后以同一三卡命令重跑。

## 2026-09-05 21:40:32 +0800 — Dexplore RL-Inspire 右手数据转换启动

- activity_id: `ACT-20260905-214032-OICM-DEXPLORE-RL-CONVERSION`
- timestamp: 2026-09-05 21:40:32 +0800
- modification_version: V1.2.2
- type: data_change / operation
- change_level: L2（sequence split、坐标系、GT geometry cache 与兼容 index）
- approval: user-approved
- approval_basis: 用户最新确认“按默认执行”；默认方案已冻结为右手、GRAB 母 split、train/val 近似 1:1 variant、test MANO、Dexplore RL source。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有 CmDecoder 用户改动）
- scope: 新增 Dexplore RL-Inspire→1538 点几何转换脚本和 V1.2.2 final plan；先执行单序列 pilot，旧 cache/index/split 不覆盖。
- run_id: `oicm-dexplore-rl-pilot-20260905-214032`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（pilot 产物、坐标与 candidate/contact 检查待生成；未据此作科研效果结论）

**文件与计划**

- [V1.2.2 执行计划](../plan/V1.2.2.md) — 冻结 split、坐标、surface sampling、pilot/full 验证和回滚边界。
- [build_dexplore_rl_cache.py](../../tools/data/build_dexplore_rl_cache.py) — 新增 MANO/RL-Inspire 统一 geometry cache、assignment、index 与 run manifest 生成器。
- [pilot output](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/) — PENDING，单序列 pilot 运行产物。

**原因与验证**

- 只使用 `inspire_rl` native q slice `[373:391]` 与 Dexplore right Inspire URDF；不读取 `inspire_geometric`。
- 目标 schema 复用 loader 已支持的 `geometry/manifest.json`，hand 固定 1538 点，object pool 固定 4096，30 Hz，future flow 由 loader 按帧产生。
- 已完成静态验证：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py` 通过。
- 下一步命令（pilot）：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode pilot --sequence s1/airplane_fly_1 --output data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot`。

## 2026-09-05 21:41:18 +0800 — pilot 首次运行失败并修复路径

- activity_id: `ACT-20260905-214118-OICM-DEXPLORE-RL-PILOT-FAILED`
- timestamp: 2026-09-05 21:41:18 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: 继承 V1.2.2 计划与本轮“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: pilot 运行后的 contact-check 路径修复；不改变数据语义、坐标、split 或旧 cache。
- run_id: `oicm-dexplore-rl-pilot-20260905-214032`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（转换阶段完成，末端 contact-check 将 sequence 根误传给 geometry 根；已定位并修正）

**验证与回滚**

- 失败命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode pilot --sequence s1/airplane_fly_1 --output data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot`。
- 证据：异常为 `FileNotFoundError`，目标多拼/少拼一层 `geometry`；未进入坐标或数值结论。
- 已用 `apply_patch` 将 pilot contact-check 输入修正为 `<sequence>/geometry`；失败生成的 37 MB pilot 目录将删除后重跑。

## 2026-09-05 21:42:49 +0800 — Dexplore RL-Inspire pilot 完成

- activity_id: `ACT-20260905-214249-OICM-DEXPLORE-RL-PILOT-COMPLETED`
- timestamp: 2026-09-05 21:42:49 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 计划与用户“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 单序列 `s1/airplane_fly_1` 的 Dexplore RL-Inspire 右手转换、loader probe 与 candidate/contact 几何 sanity check。
- run_id: `oicm-dexplore-rl-pilot-20260905-214249`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅表示 pilot 数据合同与转换 wiring 通过，不表示 Cm/解码科学效果成立）

**产物**

- [pilot cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/) — 单序列 geometry、assignment、index、manifest。
- [pilot run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/run_manifest.json) — 记录输入、q slice、URDF hash、seed、计数与 validation。
- [pilot contact check](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/pilot_contact_check.json) — native contact active 212/279，实际 5 cm 几何 active 222/279，分歧 10/279（3.58%），因此 full 仍沿用已导出的 native contact 语义并保留该偏差证据。
- [pilot validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/validation_summary.json) — object `[T,4096,3]`、hand `[T,1538,3]`、normals/pose/finite 检查通过。
- loader probe：`_SequenceView` 识别 `kind=inspire`、`effective_fps=30`；固定 stride=2 的 sample 输出 object `[1024,3]`、hand `[1538,3]`、future hand flow `[1538,3]` 且 finite。
- coordinate probe：object-frame 点到 Dexplore airplane mesh 顶点最近距离 median 0.415 mm、mean 0.417 mm；说明当前 object pose 转换使用 `native quaternion.T` 与 object-frame contract 一致。

## 2026-09-05 21:43:20 +0800 — Dexplore RL-Inspire full conversion 启动

- activity_id: `ACT-20260905-214320-OICM-DEXPLORE-RL-FULL-STARTED`
- timestamp: 2026-09-05 21:43:20 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: pilot 已通过且用户确认“按默认执行”
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 对 1255 条 GRAB/Dexplore RL 交集序列执行 assignment、MANO/RL geometry 转换、兼容 index 与 run manifest；test MANO-only，train/val variant 近似 1:1。
- run_id: `oicm-dexplore-rl-full-20260905-214320`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING full conversion；运行中不作科研效果结论）

**命令与 PENDING 产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 22:12:25 +0800 — Dexplore RL-Inspire full conversion 完成

- activity_id: `ACT-20260905-221225-OICM-DEXPLORE-RL-FULL-COMPLETED`
- timestamp: 2026-09-05 22:12:25 +0800
- modification_version: V1.2.2
- type: operation / data_change
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 final plan、pilot 通过、用户“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留既有 CmDecoder 用户改动与未跟踪实验状态）
- scope: 完成三方 intersection 的 1255 条右手 MANO/RL-Inspire geometry cache、assignment、兼容 index 和 loader 终态 probe；旧 cache/index/split 未覆盖。
- run_id: `oicm-dexplore-rl-full-20260905-221207`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（数据合同、转换 wiring、split/loader 审计通过；不代表 Cm 训练效果或未来 CmDecoderV2 科研结论）

**最终产物**

- [full cache root](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — 1255 条 sequence geometry，约 49 GB，不纳入 Git。
- [full index](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json) — 兼容 `ref2dex_object_interaction_cm_index_v1_1`，train/val/test=1004/126/125，source=501/64/125 MANO 与 503/62/0 RL-Inspire。
- [assignment](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/assignment.json) — seed=42、parent_seq_id/variant、三方 intersection、两条损坏 MANO 强制 RL 记录。
- [run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — `run_status=COMPLETED`、`conclusion=SUPPORTED`、输入 hash、URDF、计数、1255 条 validation。
- [validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/validation_summary.json) — 每条 sequence 的 shape/finite/pose 检查。

**终态验证**

- 三方 intersection：parent GRAB object cache、`dexplore_grab/sequences`、`inspire_rl` 均 1255 条，missing=0。
- 无泄露：全局 `parent_seq_id` 唯一；train/val 不跨 variant；test 全部 MANO；train/val variant 近似 1:1。
- schema：object pool `[T,4096,3]`、right hand `[T,1538,3]`、normals/pose/frame id/frame time 全部 finite；30 Hz；native q slice `[373:391]` 与 Dexplore native→URDF reorder 已写入 manifest。
- loader probe：三 split 均可由现有 `ObjectInteractionCmDataset` 读取；rows=304343/39538/36365；抽样 object `[1024,3]`、hand `[1538,3]`、future flow `[1538,3]` finite。
- 保护边界：未修改旧 `cm_object_v2_surface512_object_pose_20260830`、旧 `object_interaction_cm_v1_1/index.json`、共享 `src/base`、CmDecoder v2；未启动 Cm 训练。

**文件、原因与验证入口**

- [V1.2.2 执行计划](../plan/V1.2.2.md) — 固化本次数据 split、坐标、schema、pilot/full 顺序和回滚边界。
- [build_dexplore_rl_cache.py](../../tools/data/build_dexplore_rl_cache.py) — 实现三方 intersection、variant assignment、Dexplore q→URDF surface FK、MANO object-frame 重投影、cache/index/run manifest 与 resume。

**原因**

- 需要在同一 GRAB parent split 上混合 MANO 与 Dexplore RL-Inspire，同时禁止同一 `parent_seq_id` 跨 variant 泄露；右手-only 口径要求不伪造损坏的旧 MANO 文件。
- 保留 object_pose_t、4096 object pool、1538 hand surface、future hand flow 和 30 Hz，使后续 Cm 训练可以直接复用现有 loader 合同。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py` — PASS。
- 三方 intersection/assignment/index 计数审计 — PASS，missing=0、parent id 全局唯一、test MANO-only。
- `ObjectInteractionCmDataset` 三 split loader probe — PASS，rows `304343/39538/36365`，抽样 object/hand/future-flow finite。
- [validation summary](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/validation_summary.json) — 1255/1255 sequence shape、pose、finite 检查通过。

## 2026-09-05 22:15:44 +0800 — RL candidate mask 几何口径 refresh 启动

- activity_id: `ACT-20260905-221544-OICM-DEXPLORE-RL-CANDIDATE-REFRESH-STARTED`
- timestamp: 2026-09-05 22:15:44 +0800
- modification_version: V1.2.2
- type: data_change / operation
- change_level: L2（只更新新 cache 的 RL candidate mask，不改变点云、split、pose 或旧 cache）
- approval: user-approved
- approval_basis: pilot 已量化 native-contact 与几何 5 cm mask 的 3.58% 分歧；为公平性采用同一几何规则。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 对新 full/pilot cache 的 RL-Inspire entries，以 object-frame 生成 hand/object 近邻的 5 cm 规则重写 `obj_candidate_mask_5cm.npy`；MANO mask 不变。
- run_id: `oicm-dexplore-rl-candidate-refresh-20260905-221544`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING mask refresh 与 loader 复核）

**命令与产物**

- full：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --refresh-rl-candidates --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- pilot：PENDING，full 完成后对 pilot 同步 refresh。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING refresh 终态。

## 2026-09-05 22:42:42 +0800 — RL candidate mask 几何口径 refresh 完成

- activity_id: `ACT-20260905-224242-OICM-DEXPLORE-RL-CANDIDATE-REFRESH-COMPLETED`
- timestamp: 2026-09-05 22:42:42 +0800
- modification_version: V1.2.2
- type: data_change / operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户已确认默认执行；pilot 分歧证据支持统一 5 cm geometry active 口径
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: full 565 条 RL-Inspire sequence 与 pilot 的 candidate mask 已按生成 hand/object geometry 的 object-frame 5 cm KD-tree 规则刷新；MANO mask、点云、pose、split 未改变。
- run_id: `oicm-dexplore-rl-full-20260905-224017`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（mask/loader 数据合同通过；不代表 Cm/decoder 科研效果）

**文件、原因与验证**

- [V1.2.2 执行计划](../plan/V1.2.2.md) — 本次数据处理边界与回滚入口。
- [build_dexplore_rl_cache.py](../../tools/data/build_dexplore_rl_cache.py) — 新增 `--refresh-rl-candidates` 几何 mask refresh。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — `run_status=COMPLETED`、`candidate_refresh.rl_sequences=565`、`SUPPORTED`。
- [pilot contact check](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1_pilot/pilot_contact_check.json) — refresh 后 native/geometry active 均 222/279，分歧 0。

**原因**

- MANO 与 RL 必须使用同一 `active_only` 5 cm 几何规则；native Dexplore contact 与生成表面几何存在 3.58% pilot 分歧，不能混用两种过滤语义。
- 发现的 1 条旧 partial RL 目录 `s5/waterbottle_shake_1` 不属于最终 assignment，已移至 [recovery 目录](../../../../../data/processed_data/_recovery/object_interaction_cm_dexplore_rl_v1_stale_s5_waterbottle_shake_1_20260905/)，最终 geometry/index 均为 1255/1255，无额外 variant。

**验证**

- full index/geometry 审计：1255 entries、1255 geometry manifests、MANO 690、RL 565、extra ids=0。
- active-only loader probe：train/val/test rows=`169860/21911/20612`；抽样 object `[1024,3]`、hand `[1538,3]`、future flow `[1538,3]` finite。
- pilot/full manifest candidate semantics：`generated_rl_hand_to_object_surface_5cm`；刷新未改变 object/hand shape、pose、split 或旧 cache。

## 2026-09-05 22:10:13 +0800 — full manifest intersection 约束补记并刷新

- activity_id: `ACT-20260905-221013-OICM-DEXPLORE-RL-MANIFEST-REFRESH-STARTED`
- timestamp: 2026-09-05 22:10:13 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 默认方案要求同时核对 `dexplore_grab`、旧 object cache 与 `inspire_rl`；只补充已验证的输入约束，不改变已生成 geometry。
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: converter 现在显式检查三方 intersection；用 `--resume` 只复用 geometry 并刷新 index/run manifest。
- run_id: `oicm-dexplore-rl-manifest-refresh-20260905-221013`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING metadata refresh）

**命令与产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING（geometry 复用）。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 21:57:41 +0800 — full resume 遇到 parent 右手 MANO 损坏项

- activity_id: `ACT-20260905-215741-OICM-DEXPLORE-RL-FULL-RESUME-FAILED`
- timestamp: 2026-09-05 21:57:41 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: 继承 V1.2.2 与用户“按默认执行”确认
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: resume 在约 700 条序列处发现 `s7/headphones_lift` 的旧 right MANO mmap 损坏；只读扫描确认共有 2 条 train 序列损坏（另一个 `s2/mug_drink_2` 已分到 RL）。
- run_id: `oicm-dexplore-rl-full-resume-20260905-214854`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（转换器原 assignment 将损坏 right MANO 分给 MANO；不生成伪 MANO，改为强制 RL variant）

**证据与修正**

- right-only 文件扫描：1255 条中仅 `s2/mug_drink_2`、`s7/headphones_lift` 的 `right/hand_points_world.npy` 无法 mmap，且缺 `hand_normals_world.npy`/candidate；两条均在 train，test 无损坏项。
- 已通过 `apply_patch` 将 `mano_available` 写入 assignment：train/val 中不可用项强制 `inspire_rl`，test 若不可用则直接失败；不修改旧 cache。
- full partial 目录继续保留，下一次使用 `--resume` 会复用已完成 geometry 并只转换剩余项。

## 2026-09-05 21:58:08 +0800 — full conversion resume-2 启动

- activity_id: `ACT-20260905-215808-OICM-DEXPLORE-RL-FULL-RESUME2-STARTED`
- timestamp: 2026-09-05 21:58:08 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: parent right-hand 损坏项已证据化，按默认方案保留序列并强制其进入 RL variant
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 以新 assignment 重建并继续 full cache；已完成 geometry 复用，RL 分支不依赖损坏 MANO 文件。
- run_id: `oicm-dexplore-rl-full-resume2-20260905-215808`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING resume-2 终态）

**命令与产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 21:48:31 +0800 — full conversion 首次运行失败并切换安全 resume

- activity_id: `ACT-20260905-214831-OICM-DEXPLORE-RL-FULL-FAILED`
- timestamp: 2026-09-05 21:48:31 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.2.2 计划与 pilot 通过后的继续执行
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: full conversion 在已完成约 305 条序列后遇到旧 parent GRAB cache `s2/mug_drink_2/right/hand_points_world.npy` 的 mmap 文件损坏；修复脚本使 RL variant 不再读取无关 MANO hand，并加入不覆盖的 `--resume`。
- run_id: `oicm-dexplore-rl-full-20260905-214320`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`（失败来自转换器对 RL 分支读取无关旧 hand 文件；不是新坐标/FK 证据）

**证据与保护**

- 失败异常：`ValueError: mmap length is greater than file size`，发生在旧 cache hand mmap；当时已生成约 13 GB、305 条 sequence geometry，未覆盖旧数据。
- 已通过 `apply_patch`：RL variant 仅读取 object pool/pose；MANO variant 仍强校验右手 hand geometry；新增 `--resume` 只复用存在完整 `geometry/manifest.json` 的序列。
- 原失败 pilot 临时目录仍位于 [recovery 目录](../../../../../data/processed_data/_recovery/object_interaction_cm_dexplore_rl_v1_pilot_failed_20260905_214118/)，可人工清理；本次 full partial 目录保留用于 resume。

## 2026-09-05 21:48:54 +0800 — full conversion 安全 resume 启动

- activity_id: `ACT-20260905-214854-OICM-DEXPLORE-RL-FULL-RESUME-STARTED`
- timestamp: 2026-09-05 21:48:54 +0800
- modification_version: V1.2.2
- type: operation
- change_level: L2
- approval: user-approved
- approval_basis: 继承 V1.2.2 与 pilot 结果；resume 不改变 split/坐标/schema
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true
- scope: 在既有 full partial 目录上复用完整 sequence geometry，继续生成剩余序列并最终重建 assignment/index/run manifest。
- run_id: `oicm-dexplore-rl-full-resume-20260905-214854`
- run_status: `STARTED`
- conclusion: INCONCLUSIVE（PENDING resume 终态）

**命令与产物**

- 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py --mode full --resume --output data/processed_data/object_interaction_cm_dexplore_rl_v1`。
- [full cache](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/) — PENDING。
- [full run manifest](../../../../../data/processed_data/object_interaction_cm_dexplore_rl_v1/run_manifest.json) — PENDING。

## 2026-09-05 20:30:09 +0800 — RL-Inspire native q 与 OICM 几何接口诊断

- activity_id: `ACT-20260905-203009-OICM-RL-NATIVE-Q-GEOMETRY-CONTRACT`
- timestamp: 2026-09-05 20:30:09 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读接口与 native tensor schema 核查；未改变代码、配置、数据、cache、split、GT 或 checkpoint）
- approval: user-requested
- approval_basis: 用户确认主实验使用 Dexplore RL-Inspire variant，CmDecoderv2 延后单独建立，当前仅先训练 Cm
- skills_used: research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留已有 CmDecoder 修改与未跟踪文件）
- scope: 核对 `inspire_rl` 的 native tensor 布局与 ObjectInteractionCmDataset 当前支持的 hand geometry 布局，并将用户确认的 sequence-level embodiment 划分语义记录为后续 plan 输入。
- conclusion: INCONCLUSIVE（实验定义已基本明确，但 RL q→固定 Inspire 表面点/flow 的数据桥接和无 paired target 的测试指标尚未实现/验证）

**文件与证据**

- [Dexplore RL export README](../../../../../data/processed_data/inspire_rl_before_full_coordinate_fix_20260905/EXPORT_INFO.txt) — 记录 RL checkpoint rollout、右手、30 Hz、1335 条序列、native `(T,598)`。
- [Dexplore export schema](../../../../../data/processed_data/inspire_rl/) — 当前 RL 文件为 `interaction_hand_inspire.pt`；native tensor 的 Inspire 18-DOF slice 为 `373:391`，不能直接作为 OICM hand point cloud。
- [ObjectInteractionCm dataset](../../dataset.py) — 当前 loader 要求 hand geometry、normals 和 5 cm candidate metadata，Inspire 路径要求 `geometry/manifest.json`、`hand_points_world.npy` 等文件。

**原因**

- 用户方案应解释为：原始 GRAB sequence 只选择一个 embodiment；MANO sequence 只进入 OICM source 混合训练，RL-Inspire sequence 进入 OICM，并作为未来 Inspire-q decoder 的训练数据；同一 parent sequence 不同时出现两种 variant。
- 最终 held-out MANO → Cm → CmDecoderv2 是无 paired Inspire target 的跨 embodiment 泛化测试，不是逐帧 MANO→Inspire q 监督测试。
- 因此 decoder 训练阶段可以只使用 RL-Inspire subset；若把 MANO subset 也送入 q decoder，必须另定义 target，不能默认为 Inspire supervision。

**验证**

- 读取 RL/几何 export 说明及样例 tensor，确认 `inspire_rl` 与 `inspire_geometric` 是独立产物，且 RL 版本来自 deterministic policy rollout。
- 读取当前 OICM dataset loader，确认 q native tensor 不能直接满足 1538 点手几何合同；后续需用固定 Inspire URDF/mesh/FK 将 RL q 变成 surface points、normals 和 future flow。
- 本次未生成转换 cache、未修改索引、未启动训练或评估。

**边界、建议与回滚**

- 本条仅记录诊断与用户澄清；删除本条活动记录即可回滚本次文档差异。
- 后续 plan 需要单独冻结 embodiment assignment、sequence/object split、source-balanced sampling、RL q→geometry 的 manifest，以及无 paired target 时的 simulation/object-interaction 评价指标。

## 2026-09-05 20:25:24 +0800 — Dexplore RL 与几何重定向数据口径复核

- activity_id: `ACT-20260905-202524-OICM-DEXPLORE-RL-DATASET-CLARIFICATION`
- timestamp: 2026-09-05 20:25:24 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读数据目录与导出说明复核；未改变代码、配置、数据、cache、split、GT 或 checkpoint）
- approval: user-requested
- approval_basis: 用户明确指定 Dexplore 的 RL 数据集，而不是几何重定向数据集，并补充 Cm/decoder 实验边界
- skills_used: research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留已有 CmDecoder 修改与未跟踪文件）
- scope: 只读核对本地 `dexplore_grab`、`inspire_rl`、`inspire_geometric`、RL/几何导出说明和当前 OICM/CmDecoder 接口；不启动生成、训练或评估运行。
- conclusion: INCONCLUSIVE（已确认 RL 数据入口和用户实验语义，但 Cm 混合训练与后续 CmDecoderv2 尚未实现/验证）

**文件与证据**

- [Dexplore GRAB motion/object](../../../../../data/processed_data/dexplore_grab/) — 1335 条 GRAB 处理序列，包含 MANO motion 与 object。
- [Dexplore Inspire RL](../../../../../data/processed_data/inspire_rl/) — 1335 个 `interaction_hand_inspire.pt`，每条 shape 为 `(T, 598)`，由 `/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth` 的 Isaac Gym deterministic policy rollout 生成，右手、18-DOF、30 Hz。
- [Dexplore Inspire geometric](../../../../../data/processed_data/inspire_geometric/) — 同样 1335 条，但属于几何 retarget，不是本次目标数据。
- [RL export info](../../../../../data/processed_data/inspire_rl_before_full_coordinate_fix_20260905/EXPORT_INFO.txt) — 明确记录 `Export: GRAB -> Inspire (RL checkpoint rollout)`、`Robot: Inspire right hand`、`Rate: 30 Hz`。
- [几何 export info](../../../../../data/processed_data/inspire_geometric_before_full_coordinate_fix_20260905/EXPORT_INFO.txt) — 明确记录几何重定向及 30 Hz 下采样，作为排除项。

**原因**

- 用户的目标不是拿几何 retarget 结果充当 Inspire target，而是把 RL policy rollout 作为 Inspire variant 的 hand trajectory source，与原始 GRAB MANO variant 一起训练 Cm；未来再由独立的 `CmDecoderv2` 输出 Inspire q。
- 为避免泄露，不能让同一原始 GRAB sequence 的 MANO 和 Inspire RL variant 跨 train/test 出现；更严格的做法是先按原始 sequence/object 划分，再为每个 split 选择 variant，保留 `parent_seq_id`。
- 测试阶段只需要 held-out MANO 提取 Cm；不要求生成对应 Inspire 数据是合理的，但这样只能先验证 Cm 的跨 embodiment 表征/下游可解码性，不能在当前阶段得到 Inspire q 的定量测试结果。

**验证**

- 读取四个数据根目录的文件布局和 `EXPORT_INFO.txt`，确认 `inspire_rl` 与 `inspire_geometric` 是两套独立产物，且 RL 版本确实来自 checkpoint rollout。
- 读取样例 RL/几何 tensor，二者均为 `(T, 598)` native tensor；RL 与几何不能仅按文件名区分，数据 manifest 必须显式写入 `source_type=rl`。
- 当前仅做只读检查，未生成新数据、未改索引、未启动训练；后续代码实现仍需用户确认并定稿对应 plan。

**边界、建议与回滚**

- 本条只记录数据口径复核；删除本条活动记录即可回滚本次文档差异。
- 后续 Cm 训练建议采用右手、30 Hz、同一 source frame/horizon，并把 MANO 与 RL-Inspire 作为 source variant；`CmDecoderv2` 单独建 Task，target 只定义为 Inspire q，不把几何版本混入主实验。

## 2026-09-05 19:59:22 +0800 — Dexplore/GRAB MANO-Inspire 混合训练方案诊断

- activity_id: `ACT-20260905-195922-OICM-DEXPLORE-PIPELINE-DIAGNOSTIC`
- timestamp: 2026-09-05 19:59:22 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读仓库、数据布局、接口与实验语义核查；未改变代码、配置、数据、cache、split、GT 或 checkpoint）
- approval: user-requested
- approval_basis: 用户要求检查使用 Dexplore 将 GRAB MANO 与 Inspire 重定向数据混合训练 Cm/解码器的可行性与疑问
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留已有 CmDecoder 修改与未跟踪文件）
- scope: 读取 ObjectInteractionCm/CmDecoder 当前数据合同、模型输入输出、索引与 split 约束；检查本地 `dexplore_grab` 处理数据的序列数量、字段和与当前 GRAB cache 的交集；核对 Dexplore 官方转换脚本和模型卡。未修改代码、配置、原始数据、cache、checkpoint 或运行目录。
- conclusion: INCONCLUSIVE（工程上可构造该实验，但源/目标手、未来 hand-flow 条件、split、时间尺度、伪 GT 与采样权重仍需先固定，当前没有新的训练/评估证据支持方案效果）

**文件与证据**

- [Dexplore processed root](../../../../../data/processed_data/dexplore_grab/) — 本地 1335 个序列目录，每个当前只有 `motion.npz` 与 `object.npz`；未发现 Inspire q 或 1538 点几何输出。
- [ObjectInteractionCm dataset](../../dataset.py)、[ObjectInteractionCm model](../../model.py) — 当前 OICM 将双 MANO 手或单 Inspire 手填充到最大 3076 点；forward 把 future `hand_flow` 作为输入条件，模型本身不消费 `stride/delta_time_s`。
- [CmDecoder adapter](../../../CmDecoder/object_interaction_cm_model.py) — 当前适配器把同一个 1538 点手及其 flow 同时送入 OICM 和解码目标，尚无 MANO source → Inspire target 的分离字段。
- [当前 OICM index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 现有混合索引仍是 GRAB 与 HRDexDB Inspire；Dexplore 数据尚未接入。
- [Dexplore conversion script](https://raw.githubusercontent.com/NVlabs/dexplore/main/data_processing/convert_grab.py) — 官方转换会对 GRAB 做 `::4` 采样、逐帧 retarget 到机器人 DOF，并写出 `interaction_hand_inspire.pt`；这不是本地 `dexplore_grab` 目录现有的完整输出。
- [Dexplore model card](https://github.com/NVlabs/dexplore/blob/main/MODEL_CARD.md) — 发布模型使用约 1269 条 GRAB 中的 658 条，未提供专门的 held-out GRAB 测试 split。

**原因**

- 本地 Dexplore 处理序列归一化名称后与当前 GRAB object cache 有 1255 条交集、80 条缺失；处理帧数约为 raw 120 Hz 的四分之一，说明当前对象轨迹大致为 30 Hz，但仍需由 manifest 确认 frame mapping。
- 同一原始序列生成 MANO 与 Inspire 两个 variant 时，split 必须在 variant 生成前按原始 sequence（必要时 object）划分；否则测试 MANO 与训练 Inspire 的同序列配对会泄漏。
- 如果目标是 MANO Cm → Inspire 解码器，训练样本必须显式区分 `source_hand_*` 与 `target_hand_*`；不能把 MANO 和 Inspire 都作为无标签 target 混在同一个 decoder 监督池里。当前 decoder guidance/plan 仍是 Inspire-only。
- Dexplore 的 Inspire q/几何是 retarget 产生的伪 GT，不是实测机器人轨迹；结果应命名为 retarget-label reconstruction/retarget consistency，并额外报告关节限位、平滑性和接触关系。
- 若用未来 `hand_flow` 提取 Cm，评估属于 teacher-forced/offline 条件重建；不能直接宣称在线或未来预测。在线声明需要只用观测到的 flow/闭环 rollout 另测。
- MANO 当前可双手，而 Dexplore Inspire 输出通常是单个机器人手；需固定右手/左手口径或明确双手拼接规则，否则 padding、接触和 embodiment 差异会成为 shortcut。

**验证**

- 只读统计确认本地 Dexplore 目录包含 1335 个序列，归一化名称后与当前 GRAB object cache 交集为 1255、缺失 80；样例 raw GRAB 为 120 Hz，Dexplore 处理帧数约为 raw 的四分之一。
- 读取当前 ObjectInteractionCm 的 dataset/model、CmDecoder adapter、索引及 Task 指导/plan，确认现有 decoder 仍是 Inspire-only，同一 1538 点手/flow 同时承担 OICM 输入和 decoder target，尚未支持 MANO source → Inspire target。
- 使用 `audit_diff.py --check-links` 做活动记录审计；首次检查提示本条缺少标准“原因/验证”段，已按仓库合同补齐后重新执行。

**边界、建议与回滚**

- 本次只产生诊断记录；代码、配置、数据、cache、checkpoint、split、GT 和既有运行均未改动，删除本条活动记录即可回滚本次文档差异。
- 后续若实现该方案，需要先定稿与指导同版本的 plan，至少冻结：source/target 字段、手侧、sequence/object-disjoint split、共同物理 horizon、Dexplore converter/URDF/mesh/点采样 manifest、variant 采样权重，以及 teacher-forced 与 online 两类评估边界。

## 2026-09-05 17:18:12 +0800 — GRAB 原始 120 Hz 与 HRDexDB 物理时间跨度对齐诊断

- activity_id: `ACT-20260905-171812-OICM-GRAB-120HZ-HR-TIMESCALE`
- timestamp: 2026-09-05 17:18:12 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读时间分辨率与运动幅度核查；未改变代码、配置、数据、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求比较 GRAB 原始 120 Hz 与 HRDexDB 的运动幅度，并判断时间采样是否造成表观差异
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留 CmDecoder 用户改动与未跟踪文件）
- scope: 读取 GRAB 原始 `.npz` 的物体轴角/平移、当前 30 Hz GRAB object-pose cache、旧 cache 中的 5 cm candidate 点索引，以及当前 Inspire-F1 HRDexDB geometry；在 val 序列上按物理时间跨度重算 object-pose frame 的物体表面位移。未修改代码、配置、原始数据、cache、checkpoint 或运行目录。
- conclusion: SUPPORTED（120 Hz 的单 raw frame 位移因时间间隔缩短而变小，但在相同物理时间跨度下 GRAB 仍比 Inspire-F1 HRDexDB 大约 8–11 倍；若只把 GRAB 改成 120 Hz 而保持整数 stride `1..10`，会把 GRAB horizon 错移到约 `8–83 ms`，进一步破坏与 HR `67–667 ms` 的时间语义匹配）

**文件与证据**

- [GRAB 原始样例](../../../../../data/raw_data/GRAB/grab/s1/airplane_fly_1.npz) — 原始字段声明 `framerate=120 Hz`，物体轨迹保留每个 raw frame 的 `global_orient` 与 `transl`。
- [GRAB cache meta](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s1/airplane_lift/shared/meta.json) — 当前 cache 为 `source_fps=120`、`ds_rate=4`，有效频率为 30 Hz，物体池 4096 点。
- [ObjectInteractionCm index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 当前训练的 GRAB stride 为 `1..10`、Inspire-F1 stride 为偶数 `2..20`，两者都由各自 cache 的 frame index 解释。
- [HRDexDB geometry manifest](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/episodes/78013e164ab75a34/geometry/manifest.json) — HR geometry 的 `delta_time_median_s≈30 ms`、最大约 `60.7 ms`，并提供 4096 点池与 5 cm object candidate mask。

**原因**

- 需要把“120 Hz 逐帧看起来位移更小”和“同一物理时间内实际运动更小”分开；如果不对齐 `Δt`，直接比较整数 stride 会把采样率差异误判成数据分布差异。

**验证**

- 统计口径：GRAB 使用当前 index 的 126 条 val 序列、原始 120 Hz 物体 pose；以旧 GRAB cache 的 candidate 点索引筛选当前有接触的 source frame，每行最多抽 256 个候选物体表面点。HR 使用当前 Inspire-F1 val 的 58 条序列、1024 个固定池点，并筛选 5 cm candidate 点。位移是在当前物体 pose 的坐标系中计算；这是分布诊断，不是模型 benchmark。
- GRAB 原始 120 Hz（candidate object points，均值 / 中位数 / p95，mm）：stride 1（8.3 ms）=`2.60/1.26/9.85`；stride 4（33.3 ms）=`9.99/4.89/38.97`；stride 8（66.7 ms）=`19.67/9.61/76.44`；stride 40（333.3 ms）=`87.44/46.59/340.20`。
- Inspire-F1 HRDexDB 30 Hz（同类 candidate object points，mm）：stride 1（约 33.3 ms）=`1.36/0.64/4.55`；stride 2（约 66.7 ms）=`2.37/1.11/8.13`；stride 10（约 333 ms）=`9.58/4.37/33.92`。
- 物理时间对齐后：约 66.7 ms 的 GRAB stride 8 / HR stride 2，其均值、中位数、p95 比约为 `8.3×/8.6×/9.4×`；约 333 ms 的 GRAB stride 40 / HR stride 10，比约为 `9.1×/10.6×/10.0×`。这与既有 30 Hz active-flow 诊断（GRAB 中位数约 `9.1 mm`、HR 约 `1.0 mm`）一致，说明简单下采样没有消除域差。
- 仅比较单整数 stride 会产生误导：GRAB raw stride 1 的间隔只有 8.3 ms；把它与 HR stride 2 的 66.7 ms 直接比较，会把较短 horizon 误认为较小运动。当前模型 forward 读取 object/hand 几何与 flow，但不读取 `stride` 或 `delta_time_s`，因此不同物理 horizon 会共享同一个预测函数。

**边界、建议与回滚**

- 本次只量化了物体表面位移；GRAB 120 Hz 的全手 MANO surface 未在全量上重建。既有同口径 66.7 ms 诊断仍显示 hand flow 中位数约 GRAB `13.5 mm`、Inspire-F1 `1.9 mm`（HR human MANO 约 `5.4 mm`），方向与 object 结论一致。
- 若保留原始 120 Hz，建议先按物理 horizon 重定义 stride：GRAB raw `4..40` 才覆盖当前 30 Hz `33..333 ms`，与 HR stride 2/10 对齐分别约用 raw stride 8/40；或者先统一重采样到 30 Hz。任何 stride/采样合同变更都应单独建 plan 并做受控 ablation，不能把本诊断当作效果因果证据。
- 仅新增本活动条目；删除本条即可回滚文档差异。代码、配置、数据、cache、checkpoint 和既有运行均未改动。

## 2026-09-05 17:49:32 +0800 — 按位移幅度匹配 GRAB 原始 120 Hz 与 HRDexDB stride 诊断

- activity_id: `ACT-20260905-174932-OICM-AMPLITUDE-STRIDE-MAP`
- timestamp: 2026-09-05 17:49:32 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读幅度分布与 stride 映射核查；未改变代码、配置、数据、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求在考虑运动幅度对齐时估计 GRAB 原始 120 Hz 与 HRDexDB stride 的对应关系
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `27316ef8e9552b7b335e53400453902d745b1ebc`
- worktree_dirty: true（保留 CmDecoder 用户改动、未跟踪文件及本任务既有诊断记录）
- run_status: NOT_STARTED（只读统计；未启动正式训练/评估 run）
- conclusion: SUPPORTED（在当前 val 物体候选点口径下，中心位移幅度的经验映射约为 `GRAB raw stride ≈ ceil(HR stride / 3)`；这不是物理时间或速度语义的等价映射，映射用于训练尺度 ablation 的可行起点，收益因果仍为 INCONCLUSIVE）
- scope: 使用当前 ObjectInteractionCm index 的 GRAB val 126 条序列和 Inspire-F1 HRDexDB val 58 条序列；GRAB 读取原始 120 Hz object pose，HR 使用约 30 Hz geometry，按 5 cm candidate object points 统计多个 stride 的位移均值/中位数/p95，并为每个 HR stride 选择分布距离最近的 GRAB raw stride。未修改代码、配置、原始数据、cache、checkpoint 或运行目录。

**文件与证据**

- [ObjectInteractionCm index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 当前 HR stride 为偶数 `2..20`，GRAB cache stride 为 `1..10`。
- [GRAB 原始样例](../../../../../data/raw_data/GRAB/grab/s1/airplane_fly_1.npz) — 原始 `framerate=120 Hz`。
- [GRAB cache meta](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s1/airplane_lift/shared/meta.json) — 现有 GRAB cache 仍为 `source_fps=120`、`ds_rate=4`，即有效 30 Hz。
- [HRDexDB geometry manifest](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/episodes/78013e164ab75a34/geometry/manifest.json) — HR 几何帧间隔中位数约 30 ms。

**原因**

- 物理时间对齐回答“相同时间内移动了多少”；本条额外回答“若只让监督位移的统计幅度相近，应取哪个 raw stride”。两者目标不同，不能把幅度映射解释成动力学等价。

**验证**

- 统计口径：GRAB 每个 active source frame 最多抽取 128 个 candidate object surface points，HR 每条序列从 4096 点池中使用 5 cm candidate mask 后统计；数值为 val 分布诊断，不是 benchmark。
- GRAB raw120 的 candidate object displacement（均值/中位数/p95，mm）随 stride 近似为：`g1=2.59/1.26/9.84`、`g4=9.96/4.87/38.93`、`g8=19.62/9.56/76.37`、`g12=29.03/14.22/113.16`。
- HRDexDB Inspire-F1 约 30 Hz 的对应统计为：`h2=2.37/1.11/8.13`、`h10=9.58/4.37/33.92`、`h20=17.65/8.36/61.58`。
- 按均值/中位数/p95 的最近邻结果综合后，当前使用的 HR 偶数 stride 建议映射为：`h2→g1`、`h4→g2`、`h6→g2~3`、`h8→g3`、`h10→g4`、`h12→g4~5`、`h14→g5`、`h16→g5~6`、`h18→g6`、`h20→g7`；简化成单值即 `{1,2,2,3,4,4,5,6,6,7}`，可记作 `g≈ceil(h/3)`。
- 该映射的时间跨度明显更短：例如 HR `h2` 约 60–67 ms，而幅度匹配的 GRAB `g1` 只有 8.3 ms；HR `h10` 约 300–333 ms，而幅度匹配的 GRAB `g4` 只有 33.3 ms。相同物理时间则仍是 `h2↔g8`、`h10↔g40`，且 GRAB 位移约大 8–11 倍。
- 作为 hand-flow 边界检查，HR 当前 1538 点 MANO 在 `h2/h10/h20` 的中位数约为 `1.79/8.70/15.98 mm`；本次没有重建 GRAB raw120 全量 MANO surface，因此上述 stride 映射目前只对 object target 直接成立，不能宣称 hand 与 object 共用同一精确映射。
- 现有 GRAB 30 Hz cache（raw `g4/g8/g40`）的全手点位移中位数约为 `5.68/11.28/52.22 mm`，与 HR `h2/h10/h20` 的 `1.79/8.70/15.98 mm` 对照，线性插值给出的 hand 幅度映射更接近 `g≈0.5~0.65h`；这进一步说明 object 的 `ceil(h/3)` 不能未经验证地同时用于 hand。

**边界、建议与回滚**

- 若目标是消除 loss 中 target magnitude 的主导差异，可把 `g≈ceil(h/3)` 作为受控 ablation 的初始配对，并同时记录真实 `Δt`/速度；现有 GRAB 30 Hz cache 的 stride `1` 已相当于 raw `g4`，无法表示 `g1~3` 的短幅度 horizon。
- 若 object 与 hand loss 同时等权，建议先把这两种映射作为两个独立 ablation（或按 target RMS 做 loss reweight），不要假设存在一个同时精确对齐的 stride；hand 的 raw120 精确映射需另做 MANO 重建统计。
- 若目标是相同动作时间或可解释的未来预测，应继续使用物理时间映射（raw `g≈4h`，例如 `h2↔g8`、`h10↔g40`），并考虑显式输入 `delta_time_s` 或预测速度。当前 model forward 未读取 stride/delta-time，故任一映射都应作为明确的训练变量记录。
- 结论状态：幅度映射观察为 `SUPPORTED`；其能否改善最终效果尚未运行实验，科研结论为 `INCONCLUSIVE`。仅新增本活动条目；删除本条即可回滚文档差异，代码、配置、数据、cache、checkpoint 和既有运行均未改动。


## 2026-09-04 15:27:04 +0800 — HRDexDB MANO 配对与可适配数据集诊断

- activity_id: `ACT-20260904-152704-OICM-HR-MANO-DATASET-SEARCH`
- timestamp: 2026-09-04 15:27:04 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读数据与公开资料核查；未改变代码、配置、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求搜索可适配数据集，核对 HRDexDB 是否含 MANO 配对数据及其运动幅度
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: HRDexDB `episodes/pairings/objects` 元数据、human MANO OBJ/JSON、compact object pose、当前 Inspire-F1 选集连接、同一 object-pose/wrist-frame 的运动抽样，以及公开数据集的标注与许可证入口；未修改代码、配置、数据、cache、checkpoint 或运行目录
- conclusion: SUPPORTED（HRDexDB 完整发布目录含 human MANO 与经验证 robot→human pairing；当前 ObjectInteractionCm index 只喂 Inspire-F1，未喂 human MANO；human MANO 增大手部运动但没有改变 HR 近静止的物体运动分布。候选数据集的最终收益仍需受控实验，故因果结论为 INCONCLUSIVE）

**文件与证据**

- [HRDexDB README](../../../../../dataset/HRDexDB/v0_nonvideo/README.md) — 官方元数据说明：`train` 是完整 catalog 而非 benchmark split，`pairings` 只收录显式验证的 robot-to-human 链接。
- [episodes.parquet](../../../../../dataset/HRDexDB/v0_nonvideo/metadata/episodes.parquet)、[pairings.parquet](../../../../../dataset/HRDexDB/v0_nonvideo/metadata/pairings.parquet) — 本地目录统计：2104 条 episode、441 条 human episode、1335 条已验证 pairing；Inspire-F1 pairing 392 条。
- [当前 ObjectInteractionCm index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json)、[Inspire-F1 选集](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/selection_all_object_disjoint_seed42.json) — 当前训练实际使用 576 条 Inspire-F1 episode；与 pairing 连接后有 378 条 robot episode、377 条 unique human episode，但 human 记录尚未进入当前 index。
- [HRDexDB 全量 geometry manifest](../../../../../data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json) — 仓库已有四手型 `cmdecoder_layered_v4` cache：2088 条有效 episode（含 human 441 条），但该旧 manifest 仍是独立 CmDecoder cache，不能替代当前 ObjectInteractionCm 的 `object_pose_t` index。
- [HR cache builder](../../../../../src/task/CmDecoder/build_cache.py) — human 分支读取 `hand/mano/*.obj` 与 `mano_params/*.json`，以 1538 个 MANO face-center 保持 hand 点合同，并把 `q_semantics` 标记为 `unavailable_mano`；这说明 human MANO 可转成现有 surface/cache 表示，但不是可直接解释的机器人关节 q。
- [model.py](../../model.py) — 工作区已有用户改动，本次诊断未修改。

**原因**

- 需要区分“HRDexDB 是否拥有可用的 paired MANO 证据”和“当前训练是否真的看到了 human source”；同时用与现有 object-pose/wrist-frame 一致的时间跨度比较 hand/object flow，避免把配对关系误当作逐帧同步或把近静止物体目标误判成高动态监督。

**验证**

- 使用 `fastwam` 环境的 `pyarrow` 读取 parquet：所有 HR 元数据 `split=train`；pairings 的 `source` 均为 `grasp_result.json:human_paired_episode` 且 `verified=True`。当前选中的 377 个 unique human episode 中，MANO OBJ 数、MANO 参数数、compact object-pose 帧数与 episode `num_frames` 均一致。
- 对全部 440 个可连接 human episode 及当前选集对应的 377 个 episode 做只读抽样：解析 MANO face-center、object 4×4 pose、wrist（`joints[0]`/`global_orient[0]`），在当前 pipeline 的 object-pose/wrist frame 中按 30 Hz 计算 stride=2（约 66.7 ms）和 stride=10（约 333 ms）flow。数值是随机表面点诊断抽样，不是官方 benchmark。
- 当前选集对应 human 的 stride=2：hand flow 中位数 `5.44 mm`、p95 `15.33 mm`；object flow 中位数 `1.07 mm`、p95 `10.44 mm`、`>20 mm` 约 `0.06%`。stride=10：hand 中位数 `26.44 mm`，object 中位数 `3.34 mm`、p90 `40.95 mm`。
- 同口径既有诊断中，Inspire-F1 stride=2 的 hand/object 中位数约 `1.85/1.01 mm`，GRAB 约 `13.46/9.11 mm`；因此 human MANO 主要补手部 articulation/contact，object motion 仍接近 HR robot，远小于 GRAB。
- 公开资料核查了 GigaHands（完整物体运动与 MANO-derived hand）、HOT3D（刚性物体 6DoF + MANO）、OakInk/OakInk2（MANO + object SE(3)）、DexYCB（MANO + object 6D）、ARCTIC（高动态双手但含 articulated object）、HOGraspNet（接触/抓取标注）等候选；没有据此启动训练或改变研究变量。

**边界与回滚**

- 配对是“同对象/同任务语义”的 human↔robot 关联，不应默认逐帧同步；官方采集协议允许机器人操作者观察人类动作后按自身形态重现。因此 human MANO 更适合作为独立 source 或 contact/hand 预训练信号，不能直接当作 robot flow 的逐帧 GT。
- 仅新增本活动条目；删除本条即可回滚文档差异。代码、配置、数据、cache、checkpoint 和既有运行均未改动。

## 2026-09-04 12:25:51 +0800 — GRAB 与 Inspire-F1（HRDexDB 子集）分布诊断

- activity_id: `ACT-20260904-122551-OICM-DATA-DISTRIBUTION`
- timestamp: 2026-09-04 12:25:51 +0800
- modification_version: V1.2.1.4
- type: diagnostic
- change_level: L0（只读分布核查；未改变代码、配置、cache、split、GT 或既有运行）
- approval: user-requested
- approval_basis: 用户要求探查 GRAB 与 HRDexDB 的分布差异及其对效果的可能影响
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: 读取当前 ObjectInteractionCm index、两套 cache/geometry manifest、scale manifest、HRDexDB episodes metadata 和既有验证结果；固定 stride=2 的接触/运动抽样，以及训练 stride policy 的对照。工作区已有的 [model.py](../../model.py) 未在本次诊断中修改。
- conclusion: SUPPORTED（存在足以解释混合训练效果受限的显著域差，主差异在运动/接触/embodiment 与 split；“域差是唯一原因”仍为 INCONCLUSIVE）

**文件与证据**

- [index.json](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 当前实际混合合同：GRAB 1004/126/125 条序列，Inspire-F1 455/58/63 条序列；source probability 0.5/0.5，GRAB stride 1..10、Inspire-F1 stride 2..20，split 规则不同。
- [GRAB cache meta.json](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/meta.json)、[HRDexDB selection](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/selection_all_object_disjoint_seed42.json)、[HRDexDB metadata README](../../../../../dataset/HRDexDB/v0_nonvideo/README.md) — 当前 HRDexDB 不是全库 2104 条，而是 Inspire-F1 的 576 条 object-disjoint 子集。
- [scales_train.json](../../../../../data/processed_data/object_interaction_cm_v1_2_1/scales_train.json) — 全局 flow scale 与 source/stride group RMS。
- [当前 metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl) — epoch 52 source 指标及 interaction coverage；[旧 stride 诊断](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/results.json) — 同 cache 合同下的 zero-flow 对照。

**原因**

- 需要把“物体侧表示更细”与“两个 source 是否提供同一种可学习信号”分开；只看混合 EPE 会把近静止 HR 目标、双手/单手输入和不同 split 难度混在一起。

**验证**

- 只读统计得到：固定 stride=2 的 object flow median 约 GRAB `9.11 mm`、Inspire-F1 `1.01 mm`；`>20 mm` 比例约 `33.5%` 对 `0.56%`。hand flow median 约 `13.46 mm` 对 `1.85 mm`。
- 同一 active transition 抽样中，object 点 `<5 cm` 接触覆盖约 `73.1%` 对 `40.5%`，KNN 有效邻居约 `5.8` 对 `3.1`；当前全量 val 也显示 sampled active points `747.9` 对 `429.0`。
- object bbox diagonal median 约 `172` 对 `217 mm`，geometry scale 差异中等；但 stride=2 group RMS flow 为 object `32.30` 对 `5.51 mm`、hand `35.87` 对 `4.63 mm`（约 5.9–7.8 倍）。
- 当前全量 val source object EPE 为 GRAB `5.073 mm`、Inspire-F1 `2.225 mm`；旧同 cache stride=2 zero-flow 对照中 Inspire-F1 `2.338 mm`、模型 `2.356 mm`，说明 HR 的绝对指标本身已接近“保持不动”下限。
- 本次未启动新训练、未运行测试集、未修改研究变量；统计结果用于诊断，不等同于因果实验。

**回滚与边界**

- 仅新增本活动条目；删除本条即可回滚文档差异。代码、配置、数据、cache、checkpoint 和既有运行均未改动。

## 2026-09-04 11:27:03 +0800 — V1.2.1 前缀手点 KNN 邻域搜索优化

- activity_id: `ACT-20260904-112703-OICM-KNN-PREFIX-OPT`
- timestamp: 2026-09-04 11:27:03 +0800
- modification_version: V1.2.1.4
- type: code / diagnostic
- change_level: L1（局部实现优化；不改参数、坐标系、GT、radius=5cm、K=8 或 checkpoint schema）
- approval: user-requested
- approval_basis: 用户明确要求先修改邻域搜索并在 GPU0/1/2 做吞吐短训
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: `LocalHandInteraction` 在前缀有效 mask（CmDecoder 的 1538/3076 contract）上改用 PyTorch3D `knn_points(K=8)`，再执行 5cm radius mask；非连续 mask 和全无效情况回退原 `cdist+topk`；未实现持久化邻域 cache
- conclusion: SUPPORTED（数值 parity 与独立 kernel benchmark 通过；端到端短训吞吐已测，科研效果尚未评估）

**原因**

- 之前的 `cdist` 在 mask 之前对 3,076 个手点全部计算；CmDecoder 实际只有前 1,538 个有效点。先 compact 前缀再做 KNN 可减少显存和距离计算，同时保留 5cm/K=8 语义。

**验证**

- 随机 `[B=2,N_obj=32,N_hand=3076]` parity：新旧 `edge_indices` 和 `edge_valid_mask` 完全一致，交互输出最大绝对差 `1.49e-8`，距离最大差 `5.03e-8`。
- GPU1 邻域 kernel（`B=16,N_obj=1024,N_hand=3076,K=8`）基准：旧 `cdist+topk=3.94 ms`，新 `knn_points=0.576 ms`，约 `6.84x` 加速；该结果只代表邻域子图，不等于完整训练加速。
- 验证命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/model.py`；另以同一随机权重分别执行旧 `cdist+topk` 参考路径和新路径，比较上述输出/诊断张量。
- 修改文件：[model.py](../../model.py)。回滚入口为恢复该文件本次 diff；没有改动 checkpoint、数据或公共 `src/base`。

**运行关联**

- 端到端短训见 CmDecoder 活动 `ACT-20260904-112703-CMDECODER-KNN-012-THROUGHPUT`；该运行使用本 Task 的 frozen best Cm，不改变 Cm 权重。

## 2026-09-04 08:45:46 +0800 — V1.2.1 三卡训练完成与收敛性核查

- activity_id: `ACT-20260904-084546-OBJECTINTERACTIONCM-V121-CONVERGENCE-FINAL`
- timestamp: 2026-09-04 08:45:46 +0800
- modification_version: V1.2.1.3
- type: diagnostic / operation
- change_level: L0（只读核对终态日志、metrics、checkpoint 与验证曲线）
- approval: user-requested
- approval_basis: 用户询问 Cm 训练是否收敛；本次不启动、停止或修改训练与评估口径
- skills_used: research-experiment-workflow, research-change-control
- branch: `oyx`
- base_commit: `6ca6509ab1f8853767d11724877dd8aba5914b63`
- worktree_dirty: true（保留既有用户改动）
- scope: ObjectInteractionCm V1.2.1 三卡续训终态；不修改代码、配置、数据、cache 或 checkpoint
- run_id: `object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806`（resume）
- run_status: COMPLETED
- last_step: `202300`
- last_epoch: `52`
- best_metric: `3.649306 mm`（`val/obj/flow_epe_mm`，epoch 52）
- best_checkpoint: [best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt)（step `202300` / epoch `52`）
- latest_checkpoint: [latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt)（step `202300` / epoch `52`）
- practical_status: 基本收敛（object 指标已进入平台，最后 5 个 epoch 仅改善约 `0.066%`）
- conclusion: INCONCLUSIVE（未做 held-out test，且单次训练不能证明统计意义上的最终收敛）

**原因**

- 用户要求确认 ObjectInteractionCm 是否收敛；本次只读检查完整训练终态和最后若干 epoch 的验证曲线，区分“训练完成”“实用平台”和“科研意义上的严格收敛”。

**验证**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 日志末尾为 `Training finished at step 202300 in 08:11:05.`，epoch 52 validation 已写入。
- 总体 `val/obj/flow_epe_mm`：epoch 44=`3.669525 mm`、epoch 48=`3.651703 mm`、epoch 52=`3.649306 mm`；最后 5 个 epoch 均值=`3.650447 mm`、标准差=`0.000969 mm`，首尾改善约 `0.066%`。
- 总体 `val/loss` 最后 5 个 epoch 均值=`0.00307428`、标准差=`1.41e-6`；epoch 52 学习率=`3.30e-9`，优化步长已接近零。
- source 分支：GRAB object EPE 最终=`5.073422 mm`（全程 best）；Inspire-F1 object EPE 最优为 epoch 15 的 `2.216799 mm`，末期稳定在约 `2.22–2.25 mm`，说明后者早已进入平台。
- hand 3 cm EPE 最终=`2.404122 mm`，epoch 51 为 `2.404022 mm`，末期无实质变化；`train.log` 未发现 traceback、CUDA OOM、NCCL、NaN 或 Inf，训练进程已退出。
- 本次只读检查未停止或修改其他 GPU 任务；`git diff --check` 与 `audit_diff.py --check-links` 通过。

## 2026-09-04 00:06:29 +0800 — V1.2.1 各 source/stride 验证集快速评估

- activity_id: ACT-20260904-000629-OBJECTINTERACTIONCM-V121-STRIDE-EVAL
- timestamp: 2026-09-04 00:06:29 +0800
- modification_version: V1.2.1.3
- type: operation / diagnostic
- change_level: L0（用户请求的只读评估；仅生成独立运行产物）
- approval: user-requested
- approval_basis: 用户要求先用其他 GPU 评估各 stride；不停止或修改 GPU0/1/2 上的训练，不抢占 GPU4–7 上的既有任务
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留既有 ObjectInteractionCm 实现、训练和治理改动）
- scope: 使用训练运行的 `latest.pt`（step=129302、epoch=32）在验证集分别评估 GRAB stride `1..10` 与 Inspire-F1 stride `2..20`；每个 source/stride 均独立 forward，每组均匀抽取 512 个样本；不改代码、配置、cache、训练进程或其他 GPU 任务
- run_id: objectinteractioncm_stride_eval_subset_20260904_000408
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（这是每组 512 样本的快速诊断，不替代全验证集和最终 parity 结论）

**文件与证据**

- [运行 manifest](../../../../../outputs/research/objectinteractioncm_stride_eval_subset_20260904_000408/run_manifest.json)、[stride 结果](../../../../../outputs/research/objectinteractioncm_stride_eval_subset_20260904_000408/results.json) — 设备、checkpoint SHA256、采样合同、source/stride 的 object/hand EPE 和有效样本率。
- [当前训练运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt) — 评估输入及仍在运行的三卡训练。

**原因**

剩余 GPU 中 4–7 卡有高负载既有任务，3 卡有足球任务但显存余量足够，因此只在物理 GPU3 以小 batch 共存运行。先用每组 512 个均匀覆盖样本快速判断 stride 趋势；模型 interaction 分支使用 `hand_flow`，故每个 stride 必须独立 forward，不能复用其他 stride 的预测。

**验证**

- GPU3 评估期间总显存约 `11.4/24 GiB`，未发生 OOM；GPU0/1/2 训练进程和吞吐未被停止或改动。
- GRAB object EPE 从 stride1 的 `2.788 mm` 增至 stride10 的 `24.124 mm`；Inspire-F1 从 stride2 的 `2.314 mm` 增至 stride20 的 `10.935 mm`。对应 hand 3 cm EPE 分别为 `1.710→14.763 mm` 与 `0.730→3.205 mm`。
- stride2 的 source-mean object EPE 为 `3.799 mm`，与同 checkpoint 既有完整验证日志的 `3.776 mm` 接近，说明评估口径和 target 生成一致；快速子集结果仍不能视为最终全量指标。
- `results.json`、`run_manifest.json` 已写入 `run_status=COMPLETED`；未运行测试集、未修改研究变量。

**回滚**

删除本条活动记录即可回滚文档差异；评估输出为独立、可保留的只读产物，训练和其他 GPU 任务未受影响。

## 2026-09-03 23:54:32 +0800 — V1.2.1 首次 stride 复用评估作废

- activity_id: ACT-20260903-235432-OBJECTINTERACTIONCM-V121-STRIDE-EVAL-INVALID
- timestamp: 2026-09-03 23:54:32 +0800
- modification_version: V1.2.1.3
- type: operation / diagnostic
- change_level: L0（失败的只读评估尝试）
- approval: user-requested
- approval_basis: 用户要求先评估各 stride；该尝试未改变训练、代码或配置
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true
- scope: 评估 `latest.pt` 的 all-stride 运行；在 GRAB 完成后发现错误地假设不同 stride 可复用 forward，随后在 Inspire-F1 阶段中断
- run_id: objectinteractioncm_all_stride_eval_20260903_235432
- run_status: FAILED
- conclusion: INVALID_IMPLEMENTATION（模型显式读取 `hand_flow`，不同 stride 的输入不同；该次产生的 GRAB 数字不得引用）

**文件与证据**

- [失败运行 manifest](../../../../../outputs/research/objectinteractioncm_all_stride_eval_20260903_235432/run_manifest.json)、[残留结果](../../../../../outputs/research/objectinteractioncm_all_stride_eval_20260903_235432/results.json) — 已标记失败，仅作为审计证据。

**原因与回滚**

错误假设在继续运行前被发现并纠正；未修改训练进程。保留失败产物以避免把无效结果误当有效结果，后续有效评估见本活动上一条记录。

## 2026-09-03 21:59:31 +0800 — V1.2.1 三卡续训 epoch 进度核查

- activity_id: ACT-20260903-215931-OBJECTINTERACTIONCM-V121-EPOCH-STATUS
- timestamp: 2026-09-03 21:59:31 +0800
- modification_version: V1.2.1.3
- type: diagnostic / operation
- change_level: L0（只读核查既有训练，仅追加活动证据）
- approval: user-requested
- approval_basis: 用户询问当前训练状态及是否已经产生完整 epoch；本次不启动、停止或调整训练
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留 ObjectInteractionCm 既有未提交实现、配置与文档）
- scope: [ObjectInteractionCm Task](../../) 的既有 V1.2.1 三卡续训进程、日志、指标和 checkpoint；不修改代码、配置、cache、模型状态或 GPU 进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（resume）
- run_status: RUNNING
- conclusion: INCONCLUSIVE（训练与逐 epoch 验证链路正常，但正式训练尚未完成，不能形成架构效果结论）

**文件与证据**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[resume manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/run_manifest_resume_20260903_190509.json) — 既有三卡恢复运行及其追溯入口。
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 当前 step、epoch、训练吞吐、ETA 与逐 epoch 验证指标。
- [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt)、[checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt) — 最近完整 epoch 与当前最佳 object-flow checkpoint。

**原因**

区分日志中的“正在训练的 epoch”和已经完成训练、验证及 checkpoint 保存的 epoch，并确认从 GPU4 迁移到 GPU0/1/2 后的续训仍在正常推进。

**验证**

- `ps -p 768731 -o pid,stat,etime,cmd`：torchrun 父进程存活；rank `768852/768853/768854` 分别占用 GPU0/1/2 约 `9.4 GiB`。
- 21:59:31 日志已推进至 step `85100 / 202300`，当前为 `epoch=21`；吞吐约 `774.7 samples/s`，日志 ETA 约 `4.03 h`。
- `metrics.jsonl` 已包含 20 条完整 `train_epoch` 和 20 条 validation：最近完成 `epoch=20` / step `83666`，`val/loss=0.00379970`、object-flow EPE `4.02003 mm`、3 cm hand-flow EPE `3.48386 mm`。
- `best.pt` 为 `epoch=15` / step `64651`，按既定 object-flow 指标取得当前最佳 `3.84406 mm`；`latest.pt` 为 `epoch=20` / step `83666`。
- `train.log` 未检出 traceback、CUDA OOM、NCCL error 或 NaN；本次只读查询未改变训练状态。

**回滚**

删除本条活动记录即可回滚文档差异；训练进程与运行产物未被修改。

## 2026-09-03 22:05:52 +0800 — V1.2.1 三卡续训收敛性核查

- activity_id: ACT-20260903-220552-OBJECTINTERACTIONCM-V121-CONVERGENCE-STATUS
- timestamp: 2026-09-03 22:05:52 +0800
- modification_version: V1.2.1.3
- type: diagnostic / operation
- change_level: L0（只读分析既有训练曲线，仅追加活动证据）
- approval: user-requested
- approval_basis: 用户询问当前训练是否收敛；本次不改变训练、配置、checkpoint 或评估口径
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留 ObjectInteractionCm 既有未提交实现、配置与文档）
- scope: [ObjectInteractionCm Task](../../) 的 V1.2.1 三卡续训曲线和运行状态；不修改代码、数据、cache、模型状态或 GPU 进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（resume）
- run_status: RUNNING
- conclusion: INCONCLUSIVE（已进入较稳定的验证区间，但训练仍在进行，不能宣称最终收敛）

**文件与证据**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 逐步、逐 epoch 训练/验证曲线及运行日志。
- [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/latest.pt)、[checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/checkpoints/best.pt) — 最近完整 epoch 和当前最佳 checkpoint。

**原因**

区分“训练损失还在下降”“验证指标进入平台”“最终收敛已被证明”三个层次，避免因单个 epoch 的改善过早停止三卡训练。

**验证**

- 22:05:52 时 epoch `21` 已完成训练和验证，当前已进入 epoch `22`（step `87600`）；torchrun 及 GPU0/1/2 仍正常运行。
- epoch 15–21 的 `val/obj/flow_epe_mm` 为 `3.844–4.052 mm`，整体围绕约 `3.93 mm` 窄幅波动；epoch 21 为 `3.865 mm`，接近当前最佳 epoch 15 的 `3.844 mm`，说明 object 分支已接近平台，但尚未严格单调收敛。
- epoch 15–21 的 `val/loss` 从 `0.0041135` 降至 `0.0036606`，epoch 21 为目前最低；`val/hand/flow_epe_3cm_mm` 从 `4.172 mm` 降至 `3.289 mm`，hand 分支仍有改善。
- 最近 5 个 validation 的均值/标准差：loss `0.003855 / 0.000114`，object EPE `3.944 / 0.078 mm`，hand EPE `3.611 / 0.197 mm`；因此判断为“已进入稳定下降/平台区”，不是“完全收敛”。
- `train.log` 未检出 traceback、CUDA OOM、NCCL error 或 NaN；未执行早停、重启或任何科研变量修改。

**回滚**

删除本条活动记录即可回滚文档差异；训练进程与运行产物未被修改。

## 2026-09-03 22:16:22 +0800 — V1.2.1 与旧 hand-root Cm 的收敛速度对齐诊断

- activity_id: ACT-20260903-221622-OBJECTINTERACTIONCM-V121-CONVERGENCE-COMPARE-CM
- timestamp: 2026-09-03 22:16:22 +0800
- modification_version: V1.2.1.3
- type: diagnostic / experiment
- change_level: L0（只读比较既有 run；不启动新评估、不修改运行）
- approval: user-requested
- approval_basis: 用户要求判断当前 ObjectInteractionCm 是否相较旧 `hand_root_t` Cm 收敛异常偏快
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留 ObjectInteractionCm 与 Cm 既有未提交改动）
- scope: [ObjectInteractionCm Task](../../) 与历史 [Cm Task](../../../Cm/) 的训练日志、配置和指标；只读，不改变任何代码、配置、cache、checkpoint 或进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（当前 resume）；cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324（旧 hand-root）；cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401（旧 object-pose 近似对照）
- run_status: RUNNING（当前 OI；旧 hand-root 已 STOPPED）
- conclusion: INCONCLUSIVE（墙钟训练确实更快，且在同坐标/短 stride 近似对照下有一定优化优势，但 hand-root 数值不能直接比较）

**文件与证据**

- [当前 OI config](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/config.json)、[当前 OI metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[当前 OI train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — `object_pose_t`、eval stride 2、当前 epoch/step 和吞吐。
- [旧 hand-root Cm config](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324/config.json)、[旧 hand-root metrics](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260828_235324/metrics.jsonl)、[旧 Cm 实验记录](../../../Cm/docs/logs/experiment_log.md) — `hand_root_t`、多 stride mean-EPE 及资源竞争记录。
- [旧 object-pose Cm config](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/config.json)、[旧 object-pose metrics](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/metrics.jsonl) — 与当前同为 object-pose、stride 2 的近似架构对照。

**原因**

用户观察到当前 OI 在约 21 个 epoch 时已达到低 EPE，需要拆分墙钟吞吐、坐标/stride 指标口径、有效监督难度、架构差异和 optimizer batch 动力学，判断是否是真实的优化加速。

**验证**

- 墙钟：当前 OI epoch 1→21 validation 间隔约 `3.19 h`；旧 hand-root Cm epoch 1→21 约 `14.53 h`。旧 run 曾与 CmDecoder 争用 GPU1，记录吞吐约 `146–148 samples/s`；当前三卡恢复后约 `773 samples/s`，因此墙钟约 `4.5×` 更快主要是资源状态改变。
- 优化预算：当前 OI epoch 21 为 step `87469`，旧 hand-root epoch 21 为 step `90090`；当前累计样本实例约 `7.67M`，旧 run 约 `8.65M`，当前不是因为看过更多样本才更低。
- 指标口径：旧 hand-root 的 `val/mean_stride_epe_mm` 同时平均 stride 1/5/10；epoch 21 三个 stride 的 source-mean object EPE 约为 `2.274/8.014/14.989 mm`，其 mean=`8.426 mm`。当前 OI 只评估 stride 2，epoch 21 object EPE=`3.865 mm`；二者不能直接比较。
- 坐标/幅度：旧 hand-root config 的 object-flow target RMS=`0.09319 m`，当前 object-pose train scale `s_obj_flow=0.05789 m`，后者约低 `38%`；object-pose frame 移除了 hand-root 运动带来的大幅相对位移，任务本身更容易。
- 近似公平对照：旧 Cm 的同为 `object_pose_t`、stride 2 run 在 epoch 1/21 为 `4.673/4.367 mm`；当前 OI 为 `4.579/3.865 mm`。在更接近的口径下，当前确有一定优化优势，但旧 run 使用 C=64、不同 decoder/scale，不能归因于单一结构因素。
- 额外混杂：当前 OI 首个有效 epoch 先以 global batch=`32` 单卡运行 `11409` steps，随后才切换 global batch=`96` 三卡；旧 hand-root 从头就是 global batch=`96`。因此 epoch 编号也不代表完全相同的 optimizer dynamics。
- 未检出当前 OI 的 traceback、OOM、NCCL error 或 NaN；本次没有停止、重启或改动任何训练变量。

**回滚**

删除本条活动记录即可回滚文档差异；所有训练进程与既有产物保持不变。

## 2026-09-02 23:44:00 +0800 — V1.1 ObjectInteractionCm 架构首版实现与 smoke

- activity_id: ACT-20260902-234400-OBJECTINTERACTIONCM
- timestamp: 2026-09-02 23:44:00 +0800
- run_id: object_interaction_cm_20260902_233920
- modification_version: V1.1.1
- type: architecture / code / data / documentation / operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 V1.1 方案并明确要求开始执行、只用 hand flow、1024 点采样、完整 4096 点计算 3 cm mask、左右流严格对齐，且暂不运行实验矩阵
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true
- run_status: COMPLETED
- conclusion: N/A（工程 smoke 仅验证链路与张量合同，不代表科研效果）
- scope: 新建独立 ObjectInteractionCm Task 的 V1.1 架构、双手数据合同、Cm/flow 模型、训练/评估/抽取入口和最小测试；不修改旧 Cm/CmDecoder、src/base、原始数据、cache 或既有 checkpoint

**文件**

- [src/task/ObjectInteractionCm/](../../) — 新 Task 实现目录，包含配置、mmap/NPZ 数据读取、严格左右手校验、4096→1024 物体采样、全池 3 cm 监督掩码、局部 hand-flow interaction、SlotAttention、双 flow decoder，以及 train/eval/extract 入口。
- [src/task/ObjectInteractionCm/docs/README.md](../README.md)、[src/task/ObjectInteractionCm/docs/plan/V1.1.md](../plan/V1.1.md)、[src/task/ObjectInteractionCm/docs/指导/V1.1.md](../指导/V1.1.md)、[src/task/ObjectInteractionCm/docs/architecture/V1.1.md](../architecture/V1.1.md) — V1.1 指导、最终计划、架构冻结快照和文档导航。
- [tests/test_object_interaction_cm.py](../../../../../tests/test_object_interaction_cm.py) — 双手严格对齐、当前帧筛选/采样、模型输出和空 mask 损失测试。
- [outputs/objectinteractioncm/object_interaction_cm_20260902_233920/](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/) — 一步 CPU BaseRunner smoke 运行目录；关键产物为 [run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/summary.json)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/metrics.jsonl) 和 [checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233920/checkpoints/latest.pt)。

**原因**

将用户确认的 V1.1 研究语义落成可执行的独立 Task：输入仅包含显式 object intrinsic 与左右 hand flow，左右手必须同时存在且帧级对齐；物体候选先按当前帧 5 cm 过滤，在完整 4096 点池上计算 3 cm mask，再采样 1024 点进入 Cm；Cm 固定为 16×32，并同时输出 object flow 与共享 hand flow。实验矩阵按用户要求留待后续，不在本活动中运行。

**验证**

- `python3 -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`。
- `python3 -m pytest -q tests/test_cm_object_v2.py tests/test_cm_slot_attention.py`：`16 passed`，未触及旧 Task 实现。
- `python3 -m py_compile src/task/ObjectInteractionCm/*.py`：通过。
- 真实有效序列 `s1/scissors_offhand_1` 读取与前向 smoke：object `[1,1024,3]`、Cm `[1,16,32]`、hand `[1,3076,3]`；不完整左右手序列按已确认合同拒绝。
- `python3 -m src.task.ObjectInteractionCm.train --set data.root=/tmp/oi_cm_smoke.oTkI5G --set data.train_path=/tmp/oi_cm_smoke.oTkI5G/grab/s1/toy --set data.val_split=0 --set data.test_fraction=0 --set data.batch_size=1 --set data.num_workers=0 --set data.persistent_workers=false --set data.min_stride=1 --set data.max_stride=1 --set train.max_steps=1 --set train.epochs=1 --set train.log_every_steps=1 --set train.eval_every_epochs=null --set train.save_every_epochs=null --set train.save_every_steps=null --set train.device=cpu`：`run_id=object_interaction_cm_20260902_233920`，`run_status=COMPLETED`；manifest/summary 可解析，`kept_frame_ratio=1.0`。该结果仅为工程 wiring 证据，科研结论为 `N/A`。
- 先前 `run_id=object_interaction_cm_20260902_233825` 也完成了一步 smoke，但发现其 `kept_frame_ratio` 分母实现有误；已修正并以 `233920` 重跑。该旧目录 [outputs/objectinteractioncm/object_interaction_cm_20260902_233825/](../../../../../outputs/objectinteractioncm/object_interaction_cm_20260902_233825/) 保留作审计，不作为验证证据。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_cm_sequence_dataset.py`：`4 passed`；系统 Python 的同一测试因缺少既有依赖 `smplx` 无法收集，但未修改其代码或数据。

**回滚**

删除本活动列出的新 Task 文件和测试即可回滚代码/文档；smoke 输出位于 `outputs/`，默认不纳入提交，可单独清理。旧 Task、共享 base、数据和既有运行产物无需回退。

## 2026-09-02 23:58:35 +0800 — V1.1 GRAB/Inspire cache 兼容性诊断

- activity_id: ACT-20260902-235835-OBJECTINTERACTIONCM-CACHE-DIAGNOSTIC
- timestamp: 2026-09-02 23:58:35 +0800
- modification_version: V1.1.2
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问是否可以开始 GRAB/Inspire 训练及是否需要导出 cache；本次只读检查，不写入 cache 或启动训练
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true
- run_status: NOT_STARTED
- conclusion: N/A
- scope: 只读核对 ObjectInteractionCm V1.1 loader 与现有 GRAB/Inspire cache 的 schema、双手流、点数、object pose 和文件完整性；未修改代码、配置、数据或运行状态

**文件与证据**

- [ObjectInteractionCm Task](../../) — 本次诊断所针对的 V1.1 loader、plan 和架构合同；仅作关联入口，未修改实现。
- [定向测试](../../../../../tests/test_object_interaction_cm.py) — 用于复核当前 loader/model 合同的既有测试，未修改。
- [GRAB cache](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/) — 1255 个 `dataset_name=grab` 序列，完整 object pool 为 4096 点、每侧 hand 为 1538 点并带 `obj_pose_world`；split 入口为 [split.json](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/splits_seed42/split.json)。
- [GRAB cache meta](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/meta.json) — 记录 source GRAB、4096 点池、1538 hand 点和 30 Hz effective FPS。
- [Inspire-F1 fast cache](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/) — 576 episode 的 layered v4 cache；每 episode 有 4096 点 object pool、1538 点单一 Inspire hand、`obj_pose_world`、hand flow 和 5 cm candidate mask，但没有 `left/` 与 `right/` 双流目录。
- [Inspire-F1 selection](../../../../../data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4/selection_all_object_disjoint_seed42.json) — 现有 object-disjoint 选择清单。
- [两个损坏文件](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s7/headphones_lift/right/hand_points_world.npy)、[两个损坏文件](../../../../../data/processed_data/cm_object_v2_surface512_object_pose_20260830/s2/mug_drink_2/right/hand_points_world.npy) — NPY header 声明的帧数据大于实际文件大小；严格 loader 会失败。

**原因**

确认“GRAB + Inspire”目前不能直接作为 V1.1 双手 Task 的混合训练输入：GRAB cache 是双手格式但有两个损坏序列；Inspire cache 是单一机器人手格式，虽具备 4096 点池和 pose/flow，却不满足已确认的左右手必须同时存在且严格对齐合同。导出或适配前必须先明确是否允许把 Inspire 单手样本纳入双手合同；不允许用复制、零填充或隐式 fallback 伪造另一只手。

**验证**

- 只读扫描确认 GRAB cache 的 1255 个序列中缺失/不完整的双手序列为 `s7/headphones_lift` 和 `s2/mug_drink_2`；两者均是 right-hand 文件不完整。
- 只读检查 Inspire-F1 fast cache 的 episode geometry/task 字段：`hand_points_world`/`hand_flow` 均为单流 `[T,1538,3]`，object pool 为 `[T,4096,3]`，不存在左右 side 文件。
- 未执行训练、cache 导出、数据迁移或实验矩阵；现有 cache、split、checkpoint 和 outputs 保持不变。

**待用户确认**

1. “GRAB 和 Inspire 手”是要混合训练两种手型，还是先分别训练/评估？
2. 若混合训练，是否要把当前单一 Inspire 手流纳入 V1.1？若是，需要批准将双手合同改为“单手或双手 union”并重新定稿 plan/architecture；若否，则只能先修复 GRAB 的两个损坏序列，Inspire 暂不进入该 Task。

## 2026-09-03 00:40:47 +0800 — V1.1.3 available-hand union cache/index、smoke 与全量训练启动

- activity_id: ACT-20260903-004047-OBJECTINTERACTIONCM-V113
- timestamp: 2026-09-03 00:40:47 +0800
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948（全量训练）；object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601（真实数据 smoke）
- modification_version: V1.1.3
- type: architecture / code / data / documentation / operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 GRAB+Inspire 同训、所有真实可用手点 union、修复两条 GRAB 损坏序列、先 smoke 再无人值守全量训练，并允许本次按建议直接执行
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留既有 CmDecoder 日志改动）
- run_status: RUNNING
- conclusion: N/A（smoke 仅为工程链路证据；全量训练尚未形成科研结论）
- scope: 仅修改/新增 ObjectInteractionCm Task、V1.1 计划/架构/指导修订、定向测试和 cache index 工具；不改旧 Cm/CmDecoder、src/base、原始 cache、旧 split、checkpoint 或既有训练进程；全量训练仅使用 GPU4 单卡

**合同与数据产物**

- [V1.1 指导](../指导/V1.1.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 架构](../architecture/V1.1.md) — 冻结 available-hand union、GRAB/Inspire 0.5/0.5 混训、GRAB stride 1..10、Inspire 偶数 2..20、16×32 Cm 和 4096→1024 object sampling 合同。
- [混合 cache index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 1459 train / 184 val / 188 test 序列，其中 GRAB 1004/126/125、Inspire-F1 455/58/63；路径只引用现有 cache。
- [GRAB 修复 cache](../../../../../data/processed_data/object_interaction_cm_grab_repairs_20260903/) — 重新导出 `s7/headphones_lift` 与 `s2/mug_drink_2`，未覆盖原损坏目录。

**原因**

用户已确认将原先严格左右手合同调整为“所有真实可用手点 union”：GRAB 使用真实左右手，Inspire-F1 使用真实单手并以 valid mask 批处理；因此需要在不改原 cache 的前提下建立 source adapter/index，修复两条损坏 GRAB 序列，并用同一 16×32 架构启动混合训练。

**验证**

- `python3 -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`；`python3 -m compileall -q src/task/ObjectInteractionCm`：通过。
- 真实 GRAB/Inspire 前向+反向 smoke：两个 source 均有限值、`hand_points=[1,3076,3]`，GRAB 有效手点 3076、Inspire 有效手点 1538，batch32 一步 GPU smoke 峰值约 3.6 GB，未发生 OOM。
- [真实数据 smoke 运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601/) — `run_id=object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601`，`run_status=COMPLETED`；[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003601/summary.json)；等 source object-flow EPE = 7.645 mm，属于工程 smoke 指标，不代表科研效果。
- [全量训练运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/) — `run_id=object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948`，`run_status=RUNNING`，检查时已推进至约 step 3500；[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl) 和 [train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/train.log) 已生成；`summary.json`、`checkpoints/latest.pt` 当前为 `PENDING`。训练命令为 `CUDA_VISIBLE_DEVICES=4 setsid nohup python3 -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_inspire_f1_v1_1.yaml`。

**回滚与保护**

- 代码/文档回滚入口为删除本次新增的 `src/task/ObjectInteractionCm/`、`tests/test_object_interaction_cm.py` 及本条活动对应修改；原始 cache 和 repair 源均可独立保留。
- 未停止或修改 GPU0-2 上已有 CmDecoder 训练，未写入 `src/base/`、旧 Task 或仓库外路径；全量训练使用独立输出目录与 GPU4。

## 2026-09-03 04:56:22 +0800 — V1.1.3 GRAB/Inspire 全量训练完成

- activity_id: ACT-20260903-045622-OBJECTINTERACTIONCM-TRAIN-COMPLETED
- timestamp: 2026-09-03 04:56:22 +0800
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- modification_version: V1.1.3
- type: experiment / operation / diagnostic
- change_level: L3
- approval: user-approved
- approval_basis: 用户已批准按 V1.1 最终计划执行 GRAB+Inspire 全量无人值守训练；本条仅记录该既有运行的终态与只读指标核查
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留既有 CmDecoder 日志与 ObjectInteractionCm 未提交实现）
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（训练链路稳定完成，但尚无 held-out test 评估或基线对照，不能据此判定架构科研效果）
- scope: 只读核对全量训练进程、终态 summary、逐 epoch 验证指标、最佳/最终 checkpoint 和 NaN/错误信号；不运行新评估，不改模型、配置、cache、checkpoint 或其他进程

**文件**

- [docs/README.md](../README.md)、[docs/logs/activity_log.md](activity_log.md) — 增加实验记录导航并闭合训练终态。
- [全量训练运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/) — 完整训练产物。
- [config.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)、[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 配置、输入合同与终态摘要。
- [metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl)、[train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/train.log) — 逐步/逐 epoch 指标和日志。
- [checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt)、[checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/latest.pt) — 等 source object-flow EPE 最优 checkpoint 与最终 checkpoint。
- [experiment_log.md](experiment_log.md) — 本次正式训练的假设、结果和结论。

**原因**

全量训练已自然达到 `max_steps=202300`，需要将此前 `RUNNING` 启动事件闭合为终态，并区分工程完成证据与尚未充分验证的科研效果。

**验证**

- 终态 `summary.json`：`run_status=COMPLETED`、`global_step=202300`、`epoch=18`，训练耗时约 `04:16:24`；原训练 PID 已退出，GPU4 已释放。
- `metrics.jsonl` 共 18 次 epoch validation；所有数值字段均为有限值，`train.log` 未发现 traceback、error、NaN 或 Inf。
- `best.pt`：step 193953 / epoch 17，等 source object-flow EPE `4.089469 mm`；`latest.pt`：step 202300 / epoch 18，保存的 best metric 同为 `4.089469 mm`。
- 最终 epoch validation：GRAB object/hand EPE `5.851712 / 2.728593 mm`，Inspire-F1 object/hand EPE `2.343177 / 0.774676 mm`，等 source object EPE `4.097444 mm`。
- 未执行 held-out test 或对照实验；因此工程运行结果有效，科研结论保持 `INCONCLUSIVE`。

**回滚**

本条仅增加 Task-local 文档记录；删除本条 activity 与对应 experiment 条目即可回滚记录，不影响训练产物和 checkpoint。

## 2026-09-03 08:42:06 +0800 — V1.1.3 训练收敛与 batch/GPU 等效性核查

- activity_id: ACT-20260903-084206-OBJECTINTERACTIONCM-CONVERGENCE-CHECK
- timestamp: 2026-09-03 08:42:06 +0800
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问既有训练的收敛性、步数、batch size、GPU 数和与旧 Cm 的等效性；本次只读核对
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（当前训练在既定 cosine schedule 下已进入平台，但未进行 held-out test 或严格同算力对照）
- scope: 只读检查 ObjectInteractionCm 终态配置/曲线与旧混合 Cm、旧 CmDecoder 的实际 train_setup；未启动、停止或修改任何运行、模型、配置、cache 或 checkpoint

**文件**

- [ObjectInteractionCm config.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)、[metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 当前训练设置、18 个 epoch 验证曲线和终态。
- [旧混合 Cm train.log](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/train.log) — 旧 Cm 实际 world size 与 global batch 入口。
- [旧 CmDecoder train.log](../../../../../outputs/cmdecoder/cm_decoder_20260901_151052/train.log) — 旧 CmDecoder 实际 world size 与 global batch 入口。
- [本活动记录](activity_log.md) — 本次核查记录。

**原因**

需要区分“每卡 micro-batch 相同”“全局 batch 相同”“optimizer step 数相同”和“总样本暴露量相同”，这些条件不能互相替代。

**验证**

- 当前 ObjectInteractionCm：`world_size=1`、GPU4、`per_device_batch=32`、`global_batch=32`、`total_steps=202300`；配置的 `epochs=50` 是安全上限，实际由 `max_steps` 在 epoch 18 结束。
- 旧混合 Cm：实际 `world_size=3`、每卡 batch `32`、`global_batch=96`、`total_steps=213150`；因此当前 run 仅与其每卡 batch 相同，不与其 global batch 或单步样本量等效。
- 旧 CmDecoder：实际 `world_size=3`、每卡 batch `16`、`global_batch=48`；当前 run 也不与该运行的 global batch 等效。
- 当前验证等权 object EPE：epoch 1 `7.3993 mm`，epoch 17 最佳 `4.0895 mm`，epoch 18 `4.0974 mm`；最后 5 个 epoch 均值 `4.1216 mm`、标准差 `0.0271 mm`，最终仅比最佳高 `0.0080 mm`。epoch 18 learning rate `4.20e-7`，已接近 cosine schedule 终点。
- 当前训练无 NaN/Inf 或 traceback；最佳 checkpoint 为 epoch 17 / step 193953，最终 checkpoint 为 epoch 18 / step 202300。
- 当前步数是旧混合 Cm 的 `95.7%`，但按 optimizer step 不能视作同等样本预算：当前约 `202300×32=6.47M` 样本实例，旧混合 Cm 约 `213150×96=20.46M` 样本实例，且两者数据/模型合同不同。

**回滚**

本条只增加诊断记录；删除本条即可回滚，不影响任何运行产物。

## 2026-09-03 11:06:35 +0800 — V1.1.3 ObjectInteractionCm 全 stride 离线评估

- activity_id: `ACT-20260903-110635-OBJECTINTERACTIONCM-STRIDE-EVAL`
- timestamp: 2026-09-03 11:06:35 +0800
- modification_version: V1.1.3
- type: diagnostic / experiment
- change_level: L0（只读 checkpoint 与现有 val/test cache，生成独立评估产物）
- approval: user-requested
- approval_basis: 用户要求在代码提交后离线核查 Cm 统计是否为各 stride 均值
- skills_used: research-change-control, research-experiment-workflow
- branch: `oyx`
- base_commit: `991518adc7fa60600007111a3ecd76d609bb577c`
- worktree_dirty: true（保留根级治理文档与既有 Task-local 日志改动）
- scope: `src/task/ObjectInteractionCm/` 的既有 checkpoint/cache 离线评估；不修改模型、配置或训练进程
- run_id: `objectinteractioncm_stride_eval_20260903_104821`
- run_status: `COMPLETED`
- command: `CUDA_VISIBLE_DEVICES=4 /home2/wyy/miniconda3/envs/graspenv/bin/python -u <inline full-stride evaluator>`（batch=128，num_workers=16）
- conclusion: `INCONCLUSIVE`（评估证据有效；本条回答统计口径，不构成新架构优劣结论）

**范围与口径**

- checkpoint：`outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/checkpoints/best.pt`，SHA256=`146cbfc87fa9a17987ab544d53b47cf51efd3a505790ac94e0c031865cd660f4`。
- val/test 全量遍历 40 个组合：GRAB stride `1..10`，Inspire-F1 stride `2,4,...,20`；每个组合固定 stride、active-only、1024 object points。
- 同时记录 object-flow EPE、zero-flow EPE、3 cm 手点 EPE、有效点数和样本数；未修改 cache、checkpoint 或训练进程。

**原因**

- 需要确认既有 `experiment_log.md` 的 EPE 是否代表各 stride 均值，并补充 GRAB/Inspire-F1 全跨度的可复核证据。

**关键发现**

- `experiment_log.md` 中 best epoch 17 的 GRAB `5.823295 mm`、Inspire-F1 `2.355643 mm` 与本次 **val stride=2** 的 `5.8233/2.3556 mm` 完全对应；它们不是各自所有 stride 的均值。
- 因此，现有训练日志/实验表的 EPE 口径是：固定 `eval_stride=2`，并按 source loader（GRAB、Inspire-F1）分别统计；不是 GRAB 1–10 或 Inspire-F1 2–20 的 stride 聚合均值。
- 全 stride 结果显示 object EPE 随跨度增加：test/GRAB `2.9515→27.8845 mm`（stride1→10），test/Inspire-F1 `2.9558→17.9355 mm`（stride2→20）；zero-flow 基线和手 EPE 也随跨度增加。

**产物与验证**

- `src/task/ObjectInteractionCm/docs/logs/` — Task-local 活动、实验和历史记录目录（保留既有未提交记录）。
- [评估结果](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/results.json)
- [run manifest](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/run_manifest.json)
- [summary](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/summary.json)
- 40/40 组合正常完成，所有 EPE 为有限值；独立 GPU4 运行，GPU0–2 上的 CmDecoder V1.2 全量训练未停止。

## 2026-09-03 18:28:06 +0800 — V1.2.1 架构实现、scale/miss 诊断、smoke 完成并启动全量训练

- activity_id: ACT-20260903-182806-OBJECTINTERACTIONCM-V121
- timestamp: 2026-09-03 18:28:06 +0800
- modification_version: V1.2.1.1
- type: architecture / code / data / documentation / operation
- change_level: L2（Task 内模型、loss、配置和训练流程；不修改共享 base、坐标/GT、split 或 cache schema）
- approval: user-approved
- approval_basis: 用户明确要求按指导/V1.2.1 修改架构，直接写最终 plan，复用可用 cache，先 smoke 后全量并开始执行
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留仓库既有治理、CmDecoder 和历史 ObjectInteractionCm 修改）
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806
- run_status: RUNNING
- conclusion: INCONCLUSIVE（smoke 只证明工程链路；全量尚未完成，不能判定科研效果）
- scope: [ObjectInteractionCm V1.2.1 Task](../../)；不修改旧 V1.1 输出、旧 checkpoint、原始 cache、split、Cm/CmDecoder 或共享 `src/base`

**文件**

- [ObjectInteractionCm Task 全部本次实现范围](../../) — 新增/更新 V1.2.1 架构、最终 plan、D128/C32 模型、双路 interaction、mask-before-top-k、sample-level mask、masked SlotAttn、additive object/hand decoder、runner、配置和 scale/miss 工具。
- [V1.2.1 架构](../architecture/V1.2.1.md)、[V1.2.1 最终 plan](../plan/V1.2.1.md)、[V1.2.1 指导](../指导/V1.2.1.md) — 文档合同和执行范围。
- [train-only scales](../../../../../data/processed_data/object_interaction_cm_v1_2_1/scales_train.json) — 复用现有 `object_interaction_cm_v1_1/index.json` 的训练 split 统计，未导出 cache；`s_geo=0.0321234m`、`s_hand_flow=0.0701309m`、`s_obj_flow=0.0578926m`。
- [sampling miss](../../../../../data/processed_data/object_interaction_cm_v1_2_1/sampling_miss.json) — 512 个抽样样本，`N_active_4096` 均值 1759.42、`N_active_1024` 均值 439.99，条件 miss `0.0%`，因此 `sampling_retry_attempts=0`。
- [smoke run](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/) — `run_status=COMPLETED`；两步 CUDA smoke 的 [manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/metrics.jsonl) 和 [latest checkpoint](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708/checkpoints/latest.pt) 已生成。
- [full run](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/) — `run_status=RUNNING`；[manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/run_manifest.json)、[metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) 已生成；终态 checkpoint/summary 为 `PENDING`。
- [full launcher log](../../../../../outputs/objectinteractioncm/v1_2_1_full_launcher.log) — 全量命令输出入口。

**原因**

落实 V1.2.1 的延迟压缩和物体侧聚集设计：interaction/hand flow 在 D=128 保留，SlotAttn 后才投影到 16×32 Cm；object decoder 仅使用原始 point/normal + Cm + anchor geometry；两个 decoder 都沿 slot contribution 直接求和。空采样样本保持 B 维并用 `sample_valid` 屏蔽监督，dummy 仅保持 SlotAttn 数值合法。现有混合 index/schema 已被 loader 直接接受，因此不重复导出 cache。

**验证**

- `python3 -m py_compile src/task/ObjectInteractionCm/*.py src/task/ObjectInteractionCm/tools/data/*.py`：通过。
- `python3 -m pytest -q tests/test_object_interaction_cm.py`：`3 passed`。
- V1.1 best checkpoint 严格加载：通过兼容 legacy model path，未改变旧 checkpoint。
- scale/miss 两个只读脚本正常完成；统计产物见上方链接，条件 miss 低于启用 retry 的阈值。
- CUDA smoke `run_id=object_interaction_cm_grab_inspire_f1_v1_2_1_smoke_20260903_182708`：`COMPLETED`，2 steps，forward/backward 无 NaN，sample mask、两路 additive decoder、manifest 和 checkpoint 均正常；这是工程 smoke 证据，不是科研结论。
- 全量命令：`setsid env CUDA_VISIBLE_DEVICES=4 nohup /home2/wyy/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_inspire_f1_v1_2_1.yaml`；启动时约 `279 samples/s`，估计约 6.4 小时，当前仍为 `RUNNING`。

**回滚**

删除/恢复本条列出的 V1.2.1 Task 代码、配置、文档和诊断脚本即可回滚实现；scale/miss JSON 与 `outputs/` 运行目录为新增可删除产物，旧 V1.1 cache/checkpoint/output 不受影响。全量运行可安全停止后保留已写入 manifest/log/checkpoint。

## 2026-09-03 18:36:00 +0800 — V1.2.1 空样本数值安全与 provenance 补强

- activity_id: ACT-20260903-183600-OBJECTINTERACTIONCM-V121-SAFETY
- timestamp: 2026-09-03 18:36:00 +0800
- modification_version: V1.2.1.2
- type: code / documentation
- change_level: L1
- approval: auto
- approval_basis: 按 V1.2.1 已批准架构补齐 dummy token 的零输入和 scale manifest 元数据记录；不改变有效 interaction 路径或研究变量
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806
- run_status: RUNNING
- conclusion: INCONCLUSIVE（仅数值安全/可追溯性补强，不产生科研结论）
- scope: [ObjectInteractionCm V1.2.1 Task](../../)；当前全量 run 继续在 GPU4 运行，旧代码/cache/checkpoint 不改

**文件**

- [ObjectInteractionCm Task](../../) — SlotAttn 对全空样本强制零 dummy 输入，runner provenance 记录 scale values，训练入口说明更新。
- [全量运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/) — 运行保持 `RUNNING`，不重启、不覆盖已有产物。

**原因**

避免空样本的无效 token 传播非零值，并让后续新运行的 metadata 明确记录 `s_geo/s_hand_flow/s_obj_flow`。该补强只影响 `sample_valid=0` 的数值安全分支；当前抽样 miss 统计为 0%，不改变有效样本的 forward 或 loss 语义。

**验证**

- `python3 -m py_compile src/task/ObjectInteractionCm/model.py src/task/ObjectInteractionCm/runner.py src/task/ObjectInteractionCm/slot_attention.py`：通过。
- 直接构造全空 mask 的 SlotAttn：slots/assignment/slot_weights 全 finite，dummy_used=True，active slot weight=0。
- `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过，最新条目链接可导航。
- 全量运行检查：PID 710877 仍存活，metrics 已推进至约 step 2600，无 NaN/traceback；该运行的 manifest 已在补强前生成，scale 数值仍可由 config 指向的 JSON 复核。

**回滚**

恢复本条涉及的 Task 文件即可回滚补强；不停止或删除正在运行的全量任务及其产物。

## 2026-09-03 19:04:57 +0800 — ObjectInteractionCm 从 GPU4 恢复到 GPU0/1/2 三卡

- activity_id: ACT-20260903-190457-OBJECTINTERACTIONCM-V121-MIGRATE
- timestamp: 2026-09-03 19:04:57 +0800
- modification_version: V1.2.1.3
- type: operation / diagnostic
- change_level: L3（停止并恢复长时训练，改变 world size；训练变量保持配置不变）
- approval: user-approved
- approval_basis: 用户明确要求停止 GPU0/1/2 decoder 训练并将 ObjectInteractionCm 放到三张卡
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true
- scope: [ObjectInteractionCm Task](../../)；不修改 cache、旧 checkpoint 或其他 GPU 进程
- run_id: object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806（resume）
- run_status: RUNNING
- conclusion: INCONCLUSIVE（训练尚未完成）

**文件**

- [原 GPU4 run 目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/) — 保留 step `11409` 的 `latest.pt/best.pt`，作为恢复入口。
- [三卡 resume launcher log](../../../../../outputs/objectinteractioncm/v1_2_1_full_3gpu_resume_launcher.log) — `CUDA_VISIBLE_DEVICES=0,1,2`、torchrun 3 ranks 的启动输出。
- [resume manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/run_manifest_resume_20260903_190509.json) — 记录本次显式 resume 和 global batch=96。
- [当前 metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/metrics.jsonl)、[当前 train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_2_1_20260903_182806/train.log) — 恢复后继续追加。

**原因**

GPU0/1/2 的 decoder 已按用户指令停止且 GPU1/2 释放；GPU0 剩余 viewer 不占用训练合同。ObjectInteractionCm 原单卡运行在 step `11409` 已保存 checkpoint，因此不从零开始，而是显式从该 checkpoint 恢复到 3 卡，保持模型、数据 split、scale manifest、max_steps 和 loss 不变，仅将 world size 改为 3（per-device batch=32，global batch=96）。

**验证**

- 三卡启动成功：world size=3，rank 0/1/2 均完成 checkpoint load，日志显示 `Loaded checkpoint ... at step 11409`。
- 恢复后已推进至约 step `11700`，三张卡显存分别约 12.0/9.4/9.4 GiB；当前无 OOM、NaN、NCCL 或 traceback。
- GPU4 已释放到约 7 MiB；0 卡 viewer 进程保持运行，未误杀。
- `audit_diff.py --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过。

**回滚**

停止当前三卡 run 后可从原始 step `11409` checkpoint 重新选择设备恢复；本次未删除任何 decoder/ObjectInteractionCm 运行产物或 cache。

**验证**

- 结果文件包含 val/test × GRAB 10 个 stride × Inspire-F1 10 个 stride，共 40 条记录；`run_manifest.json` 与 `summary.json` 均为 `COMPLETED`。
- `experiment_log.md` 中的 best epoch 17 source 指标与 val stride=2 结果逐项对应，确认其统计口径不是各 stride 均值。

**回滚**

- 本条仅增加诊断记录；评估目录为独立生成产物，不影响代码、cache、checkpoint 和训练运行。

## 2026-09-03 10:38:55 +0800 — V1.1.3 运行 JSON 职责与重复字段诊断

- activity_id: ACT-20260903-103855-OBJECTINTERACTIONCM-JSON-AUDIT
- timestamp: 2026-09-03 10:38:55 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问既有训练运行目录中 JSON 是否存在职责重叠；本次只读比较文件结构、字段和值，不修改运行产物、配置或 checkpoint
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留用户既有 CmDecoder 与 ObjectInteractionCm 未提交改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（本条只诊断追溯 JSON 结构，不新增科研效果证据）
- scope: `config.json`、`metadata.json`、`run_manifest.json`、`summary.json` 的职责边界、精确重复字段和嵌套重复；未运行新实验

**文件**

- [运行目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/) — 本次只读检查范围。
- [config.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/config.json)、[metadata.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metadata.json)、[run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/run_manifest.json)、[summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 被比较的四个 JSON。

**原因**

当前运行目录同时保存配置、数据合同、运行追溯和终态指标；需要核对重复是否超出各自职责，避免后续收缩 JSON 时破坏 checkpoint/inference 兼容性。

**验证**

- `metadata.json` 的 22 个字段及其值被 `run_manifest.json.dataset_metadata` 完整复制；`run_manifest.json.contract` 另重复其中 6 个核心合同字段。
- `run_manifest.json` 与 `summary.json` 精确重复 `task`、`run_name`、`mode`、`output_dir`、`modification_version`；`summary.json` 另以 `run_id==run_name` 再记录一次运行身份。
- `config.json` 未被完整展开进 manifest，仅重复 `component_registry`、`components`、`modification_version`、`operation_category`；`summary.json` 仅重复 `modification_version` 并引用关键产物。
- 当前四个 JSON 均可解析；未修改任何产物。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响运行目录。
- 建议后续新 schema 让 manifest 引用 metadata 快照而不再内嵌完整 `dataset_metadata`；在审计 checkpoint/inference 消费者前，不删除运行目录 `metadata.json`。

## 2026-09-03 10:47:12 +0800 — V1.1.3 summary.json 消费者审计

- activity_id: ACT-20260903-104712-OBJECTINTERACTIONCM-SUMMARY-CONSUMERS
- timestamp: 2026-09-03 10:47:12 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问 summary.json 的实际读取者；本次只读搜索生成器、消费者和测试引用，不修改运行产物或代码
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留用户既有 CmDecoder 与 ObjectInteractionCm 未提交改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（本条只审计 JSON 消费关系，不新增科研效果证据）
- scope: 仓库内 `summary.json` 的生成调用、实际读取调用、测试和文档引用；重点核对 BaseRunner 运行目录

**文件**

- [BaseRunner 终态摘要写入](../../../../../src/base/base_runner.py) — train/eval 终态调用 `write_run_summary`。
- [summary.json 写入实现](../../../../../src/base/run_manifest.py) — 定义 `ref2dex.run_summary.v1` 结构。
- [运行目录 summary.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/summary.json) — 本次被审计的实际文件。
- [运行追溯与目录规范](../../../../../docs/目录规范.md) — 将 summary 定义为用户可读终态入口。

**原因**

需要确认 summary 是 BaseRunner 的机器依赖，还是仅作为用户/审计入口，从而判断它是否可以与其他运行 JSON 合并。

**验证**

- 对本次 ObjectInteractionCm 的 BaseRunner 运行，仓库内没有发现任何代码读取该 `summary.json`；BaseRunner 只负责写入，恢复训练读取 checkpoint/history，不读取 summary。
- `tests/test_run_manifest.py` 只读取临时目录中的 summary 以验证写入合同，不消费实际训练产物。
- 发现的实际 summary 读取者仅是旧 `PointWorldWAM` 自定义训练/诊断脚本，用于其自身 resume 或分析，不适用于 ObjectInteractionCm 的 BaseRunner 输出。
- `summary.json` 的当前定位因此是用户查看、activity/experiment 导航和外部审计入口，而不是训练循环的必需状态文件。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响运行目录。
- 若未来确认没有外部归档工具依赖，可考虑将“终态摘要”改为可选产物，但不建议删除；它应继续保留为稳定的人类/审计接口。

## 2026-09-03 10:47:12 +0800 — V1.1.3 config/meta、dataset_split 与 checkpoint 消费关系核对

- activity_id: ACT-20260903-104712-OBJECTINTERACTIONCM-CONTRACT-AUDIT
- timestamp: 2026-09-03 10:47:12 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问 dataset_split 是否来自 config/meta，以及 config、metadata 与 checkpoint 的区别；本次只读核对配置、数据 index 和共享 checkpoint 代码
- skills_used: research-experiment-workflow, research-change-control
- branch: oyx
- base_commit: 7b63dc50896392023c5c0f33c7d508cbe4436b50
- worktree_dirty: true（保留用户既有 CmDecoder 与 ObjectInteractionCm 未提交改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（本条只核对追溯合同，不新增科研效果证据）
- scope: ObjectInteractionCm `config.json`/`meta`、运行时 `metadata.json`、输入 index `counts`、checkpoint 内部 config/metadata 及 BaseRunner 消费路径

**文件**

- [ObjectInteractionCm 配置定义](../../../../../src/task/ObjectInteractionCm/config.py)、[active 配置](../../../../../src/task/ObjectInteractionCm/configs/active/grab_inspire_f1_v1_1.yaml) — 静态配置意图和模型/数据参数。
- [ObjectInteractionCm dataloader](../../../../../src/task/ObjectInteractionCm/dataset.py) — 根据实际 index entries 生成 `dataset_split`/`source_split`。
- [输入 index](../../../../../data/processed_data/object_interaction_cm_v1_1/index.json) — 保存 sequences 和 source-level counts。
- [checkpoint 管理](../../../../../src/base/checkpoint.py)、[BaseRunner](../../../../../src/base/base_runner.py) — 保存并消费 checkpoint 内的 config/metadata。

**原因**

需要区分静态配置、实际数据事实和 checkpoint 自包含信息，才能决定 manifest 中哪些字段可安全删除而不丢失追溯能力。

**验证**

- 本次运行的 `config.json`/`meta` 不包含解析后的 `dataset_split`；只包含 `data.index_path`、split 策略参数和静态形状/阈值。
- `dataset_split`/`source_split` 在 dataloader 读取 index 后按实际 entries 计算；输入 index 的 `counts` 保存同一 source-level 事实。
- `CheckpointManager.save` 将运行时 `metadata` 和完整 `config` 一起写入 `.pt`；`build_runner_from_checkpoint` 用 checkpoint 的 `config` 重建配置，`setup_inference` 用 checkpoint 的 `metadata` 配置数据。
- 输出目录的 `metadata.json` 是 checkpoint metadata 的外部快照；当前代码没有从该输出 JSON 反向加载 checkpoint。删除外部快照不会等价于删除 checkpoint 内部 metadata，但会损失独立查看入口。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响运行目录。
- 若用户确认精简 manifest，优先移除完整 `dataset_metadata` 展开；`dataset_split` 是否保留为最小 contract 字段，再根据是否需要 manifest 单文件快速审计决定。

## 2026-09-03 11:35:41 +0800 — ObjectInteractionCm 物体预测劣化原因诊断

- activity_id: ACT-20260903-113541-OBJECTINTERACTIONCM-OBJECT-REGRESSION-DIAGNOSIS
- timestamp: 2026-09-03 11:35:41 +0800
- modification_version: V1.1.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求比较 ObjectInteractionCm 与 Cm 的物体预测；本次仅阅读代码、配置、日志和既有运行产物，不修改实现、数据或研究变量
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户既有治理、CmDecoder 与 ObjectInteractionCm 文档改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948；comparison_run_id: cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（确认一个 KNN padding 实现缺陷及多项强烈的基线不等价因素；尚未形成研究效果结论）
- scope: ObjectInteractionCm 物体侧 KNN、padding mask、loss/target scaling、数据采样与旧 Cm C64 基线的训练初始化、batch 和指标口径

**原因**

需要解释 ObjectInteractionCm 的物体 EPE 为什么看起来不如原始 Cm，同时区分实现缺陷、训练条件不等价和物体侧聚集设计本身的影响。审计时发现工作区已有用户改动 [ObjectInteractionCm 架构文档](../architecture/V1.1.md) 与 [ObjectInteractionCm 计划文档](../plan/V1.1.md)，本次未读取其 diff 作为科学改动，也未修改它们。

**文件**

- [ObjectInteractionCm KNN 实现](../../model.py) — `LocalHandInteraction.forward` 在 `topk` 前未将无效 hand padding 距离置为无穷大。
- [ObjectInteractionCm 数据集](../../dataset.py) — Inspire hand 1538 个有效点后追加 1538 个零 padding；物体点从完整池均匀采样且全部计入 object loss。
- [已有架构文档改动](../architecture/V1.1.md)、[已有计划文档改动](../plan/V1.1.md) — 工作区既存用户改动，保留且未纳入本次诊断结论。
- [Cm 模型](../../../Cm/src/model.py)、[Cm runner](../../../Cm/src/runner.py) — 旧路径使用 DenseToken hand 特征；C64 基线启用 target scaling 与稠密 hand 监督。
- [ObjectInteractionCm 运行指标](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948/metrics.jsonl) — 记录 source-level 邻居数、交互物体点比例和验证 EPE。
- [ObjectInteractionCm 全 stride 评估](../../../../../outputs/research/objectinteractioncm_stride_eval_20260903_104821/results.json) — 检查不同 stride 下的测试行为。
- [Cm C64 基线运行](../../../../../outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401/metrics.jsonl) — closest apples-to-apples 旧基线。

**关键证据**

- Inspire 首个验证 batch 的 raw top-8 KNN 平均仅 4.31 个有效点，半径内有效邻居平均 0.12；先 mask 无效点再 top-k 时分别为 8.00 和 0.39。raw top-k 中约 41.8% 的物体点没有任何有效邻居。GRAB 无 padding，因此不受该缺陷影响。
- ObjectInteractionCm 最佳验证 equal-source object EPE 为 4.0895 mm；旧 Cm C64 为 4.0367 mm，差约 0.0527 mm（1.31%）。source-level 上 OI 的 GRAB 略好（约 0.0716 mm），但 Inspire 差约 0.1771 mm，劣化并非两个 source 对称发生。
- 旧 Cm C64 使用 global batch 96、约 153540 steps，并从 adapted DenseToken checkpoint 初始化；OI 使用 global batch 32、约 202300 steps、随机初始化。按样本实例数估算，OI 约为旧基线的 44%，因此 step 数不能视为训练量相当。
- OI 为 C32、1024 object points、16 slots（约 64 点/slot）；旧 C64 为 512 object points、16 slots（约 32 点/slot），OI 的 token 表征容量和每 slot 压缩比更差。
- OI 的 hand loss 只在 3 cm 接触区域生效（全体约 18.4%，Inspire 约 6.95%）；旧 C64 hand loss 对有效 hand 点提供稠密监督。OI 同时合并左右手为一个共享集合，旧 Cm 分侧处理，监督和任务语义也不完全相同。
- 旧 Cm object-v2 loader 的语义是按 5 cm candidate mask 产生 `obj_valid_mask`，而 OI 将采样点全部标为 valid；但抽查/汇总当前旧 GRAB cache 的 764737 帧 candidate 数均为 4096，因此该项在这次具体 C64 run 中不是主要差异，仍应在复现实验时显式对齐。

**验证**

- 只读检查上述源码、配置快照、数据 index、既有 metrics/summary 和全 stride results；未改代码或运行产物。
- KNN 对比使用同一 batch 分别执行原始 top-k 与 invalid-distance=`inf` 的 masked top-k，结论只用于实现诊断，不替代独立修复后的回归实验。

**回滚与后续**

- 删除本条 activity 即可回滚本次诊断记录，不影响代码、数据或已有运行。
- 若进入 change，首要修复是 KNN 前 mask padding；随后在相同初始化、global batch、采样/分侧语义、target scaling 和评估 stride 下重跑，才能判断物体侧聚集设计本身是否劣化。

## 2026-09-03 14:28:37 +0800 — V1.2 指导补充实现不变量与公平比较合同

- activity_id: ACT-20260903-142837-OBJECTINTERACTIONCM-V12-GUIDANCE-SUPPLEMENT
- timestamp: 2026-09-03 14:28:37 +0800
- modification_version: V1.2.1
- type: documentation
- change_level: L2
- approval: user-approved
- approval_basis: 用户明确要求浏览 V1.2 指导并将上一轮诊断结论补充进去；本次只修改指导文档，不修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户既有治理、CmDecoder 以及 ObjectInteractionCm 架构/计划文档改动）
- run_id: object_interaction_cm_grab_inspire_f1_v1_1_20260903_003948；comparison_run_id: cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_194401
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（补充已确认的 padding KNN 缺陷、比较合同和诊断边界；不宣称 object-side 方案效果成立）
- scope: `src/task/ObjectInteractionCm/docs/指导/V1.2.md`；补充 KNN mask-before-topk、无交互 object token 策略、baseline parity、处理宽度/尺度、监督/评估口径和配置阈值可追溯性

**文件**

- [V1.2 指导](../指导/V1.2.md) — 新增“本轮诊断补充”章节，将 padding KNN 缺陷升格为 P0 不变量，补充 object-side 稀疏交互、旧 Cm C64 公平比较、训练预算、source/stride 指标口径、slot collapse 排除项及配置合同要求。
- [已有 V1.1 架构文档改动](../architecture/V1.1.md)、[已有 V1.1 计划文档改动](../plan/V1.1.md) — 工作区既存用户改动，仅为 scope 审计列出，本次未修改。

**原因**

上一轮只读诊断确认 `LocalHandInteraction` 在 `topk` 前未排除 Inspire-F1 的 zero padding，且 OI 与旧 Cm 在初始化、global batch、DenseToken、Cm/processing width、监督密度和评估口径上均不完全等价。V1.2 若不把这些事项写成冻结合同，后续实现或实验会继续混入不可解释变量。

**验证**

- 只读复核 [ObjectInteractionCm model.py](../../model.py)、[dataset.py](../../dataset.py)、[runner.py](../../runner.py)、旧 [Cm model.py](../../../Cm/src/model.py) 与 [Cm runner.py](../../../Cm/src/runner.py)，以及 OI/旧 Cm 的 metrics、train.log、full-stride results。
- `git diff --check`：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 3 个 scope 变更路径一致，本地链接可导航。
- 未运行训练、评估、测试或数据处理；新增内容属于研究指导和比较合同，不构成效果证据。

**回滚与后续**

- 删除 [V1.2 指导](../指导/V1.2.md) 末尾新增章节即可回滚本次文档补充，不影响代码、数据、checkpoint 或已有运行。
- 进入 `change` 前仍需为 V1.2 起草并与用户定稿同版本 `plan/V1.2.md`；KNN 修复和 parity 实验不得在 plan 未定稿前执行。

## 2026-09-03 16:33:30 +0800 — V1.2 执行计划草案与架构提案

- activity_id: ACT-20260903-163330-OBJECTINTERACTIONCM-V12-PLAN-ARCHITECTURE
- timestamp: 2026-09-03 16:33:30 +0800
- modification_version: V1.2.2
- type: architecture / documentation
- change_level: L2
- approval: user-approved（仅批准起草计划和架构；实现仍待计划定稿）
- approval_basis: 用户明确要求先写一版 V1.2 计划和详细 architecture；本次只新增文档，不修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（设计提案和执行草案，不构成科研效果结论）
- scope: V1.2 的 hard interaction object mask、KNN mask-before-topk、128 维 late compression、interaction/hand-flow 与 object decoder 的信息隔离、masked SlotAttn、loss/评估合同和回滚边界

**文件**

- [V1.2 执行计划草案](../plan/V1.2.md) — 分阶段文件范围、张量合同、验证命令、parity 实验冻结项和审批闸门；状态为 `draft-for-review`。
- [V1.2 架构提案](../architecture/V1.2.md) — 详细的数据流、`[B,N,128]` 局部特征、`[B,16,128]→[B,16,32]` late projection、硬 mask 语义、decoder/loss 输入和诊断指标；状态为 `proposal-for-review`。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 架构](../architecture/V1.1.md)、[V1.1 计划](../plan/V1.1.md) — 作为本提案的上游/历史文档；工作区既有内容保留，未在本活动中重写。

**原因**

将用户确认的结构意图落成可评审的 V1.2 设计：物体 intrinsic 独立直连 object decoder；interaction + hand flow 先以 128 维保留，再经过只对交互物体点生效的 SlotAttn，最后压缩为 32 维 Cm。计划同时冻结 padding KNN、fallback、loss 覆盖范围和公平比较条件，防止实现阶段混入新的研究变量。

**验证**

- 新文档与现有指导、V1.1 计划/架构的相对链接已按 Task 目录结构写入。
- `git diff --check` 和新增文件的空树 diff 检查：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，7 个本地链接可导航。
- 未运行训练、评估、测试或数据处理；本条只记录文档设计，不产生工程 smoke 或科研证据。

**回滚与后续**

- 删除 [V1.2 执行计划草案](../plan/V1.2.md) 和 [V1.2 架构提案](../architecture/V1.2.md) 及本条记录即可回滚本次文档新增，不影响 V1.1、代码、cache、checkpoint 或旧输出。
- 用户确认并将 plan 标记为 `final` 后，才进入阶段 A-D 的 `change`；若需要改动数据语义、公共 `src/base` 或 cache/schema，必须重新确认范围。

## 2026-09-03 16:50:44 +0800 — KNN 半径 2 cm 交互物体点抽样统计

- activity_id: ACT-20260903-165044-OBJECTINTERACTIONCM-RADIUS-2CM-STATS
- timestamp: 2026-09-03 16:50:44 +0800
- modification_version: V1.2.3
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求只读统计 KNN 半径改为 2 cm 时的交互物体点数量；未修改代码、配置、数据或运行产物
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有工作区改动）
- run_id: object_interaction_cm_radius2cm_sample_20260903_165044
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（描述交互点密度，不代表模型效果结论）
- scope: ObjectInteractionCm V1.1 index 的 GRAB/Inspire-F1 train/val/test rows；每个 source/split 抽样最多 2000 帧，固定 stride=2、1024 物体点、base_seed=42、K=8，使用有效 hand 点计算 2 cm 半径

**结果摘要**

- val：GRAB 平均 `338/1024=33.0%` 个物体点参与交互，平均有效邻居 `2.52`；Inspire-F1 平均 `134/1024=13.1%`，平均有效邻居 `0.85`。
- train：GRAB `339/1024=33.1%`，Inspire-F1 `133/1024=13.0%`。
- test：GRAB `353/1024=34.5%`，Inspire-F1 `105/1024=10.3%`。
- val 中每帧至少有一个交互物体点的比例约为 GRAB `94.7%`、Inspire-F1 `82.9%`；因此若 SlotAttn 只收硬 mask 点，Inspire-F1 约 `17%` 抽样帧会触发空池 fallback。
- val 完整 4096 点池的近似交互点数为 GRAB `1338`、Inspire-F1 `540`；1024 点采样后的比例基本一致。

**文件**

- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.2 计划草案](../plan/V1.2.md)、[V1.1 架构](../architecture/V1.1.md)、[V1.2 架构提案](../architecture/V1.2.md) — 统计所依据的 Task 研究指导、计划和架构合同；这些文档中的工作区改动均保留，未因本次统计重写。

**原因**

需要在决定 KNN 半径前量化硬 mask 的稀疏度，特别是确认 Inspire-F1 在 2 cm 下是否会出现大量空 SlotAttn 池，以及 1024 点采样后每帧实际有多少物体点进入交互聚集。

**验证**

- 统计脚本以 `hand_valid_mask` 过滤 padding，并在有效手点上执行 cKDTree `K=8` 查询；没有使用当前实现中可能受 padding 影响的 raw top-k 结果。
- 命令：`PYTHONPATH=/home2/wyy/oyx_ws/Ref2Dex python3 /tmp/oi_radius_stats_fast.py`。
- 结果为抽样估计，不是全量逐帧扫描；抽样帧和统计口径固定，可按同一脚本复核。

## 2026-09-03 17:01:48 +0800 — 重排 V1.2 architecture 为端到端张量流

- activity_id: ACT-20260903-170148-OBJECTINTERACTIONCM-V12-ARCHITECTURE-REORDER
- timestamp: 2026-09-03 17:01:48 +0800
- modification_version: V1.2.4
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户指出原 architecture 缺少先总览后分模块的结构，并要求统一张量维度；本次仅重排和细化 V1.2 architecture，不改代码、配置、数据或 plan
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（architecture 文档重排，不构成工程或科研效果证据）
- scope: V1.2 architecture 的端到端总览、统一符号表、阶段输入/输出表、KNN→intrinsic→interaction→hard mask→SlotAttn→late projection→双 decoder 的顺序和模块张量尺寸

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 改为先给完整数据流和统一尺寸，再按执行顺序介绍数据整理、KNN、intrinsic、edge interaction、fusion、hard mask、SlotAttn、anchor、两个 decoder、loss 和诊断。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.2 计划](../plan/V1.2.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史合同，保留未改。

**原因**

原文虽然包含各模块约束，但总流程、模块顺序和中间 tensor shape 分散在不同章节，无法快速检查 interaction 是否绕过 Cm。重排后统一使用 `N=1024`、`H=3076`、`K=8`、`S=16`、`D=128`、`C=32`，并在总览表中列出每个阶段的输入、输出和消费者。

**验证**

- `git diff --no-index --check /dev/null src/task/ObjectInteractionCm/docs/architecture/V1.2.md`：通过。
- 逐段检查总览图、阶段表和模块公式的维度一致性；确认唯一 `D→C` 投影位于 SlotAttn 输出后，object decoder 只接收 `Ofeat` 和 `Cm`。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，5 个本地链接可导航。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:48:21 +0800 — 按 V1.2 指导收敛双路 additive 与 hand-flow 一级信息

- activity_id: ACT-20260903-174821-OBJECTINTERACTIONCM-V12-FLOW-VALUE-ADDITIVE
- timestamp: 2026-09-03 17:48:21 +0800
- modification_version: V1.2.9
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户要求按 V1.2 指导继续修改；落实真正 additive 聚合、hand-flow 独立 value/statistic、5 cm scaling、sample drop 和采样漏交互诊断；未修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（架构/计划合同更新，不构成工程或科研效果证据）
- scope: V1.2 architecture/plan 的 5 cm 固定半径、raw object decoder、空样本 drop、`Egeo/Eflow` 分离 value、`vbar/sbar` 显式 flow statistic、object/hand additive 求和与 sampling-induced-drop 指标

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 修正为 key 可合并而 value 分路；fusion 输入明确为 388 维；object/hand decoder 均直接求和 slot contribution；补充 5 cm 特征尺度和 sampling-induced-drop 诊断。
- [V1.2 plan](../plan/V1.2.md) — 同步双流 additive、独立 flow value/statistic、raw object decoder、drop_sample 和验证合同。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史文档，保留未改。

**原因**

V1.2 指导指出原设计的 object `softmax(candidate_flow)` 仍是 mixture，且 `Egeo+Eflow→V` 会在局部聚合前丢失 hand-flow 的独立影响。本次将两种 decoder 都改为真正 additive，并让 hand-flow 通过独立 value 和未投影的 `vbar/sbar` 保持一级输入信息。

**验证**

- 逐段检查 `D=128`、`C=32`、fusion `388→128`、object decoder `44→128→128`、hand decoder `166→128→128` 和 `B→B'` 维度。
- 确认唯一 `128→32` projection 仍位于 SlotAttn 后；flow contribution usage 只作 diagnostic，不参与前向求和。
- `git diff --check`、新增/未跟踪文档空树 diff 和 `audit_diff.py --check-links` 在本条 activity 完成后执行。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:47:14 +0800 — 按 V1.2 指导收敛双流 additive 与独立 hand-flow value

- activity_id: ACT-20260903-174714-OBJECTINTERACTIONCM-V12-ADDITIVE-FLOW-DESIGN
- timestamp: 2026-09-03 17:47:14 +0800
- modification_version: V1.2.8
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户要求按 V1.2 指导继续修改；同步落实真正 additive 的 object/hand flow、独立 hand-flow value/statistic、5 cm 和 sample drop 约束；未修改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（架构/计划合同更新，不构成工程或科研效果证据）
- scope: V1.2 architecture/plan 的真实 additive 聚合、object/hand flow 双路 value、显式 flow statistic、5 cm 特征尺度、sample-level drop 与 sampling-induced-drop 诊断

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — object/hand decoder 改为逐 slot contribution 直接求和；local interaction 使用 `Igeo/Iflow/vbar/sbar` 四路 flow-preserving 输出；补充 5 cm 几何/hand-flow scaling 和采样漏交互指标。
- [V1.2 plan](../plan/V1.2.md) — 同步双流 additive、独立 value、显式 flow statistic、drop_sample 和诊断合同。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史文档，保留未改。

**原因**

指导指出原 architecture 的 `softmax(candidate_flow)` 仍是 mixture，并且 `Egeo+Eflow→V` 会在 local aggregation 前丢失 hand-flow 的独立影响。现在将两处都改为真正 additive，同时保留未投影的 flow statistic；这使 hand-flow 仍是 object-side representation 的一级信息，并与旧 Cm 的 additive 语义一致。

**验证**

- 逐段检查 object/hand flow 的 forward 公式均为 `Σ_s contribution_s`，usage 只作范数 diagnostic。
- 检查 local interaction 的 key/value 分离、`[B,N,388]→[B,N,128]` fusion、`D=128→C=32` 唯一投影位置及 5 cm scaling 公式。
- `git diff --check` 和 `audit_diff.py --check-links` 在本条 activity 完成后执行。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:30:04 +0800 — 核对 ObjectInteractionCm 当前 slot flow 聚合方式

- activity_id: ACT-20260903-173004-OBJECTINTERACTIONCM-AGGREGATION-DIAGNOSTIC
- timestamp: 2026-09-03 17:30:04 +0800
- modification_version: V1.2.7
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问当前 object/hand flow 是否为 additive；只读核对实际代码和旧 Cm 聚合分支，未修改实现
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有工作区改动）
- run_status: COMPLETED
- conclusion: INCONCLUSIVE（确认实现语义，不代表模型效果结论）
- scope: 当前 ObjectInteractionCm 的 object/hand decoder slot aggregation，以及旧 Cm `use_additive_slot_contributions` 和 hand-flow decoder 的对照实现

**文件**

- [ObjectInteractionCm decoder](../../decoder.py) — 当前 object/hand decoder 均先生成 per-slot candidate flow，再用 `softmax(dim=slots)` 加权求和。
- [ObjectInteractionCm model](../../model.py) — 当前模型无 `use_additive_slot_contributions` 配置，并将两个 decoder 都按 mixture 路径调用。
- [旧 Cm model](../../../Cm/src/model.py) — 旧 C64 对照运行启用 additive object contributions；hand-flow decoder 在启用时直接对 slot contributions 求和。
- [V1.2 architecture](../architecture/V1.2.md)、[V1.2 plan](../plan/V1.2.md)、[V1.2 指导](../指导/V1.2.md)、[V1.1 architecture](../architecture/V1.1.md)、[V1.1 plan](../plan/V1.1.md) — 当前设计合同、上游指导和历史对照；未因本次只读核对改写。

**原因**

需要区分“当前 OI 已实现的聚合方式”和“V1.2 architecture 计划采用的 additive 方式”，避免把旧 Cm C64 运行配置或设计草案误认为当前 OI 已经具备。

**验证**

- 当前 OI object decoder：`weights=softmax(logits, dim=slots)`，`prediction=Σ weights×candidate_flow`。
- 当前 OI hand decoder：同样使用 `softmax` slot weights 后加权求和。
- 旧 Cm object decoder：`use_additive_slot_contributions=True` 时直接对 slot flow contributions 求和；旧 hand-flow decoder 在启用时也直接求和。
- 只读检查源码，未运行训练、评估、测试或数据处理。

## 2026-09-03 17:23:17 +0800 — V1.2 固定 5 cm、移除 null-Cm 并定义空样本丢弃

- activity_id: ACT-20260903-172317-OBJECTINTERACTIONCM-V12-EMPTY-SAMPLE-POLICY
- timestamp: 2026-09-03 17:23:17 +0800
- modification_version: V1.2.6
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户明确要求按 V1.2 指导继续修改架构，首版使用 5 cm，不使用 null-Cm，采样物体点无交互时跳过该样本训练；同步更新配对 plan 以避免合同不一致
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（架构/计划合同更新，不构成工程或科研效果证据）
- scope: V1.2 的 5 cm KNN、raw object point/normal decoder 输入、`sample_keep`、`B→B'` batch compaction、无交互样本 drop、空 batch optimizer 处理和跳过样本指标

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 固定 `r=0.05 m`；object decoder 使用 raw point/normal + Cm + anchor geometry；移除 null-Cm，定义 `sample_keep` 和有效子 batch `B'`。
- [V1.2 plan](../plan/V1.2.md) — 同步张量表、5 cm 默认、raw decoder 输入、drop_sample、全 batch drop 和验证标准。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史文档，保留未改。

**原因**

2 cm 统计显示 Inspire-F1 的交互点过稀；用户决定首版固定 5 cm，并明确不训练 null-Cm 路径。若 1024 个采样物体点没有任何有效邻居，则该样本从 SlotAttn、decoder、loss 和指标中移除，而不是伪造一个 Cm。

**验证**

- 逐段检查 architecture 总览图、阶段表和模块公式的 batch 维：KNN 前为 `B`，compaction 后统一为 `B'`。
- `git diff --check`、新增/未跟踪文档的空树 diff 检查：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，5 个本地链接可导航。
- 未运行训练、评估、测试或数据处理。

## 2026-09-03 17:21:21 +0800 — V1.2 固定 5 cm 并改为空交互样本丢弃

- activity_id: ACT-20260903-172121-OBJECTINTERACTIONCM-V12-DROP-EMPTY-SAMPLES
- timestamp: 2026-09-03 17:21:21 +0800
- modification_version: V1.2.5
- type: architecture / documentation
- change_level: L0
- approval: user-approved
- approval_basis: 用户明确要求 V1.2 首版继续使用 5 cm KNN、禁止 null-Cm，并在采样物体点没有任何交互点时跳过该样本训练；本次只同步 architecture 和 plan，不改代码、配置、数据或运行产物
- skills_used: research-change-control
- branch: oyx
- base_commit: 52f47ae7bf688e652aef0fc6b49e2ee2eea8d868
- worktree_dirty: true（保留用户已有治理、CmDecoder 及 ObjectInteractionCm V1.1 文档改动）
- run_status: NOT_STARTED
- conclusion: N/A（设计合同更新，不构成工程或科研效果证据）
- scope: V1.2 的 KNN radius、object decoder 原始 geometry 输入、sample-level `B→B'` compaction、空交互样本 drop、训练/评估跳过口径及相关诊断指标

**文件**

- [V1.2 architecture](../architecture/V1.2.md) — 固定 `r=0.05 m`，object decoder 改为 raw point/normal + Cm + anchor geometry，移除 null-Cm，定义 `sample_keep` 和 `B'` 子 batch。
- [V1.2 plan](../plan/V1.2.md) — 同步 object decoder 输入、5 cm 默认值、空样本 drop、全 batch drop、验证和 parity 指标。
- [V1.2 指导](../指导/V1.2.md)、[V1.1 计划](../plan/V1.1.md)、[V1.1 architecture](../architecture/V1.1.md) — 上游和历史合同，保留未改。

**原因**

2 cm 抽样统计显示 Inspire-F1 的交互物体点过稀；用户决定 V1.2 首版固定 5 cm，并明确不训练 null-Cm 路径。若 1024 个采样物体点没有任何有效 interaction，则将该样本从 SlotAttn、decoder、loss 和指标中移除，避免用伪造 Cm 或零流污染训练。

**验证**

- 逐段检查 architecture 的总览图、阶段表、`B'` 维度和 raw object decoder 的 44 维输入是否一致。
- `git diff --check`、新增 architecture 的空树 diff 检查：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm --check-links`：通过；activity 与 5 个 scope 变更路径一致，5 个本地链接可导航。
- 未运行训练、评估、测试或数据处理。

## 2026-09-14 04:22:21 +0000 — 扩大 OI-Cm 数据与 Dexplore 导出入口只读盘点

- activity_id: ACT-20260914-042221-OICM-DATA-SCOPE-DIAG
- timestamp: 2026-09-14 04:22:21 +0000
- modification_version: V1.3.2
- type: diagnostic
- operation_category: [diagnostic, documentation]
- change_level: L0
- approval: auto
- approval_basis: 用户要求浏览仓库并确认扩大 OI-Cm 数据、MANO/Inspire 混合训练及 Dexplore 重导出是否存在歧义；本次仅只读扫描
- skills_used: research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留开始前已有 activity_log 改动）
- scope: ObjectInteractionCm V1.3 指导/计划/配置、Dexplore 导出仓库、NAS 原始数据入口和现有产物可见性；不修改代码、配置、数据、cache、split、checkpoint 或运行状态
- conclusion: SUPPORTED（当前入口与扩容方向可定位）；INCONCLUSIVE（目标数据集合、Inspire 几何/RL 选择及是否纳入 ARCTIC 尚未确认）

**文件**
- [`../指导/V1.3.md`](../指导/V1.3.md) — 核对当前 630 sequence、159476 frame、右手、30 Hz、4096/1024、KNN=32/2 cm 合同。
- [`../plan/V1.3.md`](../plan/V1.3.md) — 核对现有 cache、split、source-specific stream 与全量构建顺序。
- [`../../configs/active/dexplore_rl_v1_3.yaml`](../../configs/active/dexplore_rl_v1_3.yaml) — 核对 mixed source、0.5/0.5 source probability、训练预算和 index 路径。
- [`../../../../../../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md`](../../../../../../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md) — 核对 GRAB→Inspire geometric/RL 导出说明；该仓库不含生成数据目录。
- [`../../../../../../dexplore/data_processing/convert_grab.py`](../../../../../../dexplore/data_processing/convert_grab.py) — 核对 GRAB 转换器输入/输出及重定向参数。

**原因**
用户希望在保持 OI-Cm 其他配置不变的前提下扩大 MANO 数据，并将对应动作几何重定向到 Inspire 后混合训练；必须先区分新增独立序列、现有序列重导出、Inspire 几何轨迹和 RL 实际轨迹四种输入。

**验证**
- `data/raw_data/GRAB/grab` 可读，包含 10 个 subject、1335 个 `.npz` 原始序列；MANO 模型与 GRAB subject meshes 可见。
- `data/raw_data/ARCTIC/arctic_data/data` 可见 ARCTIC 原始目录（1206 个 `.npy` 文件），但当前 OI-Cm V1.3 指导未允许直接加入 ARCTIC。
- 当前机器不存在 `/home2/wyy/oyx_ws/Ref2Dex`、`/home2/wyy/oyx_ws/dexplore`、`/home2/wyy/oyx_ws/InterAct`；`/home/wbcd/workspace/oyx_ws/dexplore` 只有代码/资产/checkpoint，无生成 `data/` 轨迹，另一份 `/home/wbcd/workspace/dex/retarget/dexplore` 仅有单条 pilot 轨迹。
- `data/processed_data` 是 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data` 软链接；该 NAS 当前可见 stage/cache 目录，但没有可直接定位的 OI-Cm V1.3 index 或 Dexplore geometric/RL 轨迹目录。
- Dexplore README 说明完整 GRAB 导出需要 InterAct canonical cache、SMPL-X/机器人资产和 `inspire.pth`；RL 导出会替换 Inspire DOF，并使用实际仿真物体状态。README 中的默认路径仍是另一台机器的绝对路径。
- 未运行导出、cache 构建、训练或评估；上述检查属于工程入口盘点，不构成科研效果结论。

**待用户确认**
- 扩容集合是否为全部 1335 条 GRAB，还是沿用 Dexplore 过滤后的 660 条，或只追加现有 630 条之外的序列；是否保留右手/排除左手接触与 `doorknob` 过滤。
- Inspire 侧使用 geometric retarget 轨迹，还是必须重新跑 `inspire.pth` 得到实际 RL object trajectory；两者不能在同一 source 名下混用。
- 第一阶段是否完全不纳入 ARCTIC；若纳入，需要新数据合同、articulated-object cache、split 和 plan。
- “其他配置不变”是否也冻结当前 0.5/0.5 source probability、stride、split 规则、202300 max steps/120 epochs；扩容后是否允许按新增样本量增加训练 steps。
- 现有 InterAct canonical/Dexplore geometric/RL 产物在 NAS 上的确切路径，或是否授权从 NAS 原始 GRAB 重新导出。

**回滚与规范反馈**

- 仅移除本 activity 条目即可回滚本次诊断记录；没有生成新的代码、配置、数据或运行产物。
- 本次未遇到目录、版本或审批阻碍；后续若改变数据集合、cache/schema、split、GT/坐标或启动长时导出/训练，需先形成对应 final plan 并取得用户确认。

## 2026-09-14 04:25:00 +0000 — InterAct 官方代码浅克隆准备

- activity_id: ACT-20260914-042500-INTERACT-CLONE
- timestamp: 2026-09-14 04:25:00 +0000
- modification_version: V1.3.2
- type: operation / diagnostic
- operation_category: [operation, diagnostic]
- change_level: L3（外部网络与本地依赖准备）
- approval: user-approved
- approval_basis: 用户明确询问并授权尝试从 GitHub 克隆 InterAct 以支持重新导出。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留开始前已有 activity_log 改动）
- run_id: interact_clone_20260914_042500
- run_status: COMPLETED
- scope: 将官方 `wzyabcas/InterAct` 仓库以 `--depth 1` 克隆到 `/home/wbcd/workspace/oyx_ws/InterAct`；不下载数据、不执行处理脚本。
- conclusion: SUPPORTED（代码入口准备完成）；不构成数据完整性或科研效果结论。

**原因**
用户确认需要重新导出扩容数据；InterAct 是当前 GRAB/ARCTIC canonical 处理链的上游代码依赖，因此先准备本地代码副本，后续再按定稿 plan 接入数据。

**文件**
- [`../../../../../../InterAct/README.md`](../../../../../../InterAct/README.md) — 官方处理说明与 GRAB/ARCTIC 目录合同。
- [`../../../../../../InterAct/process/process_grab.py`](../../../../../../InterAct/process/process_grab.py) — GRAB 处理入口。
- [`../../../../../../InterAct/process/process_arctic.py`](../../../../../../InterAct/process/process_arctic.py) — ARCTIC 处理入口。

**验证**
- 官方仓库地址：[wzyabcas/InterAct](https://github.com/wzyabcas/InterAct)。
- 克隆提交：`96180a34f7b516e7f3520b853c19ea8679b8204f`；工作树 clean。
- 首次经过失效本地代理 `127.0.0.1:7897` 的尝试失败；随后清除该代理变量后直连成功。
- 未下载 GRAB/ARCTIC 数据、未运行 InterAct 处理、未修改 Ref2Dex 代码或数据。

**回滚与规范反馈**

- 回滚入口：删除 `/home/wbcd/workspace/oyx_ws/InterAct` 目录即可；没有触碰 Ref2Dex 现有产物。
- 本次未遇到需要修改治理规则的阻碍；后续数据处理仍需扩容 plan 定稿和运行记录。

## 2026-09-14 04:30:00 +0000 — 扩容目标确认与 V1.4 计划入口

- activity_id: ACT-20260914-043000-OICM-EXPANSION-SCOPE-CONFIRMED
- timestamp: 2026-09-14 04:30:00 +0000
- modification_version: V1.3.2（新数据合同拟升级为下一版本，尚未定稿）
- type: diagnostic / documentation
- operation_category: [diagnostic, documentation]
- change_level: L2/L3（数据合同、ARCTIC 接入、cache 重建与长时导出）
- approval: user-approved（目标范围）；pending（具体手侧重定向和 ARCTIC Inspire 路线）
- approval_basis: 用户确认完整 GRAB、加入 ARCTIC、保留现有 val/test、只用 geometric Inspire、按扩容比例增加 max steps 并保持 epoch 总数。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有 activity_log 改动）
- scope: 记录用户确认的扩容方向；不修改指导、plan、代码、cache、split、配置或训练运行。
- conclusion: INCONCLUSIVE（关键数据语义仍需定稿）

**原因**
完整 GRAB + ARCTIC、左右手点合并、Inspire geometric-only 和 max-step 重算会改变 V1.3 的数据合同，必须在新 plan 中明确而不能静默覆盖 V1.3。

**文件**
- [`../指导/V1.3.md`](../指导/V1.3.md) — 当前合同，作为不可静默改写的基线。
- [`../plan/V1.3.md`](../plan/V1.3.md) — 当前 cache/训练计划，后续需建立新版本 plan。
- [`../../../../../../InterAct/README.md`](../../../../../../InterAct/README.md) — 已克隆的官方上游处理说明。

**验证**
- 用户确认：完整 GRAB；不区分左右手点；val/test 保持；仅 geometric Inspire；加入 ARCTIC；max steps 按扩容调整且 epoch 总数保持。
- 尚未确认：双手/左手如何映射到单只 Inspire；ARCTIC 是否也必须生成 Inspire geometric 对；“val/test 不变”是保持成员与帧索引，还是保留旧 cache 字节与旧 Inspire-RL 结果。
- 未运行数据导出、cache 构建、训练或评估。

**回滚与规范反馈**

- 仅移除本 activity 条目即可回滚；没有新增代码或数据产物。
- 后续 L2/L3 数据合同和长任务需新 guidance/plan 配对并在定稿后执行。

## 2026-09-14 04:35:00 +0000 — TopoRetarget 候选代码盘点

- activity_id: ACT-20260914-043500-TOPORETARGET-REVIEW
- timestamp: 2026-09-14 04:35:00 +0000
- modification_version: V1.3.2
- type: operation / diagnostic
- operation_category: [operation, diagnostic]
- change_level: L3（外部代码获取与候选重定向评估）
- approval: user-approved
- approval_basis: 用户要求检查 TopoRetarget，并在本机缺失时从 GitHub 获取。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有 activity_log 改动）
- run_id: toporetarget_clone_review_20260914_043500
- run_status: COMPLETED
- scope: 将官方公开的 `0iui0/toporetarget` 以 `--depth 1` 克隆到 `/home/wbcd/workspace/oyx_ws/toporetarget` 并只读检查其模型/输入合同；不修改 Ref2Dex，不运行数据处理。
- conclusion: SUPPORTED（候选代码可获取）；INCONCLUSIVE（不能直接替代当前 Inspire 双手重定向）。

**原因**
用户要求双手 GRAB/ARCTIC 都重定向到 Inspire；TopoRetarget 可能提供 interaction-preserving 候选，但需先核对机器人支持和输入数据合同。

**文件**
- [`../../../../../../toporetarget/README.md`](../../../../../../toporetarget/README.md) — 候选算法说明和支持列表。
- [`../../../../../../toporetarget/toporetarget/core/retarget.py`](../../../../../../toporetarget/toporetarget/core/retarget.py) — MediaPipe 21 点与 object pose 输入合同。
- [`../../../../../../toporetarget/toporetarget/core/robot.py`](../../../../../../toporetarget/toporetarget/core/robot.py) — 当前机器人模型实现。

**验证**
- 官方仓库：[0iui0/toporetarget](https://github.com/0iui0/toporetarget)，提交 `8c7fbcbb946a00cae92de803e731b7c5dacba3d7`，工作树 clean。
- 当前公开支持 Revo3、Wuji、Leap、Simplified，不包含 Inspire；输入为 MediaPipe 21 keypoints + object pose，不是 GRAB/ARCTIC 的 MANO/SMPL-X 轨迹。
- 因此若采用 TopoRetarget，至少需要 Inspire robot model、MANO/SMPL-X 到 21 点转换、双手调用与 ARCTIC 铰接物体适配；不能直接把其输出当作现有 Dexplore Inspire geometric 数据。
- 未运行 TopoRetarget demo、未安装依赖、未改代码或数据。

**回滚与规范反馈**

- 回滚入口：删除 `/home/wbcd/workspace/oyx_ws/toporetarget` 即可；Ref2Dex 无代码/数据改动。
- 本次未遇到治理阻碍；是否把 TopoRetarget 纳入正式导出必须在新 plan 中作为候选路线冻结。

## 2026-09-14 04:40:00 +0000 — V1.4 双手 GRAB/ARCTIC geometric 扩容计划定稿

- activity_id: ACT-20260914-044000-OICM-V14-PLAN-FINAL
- timestamp: 2026-09-14 04:40:00 +0000
- modification_version: V1.4
- type: documentation / config_change
- operation_category: [documentation, code, data, experiment, operation]
- change_level: L2/L3
- approval: user-approved
- approval_basis: 用户连续确认完整 GRAB、ARCTIC、双手分别重定向到 Inspire、只使用 geometric、保留 val/test、按扩容比例调整 max steps 并保持 epoch 总数；随后确认继续执行。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 新建 V1.4 guidance/plan 和 geometric 双手训练配置；未修改 V1.3 代码/cache/checkpoint/output，未启动导出或训练。
- conclusion: N/A（计划与配置入口，不构成科研效果结论）

**文件**
- [`../指导/V1.4.md`](../指导/V1.4.md) — 固化完整 GRAB/ARCTIC、双手合并、双 Inspire geometric 和 val/test 保护边界。
- [`../plan/V1.4.md`](../plan/V1.4.md) — 定稿执行范围、cache/schema、验证顺序、预算公式和回滚方式。
- [`../../configs/active/grab_arctic_inspire_geometric_v1_4.yaml`](../../configs/active/grab_arctic_inspire_geometric_v1_4.yaml) — 新正式配置入口。
- [`../../configs/active/grab_arctic_inspire_geometric_v1_4_smoke.yaml`](../../configs/active/grab_arctic_inspire_geometric_v1_4_smoke.yaml) — 新 smoke 配置入口。
- [`../../../../../docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针更新为 V1.4。

**原因**
用户确认的数据集合和双手重定向语义改变了 V1.3 的 hand/cache 合同，因此建立独立 V1.4 版本，避免覆盖旧实验并为后续导出、cache finalize 和训练提供唯一入口。

**验证**
- `git diff --check`：计划、指导和配置无空白错误。
- `audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links`：通过，最新条目 5 个本地链接可导航。
- 未执行数据导出、cache 构建、模型 smoke 或正式训练；`max_steps` 待 cache finalize 后按实际 train rows 更新。

**回滚与规范反馈**

- 删除 V1.4 guidance/plan/config 和 current_versions 指针增量即可回滚；V1.3 及更早产物不受影响。
- 本轮未遇到治理阻碍；后续实现若发现 ARCTIC/MANO 双手无法满足同一坐标或 point correspondence 合同，必须暂停并回到计划协商。

## 2026-09-14 04:45:00 +0000 — V1.4 双手无 side-ID 合并工具与合同测试

- activity_id: ACT-20260914-044500-OICM-V14-BILATERAL-MERGE
- timestamp: 2026-09-14 04:45:00 +0000
- modification_version: V1.4
- type: code_change / diagnostic
- operation_category: [code, diagnostic]
- change_level: L1（Task-local helper，保持模型外部合同）
- approval: user-approved
- approval_basis: V1.4 计划已定稿；用户确认左右手都重定向并按无 side-ID 合并。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 新增双手点/法向/future/flow 合并 helper 与 Task-local 合同测试；未改 loader/model/cache producer，未启动数据导出。
- conclusion: SUPPORTED（合并 helper 工程合同）；不构成科研效果结论。

**文件**
- [`../../tools/data/merge_dual_hand_stream.py`](../../tools/data/merge_dual_hand_stream.py) — 固定 left→right 拼接、严格帧/shape/finite 检查；明确 KNN 必须在合并后重新计算。
- [`../../tests/test_merge_dual_hand_stream.py`](../../tests/test_merge_dual_hand_stream.py) — 双手顺序、flow、帧不一致和 non-finite 测试。

**原因**
V1.4 需要把 MANO 和 Inspire 的左右手统一成单一无 side-ID stream；若直接偏移并拼接左右旧 KNN index，会漏掉跨手近邻，因此先建立独立合并合同。

**验证**
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py`：`2 passed`。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/merge_dual_hand_stream.py`：通过。
- 未运行正式导出、cache、训练或评估。

**回滚与规范反馈**

- 删除上述 helper/test 文件即可回滚；V1.3 及现有产物不受影响。
- 暂无规范反馈。

## 2026-09-14 08:35:00 +0000 — V1.4 导出器 CLI 导入兼容修正

- activity_id: ACT-20260914-083500-OICM-V14-EXPORT-CLI
- timestamp: 2026-09-14 08:35:00 +0000
- modification_version: V1.4
- type: code_change / diagnostic
- operation_category: [code, diagnostic]
- change_level: L1
- approval: user-approved
- approval_basis: V1.4 计划已定稿。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 允许双手导出器既以 package module 又以文件 CLI 运行；未改变数据语义或输出合同。
- conclusion: SUPPORTED（静态导入/编译合同）；不构成科研效果结论。

**文件**
- [`../../tools/data/export_bilateral_geometry.py`](../../tools/data/export_bilateral_geometry.py) — 增加直接脚本调用的 import fallback。

**原因**

后续 NAS 批处理可能直接调用脚本路径，导出器需要兼容该入口。

**验证**
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py`：`4 passed`。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/export_bilateral_geometry.py`：通过。

**回滚与规范反馈**

- 删除该 fallback 增量即可回滚；暂无规范反馈。

## 2026-09-14 08:33:17 +0000 — V1.4 geometric 双手导出适配器与 max_steps 计算器

- activity_id: ACT-20260914-083317-OICM-V14-EXPORT-ADAPTER
- timestamp: 2026-09-14 08:33:17 +0000
- modification_version: V1.4
- type: code_change / diagnostic
- operation_category: [code, data, diagnostic]
- change_level: L1（Task-local 可回滚适配器和纯函数测试）
- approval: user-approved
- approval_basis: V1.4 计划已定稿；用户确认双手 geometric Inspire、完整 GRAB/ARCTIC 扩容方向。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 新增通用双手 geometry 导出适配器（复制 shared、left→right 合并点/法向、union candidate mask、拒绝覆盖既有输出）和按 epoch/实际 train rows/global batch 计算绝对 max_steps 的工具；不修改 V1.3，不复制旧 KNN，不启动长时导出或训练。
- conclusion: SUPPORTED（工具合同和单元测试）；不构成科研效果结论。

**文件**
- [`../../tools/data/export_bilateral_geometry.py`](../../tools/data/export_bilateral_geometry.py) — 从 retargeter 产出的 left/right sequence 生成 V1.4 side-free geometry 目录，并要求合并后重算 KNN。
- [`../../tools/data/compute_v1_4_max_steps.py`](../../tools/data/compute_v1_4_max_steps.py) — `ceil(epochs * train_rows / global_batch)` 计算入口。
- [`../../tests/test_v1_4_data_contract.py`](../../tests/test_v1_4_data_contract.py) — 导出顺序、candidate union、旧 KNN 不复制和 max_steps 测试。

**原因**

V1.4 需要将任一 retargeter 的双手输出统一到单一无 side-ID stream，并在最终 train rows 确定后按 epoch 语义更新绝对步数；这两个适配点可以先独立验证，避免在数据尚未导出时修改旧 cache 或凭估计填写预算。

**验证**
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py`：`4 passed`。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/export_bilateral_geometry.py src/task/ObjectInteractionCm/tools/data/compute_v1_4_max_steps.py`：通过。
- 未运行正式导出、cache finalize、训练或评估；`max_steps` 仍待最终 cache train rows 解析后写回正式配置。

**回滚与规范反馈**

- 删除上述三个新增文件即可回滚；V1.3 及现有产物不受影响。
- 暂无规范反馈。

## 2026-09-14 08:50:00 +0000 — V1.4 上游双手输入与 ARCTIC raw pilot

- activity_id: ACT-20260914-085000-OICM-UPSTREAM-PILOT
- timestamp: 2026-09-14 08:50:00 +0000
- modification_version: V1.4
- type: diagnostic / operation
- operation_category: [diagnostic, operation, data]
- change_level: L2（只读输入盘点与单序列 CPU pilot）
- approval: user-approved
- approval_basis: 用户确认继续执行 V1.4 扩容和重导方向。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 只读检查 NAS 原始输入和模型资产；验证完整 GRAB 双手参数帧一致性、ARCTIC 双手 raw adapter 单序列输出；未写入 cache、未改 NAS、未启动全量导出。
- conclusion: SUPPORTED（输入完整性与 ARCTIC raw geometry pilot）；INCONCLUSIVE（GRAB geometric retarget 尚未运行）。

**文件**
- [`../../../../../data/raw_data/GRAB`](../../../../../data/raw_data/GRAB) — 1335 条 `.npz` 原始序列入口。
- [`../../../../../data/raw_data/ARCTIC`](../../../../../data/raw_data/ARCTIC) — 301 组同时含 `.mano/.object/.smplx` 的序列入口。
- [`../../../../../process/ARCTIC/raw.py`](../../../../../process/ARCTIC/raw.py) — 双手 MANO 与铰接物体 raw adapter。
- [`../../../../../../InterAct/README.md`](../../../../../../InterAct/README.md) — 上游 GRAB/ARCTIC 处理入口说明。

**原因**

重导必须先确认两侧参数和帧轴完整，且 ARCTIC 的 object articulation 字段能够在当前机器上被解析；这一步避免在上游不完整时生成不可审计的 V1.4 cache。

**验证**
- GRAB 扫描：`1335` 条序列，`missing=0`，`frame_mismatch=0`；每条均有 `lhand/rhand/object` 参数。
- ARCTIC CPU pilot：`REF2DEX_ARCTIC_ROOT=/mnt/ugreen_nas/storage/Ref2Dex_storage/arctic`，`ArcticRawAdapter(num_obj_points=64,max_frames=2)`；输出左右手均为 `[2,1538,3]`，`obj_articulation` 为 `[2,1]`，`obj_point_id` 为 `[64]`。
- NAS 已有 `MANO_LEFT/RIGHT.pkl`、`SMPLX_{MALE,FEMALE,NEUTRAL}.npz/.pkl` 和 Inspire 左右 URDF；InterAct `process/process_grab.py --help` 当前被其仓库内缺少顶层 `render` import 阻塞，尚未运行 GRAB canonical processing。
- 未执行正式导出、cache finalize、训练或评估。

**回滚与规范反馈**

- 本次仅产生诊断记录，无数据产物可回滚；暂无规范反馈。

## 2026-09-14 09:05:00 +0000 — V1.4 InterAct GRAB 入口依赖复核

- activity_id: ACT-20260914-090500-OICM-INTERACT-DEPS
- timestamp: 2026-09-14 09:05:00 +0000
- modification_version: V1.4
- type: diagnostic / operation
- operation_category: [diagnostic, operation]
- change_level: L2（只读依赖检查）
- approval: user-approved
- approval_basis: 用户确认继续尝试上游重导。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 在临时环境变量和未修改外部仓库的前提下检查 InterAct GRAB processing 入口；没有创建 symlink、安装依赖或写入外部 checkout。
- conclusion: INCONCLUSIVE（处理入口代码可见，但当前环境缺少 `pyrender` 依赖；不能据此生成 canonical 输出）。

**文件**
- [`../../../../../../InterAct/process/process_grab.py`](../../../../../../InterAct/process/process_grab.py) — GRAB 处理脚本，使用 `render.mesh_utils`。
- [`../../../../../../InterAct/text2interaction/render/mesh_utils.py`](../../../../../../InterAct/text2interaction/render/mesh_utils.py) — 当前导入链依赖 `pyrender`。
- [`../../../../../../InterAct/README.md`](../../../../../../InterAct/README.md) — 模型目录和处理命令合同。

**原因**

InterAct 脚本在模块导入阶段就加载渲染工具；先确认缺口可以避免全量处理半途失败，也避免未经用户确认改变 Python 环境。

**验证**
- `cd /home/wbcd/workspace/oyx_ws/InterAct && PYTHONPATH=.:text2interaction /home/wbcd/miniconda3/envs/ref2dex-grab/bin/python process/process_grab.py --help`：失败于 `ModuleNotFoundError: pyrender`，未进入数据写入阶段。
- 现有 NAS 模型资产已确认位于 `/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/{smplx,mano}`；完整 GRAB/ARCTIC 输入和 ARCTIC CPU pilot 见上一条 activity。
- 未安装依赖、未改 InterAct、未运行全量处理、cache 或训练。

**回滚与规范反馈**

- 本次无文件/数据产物可回滚；暂无规范反馈。后续若安装 `pyrender` 或建立 InterAct 临时 staging，应单独记录环境与命令。

## 2026-09-14 09:20:00 +0000 — V1.4 InterAct GRAB 单序列 staging smoke

- activity_id: ACT-20260914-092000-OICM-INTERACT-GRAB-SMOKE
- timestamp: 2026-09-14 09:20:00 +0000
- modification_version: V1.4
- type: operation / diagnostic
- operation_category: [operation, diagnostic, data]
- change_level: L2（临时 staging，只读输入，输出位于 `/tmp`）
- approval: user-approved
- approval_basis: 用户确认继续尝试数据重导。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 建立临时 `/tmp/interact_grab_smoke.jQqZ1o` staging，将一条 GRAB 原始序列、GRAB subject mesh、物体 mesh 和 NAS SMPL-X 模型映射到 InterAct 约定路径；运行单序列 `process_grab.py`，未写入 Ref2Dex 或 NAS。
- conclusion: ENGINEERING_PASS（InterAct 单序列 canonical processing 可运行）；不构成双手 Inspire retarget 或科研效果结论。

**文件**
- [`../../../../../../InterAct/process/process_grab.py`](../../../../../../InterAct/process/process_grab.py) — 实际运行入口。
- [`../../../../../data/raw_data/GRAB/grab/s7/cubemedium_inspect_1.npz`](../../../../../data/raw_data/GRAB/grab/s7/cubemedium_inspect_1.npz) — smoke 输入。

**原因**

上一条诊断发现 InterAct 需要 `pyrender`；本机 `ref2dex-grab` 环境已具备该依赖，因此用隔离 staging 验证真正的 GRAB 处理链，避免先改动正式数据目录。

**验证**
- 命令返回码 `0`，日志输出 `Saved s7_cubemedium_inspect_1`。
- 输出 `/tmp/interact_grab_smoke.jQqZ1o/data/grab/sequences/s7_cubemedium_inspect_1/{human,object}.npz`；`human.npz` 含 `poses [238,114]`、`trans [238,3]`、`vtemp [10475,3]`，`object.npz` 已生成。
- 未运行全量 1335 条、双手 Inspire geometric retarget、V1.4 cache 或训练。

**回滚与规范反馈**

- 临时目录可直接清理；本次无仓库/NAS 产物可回滚。暂无规范反馈。

## 2026-09-14 09:13:27 +0000 — Dexplore geometric retargeting 手型与目标合同核查

- activity_id: ACT-20260914-091327-OICM-DEXPLORE-GEOMETRIC-CONTRACT
- timestamp: 2026-09-14 09:13:27 +0000
- modification_version: V1.4
- type: diagnostic
- operation_category: [diagnostic]
- change_level: L1（只读代码核查）
- approval: user-approved
- approval_basis: 用户询问 Dexplore geometric retargeting 的手型限制和基础。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 只读检查 Dexplore `convert_grab.py`、robot config、dex-retargeting Inspire offline config 和 PositionOptimizer；未修改外部仓库或 Ref2Dex。
- conclusion: SUPPORTED（实现合同已确认）；不构成科研效果结论。

**文件**
- [`../../../../../../dexplore/data_processing/convert_grab.py`](../../../../../../dexplore/data_processing/convert_grab.py) — SMPL-X forward、21 点输入抽取、position retarget 调用。
- [`../../../../../../dexplore/data_processing/robot_configs.py`](../../../../../../dexplore/data_processing/robot_configs.py) — Inspire 18 DOF 和关节重排。
- [`../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/configs/offline/inspire_hand_right.yml`](../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/configs/offline/inspire_hand_right.yml) — Inspire right-hand position 配置。
- [`../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/configs/offline/inspire_hand_left.yml`](../../../../../../dex/retargeting/dex-retargeting/dex_retargeting/configs/offline/inspire_hand_left.yml) — 已存在的 left-hand 对称配置。
- [`../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/optimizer.py`](../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/optimizer.py) — 位置误差目标和正则项。

**原因**

需要区分“输入人的手型是否固定”和“输出机器人手型是否固定”：前者影响 SMPL-X 生成的关键点，后者由 Inspire URDF/DOF 合同决定。

**验证**
- 当前 Dexplore geometric exporter 的 `setup_retargeting` 固定使用 `HandType.right` 和 `inspire_hand_right.urdf`；这是现有脚本实现限制，不是 dex-retargeting 算法只能支持右手。依赖包同时提供 left 配置和 left URDF。
- 输入不是完整手部 mesh 拟合，而是 SMPL-X forward 后的 21 个点：手腕、15 个手指关节和 5 个指尖顶点；PositionOptimizer 用 5 个 Inspire tip link 的三维位置做 Huber 位置误差优化，并带轻微关节变化正则，不匹配法向、完整 mesh 或手指姿态矩阵。
- 输出是固定 Inspire 机器人形态：18 个 native DOF（6 个 floating wrist + 12 个手指相关 DOF，含 URDF mimic/重排处理）。不同人的 SMPL-X/MANO shape 会改变输入点位置，但不会改变 Inspire 的 link 长度、关节数量或目标手型。
- `convert_grab.py` 当前实际只构造并 retarget 右手 landmark stream；双手需要分别准备左/右 landmark、选择对应 URDF/config，并分别输出 qpos 后再合并几何点。

**回滚与规范反馈**

- 本次无代码/数据产物可回滚；暂无规范反馈。

## 2026-09-14 09:21:42 +0000 — ARCTIC→Inspire 直接重定向可行性核查

- activity_id: ACT-20260914-092142-OICM-ARCTIC-INSPIRE-FEASIBILITY
- timestamp: 2026-09-14 09:21:42 +0000
- modification_version: V1.4
- type: diagnostic
- operation_category: [diagnostic]
- change_level: L1（只读合同核查）
- approval: user-approved
- approval_basis: 用户询问 ARCTIC 是否可直接重定向到 Inspire。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 对照 ARCTIC raw adapter 与 Dexplore geometric 输入合同；未修改代码、数据或外部仓库。
- conclusion: SUPPORTED（算法路线可行）；INCONCLUSIVE（现有 Dexplore GRAB wrapper 不能无修改直接消费 ARCTIC）。

**文件**
- [`../../../../../process/ARCTIC/raw.py`](../../../../../process/ARCTIC/raw.py) — ARCTIC 左右 MANO、1538 face-center 和 articulation 输出。
- [`../../../../../../dexplore/data_processing/convert_grab.py`](../../../../../../dexplore/data_processing/convert_grab.py) — 当前仅右手、GRAB/SMPL-X 专用输入路径。
- [`../../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/configs/offline/inspire_hand_left.yml`](../../../../../../../dex/retarget/third_party/dex-retargeting/dex_retargeting/configs/offline/inspire_hand_left.yml) — 可复用的左 Inspire position 配置。

**原因**

ARCTIC 已有完整 MANO 参数和双手几何，理论上满足位置重定向；需要明确哪些是格式适配和坐标校准问题，哪些才是算法限制。

**验证**
- ARCTIC raw adapter 已输出左右手 MANO pose/betas、wrist/joint 几何、物体 articulation 和稳定 point ID。
- 现有 Dexplore converter 的 21 点输入和 `rotation_x_90` 处理是 GRAB 专用；ARCTIC 不能直接套用该坐标变换或其 SMPL-X 顶点索引。
- dex-retargeting 同时提供 Inspire left/right position 配置，因此双手分别优化在算法层面可行；需要新增 ARCTIC landmark adapter、左右调用和 merged FK geometry。

**回滚与规范反馈**

- 本次无代码/数据产物可回滚；暂无规范反馈。

## 2026-09-14 09:50:23 +0000 — ARCTIC 双手 Inspire geometric 实际 pilot

- activity_id: ACT-20260914-095023-OICM-ARCTIC-INSPIRE-PILOT
- timestamp: 2026-09-14 09:50:23 +0000
- modification_version: V1.4
- type: code_change / operation / diagnostic
- operation_category: [code, operation, diagnostic]
- change_level: L1（Task-local pilot 工具和小规模验证）
- approval: user-approved
- approval_basis: 用户确认继续 ARCTIC→Inspire pilot。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留既有用户/activity 改动）
- scope: 新增 ARCTIC MANO→21 landmark→Inspire 左右 position retargeting pilot；使用 4 帧 `s07/scissors_use_01`，未写入 cache、未修改 NAS 或 Dexplore。
- conclusion: ENGINEERING_PASS（左右均生成 finite `[4,18]` qpos）；INCONCLUSIVE（坐标约定尚未冻结，未进入全量导出）。

**文件**
- [`../../tools/data/pilot_arctic_inspire_geometric.py`](../../tools/data/pilot_arctic_inspire_geometric.py) — ARCTIC MANO landmark、左右 Inspire retarget 和 tip error 统计。
- [`../../tests/test_arctic_landmark_contract.py`](../../tests/test_arctic_landmark_contract.py) — 21 点和五指尖索引合同测试。

**原因**

需要实测验证 ARCTIC 参数能否进入 dex-retargeting，而不是只依赖格式推断；同时检查左右 URDF 的坐标约定差异。

**验证**
- ARCTIC MANO extraction：左右均输出 `[4,21,3]`，使用官方 MANO tip vertex IDs `[744,320,443,554,671]`。
- dex-retargeting 左右配置均成功构建，输出均为 finite `[4,18]` qpos。
- 未变换坐标时：left 平均 tip error `11.44 mm`，right `65.20 mm`；right landmark 采用候选 `Rx(180°)` 约定时平均误差约 `14.42 mm`。该旋转只作为 pilot 诊断参数，尚未写入正式 schema。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_arctic_landmark_contract.py src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py`：`3 passed`。
- 未运行全量 retarget、cache finalize、训练或评估。

**回滚与规范反馈**

- 删除 pilot 工具和测试即可回滚；暂无规范反馈。

## 2026-09-14 12:01:25 +0000 — OakInk2 三序列抽样试转：FAILED

- activity_id: ACT-20260914-120125-OICM-OAKINK2-FAILED
- timestamp: 2026-09-14 12:01:25 +0000
- modification_version: V1.4.1
- type: operation / diagnostic
- operation_category: [operation, diagnostic]
- primary_task_mode: run-only/operation
- change_level: L2（用户授权的独立双手几何诊断）
- approval: user-approved
- approval_basis: 用户要求使用 graspenv 安装依赖并尝试 OakInk2 转换，随后多次确认继续。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 三条序列各均匀采样 64 个原始 mocap 帧；独立 output，不改变正式 cache/split。
- run_id: oakink2_inspire_20260914T120600Z
- run_status: FAILED
- conclusion: INVALID_IMPLEMENTATION
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/run.py --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --mano-root /mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano --dex-root /home/wbcd/workspace/dex/retarget/third_party/dex-retargeting --frames 64 --output src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z`
- output: [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z)

**文件**
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z/run_manifest.json](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z/report.json](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z/report.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z/run.log](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T120600Z/run.log)

**原因**

验证官方 MANO 重建与左右 Inspire FK 的端到端诊断入口。

**验证**

首轮在误差统计处触发 NumPy advanced indexing 轴顺序错误：`(5,64,3)` 与 `(64,5,3)` 不匹配，三条序列均显式记录失败。未生成有效 NPZ；随后修正为先取位置再选择 link 轴，原失败 run 不覆盖。run_id 的时间标签不是事件时间，准确起止时间以 manifest 字段为准。

**回滚与规范反馈**

仅本次独立输出目录；失败产物保留审计。没有训练 step/epoch、checkpoint 或 metrics.jsonl。无规范阻碍。

## 2026-09-14 12:02:05 +0000 — OakInk2 三序列抽样试转：COMPLETED

- activity_id: ACT-20260914-120205-OICM-OAKINK2-COMPLETED
- timestamp: 2026-09-14 12:02:05 +0000
- modification_version: V1.4.1
- type: operation / diagnostic
- operation_category: [operation, diagnostic]
- primary_task_mode: run-only/operation
- change_level: L2（用户授权的独立双手几何诊断）
- approval: user-approved
- approval_basis: 用户要求使用 graspenv 安装依赖并尝试 OakInk2 转换，随后多次确认继续。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 三条序列各均匀采样 64 个原始 mocap 帧；独立 output，不改变正式 cache/split。
- run_id: oakink2_inspire_20260914T121000Z
- run_status: COMPLETED
- conclusion: SUPPORTED（工程可运行）；INCONCLUSIVE（接触/科研效果）
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/run.py --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --mano-root /mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano --dex-root /home/wbcd/workspace/dex/retarget/third_party/dex-retargeting --frames 64 --output src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z`
- output: [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z)

**文件**
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/run_manifest.json](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/report.json](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/report.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/run.log](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/run.log)

**原因**

修复诊断脚本索引后重新验证三条序列，保持重建和优化目标不变。

**验证**

三条序列共 192 个双手帧、384 次单手优化通过，左右均输出 `[64,18]` finite qpos。每条序列每侧前 3 帧与等价 smplx 重建相比顶点最大差异不超过 `1.35e-7 m`；所有导出帧手腕严格等于对应 `tsl`。每侧逐序列平均指尖误差约 `10.92–15.81 mm`，最大单指尖误差 `74.03 mm`；未按误差剔除帧。NPZ 全字段重读相等，所有对象 SE(3) 和原始帧 ID 检查通过。run_id 的时间标签不是事件时间，准确起止时间以 manifest 字段为准。

**回滚与规范反馈**

仅本次独立输出目录；失败产物保留审计。没有训练 step/epoch、checkpoint 或 metrics.jsonl。无规范阻碍。

## 2026-09-14 12:10:56 +0000 — graspenv 依赖就绪及 OakInk2 完整双手序列试转

- activity_id: ACT-20260914-121056-OICM-OAKINK2-FULL-PILOT
- timestamp: 2026-09-14 12:10:56 +0000
- modification_version: V1.4.1
- type: code_change / operation / diagnostic
- operation_category: [code, operation, diagnostic, documentation]
- primary_task_mode: change
- change_level: L3（用户指定环境的依赖安装）+ L2（OakInk2 独立几何诊断及坐标/帧合同）
- approval: user-approved
- approval_basis: 用户明确要求“直接用 graspenv 环境，装一下依赖”“看看 OakInk2 的数据，也尝试转一下”，后续确认继续；已体现在 [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §7。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保留已有 Cm/OI-Cm 日志、V1.4 exporter/config/tests 和用户改动）
- scope: 在 graspenv 安装 geometric/manotorch 依赖，新增 OakInk2 诊断入口与检查图工具；版本指针从 V1.4 递进至 V1.4.1。只读 NAS 原始数据，输出独立 pilot；不修改正式 GRAB/ARCTIC cache、val/test、scale、训练超参、120 epochs、V1.3 或公共 src/base。
- run_id: oakink2_full_20260914T120310Z
- run_status: COMPLETED
- run_started_at: 2026-09-14T12:03:11+00:00
- run_completed_at: 2026-09-14T12:03:17+00:00
- last_source_frame_id: 1590
- last_step / last_epoch / best_metric / checkpoint: N/A（几何数据诊断，无训练）
- conclusion: SUPPORTED（原始重建、双手几何优化和 NPZ 导出工程可行）；INCONCLUSIVE（穿透、接触保持、手掌朝向、训练收益与全量数据质量）
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/run.py --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --mano-root /mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano --dex-root /home/wbcd/workspace/dex/retarget/third_party/dex-retargeting --sequence scene_04__A008++seq__ff070467cd91f735acab__2023-04-23-11-28-29.pkl --frames 0 --output src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z`
- output: [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z)

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/README.md](../../research/oakink2_inspire_pilot/README.md)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/experiment.yaml](../../research/oakink2_inspire_pilot/experiment.yaml)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/run.py](../../research/oakink2_inspire_pilot/run.py)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/plot.py](../../research/oakink2_inspire_pilot/plot.py)

**原因**

用户希望直接复用 graspenv 并确认 OakInk2 是否可扩大双手 MANO/Inspire 数据。已在 V1.4 final plan 下补充本次明确授权的 pilot，指导及正式训练数据范围仍为 GRAB+ARCTIC；不将 OakInk2 试转等同于正式接入。

**验证**

- 安装并导入 `anytree 2.12.1`、`nlopt 2.7.1`、`pinocchio 2.7.0`、`sapien 2.2.2`、`pyrender 0.1.45`、`dex-retargeting 0.4.6`，追加官方 `manotorch 0.0.2`。`pip check`：`No broken requirements found.`；当前环境 `torch 2.0.1+cu118`、`numpy 1.24.4`。安装命令取消失效的本机代理，不修改持久代理设置。
- 新 clone 的 manotorch commit `a2a70c591f91551078b7bb2af9b5d9f275b626e0`，dex-retargeting commit `8632b2cab32e1b51ce379940c414a0f78332ff6b`；二者均 clean。真实 import 路径及 commit 见 source_provenance.json。
- NAS `downloads/hf/OakInk-v2/anno_preview` 共 627 份标注，匹配 627 份配套 frame metadata；metadata 合计 4,029,640 mocap 帧、1,004,631 RGB 帧。该总数不是全量 pose 重建成功计数。
- 官方 OakInk2 MANO 使用 quaternion wxyz、`flat_hand_mean=True`、`center_idx=0`，将标注平移加到已按手腕居中的结果；初始临时尝试曾直接套用 ARCTIC 均值/平移约定，其误差不作有效证据。最终使用官方 manotorch，并与等价 smplx 独立重建对照。
- 本次完整序列 `scene_04__A008++seq__ff070467cd91f735acab__2023-04-23-11-28-29`：1,591 帧，2 个物体，两侧各 `[1591,18]` finite native qpos。每侧前 3 帧与 smplx 最大顶点差异不超过 `1.37e-7 m`，所有 1,591 帧的手腕平移误差为零。
- 左/右平均指尖误差 `8.7163 / 8.7422 mm`，p95 `19.6628 / 18.1939 mm`，最大单指尖误差 `48.3018 / 31.8341 mm`；左 8 帧的五指平均误差超过 20 mm，右 0 帧。保留全部帧；这些误差不是穿透/接触成功率。
- 输出含左右 MANO 顶点/21 点/参数、Inspire qpos+joint names+逐 link FK、对象 ID+原始 float64 位姿、原始 mocap ID；2 个物体的 SE(3) 与帧对应检查通过。NPZ 全字段重新读取后精确相等，约 37.98 MiB。
- `plot.py` 从保存的 FK 和 URDF visual mesh 重建手部几何并生成检查图，已人工查看；物体与双手保留同一 native world，不做仅右手的全局旋转。pilot 使用依赖包的官方 Inspire URDF/config，未将 native qpos 冒充 Dexplore 的 joint reorder 格式。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/run.py`：通过。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_arctic_landmark_contract.py src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py`：`5 passed in 0.91s`。OakInk2 证据来自上述真实重建/FK/NPZ 检查，既有 tests 只用于相关合同回归。
- `git diff --check`、`run.py/plot.py` AST 语法检查及完整 NPZ SHA256/1591 帧连续 ID/双手 shape/双物体 shape 复核：通过。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix docs/current_versions.yaml --scope-prefix src/task/ObjectInteractionCm/docs/plan/V1.4.md --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --scope-prefix src/task/ObjectInteractionCm/research/oakink2_inspire_pilot --check-links`：通过，6 个选定变更路径、20 个本地链接可导航。
- 未启动正式全量导出、OI-Cm cache 构建或训练。OakInk2 正式接入仍需明确多物体样本构造和 train-only 分配。

**产物**

- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/run_manifest.json](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/config.json](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/config.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/metadata.json](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/metadata.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/report.json](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/report.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/run.log](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/run.log)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/geometry_review.png](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/geometry_review.png)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/environment.json](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/environment.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/pip_freeze.txt](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/pip_freeze.txt)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/source_provenance.json](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/source_provenance.json)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/dependency_actions.md](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/dependency_actions.md)
- [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/scene_04__A008++seq__ff070467cd91f735acab__2023-04-23-11-28-29.npz](../../research/oakink2_inspire_pilot/output/oakink2_full_20260914T120310Z/scene_04__A008++seq__ff070467cd91f735acab__2023-04-23-11-28-29.npz)

**回滚与规范反馈**

回滚为移除新增 research 诊断定义与本次独立 output、恢复本次 plan/版本指针增量；不触碰旧数据或其他工作区改动。环境安装前未保存完整 pip freeze，不能保证仅卸载顶层依赖即可精确恢复传递依赖；已保存安装后快照和操作清单，未执行卸载。

本次没有格式、目录、版本、日志或审批规则阻碍；已有明确用户授权，无需重复确认。没有修改 AGENTS/Skill/治理合同。

## 2026-09-14 13:39:36 +0000 — TopoRetarget-Repro 与 Dexplore 几何重定向方案对比诊断

- activity_id: ACT-20260914-133936-OICM-TOPO-DEX-COMPARE
- timestamp: 2026-09-14 13:39:36 +0000
- modification_version: V1.4.1
- type: diagnostic / operation
- operation_category: [diagnostic, operation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读时间轴与文件长度核对）
- change_level: L0（只读仓库、代码和资产能力检查；新增独立诊断报告与 manifest）
- approval: user-approved（用户明确要求比较 `/home/wbcd/workspace/oyx_ws/TopoRetarget-Repro` 与 Dexplore，并连续确认继续）
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- scope: 只读比较 Dexplore Inspire position retargeting 与 TopoRetarget-Repro 的目标手、输入数据、目标函数、接触/穿透处理、左右手和大规模导出适配；不修改两个外部仓库，不安装新依赖，不改变正式数据、cache、val/test、训练超参或 checkpoint。
- run_id: toporetarget_vs_dexplore_20260914T133936Z
- run_status: COMPLETED
- last_step / last_epoch / best_metric / checkpoint: N/A（方案诊断，无训练）
- external_repositories: Dexplore `c31f57f186409ce5f0de47ced2d347abffe45d06` clean；TopoRetarget-Repro `a1052368b1e9bdee77ecef71a49de571d583282d` clean。
- evidence: TopoRetarget Arti-MANO RH/LH generic model 均加载并通过 `validate(seed=4,dtype=float64)`；两侧均 22 DoF、28 links、27 joints、5 fixed，FK 交叉检查最大平移误差约 `2.8e-17 m`。仓库检索未发现 Inspire 配置/插件；ARCTIC、OakInk2 在其 README 中仍为 planned。Dexplore `convert_grab.py` 直接绑定 Inspire URDF/config，输出约定为 native 18D。
- environment: TopoRetarget 项目要求 Python `>=3.10,<3.14`；本机 `graspenv` 为 Python 3.8.20，系统 Python 3.12 缺少 Typer，故未将 CLI 启动失败误报为算法失败，也未改动环境。
- conclusion: ENGINEERING_PASS（两边的已存在基础能力已核实）；INCONCLUSIVE（当前没有同一 Inspire 目标手、同一输入和同一指标的公平算法 benchmark）；SUPPORTED（立即扩大 OI-Cm Inspire train-only 数据时选 Dexplore）。
- command: 见 [toporetarget_vs_dexplore_20260914T133936Z/commands.log](../../research/toporetarget_vs_dexplore/output/toporetarget_vs_dexplore_20260914T133936Z/commands.log)；完整对比见 [toporetarget_vs_dexplore_20260914T133936Z/report.md](../../research/toporetarget_vs_dexplore/output/toporetarget_vs_dexplore_20260914T133936Z/report.md)。
- output: [toporetarget_vs_dexplore_20260914T133936Z](../../research/toporetarget_vs_dexplore/output/toporetarget_vs_dexplore_20260914T133936Z)，manifest 为 [run_manifest.json](../../research/toporetarget_vs_dexplore/output/toporetarget_vs_dexplore_20260914T133936Z/run_manifest.json)。
- validation: 直接 RobotHandRegistry/FK/asset manifest 校验通过；两个外部仓库 `git status --short` 均为空；Ref2Dex `git diff --check` 待本条记录写入后执行。
- protected_boundary: 未修改 Dexplore、TopoRetarget-Repro、OI-Cm 正式 exporter/cache、val/test、训练配置、RL 数据或公共 `src/base`。

**原因**

用户需要在扩大 OI-Cm 数据前判断 TopoRetarget-Repro 是否能替代 Dexplore 的 Inspire 几何重定向。两套方案当前目标手和数据支持不一致，必须先做能力矩阵和运行基础校验，避免把 Arti-MANO 结果误当作 Inspire 算法对比。

**验证**

- `PYTHONPATH=src /home/wbcd/miniconda3/envs/wbcd/bin/python` 直接加载 registry、URDF、anchor 和 FK；`artimano_rh`、`artimano_lh` 均 `validate=pass`。
- `PYTHONPATH=src python3 -m toporetarget --help` 的唯一失败为当前环境缺少 `typer`；项目要求 Python 3.10+，graspenv 为 Python 3.8.20，因此未把环境缺失误记为算法失败。
- `git diff --check` 及 `audit_diff.py --check-links` 在本条记录完成后执行；报告和 manifest 保存上述能力、差异和结论标签。

**规范反馈**

本次只读诊断需在最近作用域 activity 中记录外部仓库 commit、环境阻塞和“工程可用/算法结论”分层，当前规则足以表达，无额外规范阻碍。

## 2026-09-14 13:45:00 +0000 — Inspire 目标手接入工作量边界诊断

- activity_id: ACT-20260914-134500-OICM-TOPO-INSPIRE-SCOPE
- timestamp: 2026-09-14 13:45:00 +0000
- modification_version: V1.4.1
- type: diagnostic / documentation
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读代码路径与接口边界；更新独立诊断报告）
- approval: user-approved（用户追问 Inspire 是否方便加入，要求继续当前比较）
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- scope: 判断 TopoRetarget-Repro 增加 Inspire 目标手是否只需配置，区分 generic URDF/FK 最小接入与完整 GRAB/ARCTIC/OakInk2 workflow 接入；不修改外部仓库、正式数据、训练配置或 cache。
- run_id: toporetarget_vs_dexplore_20260914T133936Z
- run_status: COMPLETED
- last_step / last_epoch / best_metric / checkpoint: N/A（接口诊断，无训练）
- evidence: generic URDF parser 已支持 `revolute`、`continuous`、`prismatic`；但 `WorkflowRequest.validate`、`WORKFLOW_ID=grab_to_artimano`、Arti-MANO loader、geometry/contact audit 和部分 profiles/path 仍存在 Arti-MANO 特化。当前仓库仍无 Inspire spec、anchor、qpos、surface/collision profile 或 ARCTIC/OakInk2 adapter。
- conclusion: 基础 Inspire FK/anchor 接入可复用 generic 层，主要是资产与配置；完整 TopoRetarget Inspire workflow 需要专门适配代码、碰撞/接触合同和回归 benchmark，属于中等改动，不是求解器重写。
- reason: 用户需要判断是否值得把 TopoRetarget 用作 Inspire 接触质量方案，避免误以为复制 URDF 即可接入。
- validation: 已读取 parser/model/registry/workflow/geometry/retarget 代码并更新对比报告；未修改两个外部仓库及正式 OI-Cm 路径。
- output: [toporetarget_vs_dexplore_20260914T133936Z/report.md](../../research/toporetarget_vs_dexplore/output/toporetarget_vs_dexplore_20260914T133936Z/report.md)。
- protected_boundary: Dexplore、TopoRetarget-Repro、正式 GRAB/ARCTIC/OakInk2 cache、val/test、训练超参、RL 数据和公共 `src/base` 均未改动。

**规范反馈**

本次问题属于上一条诊断的工作量细化，现有 activity 与独立报告结构可以表达。链接审计要求独立的原因/验证段，已补齐；不修改审计合同。

**原因**

用户询问 Inspire 接入是否只需要配置，进一步定位完整 workflow 中的手型特化边界。

**验证**

已通过 `rg` 和源码读取核实 generic parser、robot registry 和 WorkflowRequest 的实现边界；本条仅补充诊断，不运行重定向。

## 2026-09-14 14:17:42 +0000 — TopoRetarget Inspire 双手适配及同侧模型几何 pilot

- activity_id: ACT-20260914-141742-OICM-INSPIRE-TOPO-PILOT
- timestamp: 2026-09-14 14:17:42 +0000
- modification_version: V1.4.2
- type: code_change / data_change / diagnostic / operation
- operation_category: [code, data, diagnostic, operation, documentation]
- primary_task_mode: change
- change_level: L2（目标手、锚点与坐标适配）+ L3（隔离依赖环境）
- approval: user-approved
- approval_basis: 用户要求对比 TopoRetarget-Repro 与 Dexplore、询问 Inspire 接入后要求“继续”；既有 V1.4 指导允许少量适配性 pilot。左右模型选择已说明，左手按推荐的此前官方模型进行候选诊断。
- skills_used: research-change-control, research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true（保护本次之前所有 Cm/OI-Cm、V1.4 exporter/config/tests 与 OakInk2 改动）
- scope: 新增 Task-local Inspire adapter、源输入重建、原 Topo API 调用和几何图；版本指针 V1.4.1 -> V1.4.2；不修改外部两仓库、公共 src/base、正式数据/cache、val/test、训练超参、epoch 或 checkpoint。
- run_id: inspire_integration_20260914T141600Z
- run_status: COMPLETED
- child_runs: topo_right_20260914T141000Z COMPLETED（3/3 accepted）；topo_left_20260914T141200Z COMPLETED（3/3 accepted）；inspire_right_20260914T140500Z / inspire_left_20260914T141100Z COMPLETED（源重建与 position）；inspire_pilot_20260914T140000Z FAILED（Dexplore left 缺 joint1/tip links，失败保留）。
- last_source_frame_id: 242；last_step / last_epoch / best_metric / checkpoint: N/A（几何 pilot，无训练）
- conclusion: SUPPORTED（左右 Inspire FK/anchor、wrist/18D 等价与原 Topo warm/graph/refinement API 可运行）；INCONCLUSIVE（全数据集效果及 OI-Cm 训练收益）。

**文件**
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/README.md](../../research/toporetarget_vs_dexplore/README.md)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/experiment.yaml](../../research/toporetarget_vs_dexplore/experiment.yaml)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/inspire_adapter.py](../../research/toporetarget_vs_dexplore/inspire_adapter.py)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/prepare.py](../../research/toporetarget_vs_dexplore/prepare.py)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/run.py](../../research/toporetarget_vs_dexplore/run.py)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/plot.py](../../research/toporetarget_vs_dexplore/plot.py)

**原因**
Topo 有通用目标手底层，但原 parser 拒绝 mimic、上层 workflow 仍特化 Arti-MANO。采用独立适配调用原算法，固定同侧两方法的 URDF、12 独立指关节、源点、对象和帧，避免改手型后做不公平比较。

**实现与科研边界**
右手沿用 Dexplore inspire_hand_new；实际 Dexplore left 无 wrist/tips，候选左手使用此前 dex-retargeting 官方 left，左右分开统计。mimic 标签在私有副本移除以匹配 Dexplore ignore_mimic_joint=True，不能解释为真实 6 actuator 模型。Topo base 单独承载腕部 SE(3)，输出按名称还原 18D；四指 DIP 使用 PIP-tip 中点工程锚点。NAS 世界坐标/原帧/物体位姿不改变。

**验证**
- `RobotHandModel.validate(seed=42,dtype=float64)` 双手通过；原 Pinocchio 与 Topo 逐 link FK 最大差异 <=6.67e-16；随机 wrist 拆合 <=5.56e-16；21 点 Jacobian 与中心差分 <=1.48e-11。
- GRAB s1/airplane_lift native 120 Hz 帧 240..242 共用输入：MANO fullpose/PCA 重建差异 <1e-7 m；与官方 ObjectModel 对象变换差异 <2e-9 m。左右各 3 帧原 SLSQP 接受，输出均保留。
- 右侧指尖均值误差：position 16.23 mm、Topo final 48.87 mm；相同 collision samples >1 mm 穿透比例 8.49% ->0%，但表面 <=2 cm 比例 39.66% ->10.66%，<=2 mm 比例 4.17% ->0%。几何图已查看：不能将离物体更远当作接触保持改善。左侧源无接触，仅证明工程链路；指尖 10.06 ->56.56 mm。
- final_refinement 使用现有 scipy_slsqp_active_set_contact_rich_v3_fixed profile；warm、交互图、SDF 与权重未修改。reference SDF 核对两方法，collision 采样右416/左640，不代表连续全表面无穿透。
- 右/左 final 总耗时约23.38/20.77 s，各3帧；不是全量吞吐量 benchmark。不同依赖环境、solver预算、原输入窗口和未启用全部加速配置均在报告中说明。
- `src/task/ObjectInteractionCm/assets/toporetarget_pilot_env/bin/python -m pip check`：No broken requirements found。venv 复用 wbcd 的 Python3.10/Torch2.7.1，追加 trimesh4.12.2/zarr2.18.3/numcodecs0.13.1/fasteners/asciitree；未升级 graspenv 或既有 Torch/CUDA。初次 zarr2.18.7 的 Python>=3.11 要求导致安装命令失败，改用兼容版后成功，原失败无包安装。
- `python -m py_compile` 四个新增脚本通过；独立 FK/随机姿态/Jacobian 与官方重建检查直接使用真实资产，未扩跑无关 Task tests。`plot.py` 用 Pinocchio 复核导出的 final18 指尖与保存的 Topo 点一致，绘图检查通过。
- `prepare.py --help` / `run.py --help` 分别在 graspenv / 隔离 venv 通过；两个外部仓库 `git status --short` 均为空。
- `git diff --check` 通过；`audit_diff.py --check-links` 通过，activity 与 8 个变更路径一致，26 个本地链接可导航。最终右手几何图已查看。

**产物与命令**
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/report.md](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/report.md)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/run_manifest.json](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/comparison.json](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/comparison.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/config.json](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/config.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/metadata.json](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/metadata.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/pip_freeze.txt](../../research/toporetarget_vs_dexplore/output/inspire_integration_20260914T141600Z/pip_freeze.txt)
- 精确 run 命令记录在下列 manifest；prepare 的早期运行 manifest 明确为 retrospective config 重建，没有虚构启动时间。
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/run_manifest.json](../../research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/config.json](../../research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/config.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/metrics.jsonl](../../research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/report.json](../../research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/report.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/run_manifest.json](../../research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/config.json](../../research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/config.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/metrics.jsonl](../../research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/metrics.jsonl)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/report.json](../../research/toporetarget_vs_dexplore/output/topo_left_20260914T141200Z/report.json)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/right_geometry_comparison.png](../../research/toporetarget_vs_dexplore/output/topo_right_20260914T141000Z/right_geometry_comparison.png)
- [src/task/ObjectInteractionCm/research/toporetarget_vs_dexplore/output/inspire_pilot_20260914T140000Z/run_manifest.json](../../research/toporetarget_vs_dexplore/output/inspire_pilot_20260914T140000Z/run_manifest.json)

**回滚**
只移除这次新增 research 定义与独立输出、assets/inspire_topo_* 派生资产和 assets/toporetarget_pilot_env；恢复本次 plan/指针增量。外部源仓库和所有旧数据/运行/用户改动保持原样；未提交 Git。

**规范反馈**
原左手资产缺失语义点是技术适配边界，已显式区分模型来源；没有以无回复当作外部发布或正式数据替换批准。activity 审计要求独立原因/验证段，已补齐；初期 prepare 缺 manifest 的记录已追补并标记 retrospective，后续入口已自动生成。没有修改 AGENTS/Skill/公共治理合同。

## 2026-09-14 14:44:33 +0000 — ARCTIC/OakInk2 导出数据 KNN/距离查看器

- activity_id: ACT-20260914-144433-OICM-EXPORT-VIS
- timestamp: 2026-09-14 14:44:33 +0000
- modification_version: V1.4.2
- type: diagnostic / operation / documentation
- operation_category: [diagnostic, operation, documentation]
- primary_task_mode: run-only/operation
- change_level: L1（只读适配投影与查看器运行；不改变源数据或正式 cache）
- approval: user-approved（用户指定使用已有 KNN/距离可视化脚本）
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- scope: 使用现有 `visualize_grab.py` 查看此前导出的 ARCTIC 与 OakInk2 pilot；将 world-space NPZ 只读投影为查看器所需的 4096 物体点/1538 手点显示格式，原始导出、训练配置和 cache 不变。
- run_id: export_visualization_20260914T144433Z
- run_status: RUNNING（Viser `127.0.0.1:8140` 服务仍在运行；已从先前的 `0.0.0.0` 绑定重启为本地监听）
- output: [exported_20260914T151000Z](../../research/export_visualization/output/exported_20260914T151000Z)、[index.json](../../research/export_visualization/output/exported_20260914T151000Z/index.json)、[run_manifest.json](../../research/export_visualization/output/exported_20260914T151000Z/run_manifest.json)
- files: [stage_for_visualizer.py](../../research/export_visualization/stage_for_visualizer.py)、[visualize_grab.py](../../visualize_grab.py)
- entries: OakInk2 3 条已导出序列，ARCTIC `s07/scissors_use_01` 48 帧；合并 viewer 可通过轨迹下拉框切换。
- command: `env PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index src/task/ObjectInteractionCm/research/export_visualization/output/exported_20260914T151000Z/index.json --sequence s07/scissors_use_01 --frame 12 --point-display both --mesh-display off --host 127.0.0.1 --port 8140 --fps 8`
- validation: 两条 `--check-only` 均通过；OakInk2 frame 0 KNN union counts n=1/4/8/16/32/64 为 8/28/40/68/104/165，ARCTIC frame 12 为 3/8/15/24/49/97；两个独立端口的 Viser 初始渲染均启动成功。`viser 1.1.0` 及其依赖仅安装到 graspenv。
- semantics: 距离阈值按最近物体点计算；KNN 为物体点查询最近手点并取并集，沿用既有查看器定义。当前适配是可视化投影，不是训练 cache；OakInk2 多物体被合并为显示点池，mesh 关闭以避免把对象名当作 GRAB mesh。
- conclusion: SUPPORTED（查看器和高亮逻辑运行）；不构成数据质量或训练收益结论。
- protected_boundary: ARCTIC/OakInk2 原始导出、NAS 数据、正式 OI-Cm cache、split、val/test、训练配置和 checkpoint 均未修改。
- rollback: 停止 run_id 对应 Viser 进程并删除该独立 visualization output；恢复不涉及正式数据。

**原因**

用户希望直接检查刚导出的 ARCTIC/OakInk2 手点与物体距离分布，并使用现有 KNN 着色逻辑；两个 pilot NPZ 不是查看器原生 index/cache，因此需要隔离显示投影。

**验证**

`visualize_grab.py --check-only` 对两个数据源均加载成功，距离与 KNN 统计可复现；`viser` 服务在端口 8140 完成初始化。适配器通过 `graspenv` 运行，未写回输入文件。

## 2026-09-14 15:17:25 +0000 — 查看器轨迹帧间隔诊断

- activity_id: ACT-20260914-151725-OICM-EXPORT-VIS-FRAME-DIAG
- timestamp: 2026-09-14 15:17:25 +0000
- modification_version: V1.4.2
- type: diagnostic / documentation
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读检查 source_frame_id 和导出数组）
- approval: user-approved（用户询问轨迹是否跳帧及变化原因）
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- scope: 检查当前 ARCTIC/OakInk2 viewer 投影的原始 frame ID、导出采样策略和相邻手点位移；不改数据、配置或 viewer。
- files: [stage_for_visualizer.py](../../research/export_visualization/stage_for_visualizer.py)、[exported index](../../research/export_visualization/output/exported_20260914T151000Z/index.json)
- run_id: export_visualization_frame_diag_20260914T151725Z
- run_status: COMPLETED
- finding: viewer 按 staged cache 的相邻帧播放，不会额外跳帧；OakInk2 导出阶段为每条原序列均匀取 64 帧，故 source frame 间隔分别为 scene_01 `165/166`、scene_02 `72/73`、scene_04 `25/26`。ARCTIC 本次 staged 的 48 帧为连续 `0..47`，但只覆盖序列开头窗口。
- evidence: OakInk2 scene_01 手部质心相邻导出帧位移中位数/最大值约 `107/583 mm`，scene_02 `16/262 mm`，scene_04 `12/304 mm`；大变化来自稀疏均匀采样叠加真实动作，不是 KNN 或播放回调重新抽帧。导出代码 `select_frame_ids(..., limit=64)` 保存于 OakInk2 run source。
- conclusion: SUPPORTED（跳帧原因已定位）；当前图不能代表逐原始帧连续运动。
- protected_boundary: 原始 OakInk2/ARCTIC 数据、已有导出 NPZ、正式 cache、训练配置和正在运行的 viewer 未修改。
- rollback: 无代码或数据变更；删除本条诊断记录即可。

**原因**

用户观察到 viewer 中相邻画面变化大，需要区分播放器行为和导出阶段的帧抽样。

**验证**

读取三条 OakInk2 NPZ 与当前 ARCTIC staged `source_frame_id`，统计 `np.diff` 及手部质心位移；确认 viewer 只按 staged 帧索引递增。

## 2026-09-14 15:27:40 +0000 — OakInk2/ARCTIC 时间轴与场景长度核对

- activity_id: ACT-20260914-152740-OICM-EXPORT-FPS-DIAG
- timestamp: 2026-09-14 15:27:40 +0000
- modification_version: V1.4.2
- type: diagnostic / documentation
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读时间轴与文件长度核对）
- approval: user-approved（用户询问轨迹帧率、场景帧切分和轨迹长度）
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- scope: 只读 OakInk2 toolkit 常量、annotation pkl 的 `mocap_frame_id_list`/`frame_id_list`/`raw_mano`，以及 Ref2Dex ARCTIC source-fps 配置；不修改数据或 viewer。
- files: [stage_for_visualizer.py](../../research/export_visualization/stage_for_visualizer.py)、[OakInk2 meta.py](../../../../../dataset/OakInk2/src/oakink2_toolkit/meta.py)、[OakInk2 pilot config](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/config.json)、[ARCTIC stage4_cm.py](../../../../../process/ARCTIC/stage4_cm.py)
- run_id: export_fps_diag_20260914T152740Z
- run_status: COMPLETED
- finding: OakInk2 toolkit 定义 `FPS_MOCAP=120`、`FPS_VIDEO=30`；三条 pilot annotation 的 mocap 长度分别为 10449、4577、1591 帧（约 87.07、38.13、13.25 秒），RGB 帧分别为 2610、1144、397。此前 pilot 命令 `--frames 64` 仅均匀选 64 个 mocap frame ID，并未重采样为 30 Hz。Ref2Dex ARCTIC Stage4 合同为 `SOURCE_FPS=30.0`，本次 ARCTIC staged 的 48 帧连续但只取序列开头。
- conclusion: SUPPORTED（时间轴和长度已核实）；当前 viewer 投影 manifest 的 `effective_fps=30` 仅为 OI-Cm 兼容显示字段，不能覆盖 OakInk2 原始 mocap 120 Hz 及稀疏选帧事实。
- protected_boundary: OakInk2/ARCTIC 原始 annotation、已有 pilot NPZ、正式 cache、训练配置和运行中的 viewer 未修改。
- rollback: 无代码或数据变更；删除本条诊断记录即可。

**原因**

需要解释 viewer 中相邻画面变化，并区分 OakInk2 的 mocap、视频时间轴与本次 pilot 导出长度。

**验证**

读取本地 OakInk2 toolkit 常量和三条 annotation 的实际列表长度，结合 pilot `report.json/config.json` 与 ARCTIC Stage4 的 `SOURCE_FPS` 核对；未把 RGB 30 Hz 误当成 MANO mocap 频率。

## 2026-09-14 15:58:39 +0000 — ARCTIC 单物体核实与 OakInk2 裁剪范围澄清

- activity_id: ACT-20260914-155839-OICM-OBJECT-SCOPE-DIAG
- timestamp: 2026-09-14 15:58:39 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读标注清点与需求记录）
- approval: user-approved
- approval_basis: 用户要求核实 ARCTIC 是否单物体，并明确 OakInk2 先筛运动物体、再用手物最近距离严格小于 2 cm 切段。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 本次仅补充本活动记录；不实现裁剪、不运行数据导出。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- evidence: [process/ARCTIC/raw.py](../../../../../process/ARCTIC/raw.py) 的 `process_sequence`；本机 NAS `arctic/data/arctic_data/data/raw_seqs` 全量标注数组头。
- command: 使用 `/home/wbcd/miniconda3/envs/graspenv/bin/python`，对上述 NAS 目录 `Path.rglob('*.object.npy')`，逐项 `np.load(path, mmap_mode='r', allow_pickle=False).shape`，并核对同名 `.mano.npy`；阅读 `sed -n '660,725p' process/ARCTIC/raw.py`。
- finding: 301 个 `.object.npy` 全为 `[T,7]`，对应 301 个 `.mano.npy`，无异常形状或缺失配对。解析代码将每条序列映射到一个物体，7 列为铰链角、整体旋转及整体平移；铰链部件属于同一标注对象，未发现多独立标注物体的 ARCTIC 轨迹。
- requirement: ARCTIC 不做本次裁剪。OakInk2 按既定 30 Hz 时间轴先用物体位姿筛运动候选及其区间，静止物体排除，再仅对候选区间计算两手到目标物体的最近距离，以任一手距离严格小于 0.02 m 的连续帧切分。每段一个目标物体与单手或双手；不同目标物体分别输出，不能合并场景物体点池，不能跨被剔除帧拼接。
- pending_detail: 运动判定需同时考虑平移和旋转并排除跟踪抖动，数值阈值尚未确定；拟由原始 MANO 确定帧段、Inspire 沿用同一 source frame ID，尚未实现。
- conclusion: SUPPORTED（仅限本机 ARCTIC 标注清点）；裁剪与训练效果未评估。
- protected_boundary: 现有数据、导出 NPZ、查看器、正式 cache、val/test、训练配置、checkpoint 和用户既有改动均未修改。
- rollback: 本次仅追加活动条目，可单独撤销本条；无数据回滚需求。

**原因**

用户将多物体裁剪限定在 OakInk2，并要求先依据物体运动缩小计算范围，需要排除将 ARCTIC 铰链部件误认为独立物体的情况。

**验证**

全量数组头检查输出 `object_files=301, mano_files=301, shape_signatures={(2,7):301}, anomalies=[], missing_mano_pairs=[]`；结合解析代码确认字段含义。使用 `audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links` 审计本条记录和本地链接。

## 2026-09-14 16:28:13 +0000 — OakInk2 同时运动实例及物体部件归属核对

- activity_id: ACT-20260914-162813-OICM-SIMULTANEOUS-MOTION-DIAG
- timestamp: 2026-09-14 16:28:13 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（读取已有位姿与部件元数据）
- approval: user-approved
- approval_basis: 用户询问 OakInk2 是否存在两个物体同时运动。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 只读三条 pilot 对应的物体位姿；本次仅追加活动记录，未实施运动筛选或裁剪。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- input: [src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/config.json](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/config.json)、[src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/report.json](../../research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/report.json) 中的 NAS annotation 与 object_root；[dataset/OakInk2/src/oakink2_toolkit/dataset.py](../../../../../dataset/OakInk2/src/oakink2_toolkit/dataset.py) 的 `get_part_tree_root`。
- finding: `scene_02__A002++seq__e1fa69abb1738c8fea90__2023-04-17-14-54-52` 在原始帧 3204..3228（26.7..26.9 秒）中，剪刀标注部件 `O02@0035@00001` 和纸部件 `O02@0056@00001` 同时运动。每隔 4 个 mocap 帧取样，6 个连续间隔的平移步长均分别大于 3.14 mm、3.68 mm；窗口首尾净位移分别 23.00 mm、39.93 mm。
- part_identity: NAS `object_raw/obj_desc.json` 确认名称为剪刀和纸。`object_affordance/object_part_tree.json` 确认剪刀的 `O02@0035@00001/00002` 共同属于 `O02@0035@00003`，纸的 `O02@0056@00001/00002` 共同属于 `O02@0056@00003`，夹子的 `O02@0032@00001/00002` 共同属于 `O02@0032@00003`。因此部件 ID 数量不能当作独立物体数量；上述剪刀与纸的示例属于两个不同根实例。
- limitation: 临时以每个 30 Hz 间隔平移大于 3 mm 定位明显例子，不是已批准的正式运动阈值；未计算手物 2 cm 距离，未估计全数据集发生比例。
- implication: 后续若按单个物理物体切段，应先依据官方部件树归组，完整保留该目标的部件；同时运动的不同目标可分别生成时间重叠的样本。本次未修改实现或 final plan。
- conclusion: SUPPORTED（存在同时运动的两个不同物体实例）；不构成裁剪质量或训练效果结论。
- protected_boundary: 数据、代码、配置、正式 cache、val/test、Inspire 导出与查看器均未修改。
- rollback: 仅追加本条记录，可单独撤销，无数据回滚需求。

**原因**

需要用连续位姿证据回答实际是否存在同时运动，并区分独立物体与铰接/可分部件，避免用稀疏 64 帧采样推断同时性。

**验证**

使用 graspenv Python 只读载入原始 annotation；下列核心计算复现上述位移（单位米转毫米），帧率来自已核实的 120 Hz mocap 时间轴：

```python
import json, pickle
from pathlib import Path
import numpy as np
cfg = json.loads(Path('src/task/ObjectInteractionCm/research/oakink2_inspire_pilot/output/oakink2_inspire_20260914T121000Z/config.json').read_text())
p = next(Path(cfg['annotation_root']).glob('scene_02__A002++seq__e1fa69abb1738c8fea90*'))
with p.open('rb') as h:
    a = pickle.load(h)
for obj in ['O02@0035@00001', 'O02@0056@00001']:
    t = np.stack([a['obj_transf'][obj][i][:3, 3] for i in range(3204, 3229, 4)])
    print(obj, np.linalg.norm(t[-1] - t[0]) * 1000,
          np.linalg.norm(np.diff(t, axis=0), axis=-1).min() * 1000)
```

活动链接审计命令：`python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links`。

## 2026-09-14 17:29:00 +0000 — OakInk2 单物体段正式范围确认

- activity_id: ACT-20260914-172900-OICM-OAKINK2-SINGLEOBJ-SCOPE
- timestamp: 2026-09-14 17:29:00 +0000
- modification_version: V1.4.2
- operation_category: [data, experiment, operation, documentation]
- primary_task_mode: change
- change_level: L3（正式数据范围与长时训练边界）
- approval: user-approved
- approval_basis: 用户明确确认 OakInk2 只用单物体段，其他数据全量导出并开始训练。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 将用户确认追加到 [plan/V1.4.md](../plan/V1.4.md) §9；GRAB/ARCTIC 全量、OakInk2 仅单物体 primitive train 段；val/test 保护不变。当前仅完成范围记录和启动前检查，未导出或训练。
- files: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- finding: 范围已消除 OakInk2 是否纳入的歧义；但全量 producer/cache/index 尚未实现，OakInk2 运动阈值仍需在 exporter 中固定并记录，不能立即运行正式任务。
- validation: 计划 §9 已写入并保持 `final`；此前 V1.4 定向测试 5 项通过；正式 index 路径仍不存在。
- conclusion: INCONCLUSIVE（范围已获确认，工程尚未具备正式导出/训练条件）。
- protected_boundary: V1.3 cache、旧 split、val/test、pilot/output、checkpoint 和用户已有改动均未修改。
- rollback: 仅撤销本次计划和活动增量即可恢复范围记录；未生成数据或运行状态。

**原因**

用户确认 OakInk2 的正式纳入策略，需要在开始实现全量 producer 前锁定数据边界和部件归组规则。

**验证**

读取 V1.4 guidance/plan、配置和 producer 入口；确认计划新增 §9 与用户要求一致，并核对正式 index 尚不存在。活动链接审计待本条追加后执行。

## 2026-09-14 17:24:06 +0000 — V1.4 全量导出与训练启动前阻塞检查

- activity_id: ACT-20260914-172000-OICM-FULL-RUN-READINESS
- timestamp: 2026-09-14 17:24:06 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, operation]
- primary_task_mode: read-only/diagnostic
- change_level: L3（正式全量数据处理和长时训练的启动前检查）
- approval: user-approved（用户要求全量导出并开始训练；本条仅检查是否具备启动条件）
- approval_basis: 用户明确要求“现在导出全量并开始训练”。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 只读检查 V1.4 final plan、指导、配置、数据处理入口和输出路径；未启动全量导出或训练，未修改数据和配置。
- files: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[src/task/ObjectInteractionCm/docs/指导/V1.4.md](../指导/V1.4.md)、[src/task/ObjectInteractionCm/configs/active/grab_arctic_inspire_geometric_v1_4.yaml](../../configs/active/grab_arctic_inspire_geometric_v1_4.yaml)、[src/task/ObjectInteractionCm/tools/data/export_bilateral_geometry.py](../../tools/data/export_bilateral_geometry.py)
- finding: V1.4 配置引用的 `data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/index.json` 不存在；`export_bilateral_geometry.py` 仅能拼接已经存在的左右手 NPY，不能从 GRAB/ARCTIC/OakInk2 原始数据生成全量 producer/cache；`max_steps=202300` 仍是等待最终 train-row 统计的占位值。现有 OakInk2 plan §7 只允许 pilot，未把 627 条 OakInk2 序列纳入正式 train 范围。
- validation: `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py src/task/ObjectInteractionCm/tests/test_arctic_landmark_contract.py` 结果 `5 passed`；配置 index 路径检查为缺失；工作区已有 V1.4 未提交改动保持不变。
- blockers: 需先完成全量 geometric producer、OakInk2 是否正式纳入 train 的范围确认、运动抖动阈值和按最终 train rows 计算 max_steps；在此之前启动会失败或训练错误数据。
- conclusion: INCONCLUSIVE（仅说明工程尚未达到正式启动条件，不代表数据或模型效果结论）。
- protected_boundary: V1.3 cache、split、val/test、已有 pilot/output、训练配置和 checkpoint 未修改，未启动新长任务。
- rollback: 无数据或运行状态变更；仅追加本条活动记录，可单独撤销。

**原因**

用户要求马上进行不可逆成本较高的全量数据处理和长时训练；根据 final plan 先核实输入范围、producer、cache、预算和启动条件，避免将 pilot 或占位配置误当正式实验。

**验证**

读取 final guidance/plan/config，检查 index 目标和 producer 能力；定向 V1.4 测试 5 项通过。链接需以 `audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links` 复核。

## 2026-09-14 17:08:53 +0000 — OakInk2 多物体交互轨迹全量计数

- activity_id: ACT-20260914-170853-OICM-OAKINK2-MULTIOBJ-COUNT
- timestamp: 2026-09-14 17:08:53 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读 program_info 与部件树统计）
- approval: user-approved
- approval_basis: 用户要求统计涉及物体间交互的 OakInk2 轨迹并评估是否排除难例。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 扫描 NAS OakInk2 全部 627 个 `program/program_info/*.json`，按 `object_part_tree` 将部件归并到实例根；不读取手物距离、不修改数据或训练配置。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)、[OakInk2 dataset.py](../../../../../dataset/OakInk2/src/oakink2_toolkit/dataset.py)
- input: NAS `data/OakInk-v2-hub/program/program_info/*.json`、`object_affordance/object_part_tree.json`、`part_desc.json`。
- method: 每个 primitive 的 `obj_list` 去重后沿部件树追溯实例根；根实例数大于 1 计为多物体交互。部件数大于 1 但根相同（例如剪刀两刃）不计为多物体。
- result: 627 条轨迹包含 2840 个 primitive 段；658 个 primitive 段涉及至少两个不同物体根，来自 363 条独立轨迹（57.9%）。若按整条轨迹剔除，剩余 264 条不含多物体 primitive 的轨迹；若仅剔除多物体段，可保留 2177 个单物体 primitive 段。
- category_counts: 多物体轨迹去重计数：`cut/shear_paper/staple_paper_together` 44 条；`pour/pour_in_lab` 77 条；`scoop/scrape/stir/stir_experiment_substances` 84 条；插拔/取放类 USB、铅笔、灯泡、电源插头 95 条；实验器材/点火/装配类（`hold_test_tube` 等）45 条。类别有交集，不能直接相加。
- limitation: 这是官方任务/物体语义的候选数量，不等价于两个物体在每一帧都同时运动，也不等价于手物最近距离小于 2 cm；还需后续运动筛选和几何距离筛选。
- conclusion: SUPPORTED（计数规则和结果可复现）；难例取舍及训练收益尚未评估。
- protected_boundary: 原始 annotation、pilot、正式 cache、split、配置、checkpoint 和用户既有改动均未修改。
- rollback: 仅追加本条记录，可单独撤销。

**原因**

用户需要估算多物体交互规模，以决定是否从 OI-Cm 训练中排除剪切、倒液、实验器材等复杂动作。

**验证**

使用 graspenv Python 扫描 627 个 program_info 文件，逐段映射部件树根并统计；输出为 `segments=2840, multi_segments=658, multi_sequences=363`。活动链接审计通过。

## 2026-09-14 17:00:28 +0000 — 剪纸序列纸部件初始状态核对

- activity_id: ACT-20260914-170028-OICM-PAPER-PARTS-INITIAL-DIAG
- timestamp: 2026-09-14 17:00:28 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读初始位姿与物体网格）
- approval: user-approved
- approval_basis: 用户询问被剪开的纸是否从开始就是两个独立 ID。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 只读 `scene_02/A002` 的纸部件 annotation 和 object_repair 网格；未修改代码、数据或训练配置。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)、[OakInk2 dataset.py](../../../../../dataset/OakInk2/src/oakink2_toolkit/dataset.py)
- input: NAS `anno_preview/scene_02__A002++seq__e1fa69abb1738c8fea90__2023-04-17-14-54-52.pkl`；`object_repair/align_ds/O02@0056@00001/00002/model.obj`；`object_affordance/{object_part_tree,object_affordance,part_desc}.json`。
- finding: `O02@0056@00001` 与 `O02@0056@00002` 在原始第 0 帧均已有独立 4x4 `obj_transf` 和独立 paper mesh。第 0 帧两网格顶点最近距离约 0.141 mm（相邻/近似接触）；到帧 3204 时约 63.4 mm，说明后续运动使其分开。部件树始终将两者归于纸实例根 `O02@0056@00003`。
- semantics: 可以确认“两个纸部件 ID 从起始帧就存在”，不能据此声称数据记录了某一帧发生拓扑剪切；OakInk2 没有剪开时刻或剪开后新 ID。任务 primitive 的 `obj_list` 只显式列出纸部件 `00001`，原始场景 `obj_list` 仍含 `00002`，因此实现时应以部件树和任务语义共同决定是否把两个部件作为一个训练目标。
- conclusion: SUPPORTED（初始独立 ID 与初始几何关系已核实）；未评估剪切物理真实性或训练效果。
- protected_boundary: 原始 annotation、资产、pilot、cache、配置、val/test 和查看器均未修改。
- rollback: 仅追加本条记录，可单独撤销。

**原因**

需要区分“数据集预先把纸建模为两个部件”与“序列中途发生纸张断裂”，避免把静态部件 ID 误当作动态切割事件。

**验证**

用 graspenv Python 读取两个部件第 0/3204 帧的刚体变换和 trimesh 网格，并用 `scipy.spatial.cKDTree` 计算顶点近似最近距离；同时读取 `object_part_tree.json` 与 `program_info` 核对实例根和任务列表。活动链接审计通过。

## 2026-09-14 16:55:18 +0000 — OakInk2 剪纸序列语义与物体 ID 结构核对

- activity_id: ACT-20260914-164500-OICM-OAKINK2-SHEAR-SEMANTICS
- timestamp: 2026-09-14 16:55:18 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（读取已有 annotation、program 与物体元数据）
- approval: user-approved
- approval_basis: 用户询问 OakInk2 如何表示剪刀剪纸及物体 ID 分类。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 只读 `scene_02/A002` 剪纸序列的 annotation、program_info、task_target、affordance 和部件树；未修改代码、数据或训练配置。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)、[OakInk2 dataset.py](../../../../../dataset/OakInk2/src/oakink2_toolkit/dataset.py)
- input: NAS `OakInk2/downloads/hf/OakInk-v2/anno_preview/scene_02__A002++seq__e1fa69abb1738c8fea90__2023-04-17-14-54-52.pkl`；`data/OakInk-v2-hub/program/program_info/<seq>.json`；`program/task_target.json`；`object_affordance/{object_part_tree,object_affordance,part_desc,affordance_label}.json`。
- finding: task target 是 `Shear paper`；唯一 primitive 是 `shear_paper`。`program_info` 的 `obj_list` 为纸 `O02@0056@00001` 与剪刀两个部件 `O02@0035@00001/00002`；`obj_list_lh` 仅纸，`obj_list_rh` 为剪刀两个部件；`primitive_lh=hold`、`primitive_rh=shear_paper`、`interaction_mode=rh_main`。两只手的标注区间分别为 mocap 805..3379 和 1213..3863。
- identity: `object_part_tree` 将剪刀两个部件归到实例根 `O02@0035@00003`，纸两个部件归到实例根 `O02@0056@00003`；`object_affordance` 根节点分别含 `<shear, paper>` 和 `<be sheared by, scissors>`。`is_instance=false` 的部件有独立 mesh，`is_instance=true` 的根节点无单一 mesh。
- semantics: OakInk2 通过每帧 `obj_transf[obj_id]` 的刚体 4x4 位姿表达物体/部件运动，通过 program/affordance 表达“剪纸”语义；没有逐帧剪切线、纸张拓扑变化、断开时刻或剪开后的新物体 ID。纸的两个部件仍是同一纸实例，剪刀两个部件仍是同一剪刀实例。
- implication: 数据处理应按部件树根实例归组后做运动筛选和 2 cm 裁剪；这条任务应保留剪刀实例与纸实例，分别计算目标片段。不能把部件 ID 数量当成物体数量，也不能从刚体位姿推断纸已经被剪开。
- conclusion: SUPPORTED（字段语义和 ID 归属已由本机元数据确认）；未评估剪纸动作的几何接触或训练效果。
- protected_boundary: 原始 annotation、物体资产、已有 pilot、正式 cache、val/test、配置和查看器均未修改。
- rollback: 仅追加本条记录，可单独撤销。

**原因**

需要确定多物体裁剪时剪刀的两个部件和纸的两个部件应如何归并，以及 OakInk2 是否提供剪开后的物理状态标签。

**验证**

读取上述 JSON/PKL；`dataset.py` 的 `_load_primitive_task_from_def` 使用 `obj_list_lh`/`obj_list_rh` 区分手侧，`get_part_tree_root` 沿 `object_part_tree` 追溯实例根。相关本地链接已通过 activity 审计。

## 2026-09-14 16:38:43 +0000 — OakInk2 同时运动轨迹查看器启动

- activity_id: ACT-20260914-163843-OICM-VIS-SCENE02
- timestamp: 2026-09-14 16:38:43 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, operation]
- primary_task_mode: run-only/operation
- change_level: L0（启动既有查看器，不改输入或处理逻辑）
- approval: user-approved
- approval_basis: 用户要求可视化已核实的 OakInk2 同时运动轨迹。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 复用既有可视化投影，初始打开 `scene_02__A002++seq__e1fa69abb1738c8fea90__2023-04-17-14-54-52`，帧 28，监听 127.0.0.1:8141；8140 原查看器未停止。
- run_id: oakink2_scene02_visualizer_20260914T163843Z
- run_status: RUNNING
- command: `env PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index src/task/ObjectInteractionCm/research/export_visualization/output/exported_20260914T151000Z/index.json --split train --sequence 'scene_02__A002++seq__e1fa69abb1738c8fea90__2023-04-17-14-54-52' --frame 28 --point-display both --mesh-display off --host 127.0.0.1 --port 8141 --fps 8`
- output: [exported_20260914T151000Z](../../research/export_visualization/output/exported_20260914T151000Z)、[index.json](../../research/export_visualization/output/exported_20260914T151000Z/index.json)
- validation: 进程 PID 1875763 存活，`ss -ltnp` 确认 127.0.0.1:8141 LISTEN；查看器输出 HTTP `http://127.0.0.1:8141`。当前是既有 visualization projection，帧间隔仍遵循该投影的 64 帧 pilot，不作为连续 30 Hz 裁剪结果。
- conclusion: SUPPORTED（查看器服务已启动）；不构成数据质量或训练收益结论。
- protected_boundary: 原始 OakInk2 annotation、正式 cache、训练配置、val/test 和既有 8140 进程未修改。
- rollback: 停止 PID 1875763 即可关闭本次独立查看器。

**原因**

用户希望直接查看包含剪刀与纸同时运动的 OakInk2 轨迹。

**验证**

启动日志打印 `viewer=http://127.0.0.1:8141`，端口监听检查通过；本次不改变数据或代码。

## 2026-09-14 17:36:10 +0000 — V1.4 GRAB 全量 30 Hz 双手中间导出启动

- activity_id: ACT-20260914-173610-OICM-GRAB-STAGE4-FULL
- timestamp: 2026-09-14 17:36:10 +0000
- modification_version: V1.4.2
- operation_category: [data, operation]
- primary_task_mode: run-only/operation
- change_level: L3（全量数据处理长任务）
- approval: user-approved
- approval_basis: 用户要求继续直到开始训练，并确认 GRAB 全量、OakInk2 单物体段、ARCTIC 全量。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 使用 NAS GRAB 原始数据，以 `ds_rate=4` 从 120 Hz 生成 30 Hz 双手 MANO 中间 cache；不覆盖旧 cache。后续仍需 Inspire geometric retarget、OakInk2/ARCTIC 接入、KNN finalize 和训练 smoke。
- files: [process/GRAB/stage4_cm.py](../../../../../process/GRAB/stage4_cm.py)、[V1.4 plan](../plan/V1.4.md)
- run_id: grab_stage4_full_20260914T173610Z
- run_status: RUNNING
- command: `.../graspenv/bin/python -u -m process.GRAB.stage4_cm --grab-root /mnt/ugreen_nas/storage/Ref2Dex_storage/GRAB/data/GRAB --mano-path dataset/arctic/data/body_models/mano --output-root data/processed_data/oicm_v1_4_raw/grab_mano_30hz --side both --ds-rate 4 --num-obj-points 4096 --device cuda --save-compressed`
- output: [GRAB intermediate output](../../../../../data/processed_data/oicm_v1_4_raw/grab_mano_30hz)（RUNNING/PENDING）
- validation: 单序列 `s1/cup_lift` smoke 已通过，934 帧，左右 hand 文件均生成；全量任务当前已处理 `s1` 的前若干序列且进程持续运行。
- conclusion: INCONCLUSIVE（中间导出运行中；不代表 geometric retarget 或训练效果）。
- protected_boundary: 原始 NAS、旧 cache、split、val/test、配置和 checkpoint 未修改。
- rollback: 停止 run_id 对应进程并删除独立中间输出；不触及旧产物。

**原因**

为 V1.4 正式全量 pipeline 先生成 30 Hz 双手 MANO/物体中间数据，供后续 Inspire geometric 和 KNN producer 使用。

**验证**

启动日志包含 `stage4` 输出，已验证输入 NAS 路径、MANO 模型和 CUDA 可用；终态待进程结束后补写。

## 2026-09-14 17:39:42 +0000 — V1.4 ARCTIC 全量 30 Hz 双手中间导出启动

- activity_id: ACT-20260914-173942-OICM-ARCTIC-STAGE4-FULL
- timestamp: 2026-09-14 17:39:42 +0000
- modification_version: V1.4.2
- operation_category: [data, operation]
- primary_task_mode: run-only/operation
- change_level: L3（全量数据处理长任务）
- approval: user-approved
- approval_basis: 用户要求继续直到开始训练，并确认 ARCTIC 全量。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 使用 NAS ARCTIC 原始序列，`ds_rate=1`（源数据 30 Hz）生成双手 MANO/铰链物体中间 cache；不做多物体裁剪，不覆盖旧 cache。后续需 Inspire geometric retarget、KNN finalize 和训练 smoke。
- files: [process/ARCTIC/stage4_cm.py](../../../../../process/ARCTIC/stage4_cm.py)、[V1.4 plan](../plan/V1.4.md)
- run_id: arctic_stage4_full_20260914T173942Z
- run_status: RUNNING
- command: `.../graspenv/bin/python -u -m process.ARCTIC.stage4_cm --output-root data/processed_data/oicm_v1_4_raw/arctic_mano_30hz --side both --ds-rate 1 --num-obj-points 4096 --device cuda:0 --save-compressed`
- output: [ARCTIC intermediate output](../../../../../data/processed_data/oicm_v1_4_raw/arctic_mano_30hz)（RUNNING/PENDING）
- validation: 301 个原始 `.mano.npy` 已盘点；ARCTIC stage4 进程已启动并完成 MANO 模型加载。
- conclusion: INCONCLUSIVE（中间导出运行中；不代表 geometric retarget 或训练效果）。
- protected_boundary: 原始 NAS、旧 cache、split、val/test、配置和 checkpoint 未修改。
- rollback: 停止 run_id 对应进程并删除独立中间输出；不触及旧产物。

**原因**

为 V1.4 正式全量 pipeline 先生成 ARCTIC 30 Hz 双手 MANO、物体整体位姿和铰链字段，供后续 geometric producer 使用。

**验证**

启动日志已打印 `[Preprocessor] Loading MANO models...`；终态待进程结束后补写，链接以运行状态标记 PENDING。

## 2026-09-14 17:48:43 +0000 — OakInk2 单物体段索引生成

- activity_id: ACT-20260914-174843-OICM-OAK-SINGLE-INDEX
- timestamp: 2026-09-14 17:48:43 +0000
- modification_version: V1.4.2
- operation_category: [code, data]
- primary_task_mode: change
- change_level: L2（新增正式 train 候选索引，不改 val/test）
- approval: user-approved
- approval_basis: 用户明确要求 OakInk2 只使用单物体段。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 新增单物体 primitive 索引生成器，扫描 627 个 OakInk2 program_info，按官方部件树根实例筛选；未生成手/物体几何 cache，未改 val/test。
- files: [src/task/ObjectInteractionCm/tools/data/build_oakink2_single_object_index.py](../../tools/data/build_oakink2_single_object_index.py)
- run_id: oakink2_single_object_index_20260914T174843Z
- run_status: COMPLETED
- command: `.../graspenv/bin/python -m src.task.ObjectInteractionCm.tools.data.build_oakink2_single_object_index --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --output data/processed_data/oicm_v1_4_raw/oakink2_single_object_segments/index.json`
- output: [OakInk2 single-object index](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_single_object_segments/index.json)
- validation: 输出 `segment_count=2177`、`excluded_multi_object_segments=663`；脚本 `py_compile` 通过。663 包含 658 个多物体段及 5 个无对象段。
- conclusion: SUPPORTED（语义筛选索引生成）；几何重定向、2 cm 帧裁剪和训练效果尚未评估。
- protected_boundary: OakInk2 原始 annotation/资产、现有 pilot、旧 cache、val/test、训练配置和 checkpoint 未修改。
- rollback: 删除新增脚本及独立 index/output 即可回滚，不触及原始数据。

**原因**

为后续全量 producer 固定 OakInk2 的正式 train 候选边界，避免将部件 ID 或多物体 primitive 混入单物体训练样本。

**验证**

全量扫描 627 个 program_info 文件并沿 object_part_tree 追溯根实例；`segment_count=2177` 与此前统计一致。活动链接审计待本条追加后执行。

## 2026-09-14 19:14:06 +0000 — V1.4 GRAB+ARCTIC 几何重定向训练启动

- activity_id: ACT-20260914-191406-OICM-TRAIN-START
- timestamp: 2026-09-14 19:14:06 +0000
- modification_version: V1.4.2
- operation_category: [data, operation, experiment]
- primary_task_mode: run-only/operation
- change_level: L3（全量数据导出与长时训练）
- approval: user-approved
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- run_id: object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406
- run_status: RUNNING
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_arctic_inspire_geometric_v1_4.yaml --set train.distributed.enable=false --set data.num_workers=4 --set data.persistent_workers=true --device cuda`
- scope: ObjectInteractionCm V1.4 full GRAB + ARCTIC geometry training run.
- output: [run directory](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406)
- evidence: loader smoke `train_rows=402235,val_rows=26531,test_rows=24498`; first logged step=100, loss=0.0683698, obj_flow_epe=34.0439mm.
- data: full GRAB=1335 geometry sequences, full ARCTIC=301 geometry sequences, bilateral merged hand points=3076, effective FPS=30.
- max_steps: 1508382 = ceil(402235/32)*120.
- protected_boundary: original GRAB val/test split unchanged; OakInk2 single-object index has 2177 candidates but its full MANO reconstruction/geometry producer is not yet part of this run.
- conclusion: SUPPORTED（工程 smoke 与训练已启动；科研效果尚未结论）。

**原因**

按用户确认的 30 Hz、双手 Inspire 几何重定向范围启动训练，并按最终 train 行数保持 120 个 epoch。

**验证**

训练目录已生成 `run_manifest.json`、`train.log`、`metrics.jsonl`；首个有效日志 step=100。OakInk2 未静默混入，待其 producer 完成后再单独扩展索引。

## 2026-09-15 01:17:45 +0000 — OI-Cm 训练进度与剩余时间诊断

- activity_id: ACT-20260915-011745-OICM-TRAIN-STATUS
- timestamp: 2026-09-15 01:17:45 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic]
- primary_task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问当前训练状态和预计完成时间；仅查询已有进程、配置、指标并追加诊断记录。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 查询既有 run 的进程、step/epoch、验证指标、checkpoint 与 ETA；仅追加本 activity，不修改代码、配置、数据或运行。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- run_id: object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406
- run_status: RUNNING
- command: `ps -eo pid,etime,pcpu,pmem,args`；`nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader`；Python 标准库读取 metrics.jsonl、config.json、metadata.json 和 index.json，使用最近 5 个完整 epoch 的 wall time 估算 ETA。
- last_step: 175900
- last_epoch: 14
- best_metric: val/obj/flow_epe_mm=6.81080321，epoch=13，step=163410
- evidence: PID 1966387 存活；最近 step_ms 约 114.786；GPU0 利用率 95%；最近完整 epoch 均时 1523.011s，含验证及保存间隔，预计剩余 44.85h；ETA 以运行负载不变为前提。
- output: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406)
- manifest: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/run_manifest.json)
- metrics: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/metrics.jsonl)
- train_log: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/train.log)
- best_checkpoint: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/checkpoints/best.pt)
- latest_checkpoint: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406/checkpoints/latest.pt)
- plan_reference: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)
- scope_discrepancy: 已核对 index.train 只有 Inspire 源（GRAB 1068、ARCTIC 301），没有 MANO 原手型或 OakInk2；运行快照 hand_stream_mode=decoder，与 final plan 的 MANO/Inspire 混训及 unique-KNN stream 约束不符。本条不把前次实际范围变化解释为已获用户批准。
- conclusion: SUPPORTED（进程持续推进和日志状态）；INCONCLUSIVE（原定混训方案尚未实现，现有 val 值不能证明其效果）。
- protected_boundary: 进程、checkpoint、cache、split、config 和模型代码均未修改。
- rollback: 本次只有追加的诊断条目；无运行或数据回滚操作。

**原因**

响应用户状态与训练完成时间查询；区分当前运行 ETA 与原计划完成状态。

**验证**

比对存活进程、连续更新的 metrics/train.log、既有 checkpoint 和配置快照。验证 EPE 从 epoch 1 的 8.8820mm 至 epoch 13 的 6.8108mm，仅为当前数据配置下的观测。活动链接审计在追加后执行。

**规范反馈**

本次诊断无格式、路径或审批阻碍。发现前次启动范围与 final plan 不一致，已向用户说明；本次不改规范、不重写历史审批记录。

## 2026-09-15 01:25:00 +0000 — 停止旧训练并固化 NAS cache

- activity_id: ACT-20260915-012500-OICM-STOP-OLD-TRAIN-CACHE
- timestamp: 2026-09-15 01:25:00 +0000
- modification_version: V1.4.2
- operation_category: [operation, data]
- primary_task_mode: run-only/operation
- change_level: L3（停止长任务并固化全量 cache）
- approval: user-approved
- approval_basis: 用户明确要求停止当前训练、完整 cache 写入 NAS，仅使用 ARCTIC 和 GRAB 后重新训练。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 终止 run_id=object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406；核验 NAS 上 GRAB/ARCTIC 双手几何 cache、索引和 manifest；未复制数组到本地，未清理旧输出。
- run_id: object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406
- run_status: STOPPED
- command: `kill -TERM 1966387`；NAS cache shape validation；`du -sh`。
- output: [旧训练目录](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260914_191406)
- cache_manifest: [NAS cache manifest](../../../../../data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json)
- evidence: 训练主进程及 worker 均已退出；GRAB=1335、ARCTIC=301、总帧=624537、cache=110105920657 bytes；所有序列 shape 校验通过，bad_count=0。
- conclusion: SUPPORTED（用户要求的停止和 NAS cache 完整性核验完成）。

**原因**

当前训练未包含 OakInk2，且用户要求先停止并只固化 GRAB/ARCTIC 的完整 cache 后重新启动。

**验证**

逐序列检查 `obj_points_pool_world=[T,4096,3]`、`hand_points_world=[T,3076,3]`、pose/raw frame 对齐；索引仍位于 NAS 并保持 GRAB val/test 成员。

## 2026-09-15 02:29:24 +0000 — NAS 完整 cache 核验后重新启动训练

- activity_id: ACT-20260915-022924-OICM-RETRAIN-GRAB-ARCTIC
- timestamp: 2026-09-15 02:29:24 +0000
- modification_version: V1.4.2
- operation_category: [data, operation, experiment]
- primary_task_mode: run-only/operation
- change_level: L3（NAS 全量 cache 与长时训练）
- approval: user-approved
- approval_basis: 用户明确要求停止旧训练、只导出 ARCTIC/GRAB 到 NAS 并重新训练。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 使用 NAS 上已固化的 GRAB/ARCTIC 双手 Inspire geometric cache；不复制大数组到本地；保持 GRAB val/test split。
- run_id: object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758
- run_status: RUNNING
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_arctic_inspire_geometric_v1_4.yaml --set train.distributed.enable=false --set data.num_workers=4 --set data.persistent_workers=true --device cuda`
- output: [run directory](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758)
- manifest: [run manifest](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/run_manifest.json)
- metrics: [metrics](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/metrics.jsonl)
- train_log: [train log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/train.log)
- cache_manifest: [NAS cache manifest](../../../../../data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json)
- evidence: NAS cache 1636 sequences/624537 frames/110105920657 bytes；shape validation bad_count=0；loader smoke train/val/test=402235/26531/24498；首个有效日志 step=400。
- max_steps: 1508382；epochs=120；no checkpoint resume。
- conclusion: SUPPORTED（cache、loader 和训练启动工程证据）；科研效果尚未结论。

**原因**

按用户要求停止未符合范围的旧运行，核验完整 GRAB/ARCTIC cache 后从 NAS 路径重新启动独立训练。

**验证**

逐序列检查对象池 `[T,4096,3]`、双手合并 Inspire 点 `[T,3076,3]`、pose/raw 对齐；loader smoke 取样成功；step 100/200/300/400 日志连续产生。

## 2026-09-15 11:17:00 +0800 — GPU 离线 KNN 生成启动

- activity_id: ACT-20260915-111700-OICM-OFFLINE-KNN-START
- timestamp: 2026-09-15 11:17:00 +0800
- modification_version: V1.4.2
- operation_category: [data, operation]
- primary_task_mode: run-only/operation
- change_level: L3（全量 NAS cache 写入与 GPU 长任务）
- approval: user-approved
- approval_basis: 用户明确要求开始用 GPU 生成离线 KNN 文件。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 为 NAS 上 GRAB 1335 与 ARCTIC 301 条双手 Inspire geometry 生成 KNN=32 索引及 2 cm mask；复用已有 hand_points_world/hand_normals_world 为 KNN stream，不复制大数组。
- run_id: oicm_v1_4_offline_knn_gpu1_20260915_111700
- run_status: RUNNING
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.build_bilateral_offline_knn --roots /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2 /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/arctic_inspire_bilateral_v2 --device cuda:1 --obj-chunk 512`
- output: [NAS cache roots](../../../../../data/processed_data/oicm_v1_4_raw/)
- evidence: GPU1 已占用约 7.5 GiB、利用率约 28%；首批 GRAB 序列已写入，过程使用每条序列 `.partial` 文件并在完成后原子切换。
- files: 每条 geometry 目录新增 `obj_knn_indices.npy`、`obj_candidate_mask_2cm.npy`、`hand_supervision_mask_2cm.npy`、`hand_min_object_distance_m.npy`，并建立 KNN hand 点/法线链接。
- protected_boundary: 当前训练继续使用 GPU0；GRAB/ARCTIC 原始数据、既有 geometry、训练配置和 checkpoint 未修改。
- rollback: 停止该进程并删除本次生成的 KNN 文件/链接即可回滚；不触及原始 geometry。
- conclusion: INCONCLUSIVE（正在生成；完成后需全量计数、shape、索引范围和 loader unique-KNN smoke）。

**原因**

用户要求为当前 GRAB/ARCTIC NAS cache 生成 GPU 离线 KNN 文件，以便后续切换到 unique-KNN stream 或进行独立验证。

**验证**

脚本 `py_compile` 通过；GPU1 进程存活并已生成首批完整序列；首次临时文件命名问题已修正，当前任务从缺失文件序列开始安全补齐。

## 2026-09-15 03:18:24 +0000 — 离线 KNN 生成器修正后重启

- activity_id: ACT-20260915-031824-OICM-OFFLINE-KNN-RESTART
- timestamp: 2026-09-15 03:18:24 +0000
- modification_version: V1.4.2
- operation_category: [code, data, operation]
- primary_task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求 GPU 生成离线 KNN；运行中发现派生最小距离字段需使用全 hand stream 全局最小值，已修正后重启。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: GPU1 重新生成 GRAB/ARCTIC 的离线 KNN；不修改训练进程和原始 geometry；已生成文件按序列原子提交。
- run_id: oicm_v1_4_offline_knn_gpu1_20260915_031824
- run_status: RUNNING
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.build_bilateral_offline_knn --roots /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2 /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/arctic_inspire_bilateral_v2 --device cuda:1 --obj-chunk 512`
- output: [NAS KNN cache](../../../../../data/processed_data/oicm_v1_4_raw/)
- evidence: GPU1 运行中；已完成若干 GRAB 序列；每条序列新增 KNN=32 索引、物体/手 2 cm mask、全局最小距离和 KNN hand symlink。
- protected_boundary: 当前训练 GPU0 及其输出、原始 geometry、索引、split 未修改。
- rollback: 停止 KNN 进程并删除本次 KNN 派生文件/链接即可回滚；不触及原始 geometry。
- conclusion: INCONCLUSIVE（仍在生成）。

**原因**

修正首批版本的全局最小距离计算，并继续用户要求的 GPU 离线 KNN 导出。

**验证**

`py_compile` 通过；修正后的进程已在 GPU1 启动，训练 GPU0 仍存活。前次临时文件已清理，已完成序列会检查 `offline_knn_min_global` 标记后决定跳过或重算。

## 2026-09-15 03:31:54 +0000 — GPU 离线 KNN 进度与剩余时间诊断

- activity_id: ACT-20260915-033154-OICM-KNN-ETA
- timestamp: 2026-09-15 03:31:54 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户询问剩余时间；只读查询已有进程与 cache 完成标记，并记录证据。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 诊断现有 GPU1 KNN 任务并追加 activity；校正上一启动条目的时区误写（北京时间 11:17 对应 UTC 03:17），保留原 activity_id/run_id；不改运行、代码、配置或数据。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- run_id: oicm_v1_4_offline_knn_gpu1_20260915_031824
- run_status: RUNNING
- command: `ps -p 2422452 -o pid,etime,lstart,args`；`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`；graspenv Python 使用 `Path.glob('**/geometry/manifest.json')` 扫描两根目录，按 `offline_knn_min_global` 和六个派生文件/链接存在性统计完成量，以 `np.load(source_frame_id.npy, mmap_mode='r')` 读取帧数，以 `/proc/2422452/stat` 和 `/proc/uptime` 计算进程运行时间。
- output: [data/processed_data/oicm_v1_4_raw](../../../../../data/processed_data/oicm_v1_4_raw/)
- cache_manifest: [data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json)
- completed_manifest_example: [data/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2/s10/torussmall_inspect_1/geometry/manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2/s10/torussmall_inspect_1/geometry/manifest.json)
- evidence: 扫描于 UTC 03:30:23 开始，持续数秒；PID 2422452 从 UTC 03:16:55 运行。GRAB 已完成 329/1335 序列、108800/406264 帧；ARCTIC 已完成 0/301 序列、0/218273 帧；合计 108800/624537 帧（17.42%），manifest 读取错误 0。进程 stdout 持续推进至第 330 条附近。
- eta_basis: 本次进程完成 107066 帧（排除启动前已完成的 1734 帧），启动至采样约 807.73s，平均 132.55 帧/s；近 300s 约 134.49 帧/s；剩余约 515737 帧，线性估计约 65 分钟，考虑 NAS/GPU 波动向用户报告剩余 1–1.5 小时。扫描非原子快照，短窗口速率存在几秒采样误差。
- scope_limit: ETA 仅适用于当前已有 3076 点双手 Inspire geometry 的 KNN 生成，不代表 MANO/Inspire 完整混训导出或训练完成时间。
- protected_boundary: 未修改代码、配置、cache、split、训练/KNN 进程或 checkpoint。
- rollback: 本次仅 activity 文档变更；可移除本条诊断并还原上述时区文字，不涉及运行或数据回滚。
- conclusion: SUPPORTED（完成标记计数和进程推进证据）；INCONCLUSIVE（ETA 为负载条件下的外推，尚未执行全量 KNN 正确性/loader 验证，也不形成科研效果结论）。

**原因**

用实际完成帧数估计剩余时间；前次仅数索引文件会混入修正前产物，本次统一使用修正后的完整标记。

**验证**

比对进程、stdout、逐序列 manifest 与文件存在性；完成量及速度如上。追加后对本 activity 文档执行 `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log src/task/ObjectInteractionCm/docs/logs/activity_log.md --worktree --scope-prefix src/task/ObjectInteractionCm/docs/logs/activity_log.md --check-links`；审计范围只覆盖本日志，不覆盖既有 dirty 代码。

**规范反馈**

既有启动条目将北京时间标成 UTC，导致按 timestamp 选最新记录的审计会误选；本次只纠正该条时区并保留标识。无须修改公共规范，无审批阻碍。

## 2026-09-15 04:59:10 +0000 — OakInk2 主动作工具帧切分启动

- activity_id: ACT-20260915-045910-OICM-OAK-ACTIVE-TOOL
- timestamp: 2026-09-15 04:59:10 +0000
- modification_version: V1.4.2
- operation_category: [code, data, operation]
- primary_task_mode: run-only/operation
- change_level: L2（OakInk2 多物体选择与帧筛选语义）
- approval: user-approved
- approval_basis: 用户明确确认“只保留主动操作的工具”，并要求在运动区间内按手物距离切分；剪刀剪纸只保留剪刀。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 新增 OakInk2 active-tool 帧选择器；读取 NAS annotation、object_part_tree、object pose 与既有 Stage3 hand_to_obj_min_dist；输出独立 NAS 索引，不复制 Stage3 数组，不修改 GRAB/ARCTIC、既有 OakInk2 cache、split 或训练。
- files: [src/task/ObjectInteractionCm/tools/data/split_oakink2_active_tool.py](../tools/data/split_oakink2_active_tool.py)
- run_id: oakink2_active_tool_segments_v1_20260915_045910
- run_status: RUNNING
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --stage3-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/oakink2_object_centered_v1 --output /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1`
- output: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1) PENDING
- evidence: 脚本已通过 graspenv `py_compile`；全量进程 PID 2576157 存活并持续读取 NAS，当前运行约 2 分钟，输出尚未 finalize。
- selection_policy: 先按对象位姿计算运动区间（平移阈值 1 mm 或相邻帧旋转超过 1 度），再在该区间取左右手对所选工具/其部件的最小距离严格小于 2 cm；部件按 object_part_tree 合并；无法唯一确定工具写入 uncertain 清单。
- protected_boundary: 既有代码、训练、GRAB/ARCTIC cache、OakInk2 Stage3 数组和 split 未修改。
- rollback: 停止 PID 2576157 并删除独立输出目录；代码删除/恢复入口为本条新增脚本路径。
- conclusion: INCONCLUSIVE（运行中；完成后需检查选段计数、uncertain 清单、帧 ID、2 cm 条件和剪刀样例）。

**原因**

按用户确认的 active-tool-only 规则生成 OakInk2 帧选择索引，供后续 Inspire 导出使用。

**验证**

已执行 `.../graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/tools/data/split_oakink2_active_tool.py`；进程检查 `ps -p 2576157` 显示运行中。终态验证待 PENDING 输出生成。

## 2026-09-15 04:35:34 +0000 — GRAB/ARCTIC 离线 KNN 生成完成

- activity_id: ACT-20260915-043534-OICM-OFFLINE-KNN-COMPLETE
- timestamp: 2026-09-15 04:35:34 +0000
- modification_version: V1.4.2
- operation_category: [data, operation, diagnostic, documentation]
- primary_task_mode: run-only/operation
- change_level: L3（全量 NAS cache 写入的既有获批运行；本条终态核对仅追加记录）
- approval: user-approved
- approval_basis: 用户已明确要求使用 GPU 为当前 GRAB/ARCTIC 双手 Inspire geometry 生成离线 KNN；本条闭合对应运行并核对终态。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2（终态统一记录；运行发生于原 `oyx` 工作树）
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 核对既有 GRAB/ARCTIC 离线 KNN 运行终态；不重算、不修改 cache、训练配置、split 或 checkpoint。
- run_id: oicm_v1_4_offline_knn_gpu1_20260915_031824
- run_status: COMPLETED
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.build_bilateral_offline_knn --roots /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2 /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/arctic_inspire_bilateral_v2 --device cuda:1 --obj-chunk 512`
- output: [data/processed_data/oicm_v1_4_raw](../../../../../data/processed_data/oicm_v1_4_raw/)
- cache_manifest: [data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/cache_manifest.json)
- completed_manifest_example: [data/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2/s7/pyramidmedium_inspect_1/geometry/manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2/s7/pyramidmedium_inspect_1/geometry/manifest.json)
- last_step: N/A（数据处理任务）
- last_epoch: N/A（数据处理任务）
- best_metric: N/A（数据处理任务）
- evidence: 原执行器记录命令于 04:35:34 正常退出，exit code 0；GRAB 1335/1335、ARCTIC 301/301 条序列均存在 `obj_knn_indices.npy` 且 manifest 标记 `offline_knn_min_global=true`，遗留 `.partial` 文件为 0。代表性 GRAB/ARCTIC 数组分别核对为 KNN `[T,4096,32] uint16`、object mask `[T,4096] bool`、hand mask `[T,3076] bool`、min distance `[T] float32`，抽查索引范围小于 3076。
- protected_boundary: 原始 GRAB/ARCTIC、既有 geometry、训练运行、split、配置和 checkpoint 未修改。
- rollback: 本条仅补充终态记录；如需回滚数据任务，按启动条目删除四类 KNN 派生文件和对应链接，不触及原 geometry。
- conclusion: SUPPORTED（任务自然完成、全量完成标记和文件计数）；INCONCLUSIVE（尚未对全部数组逐值校验，也未完成 unique-KNN loader 全量训练）。

**原因**

旧会话结束前运行已完成，但 activity 仍停留在 `RUNNING`；本条将唯一时间线闭合为实际终态。

**验证**

只读比对旧会话执行事件、1636 个逐序列 manifest、派生文件计数、`.partial` 文件和两源代表性数组的 shape、dtype、索引范围。

**规范反馈**

该历史数据处理运行没有独立 `run_manifest.json`，仅更新逐序列 manifest 并复用总 cache manifest；本条保留这一追溯缺口，不事后伪造运行清单。后续 producer 应在启动时生成独立 run manifest。

## 2026-09-15 06:07:28 +0000 — OakInk2 主动作工具帧切分完成

- activity_id: ACT-20260915-060728-OICM-OAK-ACTIVE-TOOL-COMPLETE
- timestamp: 2026-09-15 06:07:28 +0000
- modification_version: V1.4.2
- operation_category: [data, operation, diagnostic, documentation]
- primary_task_mode: run-only/operation
- change_level: L2（主动工具、运动区间和 2 cm 帧选择语义）
- approval: user-approved
- approval_basis: 用户确认多物体段只保留主动操作工具，剪刀剪纸只保留剪刀，并要求不确定情况显式列出。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2（终态统一记录；运行发生于原 `oyx` 工作树）
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 核对既有 OakInk2 帧选择运行及输出；不把选择结果接入正式 cache、index、scale 或训练。
- run_id: oakink2_active_tool_segments_v1_20260915_045910
- run_status: COMPLETED
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --stage3-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/oakink2_object_centered_v1 --output /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1`
- output: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/)
- manifest: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/manifest.json)
- index: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json)
- last_step: N/A（数据处理任务）
- last_epoch: N/A（数据处理任务）
- best_metric: N/A（数据处理任务）
- evidence: 原执行器记录命令于 06:07:28 正常退出，exit code 0；输出含 2596 个有效选择段、2170451 个原始 frame ID、30 个显式 uncertain 段，其中 25 个主动工具不唯一、5 个没有对象标注；另有 212 个选中对象无显著运动、2 个运动段没有严格小于 2 cm 的帧，按既定筛选条件计数后排除。
- protected_boundary: OakInk2 原始 annotation、Stage3 数组、GRAB/ARCTIC cache、正式 split、scale、配置和 checkpoint 未修改。
- rollback: 删除该独立选择目录和本条终态记录即可；不触及源数据或其他 cache。
- conclusion: SUPPORTED（切分程序自然完成、manifest/index 可读且统计一致）；INCONCLUSIVE（30 个 uncertain 段尚待用户决定，结果尚未进入正式 producer 或训练）。

**原因**

旧会话中的帧选择进程已经自然结束，需要闭合运行状态并将待确认样本显式交接。

**验证**

只读解析 manifest/index，汇总 selected frame ID、uncertain reason 和对象组合；对照脚本确认运动阈值 1 mm/1 度、距离严格小于 2 cm、左右手距离 union 及部件树归并逻辑。

**规范反馈**

该运行生成 cache manifest 和 index，但没有独立 `run_manifest.json`；212 个静止段和 2 个无 2 cm 帧段只保留聚合计数，没有逐段 reject ledger。后续正式 producer 应补充 run manifest 和逐段拒绝清单，避免只靠计数追溯。

## 2026-09-15 08:17:10 +0000 — GRAB/ARCTIC Inspire-only 训练随旧会话中止

- activity_id: ACT-20260915-081710-OICM-GRAB-ARCTIC-TRAIN-STOPPED
- timestamp: 2026-09-15 08:17:10 +0000
- modification_version: V1.4.2
- operation_category: [experiment, operation, diagnostic, documentation]
- primary_task_mode: run-only/operation
- change_level: L3（长时训练终态与 checkpoint 解释）
- approval: user-approved（原训练范围）；auto（只读终态闭合）
- approval_basis: 用户已批准使用 NAS 上 GRAB/ARCTIC Inspire geometry cache 启动训练；本条仅根据执行器和产物证据记录非正常中止，不恢复 checkpoint。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2（终态统一记录；运行发生于原 `oyx` 工作树）
- base_commit: 85e70edffa85d8d1698a3e8adb22e111033cb892
- worktree_dirty: true
- scope: 闭合既有 Inspire-only 训练终态；保留运行目录和 checkpoint，不恢复、不删除、不把 TACO/OakInk2 静默混入旧运行。
- run_id: object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758
- run_status: STOPPED
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.train --config src/task/ObjectInteractionCm/configs/active/grab_arctic_inspire_geometric_v1_4.yaml --set train.distributed.enable=false --set data.num_workers=4 --set data.persistent_workers=true --device cuda`
- output: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/)
- manifest: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/run_manifest.json](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/run_manifest.json)
- metrics: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/metrics.jsonl](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/metrics.jsonl)
- train_log: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/train.log](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/train.log)
- best_checkpoint: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/checkpoints/best.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/checkpoints/best.pt)
- latest_checkpoint: [outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/checkpoints/latest.pt](../../../../../outputs/objectinteractioncm/object_interaction_cm_grab_arctic_inspire_geometric_v1_4_20260915_022758/checkpoints/latest.pt)
- last_step: 157400（日志最后有效训练 step）
- last_epoch: 13（未完成该 epoch）
- best_metric: val/obj/flow_epe_mm=6.849145640342794，epoch=12，step=150840
- latest_checkpoint_state: epoch=12，step=150840；若恢复会从最近完整 checkpoint 开始，不能声称无损续接 step 157400。
- exit_reason: 旧会话统一执行器结束时命令状态为 failed、exit code -1；训练日志末尾无 Python traceback、NaN 或 CUDA OOM，进程随后不存在，因此按外部中止记为 `STOPPED`，不归因于实现错误。
- protected_boundary: 运行目录、两个 checkpoint、metrics、train log、NAS cache、split 和配置均保留；没有覆盖或恢复。
- rollback: 本条仅补充终态记录；运行产物保持原位，可删除本条恢复文档状态，但不能恢复已丢失的 epoch 13 后半段进程状态。
- conclusion: INCONCLUSIVE（只完成约 10.4% 计划 step，且仅含 Inspire source；中间 validation 不构成完成实验结论）。

**原因**

旧会话结束时训练命令被执行器终止，原 activity 仍显示 `RUNNING`；需要按现有 checkpoint 和日志闭合终态，避免后续误判为存活训练。

**验证**

比对进程表、GPU 状态、旧会话 `item_completed` 事件、metrics/train log 尾部和 best/latest checkpoint 元数据。最佳与最近完整 checkpoint 都是 epoch 12、step 150840；训练日志继续到 step 157400 后中止。

## 2026-09-15 08:32:08 +0000 — 会话续接与 TACO NAS 几何数据盘点

- activity_id: ACT-20260915-083208-OICM-TACO-INVENTORY
- timestamp: 2026-09-15 08:32:08 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求浏览并续接指定会话，随后明确后续新增改动统一放入 `feature/objectinteractioncmv2-v1.0.2`；本条只读盘点 TACO 数据并记录状态，不改变数据语义。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 只读核对 NAS TACO-Instructions 的几何训练输入、官方列表、既有全量审计和当前分支；不修改 TACO 数据、split、cache、指导、plan、配置或模型，不启动 producer/训练。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- dataset_list: [dataset/TACO-Instructions/data_lists/v1_overall_data_sequences.txt](../../../../../dataset/TACO-Instructions/data_lists/v1_overall_data_sequences.txt)
- official_split: [dataset/TACO-Instructions/data_lists/v1_overall_data_train_test_split.txt](../../../../../dataset/TACO-Instructions/data_lists/v1_overall_data_train_test_split.txt)
- prior_audit_summary: [data/outputs/statics_exec/20260703_115622_TACO-Instructions_canonical_streaming_full_fresh_verified/ledgers/manifest.summary.json](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/statics_exec/20260703_115622_TACO-Instructions_canonical_streaming_full_fresh_verified/ledgers/manifest.summary.json)
- data_root: `data/TACO-Instructions/data`（NAS 实际路径 `/mnt/ugreen_nas/storage/Ref2Dex_storage/TACO-Instructions/data`）
- evidence: Hand_Poses 与 Object_Poses 各有 2317 条序列且交集为 2317；覆盖 151 个 `<tool, action, object>` triplet 和 206 个 `_cm.obj` 物体模型，与官方 Whole Dataset V1 数量一致。既有全量几何审计记录 2317 success、0 failed/blocked。官方 split 为 train 953、test_1 225、test_2 234、test_3 354、test_4 551。代表性序列左右手各 259 帧，逐帧字段为 `hand_pose[48]`、`hand_trans[3]`，tool/target pose 均为 `[259,4,4] float32` 且有限值。
- modality_boundary: NAS 当前没有四类 RGB/depth video 目录，但 OI-Cm 几何 producer 所需的双手 MANO 参数、tool/target SE(3) 和物体模型均存在；是否需要视频不属于当前几何训练范围。
- pending_semantics: TACO 使用官方 train 953 还是全部 2317；每条只取 `tool_*` 还是 tool/target 各自产生单物体样本；是否同时保留 native MANO 与 Inspire geometric 两个 source。上述选择会改变 split、cache/schema 和训练预算，需进入新的 final guidance/plan 后才能实现。
- protected_boundary: 当前 GRAB/ARCTIC/OakInk2 数据、V1.4 cache、旧训练 checkpoint、现有 ObjectInteractionCmv2 分支提交和所有用户未提交改动均未修改。
- rollback: 删除本条及前三条终态补录即可回滚本轮文档变更；不涉及数据或运行产物。
- conclusion: SUPPORTED（TACO Whole Dataset V1 几何标注在 NAS 上完整可用）；INCONCLUSIVE（尚未确认接入语义，也未实现 TACO producer、cache 或训练）。

**原因**

完成指定会话的状态交接，并在当前统一分支上留下可审计的终态和 TACO 可用性证据。

**验证**

只读比对官方 sequence/split 列表、NAS Hand/Object 目录交集、物体模型计数、代表性 pickle/NPY shape 与 finite、既有 2317 条全量审计结果；核对当前分支、HEAD、dirty worktree 和旧会话执行终态。

**规范反馈**

会话运行结束后、终态补录前发生分支切换，导致运行 provenance 与日志承载分支不同；本条同时保留真实运行 base commit 和当前统一记录分支，不改写历史事实。KNN 与 OakInk2 历史运行缺少独立 run manifest，后续正式 producer 需在启动时补齐。

## 2026-09-15 08:44:58 +0000 — OakInk2 30 条 uncertain 逐条语义诊断

- activity_id: ACT-20260915-084458-OICM-OAK-UNCERTAIN-DIAG
- timestamp: 2026-09-15 08:44:58 +0000
- modification_version: V1.4.2
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户要求先说明并解决 OakInk2 的 30 条 uncertain；本条仅诊断和提出待确认映射，不修改数据选择语义。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 逐条关联 uncertain index、官方 program_info、desc_info、task_target、object_part_tree、obj_desc 和原始 annotation 对象清单；不修改 index、split、cache、producer、配置、模型或 checkpoint。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- index: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json)
- manifest: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/manifest.json)
- selector: [src/task/ObjectInteractionCm/tools/data/split_oakink2_active_tool.py](../../tools/data/split_oakink2_active_tool.py)
- evidence: 30 条中 25 条为 `active_tool_not_unique`、5 条为 `no_objects`。25 条前者中，24 条是真实双主动物体 primitive：21 条的左右手对象列表可直接区分，3 条（bread+donut、bowl+plate、tripod+asbestos mesh）存在手侧列表重叠但自然语言动作仍明确同时操作两个实例；剩余 1 条 `close_laptop_lid` 将 gamecontroller 与 laptop lid 一并列入左手对象，动作描述和相邻 primitive 均指向 laptop lid。5 条空列表分别是打开 laptop lid、关闭 laptop lid、拉出 drawer、打开 book、把 asbestos mesh 放上 tripod，场景 annotation 中均存在对应对象/部件。
- proposed_resolution: 24 条双主动物体段按物体根拆成两个共享原时间窗的单物体候选；laptop/gamecontroller 条目只取 `O02@0053@00001` laptop lid；5 条空列表依次恢复为 laptop lid `O02@0053@00001`、laptop lid `O02@0053@00001`、moving drawer `O02@0019@00002`、book 两部件 `O02@0060@00001/00002`（同一实例根）、asbestos mesh `O02@0205@00001`。因此 30 条可先恢复为 54 个 object-segment 候选，再分别执行既有 motion-first 与严格小于 2 cm 过滤；54 不是最终保留数。
- protected_boundary: uncertain index 和 OakInk2 原始 annotation/Stage3、GRAB/ARCTIC cache、split、训练配置、checkpoint、现有用户未提交改动均未修改。
- rollback: 删除本条活动记录即可；无数据或运行产物需要回滚。
- conclusion: SUPPORTED（30 条的标注类型、动作语义和候选对象均可由现有文件复核）；INCONCLUSIVE（候选映射尚未获用户确认，也尚未重新运行运动/距离过滤）。

**原因**

把自动选择器的拒绝原因还原为可决策的具体动作，区分真实双主动物体、过宽对象列表和空对象列表，避免把全部 uncertain 静默删除或任意选择一个物体。

**验证**

只读解析 30 条 index 记录，并按 sequence 与 frame range 精确关联官方 program_info/desc_info/task_target；核对 object_part_tree、obj_desc 与 5 条空列表序列的 annotation 对象根。曾启动可选的逐对象 Stage3 深扫，因需要大量解压且不影响语义分类而主动终止；没有写入输出。

## 2026-09-15 09:00:03 +0000 — OakInk2 uncertain 消解实现与全量索引重算启动

- activity_id: ACT-20260915-090003-OICM-OAK-RESOLVE-START
- timestamp: 2026-09-15 09:00:03 +0000
- modification_version: V1.4.3
- operation_category: [code, data, operation, documentation]
- primary_task_mode: change → run-only/operation
- change_level: L2（主动对象选择与 index schema 小版本）+ L3（全量 NAS 数据处理长任务）
- approval: user-approved
- approval_basis: 用户逐条查看 30 条 uncertain 后明确回复“是的，你修改”，确认 24 条双主动物体拆分、1 条过宽列表纠正和 5 条空列表恢复，并确认重新生成 index。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 修改 Task-local OakInk2 active-tool selector，新增定向测试、V1.4 final plan 增补和 V1.4.3 版本指针；启动全量 motion-first/严格小于 2 cm 重算到独立 v1.1 目录。旧 v1 index 和其他数据保持只读。
- files: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[src/task/ObjectInteractionCm/tools/data/split_oakink2_active_tool.py](../../tools/data/split_oakink2_active_tool.py)、[src/task/ObjectInteractionCm/tests/test_split_oakink2_active_tool.py](../../tests/test_split_oakink2_active_tool.py)、[src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- run_id: oakink2_active_tool_segments_v1_1_20260915_090003
- run_status: STARTED
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --stage3-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/oakink2_object_centered_v1 --output /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1 --run-id oakink2_active_tool_segments_v1_1_20260915_090003 --modification-version V1.4.3`
- output: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1) PENDING
- manifest: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/manifest.json) PENDING
- run_manifest: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/run_manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/run_manifest.json) PENDING
- implementation: 双主动物体仅在 `bh_main` 且所有根均属于主手对象时按根拆分；唯一工具仍优先，避免把剪刀/纸等被作用对象纳入；6 条标注异常使用精确 sequence/frame-range override；Stage3 距离数组增加序列内只读缓存；schema 从 1.0.0 增至 1.1.0，并生成独立 run manifest。
- preflight_evidence: `py_compile` 通过；Task-local 定向测试 `5 passed`；全量 2840 primitive 轻量解析得到 single=2177、unique tool=380、unique main=253、bilateral split=24、semantic override=6、unresolved=0；旧 30 条对应 54 个 motion/distance 过滤前候选。
- protected_boundary: 旧 `oakink2_active_tool_segments_v1`、原始 OakInk2 annotation/Stage3、GRAB/ARCTIC cache、val/test、模型、训练配置、checkpoint 和其他用户未提交改动均不修改；新 index 尚不接入训练。
- rollback: 停止本次运行并删除独立 `oakink2_active_tool_segments_v1_1` 目录；恢复 selector/test/plan/version/activity 的本次增量。旧 v1 产物不受影响。
- conclusion: INCONCLUSIVE（全量数据处理已获批并准备启动；需等待新 index、manifest 和统计完成）。

**原因**

落实用户确认的 30 条 uncertain 确定性消解，同时保持单物体 OI-Cm 输入、唯一工具优先和既有运动/距离筛选语义。

**验证**

已执行 `python -m py_compile`、`pytest -q src/task/ObjectInteractionCm/tests/test_split_oakink2_active_tool.py`（5 passed）和全量 program_info 轻量语义预检；全量几何筛选终态待运行完成后补录。

## 2026-09-15 10:24:12 +0000 — OakInk2 uncertain 消解与 v1.1 索引完成

- activity_id: ACT-20260915-102412-OICM-OAK-RESOLVE-COMPLETE
- timestamp: 2026-09-15 10:24:12 +0000
- modification_version: V1.4.3
- operation_category: [code, data, operation, diagnostic, documentation]
- primary_task_mode: change → run-only/operation → read-only/diagnostic
- change_level: L2（主动对象选择、显式 override 与 index schema）+ L3（全量 NAS 数据处理）
- approval: user-approved
- approval_basis: 用户确认 30 条 uncertain 的逐条消解方案并明确要求修改；运行范围、独立输出和保护边界见同一 run_id 的启动条目。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 完成 OakInk2 active-tool selector V1.4.3 实现、测试和独立 v1.1 全量 index；核对旧 30 条 uncertain 的逐对象去向，不接入正式 cache、scale 或训练。
- files: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[src/task/ObjectInteractionCm/tools/data/split_oakink2_active_tool.py](../../tools/data/split_oakink2_active_tool.py)、[src/task/ObjectInteractionCm/tests/test_split_oakink2_active_tool.py](../../tests/test_split_oakink2_active_tool.py)、[src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- run_id: oakink2_active_tool_segments_v1_1_20260915_090003
- run_status: COMPLETED
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool --annotation-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview --object-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/data/OakInk-v2-hub --stage3-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/oakink2_object_centered_v1 --output /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1 --run-id oakink2_active_tool_segments_v1_1_20260915_090003 --modification-version V1.4.3`
- output: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/)
- index: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json)
- manifest: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/manifest.json)
- run_manifest: [data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/run_manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/run_manifest.json)
- completed_at: 2026-09-15 09:56:23 +0000
- exit_code: 0
- last_step: N/A（数据处理任务）
- last_epoch: N/A（数据处理任务）
- best_metric: N/A（数据处理任务）
- result: schema `1.1.0`；selected object-segment 2643；selected frame references 2195908；uncertain 0；static-selected candidates 219；no-2cm candidates 2。旧 v1 为 2596 段、2170451 帧引用、30 uncertain，因此净增 47 段和 25457 帧引用。
- uncertain_resolution: 旧 30 段解析为 54 个过滤前候选。24 个双主动物体段产生 48 个候选，其中 44 个通过；6 个显式 semantic override 中 3 个通过。7 个未保留候选均落入既有静止规则：`scene_01/A003/c437...` 的 bowl，三个 open/close laptop lid，以及 `scene_03/A004/3b1e...`、`scene_03/A004/b5fa...`、`scene_03/A007/2bae...` 的 alcohol burner。全局 no-2cm 计数仍为 2，说明新增候选没有因 2 cm 规则额外淘汰。
- validation: 最终进程 exit code 0；run manifest 为 `COMPLETED` 且锁定 627 个 annotation、3 个 Stage3 stats 及 object metadata hash；2643 个 `(sequence, frame_range, selected_root)` key 全部唯一，空 selected object 为 0，frame count 与数组长度全部一致，frame ID 全在 motion bounds 内。旧 uncertain 的 47 个保留候选中 selection reason 为 bilateral split 44、semantic override 3，保存的最小手物距离最大值为 0.0140528 m。
- tests: `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_split_oakink2_active_tool.py src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py src/task/ObjectInteractionCm/tests/test_arctic_landmark_contract.py` → `10 passed in 0.93s`；`py_compile` 与 `git diff --check` 通过。
- protected_boundary: 旧 v1 index、OakInk2 原始 annotation/Stage3、GRAB/ARCTIC cache、现有 val/test、模型、训练配置、checkpoint 和其他用户未提交改动均未修改；新 index 尚未接入训练。
- rollback: 删除独立 v1.1 输出目录，并恢复本次 selector/test/plan/version/activity 增量；旧 v1 仍可直接用于对照。
- conclusion: SUPPORTED（30 条 uncertain 的确定性消解、全量 index 生成和结构验证）；INCONCLUSIVE（尚未生成 OakInk2 正式几何 cache，也未评估训练收益）。

**原因**

闭合已获用户批准的 uncertain 消解实现和全量数据处理，并把语义解析、几何过滤与训练效果结论分开记录。

**验证**

核对进程 exit code、index/manifest/run manifest、v1/v1.1 计数差异、旧 30 条的 54 个候选映射、唯一键、对象/帧字段和定向测试；终态链接审计在本条追加后执行。

**规范反馈**

新版已补齐独立 `run_manifest.json`。当前 producer 对静止和 no-2cm 候选只保存聚合计数，7 个静止候选由新旧 index 与已确认映射反查得到；若后续需要长期逐条审计所有 reject，建议在下一次已确认的 schema 修改中增加 reject ledger，本次不扩大已批准范围。

## 2026-09-15 12:51:41 +0000 — OakInk2 active-tool v1.1 原生轨迹查看器启动

- activity_id: ACT-20260915-125141-OICM-OAK-VIS-RUNNING
- timestamp: 2026-09-15 12:51:41 +0000
- modification_version: V1.4.4
- operation_category: [code, diagnostic, operation, documentation]
- primary_task_mode: change → run-only/operation
- change_level: L1（Task-local 只读查看器、定向测试与独立运行目录）
- approval: user-approved
- approval_basis: 用户提出先可视化检查 OakInk 切分轨迹，并在确认原始双手 MANO、选中/上下文对象、最终保留帧与 primitive 全帧切换、47 条新增和 7 条静止对照方案后明确回复“可以”。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 新增 Task-local OakInk2 active-tool v1.1 交互查看器、实验定义、说明与定向测试；只读重建双手 quaternion MANO，并将 Stage3 对象点按原 annotation 位姿放回 OakInk2 native world。默认查看 v1→v1.1 新增段，可切换双物体拆分、语义修复、静止排除对照和全部有效段。
- files: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCm/docs/README.md](../README.md)、[src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/README.md](../../research/oakink2_segment_visualizer/README.md)、[src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/experiment.yaml](../../research/oakink2_segment_visualizer/experiment.yaml)、[src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/run.py](../../research/oakink2_segment_visualizer/run.py)、[src/task/ObjectInteractionCm/tests/test_oakink2_segment_visualizer.py](../../tests/test_oakink2_segment_visualizer.py)、[src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- run_id: oakink2_segments_v1_1_20260915T125052Z
- run_status: RUNNING
- command: `PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.research.oakink2_segment_visualizer.run --category new --host 127.0.0.1 --port 8142 --output src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z`
- viewer: `http://127.0.0.1:8142`
- output: [src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z](../../research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z/)
- run_manifest: [src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z/run_manifest.json](../../research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z/run_manifest.json)
- viewer_log: [src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z/viewer.log](../../research/oakink2_segment_visualizer/output/oakink2_segments_v1_1_20260915T125052Z/viewer.log)
- result: v1.1 有效段 2643 条；v1→v1.1 新增 47 条，其中双主物体拆分 44 条、语义修复 3 条；静止排除对照 7 条。默认加载 `selected:0075`。服务监听 `127.0.0.1:8142`，HTTP 检查返回 200。
- validation: `python -m py_compile` 通过；Task-local pytest 为 `4 passed in 0.84s`；`--category new --check-only` 与 `--category static --check-only` 均通过。两类 smoke 各抽取 3 个原始 frame ID，左右 MANO shape 均为 `[3, 778, 3]`、finite=true，选中对象均得到 4096 个 native-world 点；分类计数一致。新增样例缺少一个非选中上下文对象的 Stage3 点云，已在 GUI 状态中显式报告，不影响选中对象轨迹。
- protected_boundary: v1/v1.1 index、OakInk2 原始 annotation/Stage3、训练 cache、split、scale、配置、模型、checkpoint、既有运行和其他用户未提交改动均未修改；未批量复制 2643 条轨迹，查看器不写回选择结果。
- rollback: 停止查看器进程并删除本次独立 output；如需撤回实现，只移除 `oakink2_segment_visualizer/`、对应测试、V1.4 plan 第 11 节、README 链接、V1.4.4 指针及本条活动记录。输入数据不受影响。
- conclusion: SUPPORTED（v1.1/v1 差异分类、原生坐标读取重建链路和本地交互服务启动）；INCONCLUSIVE（尚需用户通过可视化判断切分数据质量，且不构成训练收益证据）。

**原因**

在正式接入 OakInk2 cache 与训练前，提供不会改变 index 的逐段人工复核入口，并把最终保留帧与原 primitive 时间窗清楚区分。

**验证**

已核对 schema、v1/v1.1 差集、7 条静止对照、原始 frame ID、双手 MANO finite、选中/上下文对象世界坐标、HTTP 可达性、run manifest 和运行日志；最终链接审计在本条追加后执行。

## 2026-09-15 13:47:05 +0000 — V1.4 GRAB stride=10 跨帧点流查看器启动

- activity_id: ACT-20260915-134705-OICM-GRAB-STRIDE10-VIS
- timestamp: 2026-09-15 13:47:05 +0000
- modification_version: V1.4.5
- operation_category: [code, diagnostic, operation, documentation]
- primary_task_mode: change → run-only/operation
- change_level: L1（Task-local 查看器兼容当前 V1.4 schema 和只读点流统计）
- approval: user-approved
- approval_basis: 用户明确要求复用此前带跳帧选项的 GRAB 可视化，启动 `stride=10` 版本以人工判断跨度是否不合理。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 旧 V1.2.5 index 已不在当前数据盘；最小扩展既有 GRAB Viser，使其在保留旧 v1.1/单手入口兼容的同时，只读加载当前 V1.4 index v1.2 与 bilateral geometry。未来 `Δ=1..10` 继续叠加对应 cache 帧，并新增时间跨度、原始 frame 差和手/物对应点流 median/P95/max；不改变训练 flow。
- files: [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/ObjectInteractionCm/docs/README.md](../README.md)、[src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[src/task/ObjectInteractionCm/visualize_grab.py](../../visualize_grab.py)、[src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py](../../tests/test_visualize_grab_v1_4.py)、[src/task/ObjectInteractionCm/research/grab_stride_visualization/README.md](../../research/grab_stride_visualization/README.md)、[src/task/ObjectInteractionCm/research/grab_stride_visualization/experiment.yaml](../../research/grab_stride_visualization/experiment.yaml)、[src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- run_id: grab_stride10_20260915T134656Z
- run_status: RUNNING
- command: `PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/index.json --split train --sequence grab/s1/airplane_fly_1 --frame 103 --future-delta 10 --point-display both --mesh-display off --host 127.0.0.1 --port 8143 --fps 8 --output src/task/ObjectInteractionCm/research/grab_stride_visualization/output/grab_stride10_20260915T134656Z`
- viewer: `http://127.0.0.1:8143`
- output: [src/task/ObjectInteractionCm/research/grab_stride_visualization/output/grab_stride10_20260915T134656Z](../../research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/)
- config: [src/task/ObjectInteractionCm/research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/config.json](../../research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/config.json)
- run_manifest: [src/task/ObjectInteractionCm/research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/run_manifest.json](../../research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/run_manifest.json)
- viewer_log: [src/task/ObjectInteractionCm/research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/viewer.log](../../research/grab_stride_visualization/output/grab_stride10_20260915T134656Z/viewer.log)
- initial_sample: `grab/s1/airplane_fly_1`，cache frame 103→113，source frame 412→452，30 Hz 时间跨度 0.333 秒，未夹到末帧；当前帧 hand-object 最近距离 0.894 mm，candidate active=true。物体点流 median/P95/max 为 181.93/186.74/188.05 mm，双手点流为 161.63/194.75/202.51 mm。
- trajectory_diagnostic: 该轨迹 279 帧，其中 234 个 current frame 为 2 cm active；对这些 current frame 的 `Δ=10` 双手逐帧 median flow 再统计，median=160.94 mm、P95=301.10 mm、max=694.60 mm。该量级提示 10 stride 可能跨过较大运动甚至接触状态变化，但是否调整训练 stride 等待用户可视化判断，不在本次自动修改。
- validation: `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py` 通过；Task-local pytest `2 passed in 1.04s`；真实 V1.4 `--check-only` 通过；Viser 监听 `127.0.0.1:8143` 且 HTTP 返回 200。
- limitation: 当前 V1.4 Inspire-geometric cache 保存了双手点云但没有 viewer 可读取的 Inspire qpos，因此本次默认关闭手 mesh；点流点云和数值统计完整可用，物体 mesh 可用。
- protected_boundary: GRAB/ARCTIC/OakInk2 cache 和 index、split、flow GT、stride 训练配置、模型、checkpoint、既有 8142 OakInk2 viewer 与其他用户未提交改动均未修改。
- rollback: 停止 8143 查看器并删除本次独立 output；恢复 `visualize_grab.py` 的 v1.2/bilateral 兼容与统计增量、对应测试/研究定义、plan 第 12 节、README 链接、V1.4.5 指针和本条活动记录。数据无需回滚。
- conclusion: SUPPORTED（当前 V1.4 GRAB 双手 cache 的 `t→t+10` 读取、时间跨度、点流统计及交互服务启动）；INCONCLUSIVE（单条轨迹与可视化尚不足以决定全局 stride 上限，等待人工检查和更广泛统计）。

**原因**

在继续 OakInk2 30 Hz 导出前，先用与当前训练 cache 一致的 GRAB 双手数据直观看清最大 stride 对应的真实时间与空间跨度，避免沿用不合适的未来帧范围。

**验证**

已验证旧/新 index schema 兼容、bilateral candidate mask、对应点流统计、真实 source frame 关系、运行清单、HTTP 可达性和服务进程；最终链接审计在本条追加后执行。

## 2026-09-15 14:51:31 +0000 — V1.4 GRAB Inspire 手点云散裂诊断

- activity_id: ACT-20260915-145131-OICM-INSPIRE-DOUBLE-FK-DIAG
- timestamp: 2026-09-15 14:51:31 +0000
- modification_version: V1.4.5
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读数值诊断与活动记录）
- approval: auto
- approval_basis: 用户指出正在运行的 GRAB stride 查看器中手点零散；本次只调查已有 cache 与导出实现，不修复代码或重导数据。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- parent_activity_id: ACT-20260915-134705-OICM-GRAB-STRIDE10-VIS
- scope: 对查看器初始样例 `grab/s1/airplane_fly_1` 的 frame 103/113，将 bilateral 3076 点拆为左右各 1538 点，检查包围盒、点间距和连通分量，并与同帧 Stage-4 MANO 点云及零姿态 Inspire FK 对照；不改变查看器、导出器、cache、index、split、GT 或 stride。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- evidence_cache: [当前 GRAB bilateral geometry manifest](../../../../../data/processed_data/oicm_v1_4_raw/grab_inspire_bilateral_v2/s1/airplane_fly_1/geometry/manifest.json)
- evidence_code: [Stage-4 Inspire 导出器](../../tools/data/retarget_stage4_bilateral_inspire.py)、[Inspire canonical pool 构建](../../research/hand_region_sampling/run.py)、[Inspire FK surface 实现](../../tools/data/build_dexplore_rl_cache.py)
- result: frame 103 单只 Inspire 手 bbox 分别约为 `357×149×402 mm` 和 `350×135×400 mm`，而对应 MANO 手约为 `69×121×179 mm` 和 `109×141×121 mm`。以 20 mm 半径建图时，两只 Inspire 手各仍有 7 个连通分量，MANO 均为 1 个；查看器显示的是完整 3076 点，不存在阈值过滤造成的缺点。全量 manifest 扫描显示 GRAB 1335/1335、ARCTIC 301/301 条 bilateral Inspire 序列都标记为同一 `stage4_mano_to_inspire_position_retarget` producer，因此在重导验证前均应视为受同一结构性错误影响；OakInk2 不属于该脚本的输入范围，本次不据此外推。
- root_cause: `_build_inspire_pool()` 已在零姿态下将每个 visual 的 mesh-local 三角面乘以 `link_transform @ visual.local_transform`，但 `retarget_stage4_bilateral_inspire.py` 把该整手坐标系采样结果直接交给 `_fk_surface()`；后者再次把它当作 visual-local 点乘同类 link FK，造成逐 link 的二次变换和手掌/指节散裂。零姿态复现实验中，正确单次变换 bbox 为 `95×163×246 mm`，当前双重变换变成 `230×314×384 mm`；先逆回 visual-local 再做 FK 后与原零姿态 cloud 最大误差仅 `0.0000019 mm`。
- visualization_factor: `future_delta=10` 还会同时叠加左右手在当前帧与未来帧的四组点；frame 103→113 的左右手质心移动约 113/189 mm，会进一步放大“零散”观感，但不是根因。
- run_status: 既有 `grab_stride10_20260915T134656Z` 查看器仍为 RUNNING；本次未启动新运行、未生成新数据产物。
- protected_boundary: GRAB/ARCTIC/OakInk2 原始数据与 cache、index、split、坐标系、GT、训练配置、模型、checkpoint、两个既有查看器及用户其他未提交改动均未修改。
- rollback: 仅删除本条活动记录即可回滚本次文档增量；诊断过程无数据或运行产物需要回滚。
- conclusion: SUPPORTED（当前样例散裂来自 Stage-4 Inspire surface 的逐 link 双重 FK，而非查看器漏画点）；INVALID_IMPLEMENTATION（现有 GRAB/ARCTIC bilateral Inspire 手几何及其手点流不能作为预期机器人手表面证据）；INCONCLUSIVE（尚未获批修复或重导，OakInk2 不在本次 producer 范围内）。

**原因**

确认用户看到的异常是否来自显示设置、stride 叠帧或已有训练几何，以避免基于损坏的 Inspire 点流判断 stride 合理性。

**验证**

使用 SciPy KDTree/连通分量比较 frame 103/113 的 bilateral Inspire 与 parent MANO；用同一 URDF、seed=2024 和 1538 点复现 `_build_inspire_pool → _sample_uniform_surface → _fk_surface`，并以逆回 visual-local 后的单次 FK 作为控制；扫描 GRAB/ARCTIC bilateral geometry manifest 的 producer 类型与覆盖计数。仅完成工程数据一致性诊断，未形成训练效果结论。

## 2026-09-15 15:00:15 +0000 — 新旧 GRAB 可视化几何链路对照

- activity_id: ACT-20260915-150015-OICM-GRAB-VIS-HISTORY-DIAG
- timestamp: 2026-09-15 15:00:15 +0000
- modification_version: V1.4.5
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（只读历史与实现对照、活动记录）
- approval: auto
- approval_basis: 用户追问为什么此前 GRAB 可视化没有手点散裂；本次只比较历史 viewer 输入与新旧 producer，不修复或重导。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- parent_activity_id: ACT-20260915-145131-OICM-INSPIRE-DOUBLE-FK-DIAG
- scope: 对照最早 `cm_object_v2/grab` 原生 bilateral MANO viewer、V1.2.5 MANO/Inspire-RL viewer 和当前 V1.4 GRAB→bilateral Inspire viewer 的手点来源、surface sampling 坐标语义、FK 次数及既有验证覆盖；不改代码、cache、index、训练或运行。
- files: [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md)
- evidence_code: [旧 V1.2.5 Inspire cache builder](../../tools/data/build_dexplore_rl_cache.py)、[当前 V1.4 Stage-4 Inspire adapter](../../tools/data/retarget_stage4_bilateral_inspire.py)、[V1.4 数据合同测试](../../tests/test_v1_4_data_contract.py)
- finding: 2026-09-06 最早 GRAB viewer 直接读取 `cm_object_v2/grab` 的左右 MANO 点，各 1538 点，不经过 Inspire URDF/FK。V1.2.5 的 Inspire-RL producer 使用 `InspireUrdfModel.surface_samples()`，明确保留 mesh-local 点，然后 `_fk_surface()` 只施加一次 `link_transform @ visual.local_transform`；历史验证中 Inspire mesh 与 cache 点云误差为 2.0421 mm。因此此前正常并不能覆盖 2026-09-14 新增的 V1.4 双手 adapter。
- regression_origin: V1.4 adapter 为完整 GRAB/ARCTIC 双手 geometric 扩容改用 `_build_inspire_pool()`；该 helper 的输出是已摆到 canonical zero-q 整手坐标的点，与旧 `_fk_surface()` 所要求的 mesh-local 输入语义不兼容，由此引入双重 FK。错误不是旧 viewer 或旧 cache 逐渐变化，而是切换到新 V1.4 producer 后出现。
- missed_gate: V1.4 现有定向测试验证 bilateral 拼接顺序、shape、finite、KNN 索引与文件完整性，但没有逐 side 的 hand bbox/连通性、FK 后 mesh-to-point 距离或 target fingertip residual。全量核验中的 `bad_count=0` 因而只证明 schema/数组可读，不证明几何正确；这是此前把“导出完成”误当成“几何正确”的验证缺口。
- run_status: 既有 `grab_stride10_20260915T134656Z` 查看器仍为 RUNNING；本次未启动新运行。
- protected_boundary: 新旧 GRAB/ARCTIC/OakInk2 数据与 cache、index、split、GT、stride、模型、checkpoint、既有 viewer 和用户其他未提交改动均未修改。
- rollback: 仅删除本条活动记录即可回滚文档增量。
- conclusion: SUPPORTED（此前正常是因为使用 MANO 或正确的 mesh-local→单次 FK 旧链路；散裂是 V1.4 新 adapter 的回归）；INVALID_IMPLEMENTATION（V1.4 的 shape/KNN smoke 不足以证明手几何有效，相关“全量导出成功”结论需要降级为文件层完成）。

**原因**

区分旧数据本身是否曾经正确、viewer 是否回归和 V1.4 新 producer 是否引入错误，并明确为什么已有完成性检查没有拦截该问题。

**验证**

核对历史 activity 中三代 viewer 的真实命令、index/source 与 mesh 对齐结果；静态对照旧 builder 的 mesh-local 采样注释及调用链、V1.4 adapter 的 canonical pool 调用链和当前 V1.4 测试断言范围。未运行新数据处理或训练。

## 2026-09-15 15:56:11 +0000 — 续接 Inspire 双重 FK 修复 pilot 并交付查看器

- activity_id: ACT-20260915-155611-OICM-FKFIX-PILOT-VIEWER
- timestamp: 2026-09-15 15:56:11 +0000
- modification_version: V1.4.6
- operation_category: [code, data, diagnostic, operation, documentation]
- primary_task_mode: change → run-only/operation
- change_level: L2（此前已批准的 surface 坐标修复与两条 pilot）；本次续接的代码增量为 L0 运行版本元数据。
- approval: user-approved
- approval_basis: 会话 `01a0a4fb-2cef-70d0-ba2a-1c18c834f686` 中用户先要求“那你解决目前这个问题”，随后明确限定“先导部分然后可视化给我看吧”；本次用户要求浏览该会话并继续。沿用同一授权，不扩展到全量重导。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- parent_activity_id: ACT-20260915-145131-OICM-INSPIRE-DOUBLE-FK-DIAG
- scope: 完成 V1.4 plan §13 已批准的双重 FK 修复、GRAB/ARCTIC 各一条 pilot 与只读查看器交付；本次复用此前已完成的两条产物，不重复导出。
- plan: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)

**文件**

- [src/task/ObjectInteractionCm/tools/data/retarget_stage4_bilateral_inspire.py](../../tools/data/retarget_stage4_bilateral_inspire.py) — 此前会话已完成 canonical zero-q → visual-local 的逆变换、单次 FK、指定序列和修复 provenance；本次读取和回归验证，无新增 exporter 改动。
- [src/task/ObjectInteractionCm/tests/test_retarget_stage4_bilateral_inspire.py](../../tests/test_retarget_stage4_bilateral_inspire.py) — 此前新增真实 URDF 零姿态往返、旧路径显著偏移及采样/法向回归；本次复跑。
- [src/task/ObjectInteractionCm/research/inspire_fk_repair/run.py](../../research/inspire_fk_repair/run.py)、[src/task/ObjectInteractionCm/research/inspire_fk_repair/experiment.yaml](../../research/inspire_fk_repair/experiment.yaml) — 此前新增两条 pilot 的隔离导出、报告与 viewer-only index；本次无新增修改。
- [src/task/ObjectInteractionCm/research/inspire_fk_repair/README.md](../../research/inspire_fk_repair/README.md) — 本次补充运行版本参数、Δ 操作和几何验证边界。
- [src/task/ObjectInteractionCm/visualize_grab.py](../../visualize_grab.py) — 本次新增 `--modification-version` 并由该参数写入 manifest；默认保留 V1.4.5，修复 pilot 显式使用 V1.4.6。原有未提交的 V1.4 查看器扩展保持。
- [src/task/ObjectInteractionCm/docs/README.md](../README.md)、[src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 此前会话已增加 pilot 导航、final §13 和 V1.4.6 指针；本次保持原样。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 补记因会话中断尚未交付的实现、pilot 终态、旧实例停止和本次运行。

**原因**

此前会话已在 15:49:25 完成两条修复导出，但查看器把 manifest 版本硬编码为 V1.4.5；该实例在
15:50:25 停止后会话中断。本次恢复实际输入、最终计划与授权，补齐参数、独立重启和证据记录。

**运行与产物**

- pilot_run_id: `inspire_fkfix_pilot_20260915T154906Z`；run_status: `COMPLETED`；完成于 `2026-09-15T15:49:25+00:00`，本次只读确认终态。GRAB `s1/airplane_fly_1` 279 帧，ARCTIC `s01/box_use_01` 889 帧。
- pilot_command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.research.inspire_fk_repair.run --output src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z`；两个 producer 子命令见 pilot manifest。
- pilot_output: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/)
- pilot_manifest: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/run_manifest.json](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/run_manifest.json)
- pilot_report: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/pilot_report.json](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/pilot_report.json)
- pilot_index: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/index.json](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/index.json)
- pilot_export_log: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/export.log](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/export.log)
- stopped_viewer: 旧嵌套实例 `run_id=viewer`、`run_status=STOPPED`，停止原因是版本元数据误写；原记录与产物保留，不覆写历史版本。[src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/viewer/run_manifest.json](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/viewer/run_manifest.json)
- run_id: inspire_fkfix_viewer_20260915T155554Z
- run_status: RUNNING
- pid: 3338972（独立进程；停止前须再次核对命令，避免 PID 复用）
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/index.json --split train --sequence grab/s1/airplane_fly_1 --frame 103 --future-delta 10 --point-display both --mesh-display off --host 127.0.0.1 --port 8144 --fps 8 --modification-version V1.4.6 --output src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z`
- viewer: `http://127.0.0.1:8144`
- viewer_output: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z](../../research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/)
- viewer_config: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/config.json](../../research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/config.json)
- viewer_manifest: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/run_manifest.json](../../research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/run_manifest.json)
- viewer_log: [src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/viewer.log](../../research/inspire_fk_repair/output/inspire_fkfix_viewer_20260915T155554Z/viewer.log)

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_retarget_stage4_bilateral_inspire.py src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`：`4 passed in 1.75s`。
- 两条只读 smoke 的公共命令为 `/home/wbcd/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --check-only --index src/task/ObjectInteractionCm/research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/index.json --split train --future-delta 10 --point-display both --mesh-display off --modification-version V1.4.6`，分别追加 `--sequence grab/s1/airplane_fly_1 --frame 103` 和 `--sequence arctic/s01/box_use_01 --frame 437`，均退出 0。
- GRAB 103→113 对应 source frame 412→452，Δ=0.333333 s；手点流 median/P95/max=`148.851/184.899/187.091 mm`。ARCTIC 437→447 的手点流为 `72.243/76.449/77.976 mm`。两条均读取 3076 个手点、4096 个物体点，未发生末帧 clamp。
- 已有 pilot 报告抽查帧中，两数据集左右手在 20 mm 半径下各为 1 个连通分量。GRAB 最大单侧尺寸约 239.813 mm，ARCTIC 约 228.385 mm；最大法向单位误差约 `2.043e-7`。这些数字仅覆盖报告抽查帧，不能外推所有帧和全部数据。
- `curl --noproxy '*' --max-time 10 -s -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8144` 返回 HTTP 200；`ss -ltnp '( sport = :8144 )'` 与 `ps -p 3338972 -o pid,ppid,etime,args` 确认命令及端口；manifest/config 为 V1.4.6，viewer.log 确认初始渲染流程完成且可选两条轨迹。未声称已人工审阅浏览器画面。
- `git diff --check` 通过；交接审计使用 `audit_diff.py --worktree --check-links`，以本节列出的实现文件、实验目录、计划、指针和 activity 的显式 `--scope-prefix` 限定范围，保护其余未提交改动。

**结论与保护边界**

- conclusion: `SUPPORTED`（单次 FK 回归与两条 pilot 工程读取）；`INCONCLUSIVE`（整体重定向接触保真、左右手资产正确性、stride 合理性及科研效果）。
- 当前 exporter 仍沿用左手加载失败时使用右手 surface 资产的既有回退；这是与 double-FK 不同的限制，本次未扩展修复。MANO 对照点距不是指尖残差或 mesh 对齐证明。viewer 不支持当前 geometric hand mesh，ARCTIC 物体 mesh 也不在 GRAB mesh 根下，故启动时使用纯点云显示。
- 旧 v2 全量 cache、KNN/index、OakInk2、原始 annotation/Stage3、val/test、GT 定义、stride、模型、checkpoint 及无关用户改动保持。未创建 v3 全量 root，未全量重导或启动训练。
- 本次为数据 pilot 与可视化，没有训练 step/epoch、best metric、metrics.jsonl、train.log 或 checkpoint；实际日志为上述 export.log/viewer.log。
- rollback: 核对 PID 与命令后停止本次 viewer；仅移除本次独立 viewer output 和新增版本参数/README/activity 增量即可撤回本次续接。完整 V1.4.6 修复回滚范围见 final plan §13，不能 reset 整个工作区。当前未删除任何旧产物。
- 规范反馈：无新增格式、目录、版本、日志或审批阻碍；沿用既有用户批准，使用两个 Skill 补齐运行元数据和审计，不修改治理规则。

## 2026-09-15 16:04:00 +0000 — Inspire pilot 来源与 Dexplore 方案边界纠错

- activity_id: ACT-20260915-160400-OICM-FKFIX-PROVENANCE-DIAG
- timestamp: 2026-09-15 16:04:00 +0000
- modification_version: V1.4.6
- operation_category: [diagnostic, documentation, operation]
- primary_task_mode: read-only/diagnostic
- change_level: L0（来源审计、运行停止和活动记录；未修改数据、cache 或算法）
- approval: auto
- approval_basis: 用户质疑 pilot 的几何重定向数据和 Inspire 来源；本次只读追溯并停止由本 Agent 启动的查看器。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- parent_activity_id: ACT-20260915-155611-OICM-FKFIX-PILOT-VIEWER
- scope: 审计 V1.4.6 pilot 的 MANO 输入、Inspire URDF/mesh、retargeting 配置、qpos 生成路径，并与 Dexplore RL cache 的真实 qpos 路径对照；停止端口 8144 查看器以避免继续展示未经确认的结果。

**文件与证据**

- [pilot run manifest](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/run_manifest.json) — 两个实际 producer 命令和输入根目录。
- [pilot geometry manifest](../../research/inspire_fk_repair/output/inspire_fkfix_pilot_20260915T154906Z/cache/grab/s1/airplane_fly_1/geometry/manifest.json) — 明确记录 `source_type: stage4_mano_to_inspire_position_retarget`、`surface_sampling_space: visual_mesh_local`，不是 Dexplore RL source。
- [V1.4.6 pilot exporter](../../tools/data/retarget_stage4_bilateral_inspire.py) — 第 10–12 行加载 Ref2Dex FK/helper 与 `dex_retargeting`；第 63–93 行加载 Dexplore URDF/mesh，构造临时 URDF，并建立 position retargeting；第 117–133 行从 MANO 顶点或采样点生成每帧 5 个目标点并调用 `rt.retarget`。
- [Dexplore RL cache builder](../../tools/data/build_dexplore_rl_cache.py) — 第 454–456 行从 Dexplore 的 `interaction_hand_inspire.pt` 读取 native q；第 489–494 行才使用这些 RL q 做 Inspire surface FK。这条路径没有被本 pilot 调用。
- Dexplore retarget 配置：外部 `dex-retargeting/dex_retargeting/configs/offline/inspire_hand_left.yml` 与 `inspire_hand_right.yml` 定义的是 position retargeting 的 5 个 tip link；pilot 还覆盖了其 URDF、dummy free joint 和 mimic 处理。
- Dexplore Inspire 资产：外部 `/home/wbcd/workspace/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_{left,right}.urdf` 及对应 `meshes_left/right/*.STL`；这是资产来源，不是已生成的 Inspire 动作数据。
- viewer: `run_id=inspire_fkfix_viewer_20260915T155554Z` 已发送 SIGTERM，端口 8144 已释放；该查看器的 run manifest 和日志保留为 STOPPED 历史证据。

**原因**

用户指出“几何重定向的数据从哪里来的、Inspire 手数据哪里来的、是否使用 Dexplore 方案”。审计发现此前交付把“使用 Dexplore 的资产与 dex-retargeting 配置”误说成“使用 Dexplore 的完整重定向方案”，两者不等价。

**结论**

- `SUPPORTED`：来源事实已核实——输入是 Ref2Dex 30 Hz MANO，Inspire 点是 Dexplore URDF/mesh 现场表面采样。
- `REFUTED`：该 pilot 使用 Dexplore 已生成的 Inspire 重定向轨迹，或复现 Dexplore RL/native-q 方案。
- `INVALID_IMPLEMENTATION`：把该 pilot 的结果作为“Dexplore geometric retarget 方案已修复/已验证”的此前表述无效；pilot cache、报告和查看器不得用于证明 Dexplore 方案正确。
- 保护边界：未修改原始 MANO、Dexplore 外部仓库、旧 v2 cache、Dexplore RL cache、split、GT、模型或 checkpoint；未启动全量导出或训练。

**验证**

- 逐行静态审计 exporter、pilot manifest、geometry manifest、Dexplore retarget 配置和旧 RL builder；确认实际调用链如上。
- 检查 pilot 三类输入数组：GRAB `shared.npz`/`left.npz`/`right.npz` 为 30 Hz object + MANO 字段；pilot 输出 manifest 标记 `stage4_mano_to_inspire_position_retarget`。
- 检查 `ps`、`ss` 和 HTTP：由本 Agent 启动的 8144 服务已停止；无新的数据处理或训练运行。

**纠正后的后续边界**

在用户重新确认前，不会把该 pilot 接入 cache 或训练，也不会继续声称它代表 Dexplore 方案。若要按 Dexplore 方案重做，必须以 Dexplore 已有 `interaction_hand_inspire.pt`/对应 producer 为输入，逐条核对其 MANO→Inspire 生成来源、native q 顺序、左右手资产和 surface correspondence 后另立可审计 pilot；这属于新的 L2/L3 数据与方案变更。

## 2026-09-15 16:26:28 +0000 — GRAB/ARCTIC 原始 MANO 双手轨迹查看器

- activity_id: ACT-20260915-162628-OICM-MANO-VIS-RUNNING
- timestamp: 2026-09-15 16:26:28 +0000
- modification_version: V1.4.6
- operation_category: [diagnostic, operation, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L1（Task-local 只读查看器与独立运行目录）
- approval: user-approved
- approval_basis: 用户明确要求“先把 GRAB 和 ARCTIC 的 mano 轨迹给我看”。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 直接读取 Ref2Dex 30 Hz MANO cache 的 GRAB `s1/airplane_fly_1` 与 ARCTIC `s01/box_use_01`，显示左右手 MANO mesh/点云和物体点云；不读取 Inspire、Dexplore q、pilot cache 或重定向产物。

**文件与运行**

- [MANO viewer](../../research/mano_trajectory_visualizer/run.py) — 新增只读 Viser 入口，支持两条轨迹切换和帧滑块。
- run_id: `mano_trajectory_20260915T162523Z`
- run_status: `RUNNING`
- viewer: `http://127.0.0.1:8145`
- output: [src/task/ObjectInteractionCm/research/mano_trajectory_visualizer/output/mano_trajectory_20260915T162523Z](../../research/mano_trajectory_visualizer/output/mano_trajectory_20260915T162523Z)
- run_manifest: [src/task/ObjectInteractionCm/research/mano_trajectory_visualizer/output/mano_trajectory_20260915T162523Z/run_manifest.json](../../research/mano_trajectory_visualizer/output/mano_trajectory_20260915T162523Z/run_manifest.json)

**输入与验证**

- GRAB 输入：`data/processed_data/oicm_v1_4_raw/grab_mano_30hz/s1/airplane_fly_1`；ARCTIC 输入：`data/processed_data/oicm_v1_4_raw/arctic_mano_30hz/s01/box_use_01`。
- 两条输入均确认存在 `shared.npz`、`left.npz`、`right.npz`；GRAB 279 帧、左右各 1538 点并有 778 顶点 MANO mesh；ARCTIC 889 帧、左右各 1538 点。
- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/mano_trajectory_visualizer/run.py` 通过；进程 PID 3389501，`ss` 确认 127.0.0.1:8145，HTTP 200。
- 本次 viewer 只作轨迹查看，结论为 `INCONCLUSIVE`；不据此判断重定向质量或训练效果。

**保护边界**

- 未修改任何 MANO 输入、Inspire 资产、Dexplore 外部仓库、cache、split、GT、模型或 checkpoint。
- viewer 由本 Agent 启动；用户确认不再需要后可停止 PID 3389501 并删除独立 output。随后重定向修改仍需按明确的 Dexplore offline 方案进入新的 L2 变更记录。

**原因**

先让用户直接核对未经重定向的左右手 MANO 轨迹，隔离原始 MANO 数据与后续 Inspire 重定向问题。

**验证**

已通过 Python 编译检查、三文件输入存在性检查、进程/端口检查和 HTTP 200 检查；未执行重定向、数据改写或训练。

## 2026-09-15 16:32:14 +0000 — 找回异机已全量导出的 Dexplore geometric 脚本并纠正来源判断

- activity_id: ACT-20260915-163214-OICM-DEXPLORE-HISTORY-RECOVERY
- timestamp: 2026-09-15 16:32:14 +0000
- modification_version: V1.4.6
- operation_category: [diagnostic, documentation]
- primary_task_mode: read-only/diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户提醒之前已经自写重定向脚本并在另一台机器全量导出；本次只读核查历史与本机文件，纠正此前误判。
- skills_used: research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: Dexplore 旧 converter、canonical adapter、导出说明与 Git 提交，以及本机 Cmv2 已有导出的只读交叉核验。仅追加本条记录。

**原因**

此前只比较 V1.4 bilateral adapter 与 OICm 旧 RL cache consumer，就把 native tensor 误判为 RL 专属，
并建议恢复未经对照的上游默认配置；遗漏了用户已经使用并导出过全量数据的 Dexplore geometric producer。
本条纠正 ACT-20260915-160400 的“应以 RL/native-q 链路替代 geometric”推断，以及随后重写方案的依据。

**文件与证据**

- [../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md](../../../../../../dexplore/data_processing/README_GRAB_INSPIRE_EXPORT.md) — 记录旧机器 `/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric` 的 1335 条 GRAB geometric 导出，及 2026-09-05 从 canonical cache 重导；本机旧绝对路径不可见，不声称已远程核验这些旧文件。
- [../dexplore/data_processing/adapt_interact_canonical.py](../../../../../../dexplore/data_processing/adapt_interact_canonical.py) — 已有 canonical 输入桥接；README 明确禁止把旧 `prepare_grab.py` 的 upright 旋转结果再次输入 converter，避免重复旋转。
- [../dexplore/data_processing/convert_grab.py](../../../../../../dexplore/data_processing/convert_grab.py) — 已有 GRAB→Inspire geometric producer，`setup_retargeting`、右手 21 点组装、配置指定的目标点、关节名称映射和 native DOF 重排均已实现。Dexplore Git 提交 `c31f57f` 保存这批导出工具，本机外部仓库工作树干净。
- [../dexplore/data_processing/robot_configs.py](../../../../../../dexplore/data_processing/robot_configs.py) — Inspire 18 DOF、MANO/SMPL-X landmark 重排和 native 重排合同。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 2026-09-05 的 `ACT-20260905-202524-OICM-DEXPLORE-RL-DATASET-CLARIFICATION` 已明确区分 1335 条 geometric 与 RL；2026-09-14 09:13:27 条目已记录旧 converter 固定 GRAB 右手、21 点输入和 5 tip 位置目标。本次只新增当前条目，不覆写历史。
- [src/task/ObjectInteractionCmv2/docs/logs/activity_log.md](../../../ObjectInteractionCmv2/docs/logs/activity_log.md) — 本机独立 Cmv2 导出的既有终态入口，当前只读。
- [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/) — 本机另有同一 converter 生成的筛选集合，可作对照；不能当作旧 1335 条全量或双手/ARCTIC 结果。
- [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json) — 既有 `run_id=grab-dexplore-rl-full-20260915T125105Z`、`run_status=COMPLETED`，本次未运行/修改。
- [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/validation.json](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/validation.json) — 已有验证报告 `num_sequences=656`；未在本次重跑验证。
- [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/commands.log](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/commands.log) — 确认调用本机 Dexplore `convert_grab.py --robot inspire ... --retarget-iterations 1 --retarget-stride 1`，分别输出 geometric 与后续 RL。

**纠正结果**

1. `interaction_hand_inspire.pt` 在 geometric 和 RL 两种目录中都使用相同 `[T,598]` 格式，native q slice 同为 `[373:391]`；必须由 producer/目录/manifest 区分，不能凭文件名认定 RL。
2. 旧 converter 同样使用 position optimizer 的五指尖目标。21 点是输入的 landmark 排列，并不表示优化器同时拟合 21 点；“使用五点”本身不能证明偏离原方案。
3. 旧 `setup_retargeting` 也包含 continuous→revolute 临时 URDF、已有六轴 wrist 时关闭 dummy free joint、18 DOF 全量优化和 `ignore_mimic_joint=True` 的适配。此前把这些设置单独列为错误，并建议一律恢复上游默认配置，依据不足；需以历史实际资产/输入/映射核对，不能擅改耦合语义。
4. 已证实旧 producer 存在且保存了全量导出历史；当前 bilateral adapter 的 raw MANO 输入、左手使用右手模型回退和自建 surface 路径仍需与旧 converter 对照。double-FK 的独立回归证据不因本次来源纠错而失效，但也不足以证明整体重定向正确。
5. 后续按用户“自己改重定向”的方向，应复用已找回的 converter 作为 GRAB 基准，先核对 canonical 坐标与 qpos，再处理双手/ARCTIC 接口；撤回从上游默认配置另写一套及将 geometric 换成 RL 的建议。本次尚未实施这些修改。

**验证**

- `git -C /home/wbcd/workspace/oyx_ws/dexplore show --stat c31f57f -- data_processing`、`git ... status --short`；读取 converter 的 `setup_retargeting` 和 retarget/输出段、robot config、README 与历史 activity。
- `jq` 读取既有 Cmv2 manifest/validation；`rg -n -m 5 'convert_grab.py' .../commands.log` 确认实际命令，`rg --files data/processed_data -g 'interaction_hand_inspire.pt'` 确认 geometric/RL 两类文件在本机存在。
- conclusion: SUPPORTED（旧 geometric 脚本、历史全量记录及本机另一路复用证据）；REFUTED（native 文件名仅代表 RL、五点/临时 URDF 本身证明偏离旧方案）；INCONCLUSIVE（当前双手/ARCTIC 重定向整体正确性）。仅为来源诊断，不是模型效果结论。
- 保护边界：本次未修改代码、外部 Dexplore、数据、cache、split、checkpoint 或任何运行；MANO viewer 保持。回滚入口为本条 activity 增量。
- 规范反馈：无流程阻碍；问题来自 Agent 漏读既有 producer 和历史记录，不能归因于缺少用户确认。未修改 AGENTS、Skill 或其他公共合同。

## 2026-09-16 02:17:52 +0000 — 启动旧 Dexplore geometric 导出只读可视化

- activity_id: ACT-20260916-021752-OICM-DEXPLORE-GEOMETRIC-VIEWER
- timestamp: 2026-09-16 02:17:52 +0000
- modification_version: V1.4.6
- operation_category: [diagnostic, operation, documentation]
- primary_task_mode: run-only/operation
- change_level: L0（已有查看器运行和追溯文档，不修改重定向）
- approval: user-approved
- approval_basis: 用户在找回旧 geometric producer 后要求“那你现在可视化给我看一下”。
- skills_used: research-experiment-workflow、research-change-control
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true
- scope: 复用外部旧 viewer，读取本机 Cmv2 的 656 条 GRAB 右手 geometric 轨迹及其物体；只新增诊断说明、独立运行产物和本条记录。
- run_id: `dexplore_geometric_20260916T021513Z`
- run_status: `RUNNING`
- PID: 3847849；exec session: 67591
- viewer: http://127.0.0.1:8146

**原因**

让用户直接查看旧 Dexplore converter 的已有产物，不再把 V1.4 bilateral pilot、geometric 和 RL 混淆。
原查看器的 MANO 叠加要求 InterAct `data/grab/sequences_canonical/*/human.npz`，本机未找到这些文件，
因此本次关闭 MANO 叠加；不使用其他坐标系点云替代。原始 MANO 查看器 8145 保持运行。

**文件与运行入口**

- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/README.md](../../research/dexplore_geometric_visualization/README.md) — 新增只读查看边界说明。
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/experiment.yaml](../../research/dexplore_geometric_visualization/experiment.yaml) — 外部已有入口和独立 output 路由；未新增实现脚本。
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z](../../research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/)
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/run_manifest.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/run_manifest.json)
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/config.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/config.json) — 完整启动 argv/cwd；进程内只将 Viser 监听地址限制为 127.0.0.1，外部源码不改。
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/viewer.log](../../research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/viewer.log)
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/smoke.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/smoke.json)
- [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/run_manifest.json) — 既有 producer 来源，不修改其状态。
- [../dexplore/data_processing/visualize_inspire_trajectory.py](../../../../../../dexplore/data_processing/visualize_inspire_trajectory.py) — 原查看器，外部 commit `c31f57f186409ce5f0de47ced2d347abffe45d06`，工作树仍干净。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 仅追加当前活动；保留原有未提交内容。

**验证**

- `graspenv/bin/python ../dexplore/data_processing/visualize_inspire_trajectory.py --all --source geometric --geometric-root <producer>/geometric --object-root <producer>/canonical/objects --check-only`：退出 0，656 条轨迹；首条 `s1_airplane_fly_1` 为 [279,598]，native q 列 [373:391]，18 DOF。
- 首条 frame 139 为 object-contact，旧 viewer 自带 FK 检查打印 min tip/object-center 距离约 0.017 m；该量不是表面接触误差，不证明重定向质量。
- `ss -ltnp 'sport = :8146'` 确认上述 PID 监听 localhost；`curl http://127.0.0.1:8146` 返回 HTTP 200。
- 使用 `viser-v1.1.0` 子协议连接 WebSocket，收到 121 条初始化消息、约 12.8 MB payload，包含 13 个 Inspire link mesh 与 1 个物体 mesh；656 条数据对应物体网格均存在。未执行浏览器截图或人工视觉质量评估。
- 初次 shell 后台启动未存活，改为保持的 exec 会话后正常；初次通用 WebSocket 探针因缺少版本子协议被拒绝，使用当前 Viser 协议后通过。均未改动输入或依赖。
- `git diff --check -- src/task/ObjectInteractionCm/research/dexplore_geometric_visualization src/task/ObjectInteractionCm/docs/logs/activity_log.md` 通过；交接前执行当前活动的 scope/link audit。
- conclusion: SUPPORTED（读取和可视化服务工程 smoke）；INCONCLUSIVE（重定向整体质量）。不是科研效果验证；无训练、checkpoint 或 metrics.jsonl。

**保护边界与回滚**

未修改 Dexplore、converter、URDF、MANO、数据、cache、坐标、GT、split、训练、checkpoint 或旧运行。
仅为本机 656 条筛选集，不冒充异机 1335 条全量，也不代表 ARCTIC/双手已修好。
回滚可停止本次 PID 3847849，并仅移除本次新增诊断定义、独立 output 和本条记录；当前不执行删除。
规范反馈：无审批或目录阻碍；未改治理合同。MANO 同源文件缺失是本机输入限制，不是规则阻碍。
## 2026-09-16 02:47:14 +0000 — GRAB MANO / Inspire 同序列配对查看器

- activity_id: ACT-20260916-024714-OICM-PAIRED-GRAB-MANO-INSPIRE-VIEWER
- timestamp: 2026-09-16 02:47:14 +0000
- modification_version: V1.4.7
- operation_category: [code, diagnostic, operation, documentation]
- primary_task_mode: change → run-only/operation
- change_level: L1（Task-local 只读 viewer adapter；不改变正式数据/schema/科研语义）
- approval: user-approved
- approval_basis: 用户确认本机 GRAB MANO/Inspire 数据齐全后明确要求“修改一下然后可视化给我看”。
- skills_used: research-change-control、research-experiment-workflow
- branch: feature/objectinteractioncmv2-v1.0.2
- base_commit: 3d59e14a292e2ac846e031e44c017ecd1d6096ad
- worktree_dirty: true（保留已有 V1.4、模型、日志、配置和其他 Task 未提交改动）
- scope: 为现有 GRAB Viser 增加 raw bilateral MANO NPZ 的只读内存适配；生成一条同序列 MANO/Inspire viewer-only 配对 index 并启动查看器。源数据、正式 index/cache、split、GT、重定向、训练、模型和 checkpoint 不修改。
- prepare_run_id: `paired_grab_mano_inspire_20260916T024600Z`
- prepare_run_status: `COMPLETED`
- viewer_run_id: `viewer_paired_grab_mano_inspire_20260916T024600Z`
- viewer_run_status: `RUNNING`
- viewer_pid: 3880885
- viewer: http://127.0.0.1:8147
- conclusion: SUPPORTED（配对、读取、KNN/距离着色和服务工程链路）；INCONCLUSIVE（Inspire 重定向质量与训练收益）

**文件与运行入口**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针更新为 V1.4.7。
- [src/task/ObjectInteractionCm/docs/README.md](../README.md) — 增加本诊断入口导航。
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) — 新增已批准的第 14 节实施边界。
- [src/task/ObjectInteractionCm/visualize_grab.py](../../visualize_grab.py) — 只读加载 `left.npz/right.npz/shared.npz`，按 left-then-right 合并 3076 点；保留原距离/KNN 定义并支持 bilateral MANO mesh。
- [src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py](../../tests/test_visualize_grab_v1_4.py) — 增加 raw bilateral MANO adapter 回归。
- [src/task/ObjectInteractionCm/research/paired_grab_visualization/README.md](../../research/paired_grab_visualization/README.md)、[src/task/ObjectInteractionCm/research/paired_grab_visualization/experiment.yaml](../../research/paired_grab_visualization/experiment.yaml)、[src/task/ObjectInteractionCm/research/paired_grab_visualization/prepare.py](../../research/paired_grab_visualization/prepare.py) — viewer-only 配对 index 与严格一致性闸门。
- [src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/)
- [src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/run_manifest.json](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/run_manifest.json)、[src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/index.json](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/index.json)、[src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/pair_validation.json](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/pair_validation.json)
- [src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/check_mano.json](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/check_mano.json)、[src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/check_inspire.json](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/check_inspire.json)
- [src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/viewer_paired_grab_mano_inspire_20260916T024600Z/run_manifest.json](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/viewer_paired_grab_mano_inspire_20260916T024600Z/run_manifest.json)、[src/task/ObjectInteractionCm/research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/viewer_paired_grab_mano_inspire_20260916T024600Z/viewer.log](../../research/paired_grab_visualization/output/paired_grab_mano_inspire_20260916T024600Z/viewer_paired_grab_mano_inspire_20260916T024600Z/viewer.log)

**原因**

现有 Inspire geometric 已是 viewer 原生 `geometry/*.npy`，MANO 则保留为完整的 bilateral NPZ triplet。
此次不复制大数组、不创建正式 cache；查看器只在选择 MANO 记录时加载 NPZ，并把左右手各 1538 点和
mesh 按固定顺序合并。配对准备器要求两种表示的 source frame、物体点和物体 pose 逐元素完全相等，
失败即停止。两条 GUI 记录复用同一 `grab/s1/airplane_fly_1` ID，由 variant 标签区分。

**验证**

- `python -m py_compile`：查看器与 prepare 入口通过。
- `PYTHONPATH=. .../python -m pytest -q src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`：`3 passed in 1.13s`。
- 配对报告：MANO/Inspire 均 279 帧，source frame `0..1112`；frame count、source frame ID、4096 object points、object pose 全部 exact match。
- 两条真实 `--check-only`：frame 103 均映射 source frame 412、双手均为 3076 点；MANO/Inspire 最近物距分别 `0.405826 mm` / `0.893608 mm`。KNN n=`1/4/8/16/32/64` 均成功计算；详细计数见两份 check JSON。
- MANO bilateral mesh 为 2 parts、1556 vertices、3076 faces；Inspire geometric v2 未保存 viewer 可恢复的双手 qpos，因此正式启动使用 surface point cloud、关闭 hand mesh，不伪造几何。
- `curl http://127.0.0.1:8147` 返回 HTTP 200；Viser `1.1.0` WebSocket 收到 218356-byte 初始化 payload；端口由 PID 3880885 监听。
- 初次临时输出使用了占位时间 run_id，发现后已停止服务并将该临时目录移至 `/tmp/ref2dex_paired_grab_mano_inspire_bad_run_id_20260916T000000Z`；第二个非唯一 `viewer` 子运行同样停止并移至 `/tmp/ref2dex_paired_grab_viewer_nonunique_run_id_20260916T024600Z`。最终入口只使用上列精确且唯一的 run_id，源数据未受影响。
- 本次只证明工程读取、配对与显示链路；没有训练、评估、metrics.jsonl 或 checkpoint，不把 smoke 解释为重定向质量成立。

**保护边界与回滚**

未修改 1335 条 MANO/Inspire 源轨迹、正式 V1.4 index/cache、split、坐标、单位、GT、重定向、训练配置、模型、checkpoint 或旧运行。停止前先核对 PID/命令，再停止最终 viewer；删除本次独立 output，并恢复上述 adapter、测试、研究定义、plan 第 14 节、版本指针和本条记录即可回滚。规范反馈：无格式、目录、版本或审批阻碍。
## 2026-09-16 03:35:27 +0000 — DExplore geometric q 的 MANO 2048 / Inspire 10135 点配对预览

- activity_id: `ACT-20260916-033527-OICM-GEOMETRIC-Q-SURFACE-PREVIEW`
- timestamp: `2026-09-16 03:35:27 +0000`
- modification_version: `V1.4.8`
- operation_category: `[code, data, diagnostic, operation, documentation]`
- primary_task_mode: `change → run-only/operation`
- change_level: `L2`（geometric q 来源、共同 object pose 与高分辨率表面数据语义）
- approval: `user-approved`
- approval_basis: 用户在 geometric 与 RL 两种 native-q 路线中明确选择“用你建议的方式”，即复用 DExplore geometric q 做同序列右手 MANO/Inspire pilot。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留并避开已有 V1.4、ObjectInteractionCmv2 及其他用户修改）
- plan: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §15（final）
- scope: 固定 GRAB `s1/airplane_fly_1`，读取已有 DExplore geometric tensor 的 native q `[373:391]`，在 geometric object pose 下生成右手 MANO 2048 / Inspire 10135 点单次 FK pilot，并启动只读距离/KNN 着色查看器；不使用同级 RL tensor，不重新执行五指尖优化，不扩展到全量或训练。

**文件**

- [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/run.py](../../research/geometric_q_surface_preview/run.py) — 新增当前 GRAB NPZ + DExplore geometric tensor 的严格帧对齐、共同 object-pose 变换、固定表面 correspondence、单次 FK、geometry/index/manifest 与诊断生成入口。
- [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/README.md](../../research/geometric_q_surface_preview/README.md) 与 [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/experiment.yaml](../../research/geometric_q_surface_preview/experiment.yaml) — 记录入口、输入语义、点数、seed、保护边界和可解释范围。
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §15、[src/task/ObjectInteractionCm/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 写入已批准的 V1.4.8 最终边界、导航和版本指针。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本条实施、运行与验证终态。

**运行与产物**

- run_id: `geometric_q_surface_20260916T033410Z`
- run_status: `COMPLETED`
- command: `PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.research.geometric_q_surface_preview.run --output src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z`
- input geometric q: [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/geometric/s1_airplane_fly_1/interaction_hand_inspire.pt](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/geometric/s1_airplane_fly_1/interaction_hand_inspire.pt)，SHA256 `6c6dfc516787e886ae6a1f394a1186640eb0f0363863f19b570ab527f54d09a4`。
- output: [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z](../../research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/)
- run_manifest: [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/run_manifest.json](../../research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/run_manifest.json)
- index: [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/index.json](../../research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/index.json)
- diagnostics: [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/diagnostics.json](../../research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/diagnostics.json)
- viewer_run_id: `viewer`
- viewer_run_status: `RUNNING`
- viewer_pid: `3919722`（停止前须核对命令，避免 PID 复用）
- viewer: `http://127.0.0.1:8148`
- viewer_command: `PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.visualize_grab --index src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/index.json --split train --sequence-index 1 --frame 103 --future-delta 0 --point-display both --mesh-display off --host 127.0.0.1 --port 8148 --fps 8 --modification-version V1.4.8 --output src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/viewer`
- viewer_manifest: [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/viewer/run_manifest.json](../../research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/viewer/run_manifest.json)
- viewer_log: [src/task/ObjectInteractionCm/research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/viewer/viewer.log](../../research/geometric_q_surface_preview/output/geometric_q_surface_20260916T033410Z/viewer/viewer.log)

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m py_compile src/task/ObjectInteractionCm/research/geometric_q_surface_preview/run.py`：退出 0；`--help`：退出 0。
- 输入 279 帧与 GRAB 30 Hz NPZ 严格等长，tensor `[279,598]` finite；producer 路径明确位于 `/geometric/` 而非 `/rl/`；native q 固定为 `373:391`。
- 两条 geometry 共同使用同一份 4096 点 object、pose 和 raw frame；object-local 往返最大绝对误差 `7.45e-8 m`。MANO `[279,2048,3]`、Inspire `[279,10135,3]` 全部 finite，最大法向范数误差分别为 `1.79e-7`、`5.96e-8`。
- frame 103/source frame 412：MANO bbox `109.45×141.69×120.27 mm`，Inspire bbox `151.98×197.22×116.64 mm`；两者在 20 mm 邻接下均为 1 个连通分量。Inspire 相比旧 double-FK 数据不再出现约 350–400 mm、7 个分量的结构性散裂。
- 同帧 MANO/Inspire 最近手物距离分别为 `0.469/0.197 mm`；object→hand KNN=32 的 union 手点数分别为 `352/1406`。这些无符号距离不能判定 mesh 内外或穿透。
- 对 index 的 `--check-only --sequence-index 0/1 --frame 103 --future-delta 10` 均退出 0；分别读到 2048/10135 手点、相同 source frame 412→452 和相同 object flow。手 mesh provider 不支持本 preview provenance，但点云 viewer 使用 `--mesh-display off`，未伪造 mesh。
- `curl --noproxy '*' --max-time 10 http://127.0.0.1:8148` 返回 HTTP 200；`ss` 与 `ps` 确认 PID/端口/命令，viewer manifest 为 V1.4.8、RUNNING。首次 detached `nohup` 在创建 viewer run 前退出，未占端口；随后由统一执行会话正常启动，未覆盖产物。
- `git diff --check` 通过；交接前使用限定 scope 的 `audit_diff.py --worktree --check-links` 检查本次文档和实现链接。

**结论与保护边界**

- conclusion: `SUPPORTED`（geometric native-q 来源、固定 10135 表面 correspondence、单次 FK、共同 object pose、viewer 距离/KNN 读取链路）；`INCONCLUSIVE`（是否穿透、接触保真、全量序列质量及训练效果）。
- 当前结果只覆盖 DExplore-compatible 集合中的一条右手 geometric 序列；没有证明左手、完整 1335 条 GRAB 或 ARCTIC 正确。没有把工程 smoke 报告为科研效果。
- `grab_inspire_bilateral_v2`、旧 10135 预览、DExplore geometric/RL tensor、正式 index/cache/split/GT、模型、训练配置、checkpoint 和用户其他未提交修改均未修改；未启动全量导出或训练。
- rollback: 核对 PID 与命令后停止本次 viewer，删除独立 output 和 `research/geometric_q_surface_preview/`，恢复 plan §15、README、版本指针与本条 activity；不得 reset 整个 dirty 工作区。
- 规范反馈：无新增格式、目录、版本、日志或审批阻碍。

**原因**

用历史已验证的 DExplore geometric q 替代当前错误的 bilateral surface adapter，严格复现异机高分辨率
surface replay 的核心方式，同时保留 geometric/RL 来源边界并以一条隔离 pilot 供人工检查。
## 2026-09-16 04:00:09 +0000 — 按真实 DExplore 链路启动 geometric 完整 URDF mesh viewer

- activity_id: `ACT-20260916-040009-OICM-DEXPLORE-GEOMETRIC-MESH-VIEWER`
- timestamp: `2026-09-16 04:00:09 +0000`
- modification_version: `V1.4.9`
- operation_category: `[diagnostic, operation, documentation]`
- primary_task_mode: `read-only/diagnostic → run-only/operation`
- change_level: `L2`（纠正 geometric tensor 的可视化表示与来源解释；不改变输入数据）
- approval: `user-approved`
- approval_basis: 用户明确给出异机 8105 的真实链路，并纠正目标为 `373:391` 驱动完整 URDF mesh、`198:205` 驱动物体，而不是 10135 点 surface cache。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留已有 V1.4、其他 Task 和用户未提交修改）
- plan: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §16（final）
- parent_activity_id: `ACT-20260916-033527-OICM-GEOMETRIC-Q-SURFACE-PREVIEW`
- scope: 核对用户给出的 InterAct canonical→DExplore converter→1335 geometric→规则筛选→660 异机链路；确认本机只有同 converter 生成的 656 条 geometric 集合；停止两条错误点云 viewer，修正消失的旧 8146 状态，并用 DExplore 原 viewer 启动完整 URDF mesh + geometric object 的本机 656 条浏览页面。

**来源纠正**

- [../dexplore/data_processing/adapt_interact_canonical.py](../../../../../../dexplore/data_processing/adapt_interact_canonical.py)、[../dexplore/data_processing/convert_grab.py](../../../../../../dexplore/data_processing/convert_grab.py)、[../dexplore/data_processing/filter_inspire_for_dexplore.py](../../../../../../dexplore/data_processing/filter_inspire_for_dexplore.py) 与 [../dexplore/data_processing/visualize_inspire_trajectory.py](../../../../../../dexplore/data_processing/visualize_inspire_trajectory.py) 均存在于外部 commit `c31f57f186409ce5f0de47ced2d347abffe45d06`，工作树干净，本次只读。
- 异机事实由用户提供：完整 geometric 根 1335 条；排除 658 条任意左手 contact、再排除 17 条 doorknob，得到 660 条；筛选使用 `shutil.copy2`，不重算 q。本机没有 `data/processed_data/inspire_geometric` 或 `inspire_geometric_dexplore`，无法在本机复核异机 full/filter tensor SHA256。
- 本机实际输入 [data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/manifest.json](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/manifest.json) 明确为 656 条；其命令日志确认仍调用同一个 `convert_grab.py`、`retarget-iterations=1`、`retarget-stride=1`。筛选统计为 source 1335、left-contact excluded 662、doorknob excluded 17、selected 656，并显式记录与历史 660 相差 -4。
- 样例 geometric tensor [s1_airplane_fly_1/interaction_hand_inspire.pt](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/geometric/s1_airplane_fly_1/interaction_hand_inspire.pt) 为 `[279,598]`，SHA256 `6c6dfc516787e886ae6a1f394a1186640eb0f0363863f19b570ab527f54d09a4`；本次不与不存在的异机 660 根声称相等。

**文件与运行**

- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §16、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 记录用户纠正后的 V1.4.9 表示和运行边界。
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/README.md](../../research/dexplore_geometric_visualization/README.md)、[src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/experiment.yaml](../../research/dexplore_geometric_visualization/experiment.yaml) — 明确 660/656 边界以及 viewer 直接渲染 mesh、不消费 10135 点。
- 旧 `run_id=dexplore_geometric_20260916T021513Z` 的 PID 3847849 与端口 8146 已不存在；其 [run manifest](../../research/dexplore_geometric_visualization/output/dexplore_geometric_20260916T021513Z/run_manifest.json) 从陈旧 RUNNING 修正为 `STOPPED`，停止原因是审计时进程/端口已消失。
- V1.4.7 错误点云 viewer `viewer_paired_grab_mano_inspire_20260916T024600Z`：PID 3880885 已发送 SIGTERM，run_status `STOPPED`，端口 8147 已释放。
- V1.4.8 10135 点 viewer `viewer`：PID 3919722 已发送 SIGTERM，run_status `STOPPED`，端口 8148 已释放；数据 pilot 保留作审计，不再声称符合异机 8105 表示。
- run_id: `dexplore_geometric_mesh_20260916T035811Z`
- run_status: `RUNNING`
- PID: `3936819`；exec_session_id: `5136`（停止前须核对命令，避免 PID 复用）
- viewer: `http://127.0.0.1:8105`
- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -u -c 'import functools,runpy,viser; viser.ViserServer=functools.partial(viser.ViserServer,host="127.0.0.1"); runpy.run_path("/home/wbcd/workspace/oyx_ws/dexplore/data_processing/visualize_inspire_trajectory.py",run_name="__main__")' --all --source geometric --geometric-root data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/geometric --object-root data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/canonical/objects --port 8105 --start-frame 153`
- output: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/)
- config: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/config.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/config.json)
- run_manifest: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/run_manifest.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/run_manifest.json)
- check_only_log: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/check_only.log](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/check_only.log)
- viewer_log: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/viewer.log](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/viewer.log)
- smoke: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/smoke.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T035811Z/smoke.json)

**验证**

- 原 viewer `--all --source geometric ... --check-only`：退出 0；报告 656 条，首条 `s1_airplane_fly_1` 为 `[279,598]`，qpos `[373:391]`，18 个 actuated joints；frame 139 contact 时 fingertip/object-center 最小距离约 0.017 m，此量不是表面穿透或接触误差。
- 静态 URDF 解析为 18 个非 fixed joints、13 个 visual meshes；viewer log 确认加载 `canonical/objects/airplane/airplane.obj`、656 条轨迹和完整 geometric source。代码中手 mesh 每帧由 q FK 更新，物体 mesh由 tensor `198:205` 更新。
- `curl --noproxy '*' --max-time 10 http://127.0.0.1:8105` 返回 HTTP 200；`ss` 确认 PID 3936819 仅监听 127.0.0.1:8105；`ps` 命令与 manifest 一致。未执行浏览器截图或人工视觉复核。
- 外部 viewer SHA256 `f8bd976c375658c86e32dba8ef0212cfd1697ccd03407f9c0586bc33bfe3e5b9`；URDF SHA256 `7d0023168a22191de2126d71f8d2a810f82e8cba56b74c82be6db04812b80f58`。外部 DExplore 工作树保持干净。
- 交接前执行 `git diff --check` 和限定 scope 的 `audit_diff.py --worktree --check-links`；未修改 viewer 算法或输入。

**结论与保护边界**

- conclusion: `SUPPORTED`（当前页面确实直接用 geometric q 驱动完整 URDF mesh，并用 geometric object pose 显示物体；来源不含 RL/10135 点/V1.4 bilateral adapter）；`REFUTED`（V1.4.8 10135 点 viewer 等同于异机 8105 的解释）；`INCONCLUSIVE`（本机 656 与异机 660 的四条差异原因、重定向质量、穿透与接触保真）。
- 没有修改 DExplore、URDF、geometric tensor、contact/filter、物体 pose、正式 cache/index/split/GT、模型、训练配置或 checkpoint；没有补造四条数据、重跑 converter/filter 或训练。
- rollback: 核对 PID 3936819 命令后停止 8105，删除本次独立运行目录，并恢复 plan §16、README/experiment 说明、版本指针和本条 activity；已停止的 8147/8148 不自动重启。
- 规范反馈：无治理阻碍；本机缺少异机 1335/660 原归档使其 SHA256 同一性无法本机复核，已通过明确 provenance 避免冒充。

**原因**

恢复用户指定的真实可视化语义：查看 DExplore converter 产出的 native 18D geometric 轨迹和完整
URDF mesh，而不是任何派生手点云；同时诚实保留本机 656 与异机 660 的数据边界。
## 2026-09-16 04:04:44 +0000 — geometric mesh viewer 更换至 8106

- activity_id: `ACT-20260916-040444-OICM-GEOMETRIC-VIEWER-PORT`
- timestamp: `2026-09-16 04:04:44 +0000`
- modification_version: `V1.4.10`
- operation_category: `[operation, documentation]`
- primary_task_mode: `run-only/operation`
- change_level: `L0`（只更换本地监听端口，不改变数据或科研语义）
- approval: `user-approved`
- approval_basis: 用户明确要求“换一个端口”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- parent_activity_id: `ACT-20260916-040009-OICM-DEXPLORE-GEOMETRIC-MESH-VIEWER`
- plan: [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §16.1（final）
- scope: 将已验证的 656 条 DExplore geometric 完整 URDF mesh viewer 从 127.0.0.1:8105 移到 127.0.0.1:8106；输入、q/object slice、URDF、物体 mesh、初始轨迹/帧和 FPS 保持。

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针更新为 V1.4.10。
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) — 新增用户批准的 §16.1 端口迁移边界。
- [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/README.md](../../research/dexplore_geometric_visualization/README.md) 与 [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/experiment.yaml](../../research/dexplore_geometric_visualization/experiment.yaml) — 沿用 V1.4.9 已纠正的 656/660 与完整 mesh 说明，实验版本指针更新到 V1.4.10；README 本次无新增语义。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本条停止/启动状态与验证。

**原因**

按用户要求更换访问端口，同时保留原运行作为 STOPPED 历史证据，避免覆盖运行状态或改变已确认的
geometric mesh 可视化合同。

- stopped_run_id: `dexplore_geometric_mesh_20260916T035811Z`
- stopped_run_status: `STOPPED`
- stop_reason: 用户要求换端口；PID 3936819 核对命令后发送 SIGTERM，8105 已释放。
- run_id: `dexplore_geometric_mesh_20260916T040338Z`
- run_status: `RUNNING`
- PID: `3942453`；exec_session_id: `88608`（停止前须核对命令）
- viewer: `http://127.0.0.1:8106`
- output: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/)
- config: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/config.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/config.json)
- run_manifest: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/run_manifest.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/run_manifest.json)
- viewer_log: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/viewer.log](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/viewer.log)

**验证**

- 8106 HTTP 200，PID 3942453 仅监听 127.0.0.1:8106；viewer log 确认加载首条 object mesh 和 656 条轨迹。
- 8105 HTTP 000、无监听；进程命令除端口和独立 output 外与 V1.4.9 一致。
- `git diff --check` 与限定 scope 的 activity/link audit 在本条补全后执行。

- conclusion: `SUPPORTED`（仅端口迁移与服务 smoke）；原 V1.4.9 的重定向质量结论仍为 `INCONCLUSIVE`。
- protected_boundary: geometric tensor、URDF、物体、viewer 算法、RL、10135 点、正式 cache/index/split/GT、模型、配置和 checkpoint 均未修改。
- rollback: 核对 PID 3942453 后停止 8106 并删除本次独立 output；不自动恢复 8105。
- 规范反馈：无。
## 2026-09-16 04:09:08 +0000 — 本机 656 geometric 与异机 1335→660 链路差异诊断

- activity_id: `ACT-20260916-040908-OICM-GEOMETRIC-PRODUCER-DIAG`
- timestamp: `2026-09-16 04:09:08 +0000`
- modification_version: `V1.4.11`
- operation_category: `[diagnostic, operation, documentation]`
- primary_task_mode: `read-only/diagnostic`
- change_level: `L0`（只读 producer/provenance 审计及停止误导性 viewer）
- approval: `auto`
- approval_basis: 用户指出问题在几何重定向本身并要求确认是否真正按异机 DExplore geometric 链路导出；本次不重导或修改数据。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- parent_activity_id: `ACT-20260916-040444-OICM-GEOMETRIC-VIEWER-PORT`
- scope: 对照异机的 InterAct canonicalize→`adapt_interact_canonical.py`→全量 `convert_grab.py`→filter 链路，与本机 656 条 Cmv2 重导的实际命令、canonical producer、selection 和 converter；停止 8106 viewer。未运行 converter、filter、训练或数据修改。

**文件与证据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针更新到 V1.4.11。
- [src/task/ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py](../../../ObjectInteractionCmv2/tools/data/build_grab_dexplore_export.py) — 本机 656 条使用的自定义 canonical builder；它直接读取 raw GRAB、内部复现 upright/floor/heading canonicalization，并在 build 前按 contact/doorknob 固定 selection。
- [../dexplore/data_processing/adapt_interact_canonical.py](../../../../../../dexplore/data_processing/adapt_interact_canonical.py) — 异机链路使用的 adapter；它不重算 canonical transform，而是直接读取 InterAct `sequences_canonical/<seq>/human.npz` 和 `object.npz`，再拼回 raw contact/metadata。
- [../dexplore/data_processing/convert_grab.py](../../../../../../dexplore/data_processing/convert_grab.py) — 两条链路最后都调用的 DExplore geometric converter；使用 SMPL-X 关键点和 `dex_retargeting` position optimizer，按 `INSPIRE_RETARGET_REORDER` 写入 `[373:391]`。
- [本机 full commands.log](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/full_20260915T125105Z/commands.log) — 明确先运行 `python -m src.task.ObjectInteractionCmv2.tools.data.build_grab_dexplore_export build --selection ... --stride 4`，随后才调用外部 `convert_grab.py`；没有运行 InterAct `process_grab.py`、`canonicalize_human_multi_thread.py` 或 `adapt_interact_canonical.py`。
- [本机 selection.json](../../../../../data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/selection.json) — 在 canonical/convert 前即选出 656 条：1335 source、662 left-contact excluded、17 doorknob excluded；与异机用户给出的 658/17/660 不同。
- 当前 `/home/wbcd/workspace/oyx_ws/InterAct/data/grab/sequences_canonical` 下 `human.npz` 数量为 0；本机也没有异机 `inspire_geometric`/`inspire_geometric_dexplore` 根，无法做 canonical 或 tensor 逐值/SHA256 对照。

**原因**

回答“是否和异机一样”必须区分最后的优化器与端到端 producer。仅看到文件名、q slice 或调用
`convert_grab.py` 不足以证明输入 canonical 人体/物体相同；几何优化对输入关键点敏感。

**运行状态**

- run_id: `dexplore_geometric_mesh_20260916T040338Z`
- run_status: `STOPPED`
- PID 3942453 在核对命令后发送 SIGTERM；127.0.0.1:8106 已释放。
- stop_reason: 用户指出几何重定向不一致；避免继续把本机 656 重导显示为异机 660 的等价结果。
- run_manifest: [src/task/ObjectInteractionCm/research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/run_manifest.json](../../research/dexplore_geometric_visualization/output/dexplore_geometric_mesh_20260916T040338Z/run_manifest.json)

**验证**

- 静态读取本机 commands/selection/manifest、自定义 canonical builder、异机 adapter 和 DExplore converter；本机命令证实 `convert_grab.py --robot inspire --retarget-iterations 1 --retarget-stride 1`，因此“使用 DExplore geometric optimizer”成立。
- 本机完整构建入口和异机 InterAct adapter 不同，且筛选顺序/计数不同；因此“端到端与异机相同、tensor 可视为同一批”不成立。
- `ss` 确认 8106 无监听；未改外部 DExplore、GRAB、InterAct、tensor、URDF、contact/filter、cache、模型或 checkpoint。

**结论与保护边界**

- conclusion: `SUPPORTED`（本机 `.pt` 最后确实由 DExplore `convert_grab.py`/`dex_retargeting` geometric position optimizer 导出）；`REFUTED`（本机 656 使用了与异机完全相同的端到端 canonical→adapter→全量→filter 链路）；`INCONCLUSIVE`（几何异常具体由 custom canonical 的哪一项差异导致，以及异机 660 的四条计数差异）。
- 下一步若要严格复现，应优先取得异机的一条 `sequences_canonical` human/object + full/filtered geometric tensor 做逐值/SHA 对照；若无法复制，则需经用户批准运行完整 InterAct canonicalization、adapter、1335 converter 和 filter，这属于新的长时 L3 数据运行。
- rollback: 本次无数据/代码实现需回滚；恢复 8106 只会恢复已确认不等价的查看器，不建议自动执行。活动/版本增量可按显式路径撤回，不能 reset dirty 工作区。
- 规范反馈：无治理阻碍；问题来自此前把“相同末端 converter”误当成“相同端到端 producer”。
## 2026-09-16 04:49:05 +0000 — 正式 InterAct→DExplore exact-chain 单序列 pilot 启动

- activity_id: `ACT-20260916-044905-OICM-INTERACT-DEXPLORE-EXACT-PILOT-START`
- timestamp: `2026-09-16 04:49:05 +0000`
- modification_version: `V1.4.12`
- type: `data / operation / diagnostic`
- task_mode: `change` 与 `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 用户在确认现有 656 条只复用了末端 DExplore optimizer、未复现另一台机器完整链路后明确要求“你继续吧”；V1.4 final 指导与 plan 已批准完整 GRAB、正式 InterAct canonical 和 DExplore geometric 独立导出。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留当前 ObjectInteractionCm、ObjectInteractionCmv2、Cm 及根文档的既有未提交修改）
- run_id: `grab-interact-dexplore-exact-pilot-s1-airplane-fly-1-20260916T044905Z`
- run_status: `STARTED`
- scope: 仅对 `s1_airplane_fly_1` 运行正式 InterAct `process_grab`、`canonicalize_human_multi_thread`、DExplore canonical adapter 和 `convert_grab.py`；在独立目录中验证后再决定是否扩展到 1335 条，不覆盖现有 656 条 custom-canonical 数据。

**文件与产物**

- [`src/task/ObjectInteractionCm/docs/指导/V1.4.md`](../指导/V1.4.md) — 已批准的完整 GRAB 与 DExplore geometric 研究边界。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — final 执行计划、坐标停止条件和回滚边界。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — 将 ObjectInteractionCm 指针推进到 `V1.4.12`；同文件其他既有修改不属于本次范围。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/) — 独立 pilot 运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/run_manifest.json) — `STARTED` 运行合同。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/commands.log` — PENDING。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/validation.json` — PENDING。

**原因**

现有 656 条数据的末端虽调用同一 DExplore optimizer，但上游使用 Ref2Dex 自定义 canonicalization。该 pilot 改为另一台机器记录的正式 InterAct canonical 输入，先消除坐标与手关键点来源差异，再评估 Inspire FK 几何。

**验证**

- 每阶段必须生成且仅生成 `s1_airplane_fly_1`，InterAct 输出为 30 Hz，DExplore tensor shape 为 `[T,598]`，全部有限。
- 对比正式 InterAct 与旧 custom canonical 的 human/object 位姿和最终 `373:391` q；任何帧长、坐标、对象位姿或 FK 手物对齐异常均停止，不扩大到全量。
- 当前状态：`STARTED`；工程与科研结论均为 `INCONCLUSIVE`。

**回滚**

停止运行并删除本次独立 pilot 目录、恢复 `ObjectInteractionCm` 指针即可；不触碰原始 GRAB、现有 656 条输出、V1.3 cache、checkpoint、外部仓库代码或其他运行。

## 2026-09-16 04:55:59 +0000 — 正式 InterAct→DExplore exact-chain 单序列 pilot 完成

- activity_id: `ACT-20260916-045559-OICM-INTERACT-DEXPLORE-EXACT-PILOT-COMPLETE`
- timestamp: `2026-09-16 04:55:59 +0000`
- modification_version: `V1.4.12`
- type: `code / data / operation / diagnostic`
- task_mode: `change` 与 `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 延续 `ACT-20260916-044905-OICM-INTERACT-DEXPLORE-EXACT-PILOT-START`；用户已批准复现另一台机器完整链路，pilot 按 final V1.4 plan 的停止条件执行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留所有既有未提交改动）
- run_id: `grab-interact-dexplore-exact-pilot-s1-airplane-fly-1-20260916T044905Z`
- run_status: `COMPLETED`
- last_step: `N/A`（数据处理）
- last_epoch: `N/A`（数据处理）
- best_metric: `N/A`（数据处理）
- checkpoint: `N/A`（纯 geometric retarget）
- scope: 完成单条正式 InterAct process/canonicalize、DExplore adapter/converter 及新旧链路逐值/FK 对照；未启动 1335 条全量、filter、RL rollout、cache 或训练。

**文件与产物**

- [`src/task/ObjectInteractionCm/tools/data/run_interact_grab_canonical.py`](../../tools/data/run_interact_grab_canonical.py) — 仅屏蔽 GRAB 分支未使用的 import-time SMPL-H/BodyPrior 初始化，随后直接调用外部 InterAct 原版 `process_dataset('grab', ...)`。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 当前指针 `V1.4.12`；其他既有修改保持不变。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/) — 138 MiB pilot 运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/run_manifest.json) — `COMPLETED` 依赖、输入/输出 hash 与运行合同。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/commands.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/commands.log) — 启动兼容问题、正式四阶段命令和输出。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/validation.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/validation.json) — 帧数、finite、canonical/q 差异与 FK 检查。

**原因**

pilot 证实旧 custom canonical 与正式 InterAct canonical 在该序列上几乎逐值一致，但单条 geometric q 与旧 656 条批处理结果显著不同。继续把几何异常归因于 canonicalization 不符合证据；必须考虑 converter 在排序后的序列循环中复用同一个 retargeting optimizer、跨序列 warm-start 的实际行为。

**验证**

- 正式四阶段均生成且仅生成 `s1_airplane_fly_1`；279 帧、tensor `[279,598]`、全部 finite，DExplore converter 为 `1 succeeded, 0 failed`。
- 新旧 canonical：human pose 最大差 `5.74e-08`，human translation 最大差 `4.17e-07 m`，object translation 最大差 `4.16e-07 m`，`[198:205]` 最大差 `1.19e-07`。
- 新旧 `q[373:391]` 最大差 `16.51 rad`、平均绝对差 `1.90 rad`；新链 contact frame 139 的最小指尖到物体中心距离 `0.011 m`，旧链为 `0.017 m`。
- 代码审计确认 `convert_grab.py` 在 sorted sequence loop 之前只构建一次 retargeting，且不在序列边界 reset；因此 `--filter` 单序列结果不能复现 1335 条全量运行进入该序列时的 optimizer state。
- `python -m py_compile src/task/ObjectInteractionCm/tools/data/run_interact_grab_canonical.py`：通过。工程结论：`SUPPORTED`；哪一份 q 的视觉/碰撞质量更好及完整 660 条是否与异机逐值相同仍为 `INCONCLUSIVE`。

**回滚**

删除新增 GRAB-only 启动器与独立 pilot 目录，并恢复本次版本/activity 增量；原始 GRAB、外部 InterAct/DExplore、旧 656 条、V1.3 数据、checkpoint 和其他运行均未修改。

## 2026-09-16 04:57:05 +0000 — 正式 InterAct→DExplore 1335 条全量 geometric 导出启动

- activity_id: `ACT-20260916-045705-OICM-INTERACT-DEXPLORE-EXACT-FULL-START`
- timestamp: `2026-09-16 04:57:05 +0000`
- modification_version: `V1.4.12`
- type: `data / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确要求继续复现另一台机器完整链路；`ACT-20260916-045559-OICM-INTERACT-DEXPLORE-EXACT-PILOT-COMPLETE` 已证明四阶段单序列可运行，并发现必须保留 1335 条 sorted loop 的跨序列 optimizer warm-start 才能复现批处理语义。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `grab-interact-dexplore-exact-full-20260916T045705Z`
- run_status: `STARTED`
- scope: 在新目录依次生成正式 InterAct processed/canonical 1335 条、DExplore converter 输入、完整 geometric 1335 条，再按原 filter 的 left-contact/doorknob 规则物化 geometric-only DExplore 子集；不运行 RL rollout、cache 或训练。

**文件与产物**

- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — 已批准的 L3 全量数据处理、验证和回滚计划。
- [`src/task/ObjectInteractionCm/tools/data/run_interact_grab_canonical.py`](../../tools/data/run_interact_grab_canonical.py) — GRAB-only 原版 InterAct canonicalizer 启动入口。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/) — 独立全量运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/run_manifest.json) — `STARTED` 运行合同。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/commands.log` — PENDING。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/validation.json` — PENDING。

**原因**

单序列 pilot 的 canonical 与旧 custom 数据等价，但 q 不等价；代码确认 converter 的 optimizer 状态跨 sorted sequence loop 延续。为匹配异机“先完整 1335 geometric、再筛 660”的实际语义，必须运行完整序列顺序，而不能在 converter 前筛选或用 `--filter` 逐条生成。

**验证**

- 输入枚举为 1335 条，GPU 3 空闲；GPU 1、2 的既有外部训练不停止、不抢占。
- 每阶段核对 sequence 集、帧数、finite、shape、失败清单；converter 后按 tensor contact 列 `206:222` 与名称 doorknob 执行相同筛选，并记录实际 658/17/660 是否复现。
- 当前状态：`STARTED`；工程与科研结论均为 `INCONCLUSIVE`。

**回滚**

安全停止当前进程并删除独立 `full_20260916T045705Z` 目录即可；pilot、旧 656 条、原始 GRAB、外部仓库代码、V1.3、checkpoint 和其他运行均保持不变。

## 2026-09-16 05:10:19 +0000 — 正式 InterAct→DExplore 全量运行进入 process_grab

- activity_id: `ACT-20260916-051019-OICM-INTERACT-DEXPLORE-EXACT-FULL-RUNNING`
- timestamp: `2026-09-16 05:10:19 +0000`
- modification_version: `V1.4.12`
- type: `data / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续已批准的 `ACT-20260916-045705-OICM-INTERACT-DEXPLORE-EXACT-FULL-START`；用户再次要求“继续”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `grab-interact-dexplore-exact-full-20260916T045705Z`
- run_status: `RUNNING`
- command: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/home/wbcd/workspace/oyx_ws/InterAct:/home/wbcd/workspace/oyx_ws/InterAct/text2interaction /home/wbcd/miniconda3/envs/graspenv/bin/python -u /home/wbcd/workspace/oyx_ws/InterAct/process/process_grab.py`
- process: wrapper PID `3997929`，worker PID `3997936`，GPU `3`。
- scope: 从完整 1335 条只读 GRAB 输入生成独立 InterAct processed staging；后续阶段尚未启动，旧 656 条、pilot、原始数据和 GPU 1/2 既有训练均未修改。

**文件与产物**

- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/) — 独立全量运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/run_manifest.json) — 当前阶段、PID 与 `RUNNING` 状态。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/commands.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/commands.log) — 当前命令日志；进程结束时完成刷新。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/validation.json` — PENDING；终态前生成。

**原因**

恢复会话后确认此前只建立了运行合同，没有启动全量进程。本次从既有断点启动第一阶段，不重复 pilot，也不改变已批准的数据语义。

**验证**

- 启动前输入仍为 1335 条，GPU 3 空闲；启动后 worker 存活并持续生成 `data/grab/sequences/`。
- 初始检查已生成 23 条 processed sequence；运行按实验工作流采用分钟级等待。
- 当前仅证明进程已启动，最终完整性与几何质量为 `INCONCLUSIVE`。

**回滚**

向 wrapper/worker 发送安全停止信号并保留或删除本次独立运行目录；不影响任何旧数据或其他 GPU 进程。

## 2026-09-16 05:19:25 +0000 — 全量导出停止并切换到 12 条 optimizer-prefix 可视化

- activity_id: `ACT-20260916-051925-OICM-INTERACT-DEXPLORE-PREFIX12-START`
- timestamp: `2026-09-16 05:19:25 +0000`
- modification_version: `V1.4.13`
- type: `data / diagnostic / operation / documentation`
- task_mode: `change` 与 `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 用户明确要求“先不要完整导出，用一部分给我可视化再导出”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- stopped_run_id: `grab-interact-dexplore-exact-full-20260916T045705Z`
- stopped_run_status: `STOPPED`
- stopped_stage: `interact_process_grab`，已生成 `433/1335` 条独立 processed staging；canonical、adapter、converter、filter 均未启动。
- run_id: `grab-interact-dexplore-exact-prefix12-20260916T051925Z`
- run_status: `STARTED`
- scope: 安全停止全量 worker；新建一次性 prefix pilot，按未来全量 lexicographic 顺序处理前 12 条并用完整 URDF mesh viewer 可视化。人工确认前不恢复全量。

**文件与产物**

- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §17 固化 12 条 strict prefix、warm-start、viewer 和停止边界。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进到 `V1.4.13`。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/) — 已停止的全量 staging，保留不覆盖。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/full_20260916T045705Z/run_manifest.json) — `STOPPED`、433 条和停止原因。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/` — PENDING；新的 prefix 运行目录。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/run_manifest.json` — PENDING。

**原因**

随意抽样或单条 converter 会改变已确认存在的跨序列 optimizer warm-start，无法代表未来全量。使用完整排序的严格前缀既满足“先看部分”，又使这 12 条在未来全量中可逐值复现。

**验证**

- 已向 worker PID `3997936` 发送 `SIGINT`，wrapper/worker 均退出；GPU 1/2 既有进程未改动。
- 停止时独立 staging 有 433 条 processed sequence，命令日志已刷新；原始 GRAB 和旧输出均未写入。
- prefix 尚未完成，工程与几何质量结论均为 `INCONCLUSIVE`。

**回滚**

停止 prefix 进程/viewer并删除其独立目录；保留或删除已停止全量 staging均不影响原始数据。全量 run 不自动恢复。

## 2026-09-16 05:25:53 +0000 — 12 条 exact prefix geometric 导出完成并启动完整 mesh viewer

- activity_id: `ACT-20260916-052553-OICM-INTERACT-DEXPLORE-PREFIX12-VIEWER`
- timestamp: `2026-09-16 05:25:53 +0000`
- modification_version: `V1.4.13`
- type: `data / diagnostic / operation`
- task_mode: `run-only/operation`
- change_level: `L2 / L3`
- approval: `user-approved`
- approval_basis: 延续用户“先不要完整导出，用一部分给我可视化再导出”的明确要求与 V1.4 plan §17。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `grab-interact-dexplore-exact-prefix12-20260916T051925Z`
- run_status: `RUNNING`（数据阶段完成；等待用户视觉复核）
- viewer_run_id: `grab-interact-dexplore-prefix12-viewer-20260916T052329Z`
- viewer_run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8107`
- process: wrapper PID `4019264`，worker PID `4019276`，HTTP `200`。
- last_step / last_epoch / best_metric / checkpoint: `N/A`（数据导出与可视化，无训练）。
- scope: 完成全量排序前 12 条的正式 InterAct→DExplore geometric 一次性批处理，并用 native `373:391` q、完整 URDF mesh 和 `198:205` reference object 启动浏览器；1335 条全量仍为 STOPPED。
- conclusion: `SUPPORTED`（12 条数据导出、tensor 合同、viewer/FK/object mesh 工程链路）；`INCONCLUSIVE`（几何重定向视觉质量、穿透/接触保真以及是否批准全量）。

**文件与产物**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 当前指针 `V1.4.13`。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §17 记录本次用户批准的 prefix-first 执行顺序与保护边界。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/) — prefix 运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/run_manifest.json) — 数据完成、viewer 关联和当前 visual review 状态。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/selected_sequences.txt`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/selected_sequences.txt) — 固定的全量排序前 12 条。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/commands.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/commands.log) — 四阶段命令、初始 staging 兼容错误与成功重试记录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/validation.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/validation.json) — 12/12、帧数、shape、finite 和 viewer check。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/geometric/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/geometric/) — 12 条 geometric tensor。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer/run_manifest.json) — 8107 viewer 状态、PID 和 HTTP 证据。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer/viewer.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer/viewer.log) — URDF FK、object mesh、Viser 启动日志。

**原因**

将人工复核放在全量前，并严格使用 converter 的排序前缀，使这 12 条在未来全量中保留相同 optimizer warm-start 状态。第一次 viewer 未显式传 object root，只显示手；已停止该实例并用 prefix InterAct object assets 原样重启。

**验证**

- 四阶段计数均为 12，DExplore converter：`12 succeeded, 0 failed`；所有 tensor `[T,598]` 且 finite，序列集合和顺序与 `selected_sequences.txt` 完全一致。
- viewer `--check-only`：12 条、18 个 URDF actuated joints、q columns `[373:391]`；默认 `s10_airplane_fly_1` 为 212 帧，frame 106 最小指尖到 object center 距离约 `0.027 m`。
- 在线 viewer 显式加载 `airplane.obj`，8107 HTTP 返回 `200`；完整手 mesh 与 reference object 均可见。以上是工程 smoke，不是几何质量结论。

**回滚**

停止 PID `4019264/4019276` 并移除 prefix 独立目录；全量 run 继续保持 STOPPED，不自动恢复。原始和旧数据不受影响。

## 2026-09-16 07:13:13 +0000 — 12 条 prefix viewer 从 8107 切换到 8108

- activity_id: `ACT-20260916-071313-OICM-PREFIX12-VIEWER-PORT-8108`
- timestamp: `2026-09-16 07:13:13 +0000`
- modification_version: `V1.4.14`
- type: `operation / documentation`
- task_mode: `run-only/operation`
- change_level: `L0`
- approval: `user-approved`
- approval_basis: 用户明确要求“换一个端口”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- stopped_run_id: `grab-interact-dexplore-prefix12-viewer-20260916T052329Z`
- stopped_run_status: `STOPPED`
- run_id: `grab-interact-dexplore-prefix12-viewer-8108-20260916T071155Z`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8108`
- process: wrapper PID `4083441`，worker PID `4083442`，HTTP `200`。
- scope: 仅更换同一 12 条 prefix geometric viewer 的监听端口和独立运行记录；数据、序列、q、物体、URDF、默认序列和全量停止状态保持不变。
- conclusion: `SUPPORTED`（端口切换和 viewer 工程 smoke）；几何质量仍为 `INCONCLUSIVE`。

**文件与产物**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进到 `V1.4.14`。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §17.1 记录同参数端口切换与回滚边界。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/run_manifest.json) — 父运行改指向 8108 viewer。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer/run_manifest.json) — 8107 旧实例 `STOPPED`。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/) — 8108 独立运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/run_manifest.json) — `RUNNING`、PID 与 HTTP 状态。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/viewer.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/viewer.log) — URDF、物体与 Viser 启动日志。

**原因**

按用户要求更换端口，同时保留旧 viewer 终态与新 viewer 独立证据，不覆盖先前运行记录。

**验证**

- 8107 已停止，HTTP `000` 且无监听进程；8108 HTTP `200`，worker PID `4083442` 正在监听。
- 新实例加载 12 条相同轨迹、相同 `airplane.obj` 与完整 18-joint Inspire URDF；启动日志未出现数据或 mesh 缺失。
- 本次仅为运行端口 smoke，不构成新的几何或科研结论。

**回滚**

停止 PID `4083441/4083442`；不自动恢复 8107，也不恢复已暂停的 1335 条全量导出。

## 2026-09-16 07:26:29 +0000 — DExplore Inspire geometric 解唯一性与异机差异诊断

- activity_id: `ACT-20260916-072629-OICM-DEXPLORE-RETARGET-UNIQUENESS`
- timestamp: `2026-09-16 07:26:29 +0000`
- modification_version: `V1.4.14`
- type: `diagnostic / documentation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户询问当前部分导出为何与另一台机器观感不同，以及重定向结果是否唯一；本轮只读核对 converter、dex-retargeting、运行环境和 URDF hash，并追加诊断记录。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- scope: 判断 geometric retarget 的解唯一性、确定性来源和异机复现条件；不修改 converter、optimizer、资产、tensor、viewer、全量暂停状态或科研结论。
- conclusion: `SUPPORTED`（当前求解不是全局唯一，且对初值、帧/序列顺序和求解器版本路径依赖）；`INCONCLUSIVE`（缺少异机同序列 tensor 与环境 hash，不能确定具体差异源）。

**文件与证据**

- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/validation.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/validation.json) — 当前 12 条 prefix 的 shape/finite/FK 工程证据。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_13/prefix12_20260916T051925Z/viewer_8108_20260916T071155Z/run_manifest.json) — 当前 viewer 仍为 8108 `RUNNING`。
- `dexplore/data_processing/convert_grab.py`（外部只读）— 当前 Inspire 分支将 18 个 active joint 全部设为优化变量，一次构造 retargeting 后按排序序列和逐帧连续调用，不在序列边界 reset。
- `dex-retargeting/dex_retargeting/seq_retarget.py` 与 `optimizer.py`（外部只读）— `last_qpos` 作为下一次 SLSQP 初值；position 目标仅为 5 个指尖位置，并在梯度中加入相对上一解的正则。

**原因**

1. 当前优化变量为 18D，而位置目标只有 5 个指尖的 15 个坐标，几何约束本身不足以保证唯一关节解；关节限位和 temporal prior 只是在多解中选取路径相关的局部解。
2. `SeqRetargeting.last_qpos` 每帧更新，且 converter 对全部排序序列复用同一实例；因此前一帧、前一序列、筛选方式、处理顺序和调用次数都会改变后续解。
3. 求解器为 NLopt `LD_SLSQP`。当前 `PositionOptimizer` 返回的 objective value 只有 Huber 位置损失，但提供给求解器的 gradient 额外包含 temporal norm 项；值与梯度并非同一标量目标的严格导数，可能进一步放大不同 NLopt/数值栈的路径敏感性。
4. q 求解主路径没有随机采样；对象 surface sampling 固定 seed=2024，且不参与默认 q 优化。因此在完全相同输入、排序、converter/dex-retargeting/URDF、NLopt/Pinocchio/NumPy/Torch 版本和调用次数下，应当可重复到很接近，而不应把明显异机差异解释为普通随机波动。

**验证**

- 本机实际加载 `dex-retargeting 0.4.6`，源码来自 commit `8632b2cab32e1b51ce379940c414a0f78332ff6b`；环境为 NLopt `2.7.1`、Pinocchio `2.7.0`、NumPy `1.24.4`、Torch `2.0.1+cu118`。
- converter commit `c31f57f186409ce5f0de47ced2d347abffe45d06`，`convert_grab.py` SHA256 `0662ade4…aeea`；右手 URDF SHA256 `7d002316…0f58`。
- converter 的 `inspire_hand/` 与 viewer 的 `inspire_hand_new/` 右手 URDF SHA256 完全相同，排除当前优化资产与显示资产不一致。
- 未运行新数据处理或优化器实验；以上为源码/环境诊断，不证明哪台机器的视觉结果更正确。

**下一步与回滚**

最小判别应直接取得异机同一 sequence 的 `interaction_hand_inspire.pt`，先比较 `[198:205]`、`[373:391]` 和 SHA256，再核对上述四类源码/资产/依赖 hash。若 object pose 相同而 q 不同，差异锁定在 optimizer 状态或软件栈；若 q 相同而显示不同，差异锁定在 viewer/资产。删除本诊断条目即可回滚文档记录；运行和数据无需回滚。

## 2026-09-16 07:30:49 +0000 — exact-chain `s1_airplane_fly_1` 完整 mesh viewer 启动

- activity_id: `ACT-20260916-073049-OICM-EXACT-S1-AIRPLANE-VIEWER`
- timestamp: `2026-09-16 07:30:49 +0000`
- modification_version: `V1.4.15`
- type: `diagnostic / operation / documentation`
- task_mode: `run-only/operation`
- change_level: `L0`
- approval: `user-approved`
- approval_basis: 用户判断 `.pt` 应一致并明确要求“用 s1_airplane_fly_1 试试看，给我可视化”。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- stopped_run_id: `grab-interact-dexplore-prefix12-viewer-8108-20260916T071155Z`
- stopped_run_status: `STOPPED`
- run_id: `grab-interact-dexplore-exact-s1-airplane-viewer-8108-20260916T072922Z`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8108`
- process: wrapper PID `4101873`，worker PID `4101880`，HTTP `200`。
- scope: 使用已完成 exact-chain 单序列 geometric tensor 显示 `s1_airplane_fly_1`；不重新优化、不读取 custom 656/RL、不修改数据、资产、viewer 或全量暂停状态。
- conclusion: `SUPPORTED`（同名序列 tensor 读取、native q FK、完整 mesh/object viewer 工程链路）；异机数值一致性与视觉质量仍为 `INCONCLUSIVE`。

**文件与产物**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进到 `V1.4.15`。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §17.2 固化同名序列对照与 warm-start 边界。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/run_manifest.json) — exact-chain parent run 与输入/输出 hash。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/validation.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/validation.json) — 279 帧、finite 与 frame 139 FK 检查。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/) — 新 viewer 独立运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/run_manifest.json) — `RUNNING`、PID、tensor SHA256 与 HTTP 状态。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/viewer.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/viewer.log) — URDF FK、object mesh 与 Viser 启动日志。

**原因**

用同名 `s1_airplane_fly_1` 排除上一页面默认 `s10_airplane_fly_1` 带来的受试者/动作差异，并避免重新生成数据引入新的变量。

**验证**

- 输入 tensor SHA256 `ef7c31e7…330ad2`，279 帧；启动时 frame 139 最小指尖到 object center 距离约 `0.011 m`。
- viewer 明确加载 prefix pilot 的 `airplane.obj` 与完整 18-joint Inspire URDF；8108 HTTP `200`。
- 当前 tensor 是从 optimizer 初始状态开始的单序列结果，不包含 full-batch 前序 warm-start；因此视觉 smoke 不能替代异机 `.pt` 逐值比较。

**回滚**

停止 PID `4101873/4101880`；不恢复 prefix12 viewer，也不恢复已暂停的 1335 条全量导出。

## 2026-09-16 07:34:24 +0000 — 按用户纠正将 s1_airplane_fly_1 viewer 从 8108 切换到 8112

- activity_id: `ACT-20260916-073424-OICM-EXACT-S1-AIRPLANE-PORT-8112`
- timestamp: `2026-09-16 07:34:24 +0000`
- modification_version: `V1.4.16`
- type: `operation / documentation`
- task_mode: `run-only/operation`
- change_level: `L0`
- approval: `user-approved`
- approval_basis: 用户明确纠正“不要用这个端口”，并指定换成 `8112`。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- stopped_run_id: `grab-interact-dexplore-exact-s1-airplane-viewer-8108-20260916T072922Z`
- stopped_run_status: `STOPPED`
- run_id: `grab-interact-dexplore-exact-s1-airplane-viewer-8112-20260916T073334Z`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: wrapper PID `4106340`，worker PID `4106366`，HTTP `200`；旧端口 8108 为 HTTP `000`（无监听）。
- scope: 仅迁移同一个 `s1_airplane_fly_1` viewer 的监听端口和运行记录；不重新优化、不修改 tensor、数据、资产、viewer 实现或全量暂停状态。
- conclusion: `SUPPORTED`（8112 viewer 工程链路）；异机数值一致性仍为 `INCONCLUSIVE`。

**文件与产物**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进到 `V1.4.16`。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8108_20260916T072922Z/run_manifest.json) — 旧运行终态 `STOPPED` 及原因。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8112_20260916T073334Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8112_20260916T073334Z/) — 新 viewer 独立运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8112_20260916T073334Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8112_20260916T073334Z/run_manifest.json) — `RUNNING`、PID、tensor SHA256 与 HTTP 状态。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8112_20260916T073334Z/viewer.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_12/pilot_s1_airplane_fly_1_20260916T044905Z/viewer_8112_20260916T073334Z/viewer.log) — 启动与 FK/contact 检查日志。

**原因**

前一轮错误复用了端口 8108；本次严格按用户指定改为 8112，并保留旧运行终态供审计。

**验证**

- `curl http://127.0.0.1:8112/` 返回 HTTP `200`；`curl http://127.0.0.1:8108/` 返回 HTTP `000`。
- 新 viewer 进程仅携带 `--port 8112`；序列仍为 `s1_airplane_fly_1`，起始帧仍为 139。
- 新旧 manifest 与 config 通过 `python -m json.tool`；最新 activity 通过 `audit_diff.py --check-links`。

**回滚**

停止 wrapper PID `4106340`（会连带停止 worker PID `4106366`），并将新 viewer manifest 更新为 `STOPPED`；不会删除或改写 tensor。

## 2026-09-16 07:46:37 +0000 — NAS 全量 GRAB Inspire 导出重启并启动 ARCTIC MANO viewer

- activity_id: `ACT-20260916-074637-OICM-FULL-NAS-EXPORT-ARCTIC-MANO`
- timestamp: `2026-09-16 07:46:37 +0000`
- modification_version: `V1.4.18`
- type: `data / diagnostic / operation / documentation`
- task_mode: `run-only/operation`（plan/activity/manifest 同步记录）
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认 Inspire 视觉结果无问题，明确要求开始完整 GPU 导出并查看 ARCTIC MANO；随后明确要求导出必须位于 NAS、不得放本地。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `grab-interact-dexplore-exact-full-20260916T074233Z`
- run_status: `RUNNING`
- current_stage: `interact_process_grab`
- process: wrapper PID `4119323`，worker PID `4119331`，物理 GPU `3`，复核时显存约 `10680 MiB`。
- progress_at_check: `46/1335` processed staging；终态 sequence/frame/finite/失败清单仍为 `PENDING`。
- viewer_run_id: `arctic_mano_viewer_20260916T074556Z`
- viewer_run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8113`，PID `4121020`，HTTP `200`。
- scope: 在 NAS 新目录运行完整 1335 条 GRAB exact InterAct→DExplore geometric 链路；ARCTIC viewer 复用既有 bilateral MANO viewer 和 `s01/box_use_01`。不启动 ARCTIC/OakInk2 Inspire 导出、cache、scale、训练或 RL，不修改原始数据、V1.3、checkpoint、GPU 1/2 运行或当前 8112 viewer。
- conclusion: `SUPPORTED`（NAS 路由、GPU 3 进程和 ARCTIC MANO viewer 工程启动）；完整导出与几何质量仍为 `INCONCLUSIVE`。

**文件与产物**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进到 `V1.4.18`。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §18–18.1 固化全量范围、GPU、ARCTIC viewer 与 NAS 边界。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/) — 逻辑路径；实际解析到 NAS NFS。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/run_manifest.json) — `RUNNING`、NAS resolved root、PID、GPU 和误停/重启记录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/commands.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/commands.log) — 当前阶段实时日志。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/validation.json` — PENDING，终态前生成。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_viewer_20260916T074556Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_viewer_20260916T074556Z/) — NAS viewer 运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_viewer_20260916T074556Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_viewer_20260916T074556Z/run_manifest.json) — ARCTIC/GRAB MANO 输入、PID 与 HTTP 状态。

**原因**

用户已完成 prefix/同名序列视觉复核，允许进入完整导出；所有批量产物必须留在 NAS。`data/processed_data` 经 `readlink -f`、inode/device 和 `findmnt` 核验为 NAS 软链接。Agent 曾误看仓库根容量而在 29 条时停止，确认后从头覆盖这 29 条并重启，避免误留不完整状态。

**验证**

- `data/processed_data` 解析为 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data`；逻辑路径与 NAS 绝对路径的 device/inode 均为 `62/525666022`，NFS 可用约 `52 TiB`。
- worker PID `4119331` 只占用 GPU 3；GPU 1/2 既有进程未改动。复核时 staging 已从头推进至 46 条。
- 8113 HTTP `200`；viewer 列出 `GRAB / s1/airplane_fly_1` 与 `ARCTIC / s01/box_use_01`。页面需在下拉框选择后者。
- 当前只是工程启动证据；完整性和科研质量保持 `INCONCLUSIVE`。

**回滚**

安全停止 wrapper PID `4119323`/worker PID `4119331` 与 viewer PID `4121020`，将两个 manifest 更新为 `STOPPED`；保留 NAS 运行目录供审计，不删除原始或既有产物。

## 2026-09-16 07:53:06 +0000 — ARCTIC MANO 切换为 KNN/累计距离着色 viewer

- activity_id: `ACT-20260916-075306-OICM-ARCTIC-MANO-KNN-VIEWER`
- timestamp: `2026-09-16 07:53:06 +0000`
- modification_version: `V1.4.19`
- type: `code / diagnostic / operation / documentation`
- task_mode: `change` 后切回 `run-only/operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确纠正需要此前可查看 KNN 和距离着色的脚本，并要求以后统一使用该脚本；同时已授权兼容困难时重写/适配。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- stopped_viewer_run_id: `arctic_mano_viewer_20260916T074556Z`
- stopped_viewer_run_status: `STOPPED`
- run_id: `arctic_mano_knn_viewer_20260916T075013Z/viewer`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8113`
- process: PID `4128922`，HTTP `200`。
- concurrent_export: `grab-interact-dexplore-exact-full-20260916T074233Z` 仍为 `RUNNING`，GPU 3，复核时 `330/1335` processed staging。
- scope: 复用 `src.task.ObjectInteractionCm.visualize_grab` 的红色累计距离与黄色 object-to-hand KNN 着色；仅补充 ARCTIC raw bilateral MANO 的无单刚体 pose/无 mesh viewer-only 兼容。不修改 raw NPZ、训练 index、cache、GT、split、全量导出语义或其他运行。
- conclusion: `SUPPORTED`（ARCTIC 889 帧 point/KNN/distance viewer 工程链路）；mesh、完整导出和科研质量仍为 `INCONCLUSIVE`。

**文件与产物**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进到 `V1.4.19`。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §18.2 固化以后统一使用 KNN/距离 viewer 的用户要求与兼容边界。
- [`src/task/ObjectInteractionCm/visualize_grab.py`](../../visualize_grab.py) — 支持 ARCTIC 无单刚体 pose、无 MANO mesh 的 point-only viewer；禁止伪造 articulated object mesh。
- [`src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`](../../tests/test_visualize_grab_v1_4.py) — 新增 ARCTIC 缺失 pose/mesh 定向回归。
- [`src/task/ObjectInteractionCm/docs/logs/repo_memory.md`](repo_memory.md) — 记录默认 viewer 固定偏好。
- [`src/task/ObjectInteractionCm/docs/README.md`](../README.md) — 增加 Task repo memory 导航。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_knn_viewer_20260916T075013Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_knn_viewer_20260916T075013Z/) — NAS viewer-only index 与运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_knn_viewer_20260916T075013Z/viewer/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/arctic_mano_knn_viewer_20260916T075013Z/viewer/run_manifest.json) — PID、HTTP、着色能力与输入快照。

**原因**

前一实例只显示普通 MANO 点/mesh，不具备用户要求的交互式 KNN 与累计距离着色。正确 viewer 已有这两项能力，但 ARCTIC raw 数据因铰接物体没有 `obj_pose_world`，且当前 NPZ 未保存 MANO mesh，需做最小 viewer-only 兼容。

**验证**

- `pytest -q src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`：`4 passed`。
- `visualize_grab --check-only`：889 帧、object `[889,4096,3]`、merged hands `[889,3076,3]`；frame 0 object-to-hand KNN 命中数 n=1/4/8/16/32/64 分别为 `17/38/60/89/147/234`。
- 8113 HTTP `200`；manifest 明确累计距离为红色、object-to-hand KNN 为黄色。object/hand mesh 在缺失可靠字段时保持关闭。
- 完整导出进程未中断，复核时推进至 `330/1335`。以上为工程证据，不构成科研效果结论。

**回滚**

停止 viewer PID `4128922`，删除 viewer-only index/运行目录，并恢复本次代码、测试、plan、memory、README、版本和 activity 增量；raw NPZ、完整导出及旧运行无需恢复。
## 2026-09-16 09:19:34 +0000 — NAS 全量 GRAB Inspire 导出进入 DExplore converter

- activity_id: `ACT-20260916-091934-OICM-FULL-NAS-CONVERTER-START`
- timestamp: `2026-09-16 09:19:34 +0000`
- modification_version: `V1.4.18`
- type: `data / operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户要求先启动最终转换；延续 V1.4 plan §18–18.1 已批准的完整 1335 条 GRAB exact InterAct→DExplore geometric NAS 导出。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `grab-interact-dexplore-exact-full-20260916T074233Z`
- run_status: `RUNNING`
- current_stage: `dexplore_convert_grab`
- process: wrapper PID `174637`，worker PID `174638`，物理 GPU `3`；启动复核时显存约 `1920 MiB`。
- scope: 复用已完成的 1335/1335 InterAct processed/canonical 数据，在 NAS 完成 1335/1335 adapter 后启动一次性 DExplore Inspire converter；保持排序、跨序列 optimizer warm-start、`iterations=1`、`stride=1`。不重跑前两阶段，不修改原始 GRAB、旧 geometric、V1.3、split、cache、checkpoint 或 GPU 1/2 运行。
- conclusion: `SUPPORTED`（adapter 完整性、NAS 路由、GPU 3 converter 启动）；完整 geometric 计数、finite、失败清单和几何质量仍为 `INCONCLUSIVE`。

**原因**

canonical 阶段此前已正常完成，但四阶段链路由独立命令手工衔接，未自动提交 adapter/converter。用户本轮明确要求先启动最终转换，因此从现有完整 canonical 结果续跑，避免无意义地重做前两阶段。

**文件与证据**

- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/) — NAS NFS 运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/run_manifest.json) — 已更新为 converter `RUNNING` 和当前 PID/GPU。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/commands.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/commands.log) — process、canonical、adapter 和 converter 命令/退出状态。
- `data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/validation.json` — PENDING，converter 终态后生成。

**验证**

- adapter 输出 `converter_input/sequences` 为 `1335/1335`；converter 已在 `geometric_full` 产生首批序列目录。
- `nvidia-smi` 确认 worker PID `174638` 位于物理 GPU 3；GPU 1/2 既有进程未改变。
- 当前仅为工程启动证据。终态需核对 1335 条序列、tensor shape/finite、失败清单并生成 `validation.json`。

**回滚**

安全停止 wrapper PID `174637` / worker PID `174638` 并保留 NAS 目录，不删除既有中间产物。
## 2026-09-16 09:29:28 +0000 — OakInk2 运动优先与 2 cm 双侧搜索 pilot 完成

- activity_id: `ACT-20260916-092928-OICM-OAKINK2-TEMPORAL-PILOT`
- timestamp: `2026-09-16 09:29:28 +0000`
- modification_version: `V1.4.20`
- type: `code / data / diagnostic / documentation`
- task_mode: `change` 后切回 `read-only/diagnostic`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户指出旧 motion-first 由单零件抖动错误裁剪，并确认采用“先检测可靠运动，再从运动区间向前后搜索 2 cm”的时序方案。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `pilot_20260916T092715Z`
- run_status: `COMPLETED`
- scope: 新增隔离的 OakInk2 时序切分 helper、合成测试和 10 条真实 pilot；保持 v1/v1.1 index、annotation、Stage3、split、cache、训练与 viewer staging 只读，不自动重建正式 index。
- conclusion: `SUPPORTED`（运动优先时无可靠整体运动不会读取距离；削笔器降级为 part-only；2 cm 双侧扩展按合同执行）；`INCONCLUSIVE`（阈值与扩展后的正式样本质量尚待人工复核）。

**原因**

旧 `_motion_span` 以任一零件单帧 `>1 mm / >1°` 为运动证据，并用全局最早/最晚事件合成一个 span。削笔器的一个零件小幅转动触发该规则，而更早的连续 `<2 cm` 帧被 motion span 先行裁掉。新 pilot 改用 7 帧净运动、多零件共识和独立时序分量，并把 Stage3 缓存距离延迟到可靠运动候选之后读取。

**文件与产物**

- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — §19 final 科研语义、参数、保护边界和回滚。
- [`src/task/ObjectInteractionCm/tools/data/oakink2_temporal_segments.py`](../../tools/data/oakink2_temporal_segments.py) — motion-first、多零件共识、时序分量与接触扩展实现。
- [`src/task/ObjectInteractionCm/tests/test_oakink2_temporal_segments.py`](../../tests/test_oakink2_temporal_segments.py) — 无运动不读距离、part-only、多片段与接触边界测试。
- [`src/task/ObjectInteractionCm/research/oakink2_temporal_segmentation/README.md`](../../research/oakink2_temporal_segmentation/README.md) — pilot 入口与只读边界。
- [`src/task/ObjectInteractionCm/research/oakink2_temporal_segmentation/run.py`](../../research/oakink2_temporal_segmentation/run.py) — 3 条代表样本和 7 条旧静止对照运行入口。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/) — NAS pilot 运行目录。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/run_manifest.json) — `COMPLETED` 运行合同。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json) — 逐段旧/新 mask、motion、part-only 和距离读取证据。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进至 `V1.4.20`。

**验证**

- `pytest -q src/task/ObjectInteractionCm/tests/test_oakink2_temporal_segments.py src/task/ObjectInteractionCm/tests/test_split_oakink2_active_tool.py`：`10 passed`。
- 两个新 Python 入口 `py_compile` 通过；pilot manifest/report 通过 `json.tool`。
- 真实 pilot：10 条中 `part_only_motion=1`、`selected=7`、`no_reliable_motion=2`。削笔器由旧 12 帧变为 `part_only_motion`、新选择 0 帧且距离 provider 调用 0 次；烧杯由 34 帧扩展为 149 帧；三脚架由 43 帧扩展为 639 帧。后者表明长期持握时 `<2 cm` 可覆盖几乎整个 primitive，是否增加最长上下文或保持完整持握需用户复核，不能自动定稿。
- 同期 GRAB full converter 保持 `RUNNING`，本次复核时 `459/1335`；本 pilot 未修改其进程、顺序或输出。

**回滚**

删除本次 temporal helper、测试、研究入口和独立 NAS pilot 目录，并恢复 plan §19、版本指针、README 与本 activity 增量；v1/v1.1、annotation、Stage3 和训练数据无需恢复。
## 2026-09-16 10:14:58 +0000 — NAS 全量 GRAB Inspire geometric 导出完成

- activity_id: `ACT-20260916-101458-OICM-FULL-NAS-GEOMETRIC-COMPLETE`
- timestamp: `2026-09-16 10:14:58 +0000`
- modification_version: `V1.4.18`
- type: `data / operation / diagnostic`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续 V1.4 plan §18–18.1 已批准的 1335 条完整 GRAB exact InterAct→DExplore geometric NAS 导出；用户本轮查询终态，补做全量 tensor 验证并闭合运行。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `grab-interact-dexplore-exact-full-20260916T074233Z`
- run_status: `COMPLETED`
- last_step: `N/A`（数据处理）
- last_epoch: `N/A`（数据处理）
- best_metric: `N/A`（数据处理）
- checkpoint: `N/A`（geometric retarget 数据导出）
- scope: 闭合 1335 条 GRAB process/canonical/adapter/converter 链路，逐个加载最终 tensor 检查序列集合、帧数、shape、dtype 和 finite；不执行筛选、RL、V1.4 cache 接入或训练。
- conclusion: `SUPPORTED`（1335/1335、406264 帧、0 失败、全量 `[T,598] float32` finite）；几何接触/穿透质量仍为 `INCONCLUSIVE`。

**原因**

converter 已于 `2026-09-16 09:50:08 +0000` 正常退出，但运行 manifest 仍为 `RUNNING` 且缺少终态 `validation.json`。本次根据用户状态查询完成全量只读验证并修正终态，避免仅以目录计数宣称成功。

**文件与产物**

- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/) — NAS NFS 完整运行目录。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/run_manifest.json) — `COMPLETED` 终态、序列/帧计数和验证摘要。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/validation.json`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/validation.json) — 全量 tensor shape/frame/finite/失败清单。
- [`data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/commands.log`](../../../../../data/processed_data/object_interaction_cm/grab_interact_dexplore_exact_v1_4_17/full_20260916T074233Z/commands.log) — 四阶段命令、历史恢复过程与 converter `1335 succeeded, 0 failed`。

**验证**

- converter exit code `0`；输出序列集合与 1335 条 adapter 输入完全一致，无 missing/extra。
- 逐个读取 1335 个 `interaction_hand_inspire.pt`：总计 `406264` 帧，全部 `[T,598]`、`torch.float32`、finite；逐序列 `T` 与 converter 输入 `motion.npz:n_frames` 一致，failure_count=`0`。
- 工程导出结论为 `SUPPORTED`；该验证不评价视觉接触、穿透或训练收益。

**回滚**

保留 NAS 运行目录作为独立产物；如需回滚，只移除该 run 及本终态 activity，不修改原始 GRAB、旧 geometric、cache、split 或 checkpoint。

## 2026-09-16 10:47:42 +0000 — 双手 MANO2048 cache producer 完成并启动全量 NAS 导出

- activity_id: `ACT-20260916-104742-OICM-BILATERAL-MANO-CACHE-START`
- timestamp: `2026-09-16 10:47:42 +0000`
- modification_version: `V1.4.21`
- type: `code / data / operation / diagnostic`
- task_mode: `change` 后切换为 `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认复用仓库现有 V1.3 MANO2048/KNN 实现，继续适配当前双手 GRAB/ARCTIC NPZ 并启动全量 cache 导出。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`
- run_status: `RUNNING`
- scope: 复用 V1.3 固定表面积采样和离线 KNN 合同，新增当前 V1.4 bilateral Stage-4 NPZ producer；decoder 流保持每侧 1538 点，高分辨率 KNN 流使用每侧 MANO 2048 点，GPU 0/3 分两个确定性 shard 导出全部 1335 条 GRAB 与 301 条 ARCTIC。输出只写 NAS 独立目录，不覆盖源 NPZ、Inspire cache、split、checkpoint 或旧运行。
- conclusion: `SUPPORTED`（两数据集 pilot 与代码验证）；全量终态与训练效果当前为 `INCONCLUSIVE`。

**原因**

仓库已有 V1.3 单右手 MANO2048 producer，但当前 bilateral 快速脚本把 3076 点 decoder 流直接链接为
KNN 流，且不能读取 Stage-4 双侧 NPZ 或恢复 ARCTIC MANO mesh。新 producer 只补齐这层 schema/输入适配，
不重新定义采样、KNN、半径、split 或训练语义。

**实现与 pilot**

- [`src/task/ObjectInteractionCm/tools/data/build_bilateral_mano_v1_4_cache.py`](../../tools/data/build_bilateral_mano_v1_4_cache.py) — 双流生成、ARCTIC MANO 参数重建、GPU KNN、worker/finalize/resume 和全量验证。
- [`src/task/ObjectInteractionCm/tests/test_bilateral_mano_v1_4_cache.py`](../../tests/test_bilateral_mano_v1_4_cache.py) — 面积采样、固定 correspondence、跨帧一致性和法向测试。
- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针推进至 `V1.4.21`。
- [`data/processed_data/object_interaction_cm_bilateral_mano_v1_4_21_pilot_20260916T104526Z/`](../../../../../data/processed_data/object_interaction_cm_bilateral_mano_v1_4_21_pilot_20260916T104526Z/) — NAS pilot，GRAB/ARCTIC 各一条，共 1168 帧。
- [`data/processed_data/object_interaction_cm_bilateral_mano_v1_4_21_pilot_20260916T104526Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_bilateral_mano_v1_4_21_pilot_20260916T104526Z/run_manifest.json) — pilot `COMPLETED` manifest。
- [`data/processed_data/object_interaction_cm_bilateral_mano_v1_4_21_pilot_20260916T104526Z/validation_summary.json`](../../../../../data/processed_data/object_interaction_cm_bilateral_mano_v1_4_21_pilot_20260916T104526Z/validation_summary.json) — pilot shape、finite、index 范围和帧统计。

**全量运行**

- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/) — NAS 正式输出，`RUNNING`。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — `STARTED`；预期 1636 条、两个 shard。
- `data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json` — PENDING（`run_status=RUNNING`，worker 全部完成并 finalize 后生成）。
- `data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/validation_summary.json` — PENDING（`run_status=RUNNING`，全量逐数组验证后生成）。

**验证**

- `py_compile` 通过；`pytest -q src/task/ObjectInteractionCm/tests/test_bilateral_mano_v1_4_cache.py src/task/ObjectInteractionCm/tests/test_v1_3_offline_knn.py`：`6 passed`。
- pilot decoder `[T,3076,3]` 左右半区与源 NPZ 逐元素完全一致；高分辨率流 `[T,4096,3]`、KNN index `[T,4096,32] uint16`，索引最大值 4095，法向范数最大误差 `1.79e-7`。
- ARCTIC pilot 保存两个 object part、逐帧 articulation 和真实 root pose；按相同 seed 重建的 object correspondence 与现有 4096 点 cache 最大误差 `0 m`。
- 正式运行已确认两个 worker 分别在 GPU 0/3 连续产出，首批序列均 `COMPLETED`；GPU 1/2 的既有满载任务未改变。

**回滚**

安全停止两个 worker 并保留或移除独立 NAS 输出；恢复新增 producer、测试、`V1.4.21` 指针和本 activity。源 GRAB/ARCTIC NPZ、Inspire cache、split、checkpoint 与其他运行无需恢复。

## 2026-09-16 11:43:42 +0000 — OakInk2 V1.4.20 时序切分 KNN/距离查看器启动于 8112

- activity_id: `ACT-20260916-114342-OICM-OAKINK2-TEMPORAL-KNN-VIEWER`
- timestamp: `2026-09-16 11:43:42 +0000`
- modification_version: `V1.4.21`
- type: `data / diagnostic / operation`
- task_mode: `run-only/operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户要求查看 OakInk2 数据切分，并已固定以后统一使用带 object-to-hand KNN 与累计距离着色的 `visualize_grab.py`；端口沿用用户指定的 `8112`。
- skills_used: `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `oakink2_temporal_mano_knn_viewer_20260916T114011Z/viewer_8112`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: PID `310853`，HTTP `200`。
- scope: 将 V1.4.20 motion-first + 2 cm 双侧搜索 pilot 中实际选中的 7 条轨迹只读投影为 viewer-only bilateral MANO NPZ/index，并用统一 KNN/累计距离 viewer 展示；停止旧 8112 GRAB viewer 和旧 8113 三轨迹 OakInk2 viewer。不修改 OakInk2 v1/v1.1 index、时序 pilot 报告、annotation、Stage3、正式 cache、split 或训练。
- conclusion: `SUPPORTED`（7 条 adapter、KNN/距离 viewer 和 HTTP 工程链路）；切分科研质量仍为 `INCONCLUSIVE`。

**原因**

旧 8113 OakInk2 index 只有修复前的削笔器、烧杯、三脚架三条，仍显示削笔器错误的 12 帧；旧切分专用
viewer 又没有用户要求的 KNN/距离着色。因此本次仅对 V1.4.20 pilot 的实际新选中帧生成隔离的 viewer
adapter，不把 0 帧结果伪造成轨迹。

**产物与验证**

- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/) — NAS viewer-only adapter，共 7 条、4145 帧。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/adapter_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/adapter_manifest.json) — 7 条选中轨迹和 3 条 0 帧排除原因。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/index.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/index.json) — viewer-only index。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/viewer_8112/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/viewer_8112/run_manifest.json) — `RUNNING` viewer manifest。

**验证**

- 7/7 条 `visualize_grab --check-only` 均退出 0；每帧 object 4096 点、bilateral MANO 1556 顶点，hand mesh 两部分可用。初始烧杯轨迹 149 帧，首帧最近距离 `17.891 mm`，object-to-hand KNN n=1/4/8/16/32/64 的手点并集为 `28/51/72/109/158/221`。
- 未纳入 viewer：削笔器 `part_only_motion`、bowl `no_reliable_motion`、一条 laptop `no_reliable_motion`，均为 0 帧。MANO cache 全量 worker 未停止，复核时已完成 1080/1636 条。

**回滚**

停止 PID `310853`，移除独立 viewer adapter/run 和本 activity；所有输入 index、报告和正在运行的正式 cache 不受影响。

## 2026-09-16 11:52:17 +0000 — OakInk2 烧杯首条切分静止段诊断与 viewer 定位

- activity_id: `ACT-20260916-115217-OICM-OAKINK2-BEAKER-TAIL-DIAG`
- timestamp: `2026-09-16 11:52:17 +0000`
- modification_version: `V1.4.21`
- type: `diagnostic / operation`
- task_mode: `read-only/diagnostic` 后切换为 `run-only/operation`
- change_level: `L0`
- approval: `user-approved`
- approval_basis: 用户指出 8112 第一条烧杯轨迹看起来没有明显运动，要求复核当前可视化结果。
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `oakink2_temporal_mano_knn_viewer_20260916T114011Z/viewer_8112_motion_core`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: PID `324546`，HTTP `200`。
- scope: 只读核对第一条烧杯的 motion/contact component 与 viewer object trajectory，并把 viewer 初始帧定位到 motion core；不修改切分算法、index、报告、adapter NPZ、正式 cache 或训练。
- conclusion: `SUPPORTED`（运动核心存在但无上限 2 cm 扩展带入长静止尾部）；正式切分策略是否增加扩展上限仍为 `INCONCLUSIVE`。

**原因**

烧杯 motion core 为原始帧 `1322–1390`，但连续 `<2 cm` 接触分量覆盖 `1265–1413`；当前算法按已确认的
“从运动段向两侧搜索 2 cm”语义吞入整个接触分量，因此 viewer 前 57 帧和后 23 帧几乎静止。

**验证**

- 运动核心 69 帧内，object centroid 净移动 `44.673 mm`，7 帧窗口 translation 的 median/P95/max 为
  `3.492/7.235/7.296 mm`，rotation 为 `5.367/10.384/10.511°`。
- 前 57 帧 centroid 净移动 `0.317 mm`，后 23 帧 `0.157 mm`；静止观感真实，不是 viewer 选错对象。
- 将 8112 初始位置改为轨迹内 frame index `57`（原始 frame `1322`），不修改 index、报告或 NPZ。

**产物**

- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json) — motion/contact component 证据。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/viewer_8112_motion_core/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/viewer_8112_motion_core/run_manifest.json) — 新 viewer 运行状态。

**回滚**

停止 PID `324546` 并恢复从 frame index 0 启动；所有切分输入和 cache 不变。

## 2026-09-16 12:06:49 +0000 — OakInk2 烧杯完整 primitive 与场景上下文 viewer

- activity_id: `ACT-20260916-120649-OICM-OAKINK2-BEAKER-FULL-SCENE`
- timestamp: `2026-09-16 12:06:49 +0000`
- modification_version: `V1.4.21`
- type: `code / data / diagnostic / operation`
- task_mode: `change` 后切换为 `run-only/operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户澄清需要先查看第一条烧杯轨迹的完整 primitive 和场景，以判断目标对象是否真的被移动；继续使用固定的 KNN/累计距离 viewer。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `oakink2_temporal_mano_knn_viewer_20260916T114011Z/full_scene_selected_1772/viewer_8112`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: PID `336329`，HTTP `200`。
- scope: 将烧杯 `selected:1772` 的完整 primitive 原始帧 `941–1413` 投影为 viewer-only bilateral MANO 轨迹；橙色 4096 点仅为被选中烧杯，灰色 1536 点为同场景另外 3 个对象，KNN/红色距离仍只对烧杯计算。不修改时序切分、index、annotation、Stage3、正式 cache 或训练。
- conclusion: `SUPPORTED`（完整场景读取、上下文显示和 viewer 工程链路）；目标对象语义及切分科研质量等待用户人工判断，保持 `INCONCLUSIVE`。

**原因**

此前 8112 只显示时序筛选后的 149 帧和选中对象，无法判断该 primitive 的前因后果或场景中是否存在
更合理的主动物体。本次增加 viewer-only 可选 `context_points_world`，上下文不参与 KNN/距离，避免改变目标对象定义。

**文件与产物**

- [`src/task/ObjectInteractionCm/visualize_grab.py`](../../visualize_grab.py) — raw bilateral viewer 可选读取并灰色显示场景上下文点。
- [`src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`](../../tests/test_visualize_grab_v1_4.py) — context 点读取回归。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/full_scene_selected_1772/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/full_scene_selected_1772/) — 完整 primitive viewer-only adapter。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/full_scene_selected_1772/viewer_8112/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/full_scene_selected_1772/viewer_8112/run_manifest.json) — `RUNNING` viewer manifest。

**验证**

- `py_compile` 通过；`pytest -q src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`：`4 passed`。
- `visualize_grab --check-only`：473 帧、目标物体 4096 点、上下文 1536 点、双手 1556 顶点、finite；首帧原始 frame `941`。
- adapter 确认上下文对象 3 个、缺失上下文对象 0；HTTP 8112 返回 `200`。

**回滚**

停止 PID `336329`，删除独立 full-scene adapter/run，并恢复 viewer context 可选字段及测试；源数据和切分结果无需恢复。

## 2026-09-16 12:27:39 +0000 — OakInk2 任务优先语义复核与其余六条完整轨迹 viewer

- activity_id: `ACT-20260916-122739-OICM-OAKINK2-TASK-FIRST-REVIEW`
- timestamp: `2026-09-16 12:27:39 +0000`
- modification_version: `V1.4.21`
- type: `data / diagnostic / operation`
- task_mode: `run-only/operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户确认“拿试管放进烧杯”任务应以试管为主导对象，烧杯几乎不动，要求排除烧杯候选，并继续展示其他轨迹。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/viewer_8112`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: PID `178759`，HTTP `200`。
- scope: 按“任务主导对象语义先于运动/contact”的人工复核规则，将烧杯 `selected:1772` 标记为 viewer/formal 候选排除；其余 6 条非零时序候选以完整 primitive、双手 MANO、橙色当前候选和灰色场景上下文生成只读 viewer adapter。不修改 v1/v1.1 source index、V1.4.20 时序报告、annotation、Stage3、正式 cache、split 或训练。
- previous_run: `full_scene_selected_1772/viewer_8112` 已于 `2026-09-16T12:27:05+00:00` 停止，终态 `STOPPED`。
- conclusion: `SUPPORTED`（任务优先排除规则、六条完整轨迹 adapter、KNN/距离 viewer 和 HTTP 工程链路）；其余六条的任务语义正确性等待用户逐条人工判断，保持 `INCONCLUSIVE`。

**原因**

前一条烧杯结果证明仅凭物体运动与手物距离，无法决定任务中的主导对象；“试管放入烧杯”的语义要求先选试管，再对试管执行运动与接触检测。因此先排除已确认错误的烧杯候选，并把剩余非零候选连同完整场景交给人工复核。

**语义结论与展示范围**

- 烧杯 `selected:1772` 是“试管放入烧杯”任务中的被作用容器，不是主导移动对象；不得进入后续正式候选。该结论来自用户对完整 primitive 的明确确认，不由运动或 2 cm 距离阈值覆盖。
- viewer 展示 6 条待复核候选，共 5314 个完整 primitive 帧：tripod/rearrange 641 帧、laptop/open 1930 帧、laptop/close 882 帧、三条 alcohol burner/rearrange 分别 641/540/680 帧。
- tripod 与第一条 alcohol burner 是同一双物体 rearrange 的两个候选视角，故并列保留供任务语义判断；不因两者均有运动/contact 就自动同时接纳。
- 另有 3 条零帧候选不展示：pencil sharpener `part_only_motion`、bowl `no_reliable_motion`、laptop `no_reliable_motion`。

**产物与验证**

- [`src/task/ObjectInteractionCm/visualize_grab.py`](../../visualize_grab.py) — 复用本轮前一活动已增加的灰色上下文读取与显示能力，本次未继续改动代码。
- [`src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`](../../tests/test_visualize_grab_v1_4.py) — 覆盖上述上下文读取合同，本次未继续改动测试。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/) — NAS viewer-only adapter；烧杯排除原因与六条轨迹明细见 `adapter_manifest.json`。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/index.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/index.json) — 6 条完整 primitive viewer-only index。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/viewer_8112/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/viewer_8112/run_manifest.json) — `RUNNING` viewer manifest。

**验证**

- 6/6 条 `visualize_grab --check-only` 均退出 0；每条目标物体 4096 点、双手 MANO 1556 顶点，灰色上下文为 512–7680 点；hand mesh 两部分可用。
- viewer 仅将橙色当前候选用于累计距离红色着色和 object-to-hand KNN 黄色着色；灰色上下文只用于理解任务，不参与 KNN/距离。
- `curl http://127.0.0.1:8112` 返回 HTTP `200`，监听进程 PID 为 `178759`。

**回滚**

停止 PID `178759` 并删除独立 `task_first_review_remaining_6/` adapter/run；源 index、时序报告和正式数据不受影响。烧杯排除是用户确认的任务语义决定，若要恢复须重新获得用户确认。

## 2026-09-16 12:46:06 +0000 — OakInk2 alcohol burner 与 laptop 视觉重复诊断

- activity_id: `ACT-20260916-124606-OICM-OAKINK2-REPEAT-DIAG`
- timestamp: `2026-09-16 12:46:06 +0000`
- modification_version: `V1.4.21`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-requested`
- approval_basis: 用户指出 `2647/2648/2649` 看起来没有区别，两条 laptop 轨迹也近似相同，要求解释。
- skills_used: `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/viewer_8112`
- run_status: `RUNNING`
- scope: 只读比较五条 adapter NPZ 的 hash、源 sequence、primitive、对象 ID、帧区间、位移统计及 V1.4.20 motion component；不修改 viewer、adapter、source index、时序报告、正式 cache 或训练。
- conclusion: `SUPPORTED`（确认不存在误加载同一文件，同时确认候选语义重复和慢动作检测稀疏）；这些候选是否保留仍为 `INCONCLUSIVE`。

**原因**

需要区分两种可能：viewer 下拉选择失效导致同一轨迹重复显示，或源数据本身是相同任务的不同 take。另需检查 laptop 开/关动作相似是否来自长静止区间及运动阈值漏检。

**验证**

- `2647/2648/2649` 的 `shared/left/right.npz` SHA-256 前缀均不同，源 sequence 和 primitive frame range 分别为 `9240–9880`、`4510–5049`、`19940–20619`；不是同一文件或同一帧区间。
- 三条均为 `rearrange`，目标均为 alcohol burner `O02@0206@00002`，并具有相同双手任务分工；因此是不同录制的同语义重复 take。目标质心首尾净位移分别为 `104.681/110.582/88.805 mm`。
- laptop 两条来自不同 sequence，分别为 `open_laptop_lid` 的 `2784–4713` 和 `close_laptop_lid` 的 `360–1241`，目标均为 lid `O02@0053@00001`；目标质心首尾净位移分别为 `179.243/175.658 mm`，说明不是静态复制，而是同一铰链路径的相反动作。
- open laptop 的可靠运动证据仅 `25/1930` 帧（`1.3%`），却经连续 `<2 cm` 接触扩展为 `1345` 帧；close laptop 为 `157/882` 帧（`17.8%`），扩展为 `532` 帧。完整 primitive 播放中的大量慢速/静止帧会掩盖开关方向。
- `2647/2648/2649` 的运动证据比例分别为 `22.0%/40.2%/19.4%`；三条动作本就相同，并均包含明显非运动区间。
- viewer 轨迹 label 使用唯一 `sequence_id`，下拉框不存在同名匹配到第一条的问题；HTTP 8112 服务保持运行。

**产物**

- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json) — motion component 与 2 cm 扩展证据。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/index.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/index.json) — 五条轨迹的唯一 sequence/path 映射。

**回滚**

本次仅新增诊断记录，无数据或代码变更；删除本 activity 条目即可回滚记录。viewer 与所有输入保持不变。

## 2026-09-16 12:55:44 +0000 — OakInk2 当前 viewer 与时序 pilot 帧率语义诊断

- activity_id: `ACT-20260916-125544-OICM-OAKINK2-FPS-DIAG`
- timestamp: `2026-09-16 12:55:44 +0000`
- modification_version: `V1.4.21`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-requested`
- approval_basis: 用户询问 OakInk2 帧率是否为 30 Hz。
- skills_used: `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `oakink2_temporal_mano_knn_viewer_20260916T114011Z/task_first_review_remaining_6/viewer_8112`
- run_status: `RUNNING`
- scope: 只读核对 OakInk2 toolkit 帧率常量、adapter `source_fps/ds_rate/raw_frame_id` 与 viewer 播放参数；不修改 viewer、adapter、时序报告、source index、正式 cache 或训练。
- conclusion: `INVALID_IMPLEMENTATION`（当前 OakInk2 viewer adapter 和 V1.4.20 时序 pilot 错把连续 120 Hz mocap 帧按 30 Hz 语义处理）；OakInk2 源数据本身未损坏。

**原因**

需要区分 OakInk2 的 RGB 视频帧率、MANO/物体 mocap 帧率以及 viewer GUI 播放 FPS。三者不能仅凭 adapter 元数据视为同一时间轴。

**验证**

- `dataset/OakInk2/src/oakink2_toolkit/meta.py` 明确定义 `FPS_MOCAP=120`、`FPS_VIDEO=30`；preview stream 也定义 `FPS_MOCAP=120`。
- 当前 6 条 adapter 的 `raw_frame_id` 均逐帧连续，例如 `9240,9241,9242`，说明保留的是完整 120 Hz mocap，并未执行每 4 帧下采样。
- adapter 却写入 `source_fps=30.0`、`ds_rate=1`；viewer 以 `--fps 30` 顺序播放所有源帧，因此视觉时间被放慢 4 倍。
- V1.4.20 temporal pilot 直接在相邻 mocap frame 上使用 7 帧窗口和按 30 Hz 讨论的位移/旋转阈值；实际窗口物理时长仅为预期的四分之一，上一活动发现的 laptop 慢动作漏检与此一致。
- 当前 8112 仅作为错误定位证据继续运行；其视觉速度、运动分段和基于该分段的 2 cm 扩展不得作为有效数据结论。

**产物**

- [`dataset/OakInk2/src/oakink2_toolkit/meta.py`](../../../../../dataset/OakInk2/src/oakink2_toolkit/meta.py) — OakInk2 mocap/video 帧率事实源。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_20/pilot_20260916T092715Z/temporal_segmentation_report.json) — 当前被判无效的时序 pilot 证据入口；产物保留用于审计，不覆盖。

**回滚**

本次仅新增诊断记录，无代码或数据改动；删除本 activity 条目即可回滚记录。修复需另建独立产物，旧 pilot 保留只读。

## 2026-09-16 13:14:05 +0000 — OakInk2 官方 30 Hz 时序切分重跑与 8112 viewer

- activity_id: `ACT-20260916-131405-OICM-OAKINK2-OFFICIAL30HZ`
- timestamp: `2026-09-16 13:14:05 +0000`
- modification_version: `V1.4.22`
- type: `code / data / diagnostic / operation`
- task_mode: `change` 后切换为 `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户在确认 OakInk2 mocap 为 120 Hz、当前 pilot 错按 30 Hz 处理后，明确同意按真实 30 Hz 时间轴修正并重跑；最终 plan 已要求 OakInk2 使用 30 Hz 时间轴。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `pilot_official30hz_20260916T130328Z`
- run_status: `COMPLETED`
- adapter_run_id: `viewer_adapter_official30hz_20260916T130706Z`
- adapter_run_status: `COMPLETED`
- viewer_run_id: `viewer_adapter_official30hz_20260916T130706Z/viewer_8112`
- viewer_run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: PID `2430234`，HTTP `200`。
- previous_run: V1.4.20 `task_first_review_remaining_6/viewer_8112` 已于 `2026-09-16T13:13:33+00:00` 停止，终态 `STOPPED`。
- scope: 使用 annotation 官方 `frame_id_list` 将 OakInk2 120 Hz mocap 对齐到连续 30 Hz 视频时间轴；在任务语义计算前显式排除烧杯 `selected:1772`，重新运行运动优先与 2 cm 双侧搜索，并生成 7 条候选的 cut/full 成对 KNN/距离 viewer adapter。旧 V1.4.20 产物、v1/v1.1 index、annotation、Stage3、正式 cache、split 和训练保持只读。
- conclusion: `SUPPORTED`（官方 30 Hz 采样、时序连通、任务排除、adapter 与 viewer 工程链路）；7 条候选最终数据质量和训练收益仍为 `INCONCLUSIVE`。旧 V1.4.20 时序结论保持 `INVALID_IMPLEMENTATION`。

**原因**

OakInk2 官方定义 mocap 120 Hz、视频 30 Hz；旧 pilot 直接在连续 mocap frame 上应用按 30 Hz 设计的窗口，并把连续帧 adapter 标记为 30 Hz，导致运动窗口物理时长缩短四倍、播放慢四倍。官方 annotation 的 `frame_id_list` 是视频帧到 mocap frame 的对齐事实源，相邻 raw ID 实测可为 3、4 或 5，不能用固定 `%4` 或固定 raw stride 替代。

**文件**

- [`docs/current_versions.yaml`](../../../../../docs/current_versions.yaml) — ObjectInteractionCm 指针更新为 `V1.4.22`。
- [`src/task/ObjectInteractionCm/tools/data/oakink2_temporal_segments.py`](../../tools/data/oakink2_temporal_segments.py) — 时序组件支持独立 timeline position，输出仍保留原始 mocap frame ID。
- [`src/task/ObjectInteractionCm/research/oakink2_temporal_segmentation/run.py`](../../research/oakink2_temporal_segmentation/run.py) — 使用官方 `frame_id_list`、记录 120→30 Hz 合同，并支持在运动/contact 前显式排除任务语义错误候选。
- [`src/task/ObjectInteractionCm/research/oakink2_temporal_segmentation/build_viewer_adapter.py`](../../research/oakink2_temporal_segmentation/build_viewer_adapter.py) — 生成 cut/full 成对的双手 MANO、灰色上下文及 KNN/距离 viewer adapter。
- [`src/task/ObjectInteractionCm/research/oakink2_segment_visualizer/run.py`](../../research/oakink2_segment_visualizer/run.py) — trajectory loader 支持经过验证的显式 frame ID 子集。
- [`src/task/ObjectInteractionCm/tests/test_oakink2_temporal_segments.py`](../../tests/test_oakink2_temporal_segments.py) — 覆盖官方非固定 raw ID 对齐和独立时序连通位置。
- [`src/task/ObjectInteractionCm/research/oakink2_temporal_segmentation/README.md`](../../research/oakink2_temporal_segmentation/README.md) 与 [`experiment.yaml`](../../research/oakink2_temporal_segmentation/experiment.yaml) — 更新 30 Hz 数据合同、运行入口和产物根。
- [`src/task/ObjectInteractionCm/docs/plan/V1.4.md`](../plan/V1.4.md) — 已定稿且经用户确认的 30 Hz、任务语义优先、运动后 2 cm 搜索边界。

**产物**

- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/) — 新 30 Hz pilot 与终态 manifest。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/temporal_segmentation_report.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/temporal_segmentation_report.json) — 9 条计算结果、1 条任务语义排除与采样 provenance。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/) — 14 条 cut/full viewer adapter 和终态 manifest。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/viewer_8112/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/viewer_8112/run_manifest.json) — `RUNNING` viewer manifest。

**验证**

- `py_compile`：temporal helper、pilot runner、viewer adapter builder 和 segment visualizer 均通过。
- `pytest -q test_oakink2_temporal_segments.py test_oakink2_segment_visualizer.py test_visualize_grab_v1_4.py`：`15 passed`。
- pilot：烧杯 `selected:1772` 在计算前排除；其余 9 条中 7 条 `selected`、削笔器和 bowl 为 `no_reliable_motion`。修正后此前漏掉的 laptop `2646` 检出 22 个运动证据帧并得到 103 帧 cut。
- adapter：7 条候选生成 14 条 cut/full 记录；完整片段共 1435 个 30 Hz 帧，cut 共 1102 帧；每条保存 `source_fps=120`、`ds_rate=4` 和官方映射的原始 frame ID。
- 14/14 条 `visualize_grab --check-only` 退出 0；目标对象 4096 点、双手 1556 顶点、灰色上下文 512–7680 点，hand mesh 可用。
- 8112 返回 HTTP `200`；初始项为 `official30hz_selected_1726_cut`，同一候选的 `full` 项紧随其后。

**回滚**

停止 PID `2430234`，删除独立 `oakink2_temporal_segmentation_v1_4_22/` 产物，恢复上述 Task-local 代码、测试、README/experiment、版本指针和本 activity。旧 V1.4.20、源 index、annotation、Stage3 和正式训练数据无需恢复。

## 2026-09-16 13:19:44 +0000 — KNN/距离 viewer 未来帧上限扩展至 30

- activity_id: `ACT-20260916-131944-OICM-VIEWER-FUTURE30`
- timestamp: `2026-09-16 13:19:44 +0000`
- modification_version: `V1.4.22`
- type: `code / diagnostic / operation`
- task_mode: `change` 后切换为 `run-only/operation`
- change_level: `L1`
- approval: `user-approved`
- approval_basis: 用户明确要求“把未来可视化里面未来帧的上限设置为30”。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- run_id: `viewer_adapter_official30hz_20260916T130706Z/viewer_8112_future30`
- run_status: `RUNNING`
- viewer_url: `http://127.0.0.1:8112`
- process: PID `2435936`，HTTP `200`。
- previous_run: `viewer_adapter_official30hz_20260916T130706Z/viewer_8112` 已于 `2026-09-16T13:19:23+00:00` 停止，终态 `STOPPED`。
- scope: 仅将统一 KNN/累计距离 viewer 的未来帧 CLI 与 GUI 范围从 `0..10` 扩展为 `0..30`，同步帮助文本和测试，并用同一 OakInk2 V1.4.22 index 重启 8112。不修改当前/未来帧索引语义、末帧夹取、数据、KNN、距离、训练 stride、cache、split 或模型。
- conclusion: `SUPPORTED`（未来帧 30 的参数、索引、时间跨度和 viewer 服务工程验证）；不形成科研效果结论。

**文件**

- [`src/task/ObjectInteractionCm/visualize_grab.py`](../../visualize_grab.py) — 新增统一 `FUTURE_DELTA_MAX=30`，供 GUI slider、CLI choices、帮助文本共同使用。
- [`src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`](../../tests/test_visualize_grab_v1_4.py) — 覆盖 30 可接受、31 被拒绝及末帧夹取。

**原因**

原 viewer 将未来叠加范围硬编码为 `0..10`；仅修改 GUI 会导致 CLI 和帮助文本仍拒绝 30，因此用单一常量同步三个入口，保持既有未来帧计算不变。

**产物**

- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/viewer_8112_future30/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/viewer_8112_future30/run_manifest.json) — `RUNNING` viewer manifest。

**验证**

- `python -m py_compile src/task/ObjectInteractionCm/visualize_grab.py` 通过。
- `pytest -q src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`：`5 passed`。
- CLI `--help` 显示 `--future-delta {0,...,30}`，帮助文本声明 `1..30`。
- 对当前 OakInk2 首条 cut 运行 `--future-delta 30 --check-only`：frame `0→30`、raw mocap frame `9240→9360`、物理跨度 `1.0 s`、`future_clamped=false`。
- 8112 返回 HTTP `200`，监听 PID `2435936`。

**回滚**

停止 PID `2435936`，恢复 viewer 常量/GUI/CLI/帮助文本及对应测试，并可用原 index 重新启动；所有数据与 cache 无需恢复。

## 2026-09-16 13:26:22 +0000 — OakInk2 全量导出状态核对

- activity_id: `ACT-20260916-132622-OICM-OAKINK2-FULL-EXPORT-STATUS`
- timestamp: `2026-09-16 13:26:22 +0000`
- modification_version: `V1.4.22`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-requested`
- approval_basis: 用户询问是否所有 OakInk2 数据均已导出。
- skills_used: `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- scope: 只读核对 NAS 上 OakInk2 Stage3、active-tool index、V1.4.22 pilot/viewer adapter 与正式 V1.4 cache/index manifest；不启动任务或修改数据、代码、cache、split、配置和训练。
- conclusion: `SUPPORTED`（当前仅完成 9 条官方 30 Hz pilot 与 7 条 viewer adapter，尚未完成 OakInk2 全量正式导出）。

**原因**

需要区分全量上游 Stage3/旧选择索引、少量时序 pilot/viewer adapter，以及可直接供正式训练使用的双手几何/KNN cache；这些产物不能互相替代。

**验证**

- V1.4.22 pilot manifest 为 `COMPLETED`，范围仅 9 条：7 条 selected、2 条 no-reliable-motion；viewer adapter 为 7 条候选、14 个 cut/full 展示项。
- OakInk2 v1.1 active-tool 是全量旧索引，但尚未按 V1.4.22 官方 30 Hz 时间轴全量重算，且其 manifest 明确没有自动接入正式 cache/训练。
- 未找到 OakInk2 V1.4.22 全量时序 report、全量双手 MANO/Inspire geometric cache、全量 KNN cache 或正式训练 index/manifest。
- 当前 `object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json` 仍为 `STARTED`，目录名与输入范围仅为 GRAB/ARCTIC，且顶层 `manifest.json`、`index.json` 尚不存在；它不能证明 OakInk2 已导出。

**证据入口**

- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/run_manifest.json) — 已完成的 9 条 pilot。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/viewer_adapter_official30hz_20260916T130706Z/run_manifest.json) — 已完成的 7 条 viewer adapter。
- [`data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/run_manifest.json`](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/run_manifest.json) — 修正前的全量 active-tool 索引运行记录。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — 尚未 finalize 且不含 OakInk2 的 GRAB/ARCTIC cache 运行。

**回滚**

本次仅新增状态核对记录，无代码或数据改动；删除本 activity 条目即可回滚记录。

## 2026-09-16 13:32:37 +0000 — OakInk2 后续人工切分审查范围诊断

- activity_id: `ACT-20260916-133237-OICM-OAKINK2-REVIEW-SCOPE`
- timestamp: `2026-09-16 13:32:37 +0000`
- modification_version: `V1.4.22`
- type: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-requested`
- approval_basis: 用户询问还有哪些 OakInk2 切分需要人工审查。
- skills_used: `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户与当前任务修改）
- scope: 只读归组 v1→v1.1 的新增候选、selection reason、primitive、目标对象和当前 V1.4.22 pilot 覆盖；不生成新 viewer、不修改筛选、数据、cache、split 或训练。
- conclusion: `SUPPORTED`（确定后续应以语义风险和全量 30 Hz 异常值抽样审查，不需要逐条审查全部数据）。

**原因**

旧 v1.1 新增项包含同一任务的双对象候选和多个重复 take；在官方 30 Hz 全量重算前逐条观看旧 cut 会混入已知无效的时间尺度。人工成本应集中在任务主导对象可能选反、显式 semantic override 和全量重跑后的定量异常项。

**验证**

- v1→v1.1 有 47 条新增 selected candidate：44 条 `bilateral_multi_active_split`、3 条 `semantic_object_override`，按 sequence/primitive 合并后为 27 个任务场景。
- 3 条 override 为 drawer `selected:1081`、book `selected:1511`、asbestos mesh `selected:2258`，应全部人工查看。
- 已确认错误的 beaker `selected:1772` 的同场景 counterpart 为 test tube `selected:1773`，后者必须在官方 30 Hz 下补看。
- 高风险关系 primitive 包括 `place_inside`、`take_outside`、`assemble`、成对 `hold` 及 tripod/asbestos mesh 组合；普通独立 `rearrange` 只需代表样本和异常项。
- 当前 V1.4.22 viewer 的三条 laptop 以及 cut/full 差异较大的 alcohol burner `2649` 仍需人工复核；`2647/2648` 是同任务重复 take，可只看代表。
- 全量 V1.4.22 尚未运行，因此短 cut、超高 contact-extension ratio、多 motion component 和重复 take 的最终审查名单仍为 `PENDING`，不能用旧 v1.1 frame count 代替。

**证据入口**

- [`data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json`](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json) — 47 条新增候选及任务语义字段。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/temporal_segmentation_report.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/pilot_official30hz_20260916T130328Z/temporal_segmentation_report.json) — 当前 9 条 pilot 覆盖。

**回滚**

本次仅新增诊断记录，无代码或数据改动；删除本 activity 条目即可回滚记录。

## 2026-09-16 14:00:46 +0000 — OakInk2 待审查完整轨迹可视化与 GRAB/ARCTIC cache 状态核对

- activity_id: `ACT-20260916-140046-OICM-OAKINK2-REVIEW-GRAB-ARCTIC-STATUS`
- timestamp: `2026-09-16 14:00:46 +0000`
- modification_version: `V1.4.22`
- type: `diagnostic / operation / documentation`
- task_mode: `read-only/diagnostic` 后切换为 `run-only/operation`
- change_level: `L0`
- approval: `user-requested`
- approval_basis: 用户要求展示拿不准的 OakInk2 轨迹，并在运行中要求核对 GRAB/ARCTIC 导出及 OakInk2 实际播放速度。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有工作区修改）
- run_id: `review_uncertain_full30hz_20260916T133856Z/viewer_8112_full`
- run_status: `RUNNING`（adapter `COMPLETED`；GRAB/ARCTIC MANO2048/KNN 全量 cache 进程已退出，终态原因未知）
- viewer_url: `http://127.0.0.1:8112`
- scope: 仅为人工复核构建 29 条官方映射 30 Hz 完整 primitive 的独立 viewer adapter，并沿用上一轮 14 个 cut/full 项；核对已存在的 GRAB/ARCTIC 原始 MANO 与后续正式 cache 状态。不修改源 annotation、v1/v1.1 index、V1.4.22 时序 pilot、正式 cache、split、训练或科研结论。
- conclusion: `SUPPORTED`（29 条完整场景和现有 14 项可由统一 KNN/累计距离 viewer 读取，且使用官方 30 Hz 映射）；`INCONCLUSIVE`（样本保留判定、浏览器实际播放 FPS、GRAB/ARCTIC cache worker 退出原因）。

**原因**

上一诊断列出任务主导对象、显式语义修复、关系任务和重复 take 等待审查项。用户指出视觉上仍慢，因此同时区分源 mocap 120 Hz、官方映射后 30 Hz 帧序列与浏览器实际渲染帧率。旧 V1.4.20 的连续 120 Hz 误播产物未加入新 index。

**命令与产物**

- `PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python -u /tmp/oakink2_review_build.py`；脚本快照：[`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/build.py`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/build.py)。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/) — NAS 独立诊断运行目录，29 条完整轨迹与原有 14 项合成 43 项。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/index.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/index.json) — viewer-only 目录。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/run_manifest.json) — adapter `COMPLETED`。
- [`data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/viewer_8112_full/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm/oakink2_temporal_segmentation_v1_4_22/review_uncertain_full30hz_20260916T133856Z/viewer_8112_full/run_manifest.json) — 8112 viewer `RUNNING`。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — GRAB/ARCTIC 后续正式 cache 仍为过期 `STARTED` 状态，不代表进程仍在运行。

**验证**

- 29/29 新完整轨迹生成；index 共 43 项，每项 `shared.npz`、`left.npz`、`right.npz` 均存在。初始 laptop `2644` 的 483 个 viewer 帧对应 raw mocap `2784–4713`，相邻 raw ID 主要差 4，`source_fps=120`、`ds_rate=4`，理想播放时长约 16.1 秒；第 42 项 `--check-only --future-delta 30` 成功。
- 4 个关系任务候选有少量 raw ID 差 6 或 8，来自官方 `frame_id_list`；未使用固定 `%4` 或固定 raw stride。`curl http://127.0.0.1:8112` 返回 HTTP `200`，新 viewer 载入 43 项。
- `visualize_grab.py` 逐帧同步执行 `render()`，播放设置 30 FPS 不保证浏览器实际达到 30 FPS；本次没有测得客户端实际 FPS。
- GRAB/ARCTIC 原始 MANO 中间数据分别有 1335/301 条。后续 MANO2048/KNN cache 仅有 GRAB 1068/1335、ARCTIC 65/301，合计 1133/1636 条逐序列 manifest；两个 worker 进程均不存在，最后产物时间约 11:49 UTC，顶层 `index.json`、`manifest.json`、`validation_summary.json` 均不存在。旧 worker 报告仍为 `STARTED`，无法据此认定正常结束或失败原因。

**回滚**

停止当前 8112 viewer 并移除独立 `review_uncertain_full30hz_20260916T133856Z` 诊断目录及本 activity；源数据和正式 cache 不受影响。GRAB/ARCTIC cache 只读核对没有改变其状态或产物。

## 2026-09-16 14:07:53 +0000 — GRAB/ARCTIC MANO2048/KNN cache NAS 恢复启动

- activity_id: `ACT-20260916-140753-OICM-MANO-CACHE-RESUME`
- timestamp: `2026-09-16 14:07:53 +0000`
- modification_version: `V1.4.22`
- type: `diagnostic / operation / data`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确要求继续完成 GRAB/ARCTIC cache 导出，并重点核对本地磁盘是否耗尽；沿用已定稿 V1.4 计划和 V1.4.21 producer 合同。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`（保留既有用户修改）
- run_id: `object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`
- run_status: `RUNNING`
- scope: 仅从已完成 1133/1636 条恢复缺失 503 条；保持每侧 MANO2048 KNN、每侧 1538 decoder、K=32、2 cm 半径和既有 split。GPU 0 单 worker；不占用 GPU 3，不重写已完成序列或源 NPZ、旧 cache、训练配置。
- conclusion: `SUPPORTED`（输出路径确认为 NAS，前 4 条恢复序列成功）；全量结果仍为 `INCONCLUSIVE`。

**原因**

先前两个 worker 进程已经退出，旧 worker report 留在 `STARTED`，无法确定退出原因。根盘 99%（剩余约 42 GB），但 `data/processed_data` 与 NAS NFS 目标具有相同 device/inode，目标 NAS 剩余约 52 TB；现有约 190 GB cache 实际写在 NAS。本次将临时目录、命令脚本和 stdout log 也放在 NAS，避免本地根盘继续承载产物。

**命令与产物**

- `bash data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_worker_20260916T1402.sh`；脚本：[`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_worker_20260916T1402.sh`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_worker_20260916T1402.sh)。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/) — NAS 输出目录。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — `RUNNING`，保留原 `run_id` 并记录恢复时间。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_missing_20260916T1402.txt`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_missing_20260916T1402.txt) — 固定缺失 503 条清单。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_worker_20260916T1402.log`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_worker_20260916T1402.log) — 恢复运行 stdout/stderr。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/interrupted_20260916T1149/`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/interrupted_20260916T1149/) — 两份旧 `.partial` 与旧 worker 启动报告的 NAS 留档。
- `index.json` 与 `validation_summary.json`：PENDING（`run_status=RUNNING`，503 条完成并 finalize 后生成）。

**验证**

- `df -hT`、`readlink -f`、`findmnt`、`stat -L`：输出与 `data/processed_data` 为同一 NFS 目录；本地根盘剩余 42 GB，NAS 剩余 52 TB。
- `_entries` 精确比对：GRAB 1068/1335、ARCTIC 65/301 已完成，缺 GRAB 267、ARCTIC 236；两份 `.partial` 原样移至 NAS 留档，未删除。
- 恢复 worker 前 4 条 `COMPLETED`，输出路径均为 NAS；启动时没有修改代码、配置、split、GT 或 checkpoint。

**回滚**

停止恢复 worker；保留已完成的独立逐序列产物和旧 `.partial` 留档供审计，必要时仅回滚本次新增的运行状态、脚本和活动条目。源数据与旧 cache 不受影响。

## 2026-09-16 14:20:22 +0000 — GRAB/ARCTIC cache 恢复改为 GPU 0 双 worker

- activity_id: `ACT-20260916-142022-OICM-MANO-CACHE-RESHARD`
- timestamp: `2026-09-16 14:20:22 +0000`
- modification_version: `V1.4.22`
- type: `operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户要求继续完成已批准的 GRAB/ARCTIC 全量 cache；本次只调整同一 GPU 0 上的恢复并行度，不改变研究变量或输出合同。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`
- run_status: `RUNNING`
- scope: 单 worker 已完成 68/503 条后安全中断，保存其一份未完成 `.partial` 至 NAS；从剩余 435 条按确定性索引奇偶拆为 218/217 条，在 GPU 0 启动两个独立 worker，继续保持 GPU 3 上的其他任务不受本运行占用。
- conclusion: `SUPPORTED`（两个 shard 前各 6 条无失败）；全量仍为 `INCONCLUSIVE`。

**原因**

单 worker 受 NAS 等待影响，剩余时间较长；两进程只在同一已批准 GPU 0 上处理互斥的序列集合，并保持源数据、采样/KNN 参数和输出 schema 不变。

**产物与命令**

- `bash data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_sharded_20260916T1418.sh 0` 与同命令参数 `1`。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_sharded_20260916T1418.sh`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_sharded_20260916T1418.sh) — NAS 上可复现命令。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_missing_20260916T1418.txt`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_missing_20260916T1418.txt) — 两 shard 的固定输入全集。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_0_20260916T1418.log`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_0_20260916T1418.log) 与 [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_1_20260916T1418.log`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_1_20260916T1418.log) — 实时 stdout/stderr。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/interrupted_20260916T1418/`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/interrupted_20260916T1418/) — 中断单 worker 的一份 `.partial` 原样留档。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — `RUNNING` 与重分片信息。

**验证**

- `_entries` 在中断后重新计算缺失 435 条，shard 0/1 分别 218/217 条；目标路径均不存在 `.partial` 或不完整正式目录。
- 两 shard 各 6 条 `COMPLETED`、0 `FAILED`，GPU 0 可用显存充足，输出仍在同一 NAS NFS 目录。

**回滚**

停止两个 worker，保留已完成逐序列产物、两份中断 `.partial` 留档和运行日志；无需恢复源数据或旧 cache。

## 2026-09-16 15:11:18 +0000 — GRAB/ARCTIC cache 会话中断后恢复

- activity_id: `ACT-20260916-151118-OICM-MANO-CACHE-SESSION-RESUME`
- timestamp: `2026-09-16 15:11:18 +0000`
- modification_version: `V1.4.22`
- type: `diagnostic / operation / data`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户在前序会话明确要求完成 GRAB/ARCTIC cache，并在本会话要求读取前序会话后继续；仅恢复同一 run_id、同一固定清单和既有数据合同。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`
- run_status: `RUNNING`
- scope: 核对上一会话两个 GPU 0 worker 已退出后，从现有 1315/1636 条继续原 435 条互斥分片；不改变每侧 MANO2048、decoder 1538、K=32、2 cm、split、GT 或输出 schema。
- conclusion: `SUPPORTED`（恢复进程已独立运行且路径仍为 NAS）；全量与 finalize 结论仍为 `INCONCLUSIVE`。

**原因**

上一会话结束后两份 shard 日志均停在 57 条、失败 0，worker 进程不存在，顶层 manifest 仍为 `RUNNING`。目标目录有两份约 211 MB/148 MB 的 `.partial`；为避免 `--resume` 将其当作已存在而跳过，先原样移入 NAS 审计目录，再恢复同一脚本。

**命令与产物**

- `nohup setsid bash data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_sharded_20260916T1418.sh 0` 与参数 `1`；PID `2920190`、`2920191`。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — 同一 run 的恢复状态。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_0_20260916T1433.log`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_0_20260916T1433.log) 与 [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_1_20260916T1433.log`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/resume_shard_1_20260916T1433.log) — 本次恢复日志。
- [`data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/interrupted_20260916T1433/partials/`](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/interrupted_20260916T1433/partials/) — 两份未完成目录的 NAS 留档。
- `index.json` 与 `validation_summary.json`：PENDING（`run_status=RUNNING`，所有序列完成并 finalize 后生成）。

**验证**

- 只读计数：恢复前 GRAB 1068/1335、ARCTIC 247/301，合计 1315/1636；旧 shard 各 57 条完成、失败 0。
- `ps` 显示两个 worker 的 PPID 均为 1；GPU 0 恢复前剩余约 39.8 GiB，GPU 1/2/3 上既有进程未被改动。
- `df -hT`：NAS 剩余约 52 TB；本地根盘仍剩约 42 GB。新日志、临时目录和 cache 均位于 NAS。

**回滚**

停止 PID `2920190`、`2920191`；保留已完成序列、恢复日志和 `.partial` 审计目录。源数据、旧 cache 与 V1.2 计划无需恢复。

## 2026-09-16 15:54:39 +0000 — 接续会话并恢复 MANO cache 全量 finalize

- activity_id: `ACT-20260916-155438-OICM-MANO-FINALIZE-RESUME`
- timestamp: `2026-09-16 15:54:39 +0000`
- modification_version: `V1.4.22`
- type: `operation / diagnostic / documentation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户要求读取会话 `01a0aac3-48ec-7620-a492-634f1c26a297` 后继续，沿用已批准的全量 GRAB/ARCTIC MANO cache 导出；不新增数据合同或训练变量。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`
- run_status: `RUNNING`
- scope: 重跑既有 producer 的 finalize；仅追加本活动记录与 NAS 运行记录/校验产物，保留源码、GT、split、坐标、MANO correspondence、旧 cache 和其他运行。producer/index 继续使用原产物版本 `V1.4.21`，本次运行状态记录沿用当前 Task `V1.4.22`。

**原因**

两个恢复 worker 已于 15:30:52/53 UTC 正常完成各 217/218 条，均无失败。前会话 finalize 在等待期间被用户中断，本次检查不存在 producer 进程，且顶层 index/validation 未生成，因此恢复同一校验，并将进程与会话分离，记录 stdout、退出码和开始/结束时间。

**文件、命令与产物**

- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本次恢复与终态入口。
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) — 既有 final plan；MANO producer 的后续用户批准见 `ACT-20260916-104742-OICM-BILATERAL-MANO-CACHE-START`。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/) — 同一 NAS cache 目录。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.json) — 完整命令 argv、wrapper PID `2974798` 和本次校验状态。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.log](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.log) — 本次校验 stdout/stderr。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z_previous_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z_previous_manifest.json) — finalize 前 manifest 原样留档，保存历史恢复记录。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — 本次 finalize 信息及 producer 终态写入入口。
- 命令：`PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCm.tools.data.build_bilateral_mano_v1_4_cache finalize --grab-root data/processed_data/oicm_v1_4_raw/grab_mano_30hz --arctic-root data/processed_data/oicm_v1_4_raw/arctic_mano_30hz --split-index data/processed_data/object_interaction_cm_grab_arctic_inspire_geometric_v1_4/index.json --output-root data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4 --run-id object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`。

**验证**

- `workers/shard_00.json` 和 `workers/shard_01.json` 均为 `COMPLETED`，failures 为 0；原逐序列几何保持原样。
- 启动前按 `/proc/*/cmdline` 精确匹配 producer module，未发现重复进程；本次 wrapper 独立 session 启动。
- 最终 shape、finite、KNN 范围、覆盖和 index 仍待 finalize 结果；当前结论为 `INCONCLUSIVE`。

**回滚**

如需停止，仅停止本次 wrapper 的进程组，保留已有逐序列 cache、NAS 日志和前 manifest 快照，不删除或重建已完成序列。

## 2026-09-17 02:12:44 +0000 — GRAB/ARCTIC MANO cache 全量校验终态核对

- activity_id: `ACT-20260917-021244-OICM-MANO-FINALIZE-COMPLETE`
- timestamp: `2026-09-17 02:12:44 +0000`
- modification_version: `V1.4.22`
- type: `operation / diagnostic / documentation`
- task_mode: `run-only/operation`
- change_level: `L0`（仅补记已完成运行的终态）
- approval: `user-approved`
- approval_basis: 用户此前批准完成同一 GRAB/ARCTIC cache 导出及 finalize；本次要求浏览会话后继续，延续同一 `run_id`，不重跑或更改数据。
- skills_used: `research-experiment-workflow`, `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `3d59e14a292e2ac846e031e44c017ecd1d6096ad`
- worktree_dirty: `true`
- run_id: `object_interaction_cm_grab_arctic_mano_geometric_v1_4_20260916T105100Z`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（全量 cache 工程校验）；模型训练效果 `INCONCLUSIVE`。
- scope: 只读核对已结束的 finalize 并补记终态；未修改 cache 数组、index、manifest、GT、split、源码或其他运行。

**文件与证据入口**

- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本次终态记录及回滚入口。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/) — NAS 运行目录。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/run_manifest.json) — producer `COMPLETED` 终态。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/index.json) — 顶层序列索引。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/validation_summary.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/validation_summary.json) — 全量校验结果。
- [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.json](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.json) 与 [data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.log](../../../../../data/processed_data/object_interaction_cm_grab_arctic_mano_geometric_v1_4/finalize_20260916T155438Z.log) — finalize 命令、退出码及标准输出。

**原因**

前一条活动记录停在 `RUNNING`，而独立 wrapper 和 producer manifest 已于 2026-09-16 16:23:08 UTC 进入终态；活动时间线需与已生成证据一致。

**验证**

- 只读解析 `finalize_20260916T155438Z.json`：`exit_code=0`、`run_status=COMPLETED`；日志末行同样为 `COMPLETED`。
- `validation_summary.json`：1636 条、624537 帧通过，GRAB 1335、ARCTIC 301，`failures=[]`；split 为 train 1369、val 134、test 133。
- `run_manifest.json`：`run_status=COMPLETED`、结果 `SUPPORTED`；顶层 `index.json` 与校验报告均已存在。
- `readlink -f` 和 `df -hT`：运行目录解析到 `/mnt/ugreen_nas/.../processed_data`，文件系统为 NFS，剩余约 52 TB；无本地根盘 cache 输出证据。
- 本次没有训练 step、epoch、best metric、checkpoint、`metrics.jsonl` 或 `train.log`；这是数据 cache 运行。

**回滚**

如记录需修正，仅撤销本条终态说明；保留已完成的 NAS cache、index、manifest、校验报告和先前运行记录。

## 2026-09-17 07:31:41 +0000 — 归档提交 V1.4 已完成实现

- timestamp: `2026-09-17 07:31:41 +0000`
- activity_id: `ACT-20260917-073141-OICM-V14-COMMIT`
- modification_version: `V1.4.22`
- type: `operation, documentation`
- change_level: `L0`（仅整理和提交既有工作区内容；原实现影响等级见各历史活动）
- approval: `user-approved`
- approval_basis: 用户明确要求将其他代码分门别类提交。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `0cd030831307ea1d97223a00e69f37129acdae44`
- worktree_dirty: `true`
- scope: 归档本 Task 已完成的 V1.4 数据工具、可视化、研究脚本、测试及其文档；不修改科研语义或运行产物。

**文件**

- `src/task/ObjectInteractionCm/` — 本次提交的 Task 内源码、配置、测试、研究定义与文档；逐项路径以提交 diff 为准。
- [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) 和 [src/task/ObjectInteractionCm/docs/指导/V1.4.md](../指导/V1.4.md) — 原有最终计划和指导。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 已发生变更和运行的原始活动记录。
- `src/task/ObjectInteractionCm/research/full_export_smoke/grab/` — 生成的 NPZ、CSV、JSON 样本继续留在工作区，不纳入提交。

**原因**

按用户要求将此前已实现、已记录的本 Task 工作独立提交，保持与 Cmv2 及根级记录的提交边界。

**验证**

- `/home/wbcd/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCm/tests/test_arctic_landmark_contract.py src/task/ObjectInteractionCm/tests/test_bilateral_mano_v1_4_cache.py src/task/ObjectInteractionCm/tests/test_merge_dual_hand_stream.py src/task/ObjectInteractionCm/tests/test_oakink2_segment_visualizer.py src/task/ObjectInteractionCm/tests/test_oakink2_temporal_segments.py src/task/ObjectInteractionCm/tests/test_retarget_stage4_bilateral_inspire.py src/task/ObjectInteractionCm/tests/test_split_oakink2_active_tool.py src/task/ObjectInteractionCm/tests/test_v1_4_data_contract.py src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`：30 passed。
- `git diff --check` 通过；暂存差异及链接另行审计。既有运行结论沿用各历史记录，本次提交本身不产生新科研结论。

**回滚**

本次提交作为独立 Git commit，可按提交范围反向应用；NAS cache、旧 checkpoint 和运行输出未修改。

## 2026-09-17 07:38:31 +0000 — 忽略 full_export_smoke 生成样本

- timestamp: `2026-09-17 07:38:31 +0000`
- activity_id: `ACT-20260917-073831-OICM-SMOKE-IGNORE`
- modification_version: `V1.4.22`
- type: `documentation, operation`
- change_level: `L0`（Task-local Git 忽略规则，不改变代码、数据语义或公共合同）
- approval: `user-approved`
- approval_basis: 用户要求对剩余未提交内容选择忽略、取消更改或提交；此目录经检查为生成产物。
- skills_used: `research-change-control`
- branch: `feature/objectinteractioncmv2-v1.0.2`
- base_commit: `e62df715d39c9e9b806eae955bd20aa3d620a478`
- worktree_dirty: `true`
- scope: 仅忽略 `src/task/ObjectInteractionCm/research/full_export_smoke/grab/` 中已有和后续生成文件；样本原位保留。

**文件**

- [src/task/ObjectInteractionCm/research/full_export_smoke/.gitignore](../../research/full_export_smoke/.gitignore) — 将 `/grab/` 标记为生成产物目录。
- [src/task/ObjectInteractionCm/docs/logs/activity_log.md](activity_log.md) — 本次 Git 工作区清理记录。

**原因**

该目录仅有一次 smoke 导出的 `left.npz`、`right.npz`、`shared.npz`、`manifest.csv` 和 `meta.json`；它们属于研究生成产物，不应进入源码提交。

**验证**

- `git check-ignore -v` 对 5 个文件均命中该定向规则；`git status --short` 仅显示本次 `.gitignore` 与活动记录，提交后应为空。
- `git diff --cached --check` 与最新活动链接审计通过。本次无工程 smoke 或科研效果结论，结论 `INCONCLUSIVE`。

**回滚**

反向应用本次提交即可恢复原 Git 可见性；生成样本始终原位保留。

## 2026-09-17 09:08:08 +0000 — 启动 OakInk2 V1.4 Inspire cache 导出

- timestamp: `2026-09-17 09:08:08 +0000`
- activity_id: `ACT-20260917-090808-OICM-OAKINK2-CACHE-START`
- modification_version: `V1.4.23`
- type: `code, data, operation`
- change_level: `L3`（新增 OakInk2 正式选择与双手 Inspire geometric/KNN producer，并写入独立 NAS cache）
- approval: `user-approved`
- approval_basis: 用户明确要求“现在开始导出 OakInk2 的 cache”；范围沿用已定稿 [src/task/ObjectInteractionCm/docs/plan/V1.4.md](../plan/V1.4.md) §9、§10、§19。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `e444b1d0cce192dc9a310f0dcfd7bb67eea8f67c`
- worktree_dirty: `true`（本次新增 producer 尚未提交；既有用户改动未覆盖）
- scope: 读取 OakInk2 单物体 primitive index、原始双手 quaternion MANO、Stage3 几何；按官方 30 Hz、7 帧中心窗、2 mm/2 deg、运动优先和严格 `<2 cm` 规则创建独立选择 index；随后生成双手 Inspire 3076 点、4096 物体点和离线 KNN cache。只写 NAS 独立目录，不修改原始 annotation/Stage3、旧 v1/v1.1 index、GRAB/ARCTIC cache、val/test、训练配置和 checkpoint。

**运行**

- smoke selection/export：`run_id=oakink2_v1423_cache_smoke_20260917T090808Z`，`run_status=STARTED`，GPU `cuda:2`；目标目录和 manifest 在 smoke 创建后补写。
- full selection/export：`run_id=oakink2_v1423_cache_full_20260917T090808Z`，`run_status=PENDING`；等待 smoke 通过后启动后台运行。
- [src/task/ObjectInteractionCm/tools/data/export_oakink2_inspire_v1_4.py](../../tools/data/export_oakink2_inspire_v1_4.py) — 新增 producer，旧数据处理脚本保持不变。
- `data/processed_data/oicm_v1_4_raw/oakink2_inspire_selection_v1_4_23/` — smoke/full 选择 index 目录，NAS 路径解析后写入。
- `data/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/` — smoke/full cache 目录，终态链接待运行完成后补齐（当前 `PENDING`）。

**验证**

- 已通过 producer `py_compile`、导入检查和 `git diff --check`；尚未形成 cache 工程或科研结论。
- smoke 运行终态补记 selection/export 的 `run_manifest.json`、关键数组 shape/finite/KNN 校验和输出目录；full 运行终态另行追加。
- 当前结论：`INCONCLUSIVE`（代码链路尚未经过 smoke；不把训练 val 指标当作 OakInk2 cache 结论）。

**保护与回滚**

- 既有 GRAB/ARCTIC cache、旧 OakInk2 index、训练输出和外部 DExplore/dex-retargeting 仓库保持只读。
- 代码回滚入口为本次 producer 与版本/activity 增量；运行回滚只停止对应 run 并保留 NAS 审计产物，不删除既有数据。

## 2026-09-17 09:17:05 +0000 — OakInk2 全量选择与 cache 流水线后台运行

- timestamp: `2026-09-17 09:17:05 +0000`
- activity_id: `ACT-20260917-091705-OICM-OAKINK2-FULL-RUNNING`
- modification_version: `V1.4.23`
- type: `operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 延续用户“现在开始导出 OakInk2 的 cache”的明确指令和 V1.4 §9/§10/§19；smoke 已通过后启动全量。
- skills_used: `research-change-control`, `research-experiment-workflow`
- branch: `oyx`
- base_commit: `9f9b827db662aaa7d1c51754d083e421ccf06150`
- worktree_dirty: `false`（启动时；运行日志不写 Git 工作区）
- run_id: `oakink2_v1423_cache_full_20260917T091507Z`
- run_status: `RUNNING`
- command: NAS `run_full.sh`；先执行 `export_oakink2_inspire_v1_4.py select`，成功后执行 `export ... export --device cuda:2 --resume`。
- pid: `4002771`（wrapper）；selection 子进程 `4002773`。
- scope: 全部 `2177` 条单物体 primitive 候选；选择输出和 cache 输出均为独立 NAS 目录。GPU 2 仅用于后续 MANO/KNN，GPU 3 外部仿真保持不动。

**原因**

用户已批准 OakInk2 仅保留单物体 primitive，并要求开始生成正式 cache；smoke 已验证 producer 的输入、双手 Inspire retarget 和 KNN 合同可以运行，因此进入全量操作。

**运行与证据**

- [data/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/oakink2_v1423_cache_full_20260917T091507Z/pipeline_manifest.json](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/oakink2_v1423_cache_full_20260917T091507Z/pipeline_manifest.json) — `STARTED/RESTARTED` wrapper manifest；首次普通 `nohup` 子进程组被执行器回收，未产生选择产物，随后用 `setsid` 重启，未修改输入。
- [data/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/oakink2_v1423_cache_full_20260917T091507Z/pipeline.log](../../../../../data/processed_data/oicm_v1_4_raw/oakink2_inspire_full_runs/oakink2_v1423_cache_full_20260917T091507Z/pipeline.log) — 实时日志；当前已处理 `10/517` 个序列，`19` 个候选，`0` 个失败。
- `data/processed_data/oicm_v1_4_raw/oakink2_inspire_selection_v1_4_23/full_20260917T091507Z/` — `PENDING`；选择输出尚未终态，目录在选择完成后写入，运行中不把它当作终态证据。
- `data/processed_data/oicm_v1_4_raw/oakink2_inspire_bilateral_v1_4_23/full_20260917T091507Z/` — `PENDING`；cache 输出尚未开始。

**验证与状态**

- smoke selection：`1` 段、`1246` 帧、`0` 失败；smoke cache：双手 `3076` 点、物体 `4096` 点、KNN `K=32`、所有数组 finite，工程结论 `SUPPORTED`。
- full 运行阶段结论仍为 `INCONCLUSIVE`；最终以 selection/export 两个 run manifest、index、cache_manifest 和全量校验为准，不把 smoke 结果外推到全量或科研效果。

**保护与回滚**

- 现有 GRAB/ARCTIC cache、旧 OakInk2 v1/v1.1 index、训练权重、split、源 annotation/Stage3 和外部仓库未修改。
- 停止入口为 wrapper PID `4002771`；保留已写入的选择/缓存和 manifest，不删除既有产物。
