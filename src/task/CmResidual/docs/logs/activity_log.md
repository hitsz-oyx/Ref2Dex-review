## 2026-09-16 12:08:34 +0800 — V1.8 safe residual 实现、gate 与提交交接

- activity_id: `ACT-20260916-120834-CMRESIDUAL-V18-COMMIT`
- timestamp: `2026-09-16 12:08:34 +0800`
- modification_version: `V1.8`
- operation_category: `code`、`config`、`experiment`、`diagnostic`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准 V1.8 最小因果实现、B0/T10/E10/T20/E20 gate，并明确要求提交代码。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（提交前包含本 V1.8 全部已批准差异及无关的 ObjectInteractionCm 用户修改；后者明确排除）
- scope: 汇总提交 CmResidual V1.8 指导/最终 plan、no-Cm safe residual 实现、固定 sigma、训练/评估工具、合同测试、版本指针和完整实验记录；不提交 outputs、checkpoint、cache 或其他 Task 差异。
- conclusion: `SUPPORTED`（工程实现、单 seed 单场景 20-epoch preservation stability）；抓取改善、泛化、Cm 因果与统计显著性 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新到 V1.8。
- [src/task/CmResidual/docs/指导/V1.8.md](../指导/V1.8.md) — 用户提供的 V1.8 研究指导。
- [src/task/CmResidual/docs/plan/V1.8.md](../plan/V1.8.md) — 用户确认的最终计划和修订后 preservation metric。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态和结论边界。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 实现、失败处置、运行与终态时间线。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — B0/T10/E10/T20/E20 正式实验汇总。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — safe policy、gate、NumPy 和 epoch budget 合同测试。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — deterministic B0/E10/E20 evaluator。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — T10/T20 runner、budget 和 checkpoint 终态验证。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml) 和 [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml) — 默认显式保留 OI-Cm 路径。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafePPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafePPO.yaml) — 零 mean、固定 `sigma=0.1` safe 配置。
- [third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) — 显式 `learn_sigma=false` 冻结路径。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 1442-D no-Cm observation 路径。
- [outputs/CmResidual/cmresidual_v18_e20_20260916_120009/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/run_manifest.json) — 最终 E20 `COMPLETED / SUPPORTED` 证据；outputs 不纳入提交。

**原因**

V1.7 的大探索噪声训练退化无法区分 residual 初始化与 Cm 影响。V1.8 用 no-Cm、18D、zero-mean、固定小 sigma 的最小变量组合完成同协议短 gate，并将实现、运行证据与科研结论边界一起纳入版本控制。

**验证**

- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`17 passed, 2 warnings`；warnings 均为 Hydra 1.1 兼容提示。
- 修改/新增 Python 文件 `py_compile` 通过；`git diff --check` 通过。
- B0/T10/E10/T20/E20 正式 run 均为 `COMPLETED / SUPPORTED`；checkpoint model/optimizer/action finite，sigma=`0.1`，reload diff=`0.0`。
- E10 lift/contact ratio=`1.003871/1.164776`，E20=`0.973333/1.683597`，均通过 0.8 gate。
- 提交前使用显式路径暂存并运行 staged activity/link audit；不把工程 gate 误报为抓取提升。

**保护边界与回滚**

- 未修改 action mapper、mimic/clamp、reward、residual scale、DExplore/OI-Cm 权重、reference/source 数据、共享 `src/base/` 或其他 Task；未提交 outputs/checkpoint。
- Git 回滚入口为本次单一 V1.8 提交；实验产物保留在 ignored outputs 中，回滚代码不删除证据。

## 2026-09-16 12:00:09 +0800 — V1.8 E20 deterministic preservation 评估通过

- activity_id: `ACT-20260916-120009-CMRESIDUAL-V18-E20`
- timestamp: `2026-09-16 12:00:09 +0800`
- modification_version: `V1.8`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`
- task_mode: `run-only/operation -> change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准继续；T20 已从唯一合格 T10 checkpoint 恢复并完成 epoch 20，checkpoint 合同通过，按最终计划执行 E20。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: 加载明确的 T20 epoch-20 checkpoint，在与 B0/E10 完全相同的 deterministic GPU5/64 env/seed42/367 steps/no-Cm 协议下执行最终 preservation gate；并将已暴露的 checkpoint gate 失败终态从矛盾的 `COMPLETED` 修正为 `FAILED`。
- run_id: `cmresidual_v18_e20_20260916_120009`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 12:01:49 +0800`
- last_step: `367 / 367`
- last_epoch: `20`（被评估 checkpoint）
- best_metric: preservation gate `lift_ratio=0.9733328208`、`contact_ratio=1.6835971901`
- checkpoint: [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth)
- exit_reason: 完成 E20 预算；lift/contact 双 80% gate 通过，V1.8 计划内短程 stability 序列完成。
- conclusion: `SUPPORTED`（单 seed、单场景、20-epoch preservation stability）；抓取改善、泛化、Cm 因果与统计显著性 `INCONCLUSIVE`

**文件**

- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth) — 明确加载的 T20 checkpoint。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json) — 固定 B0 baseline。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态更新为 V1.8 stability gate 完整通过。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — B0/T10/E10/T20/E20 汇总、偏差处置和结论边界。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — checkpoint gate 失败时写入一致的 `FAILED` 终态。
- [outputs/CmResidual/cmresidual_v18_e20_20260916_120009](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009) — 完整 E20 目录。
- [outputs/CmResidual/cmresidual_v18_e20_20260916_120009/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/run_manifest.json) — `COMPLETED / SUPPORTED` 最终 gate。
- [outputs/CmResidual/cmresidual_v18_e20_20260916_120009/config.json](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/config.json) — resolved config、checkpoint 和协议快照。
- [outputs/CmResidual/cmresidual_v18_e20_20260916_120009/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/metrics.jsonl) — 367 行指标。
- [outputs/CmResidual/cmresidual_v18_e20_20260916_120009/eval.log](../../../../../outputs/CmResidual/cmresidual_v18_e20_20260916_120009/eval.log) — 评估日志。

**原因**

E20 是 V1.8 短程 stability 的最终 gate；训练统计不用于替代该 deterministic 同协议比较。

**验证**

- T20 `run_status=COMPLETED`，初始/最终 epoch=`10/20`，本 stage 10 行 metrics；checkpoint reload diff `0.0`，sigma 保持 `0.1`。
- E20 与 B0 protocol JSON 完全一致，367 行 observation/target finite。
- `mean_env_max_lift_m=0.082827`、mean contact occupancy `0.325996`；相对 B0 ratio=`0.973333 / 1.683597`，双 gate 通过。
- batch/time mean lift `-0.301940 m` 仅作诊断；没有追加 T30、重复 seed 或独立场景测试。
- runner 终态分支经 `py_compile` 与 Task tests 覆盖；gate 通过仍为 `COMPLETED`，失败路径现在直接写 `FAILED`。

**保护边界与回滚**

- 不覆盖 B0/T10/E10/T20，不改变 checkpoint、协议或阈值；E20 后不追加未规划训练。
- E20 使用独立目录；gate 失败则按计划终止并保留证据。

## 2026-09-16 11:58:28 +0800 — V1.8 T20 checkpoint 恢复训练完成

- activity_id: `ACT-20260916-115828-CMRESIDUAL-V18-T20`
- timestamp: `2026-09-16 11:58:28 +0800`
- modification_version: `V1.8`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准继续，E10 的 lift/contact preservation ratio 均通过 0.8 gate；按最终计划从明确的 T10 checkpoint 恢复到总 epoch 20。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: GPU5、64 env、seed42，从合格 T10 epoch-10 checkpoint 恢复到总 epoch 20；no-Cm、18D、固定 sigma/reward/scale/reference 不变。只有训练合同通过才执行 E20。
- run_id: `cmresidual_v18_t20_20260916_115828`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 12:00:00 +0800`
- last_step: `40960`
- last_epoch: `20 / 20`（从 epoch 10 恢复）
- best_metric: checkpoint deterministic reload action max abs diff `0.0`
- checkpoint: [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/CmResidualSafeT20/nn/last_CmResidualSafe_ep_20_rew__6.37_.pth)
- exit_reason: 从合格 T10 checkpoint 完成 epoch 11–20；训练和 checkpoint 合同全部通过。
- conclusion: `SUPPORTED`（T20 工程训练与 checkpoint 合同）；科研结论 `INCONCLUSIVE`

**文件**

- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth) — 唯一允许的初始 checkpoint。
- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828) — 完整 T20 目录。
- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/run_manifest.json) — `COMPLETED / SUPPORTED` 终态。
- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/config.json](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/config.json) — resolved config 与初始 checkpoint SHA。
- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/metrics.jsonl) — epoch 11–20 共 10 行指标。
- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/train.log](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/train.log) — 恢复训练日志。
- [outputs/CmResidual/cmresidual_v18_t20_20260916_115828/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v18_t20_20260916_115828/checkpoint_validation.json) — epoch/frame、finite、sigma 与重载验证。

**原因**

E10 已证明 T10 deterministic mean policy 在同协议下保持 B0：lift ratio `1.003871`、contact ratio `1.164776`。按计划只追加 epoch 11–20，验证短程继续学习后是否仍保持该行为。

**验证**

- E10 `run_status=COMPLETED`，367 行 finite；protocol 与 B0 JSON 完全一致，双 gate 均通过。
- T20 从 epoch 10 恢复并严格完成到 epoch 20，本 stage 有 10 行 metrics；model/optimizer/action finite。
- checkpoint epoch/frame=`20/40960`、reload diff=`0.0`、sigma min/max=`0.099999994`。
- epoch 20 training rollout residual RMS `0.107345`、success fraction `0.0`；效果判断留给 E20。

**保护边界与回滚**

- 不加载超预算失败 run 的 checkpoint，不从 latest 模糊选择，不覆盖 T10/E10；不改变预算或科研变量。
- 异常或 checkpoint 合同失败时停止且不执行 E20；新 run 可独立移除。

## 2026-09-16 11:56:33 +0800 — V1.8 E10 deterministic preservation 评估通过

- activity_id: `ACT-20260916-115633-CMRESIDUAL-V18-E10`
- timestamp: `2026-09-16 11:56:33 +0800`
- modification_version: `V1.8`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准继续；修复后的 T10 已完成 10 epochs 并通过 checkpoint 重载、finite 与固定 sigma 合同，按最终计划进入 E10。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: 加载明确的 T10 epoch-10 checkpoint，在与新 B0 完全相同的 GPU5/64 env/seed42/367 steps/no-Cm 协议下执行 deterministic mean-action rollout；按 lift/contact 双 80% gate 判定，失败则不启动 T20。
- run_id: `cmresidual_v18_e10_20260916_115633`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 11:58:16 +0800`
- last_step: `367 / 367`
- last_epoch: `10`（被评估 checkpoint）
- best_metric: `mean_env_max_lift_m=0.0854256004`
- checkpoint: [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth)
- exit_reason: 完成 E10 预算；lift/contact ratio=`1.003871 / 1.164776`，双 80% gate 通过。
- conclusion: `SUPPORTED`（T10 preservation stability）；抓取改善科研结论 `INCONCLUSIVE`

**文件**

- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth) — 明确加载的 T10 checkpoint。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json) — B0 协议与 preservation baseline。
- [outputs/CmResidual/cmresidual_v18_e10_20260916_115633](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633) — 完整 E10 目录。
- [outputs/CmResidual/cmresidual_v18_e10_20260916_115633/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633/run_manifest.json) — `COMPLETED / SUPPORTED` 双 gate 终态。
- [outputs/CmResidual/cmresidual_v18_e10_20260916_115633/config.json](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633/config.json) — resolved config、checkpoint 和协议快照。
- [outputs/CmResidual/cmresidual_v18_e10_20260916_115633/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633/metrics.jsonl) — 367 行指标。
- [outputs/CmResidual/cmresidual_v18_e10_20260916_115633/eval.log](../../../../../outputs/CmResidual/cmresidual_v18_e10_20260916_115633/eval.log) — 评估日志。

**原因**

训练 rollout 指标不能替代 deterministic preservation 测试。E10 只比较 checkpoint mean action 与同协议 B0 的 mean-env-max-lift/contact，阈值固定为 `0.0680769563 m / 0.1549046345`。

**验证**

- T10 `run_status=COMPLETED`、epoch/frame=`10/20480`、10 行 metrics；model/optimizer/action finite，checkpoint reload action diff `0.0`，sigma 为 `0.1` 且冻结。
- E10 与 B0 protocol JSON 完全一致；所有 observation/target finite。
- `mean_env_max_lift_m=0.085426`、mean contact occupancy `0.225536`，相对 B0 ratio=`1.003871 / 1.164776`，双 gate 通过。
- batch/time `mean_lift_m=-0.449473` 仅为诊断量；本结果只支持 preservation stability，不证明抓取改善。

**保护边界与回滚**

- 不修改或覆盖 B0/T10 checkpoint，不改变评估协议，不使用 stochastic action，不启动 T20。
- E10 使用独立目录；失败或 gate 未通过时保留证据并停止。

## 2026-09-16 11:53:31 +0800 — V1.8 T10 epoch 预算与重载校验修复终态

- activity_id: `ACT-20260916-115331-CMRESIDUAL-V18-T10-RETRY`
- timestamp: `2026-09-16 11:53:31 +0800`
- modification_version: `V1.8`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`
- task_mode: `run-only/operation -> change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准继续 V1.8 gate；首个 T10 暴露 rl_games epoch override 路径错误。修复只确保既定 10-epoch 预算真实生效，经合同测试后从零重跑，不改变科研变量。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: 将 T10/T20 epoch override 从无效的根级 `max_iterations` 改为 rl_games 实际读取的 `train.params.config.max_epochs`，启动前强制校验 resolved budget；用新 run_id 从零重跑 T10，不复用超预算 run 的任何 checkpoint。
- run_id: `cmresidual_v18_t10_retry_20260916_115331`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 11:55:27 +0800`
- last_step: `20480`
- last_epoch: `10 / 10`
- best_metric: checkpoint deterministic reload action max abs diff `0.0`
- checkpoint: [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/CmResidualSafeT10/nn/last_CmResidualSafe_ep_10_rew__6.49_.pth)
- exit_reason: 完成严格 10-epoch 预算；将 checkpoint model 切到 eval mode 后，重载重复 action 差为 `0.0`，全部合同通过。
- conclusion: `SUPPORTED`（T10 工程训练与 checkpoint 合同）；科研结论 `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — 正确覆盖并校验 rl_games `max_epochs`。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — resolved epoch budget 合同测试。
- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331) — 完整 T10 目录。
- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/run_manifest.json) — `COMPLETED / SUPPORTED` 终态。
- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/config.json](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/config.json) — resolved `max_epochs=10` 与 safe-policy preflight。
- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/metrics.jsonl) — 10 行 epoch 指标。
- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/train.log](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/train.log) — 训练日志。
- [outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v18_t10_retry_20260916_115331/checkpoint_validation.json) — epoch/frame、finite、sigma 和 deterministic reload 验证。

**原因**

IsaacGymEnvs 根级 `max_iterations` 没有覆盖本 safe config 显式声明的 `train.params.config.max_epochs=20`，导致首个 T10 实际跑到 epoch 20。修复直接作用于唯一消费方字段，并在创建训练子进程前拒绝任何 budget drift。

**验证**

- `python3 -m py_compile src/task/CmResidual/tools/run_ppo_stability.py`：通过。
- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`17 passed, 2 warnings`。
- 新合同确认根级 `max_iterations` 与训练预算解耦，`train.params.config.max_epochs=10` 才是 T10 权威值；`git diff --check` 通过。
- 严格完成 10 epochs，TensorBoard/metrics 共 10 条；epoch/frame=`10/20480`。
- checkpoint model/optimizer/action 均 finite；sigma min/max=`0.099999994`，deterministic reload action max abs diff=`0.0`。
- epoch 10 training rollout：residual RMS `0.105227`、success fraction `0.0`；这些训练统计不替代 E10。

**保护边界与回滚**

- 不使用首个超预算 run 的 epoch-10/20 checkpoint，不覆盖其证据；不改 PPO 超参数、B0、reward、scale、输入或 GPU。
- 回滚 runner/test 两处 budget 修复即可撤销实现变化；新 run 使用独立目录。

## 2026-09-16 11:50:10 +0800 — V1.8 T10 safe residual PPO 超预算失败

- activity_id: `ACT-20260916-115010-CMRESIDUAL-V18-T10`
- timestamp: `2026-09-16 11:50:10 +0800`
- modification_version: `V1.8`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 修订后的 B0 已通过，用户明确要求继续；按 V1.8 最终计划启动 T10，只有训练/checkpoint 合同通过后才允许 E10。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（运行使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: GPU5、64 env、seed42、10 epochs、horizon32、minibatch2048、no-Cm、18D residual，mean 精确零初始化且 sigma 固定为 `0.1`；不加载 residual checkpoint，不改变 B0、reward、scale 或 reference。
- run_id: `cmresidual_v18_t10_20260916_115010`
- run_status: `FAILED`
- completed_at: `2026-09-16 11:52:13 +0800`
- last_step: `40960`
- last_epoch: `20 / 10 requested`
- best_metric: `null`（runner 在预算审计失败后未接受训练指标）
- checkpoint: `null`（产出的 epoch-10/20 文件均因 run 超预算而不准入）
- exit_reason: resolved `train.params.config.max_epochs=20`，TensorBoard 得到 20 epochs，而 T10 只批准 10 epochs。
- conclusion: `INVALID_IMPLEMENTATION`；科研结论 `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/docs/plan/V1.8.md](../plan/V1.8.md) — T10 预算、合同与停止条件。
- [outputs/CmResidual/cmresidual_v18_t10_20260916_115010](../../../../../outputs/CmResidual/cmresidual_v18_t10_20260916_115010) — 超预算失败运行目录。
- [outputs/CmResidual/cmresidual_v18_t10_20260916_115010/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_t10_20260916_115010/run_manifest.json) — `FAILED / INVALID_IMPLEMENTATION` 终态。
- [outputs/CmResidual/cmresidual_v18_t10_20260916_115010/config.json](../../../../../outputs/CmResidual/cmresidual_v18_t10_20260916_115010/config.json) — 直接证据：root `max_iterations=10`，rl_games `max_epochs=20`。
- [outputs/CmResidual/cmresidual_v18_t10_20260916_115010/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_t10_20260916_115010/metrics.jsonl) — 失败状态行，不作为 epoch 指标。
- [outputs/CmResidual/cmresidual_v18_t10_20260916_115010/train.log](../../../../../outputs/CmResidual/cmresidual_v18_t10_20260916_115010/train.log) — 20 epochs 与 checkpoint 写入日志。

**原因**

验证固定小探索噪声、零 mean 的 18D residual PPO 能否完成首个 10-epoch 工程训练阶段，并产出可重载、finite、sigma 保持 `0.1` 的 checkpoint，随后再由独立 E10 判断是否保持 B0 行为。

**验证**

- 启动前确认新 B0 manifest 为 `COMPLETED`、baseline usable 且 continuation allowed；阈值为 lift `0.0680769563 m`、contact `0.1549046345`。
- runner `--help` 通过；GPU5 为 `8 MiB / 0%`，无 compute process。
- runner 发现 TensorBoard 有 20 而非 10 epochs 后将 run 标为 `FAILED`；epoch-10/20 checkpoint 均不准入，未启动 E10。

**保护边界与回滚**

- 不覆盖 B0/V1.7 或其他 run，不加载旧 residual checkpoint，不启用 OI-Cm，不修改代码、配置或训练预算。
- 异常、non-finite、sigma 漂移或 checkpoint 合同失败时立即停止并保留新运行证据；回滚只处理本 run。

## 2026-09-16 11:41:34 +0800 — V1.8 preservation lift 修订与 B0 重跑终态

- activity_id: `ACT-20260916-114134-CMRESIDUAL-V18-B0-ENVMAX`
- timestamp: `2026-09-16 11:41:34 +0800`
- modification_version: `V1.8`
- operation_category: `code`、`experiment`、`diagnostic`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认将不可解释的整段 batch/time mean lift gate 改为 mean per-environment maximum lift，并重跑 B0；其余协议保持不变。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（运行使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: 修订 V1.8 plan 与 evaluator 的 preservation lift 指标，增加 metric schema 锁定和合同测试；按同一 GPU5/64 env/seed42/367 steps/no-Cm/zero-residual 协议重跑 B0，不启动 T10。
- run_id: `cmresidual_v18_b0_envmax_20260916_114134`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 11:42:59 +0800`
- last_step: `367 / 367`
- last_epoch: `null`
- best_metric: `mean_env_max_lift_m=0.0850961953`
- checkpoint: `null`
- exit_reason: 完成修订后的 B0 预算，zero residual parity 与正 preservation baseline 均通过；按本轮边界停在 T10 之前。
- conclusion: `SUPPORTED`（运行、zero-residual parity 与 preservation baseline）；科研结论 `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/docs/plan/V1.8.md](../plan/V1.8.md) — 最终计划明确 `mean_env_max_lift_m` 定义与 80% gate。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — 跨 reset 累计每个环境的 rollout max lift，并写入 protocol/metrics/summary。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — preservation gate schema 更新。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态更新为新 B0 可用、T10 未启动。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 正式 B0 结果与后续固定阈值。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134) — 完整运行目录。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/run_manifest.json) — `COMPLETED` 终态、metric schema 与 continuation gate。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/config.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/config.json) — resolved config 与运行 override。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/metrics.jsonl) — 367 行逐步指标。
- [outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/eval.log](../../../../../outputs/CmResidual/cmresidual_v18_b0_envmax_20260916_114134/eval.log) — 仿真初始化和评估日志。

**原因**

success reset 与未成功环境后续下落会把整段 batch/time mean lift 拉成负值，不适合作为 preservation ratio 分母。新指标对每个环境从 `0 m` 起保留 367 步内达到的最大 lift，再对 64 个环境等权平均；不改变物理、reset、reward 或训练目标。

**验证**

- `python3 -m py_compile src/task/CmResidual/tools/eval_residual_stability.py`：通过。
- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`16 passed, 1 warning`。
- evaluator `--help` 与 `git diff --check`：通过；GPU5 启动前为 `8 MiB / 0%`。
- B0 完成 `367/367` steps；所有 observation/target finite，`max_residual_target_delta=0`。
- `mean_env_max_lift_m=0.085096`、mean contact occupancy `0.193631`；二者均为正，baseline 可用。
- 后续 80% 阈值固定为 lift `0.0680769563 m`、contact `0.1549046345`；本轮未启动 T10。

**保护边界与回滚**

- 未修改 reward、residual scale、action、reset、reference/source、checkpoint、GPU、OI-Cm 或训练预算；旧 B0 证据保留。
- 回滚本次 plan/evaluator/test 差异即可恢复旧 gate；新 run 使用独立目录，不覆盖历史产物。

## 2026-09-16 11:33:18 +0800 — V1.8 NumPy 兼容修复与 B0 重试终态

- activity_id: `ACT-20260916-113318-CMRESIDUAL-V18-B0-RETRY`
- timestamp: `2026-09-16 11:33:18 +0800`
- modification_version: `V1.8`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`
- task_mode: `run-only/operation -> change -> run-only/operation`
- change_level: `L2`（Task-local L1 兼容修复 + 已批准的 L2 gate 运行）
- approval: `user-approved`
- approval_basis: 用户要求执行 V1.8 测试；首个 B0 在 step 0 暴露 NumPy 1.24 与 pinned Isaac Gym 的环境兼容错误。兼容修复不改变科研变量，完成定向测试后以新 run_id 重试同一 B0。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（使用已获批但尚未提交的 V1.8 实现；其他 Task 的既有用户修改未触碰）
- scope: 仅在 Task-local evaluator 和 V1.8 training 子进程入口恢复 pinned Isaac Gym 唯一使用的 `np.float` alias；随后按原 GPU5/64 env/seed42/367 steps/no-Cm/zero-residual 协议重试 B0。
- run_id: `cmresidual_v18_b0_retry_20260916_113318`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 11:35:01 +0800`
- last_step: `367 / 367`
- last_epoch: `null`
- best_metric: `mean_lift_m=-0.4904631897`
- checkpoint: `null`
- exit_reason: 完成 B0 预算；因 baseline mean lift 非正，planned E10/E20 lift preservation ratio 无定义，停止后续 T10。
- conclusion: `SUPPORTED`（运行、协议和 zero-residual parity）；`INVALID_IMPLEMENTATION`（planned lift-ratio baseline）；科研结论 `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — 仿真导入前安装最小 NumPy alias。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — 训练子进程用同一兼容 bootstrap，避免 T10 重现该环境错误。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — 新增 alias 范围合同测试。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 正式实验结果、指标语义与停止边界。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前阶段更新为 B0 完成、T10 阻断。
- [outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318) — 完整运行目录。
- [outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/run_manifest.json) — `COMPLETED` 终态、协议、输入 SHA 和 continuation blocker。
- [outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/config.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/config.json) — resolved config 与运行 override。
- [outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/metrics.jsonl) — 367 行逐步指标。
- [outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/eval.log](../../../../../outputs/CmResidual/cmresidual_v18_b0_retry_20260916_113318/eval.log) — 仿真初始化和评估日志。

**原因**

NumPy 1.24 已移除 `np.float`，而 pinned Isaac Gym 的 `torch_utils.py` 在导入时仍引用该 alias。修复限定在 V1.8 进程入口，不修改外部 Isaac Gym 或共享 vendor，也不改变输入、仿真、策略或指标。

**验证**

- `python3 -m py_compile` 覆盖两个 V1.8 工具：通过。
- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`16 passed, 1 warning`。
- `git diff --check`：通过。
- B0 完成 `367/367` steps；所有 observation/target finite，`max_residual_target_delta=0`。
- `mean_lift_m=-0.490463`、`max_lift_m=0.069457`、mean contact occupancy `0.206335`、mean tip distance
  `0.175315 m`、max success fraction `0.359375`；累计 `done_count=302`。
- planned preservation gate 需要正 baseline 才能解释比例；因此 `continuation_allowed=false`，未启动 T10。

**保护边界与回滚**

- 未修改 external Isaac Gym、vendor `train.py`、NumPy 环境、reward、scale、action、reference、checkpoint 或 GPU。
- 回滚 alias helper、训练 bootstrap 和对应测试即可撤销兼容修复；首个失败 run 保留，不覆盖。

## 2026-09-16 11:30:59 +0800 — V1.8 B0 同协议零残差基线失败

- activity_id: `ACT-20260916-113059-CMRESIDUAL-V18-B0`
- timestamp: `2026-09-16 11:30:59 +0800`
- modification_version: `V1.8`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户在 V1.8 最终计划与实现完成后明确要求“现在测试一下”；按计划顺序只启动第一道 B0，不跳过 gate 启动 T10。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（运行使用已获批但尚未提交的 V1.8 实现；保留其他 Task 的既有用户修改）
- scope: GPU5、64 env、seed42、367 control steps、`terminateOnSuccess=true`、no-Cm、全零 residual 的 same-protocol baseline；不训练、不加载 residual checkpoint、不修改研究变量。
- run_id: `cmresidual_v18_b0_20260916_113059`
- run_status: `FAILED`
- completed_at: `2026-09-16 11:31:47 +0800`
- last_step: `0 / 367`
- best_metric: `null`
- checkpoint: `null`
- exit_reason: NumPy 1.24 已移除 pinned Isaac Gym `torch_utils.py` 使用的 `np.float`，仿真环境创建前抛出 `AttributeError`。
- conclusion: `INVALID_IMPLEMENTATION`；科研结论 `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/docs/plan/V1.8.md](../plan/V1.8.md) — 本次运行依据的最终计划。
- [outputs/CmResidual/cmresidual_v18_b0_20260916_113059](../../../../../outputs/CmResidual/cmresidual_v18_b0_20260916_113059) — 失败运行目录。
- [outputs/CmResidual/cmresidual_v18_b0_20260916_113059/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v18_b0_20260916_113059/run_manifest.json) — `FAILED` 终态与输入/协议快照。
- `outputs/CmResidual/cmresidual_v18_b0_20260916_113059/metrics.jsonl` — 未进入 rollout，因此未生成且不作为链接。
- [outputs/CmResidual/cmresidual_v18_b0_20260916_113059/eval.log](../../../../../outputs/CmResidual/cmresidual_v18_b0_20260916_113059/eval.log) — NumPy/Isaac Gym traceback。

**原因**

在启动任何 residual PPO 前，用与后续 E10/E20 完全一致的 GPU、环境数、seed、控制步数、reset、输入 SHA、reward 和 residual scale，建立可比较的全零 residual 基线。

**验证**

- 启动前确认 GPU5 显存占用 `8 MiB`、利用率 `0%`，无 compute process。
- DExplore checkpoint、eligible v2 reference 与 legacy-parity source tensor 均存在；runner 将在初始化时复核 SHA 和协议合同。
- runner 在仿真环境创建前终止，manifest 为 `FAILED`，`last_step=0`；没有产生 metrics 或 checkpoint，不形成科研结论。

**保护边界与回滚**

- 不覆盖旧 run，不加载 V1.7 residual checkpoint，不启动 T10/T20，不改变 reward、scale、action、reference 或设备。
- 如初始化或 rollout 失败，按计划停止并保留新运行目录作为诊断证据；回滚仅移除本次新产物和 activity，不影响历史运行。

## 2026-09-16 11:17:44 +0800 — V1.8 safe residual 实现与工程验证

- activity_id: `ACT-20260916-111744-CMRESIDUAL-V18-SAFE-RESIDUAL`
- timestamp: `2026-09-16 11:17:44 +0800`
- modification_version: `V1.8`
- operation_category: `code`、`config`、`experiment`、`diagnostic`、`documentation`
- task_mode: `read-only/diagnostic -> change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认采用 V1.8 协商口径：18D、actor/critic 均 no-Cm、mean 精确零初始化、固定
  `sigma=0.1`，不改 action/mimic/clamp/reward，并以 B0/T10/E10/T20/E20 同协议 gate 验证。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `35e8cd87cc9de2c9ae4e51f9cfa551fd5448f2db`
- worktree_dirty: `true`（接手时已有用户未跟踪的 V1.8 指导和其他 Task 的 activity 修改；均保留且未纳入本次改动）
- scope: 定稿 V1.8 plan，实现 no-Cm safe policy、冻结小 sigma、分阶段训练/确定性评估工具和合同测试；
  本次只实现与验证，未启动 GPU 训练或评估，未生成 run_id、run manifest 或实验产物。
- conclusion: `SUPPORTED`（工程合同与单元测试）；`INCONCLUSIVE`（residual stability、抓取效果和 Cm 因果）

**文件**

- [src/task/CmResidual/docs/指导/V1.8.md](../指导/V1.8.md) — 用户提供的研究指导；本次只读且未修改。
- [src/task/CmResidual/docs/plan/V1.8.md](../plan/V1.8.md) — 经用户确认的最终执行计划、保护边界与 gate。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态和结论边界更新到 V1.8。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — safe policy 与 preservation gate 合同测试。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — 同协议 B0/E10/E20 确定性评估与 80% gate。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — 仅允许 T10 或由 T10 checkpoint 恢复的 T20 runner。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 显式 no-Cm observation 路径。
- [third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) — 显式 `learn_sigma=false` 时冻结 sigma。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafePPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafePPO.yaml) — 零 mean、固定 `sigma=0.1` 的 safe 配置。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml) 和 [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml) — 默认显式保留 OI-Cm。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新到 V1.8。

**原因**

- V1.7 rollout 退化不足以区分 residual 初始化/探索噪声、Cm 输入或训练预算的影响。V1.8 先隔离为
  no-Cm + zero-mean + small fixed sigma 的最小变量组合，不把当前改动解释为 Cm 改进。
- `useOiCmContext=false` 时跳过 surface geometry、OI-Cm checkpoint 加载和推理，policy observation 为
  1442-D base observation；默认值和 canonical task config 均保持 `true`，兼容旧 2005-D 路径。
- safe 配置保持 18D action，mean head 权重/bias 为零，`log_sigma=ln(0.1)` 且不参与梯度更新；旧配置未
  声明 `learn_sigma=false` 时保持原行为。
- runner 固定 GPU5、64 env、seed42 和 eligible v2 reference；T20 必须显式接收同版本 T10 checkpoint。
  evaluator 固定 367 steps，并要求 E10/E20 与 B0 协议一致后才比较 mean lift/contact occupancy。

**验证**

- `python3 -m py_compile` 覆盖 task、network builder 和两个新增工具：通过。
- `PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests`：
  `15 passed, 1 warning`；warning 为 Hydra 1.1 兼容提示。
- 两个新增工具的 `--help`：通过；未初始化 Isaac Gym、未占用 GPU、未产生运行目录。
- `git diff --check`：通过。
- 未执行 GPU smoke、B0、T10/E10 或 T20/E20；工程 smoke 不构成科研效果证据，科研结论保持
  `INCONCLUSIVE`。

**保护边界与回滚**

- 未修改 18D action mapper、mimic/null directions、physical residual clamp、`0.015 m / 0.20 rad / 0.08 rad`
  scale、reward、DExplore/OI-Cm 权重、reference/source 数据、checkpoint、共享 `src/base/` 或其他 Task。
- 未恢复 V1.7、未启动新训练/评估、未覆盖旧 run 或产物；其他 Task 的已有工作树修改保持原样。
- Git 回滚入口为本 activity 所列 V1.8 文件差异；移除 safe 配置/工具并回退 task/builder/config/test/文档
  差异即可恢复 V1.7 默认行为，无数据迁移或产物清理。

## 2026-09-16 10:42:38 +0800 — V1.7.1 正式 PPO 训练终态与部分结果归档

- activity_id: `ACT-20260916-104238-CMRESIDUAL-V171-PPO-TERMINAL`
- timestamp: `2026-09-16 10:42:38 +0800`
- modification_version: `V1.7.1`
- operation_category: `experiment`、`diagnostic`、`operation`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认按 V1.7.1 记录外部中断、部分指标、checkpoint 解释和科研结论边界，并提交相关记录。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `e4868b78581e723691e8581a1fe69e36b99f2438`
- worktree_dirty: `true`（接手时已有同一 run 的 ETA activity；本次保留并一并归档）
- scope: V1.7 正式 PPO 运行终态化、部分 TensorBoard 指标导出、Task 状态与实验结论同步；不恢复或复跑训练，不改代码、配置、数据、checkpoint 或研究变量。
- run_id: `cmresidual_ppo_formal_v17_20260915_2345`
- run_status: `FAILED`
- last_step: `335872`
- last_epoch: `165 / 1000`
- best_metric: epoch 50 checkpoint eligible `rewards/iter=5.883934020996094`
- best_checkpoint: [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/CmResidual.pth](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/CmResidual.pth)
- latest_checkpoint: [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/last_CmResidual_ep_100_rew_-6470.568.pth](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/last_CmResidual_ep_100_rew_-6470.568.pth)
- exit_reason: 外部执行会话以 `exit_code=-1` 终止；无 Python traceback、CUDA OOM 或内核 OOM 证据。
- conclusion: `INVALID_IMPLEMENTATION`（计划运行与终态验证未完成）；科研结论 `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/docs/plan/V1.7.md](../plan/V1.7.md) — 本次沿用的最终计划和停止条件。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态更新到 V1.7.1。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 本次终态唯一活动入口。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 部分结果、对照边界和科研解释。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新到 V1.7.1。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345) — 原运行目录；未覆盖 checkpoint。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/config.json](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/config.json) — 原 resolved config。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/run_manifest.json) — 补记 `FAILED`、中断原因、最后进度和 checkpoint。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/metrics.jsonl) — 从原始 TensorBoard 导出的 165 行部分 epoch 指标。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/train.log) — 原训练日志，行缓冲止于 epoch 101。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/summaries/events.out.tfevents.1789486158.server](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/summaries/events.out.tfevents.1789486158.server) — epoch 165 的原始标量证据。

**原因**

- 原会话在用户查询状态时遇到服务过载，长期命令随后以 `exit_code=-1` 结束；runner 未执行终态收尾，manifest 因而错误停留在 `STARTED`。
- zero-residual gate 有接触与抬升工程证据；V1.7 epoch 165 的 lift、tip distance、contact、success fraction 和 residual RMS 显示训练 rollout 明显退化。两者不是同协议统计对照，故不把观察上升为 residual 或 OI-Cm 的因果结论。

**验证**

- TensorBoard `info/epochs` 共 `165` 条，最后 `epoch=165`、`frame=335872`；runner 所需 10 个 scalar tag 均存在，已按既有 schema 导出部分 `metrics.jsonl`。
- epoch 165：`lift_mean=-0.545820 m`、`tip_distance_mean=1.049141 m`、`contact_occupancy=0.165625`、`success_fraction=0`、`residual_rms=0.852116`。
- epoch 50/100 checkpoint 可读取；策略平均标准差约 `1.0051 / 0.9984`。未执行仿真恢复或 deterministic policy evaluation。
- 进程检查确认 runner/训练子进程均不存在；GPU5 无该 run 的计算进程。训练效果、Cm 因果和普遍抓取结论保持 `INCONCLUSIVE`。

**保护边界与回滚**

- 未修改代码、配置、reference/source 数据、reward、模型权重、checkpoint、V1.7 plan、指导、共享 `src/base/` 或其他 Task；未启动恢复、复跑或新评估。
- Git 回滚入口为本次文档提交；生成产物回滚只删除新导出的 `metrics.jsonl`，并将同一 run manifest 恢复到归档前副本/字段。原 TensorBoard、train log 和 checkpoint 不受影响。

## 2026-09-16 00:21:56 +0800 — V1.7 正式训练耗时查询

- timestamp: `2026-09-16 00:21:56 +0800`
- activity_id: `ACT-20260916-002156-CMRESIDUAL-TRAIN-ETA`
- modification_version: `V1.7`（沿用被查询运行版本；版本指针仍为 V1.6，本次不调整）
- operation_category: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户查询训练预计耗时；只读检查并记录状态。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `e4868b78581e723691e8581a1fe69e36b99f2438`
- scope: 训练进度与耗时估算；不改变代码、配置、训练预算或进程。
- run_id: `cmresidual_ppo_formal_v17_20260915_2345`
- run_status: `RUNNING`（PID 576668 存在；runner manifest 在运行中仍写 STARTED）
- last_epoch: `10` / `1000`
- conclusion: `SUPPORTED`（当前进度证据）；`INCONCLUSIVE`（预计完成时间和训练效果）

**文件**

- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 仅新增本次查询记录；移除此条即可回滚。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345)；[outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/run_manifest.json)；[outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/train.log)。
- [outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/summaries/events.out.tfevents.1789486158.server](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/summaries/events.out.tfevents.1789486158.server) — 直接读取原始 TensorBoard 标量，未生成或改写训练产物。

**原因**

用户询问训练需要多久，根据真实完成 epoch 的间隔估算，预算保持 1000 epochs。

**验证**

使用 `EventAccumulator(..., size_guidance={"scalars": 0}).Reload()` 读取 `info/epochs`；最近完成 epoch 间隔均值 `300.19 s/epoch`。
剩余约 `82.5 h`，推算结束时间 `2026-09-19 10:51 +0800`；该时间取决于后续速度和资源占用，不能作为收敛承诺。

## 2026-09-15 21:55:00 +0800 — V1.5.4 PPO wiring pilot 通过

- activity_id: `ACT-20260915-215500-CMRESIDUAL-V154-PPO-SUPPORTED`
- timestamp: `2026-09-15 21:55:00 +0800`
- modification_version: `V1.5.4`
- operation_category: `code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户批准删除 `num_subscenes=1` override，并按原 GPU5/seed42/64 env × 2 updates 预算复跑。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（保留接手前 V1.5 dirty diff；未覆盖或暂存用户指导）
- scope: pilot wrapper 删除 `num_subscenes=1`，使用 canonical `num_subscenes=4`；buffer `8388608/5.0`、GPU5、seed42、64 env × 2 updates、horizon32、minibatch2048 和输入 SHA 冻结
- run_id: `cmresidual_ppo_pilot_v154_20260915_214302`
- run_status: `COMPLETED`
- last_step: `4096`
- last_epoch: `2`
- best_metric: `deterministic_reload_action_max_abs_diff=0.0`
- checkpoint: [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/rlgames/nn/last_CmResidualPilot_ep_2_rew__5.84_.pth](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/rlgames/nn/last_CmResidualPilot_ep_2_rew__5.84_.pth)
- output: [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/config.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/run_manifest.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/train.log)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/checkpoint_validation.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/initialization_smoke.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/initialization_smoke.json)
- exit_reason: 固定两次 update 完成，进程正常退出
- conclusion: `SUPPORTED`（PPO wiring smoke）；`INCONCLUSIVE`（residual 科研效果、收敛和抓取改善）

**文件**

- [src/task/CmResidual/tools/run_ppo_pilot.py](../../tools/run_ppo_pilot.py) — 本轮修改/记录。
- [src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md) — 本轮修改/记录。
- [src/task/CmResidual/docs/README.md](../README.md) — 本轮修改/记录。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 本轮修改/记录。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 本轮修改/记录。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 本轮修改/记录。

**原因**

- 删除 `num_subscenes=1` 后，1-env prepare smoke 成功；64 env pilot 完成 2 epochs。相对 V1.5.3 失败运行，resolved config 只改变顶层/PhysX `num_subscenes 1→4` 与两项已批准 buffer 值，输入 references 和 budget 不变。

**验证**

- 1-env initialization smoke：`COMPLETED`，无 SIGSEGV，日志见 [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/initialization_smoke.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/initialization_smoke.log)。
- 64 env × 2 updates：完成 episode，model/optimizer/normalizer/action 全部 finite；event scalar 完整。
- checkpoint 独立 CPU 双构造重载 deterministic mean action 最大差 `0.0`；SHA256 和 epoch/frame 见 [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/checkpoint_validation.json)。
- `metrics.jsonl` 两行 epoch 指标见 [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl)；最终 residual RMS `0.7315449`、saturation ratio `0.0034722`，仅为 wiring smoke 观测。
- Task pytest `12 passed`、py_compile、`git diff --check` 和 activity link audit 通过。

**保护边界与回滚**

- 未修改 reward、18D action、reference、checkpoint 初始值、资产、actor 属性、共享 base、外部 DExplore 或其他 Task；未启动长训或效果实验。
- 回滚只恢复 wrapper 的 `num_subscenes=1` 行及本轮版本/文档记录；保留 V1.5.3 失败运行和本次 ignored 输出。

## 2026-09-15 21:43:02 +0800 — V1.5.4 删除 num_subscenes=1 后 pilot 启动

- activity_id: `ACT-20260915-214302-CMRESIDUAL-V154-PPO`
- timestamp: `2026-09-15 21:43:02 +0800`
- modification_version: `V1.5.4`
- operation_category: `code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: GPU prepare_sim 二分诊断后，用户明确回复“可以”，批准删除 pilot wrapper 的 `num_subscenes=1` override，并按 GPU5/seed42/64 env × 2 updates 单次复跑。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（保留接手前 V1.5 dirty diff；未覆盖或暂存用户指导）
- scope: 仅删除 pilot wrapper 的 `num_subscenes=1` override，保留 buffer `8388608/5.0`；GPU5、seed42、64 env × 2 updates、horizon32、minibatch2048 和全部输入冻结
- run_id: `cmresidual_ppo_pilot_v154_20260915_214302`
- run_status: `STARTED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs /usr/bin/python3 src/task/CmResidual/tools/run_ppo_pilot.py --run-id cmresidual_ppo_pilot_v154_20260915_214302 --activity-id ACT-20260915-214302-CMRESIDUAL-V154-PPO --gpu 5 --seed 42`
- output: [outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302)（PENDING）；[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/run_manifest.json)（PENDING）；[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/config.json)（PENDING）；[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/metrics.jsonl)（PENDING）；[outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v154_20260915_214302/train.log)（PENDING）
- conclusion: `INCONCLUSIVE`（运行中）

**文件**

- [src/task/CmResidual/tools/run_ppo_pilot.py](../../tools/run_ppo_pilot.py) — 本轮修改/记录。
- [src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md) — 本轮修改/记录。
- [src/task/CmResidual/docs/README.md](../README.md) — 本轮修改/记录。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 本轮修改/记录。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 本轮修改/记录。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 本轮修改/记录。

**原因**

- V1.5.3 与同 buffer、同 GPU、同预算仍在 prepare_sim 崩溃；二分诊断显示 `num_subscenes=1` 为高可信触发条件。移除 wrapper override，使 canonical `num_subscenes=4` 生效；GPU PhysX 会自行强制单场景。

**验证**

- `py_compile`、Task pytest `12 passed`、`git diff --check` 通过。
- resolved config 比较预期只含顶层/PhysX 子字段 `num_subscenes 1→4` 与两项 buffer 差异；先执行 1-env prepare smoke，再执行固定预算 pilot。

**保护边界与回滚**

- 不修改 reward、18D action、reference、checkpoint、资产、actor 属性、共享 base、外部 DExplore 或其他 Task；失败即停止，不切卡、不切 CPU、不追加运行。

## 2026-09-15 21:33:09 +0800 — V1.5.4 GPU prepare_sim 二分诊断

- activity_id: `ACT-20260915-213309-CMRESIDUAL-V154-PREPARE-DIAGNOSTIC`
- timestamp: `2026-09-15 21:33:09 +0800`
- modification_version: `V1.5.3`
- operation_category: `diagnostic`、`operation`、`documentation`
- task_mode: `run-only/operation -> read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求开始定位 GPU PhysX 崩溃；本条只归档已完成的只读诊断，不修改代码、配置或研究变量。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: GPU5、seed42、buffer `8388608/5.0`、1 env、0 PPO update；hand/hand+table/hand+object/full actor 二分和 `num_subscenes=4` 对照
- run_id: `cmresidual_prepare_probe_v154_20260915_2120`
- run_status: `COMPLETED`
- output: [outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120)、[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/run_manifest.json)
- conclusion: `SUPPORTED`（`num_subscenes=1` 在 prepare_sim 崩溃、`num_subscenes=4` 越过 prepare_sim 的工程诊断证据）；`INCONCLUSIVE`（最终修复和科研效果）

**文件**

- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 记录本次诊断。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 记录证据和结论上限。

**原因**

- hand、hand+table、hand+object、full 四种 actor 组合均在 `Subscene 0 has 1 articulations` 后 SIGSEGV，资产组合不是必要条件。
- 同一输入将 `num_subscenes` 改为 `4` 后未再 SIGSEGV，进入 rl_games 的 batch/minibatch 校验；DExplore 原始入口默认 `subscenes=0`，当前 wrapper 强制 `num_subscenes=1`。

**验证**

- [outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand.log)、[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand_table.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand_table.log)、[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand_object.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/hand_object.log)、[outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/full.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/full.log)：四种 actor 二分均同一崩溃边界。
- [outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/subscenes4_probe.log](../../../../../outputs/CmResidual/cmresidual_prepare_probe_v154_20260915_2120/subscenes4_probe.log)：`num_subscenes=4` 越过 prepare_sim，随后因 1 env 的 batch size 32 不可被 minibatch 2048 整除而退出；这不是 PhysX 崩溃。
- 临时阶段/aggregate/asset 探针均已恢复；当前生产 task 文件只保留接手前 V1.5 变更。

**保护边界与回滚**

- 未修改训练代码、物理参数、actor 属性、reference、checkpoint、数据或共享基础设施；证据目录为 ignored 输出。
- 本条为诊断记录，删除本条和对应 ignored 输出即可回滚。

## 2026-09-15 21:09:57 +0800 — CmResidual V1.5.3 缩小 buffer 后仍发生原生段错误

- activity_id: `ACT-20260915-210957-CMRESIDUAL-V153-PPO-FAILED`
- timestamp: `2026-09-15 21:09:57 +0800`
- modification_version: `V1.5.3`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation -> read-only/diagnostic`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户要求阅读会话 `01a0a426-da25-7b93-bf6d-8eb4bd05c3c3`“然后继续”；该会话末尾具体方案为仅覆盖两项 buffer、GPU5/seed 42/64 env × 2 updates 复跑一次。先将范围写入 [src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md)，本次未扩展失败后的运行预算。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（接手时的 V1.5 dirty diff 和用户指导保留；本轮没有 stage 或 commit）
- scope: 仅 pilot wrapper 的 PhysX 容量与版本追溯、同卡同 seed 同预算单次复跑、六个明确文件的文档闭环
- run_id: `cmresidual_ppo_pilot_v153_20260915_210542`
- run_status: `FAILED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs /usr/bin/python3 src/task/CmResidual/tools/run_ppo_pilot.py --run-id cmresidual_ppo_pilot_v153_20260915_210542 --activity-id ACT-20260915-210542-CMRESIDUAL-V153-PPO --gpu 5 --seed 42`
- last_step: `null`
- last_epoch: `null`
- best_metric: `null`
- checkpoint: `null`（没有 checkpoint 或 TensorBoard event）
- exit_reason: 子进程于 `2026-09-15T21:05:54+08:00` 以 `SIGSEGV (-11)` 退出；wrapper 退出码为 `1`，日志止于 `GPU Pipeline: enabled`，无完成 PPO update 的证据
- output: [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542)；[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/config.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/run_manifest.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/metrics.jsonl)、[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/train.log)
- conclusion: `INVALID_IMPLEMENTATION`（Gate B 仍失败）；`REFUTED`（仅采用 `8388608 / 5.0` 足以修复本次启动的假设）；`INCONCLUSIVE`（精确根因和 residual 科研效果）

**文件**

- [src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md) — 记录 V1.5.3 两项容量覆盖、授权来源与单次复跑边界。
- [src/task/CmResidual/tools/run_ppo_pilot.py](../../tools/run_ppo_pilot.py) — 仅增加两条 pilot PhysX CLI override，并更新运行版本和默认 activity 标识。
- [src/task/CmResidual/docs/README.md](../README.md) — 更新当前失败状态和诊断边界。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 追加本次启动、终态、验证和产物导航。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 记录单变量范围复跑证据与假设结论。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 仅将 CmResidual 指针推进为 V1.5.3。

**原因**

- 旧 pilot 使用 `41943040 / 25.0`；本次只在独立 wrapper 使用 DExplore 小批量 export 的 `8388608 / 5.0`。
  同 GPU5、解释器、seed、预算和输入条件下依然失败，说明该容量缩小不足以修复；不能进一步推导为所有显存/PhysX 问题均被排除。
- resolved config 的递归比较只发现两项 buffer 差异，输入 path/size/mtime/SHA256 完全相同；子进程实际保存的
  [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/hydra/runs/CmResidualPilot_15-21-05-47/config.yaml](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/hydra/runs/CmResidualPilot_15-21-05-47/config.yaml) 也包含两项新值。对照证据：[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/config_comparison.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/config_comparison.json)。
- 原生日志没有足够的分阶段输出或 backtrace，不能据最后一行定位到某个 allocator、`create_sim`、asset/actor
  创建或 `prepare_sim` 调用，更不能归因为 residual 数学逻辑或 GPU5 硬件损坏。
- 只读源码对照发现 DExplore `base_task.py` 本身也在 `create_sim` 前创建 CUDA buffer，因此“提前初始化 CUDA”
  本身不足以构成差异归因。DExplore 使用 actor aggregate，而当前 CmResidual 没有；这仅是后续可检查的初始化
  差异，未经崩溃栈验证不认定为根因。下一步应先做有阶段输出与原生栈的初始化诊断，另行确认运行边界。

**验证**

- `python3 -m py_compile src/task/CmResidual/tools/run_ppo_pilot.py`：通过。
- `PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests`：`12 passed in 6.26s`。
- pilot `_resolved_config` 对旧 `config.json` 做递归比较：仅两项 buffer 不同；实际输入 SHA、GPU5、seed 42、
  64 env、2 epochs、horizon 32、minibatch 2048 不变。
- 终态 JSON、Hydra config 和所有实际产物可重载；`metrics.jsonl` 仅有失败状态行，不能当作训练曲线。
  checkpoint/event 数为 `0/0`，GPU5 回落至 `6 MiB / 0%`，`/proc` 检查无残留 pilot 子进程。
- 接手时其余 12 个 dirty/untracked 文件 SHA256 未变：[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/terminal_checks.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/terminal_checks.json)。
- `git diff --check` 与 latest activity 的六个显式路径 `audit_diff.py --worktree --check-links`：交接前执行并留存
  [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/validation.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/validation.log)。

**保护边界与回滚**

- 不修改 reward、18D action、mimic、reference/eligibility、DExplore/OI-Cm 权重、canonical 配置、共享 `src/base/`、
  外部 DExplore、其他 Task 或任何旧运行。失败后没有继续训练、切卡、切 CPU、调参或复跑。
- 回滚仅撤回本轮六文件增量；接手前原稿位于 [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/before](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/before)，逐文件增量见
  [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/changes.patch](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/continuation_evidence/changes.patch)。不能用整体 reset/revert 丢掉先前 V1.5 未提交成果。
- 规范反馈：无格式、目录、版本、日志或审批阻碍；按已确认计划的失败停止边界结束本次运行，不修改治理规则。

## 2026-09-15 21:05:42 +0800 — CmResidual V1.5.3 同卡缩小 PhysX buffer 复跑启动

- activity_id: `ACT-20260915-210542-CMRESIDUAL-V153-PPO`
- timestamp: `2026-09-15 21:05:42 +0800`
- modification_version: `V1.5.3`
- operation_category: `code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 指定会话末尾已交接两项 buffer 覆盖、GPU5/seed 42/64 env × 2 updates 单次复跑方案；用户要求阅读该会话“然后继续”，据此续接，并在 [src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md) 记录授权依据。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（保留接手时全部 V1.5 未提交实现与用户指导）
- scope: 仅 pilot wrapper 覆盖 buffer 为 `8388608 / 5.0` 和版本追溯；GPU5、seed 42、64 env × 2 updates、horizon 32、minibatch 2048 及其他配置冻结
- run_id: `cmresidual_ppo_pilot_v153_20260915_210542`
- run_status: `STARTED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs /usr/bin/python3 src/task/CmResidual/tools/run_ppo_pilot.py --run-id cmresidual_ppo_pilot_v153_20260915_210542 --activity-id ACT-20260915-210542-CMRESIDUAL-V153-PPO --gpu 5 --seed 42`
- output: [outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/config.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/run_manifest.json)、[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/metrics.jsonl)、[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542/train.log)（PENDING）；[outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v153_20260915_210542)（PENDING）
- conclusion: `INCONCLUSIVE`（等待单次复跑终态）

**文件**

- [src/task/CmResidual/docs/plan/V1.5.md](../plan/V1.5.md)
- [src/task/CmResidual/tools/run_ppo_pilot.py](../../tools/run_ppo_pilot.py)
- [src/task/CmResidual/docs/README.md](../README.md)
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md)
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md)
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)

**原因**

- V1.5.2 在任何 update 前出现 SIGSEGV。仅检验 DExplore 小批量 export 的 buffer 容量设置是否足以解决同卡初始化崩溃，不改变 solver、reward、reference、checkpoint 或 PPO 数学逻辑。

**验证**

- `python3 -m py_compile src/task/CmResidual/tools/run_ppo_pilot.py` 通过。
- `PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests`：`12 passed in 6.26s`。
- 通过 pilot `_resolved_config` 和旧 config 递归比较，仅两项 buffer 不同；`git diff --check` 通过。
- 启动前 GPU5 `6 MiB / 0%`，无残留 pilot 进程；仍使用前次的 `/usr/bin/python3`。

**回滚**

- 只撤回上述六个文件的本次增量，接手前原稿已单独备份；不回退或覆盖先前 V1.5 dirty diff，不修改用户指导、其他 Task、`src/base/`、数据或旧输出。失败即停止，不切卡、不追加运行。

## 2026-09-15 20:55:20 +0800 — CmResidual V1.5.2 PPO wiring pilot 在 epoch 0 前失败

- activity_id: `ACT-20260915-205520-CMRESIDUAL-V152-PPO-FAILED`
- timestamp: `2026-09-15 20:55:20 +0800`
- modification_version: `V1.5.2`
- operation_category: `diagnostic`、`experiment`、`operation`、`documentation`
- task_mode: `run-only/operation -> read-only/diagnostic`
- change_level: `L3`
- approval: `user-approved`（[V1.5 最终计划](../plan/V1.5.md) 的固定两次 update pilot；失败后按停止条件只读诊断，未复跑）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: [V1.5 最终计划](../plan/V1.5.md) 所列准入、gate/pilot 工具、Task config/test、文档和版本指针；physical GPU 5（进程内 `cuda:0`）、seed 42、64 env、2 PPO updates、horizon 32、minibatch 2048；失败后的源码、历史日志、GPU 状态和 Python/CUDA 环境只读诊断
- run_id: `cmresidual_ppo_pilot_v15_20260915_204915`
- run_status: `FAILED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/run_ppo_pilot.py --run-id cmresidual_ppo_pilot_v15_20260915_204915 --activity-id ACT-20260915-204915-CMRESIDUAL-V152-PPO-PILOT --gpu 5`
- last_step: `null`
- last_epoch: `null`
- best_metric: `null`
- checkpoint: `null`（训练更新前失败，未生成 checkpoint 或 TensorBoard event）
- exit_reason: 训练子进程在 GPU PhysX 创建阶段以 `SIGSEGV`（return code `-11`）退出
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/metrics.jsonl)、[train.log](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/train.log)、[Hydra config](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/hydra/runs/CmResidualPilot_15-20-50-00/config.yaml)
- conclusion: `INVALID_IMPLEMENTATION`（Gate B 未进入 PPO update）；`INCONCLUSIVE`（residual 学习、收敛和抓取效果）

**文件**

- `src/task/CmResidual/docs/指导/V1.5.md`
- `src/task/CmResidual/docs/plan/V1.5.md`
- `src/task/CmResidual/docs/README.md`
- `src/task/CmResidual/docs/logs/activity_log.md`
- `src/task/CmResidual/docs/logs/experiment_log.md`
- `src/task/CmResidual/tests/test_reference_contract.py`
- `src/task/CmResidual/tools/eval_zero_residual.py`
- `src/task/CmResidual/tools/eval_nonzero_residual.py`
- `src/task/CmResidual/tools/run_ppo_pilot.py`
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml`
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml`
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPilotPPO.yaml`
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py`
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py`
- `docs/current_versions.yaml`

**原因**

- 日志完成 Isaac Gym binding 和 gymtorch 加载，只输出到 `+++ Using GPU PhysX`、`Physics Device: cuda:0`、
  `GPU Pipeline: enabled`，随后进程段错误；没有 `Started to train`、epoch、checkpoint 或 TensorBoard event。
- 启动前及退出后 physical GPU 5 分别仅占约 `6 MiB`，全部 8 张卡均为 RTX 3090；系统 Python 与
  `graspenv` 均为 PyTorch `2.4.1+cu121`，因此没有证据支持普通显存占用、不同 Torch/CUDA 栈或 residual
  数值计算导致本次失败。
- 历史同任务证据显示旧配置下 GPU0 可完成 GPU PhysX，而 GPU1/GPU3 在 64 env 首次 rollout 触发 PhysX
  allocator error 700；当前失败同样发生在 epoch 0 前，但因为 Python stdout 缓冲且 core dump 被禁用，不能
  从本次日志确认同一原生调用栈。
- 当前配置直接给小规模 pilot 使用 `max_gpu_contact_pairs=41943040` 和
  `default_buffer_size_multiplier=25.0`。外部 DExplore 的受版本控制源码只在小批量 `export_rl` 路径把它们
  降为 `8388608` 和 `5.0`，并明确记录大缓冲可能耗尽 24-GB GPU、表现为异步 illegal access。该差异与本次
  崩溃阶段一致，是当前最高可信根因假设；未经同卡、同预算、仅缩小 PhysX buffer 的复跑，仍不能标记为已证实。

**验证**

- V1.5 实现完成后，`py_compile` 通过；Task-local pytest 为 `12 passed`；pilot Hydra compose 确认 64 env、
  2 epochs、horizon 32、minibatch 2048 和显式 `ppo_pilot` 准入；`git diff --check` 通过。
- 失败 manifest 终态字段及其 `config_snapshot`、`metrics`、`log` 入口存在性断言通过；GPU5 退出后没有残留训练
  进程或显存占用。系统禁止 core dump 且当前用户无 kernel journal 权限，因此没有原生 backtrace。

**停止边界与回滚**

- 按最终计划“任一 gate 失败即停止”，没有自动复跑、切换 GPU/CPU PhysX、增加预算或修改 reward、action、
  reference、checkpoint、外部 DExplore 和 `src/base/`。
- 如用户批准修订，最小下一步是在独立 pilot 配置中采用 DExplore 小批量值并保持 GPU5、seed 42、64 env ×
  2 updates 和其余变量不变；失败运行永久保留且不覆盖。代码回滚入口仍为 base commit
  `9167f8d5bc550b058d61d8904546ef67f2719770`。

## 2026-09-15 20:49:15 +0800 — CmResidual V1.5.2 PPO wiring pilot 启动

- activity_id: `ACT-20260915-204915-CMRESIDUAL-V152-PPO-PILOT`
- timestamp: `2026-09-15 20:49:15 +0800`
- modification_version: `V1.5.2`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`（[V1.5 最终计划](../plan/V1.5.md)；Gate A 已通过）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: physical GPU 5（进程内 `cuda:0`）、seed 42、64 env、2 PPO updates、horizon 32、minibatch 2048、从随机 residual policy 初始化；显式 `reference.allowIneligibleFor=ppo_pilot`；不含长训、超参搜索、效果对照或科学变量修改
- run_id: `cmresidual_ppo_pilot_v15_20260915_204915`
- run_status: `STARTED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/run_ppo_pilot.py --run-id cmresidual_ppo_pilot_v15_20260915_204915 --activity-id ACT-20260915-204915-CMRESIDUAL-V152-PPO-PILOT --gpu 5`
- output: [运行目录（PENDING）](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915)、[run_manifest.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/run_manifest.json)、[config.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/config.json)、[metrics.jsonl（PENDING）](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/metrics.jsonl)、[train.log（PENDING）](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/train.log)、[checkpoint validation（PENDING）](../../../../../outputs/CmResidual/cmresidual_ppo_pilot_v15_20260915_204915/checkpoint_validation.json)
- conclusion: `INCONCLUSIVE`（运行中）

**原因**

- Gate A 已满足进入最小 PPO wiring smoke 的全部前置条件；本次只执行用户批准的 2-update 上限。

**验证**

- 启动前 GPU 5 显存占用 `6 MiB`、利用率 0%；resolved pilot config 已通过 64 env/2 epochs/32 horizon/2048 minibatch 检查。

## 2026-09-15 20:49:14 +0800 — 修订后的 V1.5.1 非零 residual gate 通过

- activity_id: `ACT-20260915-204914-CMRESIDUAL-V151-NONZERO-SUPPORTED`
- timestamp: `2026-09-15 20:49:14 +0800`
- modification_version: `V1.5.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`（[V1.5 最终计划](../plan/V1.5.md) 的修订 gate）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: CPU PhysX、seed 42、4 env × 32 steps、zero/zero/+0.25/-0.25 residual；PPO 尚未启动
- run_id: `cmresidual_nonzero_v15_rerun_20260915_204817`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_nonzero_residual.py --run-id cmresidual_nonzero_v15_rerun_20260915_204817 --activity-id ACT-20260915-204817-CMRESIDUAL-V151-NONZERO-RERUN`
- last_step: `32`
- last_epoch: `null`
- best_metric: `final_signed_response_separation=0.3832877874`（工程响应量）
- checkpoint: `null`
- exit_reason: 达到固定 32 步预算且全部 Gate A 条件通过，进程正常退出
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/eval.log)
- conclusion: `SUPPORTED`（非零 residual physical target 工程合同）；`INCONCLUSIVE`（PPO 与科研效果）

**原因**

- runtime mimic、requested delta、ignored mimic delta、zero target delta、saturation、joint limit 与 indexed reset isolation 的最大误差均为 0；非零 applied delta 约为 `0.05`，正负 residual 的末步 DOF 响应分离为 `0.383288`。

**验证**

- 32 步 observation/reward/target/object pose 全部 finite，reset count=0，无 simulator error；独立 PhysX trajectory 差异只作诊断，未用于 gate。

## 2026-09-15 20:48:17 +0800 — 修订判据后的 V1.5.1 非零 residual gate 启动

- activity_id: `ACT-20260915-204817-CMRESIDUAL-V151-NONZERO-RERUN`
- timestamp: `2026-09-15 20:48:17 +0800`
- modification_version: `V1.5.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`（用户在首次 gate 证据交接后明确回复“确定”，批准 [V1.5 最终计划](../plan/V1.5.md) 中的修订判据与同预算复跑）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: CPU PhysX、seed 42、4 env × 32 steps、固定 zero/zero/+0.25/-0.25 residual；runtime mimic contract 与 indexed-reset isolation；不含 PPO、reward/scale/schema/reference/checkpoint 修改
- run_id: `cmresidual_nonzero_v15_rerun_20260915_204817`
- run_status: `STARTED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_nonzero_residual.py --run-id cmresidual_nonzero_v15_rerun_20260915_204817 --activity-id ACT-20260915-204817-CMRESIDUAL-V151-NONZERO-RERUN`
- output: [运行目录（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817)、[run_manifest.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/run_manifest.json)、[config.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/config.json)、[metrics.jsonl（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/metrics.jsonl)、[eval.log（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_rerun_20260915_204817/eval.log)
- conclusion: `INCONCLUSIVE`（运行中；PPO 尚未启动）

**原因**

- 首次 gate 证明 residual 控制路径局部成立，但验收脚本使用了错误的 mimic 常量和无效的跨 PhysX trajectory 等值判据；本次只修正验收实现，不改变 action、reward、reference 或预算。

**验证**

- 修订后 `py_compile`、Task-local `pytest`（`12 passed`）和 `git diff --check` 通过。

## 2026-09-15 20:38:45 +0800 — CmResidual V1.5.1 首次非零 residual gate 未通过并停止

- activity_id: `ACT-20260915-203845-CMRESIDUAL-V151-NONZERO-END`
- timestamp: `2026-09-15 20:38:45 +0800`
- modification_version: `V1.5.1`
- type: `architecture`、`code`、`experiment`、`operation`、`documentation`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`（[V1.5 指导](../指导/V1.5.md) 与 [V1.5 最终计划](../plan/V1.5.md)）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（V1.5 已批准实现和记录尚未提交；用户既有 V1.2/V1.3/V1.4 指导未修改）
- scope: V1.5 fail-closed reference override、Task config/test、Gate A/PPO pilot 工具和本次 4 env × 32 steps Gate A；没有运行 PPO，没有修改 reward、scale、18D schema、reference/eligibility、checkpoint、外部 DExplore、其他 Task 或 `src/base/`
- run_id: `cmresidual_nonzero_v15_20260915_203620`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_nonzero_residual.py --run-id cmresidual_nonzero_v15_20260915_203620 --activity-id ACT-20260915-203620-CMRESIDUAL-V151-NONZERO`
- last_step: `32`
- last_epoch: `null`
- best_metric: `final_signed_response_separation=0.3832877874`（工程响应量，不是效果指标）
- checkpoint: `null`
- exit_reason: 完成 32 步预算后，gate 因两个验收实现问题返回非零；按停止条件未启动 PPO
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/eval.log)
- conclusion: `SUPPORTED`（requested delta、零 residual 旁路、joint limits、finite 与非零响应的局部合同）；`INVALID_IMPLEMENTATION`（本次完整 Gate A 验收工具）；`INCONCLUSIVE`（PPO 与科研效果）

**文件**

- [V1.5 指导](../指导/V1.5.md)、[V1.5 最终计划](../plan/V1.5.md)、[Task config 目录](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/)、[pilot config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPilotPPO.yaml)、[CmResidual 实现目录](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/)、[Task tests](../../tests/)、[Task tools](../../tools/)。

**原因**

- 实测 requested delta 误差、ignored mimic requested delta、零 residual target delta、saturation 和 joint-limit error 均为 0；非零 applied delta 为约 `0.05`，正负环境末步状态分离为 `0.3833`，说明 residual 确实进入 physical target。
- gate 脚本把 ring mimic scale 写成 corrected-reference manifest 的 `1.18`，而当前已批准 DExplore runtime 合同为 `1.05`，形成最高 `0.18056` 的伪 mismatch。
- gate 又要求两个独立 PhysX 环境的完整接触后 trajectory 逐位相同；首步 DOF state 差仅 `4.29e-6`，随后接触动力学分叉，完整 state/velocity 比较增至 `18.32`。这不等同于 residual 跨 env 污染；直接隔离证据 `zero_target_delta_max=0` 已通过，但修改该验收口径需要重新确认最终计划。

**验证**

- 运行前 `py_compile`、Task-local `pytest`（`12 passed`）、pilot Hydra compose 和 `git diff --check` 通过。
- Gate A 的 32 步 observation/reward/target/object pose 全部 finite、reset count=0；PPO pilot 未启动，未生成或修改 checkpoint。

## 2026-09-15 20:36:20 +0800 — CmResidual V1.5.1 固定非零 residual gate 启动

- activity_id: `ACT-20260915-203620-CMRESIDUAL-V151-NONZERO`
- timestamp: `2026-09-15 20:36:20 +0800`
- modification_version: `V1.5.1`
- type: `architecture`、`code`、`experiment`、`operation`、`documentation`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户确认将本轮要求整理为 [V1.5 指导](../指导/V1.5.md)，并同意 [V1.5 最终计划](../plan/V1.5.md) 所列“固定非零 gate 通过后再运行 64 env × 2 updates PPO pilot”的范围。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（V1.5 已批准实现和记录尚未提交；用户已有 V1.2/V1.3/V1.4 指导保持未跟踪且未修改）
- scope: [V1.5 最终计划](../plan/V1.5.md) 的 fail-closed reference override、Task config/test、非零 gate 和 PPO pilot 工具；Gate A 固定 CPU PhysX、seed 42、4 env × 32 steps，不修改 reward、scale、18D schema、reference/eligibility、checkpoint 或外部 DExplore
- run_id: `cmresidual_nonzero_v15_20260915_203620`
- run_status: `STARTED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_nonzero_residual.py --run-id cmresidual_nonzero_v15_20260915_203620 --activity-id ACT-20260915-203620-CMRESIDUAL-V151-NONZERO`
- output: [运行目录（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620)、[run_manifest.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/run_manifest.json)、[config.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/config.json)、[metrics.jsonl（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/metrics.jsonl)、[eval.log（PENDING）](../../../../../outputs/CmResidual/cmresidual_nonzero_v15_20260915_203620/eval.log)
- conclusion: `INCONCLUSIVE`（运行中；PPO 尚未启动）

**原因**

- zero-residual baseline 已通过；本 gate 验证有权 residual 的 requested/applied physical delta、mimic/limit、零对照和跨环境隔离，避免把 PPO 首次运行同时当作控制实现测试。

**验证**

- 运行前 `py_compile`、Task-local `pytest`（`12 passed`）、pilot Hydra compose 和 `git diff --check` 均通过。

## 2026-09-15 20:16:01 +0800 — 非零 residual gate 与 PPO pilot 前置边界诊断

- activity_id: `ACT-20260915-201601-CMRESIDUAL-V142-PREP-DIAGNOSTIC`
- timestamp: `2026-09-15 20:16:01 +0800`
- modification_version: `V1.4.2`
- type: `diagnostic`、`documentation`
- operation_category: `diagnostic`、`documentation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`（用户要求先做非零 residual 工程门禁，再进入小规模 PPO pilot；本条只核对现有合同，不修改实现或启动运行）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`（仅有用户提供的三个未跟踪指导文件；本次诊断开始时 tracked worktree 干净）
- scope: [V1.4 最终计划](../plan/V1.4.md)、[CmResidual task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[residual mapping](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py)、[PPO config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml) 与 corrected-reference manifest；不改代码、配置、reference/eligibility、reward、action schema 或 checkpoint，不启动仿真和 PPO
- conclusion: `SUPPORTED`（下述现有代码/配置事实）；`INCONCLUSIVE`（非零 residual gate 与 PPO pilot 尚未运行）

**文件**

- [活动记录](activity_log.md) — 只新增本次前置诊断；未创建或修改 plan/指导、实现和运行产物。

**原因**

- V1.4 最终计划明确排除 PPO；新方向需要新的指导/plan 边界，不能静默复用 V1.4.2 运行。
- 现有 18D residual action 中只有 wrist 6D 与独立 finger indices `[6,8,10,12,14,15]` 进入 requested delta；其余 6 个 mimic action 输出没有直接 target 权限，但仍进入 action penalty。这是既有 V1.3 合同，若要改为 12D 会改变 action/checkpoint schema。
- `ReferenceProvider` 读取 corrected manifest 的 `training_eligible=false`，但 task/train 入口没有 fail-closed 检查；直接启动 PPO 会用 legacy DExplore source 构造训练 observation/reset，同时只把 corrected artifact 当作审计 metadata。

**验证**

- 只读核对 `action_mapping.py`、`task.py`、`CmResidualPPO.yaml`、V1.0/V1.3/V1.4 plan 和 corrected `manifest.json`；当前 manifest 的 `training_eligible=false`、PPO 默认 `max_epochs=1000`、`horizon_length=32`、`minibatch_size=2048`。
- 本次没有运行 smoke、仿真或训练，不产生工程或科研效果结论。用户提供的指导文件、既有 outputs、数据和 checkpoint 均未修改。

## 2026-09-15 19:19:53 +0800 — CmResidual V1.4.2 上游 parity 修正与最终零残差门禁完成

- activity_id: `ACT-20260915-191707-CMRESIDUAL-V142-ZERO-FINAL`
- timestamp: `2026-09-15 19:19:53 +0800`
- modification_version: `V1.4.2`
- type: `architecture`、`code`、`diagnostic`、`experiment`、`operation`、`documentation`
- operation_category: `architecture`、`code`、`diagnostic`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`（[V1.4 最终计划](../plan/V1.4.md) 的正式 gate）
- branch: `oyx`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- worktree_dirty: `true`（本次 V1.4 diff 尚未提交；用户提供的三个指导文件保持未跟踪且未暂存）
- scope: [V1.4 最终计划](../plan/V1.4.md) 所列 CmResidual actor/reference/observation/action/task/config/test/eval、Task 文档与版本指针；不含 corrected reference、reward 数值、OI-Cm/residual 权限、PPO、外部 DExplore、其他 Task 或 `src/base/`
- run_id: `cmresidual_zero_v14_final_20260915_191707`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_zero_residual.py --run-id cmresidual_zero_v14_final_20260915_191707 --steps 367 --num-envs 4 --activity-id ACT-20260915-191707-CMRESIDUAL-V142-ZERO-FINAL`
- last_step: `367`
- last_epoch: `null`
- best_metric: `max_lift_m=0.2123541832`（gate 行为观测量；不是独立抓取 benchmark）
- checkpoint: `null`
- exit_reason: 达到 4 env × 367 control-step 预算，所有 gate 条件通过，并按独立进程退出正常回收 simulator
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_final_20260915_191707/eval.log)
- conclusion: `SUPPORTED`（DExplore 上游 parity、reset、zero-target、finite、近物/接触行为和运行终态）；`INCONCLUSIVE`（Cm/PPO 增益与科研效果）

**原因**

- 根因是 provider 把 DExplore 原语句 `body_pos[..., 6:]` 的空切片误解释为 body 维 remainder，令后 10 个关键点坐标被模到 `[0,2π)`，产生最高 `188.4956 m/s` 的伪 reference 速度并破坏 object-surface IG。

**实际修改**

- frozen actor 改为发布 checkpoint 的 ReLU MLP，并复现 float32 RMS normalization 与 `[-5,5]` clamp；721D builder 逐字段复现 DExplore body/object/IG，两个 offset 仍为 `+1/+16`。
- legacy source 与 corrected reference 分离：前者按固定 SHA 和 `[44,410]` 重建 teacher/reset；后者只作 eligibility/audit，仍为 `training_eligible=false`。
- hand/object/table reset、DExplore base target/mimic、严格 zero-residual clamp 旁路，以及 `substeps=4`、PhysX/plane/asset/shape-filter 参数已与发布源码对齐；OI-Cm、reward 和 residual 权限未改。
- 修改路径：[版本指针](../../../../../docs/current_versions.yaml)、[Task README](../README.md)、[V1.4 最终计划](../plan/V1.4.md)、[实验记录](experiment_log.md)、[Task tests](../../tests/)、[评估工具](../../tools/eval_zero_residual.py)、[CmResidual 实现目录](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/)、[canonical task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml)、[online task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml)。

**验证**

- `python3 -m py_compile ...` 通过；`PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests` 为 `11 passed`；`git diff --check` 通过。
- 注册项目 OmegaConf resolver 后，`CmResidual` 与 `CmResidualOnline` 的 task/train Hydra compose 均通过，并确认 `numObservations=2005`、`numActions=18`、`substeps=4`、source frame `[44,410]`。
- source-level golden test 在同一输入上直接调用未经修改的 DExplore 函数，721D 输出最大误差为 0；对真实 DExplore 首帧 state/reference 的 1442D 重建最大误差为 `2.10e-5`（CPU/GPU 浮点差）。
- 正式 gate 初始 q/wrist/object/table 最大位置误差为 `2.09e-7`，367 步无 reset；observation/target/object pose 全程 finite，zero residual target delta 与 target saturation 始终为 0。
- 第 16 步平均 tip distance=`0.045808 m`、实测 contact occupancy=`0.65`；全程平均/最大 contact occupancy=`0.480790/0.75`，最小 tip distance=`0.037769 m`。`success_fraction` 从第 34 步出现，最大为 1.0、末步为 0.75；success-triggered reset 在 gate 中关闭。

**保护边界与回滚**

- 没有启动 PPO，没有修改或覆盖 DExplore/OI-Cm checkpoint、corrected reference、cache、旧 outputs、外部 DExplore checkout、用户指导、其他 Task 或共享 runtime。
- 回滚入口为 base commit `b1dac1c7955c9b960481279425f9e4e2e2269a98`、[V1.4 最终计划](../plan/V1.4.md) 中的显式文件和本条独立运行目录；旧失败运行保留为审计证据。

## 2026-09-15 19:16:58 +0800 — legacy body reference 修正后的单环境诊断通过

- activity_id: `ACT-20260915-191608-CMRESIDUAL-V142-REFERENCE-SMOKE`
- timestamp: `2026-09-15 19:16:58 +0800`
- modification_version: `V1.4.2`
- operation_category: `diagnostic`、`operation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`（V1.4 上游 DExplore parity 修正范围）
- branch: `oyx`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- run_id: `cmresidual_zero_v14_reference_smoke_20260915_191608`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_zero_residual.py --run-id cmresidual_zero_v14_reference_smoke_20260915_191608 --steps 20 --num-envs 1 --activity-id ACT-20260915-191608-CMRESIDUAL-V142-REFERENCE-SMOKE`
- last_step: `20`
- last_epoch: `null`
- best_metric: `max_contact_occupancy=0.8`（工程行为门指标，不是抓取成绩）
- checkpoint: `null`
- exit_reason: 达到 20 步诊断预算且 source 接触阶段行为门通过，正常退出
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_zero_v14_reference_smoke_20260915_191608)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_reference_smoke_20260915_191608/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_reference_smoke_20260915_191608/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_reference_smoke_20260915_191608/eval.log)
- conclusion: `SUPPORTED`（legacy reference、近物/接触、finite、zero-target parity 和正常退出）；`INCONCLUSIVE`（完整 367 步 gate、PPO 与科研效果）

**关键证据**

- 第 16 步 `reference_contact_occupancy=0.4`，实测 `contact_occupancy=0.6`，平均 tip distance=`0.045783 m`；20 步内最大 contact occupancy=`0.8`。
- 对未经修改的 DExplore 单环境首帧 state/ref 重建后，reference 最大差从 `188.4956` 降到 `3.22e-5`，完整 1442D observation 最大差为 `2.10e-5`（CPU/GPU 浮点差）。
- observation/target/object pose 均 finite，reset_count=0，zero residual target delta=0；未启动 PPO。

## 2026-09-15 19:05:16 +0800 — DExplore physics parity 单环境诊断未通过

- activity_id: `ACT-20260915-190424-CMRESIDUAL-V142-PHYSICS-SMOKE`
- timestamp: `2026-09-15 19:05:16 +0800`
- modification_version: `V1.4.2`
- operation_category: `diagnostic`、`operation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`（V1.4 上游 DExplore parity 修正范围）
- branch: `oyx`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- run_id: `cmresidual_zero_v14_physics_smoke_20260915_190424`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_zero_residual.py --run-id cmresidual_zero_v14_physics_smoke_20260915_190424 --steps 20 --num-envs 1 --activity-id ACT-20260915-190424-CMRESIDUAL-V142-PHYSICS-SMOKE`
- last_step: `20`
- last_epoch: `null`
- best_metric: `min_tip_distance_m=0.1342651`
- checkpoint: `null`
- exit_reason: 达到 20 步诊断预算并正常退出；第 16 个 source 接触帧仍未近物或接触
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_zero_v14_physics_smoke_20260915_190424)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_physics_smoke_20260915_190424/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_physics_smoke_20260915_190424/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_physics_smoke_20260915_190424/eval.log)
- conclusion: `INVALID_IMPLEMENTATION`（physics parity 本身未消除行为偏离）；`INCONCLUSIVE`（科研效果）

**诊断证据**

- 同步 simulator 后第 16 步 `reference_contact_occupancy=0.4`，但 `contact=0`、平均 tip distance=`0.473640 m`。
- 将未经修改的 DExplore 单环境首帧 state/ref 保存到临时诊断目录，再用本地 provider+builder 重建；最大 observation 差为 `188.4955`，定位到 provider 对原源码空切片 `body_pos[..., 6:]` 的错误解释，而不是物理参数。

## 2026-09-15 18:52:36 +0800 — CmResidual V1.4.2 首次正式零残差门禁未通过

- activity_id: `ACT-20260915-184916-CMRESIDUAL-V142-ZERO`
- timestamp: `2026-09-15 18:52:36 +0800`
- modification_version: `V1.4.2`
- type: `experiment`、`operation`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L3`（DExplore legacy reference/reset 和 4 env × 367 步正式门禁）
- approval: `user-approved`（[V1.4 最终计划](../plan/V1.4.md) 已明确列出本 gate，用户回复“可以，你直接开始修改”）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- worktree_dirty: `true`（V1.4 已批准实现尚未提交；用户指导保持未跟踪且不暂存）
- scope: CPU PhysX、seed 42、4 env × 367 steps、严格全零 residual；不含 PPO、参数更新、reward/参考/schema 变更
- run_id: `cmresidual_zero_v14_20260915_184916`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_zero_residual.py --run-id cmresidual_zero_v14_20260915_184916 --steps 367 --num-envs 4 --activity-id ACT-20260915-184916-CMRESIDUAL-V142-ZERO`
- last_step: `367`
- last_epoch: `null`
- best_metric: `max_lift_m=0.0004970878`（没有接触，不作为抓取成绩）
- checkpoint: `null`
- exit_reason: 达到 367 步预算并通过独立进程退出正常回收；行为 gate 在 source 首次接触帧未达到近物/接触条件
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_20260915_184916/eval.log)
- conclusion: `SUPPORTED`（367 步、finite、reset 对齐、zero-target parity 和正常退出）；`INVALID_IMPLEMENTATION`（source 接触阶段行为 gate）；`INCONCLUSIVE`（PPO 和科研效果）

**结果、后续诊断与停止边界**

- 初始 q/wrist/object/table 位置误差最大为 `1.31e-7`；367 步无 reset，observation/target/object pose finite，zero residual target delta 和 saturation ratio 均为 0，进程正常退出。
- 第 16 个 source 接触帧的 reference contact occupancy 为 `0.4`，但实测 contact 为 0、平均 tip distance 为 `0.444138 m`；全程最大实测 contact 仍为 0。因此行为 gate 为 `false`。
- 外部源码复核发现尚未同步 `substeps=4`、PhysX 接触参数、friction/restitution、asset angular velocity 和发布 shape-filter 行为；同时首帧离线 FK velocity 替代并非原路径。已回到 `change` 模式修正这些上游 parity 项，并保持不启动 PPO。

## 2026-09-15 18:47:29 +0800 — CmResidual V1.4 单环境 smoke 完成

- activity_id: `ACT-20260915-184653-CMRESIDUAL-V141-SMOKE`
- timestamp: `2026-09-15 18:47:29 +0800`
- modification_version: `V1.4.2`
- type: `architecture`、`code`、`diagnostic`、`operation`、`documentation`
- operation_category: `architecture`、`code`、`diagnostic`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`（DExplore checkpoint/observation、legacy reference/reset 与 zero-residual gate 合同）
- approval: `user-approved`（用户在 V1.4 诊断和范围交接后明确回复“可以，你直接开始修改”）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `b1dac1c7955c9b960481279425f9e4e2e2269a98`
- worktree_dirty: `true`（仅修改 [V1.4 最终计划](../plan/V1.4.md) 所列范围；保留且不暂存用户提供的 `指导/V1.2.md`、`指导/V1.3.md`、`指导/V1.4.md`）
- scope: [V1.4 最终计划](../plan/V1.4.md) 所列 CmResidual task/config/test/eval 与 Task 文档；不含 corrected reference、reward 数值、OI-Cm/residual 权限、PPO、外部 DExplore、其他 Task 或共享 `src/base/`
- run_id: `cmresidual_zero_v14_smoke_20260915_184653`
- run_status: `COMPLETED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_zero_residual.py --run-id cmresidual_zero_v14_smoke_20260915_184653 --steps 4 --num-envs 1 --activity-id ACT-20260915-184653-CMRESIDUAL-V141-SMOKE`
- last_step: `4`
- last_epoch: `null`
- best_metric: `max_lift_m=0.0004968047`（仅为 smoke 观测量，不是抓取成绩）
- checkpoint: `null`
- exit_reason: 达到 4 步 smoke 预算；按计划使用独立进程退出回收 simulator，没有调用已知会阻塞的显式 `destroy_sim`
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_zero_v14_smoke_20260915_184653)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_smoke_20260915_184653/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_zero_v14_smoke_20260915_184653/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v14_smoke_20260915_184653/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v14_smoke_20260915_184653/eval.log)
- conclusion: `SUPPORTED`（单环境初始化、finite、zero-target parity 和进程终态）；`INCONCLUSIVE`（接触、完整门禁、PPO 与科研效果）

**结果与保护边界**

- 初始 native q、wrist、object、table 位置误差均为 0；4 步均无 reset，最小平均 tip distance 为 `0.127061 m`，observation/target finite，zero residual target delta 与 saturation ratio 均为 0。
- `python3 -m py_compile ...` 通过；`PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests` 为 `11 passed`；源码级 DExplore 721D golden parity 最大绝对误差为 0。
- corrected reference 保持 `training_eligible=false`；没有启动 PPO，没有修改或覆盖 checkpoint、reference、旧运行、外部 DExplore、共享 runtime 或用户指导。

## 2026-09-15 16:53:16 +0800 — CmResidual V1.3 zero-residual gate 失败并停止

- activity_id: `ACT-20260915-165316-CMRESIDUAL-V13-ZERO-END`
- timestamp: `2026-09-15 16:53:16 +0800`
- modification_version: `V1.3.2`
- type: `architecture`、`code`、`experiment`、`operation`、`documentation`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`
- approval: `user-approved`（用户在 V1.3 计划交接后明确回复“你直接开始执行吧”）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `95d5aea25a8d75306982d4266d1040911fd25efa`
- worktree_dirty: `true`（只保留并隔离用户提供且未跟踪的 `指导/V1.2.md`、`指导/V1.3.md`）
- scope: V1.3 最终计划列出的 CmResidual task/config/network、Task-local test/eval 工具、Task 文档与版本指针；不含 checkpoint、reference/schema/reward、用户指导、其他 Task、共享 `src/base/` 或 PPO 训练
- run_id: `cmresidual_zero_v13_20260915_164405`
- run_status: `FAILED`
- last_step: `367`
- last_epoch: `null`
- best_metric: `max_lift_m=0.0885452628`（无接触条件下的观测量，不作为抓取成绩）
- checkpoint: `null`
- exit_reason: 367 步 metrics 完整写入后，Isaac Gym `destroy_sim` 超过 7 分钟未返回；进程以 `SIGTERM` 停止。rollout 同时出现全程 contact occupancy=0 与 mean tip distance=1.421636 m，行为 gate 无效。
- conclusion: `SUPPORTED`（zero-residual target parity 与 finite 工程不变量）；`INVALID_IMPLEMENTATION`（完整 zero-residual gate）；`INCONCLUSIVE`（PPO、抓取效果与科研假设）

**原因**

- 用户指导 V1.3 要求先修复真实输入、contact 观测和 residual 权限，再以严格零残差验证 frozen DExplore 行为；当前 reference 不具备训练资格，因此 gate 是本次批准范围的终点。

**实际修改范围**

- 计划与记录：[最终计划 V1.3](../plan/V1.3.md)、[Task README](../README.md)、[活动记录](activity_log.md)、[实验记录](experiment_log.md)、[版本指针](../../../../../docs/current_versions.yaml)。
- 配置：[CmResidual.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml)、[CmResidualOnline.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml)、[CmResidualPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml)、[CmResidualOnlinePPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualOnlinePPO.yaml)。
- 实现：[task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[action_mapping.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py)、[dexplore_observation.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/dexplore_observation.py)、[reference_provider.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py)、[cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py)。
- 验证工具与测试：[eval_zero_residual.py](../../tools/eval_zero_residual.py)、[test_reference_contract.py](../../tests/test_reference_contract.py)。

**验证**

- canonical/online 两个入口已对齐 2005-D observation、32-D/16-slot OI-Cm、真实输入路径和 SHA256；Task 是唯一 OI-Cm owner，并验证 checkpoint 元数据与 scale manifest。
- DExplore contact 已改取 net contact force；residual 在 base target 转换后按 0.015 m / 0.20 rad / 0.08 rad 施加，mimic 与 joint-limit clamp 保持。
- `python3 -m py_compile ...`：通过；`PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests`：`8 passed`；两个 task/train alias 的 Hydra compose：通过；`git diff --check`：通过。
- 运行产物：[运行目录](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/eval.log)。

**停止、保护边界与回滚**

- 按 [最终计划 V1.3](../plan/V1.3.md) 的失败停止条件，没有启动 PPO，也没有自动修改 reference/schema/reward/初始化语义。reference manifest 仍为 `training_eligible=false`。
- 未修改 DExplore/OI-Cm checkpoint、reference、cache、历史 outputs、用户指导、其他 Task 或共享 `src/base/`；失败运行保留为只读审计证据。
- 回滚入口为 base commit `95d5aea25a8d75306982d4266d1040911fd25efa`、本条列出的显式修改文件与独立运行目录；不删除或覆盖既有产物。

## 2026-09-15 16:44:05 +0800 — CmResidual V1.3 修正完成并启动零残差评估

- activity_id: `ACT-20260915-164405-CMRESIDUAL-V13-ZERO-START`
- timestamp: `2026-09-15 16:44:05 +0800`
- modification_version: `V1.3.2`
- type: `architecture`、`code`、`experiment`、`operation`、`documentation`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L3`（contact 观测与 physical residual 合同、真实 checkpoint/reference 校验和 Isaac Gym 评估）
- approval: `user-approved`（用户在 V1.3 计划交接后明确回复“你直接开始执行吧”）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `95d5aea25a8d75306982d4266d1040911fd25efa`
- worktree_dirty: `true`（保留用户提供且未跟踪的 `指导/V1.2.md`、`指导/V1.3.md`，不修改、不暂存）
- scope: [最终计划 V1.3](../plan/V1.3.md)、CmResidual task/config/network、Task-local tests 与 zero-residual 工具；不含 reference/schema/reward、checkpoint、其他 Task、共享 `src/base/` 或 PPO 训练
- run_id: `cmresidual_zero_v13_20260915_164405`
- run_status: `STARTED`
- command: `PYTHONPATH=third_party/IsaacGymEnvs python3 src/task/CmResidual/tools/eval_zero_residual.py --run-id cmresidual_zero_v13_20260915_164405`
- output: [运行目录（PENDING）](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405)、[run_manifest.json（PENDING）](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/run_manifest.json)、[metrics.jsonl（PENDING）](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/metrics.jsonl)、[eval.log（PENDING）](../../../../../outputs/CmResidual/cmresidual_zero_v13_20260915_164405/eval.log)
- conclusion: `INCONCLUSIVE`（评估运行中；工程 smoke 不作为 PPO 或抓取效果结论）

**已完成的实现与启动前验证**

- 两个注册配置入口已锁定 32-D OI-Cm、真实输入及其 SHA256；Task 成为唯一 OI-Cm checkpoint owner，并 fail closed 校验 checkpoint 元数据与 scale manifest。
- DExplore contact 改取 Isaac Gym net contact force；18-D residual 改为 DExplore target 转换后的 0.015 m / 0.20 rad / 0.08 rad 有界物理修正，zero residual 与原 base-only 映射逐元素相同。
- `python3 -m py_compile ...` 通过；`PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests` 为 `8 passed`；canonical 与 online alias 的 Hydra compose 均得到 2005-D observation。
- 当前 reference manifest 的 `training_eligible=false` 未被修改；本次只运行 zero-residual gate，不启动 PPO。

**保护边界与回滚**

- 不修改或覆盖 DExplore/OI-Cm checkpoint、reference、cache、历史 outputs、用户指导和其他 dirty diff。
- 代码回滚入口为 base commit `95d5aea25a8d75306982d4266d1040911fd25efa` 与 [最终计划 V1.3](../plan/V1.3.md) 中列出的显式文件；运行产物位于独立新目录。

## 2026-09-15 11:43:48 +0800 — 按指导 V1.1 修订 reference 与 OI-Cm 链路

- activity_id: `ACT-20260915-114348-CMRESIDUAL-V11-REVISE`
- timestamp: `2026-09-15 11:43:48 +0800`
- modification_version: `V1.1.4`
- operation_category: `architecture`、`code`、`documentation`
- task_mode: `change`
- change_level: `L3`（DExplore 观测语义、外部 checkpoint 接入和 actor/critic 输入合同）
- approval: `user-approved`（用户要求按照 `指导/V1.1.md` 修订当前代码）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `13b57f6`
- worktree_dirty: `true`（保留其他既有差异；未覆盖用户修改）
- scope: `src/task/CmResidual/docs/指导/V1.1.md`、`src/task/CmResidual/docs/plan/V1.1.md`、`third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/`、`third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml`、`third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml`、`third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py`、`src/task/CmResidual/docs/README.md`、`docs/current_versions.yaml`、`docs/项目总览.md`
- run_id: `smoke_v11_20260915c`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（工程 smoke；不代表直接抓取或科研效果成立）

**原因**

- 指导 V1.1 指出旧实现重复当前观测、使用通用 MLP 伪装 Cm，无法验证 reference-conditioned DExplore 与 OI-Cm 的作用，因此先修正输入和表征合同。

**修改**

- [指导 V1.1](../指导/V1.1.md) 与 [计划 V1.1](../plan/V1.1.md) 作为本次设计与执行入口；[Task README](../README.md)、[版本指针](../../../../../docs/current_versions.yaml) 和 [项目总览](../../../../../docs/项目总览.md) 已同步。
- [smoke 日志](../../../../../outputs/CmResidual/smoke_v11_20260915c/train.log) 记录真实 reference 与 OI-Cm 的验证输出。
- 新增 `ReferenceProvider`，校验 world/xyzw reference manifest，并按 `+1/+16` 返回 q、link pose、object twist 和 phase。
- DExplore 721-D offset 使用 reference pose/object state 的差分字段；两个 offset 不再复制当前状态。
- PPO network 加载 ObjectInteractionCm V1.3 `best.pt`，从同步观测构造 hand/object point-flow 输入，冻结 OI-Cm，pooled Cm token 同时输入 actor/critic；不再使用 `MLP(obs)` 作为 Cm。
- V1.1 plan 改为与 `指导/V1.1.md` 配对，并明确首个有效实验冻结 OI-Cm。

**验证**

- `python3 -m py_compile third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/*.py third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_*.py`：通过。
- 真实 reference + OI-Cm checkpoint 的 CPU 单环境 PPO `max_iterations=1`：epoch 1 完成，无 traceback，生成 `outputs/CmResidual/smoke_v11_20260915c/train.log`。
- 721-D builder 独立 shape/finite 检查：通过。

**保护边界与回滚**

- 未修改 `inspire.pth`、OI-Cm checkpoint、外部 DExplore checkout、训练输出或原始数据；未启动新的长训。回滚入口为本次代码/配置提交及本条 activity。

## 2026-09-15 10:07:48 +0800 — 修复后三卡训练完成

- activity_id: `ACT-20260915-181500-CMRESIDUAL-TRAIN-3GPU-V2-END`
- timestamp: `2026-09-15 10:07:48 +0800`
- modification_version: `V1.1.3`
- operation_category: `operation`、`experiment`、`documentation`
- task_mode: `run-only/operation`
- change_level: `L3`
- approval: `user-approved`
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `8265e4d`
- worktree_dirty: `true`（保留根目录及 CmDecoderv2 既有差异）
- scope: 三个独立 `outputs/CmResidual/cmresidual_dexplore_v2_gpu*_s*/` 运行目录及其 manifest、train.log、checkpoint；既有差异 `docs/current_versions.yaml`、`docs/logs/activity_log.md`、`docs/项目总览.md`、`src/task/CmDecoderv2/docs/logs/activity_log.md` 保留且未暂存。
- run_id: `cmresidual_dexplore_v2_gpu0_s201`、`cmresidual_dexplore_v2_gpu1_s202`、`cmresidual_dexplore_v2_gpu3_s203`
- run_status: `COMPLETED`（三组均正常达到 `max_epochs=1000`，无 traceback）
- conclusion: `INCONCLUSIVE`（工程训练完成；未做独立评估，不能推出策略效果）

**原因**

- 修复后的三卡运行需要独立工作目录和终态证据；三组均达到 max_epochs，故可结束运行并登记 checkpoint。

**验证**

- 三份 train.log 均包含 `MAX EPOCHS NUM!` 且不含 `Traceback`；manifest 与 checkpoint 已逐一重载核对。

**终态产物**

- GPU0：manifest [outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/run_manifest.json)，日志 [train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/train.log)，最近 checkpoint [last_CmResidual_ep_900_rew_-33.907387.pth](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/runs/CmResidual_15-09-42-05/nn/last_CmResidual_ep_900_rew_-33.907387.pth)，最佳 checkpoint [CmResidual.pth](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/runs/CmResidual_15-09-42-05/nn/CmResidual.pth)，checkpoint 内 best `last_mean_rewards=29.805084`（epoch 599）。
- GPU1：manifest [outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/run_manifest.json)，日志 [train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/train.log)，最近 checkpoint [last_CmResidual_ep_900_rew_8.34217.pth](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/runs/CmResidual_15-09-42-05/nn/last_CmResidual_ep_900_rew_8.34217.pth)，最佳 checkpoint [CmResidual.pth](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/runs/CmResidual_15-09-42-05/nn/CmResidual.pth)，checkpoint 内 best `last_mean_rewards=25.777279`（epoch 719）。
- GPU3：manifest [outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/run_manifest.json)，日志 [train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/train.log)，最近 checkpoint [last_CmResidual_ep_900_rew_2.2933483.pth](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/runs/CmResidual_15-09-42-05/nn/last_CmResidual_ep_900_rew_2.2933483.pth)，最佳 checkpoint [CmResidual.pth](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/runs/CmResidual_15-09-42-05/nn/CmResidual.pth)，checkpoint 内 best `last_mean_rewards=18.345146`（epoch 959）。

**验证与边界**

- 三份日志均包含 `MAX EPOCHS NUM!` 且无 `Traceback`；三份 checkpoint 均可用 CPU `torch.load` 重载并包含 model/optimizer/epoch/frame。
- reward 仅是训练期滑动回报，不是成功率；没有生成 `metrics.jsonl`，因此本条不声明科研假设成立。
- 未修改旧运行、外部 DExplore checkout、`inspire.pth`、原始数据和用户既有 dirty diff。

## 2026-09-15 09:45:45 +0800 — 修复后三卡并行训练启动

- activity_id: `ACT-20260915-094545-CMRESIDUAL-TRAIN-3GPU-V2`
- timestamp: `2026-09-15 09:45:45 +0800`
- modification_version: `V1.1.3`
- operation_category: `operation`、`experiment`
- task_mode: `run-only/operation`
- change_level: `L3`（长时三卡训练）
- approval: `user-approved`（用户明确要求修复后开始训练并使用三张卡）
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `8265e4d`
- worktree_dirty: `true`（保留根目录及 CmDecoderv2 既有差异）
- scope: 三个独立 `outputs/CmResidual/cmresidual_dexplore_v2_gpu*_s*/` 运行目录；代码和配置来自提交 `8265e4d`；工作区既有差异 `docs/current_versions.yaml`、`docs/logs/activity_log.md`、`docs/项目总览.md`、`src/task/CmDecoderv2/docs/logs/activity_log.md` 保留且未暂存。
- run_id: `cmresidual_dexplore_v2_gpu0_s201`、`cmresidual_dexplore_v2_gpu1_s202`、`cmresidual_dexplore_v2_gpu3_s203`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`（训练进行中，尚无科研结论）

**启动参数**

- 三组均使用 `task.env.numEnvs=64`、`max_iterations=1000`、`horizon_length=32`、`minibatch_size=2048`、`pipeline=cpu`；PPO 分别使用 `cuda:0`、`cuda:1`、`cuda:3`。
- 基础策略 checkpoint 为 `/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth`，SHA256=`8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。

**运行证据**

- GPU0 manifest：[outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/run_manifest.json)，日志：[train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu0_s201/train.log)。
- GPU1 manifest：[outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/run_manifest.json)，日志：[train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu1_s202/train.log)。
- GPU3 manifest：[outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/run_manifest.json)，日志：[train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_v2_gpu3_s203/train.log)。
- 启动后约 3 分钟检查：三个 PID 仍存活，分别已达到约 epoch 111、190、191；各目录已生成运行 checkpoint。

**边界**

- 三组均采用独立工作目录，rl_games 的 `runs/` 不共享；未触碰旧运行、外部 DExplore checkout、checkpoint 和用户既有 dirty diff。终态前保持 `RUNNING`，完成后补充 last epoch、best/last checkpoint、日志和结论。

## 2026-09-15 09:40:00 +0800 — DExplore 1442 维观测与 Cm PPO 链路修复

- activity_id: `ACT-20260915-094000-CMRESIDUAL-FIX`
- timestamp: `2026-09-15 09:40:00 +0800`
- modification_version: `V1.1.3`
- operation_category: `code`、`documentation`
- task_mode: `change`
- change_level: `L2`（观测 schema、策略网络和训练注册）
- approval: `user-approved`（用户明确要求修复问题并开始训练）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `bbb1ce28089fc202921d2660ddd45409f487a6cd`
- worktree_dirty: `true`（保留根目录及 CmDecoderv2 既有差异）
- scope: `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/`、`third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_*.py`、`third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml`、`third_party/IsaacGymEnvs/isaacgymenvs/train.py`
- run_id: `smoke_fixed_20260915d`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（工程 smoke；不代表科研效果成立）

**修改**

- 观测改为显式构造两个 DExplore 721-D offset，输出严格为 1442-D；移除旧 71-D 向量加零尾逻辑。
- 新增 `cm_continuous`/`cm_actor_critic` 注册；Cm online 特征进入 actor/critic，target 以 EMA 更新并冻结，PPO 梯度回传 online 分支。
- 关闭 rl_games 自动 torch.compile，规避当前 PyTorch 2.4.1 环境的编译兼容错误。

**验证**

- `python3 -m py_compile ...`：通过。
- `PYTHONPATH=. python3 -m pytest -q src/task/CmResidual/tests/test_reference_contract.py`：`2 passed`。
- CPU 单环境 `max_iterations=1`：完成 epoch 1，生成 `runs/CmResidual_15-09-38-37/nn/last_CmResidual_ep_1_rew__2.47_.pth`，无 traceback。

**保护边界与回滚**

- 未修改 DExplore checkout、`inspire.pth`、原始数据、旧运行或既有用户 dirty diff；新训练使用独立输出目录。回滚入口为本次代码提交及本条 activity。

# CmResidual 活动记录

## 2026-09-15 12:58:28 +0800 — 按指导 V1.2 修正几何与冻结 OI-Cm 链路

- activity_id: `ACT-20260915-125828-CMRESIDUAL-V12-FIX`
- timestamp: `2026-09-15 12:58:28 +0800`
- modification_version: `V1.2.0`
- operation_category: `architecture`、`code`、`documentation`
- task_mode: `change`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户明确认可 [V1.2 最终计划](../plan/V1.2.md) 并要求开始修改。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `c73190c1b8a3b1bb0dd4770d78ca8bca2d3694cf`
- worktree_dirty: `true`（保留指导文件及其他既有差异）
- scope: `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/`、CmResidual 配置、CmResidual 计划/README、版本指针及定向测试。

**文件**

- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py` — 从当前 hand/object URDF visual mesh 确定性采样点/法向，按实测 link/object pose 变换，生成 hand flow 与 reference contact。
- `src/task/CmResidual/docs/指导/V1.2.md` — 用户提供的本版本研究指导，作为本次修正依据。
- `src/task/CmResidual/tests/test_reference_contract.py` — 增加几何点云形状、法向和 flow 合同测试。
- `reference_provider.py` — 增加 reference link linear/angular velocity（固定 `dt=1/30` 差分）。
- `task.py` — 移除 `CmOnlineTarget`，使用真实 reference velocity/contact 和冻结 OI-Cm 输出（slot、anchor、object effect）作为策略上下文；teacher 仅读取 1442-D 前缀。
- `third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py` — 消费 task 生成的冻结 OI-Cm context，不再从 DExplore 向量伪造点云或重复 pooled token。
- `cm_adapter.py`、`residual_policy.py`、package 导出 — 删除废弃 online/target 路径。
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml` — 统一 airplane 资产入口，声明 OI-Cm checkpoint 与几何采样参数。
- [V1.2 最终计划](../plan/V1.2.md)、[Task README](../README.md)、[版本指针](../../../../../docs/current_versions.yaml) — 同步执行状态。

**原因**

V1.1.4 将 DExplore tracking-state 切片重解释为 OI-Cm 点云，并向 reference observation 传入全零 velocity/contact；这只能证明形状兼容，不能证明语义有效。V1.2 将几何和 reference 字段改为来自当前 URDF、仿真状态及 reference pose 的可追溯计算，并把 interaction slots、anchors 和 object-effect 暴露给 actor/critic。

**验证**

- `python3 -m py_compile third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/*.py third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_*.py`：通过。
- `PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests`：`3 passed`。
- `git diff --check`：通过；`dexplore_observation.py` 对 reference link/contact shape 增加 fail-closed 断言。
- CPU 单环境 `max_iterations=1` smoke（`horizon_length=4`、`minibatch_size=4`）完成，观测空间为 `2005`（1442-D teacher prefix + 563-D frozen context），无 traceback；运行入口为 [run_manifest.json](../../../../../outputs/CmResidual/smoke_v12_20260915_125828/run_manifest.json)，日志为 [train.log](../../../../../outputs/CmResidual/smoke_v12_20260915_125828/train.log)，checkpoint 为 `third_party/IsaacGymEnvs/runs/CmResidual_15-13-01-55/nn/last_CmResidual_ep_1_rew__1.89_.pth`。
- smoke 仅是工程 wiring/finite/冻结合同证据，科研结论为 `INCONCLUSIVE`；未启动长训。

**保护边界与回滚**

- 未修改原始数据、reference.npz、OI-Cm/DExplore checkpoint、外部 DExplore 工作树、历史 outputs/cache/checkpoint 或其他 Task。
- 回滚入口为恢复本次代码、配置和文档差异至 `c73190c`；历史产物未删除。

## 2026-09-15 09:18:52 +0800 — 三卡训练终态复核及实现无效更正

- activity_id: `ACT-20260915-091852-CMRESIDUAL-TERMINAL-AUDIT`
- timestamp: `2026-09-15 09:18:52 +0800`
- modification_version: `V1.1.2`（本次终态审计；被审计运行原标记为 `V1.1.1`）
- operation_category: `diagnostic`、`operation`、`documentation`
- task_mode: `run-only/operation`（终态查询与记录同步）
- change_level: `L0`（更正状态及记录实现缺陷，不修改训练实现、数据或指标定义）
- approval: `user-approved`（用户查询训练状态；沿用已有运行与记录授权）
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `bbb1ce28089fc202921d2660ddd45409f487a6cd`
- worktree_dirty: `true`（保留既有四处根/CmDecoderv2 文档差异）
- scope: `src/task/CmResidual/docs/logs/activity_log.md`、`outputs/CmResidual/cmresidual_dexplore*/run_manifest.json` 及各原 manifest 的一次性备份；没有启动训练、修改代码或 checkpoint。
- run_status: `COMPLETED`（3 组最终尝试）；`FAILED`（6 组之前的尝试）
- conclusion: `INVALID_IMPLEMENTATION`（相对用户要求的完整 DExplore + Cm 残差链路）

**原因**

- 先前将“可完成 PPO iteration”当成可以启动三卡的依据不成立。实际输入只是旧 71D 状态加 1371 个零，没有 DExplore 的两个 reference offset/layout；Cm 只被实例化，没有参与实际 PPO 的特征输入、replay 监督、梯度或 EMA 更新。
- GPU1/GPU3 最终尝试共用 `runs/CmResidual_15-01-48-27/`，最佳 checkpoint、config 与 TensorBoard 不能视为按 seed 隔离。最初三个尝试也共用 `runs/CmResidual_15-01-39-36/`。本次保留所有文件，只按独立 stdout 路径定位唯一的终态 checkpoint。
- 三组最终尝试均出现 `MAX EPOCHS NUM!`，终态 checkpoint 为 epoch 1000；对应进程已退出。旧 manifest 的 `RUNNING` 是未更新状态，现已更正。
- CPU/GPU 仿真和 64/256 env 数量不同，回报不可用作受控三 seed 比较；没有独立评估。本批结果不能证明 Cm 有效或无效。

**验证**

- `ps -p 2510695,2535229,2535230 -o pid=,stat=,etime=,args=`：三个 PID 均不存在；全进程扫描没有 `train.py task=CmResidual`。
- 逐一重载 9 份 manifest，读取 `train.log` 的完成标记、traceback、最后 epoch/frame 和 checkpoint 文件名；使用 CPU `torch.load` 读取 3 个终态 checkpoint 的 epoch/frame/last_mean_rewards/model keys。
- 三个 model state_dict 均没有 Cm 参数；`rg` 检查 `CmResidualActor`、`begin_ppo_block`、`supervised_loss` 仅定义/导出，无训练调用；配置选择普通 `actor_critic`。
- checkpoint 的 `last_mean_rewards` 是训练期历史最高回报，终态 checkpoint 文件名中的回报是最后一期滑动平均；均不是任务成功率或独立评估。stdout 的 frames 在迭代开始打印，最后累计步数以 checkpoint 的 frame 为准。
- 原始训练未生成 `metrics.jsonl`；本次不将共享 TensorBoard 数据伪造为独立逐 seed 曲线。

**文件与终态**

- run_id: `cmresidual_dexplore_gpu0_s101`；run_status: `COMPLETED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `1000`；last_step: `8192000`；best_metric: `4.765340328216553`；退出依据：达到 max_epochs=1000；原启动器未记录操作系统退出码，按完成标记、checkpoint 内容及进程退出确认终态。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu0_s101](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu0_s101/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu0_s101/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/train.log)。
  最近 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/last_CmResidual_ep_1000_rew__-121.24_.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/last_CmResidual_ep_1000_rew__-121.24_.pth)；最终训练回报：`-121.24`。
  最佳 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/CmResidual.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/CmResidual.pth)。
  GPU0 的最佳文件存在；目录也曾被同秒失败的 GPU1/3 初始化，不能把共享 config.yaml 当作本组独占配置。
- run_id: `cmresidual_dexplore_gpu1_s102`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu1_s102_cpu_sim`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：CPU 仿真 reset_idx 使用了 GPU 索引。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu1_s102_cpu_sim_retry2`；run_status: `COMPLETED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `1000`；last_step: `2048000`；best_metric: `2.997872829437256`；退出依据：达到 max_epochs=1000；原启动器未记录操作系统退出码，按完成标记、checkpoint 内容及进程退出确认终态。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/train.log)。
  最近 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-838.6_.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-838.6_.pth)；最终训练回报：`-838.6`。
  GPU1/3 的 CmResidual.pth、config.yaml、TensorBoard 路径相同；最佳文件归属不能仅由文件名确定。保留按本组 train.log 唯一文件名定位的终态 checkpoint。
- run_id: `cmresidual_dexplore_gpu1_s102_retry64`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu3_s103`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu3_s103_cpu_sim`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：CPU 仿真 reset_idx 使用了 GPU 索引。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu3_s103_cpu_sim_retry2`；run_status: `COMPLETED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `1000`；last_step: `2048000`；best_metric: `5.319734573364258`；退出依据：达到 max_epochs=1000；原启动器未记录操作系统退出码，按完成标记、checkpoint 内容及进程退出确认终态。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/train.log)。
  最近 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-70.02_.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-70.02_.pth)；最终训练回报：`-70.02`。
  GPU1/3 的 CmResidual.pth、config.yaml、TensorBoard 路径相同；最佳文件归属不能仅由文件名确定。保留按本组 train.log 唯一文件名定位的终态 checkpoint。
- run_id: `cmresidual_dexplore_gpu3_s103_retry64`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。

**保护边界与回滚**

- 代码、训练配置、checkpoint、原始日志、外部 DExplore、用户四处既有 dirty diff 均未修改；未创建或恢复运行。回滚本次记录可恢复每组 `run_manifest.before_terminal_audit.json` 并删除本条活动，不删除训练产物。

**规范反馈**

- 已有版本指针和 Task README 仍为 `V1.0.1`，与历史运行 `V1.1.1` 不一致；V1.1 plan 还配对 V1.0 指导。当前根指针属于既有 dirty diff，本次只登记冲突，没有静默改写用户指导或治理文件。运行终态查询不因该冲突被阻塞；后续实现应同步修正受影响导航和计划配对。无需新增规则，现有隔离/manifest 规则执行不到位。

## 2026-09-15 01:49:51 +0800 — 三卡训练资源调整后保持运行

- activity_id: `ACT-20260915-014951-CMRESIDUAL-TRAIN-3GPU-RETRY`
- timestamp: `2026-09-15 01:49:51 +0800`
- modification_version: `V1.1.1`
- operation_category: `operation`、`experiment`
- change_level: `L3`
- approval: `user-approved`（沿用三卡并行训练授权）
- branch: `oyx`
- base_commit: `03ea264`
- scope: three-run DExplore residual PPO operation; manifests/logs under `outputs/CmResidual/`; no source/cache/checkpoint mutation.
- run_id: `cmresidual_dexplore_gpu0_s101`、`cmresidual_dexplore_gpu1_s102_cpu_sim_retry2`、`cmresidual_dexplore_gpu3_s103_cpu_sim_retry2`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`

**状态**

- GPU0 使用 GPU PhysX，已运行至约 epoch 110/1000。
- GPU1/GPU3 的 GPU PhysX 在初始化阶段触发 Isaac Gym allocator error 700；未继续使用故障模式，改为 CPU PhysX + 对应 GPU 上的 PPO 网络，两个 retry2 进程已分别运行至 epoch 26/30。
- 失败尝试保留在 `outputs/CmResidual/cmresidual_dexplore_gpu{1,3}_s{102,103}*` 的 manifest/log 中，未覆盖重试目录。
- 当前三组 PID 与完整命令记录在各自 `run_manifest.json`；终态前不得把运行标为完成。

**scope**

- 三组运行 manifest 和日志分别位于 [GPU0 run](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/run_manifest.json)、[GPU1 retry run](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/run_manifest.json) 和 [GPU3 retry run](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/run_manifest.json)。

**验证**

- 启动后检查确认三个 PID 仍在运行；GPU0/GPU1/GPU3 分别有训练进程和对应显存占用。GPU1/GPU3 的 CPU PhysX retry 已通过 30 个以上 epoch，无 CUDA allocator error。
- 各目录的 [GPU0 train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/train.log)、[GPU1 train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/train.log)、[GPU3 train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/train.log) 持续写入；终态字段尚未产生。

**原因**

- GPU1/GPU3 的 GPU PhysX 在当前节点触发固定的 allocator error 700，继续重试同一模式没有证据会恢复；保留 GPU0 GPU PhysX，同时将另外两组切到 CPU PhysX、GPU PPO，以维持三卡并行且避免反复触发非法内存访问。

**验证与边界**

- GPU0、CPU smoke 和三组当前运行均使用 `inspire.pth` SHA256 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- 本条记录的是运行状态，不把 epoch/reward 作为科研结论；正式结论须等待独立评估和完整指标。

## 2026-09-15 01:37:55 +0800 — 三卡并行训练启动

- activity_id: `ACT-20260915-013755-CMRESIDUAL-TRAIN-3GPU`
- timestamp: `2026-09-15 01:37:55 +0800`
- modification_version: `V1.1.1`
- operation_category: `operation`、`experiment`
- change_level: `L3`（三卡并行 PPO 长任务）
- approval: `user-approved`
- approval_basis: 用户明确要求“直接完成整条链路，然后用三张空闲的卡并行训练”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `a22223e7c35570e79a53e165ff6ad5e1dc5f17d2`
- worktree_dirty: `true`（根目录既有文档差异未纳入运行）
- scope: vendor CmResidual DExplore package；外部 `inspire.pth` 只读；三次独立 seed、独立输出目录。
- run_id: `cmresidual_dexplore_gpu0_s101`、`cmresidual_dexplore_gpu1_s102`、`cmresidual_dexplore_gpu3_s103`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`（运行尚未结束；训练结果不能由启动状态推断）

**运行合同**

- command template: `PYTHONPATH=/home2/wyy/isaac-gym/isaacgym/python:/home2/wyy/oyx_ws/Ref2Dex:/home2/wyy/oyx_ws/Ref2Dex/third_party/IsaacGymEnvs python3 isaacgymenvs/train.py task=CmResidual headless=True force_render=False pipeline=gpu sim_device=cuda:<gpu> rl_device=cuda:<gpu> graphics_device_id=<gpu> task.env.numEnvs=256 max_iterations=1000 seed=<seed> task.basePolicy.checkpoint=/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth train.params.config.horizon_length=32 train.params.config.minibatch_size=2048`
- GPU/seed mapping: `cuda:0/101`、`cuda:1/102`、`cuda:3/103`；三次运行不共享 checkpoint 或 optimizer。
- base checkpoint SHA256: `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- outputs: `outputs/CmResidual/cmresidual_dexplore_gpu{0,1,3}_s{101,102,103}/`；日志与 manifest 在各自目录，均为 `PENDING` 直至进程生成。

**启动证据**

- 三张卡启动前显存/利用率检查：GPU0 `2553 MiB/0%`、GPU1 `776 MiB/0%`、GPU3 `1997 MiB/0%`；GPU2/4/5/6 正在使用，GPU7 保留作故障回退。
- 单卡 GPU0 1-iteration smoke 已完成（8 envs）；CPU 1-iteration smoke 也完成。正式三卡运行不继承 smoke checkpoint。
- 终态必须补写 `last_step`/`last_epoch`、best/latest checkpoint、`train.log`、`metrics.jsonl`（若生成）和失败/停止原因。

**保护边界与回滚**

- 不修改外部 DExplore checkout、`inspire.pth`、旧 reference/cache、用户既有根/CmDecoderv2 dirty diff。
- 停止入口：按 run_id 单独终止对应 PID；删除各自 ignored output 目录即可回滚运行产物。

## 2026-09-15 01:08:43 +0800 — V1.1 DExplore 18D package contract

- activity_id: `ACT-20260915-010843-CMRESIDUAL-V11-PACKAGE`
- timestamp: `2026-09-15 01:08:43 +0800`
- modification_version: `V1.1.0`
- operation_category: `governance`、`architecture`、`code`、`documentation`
- change_level: `L3`（vendor Task package、公共 import、checkpoint/obs/action contract）
- approval: `user-approved`
- approval_basis: 用户明确确认取消 vendor `logs*` 忽略、使用冻结 DExplore `inspire.pth`、同步 1442D obs 和原生 18D action，并按既定 Cm online/target 方案执行。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9654d657702a8a8da5f6c02c2ddd52dd1b2e7d54`
- worktree_dirty: `true`（保留既有根/CmDecoderv2 用户差异，未覆盖、未暂存）
- scope: `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/`、vendor configs、vendor `.gitignore`、V1.1 plan；旧数据、checkpoint、output 和 `src/base/` 未修改。
- conclusion: `SUPPORTED`（工程接口 smoke；不代表 PPO 效果或科研结论）

**文件与变更**

- [V1.1 计划](../plan/V1.1.md) — 锁定 DExplore teacher、1442D obs、18D residual/action、Cm online/target、回滚和验证边界。
- [DExplore base policy](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/base_policy.py) — 加载并冻结 `inspire.pth`，校验网络形状与 running stats。
- [Cm adapter](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_adapter.py) — target 冻结、PPO block EMA 和 replay feature 接口。
- [Residual actor](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/residual_policy.py) — 将冻结 teacher 与 Cm feature 拼接，输出有界 18D residual action。
- [Task package](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 18D residual 与 DExplore PD/mimic 合成；`__init__.py` 保持 `CmResidual` import 入口。
- [架构记录](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/docs/logs/architecture_log.md) — 追加 V1.1 用户确认快照；vendor `logs*` 忽略已删除。

**验证**

- `python3 -m py_compile third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/*.py`：通过。
- 直接加载 `inspire.pth` 并前向零输入：输出 shape `(2,18)`，值域在 `[-1,1]`；checkpoint SHA256=`8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- `CmOnlineTarget(1442)` smoke：online/target feature shape 均为 `(2,128)`，通过。
- 未启动 Isaac Gym、PPO 或长时运行；因此本条不产生 `run_id`，也不宣称训练收益。

**原因**

- 旧入口是 12D CmDecoder residual，无法直接承载 DExplore teacher 的 18D wrist+finger action，也无法
  复用其 1442D policy observation。将同名脚本转换为 package 后，base checkpoint、Cm 生命周期和
  DExplore action mapping 有独立可测试边界，训练入口继续使用原 `tasks` 注册表。

**保护边界与回滚**

- 未提交外部 DExplore checkout、`inspire.pth`、旧 reference/cache/output/checkpoint、共享 `src/base/` 或用户已有 dirty diff。
- 回滚入口：回退本条 V1.1 文件/配置和 package 迁移提交，恢复旧 `tasks/cm_residual.py` 与 `.gitignore` 的 `logs*` 规则。

## 2026-09-14 18:41:47 +0800 — 建立 canonical Task 并完成首轮 reference 连续性诊断

- activity_id: `ACT-20260914-184147-CMRESIDUAL-GOVERNANCE-D0`
- timestamp: `2026-09-14 18:41:47 +0800`
- modification_version: `V1.0`
- operation_category: `governance`、`documentation`、`diagnostic`
- change_level: `L3`（新增 Task owner、根 Task 索引和版本指针），内含 `L0` 只读数据诊断
- approval: `user-approved`（Task owner 与计划方向）；`execution-pending`（代码、数据、仿真和训练）
- approval_basis: 用户明确回复“是的”，确认 V1.0 采用 `src/task/CmResidual` owner、单轨迹范围、累计
  base residual、联合 RSI、gate/pilot 优先且不直接长训
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留用户已有根/CmDecoderv2 activity diff，不覆盖、不暂存）
- scope: `docs/current_versions.yaml`、`docs/项目总览.md`、`src/task/CmResidual/`；不包含其他已有工作树差异
- conclusion: `INCONCLUSIVE`（Task 边界已明确；现有几何 reference 尚不足以安全冻结 PPO 数据与控制合同）

**范围与文件**

- 将用户原始 [指导/V1.0.md](../指导/V1.0.md) 原文迁入 canonical Task；迁移前后 SHA256 均为
  `bd2d555ae89f1b5a4d30393cb1bfaa776e7f83a7817cb36cc4e8fa30bb8df6a4`，未修改其内容。
- 将 [plan/V1.0.md](../plan/V1.0.md) 草案迁入 canonical Task，记录用户已批准方向和新增 `D0` 闸门；
  `status` 仍为 `draft`、`execution_approval` 仍为 `pending`。
- 新增 [Task 入口](../README.md)，并更新
  [根 Task 索引](../../../../../docs/项目总览.md) 与
  [当前版本指针](../../../../../docs/current_versions.yaml)。
- 未修改 `cm_residual.py`、Hydra 配置、共享 `src/base`、旧 cache/data/checkpoint/output 或用户指导；未启动
  Isaac Gym、PPO、数据生成或长时任务。

**D0 新证据**

- `s1/airplane_lift` sidecar 为 432 帧、30 Hz，`source_frame_id=0,4,...,1724`，映射连续且 object/q/wrist
  长度一致。独立 q6 单元素步差 p50/p95/p99/max 为 `0.000283/0.018418/0.071911/0.184409 rad`。
- wrist world 单步 translation p50/p95/p99/max 为 `5.90/53.68/85.08/100.14 mm`，rotation 为
  `0.109/6.520/11.542/16.946 deg`；object translation p95/max 为 `15.25/20.50 mm`。
- reference index `44..410` 为连续 `<=2 cm` hand-object 几何区间。排除远离物体的首尾后，腕姿 p95 明显
  降低，但 index `43/46/50/53/149/260/286/339/342` 等仍有局部旋转尖峰；index `342` 的 robot wrist
  为 `27.38 mm/10.55 deg`，同源原始 GRAB 右腕仅 `5.46 mm/1.81 deg`。
- 原始 GRAB 手指在 index `342` 同时存在约 `31.79 deg` 的单关节快速运动。producer 代码会把 6 个 virtual
  wrist 与 12 个 finger/mimic DOF 一起交给逐帧 warm-start retarget 优化，因此手指快速变化可能由 wrist
  补偿放大；后续 coupled fit 明确保持 native wrist `q[:,0:6]` 不变，不能修复该来源问题。
- 按 source URDF 的 joint velocity limit 和 30 Hz 差分，现有 native q 有 44 个 transition 至少一关节超限，
  其中 17 个位于 reference transition `43..409`。当前 legacy 环境又把全部 DOF velocity 覆盖为 `7.0`，
  因此不能把 legacy smoke 的可运行性当作 reference 物理可跟踪性。
- `inspire_geometric` 原始导出根没有 converter run manifest；筛选 manifest 和后续 coupled-fit manifest 能锁定
  筛选/拟合，但不能锁定首次 retarget 的精确命令、迭代参数和 producer commit。当前 provenance 不满足拟定的
  frozen reference 合同。

**原因**

- 开头/结尾的大位移在原始 GRAB 右手轨迹中同步出现，主要是实际接近/撤离，不应一概误报为 stage 拼接。
- interaction 区间内的部分腕姿放大和 joint-limit 超限仍未解决；静默裁剪、平滑或重采样会改变 GT、时间合同和
  checkpoint 解释，不能在 loader 中自行修复。
- 因此 `D0` 当前为 `INCONCLUSIVE`：可以继续只读定位和提出数据方案，但不得把计划标为 final，不得进入
  reference builder、physics playback 或 PPO。

**验证**

- 只读加载 sidecar/geometry/raw GRAB、producer 代码与 URDF，核对 shape、frame mapping、SE(3) 步差、接触距离、
  原始人体右腕/手指变化、native q 差分和 URDF velocity limit；未写入任何诊断 data/output。
- 使用 `sha256sum` 确认指导迁移前后内容一致；交接前使用 `audit_diff.py --check-links` 检查本条链接，并运行
  `git diff --check`。

**回滚**

- 回滚入口：只回退 [根 Task 索引](../../../../../docs/项目总览.md)、
  [当前版本指针](../../../../../docs/current_versions.yaml) 和本 Task 新增文档；若恢复迁移前布局，只移动指导/plan
  回原 vendor-local 文档目录。不得 reset 用户已有根/CmDecoderv2 activity diff。

## 2026-09-14 19:05:21 +0800 — 定位 wrist quaternion 合同并批准新 reference 方向

- activity_id: `ACT-20260914-190521-CMRESIDUAL-D0-WRIST-CONTRACT`
- timestamp: `2026-09-14 19:05:21 +0800`
- modification_version: `V1.0`
- operation_category: `diagnostic`、`architecture`、`documentation`
- change_level: `L2`（坐标、GT/reference、数据 schema 和验收合同）
- approval: `user-approved`（固定 wrist SE(3) mapping、finger-only constrained retarget、新版本产物且旧 sidecar
  不覆盖）；`execution-pending`（主帧区间、定量合同、代码、数据和物理 gate）
- approval_basis: 用户对“显式固定 GRAB 右腕→robot wrist SE(3) 映射，只优化 6 个独立手指 DOF 并施加
  速度/加速度约束，同时保留旧 sidecar 不覆盖”的推荐方案明确回复“是的”
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留用户已有根/CmDecoderv2 activity diff，不覆盖、不暂存）
- scope: `docs/current_versions.yaml`、`docs/项目总览.md`、`src/task/CmResidual/docs/指导/V1.0.md`、
  `src/task/CmResidual/docs/plan/V1.0.md`、`src/task/CmResidual/docs/README.md`、
  `src/task/CmResidual/docs/logs/activity_log.md`；本次新增实质编辑仅为 plan、Task 入口和本条 activity，前三项
  是同一未提交 Task diff 中已由上一条 activity 记录的版本指针、根导航和原文迁移
- conclusion: `REFUTED`（旧 reference 不能直接作为 training GT）；新 fixed-`X_HB` replacement 在 D1 生成和
  验证前仍为 `INCONCLUSIVE`

**新增坐标证据**

- DExplore tensor columns `51:57` 保存右腕 translation/exp-map，global joint quaternion block 位于
  `245:373`，其中右腕 quaternion 为 columns `309:313`、顺序 `xyzw`。对 432 帧比较两种旋转，SO(3)
  geodesic 差异 min/median/p95/max=`24.88/49.99/113.81/174.09 deg`，全部帧不等价。
- producer 的 `quat_to_exp_map` 对负 `w` 用 `abs(w)` 计算角度，却没有同步翻转 vector part；本序列右腕
  quaternion 的 `w` 全为负。新 schema 因此显式拒绝 columns `54:57`，只接受单位化的 columns `309:313`。
- 用同一 tensor 的 wrist/MCP landmarks 构造 human palm frame，得到其相对 global wrist quaternion 的旋转在
  432 帧内为数值常量，独立支持 quaternion 字段和 `xyzw` 解释。
- 采用正确 quaternion 后，原 indices `[44,410]` 的连续 `<=2 cm` interaction 区间共有 367 帧，对应 raw
  `source_frame_id=[176,1640]`；human wrist linear/angular speed 最大值=`1.104 m/s` / `2.248 rad/s`，低于
  source URDF virtual wrist limit `2.0 m/s` / `3.14 rad/s`。完整 432 帧最大值仍达到 `2.080 m/s` /
  `4.873 rad/s`，不推荐直接进入 V1.0 训练。
- 新合同将 robot wrist 固定为 `T_WB(t)=T_WH(t) X_HB`，整段只允许一个 `X_HB ∈ SE(3)`；逐帧变量仅 q6，
  mimic 固定，q6 写出前必须通过 position、`1.0 rad/s` velocity 和计划预注册的 `20.0 rad/s^2` acceleration
  硬门。不得从旧 robot wrist 拟合逐帧 offset。

**原因**

- 新坐标证据推翻了旧 exp-map 的 GT 身份；继续沿用旧 wrist 或用其拟合逐帧 offset 会把 converter 错误和
  finger compensation 固化进 base checkpoint。固定 `X_HB` 能表达一次性的 human→robot morphology 映射，
  同时阻断逐帧 wrist 泄漏；q6 约束则让新 reference 在进入 Isaac Gym 前先满足明确的运动学合同。

**计划与保护边界**

- 更新 [V1.0 计划草案](../plan/V1.0.md)，加入权威 tensor columns、fixed-`X_HB` 求解、q6 硬门、367 帧推荐
  区间、精确 action/reward/RSI/PPO 工程 gate 和稳定 CLI 合同；状态仍为 `draft`、执行审批仍为 `pending`。
- 旧 sidecar 与同源 [geometry manifest](../../../../../data/processed_data/coupled_geometric_cache_v1_20260912/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)
  只读保留；未裁剪、平滑、重采样或重导出，未修改外部 DExplore checkout、vendor adapter、Hydra 配置、
  `src/base/`、checkpoint、output 或现有运行进程。
- 一次 CPU 内存 feasibility probe 未捕获到可复核 stdout，进程随后已不存在且未写文件；该尝试不登记为正式
  run、不作为阈值或方向证据，也未重复启动。

**验证**

- 只读加载 raw tensor、同源 geometry、producer 和 URDF，检查 quaternion/exp-map geodesic、palm-frame
  一致性、frame mapping 以及 30 Hz wrist velocity；所有数值仅用于 D0 诊断。
- 交接前运行 `audit_diff.py --check-links` 和 `git diff --check`。

**回滚**

- 文档改动的回滚入口为本条 activity、[Task 入口](../README.md) 和 [V1.0 计划草案](../plan/V1.0.md)；
  同一未提交 Task diff 还包括 [指导原文](../指导/V1.0.md)、
  [根 Task 索引](../../../../../docs/项目总览.md) 与 [当前版本指针](../../../../../docs/current_versions.yaml)。
  不得 reset 用户已有根/CmDecoderv2 activity diff。

## 2026-09-14 20:41:02 +0800 — V1.0 定稿后发现 pinky mimic 合同冲突并暂停 D1

- activity_id: `ACT-20260914-204102-CMRESIDUAL-D1-MIMIC-BLOCKER`
- timestamp: `2026-09-14 20:41:02 +0800`
- modification_version: `V1.0.1`
- operation_category: `governance`、`architecture`、`diagnostic`、`documentation`
- change_level: `L2`（mimic、native q、FK 与未来 checkpoint 解释）
- approval: `user-approved`（V1.0 final、indices `[44,410]`、D1 builder/artifact/offline validation）；
  `amendment-pending`（pinky mimic `1.05` 或 `1.18`）
- approval_basis: 用户对“367 帧主 reference、计划定稿、只执行 D1 且不运行 Isaac Gym/PPO”明确回复“同意”；
  mimic 冲突是在定稿后的实现前检查中新增发现，不能沿用原批准静默选择
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留用户已有根/CmDecoderv2 activity diff，不覆盖、不暂存）
- scope: `docs/current_versions.yaml`、`docs/项目总览.md`、`src/task/CmResidual/docs/指导/V1.0.md`、
  `src/task/CmResidual/docs/plan/V1.0.md`、`src/task/CmResidual/docs/README.md`、
  `src/task/CmResidual/docs/logs/activity_log.md`；未修改代码、配置、旧数据、cache、vendor 或外部 DExplore
- run_id: `N/A`（仅内存 feasibility probe，不是正式数据运行）
- run_status: `N/A`
- conclusion: `INCONCLUSIVE`（human target 与 fixed-`X_HB`/q6 可行性已有正证据，但 mimic 合同未决，D1
  尚不得生成）

**文件**

- [V1.0 最终计划](../plan/V1.0.md) — 记录用户的 final/D1-only 批准，并在发现 mimic 冲突后将执行状态设为
  `paused-pending-mimic-contract-amendment`。
- [Task 入口](../README.md) — 更新 `V1.0.1`、D1-only 边界和当前 blocker。
- [当前版本指针](../../../../../docs/current_versions.yaml) 与
  [根 Task 索引](../../../../../docs/项目总览.md) — 更新到 D1 `V1.0.1` 和 final 计划状态。
- [指导原文](../指导/V1.0.md) — 同一未提交 Task diff 的既有迁移文件，本轮未改内容。

**原因**

- legacy coupled/CmResidual contract 把 native pinky `q11` 展开为 `1.05*q10`；当前锁定 URDF 的 `<mimic>`
  明确为 `1.18*pinky_proximal`，且仓库与 DExplore 的三份同名 URDF SHA256 相同。两者不可同时作为
  “固定 mimic”真值；静默选择会改变 reference geometry、速度门和 future checkpoint 身份。
- `research-change-control` 要求 final plan 无法按原语义执行时暂停并重新协商，因此没有开始 builder 文件编辑
  或正式数据写出。

**验证**

- 在 GPU6 只用内存重建同源 432 帧 SMPL-X 五指尖；对齐后旧 DExplore robot→human tip EPE
  mean/p95/max=`11.585/22.807/51.109 mm`，支持 target 重建链可复核。
- 对批准的 367 帧做内存 fixed-`X_HB` + q6 feasibility probe：相对 neutral-q6 baseline 的 tip EPE mean
  从 `18.421 mm` 降至 `9.156 mm`（改善 `50.3%`），p95/max=`21.618/35.099 mm`；投影后 q6
  velocity/acceleration max=`1.000 rad/s` / `20.000 rad/s^2`。这些数值仅证明优化可行，不是正式 artifact、
  P0 smoke 或科研结论，未写输出文件。
- 核对锁定 URDF：index/middle/ring=`1.05`、pinky=`1.18`、thumb=`0.60/0.8`；核对 legacy 代码与 coupled
  manifest：四个非 thumb source 均硬编码 `1.05`。运行 `git diff --check` 通过。

**回滚**

- 当前仅有文档/版本指针变化；回滚入口为 [V1.0 最终计划](../plan/V1.0.md)、[Task 入口](../README.md)、
  [根 Task 索引](../../../../../docs/项目总览.md)、[当前版本指针](../../../../../docs/current_versions.yaml) 和
  本条 activity。未产生需要删除的数据或运行目录，不得 reset 用户已有根/CmDecoderv2 activity diff。

## 2026-09-14 21:53:28 +0800 — D1 fixed-wrist constrained-q6 reference artifact

- activity_id: `ACT-20260914-215328-CMRESIDUAL-D1-BUILD`
- timestamp: `2026-09-14 21:53:28 +0800`
- modification_version: `V1.0.1`
- operation_category: `code`、`data`、`operation`、`documentation`
- change_level: `L2`（reference、坐标、mimic、数据 schema 与离线验收）
- approval: `user-approved`
- approval_basis: 用户在恢复会话后要求继续；沿用 D1-only 批准并确认采用锁定 URDF pinky mimic `1.18`，不启动 Isaac Gym/PPO。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留既有根/CmDecoderv2 文档差异；未覆盖或暂存无关改动）
- scope: `src/task/CmResidual/tools/data/build_reference.py`、`src/task/CmResidual/tests/test_reference_contract.py`、`src/task/CmResidual/docs/plan/V1.0.md`、`src/task/CmResidual/docs/README.md`；新数据产物位于 ignored `data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/`
- run_id: `s1_airplane_lift`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（schema/SE(3)/mimic/q6 运动门和五指尖改善通过；源 cache 无 raw contact flag，接触保留门不可计算，故 artifact 暂不具备 training-eligible 状态）

**文件**

- [指导原文](../指导/V1.0.md) — 既有同一 Task dirty diff，本轮未改内容。
- [D1 builder](../../tools/data/build_reference.py) — 读取同源 quaternion/object pose，固定单一 `X_HB`，逐帧仅优化 q6，按 URDF mimic 展开并执行硬门；不修改旧 sidecar。
- [D1 contract tests](../../tests/test_reference_contract.py) — 验证 URDF-authoritative mimic 和 367 帧/source frame 合同。
- [D1 reference artifact](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/reference.npz) — 367 帧新 reference、wrist/object twist、link poses、mask。
- [D1 manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/manifest.json) — 输入 SHA、X_HB、mimic、优化器和 gate 状态。
- [run manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/run_manifest.json) — 运行命令、base commit 和终态。

**原因**

- 旧 sidecar 的 pinky `1.05` 与锁定 URDF 的 `1.18` 不兼容；新 artifact 明确记录 q13=`1.18*q12`，旧数据继续只作历史兼容。
- 固定 `X_HB` 后在批准的 `[44,410]` 区间生成 367 帧；五指尖 RMS `57.36 mm`，相对同一固定腕映射的 neutral baseline `88.88 mm` 改善 `35.46%`，超过计划的 `20%` 工程门。

**验证**

- `PYTHONPATH=. python3 src/task/CmResidual/tools/data/build_reference.py --output data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift`：完成，未覆盖既有输出。
- `PYTHONPATH=. python3 -m pytest -q src/task/CmResidual/tests/test_reference_contract.py`：`2 passed`。
- 独立 `numpy.load`/JSON 重载核对：所有字段 finite；367 帧，Task-local frame `0..366` 与 source frame `176..1640` 分离；X_HB 旋转行列式 `1.0`、正交误差 `<4e-16`、最后齐次行正确；mimic 最大误差 `<1.2e-7 rad`；独立 q6 速度最大 `1.000001 rad/s`（float32 保存舍入，容差 `1e-6`），加速度最大 `11.83 rad/s²`。
- `contact_flag=UNAVAILABLE`：输入 geometry manifest 未提供 raw contact flag，未用距离阈值伪造接触标签；因此 manifest/run manifest 均标记 `training_eligible=false` 与 `INCONCLUSIVE`。
- `git diff --check`：通过。

**保护边界与回滚**

- 未修改旧 coupled-geometric sidecar、CmDecoder/CmResidual legacy 配置、vendor adapter、`src/base/`、外部 DExplore checkout、checkpoint 或训练输出；未启动物理 gate/PPO。
- 回滚入口：删除新 ignored artifact 目录，并回退本条 builder/test、plan/README 和 activity 差异；旧输入与旧 reference 不受影响。


## 2026-09-14 22:02:00 +0800 — D1 contact flag source audit

- activity_id: `ACT-20260914-220200-CMRESIDUAL-CONTACT-AUDIT`
- timestamp: `2026-09-14 22:02:00 +0800`
- modification_version: `V1.0.1`
- operation_category: `diagnostic`、`documentation`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求继续；仅检查现有 source/parent cache 的 contact 字段，不修改数据或运行配置。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `6d54106`
- worktree_dirty: `true`
- scope: coupled geometric source、cm_object_v2 parent cache 和其 manifest；只读。
- run_id: `cmresidual_contact_flag_audit_20260914_220200`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`

**文件**

- [D1 manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/manifest.json) — 对照现有 artifact 的接触字段状态。
- [source geometry manifest](../../../../../data/processed_data/coupled_geometric_source_v1_20260912/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json) — 输入 schema 与字段清单。

**原因**

- D1 的接触保留门要求 raw contact flag；现有 source/parent cache 没有该标注，几何候选 mask 不能替代它，因此维持 `INCONCLUSIVE`。

**验证**

- source geometry manifest 只包含 pose、surface、frame mapping 和 q provenance，没有 raw contact/touch flag。
- parent cache 的 `right/candidate_active_5cm.npy` 语义为生成的 5 cm 几何候选 mask，不是传感器或标注 contact flag，不能代替接触真值。
- 未修改 cache、artifact、checkpoint、配置或运行进程。

**回滚**

- 本条仅为诊断记录；删除本条即可回滚。
## 2026-09-15 23:17:54 +0800 — V1.6 corrected reference 训练准入通过

- activity_id: `ACT-20260915-231754-CMRESIDUAL-V16-REFERENCE-ELIGIBLE`
- timestamp: `2026-09-15 23:17:54 +0800`
- modification_version: `V1.6`
- operation_category: `data`、`code`、`config`、`diagnostic`、`documentation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户要求使用具备训练资格的 corrected reference，并授权其余技术决策；采用已批准的 V1.0 D1 门禁。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: CmResidual corrected reference eligibility；保留旧 v1、source/cache、checkpoint、旧运行和用户已有 dirty diff。
- run_id: `cmresidual_reference_v2_build_20260915_231227`
- run_status: `COMPLETED`

**文件**

- [V1.6 指导](../指导/V1.6.md) — 记录用户目标和保护边界。
- [V1.6 最终计划](../plan/V1.6.md) — 固定 raw contact 与 20 mm 距离准入合同。
- [builder](../../tools/data/build_reference.py) — 读取 source tensor `205:206` raw object contact，并用固定 URDF/采样参数执行距离门。
- [contract tests](../../tests/test_reference_contract.py) — 增加 raw contact 列合同和 eligible manifest 回归断言。
- [CmResidual.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml)、[CmResidualOnline.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml) — 默认切换到 v2 artifact。
- [v2 reference manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/manifest.json) — `training_eligible=true`。
- [v2 run manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v2/s1_airplane_lift/run_manifest.json) — 数据构建运行证据。

**接手时既有未提交改动（本次未重写）**

- `docs/current_versions.yaml`
- `src/task/CmResidual/docs/README.md`
- `src/task/CmResidual/docs/logs/experiment_log.md`
- `src/task/CmResidual/tools/eval_zero_residual.py`
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py`
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py`

**原因**

旧 v1 曾因 geometry manifest 未声明接触而保持 `training_eligible=false`。只读审计确认 source tensor 的固定列合同提供 raw object contact；主区间 296 个 raw-contact 帧中 295 个 corrected hand/object surface minimum distance `<=0.020 m`，比例 `0.9966216216`，达到预注册 `0.90` 门槛。未使用 5 cm candidate mask 重新标注接触，也未覆盖旧产物。

**验证**

- builder 完成并生成 367 帧 v2 artifact；schema、SE(3)、mimic、q-limit、tip-improvement、raw-contact-distance gates 全部 `PASS`。
- `training_eligible=true`；`raw_contact_frames=296`、`raw_contact_distance_pass_frames=295`、`raw_contact_distance_pass_fraction=0.9966216216`。
- `PYTHONPATH=third_party/IsaacGymEnvs python3 -m pytest -q src/task/CmResidual/tests`：`13 passed`。
- `python3 -m py_compile src/task/CmResidual/tools/data/build_reference.py`：通过；`git diff --check`：通过。
- 工程/data-contract 结论：`SUPPORTED`；正式训练效果、收敛和抓取结论：尚未产生，保持 `INCONCLUSIVE`。

**保护边界与回滚**

- 未修改旧 v1 reference、source tensor、geometry cache、checkpoint、共享 `src/base/`、外部 DExplore checkout 或既有运行输出。
- 回滚入口：删除 ignored `data/processed_data/cm_residual/reference_tracking_v2/`，撤回 V1.6 显式代码/config/文档增量；旧 v1 仍可用作历史诊断。
## 2026-09-15 23:37:04 +0800 — V1.7 正式 PPO 训练启动

- activity_id: `ACT-20260915-234500-CMRESIDUAL-V17-PPO-FORMAL`
- timestamp: `2026-09-15 23:37:04 +0800`
- modification_version: `V1.7`
- operation_category: `experiment`、`operation`
- change_level: `L3`
- approval: `user-approved`
- approval_basis: 用户要求使用具备训练资格的 corrected reference，并授权其余训练参数自主确定；V1.6 reference gate 已通过。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9167f8d5bc550b058d61d8904546ef67f2719770`
- worktree_dirty: `true`
- scope: V1.7 formal PPO；v2 eligible reference、seed42、GPU5、64 env、1000 epochs、canonical reward/action/checkpoints。
- run_id: `cmresidual_ppo_formal_v17_20260915_2345`
- run_status: `RUNNING`
- last_epoch: `2`（TensorBoard 已写入；终态字段 `PENDING`）

**文件**

- [V1.7 指导](../指导/V1.7.md) — 训练目标和保护边界。
- [V1.7 最终计划](../plan/V1.7.md) — 固定输入、预算、停止条件和回滚。
- [formal runner](../../tools/run_ppo_formal.py) — 读取 v2 manifest，`allowIneligibleFor` 为空；smoke 已通过。
- [formal output](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/) — 当前运行目录。
- [config.json](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/config.json) — 已生成。
- [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/run_manifest.json) — `RUNNING`。
- [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/metrics.jsonl) — `PENDING`，终态生成。
- [train.log](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/train.log) — 已生成。
- [checkpoint](../../../../../outputs/CmResidual/cmresidual_ppo_formal_v17_20260915_2345/rlgames_formal/nn/) — `PENDING`，终态生成。

**验证**

- V1.6 v2 manifest：`training_eligible=true`；运行时 metadata snapshot 已指向 v2。
- 1-env、0-update initialization smoke：`COMPLETED` / `SUPPORTED`，确认 GPU PhysX 初始化、reference 准入和任务构造通过。
- 正式训练已越过 GPU `prepare_sim`，TensorBoard 已记录第 1 个 epoch；截至本条未出现 non-finite 或 simulator error。
- 训练效果、收敛和抓取科研结论在终态前保持 `INCONCLUSIVE`。

**原因**

v2 corrected reference 已通过 raw-contact 距离准入，正式训练因此可以在 fail-closed 配置下启动；本次不使用旧 pilot checkpoint，也不放宽 reference eligibility。

**接手时既有未提交改动（本次未重写）**

- `docs/current_versions.yaml`
- `src/task/CmResidual/docs/README.md`
- `src/task/CmResidual/docs/logs/experiment_log.md`
- `src/task/CmResidual/tests/test_reference_contract.py`
- `src/task/CmResidual/tools/data/build_reference.py`
- `src/task/CmResidual/tools/eval_zero_residual.py`
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml`
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualOnline.yaml`
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py`
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py`
