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

**原因**

用户询问 reward 是否仍在增长；该判断需要用最新训练日志比较最近窗口、峰值和当前值，而不是只看单个 epoch。

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
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
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
## 2026-09-16 13:24:33 +0800 — V1.9 critic-only Cm A/B 实现与合同验证

- activity_id: `ACT-20260916-132433-CMRESIDUAL-V19-IMPLEMENTATION`
- timestamp: `2026-09-16 13:24:33 +0800`
- modification_version: `V1.9`
- operation_category: `code`、`diagnostic`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认 V1.9 歧义收敛方案，审阅计划草案后明确要求继续实现。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`
- worktree_dirty: `true`（开始时存在用户提供的 V1.9 指导和无关 ObjectInteractionCm 活动日志差异；后者未触碰、未纳入本次范围）
- scope: 按 V1.9 最终计划实现 matched control / critic-only Cm 的非对称 actor/critic 输入、配置、训练/评估入口和合同测试；只完成 CPU/静态验证，不启动 GPU smoke、训练或评估。
- conclusion: `SUPPORTED`（工程 observation、actor parity、配置和入口合同）；Cm 的 RL utility、抓取改善、统计显著性与 Experiment C 均为 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 当前指针更新为 V1.9。
- [src/task/CmResidual/docs/指导/V1.9.md](../指导/V1.9.md) — 用户提供的研究指导。
- [src/task/CmResidual/docs/plan/V1.9.md](../plan/V1.9.md) — 用户确认的最终执行边界、A/B 协议和停止条件。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态更新为实现完成、运行未启动。
- [third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) — 显式 actor/critic 输入宽度和前缀切片。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml) — 1442-D actor / 1442-D critic matched control。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml) — 1442-D actor / 2005-D critic-Cm 变体。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — V1.9 variant、输入 provenance、actor 初始化 parity 和 checkpoint schema gate。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — V1.9 B0/E10/E20 同协议 deterministic evaluator 与 episode-return 记录。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — V1.9 actor/critic 非对称输入、初始化 parity 和 V1.8 兼容回归。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 本次实现的唯一活动入口；本轮无运行目录或 manifest。

**原因**

V1.8 已证明小探索 residual 能在短程内保持 base 行为，但没有检验 Cm 的 RL utility。V1.9 将唯一实验变量收敛为 critic 是否读取现有冻结 563-D Cm 上下文；A/B 都计算相同 Cm 并共享 2005-D 环境路径，actor 始终只读 1442-D prefix，以避免环境路径、探索和 actor 初始化成为混杂变量。

**验证**

- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`19 passed, 4 warnings`；warnings 均为 Hydra 1.1 兼容提示。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile ...`：builder、train/eval runner 和测试文件全部通过。
- 纯配置 preflight：A/B 均为 2005-D observation、1442-D actor；critic 分别为 1442/2005-D；相同 seed 下 actor state 无 mismatch，跨 variant mean/sigma diff 和后缀扰动 action diff 均为 `0.0`。
- train/eval runner `--help` 均暴露 `v18_no_cm|control|critic_cm` 与显式 OI-Cm checkpoint 参数；`git diff --check` 和 staged activity/link audit 在提交前执行。
- 本轮未运行 Isaac Gym/GPU smoke，不能将静态合同通过解释为训练可运行或 Cm 有效。

**保护边界与回滚**

- 未修改 action mapper、reward、residual scale、reset、坐标系、数据/cache/schema、OI-Cm 模型或 checkpoint、共享 `src/base/`、外部 DExplore、旧输出和其他 Task；未启动或覆盖任何运行。
- 回退本次 V1.9 的 builder、两个新配置、runner/evaluator、测试和文档差异即可恢复 V1.8；现有 checkpoint/output 不受影响。
## 2026-09-16 15:21:52 +0800 — V1.9.1 critic-Cm smoke 完成并发现配对 RNG 失效

- activity_id: `ACT-20260916-152152-CMRESIDUAL-V191-CRITICCM-SMOKE`
- timestamp: `2026-09-16 15:21:52 +0800`
- modification_version: `V1.9.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准 V1.9 Gate II 实验；matched control 已完成 2 epochs 并通过全部 wiring/checkpoint 合同，按最终计划启动配对 critic-Cm smoke。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`
- worktree_dirty: `true`（使用已批准但未提交的 V1.9.1 实现；无关 ObjectInteractionCm 用户差异未触碰）
- scope: GPU5、64 env、seed42、2 epochs、critic-Cm（2005-D 环境 observation，actor 读取 1442-D prefix，critic 读取完整 2005-D）；其余变量与 control smoke 相同。
- run_id: `cmresidual_v19_critic_cm_smoke_20260916_152152`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 15:31:45 +0800`
- last_step: `4096`
- last_epoch: `2 / 2`
- best_metric: checkpoint deterministic reload action max abs diff `0.0`
- checkpoint: [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/CmResidualV19CriticCmT2/nn/last_CmResidualSafeCriticCm_ep_2_rew__6.79_.pth](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/CmResidualV19CriticCmT2/nn/last_CmResidualSafeCriticCm_ep_2_rew__6.79_.pth)
- exit_reason: 完成严格 2-epoch wiring smoke；单侧训练/checkpoint 合同通过，但 paired stochastic RNG 合同失败，按计划停止在 Gate II。
- conclusion: `SUPPORTED`（critic-Cm 单侧工程 wiring）；`INVALID_IMPLEMENTATION`（严格配对 RNG 合同）；Cm utility 和抓取效果 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 当前指针更新为 V1.9.1。
- [src/task/CmResidual/docs/指导/V1.9.md](../指导/V1.9.md) 和 [src/task/CmResidual/docs/plan/V1.9.md](../plan/V1.9.md) — 本轮遵循的指导与最终计划。
- [src/task/CmResidual/docs/README.md](../README.md) — 更新为 Gate II 后停止及 RNG 阻断状态。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) 和 [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 两侧运行终态、证据与结论边界。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — 新增显式 2-epoch/save-frequency 合同并保留 A/B 静态 parity 测试。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — 计划内最小修正：允许 2-epoch smoke、每 epoch 保存并记录 V1.9.1。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — V1.9.1 manifest 版本一致性；本轮未启动 evaluation。
- [third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) — V1.9 非对称 actor/critic 输入实现；本轮诊断定位其 critic replacement RNG 消耗差异。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml) 和 [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml) — 本轮实际运行的 1442/1442 与 1442/2005 配置。

**原因**

最终计划要求正式 A/B 前先完成各 2 epochs wiring smoke。runner 原只接受 10/20 epochs，因此先做不改变科研变量的最小预算修正；两侧完成后，首 epoch trajectory 差异进一步暴露静态 actor parity 未覆盖的 post-build RNG 状态差异。按计划停止并记录，不用无效配对结果判断 Cm 效果。

**产物与结果**

- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152) — 完整 critic-Cm smoke 目录。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/run_manifest.json) — 单侧 runner `COMPLETED / SUPPORTED` 终态。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/config.json](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/config.json) — resolved config、输入 SHA 和 actor 参数 parity。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/metrics.jsonl) — 2 行 epoch 指标。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/train.log](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/train.log) — 完整训练日志。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_20260916_152152/checkpoint_validation.json) — model/optimizer/action finite、固定 sigma 和重载验证。

**验证**

- initial mean=`0`，sigma min/max=`0.099999994` 且不可训练；checkpoint epoch/frame=`2/4096`，model/optimizer/action finite，重载 action diff=`0.0`。
- epoch 2：`c_loss=4.212568`、`a_loss=0.089947`、`KL=0.199856`、`success_fraction=0.09375`、`residual_rms=0.105254`、saturation=`0.001736`，全部 finite。
- control 与 critic-Cm 的 actor state、初始 mean/sigma 完全一致，但首 epoch success fraction 已为 `0.15625 / 0.203125`。离线同 seed 构造后下一组 `torch.rand(8)` 不相等，最大绝对差 `0.6465547085`；根因是 1442/2005-D critic replacement 消耗不同数量的 RNG draws。
- 因此两个 smoke 只支持各自 wiring，不是有效效果对照；未启动 B0、T10、E10 或更长训练。修复应让两侧按固定候选顺序消耗相同 RNG，并新增 post-build RNG parity 与首轮 action-noise parity gate，需重新确认后实施。
- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`20 passed, 5 warnings`；warnings 均为 Hydra 1.1 兼容提示。
- 修改 Python 文件 `py_compile` 与 `git diff --check` 通过；本轮没有残留 CmResidual 运行进程。

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_ppo_stability.py --variant critic_cm --max-epochs 2 --gpu 5 --seed 42 --num-envs 64 --run-id cmresidual_v19_critic_cm_smoke_20260916_152152 --activity-id ACT-20260916-152152-CMRESIDUAL-V191-CRITICCM-SMOKE`

**停止边界**

只运行 2 epochs；若 finite、checkpoint、固定 sigma、actor parity 或重载合同失败，则记录终态并停止，不进入 T10/E10。

## 2026-09-16 15:12:09 +0800 — V1.9.1 control GPU wiring smoke 完成

- activity_id: `ACT-20260916-151209-CMRESIDUAL-V191-CONTROL-SMOKE`
- timestamp: `2026-09-16 15:12:09 +0800`
- modification_version: `V1.9.1`
- operation_category: `code`、`experiment`、`operation`
- task_mode: `run-only/operation -> change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户要求“实验一下”；按 V1.9 最终计划只启动 Gate II A/B 各 2 epochs wiring smoke，不进入 T10/E10。runner 原只允许 10/20 epochs，先做计划内最小修正开放 2 epochs 并强制每 epoch 保存。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`
- worktree_dirty: `true`（使用已批准但未提交的 V1.9 实现；无关 ObjectInteractionCm 用户差异未触碰）
- scope: GPU5、64 env、seed42、2 epochs、matched control（2005-D 环境 observation，actor/critic 都只读取 1442-D prefix）；零 mean、固定 sigma=0.1、冻结 OI-Cm 和 V1.8 其余科研变量不变。
- run_id: `cmresidual_v19_control_smoke_20260916_151209`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 15:21:45 +0800`
- last_step: `4096`
- last_epoch: `2 / 2`
- best_metric: checkpoint deterministic reload action max abs diff `0.0`
- checkpoint: [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/CmResidualV19ControlT2/nn/last_CmResidualSafeCriticControl_ep_2_rew__6.78_.pth](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/CmResidualV19ControlT2/nn/last_CmResidualSafeCriticControl_ep_2_rew__6.78_.pth)
- exit_reason: 完成严格 2-epoch wiring smoke；全部训练、finite、固定 sigma、actor parity 和 checkpoint 重载合同通过。
- conclusion: `SUPPORTED`（工程 wiring）；Cm utility 和抓取效果 `INCONCLUSIVE`

**产物与结果**

- [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209) — 完整 control smoke 目录。
- [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/run_manifest.json) — `COMPLETED / SUPPORTED` 终态。
- [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/config.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/config.json) — resolved config、输入 SHA 和 actor parity。
- [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/metrics.jsonl) — 2 行 epoch 指标。
- [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/train.log](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/train.log) — 完整训练日志。
- [outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_20260916_151209/checkpoint_validation.json) — model/optimizer/action finite、固定 sigma 和重载验证。

**验证结果**

- initial mean=`0`，sigma min/max=`0.099999994` 且不可训练；actor state mismatch 为空，Cm 后缀与跨 variant mean/sigma diff 均为 `0.0`。
- epoch 2：`c_loss=3.538330`、`a_loss=-0.002872`、`KL=0.017957`、`success_fraction=0.0625`、`residual_rms=0.105352`、saturation=`0.001736`，全部 finite。
- checkpoint epoch/frame=`2/4096`，model/optimizer/action finite，独立重载 action diff=`0.0`。

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_ppo_stability.py --variant control --max-epochs 2 --gpu 5 --seed 42 --num-envs 64 --run-id cmresidual_v19_control_smoke_20260916_151209 --activity-id ACT-20260916-151209-CMRESIDUAL-V191-CONTROL-SMOKE`

**停止边界**

只运行 2 epochs；若进程、finite、checkpoint、固定 sigma、actor parity 或重载合同失败，则记录终态并停止，不启动 critic-Cm smoke。
## 2026-09-16 15:49:00 +0800 — V1.9.2 critic-Cm smoke 复跑完成

- activity_id: `ACT-20260916-154900-CMRESIDUAL-V192-CRITICCM-SMOKE`
- timestamp: `2026-09-16 15:49:00 +0800`
- modification_version: `V1.9.2`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准继续修复与复跑；V1.9.2 control 已完成并通过 RNG、finite、固定 sigma 与 checkpoint 合同，按相同协议启动 critic-Cm。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`
- worktree_dirty: `true`（使用已批准但未提交的 V1.9.2 实现；无关 ObjectInteractionCm 用户差异未触碰）
- scope: GPU5、64 env、seed42、2 epochs、critic-Cm；actor 1442-D、critic 2005-D，固定候选构造顺序和全部科研变量与 V1.9.2 control 一致。
- run_id: `cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 15:59:14 +0800`
- last_step: `4096`
- last_epoch: `2 / 2`
- best_metric: checkpoint deterministic reload action max abs diff `0.0`
- checkpoint: [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/CmResidualV19CriticCmT2/nn/last_CmResidualSafeCriticCm_ep_2_rew__6.81_.pth](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/CmResidualV19CriticCmT2/nn/last_CmResidualSafeCriticCm_ep_2_rew__6.81_.pth)
- exit_reason: 完成严格 2-epoch V1.9.2 critic-Cm smoke；RNG、finite、固定 sigma、actor parity 和 checkpoint 重载合同全部通过。
- conclusion: `SUPPORTED`（V1.9.2 两侧静态 parity 与各自 GPU wiring）；跨独立 GPU 进程的逐 trajectory bitwise parity `INCONCLUSIVE`；Cm utility `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 当前指针更新为 V1.9.2。
- [src/task/CmResidual/docs/指导/V1.9.md](../指导/V1.9.md) 和 [src/task/CmResidual/docs/plan/V1.9.md](../plan/V1.9.md) — 本轮遵循的研究指导和最终计划。
- [src/task/CmResidual/docs/README.md](../README.md) — 更新 Gate II 终态和 GPU 非严格确定性边界。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) 和 [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — V1.9.1 失败 smoke、V1.9.2 修复复跑和结论边界。
- [third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) — 固定顺序构造 1442/2005-D critic 候选，恢复 post-build RNG parity。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml) 和 [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml) — 声明相同 `[1442, 2005]` critic 候选顺序。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — post-build RNG、stochastic action parity 硬 gate，2-epoch smoke 和 V1.9.2 manifest。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — V1.9.2 manifest 版本一致性；本轮未运行 evaluation。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — RNG state 与首轮 stochastic action 精确一致回归。

**原因**

V1.9.1 的 actor 参数相同不足以保证模型构造后 RNG 相同；不同 critic 宽度会消耗不同数量的初始化随机数。V1.9.2 在不改变 critic 实际输入、actor、reward、数据或训练变量的前提下，让两侧以固定候选顺序消耗相同 RNG，并以新 run_id 复跑 Gate II。两侧 GPU smoke 完成后仍保留分进程 trajectory 非 bitwise 一致的边界，不把该差异归因于 Cm。

**产物**

- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913) — V1.9.2 control 运行目录。
- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/metrics.jsonl)、[train.log](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/train.log)、[checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/checkpoint_validation.json) — control 终态证据。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900) — V1.9.2 critic-Cm 运行目录。
- [outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/metrics.jsonl)、[train.log](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/train.log)、[checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/checkpoint_validation.json) — critic-Cm 终态证据。

**验证**

- 静态 parity：actor state mismatch 为空，mean/sigma/Cm 后缀 action diff=`0.0`，post-build CPU RNG equal=`true`，首轮 stochastic action max abs diff=`0.0`。
- control / critic-Cm 均为 epoch/frame=`2/4096`，model/optimizer/action finite，sigma min/max=`0.099999994`，checkpoint reload diff=`0.0`。
- epoch 1：control / critic-Cm residual RMS 均为 `0.1032117531`、saturation 均为 `0`；success fraction=`0.1875/0.15625`，因此不声称实际 GPU trajectory bitwise 相同。
- epoch 2：control / critic-Cm `c_loss=3.707792/3.724522`、KL=`0.028788/0.024047`、success fraction=`0.03125/0.078125`、residual RMS=`0.105473/0.105276`；仅作 smoke 诊断，不比较效果。
- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`21 passed, 6 warnings`；warnings 均为 Hydra 1.1 兼容提示。
- 修改 Python 文件 `py_compile` 和 `git diff --check` 通过；没有残留 CmResidual 进程，未启动 B0/T10/E10。

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_ppo_stability.py --variant critic_cm --max-epochs 2 --gpu 5 --seed 42 --num-envs 64 --run-id cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900 --activity-id ACT-20260916-154900-CMRESIDUAL-V192-CRITICCM-SMOKE`

**停止边界**

只运行 2 epochs；完成后核对单侧 checkpoint 合同与两侧 epoch-1 初始 trajectory 指标。无论结果如何均不自动进入 T10/E10。

## 2026-09-16 15:39:13 +0800 — V1.9.2 RNG parity 修复并启动 control smoke 复跑

- activity_id: `ACT-20260916-153913-CMRESIDUAL-V192-CONTROL-SMOKE`
- timestamp: `2026-09-16 15:39:13 +0800`
- modification_version: `V1.9.2`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: V1.9.1 smoke 暴露 critic 宽度导致 post-build RNG 不一致后，用户明确要求继续；限定修复 RNG parity 并用新 run_id 重跑 Gate II，不进入 T10/E10。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`
- worktree_dirty: `true`（使用已批准但未提交的 V1.9 系列实现；无关 ObjectInteractionCm 用户差异未触碰）
- scope: 两侧按固定 `[1442, 2005]` 顺序构造 critic 候选并选择各自输入宽度，使构造后的 CPU RNG 与首轮 stochastic action 严格一致；随后在 GPU5、64 env、seed42、2 epochs 下先复跑 control。
- run_id: `cmresidual_v19_control_smoke_rngfix_20260916_153913`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 15:48:53 +0800`
- last_step: `4096`
- last_epoch: `2 / 2`
- best_metric: checkpoint deterministic reload action max abs diff `0.0`
- checkpoint: [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/CmResidualV19ControlT2/nn/last_CmResidualSafeCriticControl_ep_2_rew__7._.pth](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/CmResidualV19ControlT2/nn/last_CmResidualSafeCriticControl_ep_2_rew__7._.pth)
- exit_reason: 完成严格 2-epoch V1.9.2 control smoke；RNG、finite、固定 sigma、actor parity 和 checkpoint 重载合同全部通过。
- conclusion: `SUPPORTED`（V1.9.2 control 工程 wiring 与静态 RNG parity）；Cm utility `INCONCLUSIVE`

**产物**

- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913) — 完整运行目录。
- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/run_manifest.json) — `COMPLETED / SUPPORTED` 终态及 RNG parity 合同。
- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/config.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/config.json) — resolved config 与输入身份。
- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/metrics.jsonl) — 2 行 epoch 指标。
- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/train.log](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/train.log) — 完整训练日志。
- [outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/checkpoint_validation.json) — finite、sigma 与重载证据。

**运行结果**

- epoch 1：`c_loss=1.130573`、success fraction=`0.1875`、residual RMS=`0.103212`。
- epoch 2：`c_loss=3.707792`、`KL=0.028788`、success fraction=`0.03125`、residual RMS=`0.105473`、saturation=`0.001736`。
- checkpoint epoch/frame=`2/4096`，model/optimizer/action finite，sigma min/max=`0.099999994`，reload diff=`0.0`。

**实现与启动前验证**

- 两侧 post-build CPU RNG state 完全相等，首轮 stochastic action max abs diff=`0.0`；actor state、mean、sigma 和 Cm 后缀不变性继续为 `0.0` mismatch。
- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`21 passed, 6 warnings`；warnings 均为 Hydra 1.1 兼容提示。
- 修改 Python 文件 `py_compile` 与 `git diff --check` 通过；GPU5 启动前为 `8 MiB / 0%`，无残留 CmResidual 进程。

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_ppo_stability.py --variant control --max-epochs 2 --gpu 5 --seed 42 --num-envs 64 --run-id cmresidual_v19_control_smoke_rngfix_20260916_153913 --activity-id ACT-20260916-153913-CMRESIDUAL-V192-CONTROL-SMOKE`

**停止边界**

只运行 2 epochs；单侧合同失败则停止。control 通过后才启动 critic-Cm；两侧结果仍只作 wiring 与配对协议验证，不判断 Cm 效果。
## 2026-09-16 16:03:39 +0800 — V1.9.2 实现、RNG 修复与 Gate II 证据提交交接

- activity_id: `ACT-20260916-160339-CMRESIDUAL-V192-COMMIT`
- timestamp: `2026-09-16 16:03:39 +0800`
- modification_version: `V1.9.2`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准 V1.9 critic-only Cm 计划、实现、Gate II smoke 与 RNG parity 修复，并明确要求提交代码。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `c619a99f579bd1eef209073e1b8b5c660e71a5e6`
- worktree_dirty: `true`（包含本次 CmResidual V1.9.2 全部已批准差异及无关 ObjectInteractionCm 用户修改；后者明确排除）
- scope: 汇总提交 V1.9 指导/最终 plan、critic-only Cm 非对称 actor/critic 实现、匹配配置、runner/evaluator、RNG parity 修复、Task 测试、版本指针和 Gate II 活动/实验记录；不提交 outputs、checkpoint、cache 或其他 Task 差异。
- conclusion: `SUPPORTED`（静态 actor/RNG parity 与两侧 GPU wiring/checkpoint）；跨独立 GPU 进程 trajectory bitwise parity、Cm utility、抓取改善与统计显著性 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新为 V1.9.2。
- [src/task/CmResidual/docs/指导/V1.9.md](../指导/V1.9.md) — 用户提供的研究指导。
- [src/task/CmResidual/docs/plan/V1.9.md](../plan/V1.9.md) — 用户确认的最终执行计划。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前实现、Gate II 终态和结论边界。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — V1.9 实现、V1.9.1 失败 smoke、V1.9.2 修复复跑与提交交接。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 两轮 Gate II 的假设、证据与解释边界。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — actor/critic 输入、RNG、stochastic action、预算和兼容回归。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — V1.9 variants、2-epoch gate、manifest、checkpoint 与 parity preflight。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — V1.9 deterministic evaluation 与 episode-return 记录入口。
- [third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) — 非对称输入与固定候选构造顺序。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticControlPPO.yaml) — matched control 配置。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualSafeCriticCmPPO.yaml) — critic-only Cm 配置。

**原因**

V1.8 只证明安全 residual 的短程保持能力。V1.9.2 将 Cm 唯一引入 critic，并通过相同环境路径、相同 actor、固定 sigma、固定 critic 候选构造顺序和显式 RNG/action parity gate 控制混杂变量。Gate II 证明两侧各自可运行，但尚未运行 B0/T10/E10，不能形成 Cm 效果结论。

**验证**

- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`21 passed, 6 warnings`；warnings 均为 Hydra 1.1 兼容提示。
- 修改 Python 文件 `py_compile`、`git diff --check` 通过；actor state mismatch 为空，post-build CPU RNG equal=`true`，首轮 stochastic action diff=`0.0`。
- V1.9.2 control 与 critic-Cm run 均为 `COMPLETED / SUPPORTED`，epoch/frame=`2/4096`，model/optimizer/action finite，sigma 约 `0.1`，checkpoint reload diff=`0.0`。
- control [manifest](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v19_control_smoke_rngfix_20260916_153913/checkpoint_validation.json)。
- critic-Cm [manifest](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v19_critic_cm_smoke_rngfix_20260916_154900/checkpoint_validation.json)。
- 提交前只暂存上述显式路径并执行 staged activity/link audit；不把 smoke 指标误报为 Cm 效果。

**保护边界与回滚**

- 未修改 OI-Cm 模型/checkpoint、action mapper、reward、residual scale、reset、坐标系、数据/cache/schema、共享 `src/base/`、外部 DExplore、旧 outputs 或其他 Task；未提交生成 checkpoint。
- 本次单一提交可用 `git revert <commit>` 回滚；ignored outputs 证据保持独立，不随代码回滚删除。
## 2026-09-16 17:01:59 +0800 — V1.10.1 计划定稿与 Gate 0 通过

- activity_id: `ACT-20260916-170159-CMRESIDUAL-V1101-GATE0`
- timestamp: `2026-09-16 17:01:59 +0800`
- modification_version: `V1.10.1`
- operation_category: `code`、`diagnostic`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户要求按 V1.10 指导继续，审阅计划草案后明确确认；本阶段只定稿计划、补齐运行版本追溯并执行 Gate 0，未启动 GPU 正式实验。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（包含用户提供的 V1.10 指导、本次 CmResidual 差异及无关 ObjectInteractionCm 用户修改；后者未触碰、未纳入范围）
- scope: 定稿 V1.10 seed42 B0/T10/E10 计划；为既有 V1.9 critic-only Cm runner/evaluator 增加显式 `V1.10.1` provenance，同时保留 V1.8/V1.9.2 默认解释；完成 Task-local Gate 0，不改变模型、RNG、训练变量、数据、指标或仿真协议。
- conclusion: `SUPPORTED`（计划内版本追溯、合同测试与 Gate 0 工程条件）；B0/T10/E10、Cm utility、抓取改善和统计显著性 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 当前指针更新为 V1.10.1。
- [src/task/CmResidual/docs/指导/V1.10.md](../指导/V1.10.md) — 用户提供的 V1.10 研究指导，未由 Agent 改写。
- [src/task/CmResidual/docs/plan/V1.10.md](../plan/V1.10.md) — 用户确认的最终 B0/T10/E10 执行计划、停止条件和结论边界。
- [src/task/CmResidual/docs/README.md](../README.md) — 更新当前阶段、计划入口和 Gate 0 状态。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 记录本次修改、验证和保护边界。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) — 允许 V1.10 control/critic-Cm 训练显式写入 `V1.10.1`，旧入口默认版本保持不变。
- [src/task/CmResidual/tools/eval_residual_stability.py](../../tools/eval_residual_stability.py) — 集中校验 variant 与允许的 provenance 版本，V1.10 评估须显式传入 `V1.10.1`。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — 覆盖 V1.10 显式版本、V1.8/V1.9.2 legacy 默认和非法组合拒绝。

**原因**

V1.10 只执行已通过 wiring smoke 的 critic-only Cm 对照，不应为新运行继续沿用 V1.9.2 manifest 版本，也不能通过直接替换常量破坏旧命令的默认解释。显式版本参数将运行 provenance 与算法/实验配置解耦，并在启动前拒绝 V1.8 与 V1.10.1 等非法组合。

**验证**

- `PYTHONPATH=third_party/IsaacGymEnvs /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests`：`22 passed, 6 warnings`；warnings 均为现有 Hydra 1.1 兼容提示。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmResidual/tools/run_ppo_stability.py src/task/CmResidual/tools/eval_residual_stability.py`：通过。
- 两个工具的 `--help` 均显示 `--modification-version`；合同测试确认 control/critic-Cm 可显式解析为 V1.10.1，V1.8/V1.9.2 默认值不变，非法版本组合抛出错误。
- V1.9.2 的 actor/critic 维度、actor state parity、Cm 后缀不变性、post-build RNG 与首轮 stochastic action parity 回归通过；没有修改 builder 或训练配置。
- `git diff --check`：通过；GPU5 查询为 `6 MiB / 24576 MiB`、利用率 `0%`，无计算进程；无遗留 CmResidual runner/evaluator/train 进程。
- 本阶段没有训练、评估或 simulator rollout；工程 Gate 0 通过不构成 Cm 效果证据。

**保护边界与回滚**

- 未修改 actor/critic 构造、critic candidate 顺序、PPO 超参数、reward、residual scale、OI-Cm/DExplore checkpoint、数据/schema、共享 `src/base/`、ObjectInteractionCm、旧 output 或 checkpoint。
- B0、A/B T10/E10、T20/E20、多 seed 和 OI-Cm V2 均未启动；outputs/checkpoint/cache 未生成或覆盖。
- 回滚本次未提交的 V1.10.1 Task-local 工具、测试、版本指针、README、plan 和活动记录即可恢复 V1.9.2；用户提供的指导和无关 ObjectInteractionCm 差异保持不变。
## 2026-09-16 17:06:51 +0800 — V1.10.1 fresh B0-Cm-path 启动

- activity_id: `ACT-20260916-170651-CMRESIDUAL-V1101-B0`
- timestamp: `2026-09-16 17:06:51 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认 V1.10 最终计划并明确要求开始实验；Gate 0 已通过，按固定顺序先执行 fresh B0-Cm-path。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（使用已批准但未提交的 V1.10.1 provenance 实现；无关 ObjectInteractionCm 用户差异未触碰）
- scope: GPU5、64 env、seed42、367 steps、2005-D Cm 环境路径、control variant、严格全零 18-D residual；只建立 A/B 共用 preservation baseline，通过后才允许 T10。
- run_id: `cmresidual_v110_b0_cm_path_20260916_170651`
- run_status: `STARTED`
- output: [outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651) — `PENDING`
- manifest: [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json) — `PENDING`
- metrics: [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/metrics.jsonl) — `PENDING`
- log: [eval.log](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/eval.log) — `PENDING`
- conclusion: `INCONCLUSIVE`（运行中）

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/eval_residual_stability.py --variant control --modification-version V1.10.1 --gpu 5 --seed 42 --num-envs 64 --steps 367 --run-id cmresidual_v110_b0_cm_path_20260916_170651 --activity-id ACT-20260916-170651-CMRESIDUAL-V1101-B0`

**停止条件**

非正常退出、non-finite、zero-target parity 失败、lift/contact baseline 非正或协议漂移时立即停止，不启动 T10。
## 2026-09-16 17:19:53 +0800 — DexYCB base-policy 评估可行性只读诊断

- activity_id: `ACT-20260916-171953-CMRESIDUAL-DEXYCB-DIAGNOSTIC`
- timestamp: `2026-09-16 17:19:53 +0800`
- modification_version: `V1.10.1`
- operation_category: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求在不停止当前 B0 的情况下确认本地 DexYCB 数据以及用它观察 base 策略效果是否存在歧义；仅执行只读盘点和合同分析。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（当前 V1.10.1 已批准差异、运行活动记录及无关 ObjectInteractionCm 用户修改；未修改数据、代码或正在运行的 B0）
- scope: 核对本地 DexYCB raw/cache/split、已有 Cm 跨数据集评估和当前 frozen DExplore/CmResidual 输入与物理资产合同；不创建适配器、reference、仿真资产、实验计划或新运行。
- concurrent_run_id: `cmresidual_v110_b0_cm_path_20260916_170651`
- concurrent_run_status: `STARTED`（进程继续运行，本诊断未发送停止或交互信号）
- conclusion: `SUPPORTED`（本地 subject-10/right DexYCB 数据与修复 cache 可用、现有 cache 不能直接作为 DExplore 物理 rollout 输入）；DexYCB base-policy 效果 `INCONCLUSIVE`

**证据**

- [data/raw_data/DexYCB](../../../../../data/raw_data/DexYCB) — 本地 raw 根约 22 GB，当前 capture 根为 subject-10，并包含 YCB object meshes/XML 和 calibration。
- [data/processed_data/stage4/data/dexycb/meta.json](../../../../../data/processed_data/stage4/data/dexycb/meta.json) — 修复版 Stage4 cache：50 条 right-hand sequence、2853 帧、30 Hz、reference-camera/world frame。
- [data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json](../../../../../data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json) — 50 条序列全部用于既有 held-out test；train 仅为 loader 占位。
- [process/DexYCB/raw.py](../../../../../process/DexYCB/raw.py) 和 [process/DexYCB/stage4_cm.py](../../../../../process/DexYCB/stage4_cm.py) — 当前修复版 xyzw/SE(3)/MANO PCA/right-hand 转换入口。
- [src/task/Cm/docs/logs/experiment_log.md](../../../Cm/docs/logs/experiment_log.md) — EXP-010 已评估 Cm object-flow 泛化，不是 DExplore 控制策略 rollout。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidual.yaml) — 当前物理任务锁定 Inspire、airplane asset、airplane reference/source tensor。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/base_policy.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/base_policy.py) — frozen DExplore actor 要求 1442-D Inspire observation 和原 checkpoint normalization。

**原因**

DexYCB cache 提供人手 MANO、物体几何/位姿和 flow，不提供可直接执行的 Inspire 18-D action、DExplore 428-D reference 或 598-D source tensor。要判断 base 物理效果，至少需要冻结地定义：base 是 frozen DExplore zero-residual；选择哪些 subject-10/right 序列与物体；如何将 MANO wrist/fingers retarget 到 Inspire；如何将 YCB mesh/XML 接入 Isaac Gym 并定义桌面/reset；以及评价 reference tracking、接触、抬升还是任务成功。不同选择会改变 GT、坐标、资产和结论，属于新的 L2 数据/实验计划边界。

**验证**

- `du -sh data/raw_data/DexYCB data/processed_data/stage4/data/dexycb`：raw/cache 分别约 `22G/383M`。
- cache `meta.json` 与 `manifest.csv` 核对为 50 条 sequence、50 个 right stream、2853 帧；首个 shared NPZ 为 schema `3.0.0`，包含逐帧 `obj_points_world [T,4096,3]`，不是 DExplore source tensor。
- split JSON 明确 `num_test_streams=50`、`train_placeholder_only=true`；现有 Cm EXP-010 的 `14.92 mm` 是 object-flow EPE，不是 frozen base 的物理抓取结果。
- 当前 `CmResidual.yaml`、`ReferenceProvider` 与 `InspireDExplorePolicy` 静态核对确认 airplane asset/reference/source 和 1442-D observation 合同均为硬约束。

**保护边界**

- 未改 DexYCB raw/cache/split、CmResidual/DExplore/OI-Cm、配置、指标、checkpoint 或仿真资产；未启动 DexYCB 新运行。
- 当前 V1.10 B0 保持独立运行，不因本诊断改变其 GPU、进程、manifest 或停止条件。

## 2026-09-16 17:42:20 +0800 — V1.11.1 DexYCB base-only 实现、离线构建与单次 Gate C

- activity_id: `ACT-20260916-174041-CMRESIDUAL-V1111-DEXYCB`
- timestamp: `2026-09-16 17:42:20 +0800`
- modification_version: `V1.11.1`
- operation_category: `code`、`data`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户确认 frozen DExplore、严格 zero residual、固定 `subject-10/20201022_110806/right` 与
  `002_master_chef_can`、隔离配置/资产/output、GPU6 单次评估，并要求直接实施且不影响当前 V1.10 轨迹。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（包含已批准的 V1.10.1/V1.11.1 CmResidual 差异及无关 ObjectInteractionCm 用户修改；后者未触碰）
- scope: 汇总当前未提交的已批准 V1.10.1 provenance 差异，并新增 V1.11.1 指导/最终计划、DexYCB builder、
  独立 task config/alias、task actor-name 与 reference-tip metric、zero-residual evaluator、测试、版本/入口/活动/实验记录；
  生成 reference/assets 与 GPU output 独立留存，不改 canonical airplane 配置和正在运行的 V1.10 B0。
- run_id: `cmresidual_v111_dexycb_base_20260916_174041`
- run_status: `COMPLETED`
- last_step / last_epoch: `72 / null`
- best_metric: `min_tip_distance_m=0.3027995825`（仅描述量，Gate 无效时不得解释为效果）
- checkpoint: `null`（冻结 DExplore checkpoint 只读；不训练 residual/Cm）
- exit_reason: 完成固定 72-step 预算；initial wrist alignment gate 失败后按计划停止，不改协议重跑。
- conclusion: `INVALID_IMPLEMENTATION`（GPU reset wrist/FK 对齐）；DexYCB base 效果 `INCONCLUSIVE`

**实现与数据**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新为 V1.11.1。
- [src/task/CmResidual/docs/README.md](../README.md) 与 [experiment_log.md](experiment_log.md) — 当前状态和正式实验解释边界。
- [src/task/CmResidual/docs/指导/V1.10.md](../指导/V1.10.md) 与 [V1.10 最终计划](../plan/V1.10.md) — 当前 worktree 中一并保留的上一阶段已批准输入/计划。
- [src/task/CmResidual/docs/指导/V1.11.md](../指导/V1.11.md) 与 [最终计划](../plan/V1.11.md) — 已确认研究边界和停止条件。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py)、[eval_residual_stability.py](../../tools/eval_residual_stability.py) 与 [test_reference_contract.py](../../tests/test_reference_contract.py) — V1.10.1 已批准 provenance 支持及回归。
- [src/task/CmResidual/tools/data/build_dexycb_base_reference.py](../../tools/data/build_dexycb_base_reference.py) — 固定 point identity、统一 SE(3)、Inspire q6 retarget、约束投影、598-D compatibility source 和独立资产构建。
- [src/task/CmResidual/tools/eval_dexycb_base.py](../../tools/eval_dexycb_base.py) — GPU6/seed42/4 env/72 steps/strict-zero evaluator 与终态 manifest。
- [src/task/CmResidual/tests/test_dexycb_base.py](../../tests/test_dexycb_base.py) — tip identity、wrist round-trip、独立 config、padded source 和生成 URDF 合同。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml) — 1442-D、OI-Cm disabled、独立 reference/asset 配置；canonical airplane config 未改。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py) 与 [cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 新 task alias、配置化 object actor name 和只读 tracking metric；默认 airplane 行为不变。
- [data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806/manifest.json](../../../../../data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806/manifest.json) — Gate A、输入 SHA、坐标变换、point identity 与 retarget 指标。
- [data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806/run_manifest.json](../../../../../data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806/run_manifest.json) — 离线构建终态。

**原因**

DexYCB Stage4 cache 只有 MANO/object 几何与位姿，不能直接满足 frozen DExplore 的 Inspire native q、428-D
reference、598-D source 和 Isaac 物理资产合同。本实现用隔离 adapter 补齐这些输入，同时把该 artifact 锁为
evaluation-only；单次 Gate C 的 reset hard gate 失败后停止，避免把 wiring/初始化错误误报为跨数据集效果。

**验证**

Gate A/B：

- 72 帧 reference 与前后 padding 后 `[74,598]` compatibility source 均 finite；frame interval=`[1,72]`，
  `training_eligible=false`、`evaluation_eligible=true`。
- retarget tip RMS=`0.0323858 m`，neutral RMS=`0.0475794 m`，改善=`31.9332%`；joint limit、mimic、
  `1 rad/s`、`20 rad/s²`、首帧 object z-up 回代全部 PASS。该门只支持离线 adapter 工程合同。
- `pytest` 覆盖 CmResidual reference/DexYCB adapter/raw：`32 passed, 9 warnings`；warnings 为既有
  NumPy/Hydra 迁移提示。Python compile、工具 `--help`、Hydra runtime resolve、XML parse 与 `git diff --check` 通过。

**Gate C 终态证据**

- [outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041) — 完整隔离输出。
- [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/run_manifest.json) — `COMPLETED / INVALID_IMPLEMENTATION` 唯一终态。
- [config.json](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v111_dexycb_base_20260916_174041/eval.log) — resolved config、72 行指标与日志。
- native q/object/table 初始误差=`0 / 1.11e-16 m / 0`，但 wrist body/FK 初始误差=`0.486826 m`；
  zero residual target delta=`0` 且 observation/target/object pose 全程 finite。只读定位显示离线首帧 wrist
  translation norm=`0.504820 m`，第一物理步 root reference error 已降为 `0.052096 m`，说明阻断点位于
  reset 后 rigid-body/FK 初态同步合同，不允许用本 run 的 lift/contact/tracking 数值评价 base 泛化。

**保护与回滚**

- 未修改 canonical `CmResidual.yaml`、DExplore/OI-Cm checkpoint、base observation/action mapping、reward、
  raw DexYCB、Stage4 cache/schema、共享 `src/base`、V1.10 runner 或其 GPU5 运行。
- 当前 V1.10 B0 进程仍运行；V1.11 只使用 GPU6 和独立目录。按 Gate C 失败停止条件，没有修复后复跑、
  更换轨迹/物体/阈值/checkpoint、训练 PPO/Cm 或创建多 seed 结果。
- 代码回滚入口为 V1.11 新 builder/evaluator/config/alias/test、task.py actor-name/metric 最小差异及本次文档；
  已生成 data/output 是独立证据，未删除且不纳入 Git。

## 2026-09-16 18:17:53 +0800 — V1.11.2 DexYCB reset 同步修复与复跑前闸门

- activity_id: `ACT-20260916-181753-CMRESIDUAL-V1112-RESET-FIX`
- timestamp: `2026-09-16 18:17:53 +0800`
- modification_version: `V1.11.2`
- operation_category: `code`、`diagnostic`、`operation`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户在 V1.11.1 reset hard gate 失败后明确要求先修复再继续；保持同一序列、物体、reference、
  checkpoint、seed、env 数和步数，只修 reset 初始化同步。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（保留已批准 V1.10/V1.11 差异及无关 ObjectInteractionCm 用户修改；后者未触碰）
- scope: 仅为 DexYCB asset config 增加 opt-in actor-creation DOF 初始化，在 actor 创建时写入与 reset 相同的
  native q/dq 和 position target；canonical airplane config 未声明该开关，旧路径保持原行为。用 1 env GPU6
  探针验证初态，不生成正式 run；因 GPU6 随后被外部任务占用，正式 V1.11.2 Gate C 尚未启动。
- formal_rerun: `not_started`（等待 GPU 选择确认，尚无 run_id/run_status）
- concurrent_run_id: `cmresidual_v110_b0_cm_path_20260916_170651`
- concurrent_run_status: `COMPLETED / SUPPORTED`（V1.11 未干预）
- conclusion: `SUPPORTED`（reset 同步修复和定向验证）；DexYCB base 效果 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新为 V1.11.2。
- [src/task/CmResidual/docs/README.md](../README.md) — 更新修复、资源和复跑等待状态。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 补记 V1.10 B0 终态；V1.11.1 无效运行解释不变。
- [src/task/CmResidual/tools/eval_dexycb_base.py](../../tools/eval_dexycb_base.py) — 后续新 run 写入 V1.11.2 provenance。
- [src/task/CmResidual/tests/test_dexycb_base.py](../../tests/test_dexycb_base.py) — 验证 DexYCB opt-in、canonical config 不启用。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml) — 仅本变体启用 `initializeDofsAtCreation=true`。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — opt-in actor 创建初态写入；默认值为 false。

**原因**

V1.11.1 在 tensor reset 后立即读取的 rigid-body pose 仍是 actor 创建时零 DOF 姿态；等到第一次 simulate
才更新 FK，但 reference velocity/碰撞已使状态偏离。把批准的 initial q/dq 在 actor 创建时写入，可让 prepare_sim
建立正确初态，同时避免为 canonical airplane 路径新增隐式 warm-up step 或改变 episode 预算。

**验证**

- 修复前 1-env GPU6 探针：reset wrist error=`0.4868260920 m`；一次/twice simulate 后为
  `0.0205058455 / 0.0255725142 m`，证明不能用额外 physics step 充当精确 reset。
- 修复后同一探针：actual root=`[-0.23049173,-0.05781597,0.44539154]`，FK expected=
  `[-0.23049177,-0.05781598,0.44539157]`，error=`5.6864e-08 m`。
- `pytest -q src/task/CmResidual/tests/test_dexycb_base.py src/task/CmResidual/tests/test_reference_contract.py tests/test_dexycb_raw.py`：
  `32 passed, 9 warnings`；warnings 为既有 NumPy/Hydra 迁移提示。
- 修改 Python 文件 `py_compile` 与 `git diff --check` 通过。
- GPU6 当前由外部 PID `1908027/1908028` 占用约 `4.1 GiB/card` 且利用率约 `64%/73%`；未启动正式复跑。

**保护与回滚**

- 未改轨迹、物体、坐标、reference/source、checkpoint、seed、reward、observation/action、residual、Cm、
  canonical `CmResidual.yaml`、V1.10 runner 或生成数据；未停止外部 GPU 任务。
- 删除 opt-in config 字段及 task.py 对应创建时初始化块即可回滚修复；V1.11.1 失败证据保持只读。

## 2026-09-16 18:28:09 +0800 — V1.11.2 DexYCB GPU5 Gate C 完成

- activity_id: `ACT-20260916-182728-CMRESIDUAL-V1112-DEXYCB-GPU5`
- timestamp: `2026-09-16 18:28:09 +0800`
- modification_version: `V1.11.2`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change -> run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户批准 reset 修复后继续，并在 GPU6 被外部双卡训练占用、V1.10 B0 已完成后明确批准
  将同一 V1.11.2 正式协议迁移到空闲 GPU5。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（使用已批准 V1.10/V1.11 实现；无关 ObjectInteractionCm 用户修改未触碰）
- scope: 更新 V1.11 最终计划中的 GPU 资源修订，将 evaluator 锁为 GPU5/V1.11.2，完成定向回归后以新
  run_id 运行 frozen DExplore、4 env × 72 steps、strict zero residual；不覆盖 V1.11.1 失败证据。
- run_id: `cmresidual_v1112_dexycb_base_gpu5_20260916_182728`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 18:27:50 +0800`
- last_step / last_epoch: `72 / null`
- best_metric: `min_tip_distance_m=0.2998754382`（描述量）
- checkpoint: `null`（frozen DExplore 只读）
- exit_reason: 完成批准的 V1.11.2 72-step GPU5 deterministic evaluation budget。
- conclusion: `SUPPORTED`（工程 Gate C）；固定单轨迹成功抓取 `REFUTED`；DexYCB 泛化 `INCONCLUSIVE`

**文件**

- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 当前指针保持 V1.11.2。
- [src/task/CmResidual/docs/plan/V1.11.md](../plan/V1.11.md) — 记录用户批准的 GPU6→GPU5 资源替换；其余协议不变。
- [src/task/CmResidual/tools/eval_dexycb_base.py](../../tools/eval_dexycb_base.py) — 锁定 GPU5 与 V1.11.2 provenance。
- [src/task/CmResidual/docs/README.md](../README.md) 与 [experiment_log.md](experiment_log.md) — 更新工程和行为结论边界。
- [third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml)、[cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) 与 [test_dexycb_base.py](../../tests/test_dexycb_base.py) — 本次运行使用的 opt-in reset 修复及回归。

**原因**

V1.11.1 因 reset rigid-body/FK 不同步无法解释。V1.11.2 修复后必须用同一数据和 checkpoint 重新执行完整
物理 gate；GPU 迁移只解决资源冲突，不改变研究变量。工程 gate 通过后，接触、距离与 tracking 数据才可用于
回答这条固定 trajectory 是否观察到抓取，但单轨迹仍不能外推整个 DexYCB。

**验证**

- `pytest -q src/task/CmResidual/tests/test_dexycb_base.py src/task/CmResidual/tests/test_reference_contract.py tests/test_dexycb_raw.py`：
  `32 passed, 9 warnings`；Python `py_compile`、evaluator `--help` 和 `git diff --check` 通过。
- 启动前 GPU5=`6 MiB / 0%` 且无 CmResidual 进程；运行后恢复为 `6 MiB / 0%`。
- initial native q/wrist/object/table error=`0 / 1.1733e-07 m / 1.11e-16 m / 0`；72 行 finite，
  max residual target delta=`0`，manifest gate 四项全部通过。
- [outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728)、
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/run_manifest.json)、
  [config.json](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/config.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/metrics.jsonl)、
  [eval.log](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/eval.log) 均存在并可导航。

**保护与回滚**

- 未改 trajectory、object、坐标、reference/source、checkpoint、seed、env/step budget、reward、base observation/
  action、residual/Cm 或 canonical airplane config；未写入 V1.10 B0 和 V1.11.1 output。
- 回滚 V1.11.2 时删除 opt-in reset block/field并恢复 evaluator GPU 锁即可；生成 outputs 作为审计证据保留。

## 2026-09-16 18:39:34 +0800 — V1.10.1 control A-T10 启动

- activity_id: `ACT-20260916-183934-CMRESIDUAL-V1101-CONTROL-T10`
- timestamp: `2026-09-16 18:39:34 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确要求继续原 V1.10 实验；B0-Cm-path 已 `COMPLETED / SUPPORTED`，按最终计划固定顺序进入 A-control T10。
- skills_used: `research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（使用已批准但未提交的 V1.10 provenance/V1.9.2 critic-only 实现，并保留 V1.11 差异；无关 ObjectInteractionCm 修改未触碰）
- scope: GPU5、64 env、seed42、control variant、2005-D environment/1442-D actor/1442-D critic、10 epochs，
  从零初始化且不加载任何 smoke/V1.8/residual checkpoint；本阶段只运行 A-T10，完成并通过后才允许 B-T10。
- run_id: `cmresidual_v110_control_t10_20260916_183934`
- run_status: `STARTED`
- output: [outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934) — `PENDING`
- manifest: [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/run_manifest.json) — `PENDING`
- metrics: [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/metrics.jsonl) — `PENDING`
- train_log: [train.log](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/train.log) — `PENDING`
- checkpoint_validation: [checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/checkpoint_validation.json) — `PENDING`
- conclusion: `INCONCLUSIVE`（运行中）

**命令**

`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_ppo_stability.py --variant control --modification-version V1.10.1 --max-epochs 10 --gpu 5 --seed 42 --num-envs 64 --run-id cmresidual_v110_control_t10_20260916_183934 --activity-id ACT-20260916-183934-CMRESIDUAL-V1101-CONTROL-T10`

**原因**

B0 已满足 zero-residual parity、正 lift/contact baseline 和协议 SHA gate。V1.10 固定顺序要求先训练 control A，
防止并发资源竞争和事后选择；训练 loss/success 仅作诊断，不在 B-T10/E10 完成前形成 Cm utility 结论。

**验证**

- B0：`367 steps / COMPLETED / SUPPORTED`，mean-env-max lift=`0.0848945 m`、mean contact=`0.2258515`、max residual target delta=`0`。
- 启动前相关回归最近结果为 `32 passed, 9 warnings`，runner `--help`、Python compile 与 `git diff --check` 通过。
- GPU5 启动前=`6 MiB / 0%`，无遗留 CmResidual 进程和 V1.10 T10/E10 输出目录。

**停止条件**

非正常退出、epoch/frame 不等于 10/20480、non-finite、sigma 漂移/可训练、checkpoint reload diff `>1e-6`、
variant/schema/RNG/action parity 或输入 SHA 漂移时停止，不启动 B-T10。

## 2026-09-16 19:26:29 +0800 — V1.10.1 control A-T10 完成

- activity_id: `ACT-20260916-183934-CMRESIDUAL-V1101-CONTROL-T10`
- timestamp: `2026-09-16 19:26:29 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确要求继续原 V1.10 实验；本终态对应已批准最终计划中的 A-control T10。
- skills_used: `research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（沿用启动时已批准实现；本次训练未修改代码或配置，且未触碰 ObjectInteractionCm 用户修改）
- scope: GPU5、64 env、seed42、control variant、2005-D environment/1442-D actor/1442-D critic、10 epochs，
  从零初始化，不加载历史 residual checkpoint。
- run_id: `cmresidual_v110_control_t10_20260916_183934`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 19:26:29 +0800`
- last_step / last_epoch: `20480 / 10`
- best_metric: `deterministic_reload_max_abs_diff=0`（工程门禁指标）
- checkpoint: [last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth)
- checkpoint_sha256: `fdcf7e00f3d8f250686c4fb30d3527577db5eb9fbfe2e6db39be29f8b6009278`
- exit_reason: 达到批准的 10 epoch / 20480 frame 预算并正常退出。
- conclusion: `SUPPORTED`（A-T10 工程门禁）；Cm utility `INCONCLUSIVE`

**原因**

A-T10 已满足全部工程 gate，固定顺序要求使用 matched actor initialization 训练仅 critic 可见 Cm 的 B 变体；
本条将已完成运行落为终态，避免界面等待中断被误解为训练进程终止。

**验证**

- [outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934)、
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/metrics.jsonl)、
  [train.log](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/train.log)、
  [checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/checkpoint_validation.json) 均存在。
- model、optimizer、deterministic action 均 finite；reload max-abs diff=`0`；sigma min/max=`0.099999994` 且由 safe contract 冻结。
- safe policy contract、matched actor initialization、schema/SHA/RNG/首个随机动作 parity 全部通过；共写入 10 条 epoch metrics。
- epoch 1→10 的诊断量：critic loss `1.17394→0.0655944`，KL `0.003009→0.00869283`，success
  `0.1875→0.015625`，residual RMS `0.103212→0.103786`；训练量不用于单独判断 Cm 效果。

**保护与回滚**

- 未改 reward、observation/action、reference、数据 split、checkpoint 解释或 B/E 阶段参数；未覆盖既有 output。
- 回滚入口为保留本次 manifest/checkpoint 作审计，不在后续 B/E 命令中引用 A checkpoint 以外的未批准产物。

## 2026-09-16 19:27:29 +0800 — V1.10.1 critic-Cm B-T10 启动

- activity_id: `ACT-20260916-192729-CMRESIDUAL-V1101-CRITIC-CM-T10`
- timestamp: `2026-09-16 19:27:29 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: A-control T10 已 `COMPLETED / SUPPORTED`，按 V1.10 最终计划固定顺序进入 B-critic-Cm T10。
- skills_used: `research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（使用与 A 相同的已批准实现与输入；无关 ObjectInteractionCm 用户修改未触碰）
- scope: GPU5、64 env、seed42、critic_cm variant、2005-D environment/1442-D actor/2005-D critic、10 epochs；
  与 A matched actor initialization，从零初始化且不加载 A 或历史 residual checkpoint。
- run_id: `cmresidual_v110_critic_cm_t10_20260916_192729`
- run_status: `STARTED`
- output: [outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729) — `PENDING`
- manifest: [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/run_manifest.json) — `PENDING`
- metrics: [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/metrics.jsonl) — `PENDING`
- train_log: [train.log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/train.log) — `PENDING`
- checkpoint_validation: [checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/checkpoint_validation.json) — `PENDING`
- conclusion: `INCONCLUSIVE`（运行中）

**命令**

`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_ppo_stability.py --variant critic_cm --modification-version V1.10.1 --max-epochs 10 --gpu 5 --seed 42 --num-envs 64 --run-id cmresidual_v110_critic_cm_t10_20260916_192729 --activity-id ACT-20260916-192729-CMRESIDUAL-V1101-CRITIC-CM-T10`

**启动检查与停止条件**

- GPU5 启动前=`6 MiB / 0%` 且无遗留 CmResidual 进程；A checkpoint validation 全部通过。
- 非正常退出、epoch/frame 不等于 10/20480、non-finite、sigma 漂移/可训练、checkpoint reload diff `>1e-6`、
  matched actor/schema/SHA/RNG/action parity 或输入漂移时停止，不启动 E10。

## 2026-09-16 20:16:22 +0800 — V1.10.1 critic-Cm B-T10 完成

- activity_id: `ACT-20260916-192729-CMRESIDUAL-V1101-CRITIC-CM-T10`
- timestamp: `2026-09-16 20:16:22 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户明确要求继续原 V1.10 实验；本终态对应最终计划中的 B-critic-Cm T10。
- skills_used: `research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（沿用启动时已批准实现；训练未修改代码或配置，且未触碰 ObjectInteractionCm 用户修改）
- scope: GPU5、64 env、seed42、critic_cm variant、2005-D environment/1442-D actor/2005-D critic、10 epochs；
  与 A matched actor initialization，从零初始化且未加载 A 或历史 residual checkpoint。
- run_id: `cmresidual_v110_critic_cm_t10_20260916_192729`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 20:16:22 +0800`
- last_step / last_epoch: `20480 / 10`
- best_metric: `deterministic_reload_max_abs_diff=0`（工程门禁指标）
- checkpoint: [last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth)
- checkpoint_sha256: `1a12b955cd0cefcecfde3560df235beb0408ec7340784fa18250359a54776cb1`
- exit_reason: 达到批准的 10 epoch / 20480 frame 预算并正常退出。
- conclusion: `SUPPORTED`（B-T10 工程门禁）；Cm utility `INCONCLUSIVE`

**原因**

A-T10 已满足全部工程 gate，固定顺序要求使用 matched actor initialization 训练仅 critic 可见 Cm 的 B 变体；
本条将已完成运行落为终态，避免界面等待中断被误解为训练进程终止。

**验证**

- [outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729)、
  [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/run_manifest.json)、
  [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/metrics.jsonl)、
  [train.log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/train.log)、
  [checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/checkpoint_validation.json) 均存在。
- model、optimizer、deterministic action 均 finite；reload max-abs diff=`0`；sigma min/max=`0.099999994` 且冻结。
- safe policy contract、matched actor initialization、schema/SHA/RNG/首个随机动作 parity 全部通过；共写入 10 条 epoch metrics。
- 最终诊断量：critic loss=`0.0671258`、KL=`0.0105094`、success=`0`、residual RMS=`0.102356`、
  saturation ratio=`0.00173611`；训练量不用于单独判断 Cm 效果。
- 运行结束后 GPU5=`6 MiB / 0%`，无遗留 CmResidual 训练进程。

**保护与下一阶段**

- 未改 reward、observation/action、reference、数据 split、checkpoint 解释或 E10 参数；未覆盖 A/B0 及既有 output。
- A-T10 与 B-T10 工程 gate 均已通过，满足最终计划进入 deterministic A-E10/B-E10 的前置条件；截至本条记录，
  E10 尚未启动，Cm scientific conclusion 保持 `INCONCLUSIVE`。

## 2026-09-16 20:26:38 +0800 — V1.10/V1.11 实现、实验记录与提交归档

- activity_id: `ACT-20260916-202638-CMRESIDUAL-V1112-COMMIT`
- timestamp: `2026-09-16 20:26:38 +0800`
- modification_version: `V1.11.2`（同时归档已批准 V1.10.1 计划边界及运行记录）
- operation_category: `code`、`data`、`experiment`、`diagnostic`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户逐阶段确认 V1.10/V1.11 方案、实现与运行，本次明确要求提交代码。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `73001ed76316a3de591d90488eaac18749d7b98a`
- worktree_dirty: `true`（提交前工作树包含本 scope 差异及无关 ObjectInteractionCm 用户修改；后者明确排除）
- scope: 归档 V1.10 显式 provenance 与 paired T10、V1.11 固定 DexYCB base-only 构建/evaluator/reset 修复、
  定向测试、最终计划和实验记录；不继续启动 E10，不提交数据、cache、outputs 或 checkpoint。
- conclusion: `SUPPORTED`（实现、定向测试和已运行工程 gate）；V1.10 Cm utility `INCONCLUSIVE`；
  V1.11 固定单轨迹抓取 `REFUTED`、DexYCB 泛化 `INCONCLUSIVE`

**文件**

- `docs/current_versions.yaml` — 将 CmResidual 当前指针更新到 V1.11.2。
- `src/task/CmResidual/docs/README.md` — 汇总 V1.10/V1.11 当前状态与结论边界。
- `src/task/CmResidual/docs/logs/activity_log.md` — 记录计划、实现、运行与终态唯一时间线。
- `src/task/CmResidual/docs/logs/experiment_log.md` — 记录 B0、paired T10 和 DexYCB Gate 证据。
- [V1.10 最终计划](../plan/V1.10.md) 与 [V1.11 最终计划](../plan/V1.11.md) — 已批准执行边界。
- `src/task/CmResidual/docs/指导/V1.10.md`、`src/task/CmResidual/docs/指导/V1.11.md` — 用户研究指导。
- `src/task/CmResidual/tests/test_reference_contract.py` — V1.10 显式版本兼容合同。
- `src/task/CmResidual/tests/test_dexycb_base.py` — DexYCB 构建、配置、runner 和 reset 合同测试。
- `src/task/CmResidual/tools/eval_residual_stability.py`、`src/task/CmResidual/tools/run_ppo_stability.py` —
  V1.10.1 显式 run provenance，保留 V1.8/V1.9 legacy 默认值。
- `src/task/CmResidual/tools/data/build_dexycb_base_reference.py` — 固定 DexYCB 序列/物体的 evaluation-only 构建器。
- `src/task/CmResidual/tools/eval_dexycb_base.py` — frozen DExplore、strict zero residual 的固定 Gate C evaluator。
- `third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml` — 独立 opt-in DexYCB config。
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py` — 注册隔离的 DexYCB task 名。
- `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py` — opt-in actor-creation DOF 初始化、物体名和 tip tracking 诊断。

**原因**

V1.10 需要把正式 B0/T10 运行与现有 V1.9 runner 通过显式版本字段关联，避免改变 legacy 默认解释；V1.11
需要在不影响 airplane 轨迹的独立 config/runner 中检验 frozen base checkpoint 的固定 DexYCB 轨迹，并修复已证实的
actor creation/reset rigid-body 不同步。提交前同步 README 和 experiment log，避免文档仍误报 T10 未启动。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_dexycb_base.py src/task/CmResidual/tests/test_reference_contract.py tests/test_dexycb_raw.py`：`32 passed, 9 warnings`。
- `python -m py_compile`：四个本次工具脚本通过；`git diff --check` 通过。
- V1.10 A/B T10 均 `COMPLETED / SUPPORTED`，checkpoint finite、reload diff=`0`、sigma frozen；
  [A manifest](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/run_manifest.json) 与
  [B manifest](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/run_manifest.json)。
- V1.11.2 GPU5 Gate C `COMPLETED / SUPPORTED`；
  [run manifest](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/run_manifest.json)。

**保护与回滚**

- 不暂存 `src/task/ObjectInteractionCm/docs/logs/activity_log.md`；不提交 outputs、checkpoint、processed data 或 cache。
- airplane 默认配置与旧 V1.8/V1.9 provenance 保持兼容；DexYCB 行为均由独立 task/config opt-in。
- 回滚入口为本次单一 Git commit；运行证据保留在忽略的 outputs 中，不随源码回滚删除。

## 2026-09-16 20:50:28 +0800 — V1.10.1 control A-E10 启动

- activity_id: `ACT-20260916-205028-CMRESIDUAL-V1101-CONTROL-E10`
- timestamp: `2026-09-16 20:50:28 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户已确认 [V1.10 最终计划](../plan/V1.10.md) 并要求继续原实验；A/B T10 工程门禁均通过。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `759e732d9c359c883c49c2b912009e357ed74bc0`
- worktree_dirty: `true`（仅有未触碰的 ObjectInteractionCm 用户日志改动及本条运行记录）
- scope: GPU5、64 env、seed42、367 steps、deterministic mean-action；显式加载 control epoch-10 checkpoint，与 B0 比较 preservation gate。
- run_id: `cmresidual_v110_control_e10_20260916_205028`
- run_status: `STARTED`
- output: [outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028) — `PENDING`
- manifest: [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/run_manifest.json) — `PENDING`
- config: [config.json](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/config.json) — `PENDING`
- metrics: [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/metrics.jsonl) — `PENDING`
- log: [eval.log](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/eval.log) — `PENDING`
- checkpoint: [outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth)
- checkpoint_sha256: `fdcf7e00f3d8f250686c4fb30d3527577db5eb9fbfe2e6db39be29f8b6009278`
- baseline: [outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json)
- conclusion: `INCONCLUSIVE`（运行中）

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/eval_residual_stability.py --variant control --modification-version V1.10.1 --gpu 5 --seed 42 --num-envs 64 --steps 367 --checkpoint outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth --baseline-manifest outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json --run-id cmresidual_v110_control_e10_20260916_205028 --activity-id ACT-20260916-205028-CMRESIDUAL-V1101-CONTROL-E10`

**原因**

按 V1.10 最终计划在两侧 T10 门禁通过后开始固定顺序的 A-E10/B-E10；本次仅运行 A-E10，不改变协议或科研变量。

**验证**

- 启动前 GPU5=`6 MiB / 0%`，无 CmResidual 进程；A/B checkpoint epoch/frame=`10/20480`、finite、reload diff=`0`、sigma 冻结。
- 两侧 checkpoint SHA256 与各自 `checkpoint_validation.json` 一致；B0 manifest 为 `COMPLETED` 且 lift/contact 均为正。

**保护与回滚**

- 不修改代码、配置、数据、旧 output 或无关 ObjectInteractionCm 日志；新运行目录独立保存，停止时保留证据。

## 2026-09-16 21:33:56 +0800 — V1.10.1 control A-E10 完成

- activity_id: `ACT-20260916-205028-CMRESIDUAL-V1101-CONTROL-E10`
- timestamp: `2026-09-16 21:33:56 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: [V1.10 最终计划](../plan/V1.10.md) 固定的 A-E10 阶段。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `759e732d9c359c883c49c2b912009e357ed74bc0`
- worktree_dirty: `true`（本次 activity 与无关 ObjectInteractionCm 用户改动）
- run_id: `cmresidual_v110_control_e10_20260916_205028`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 21:33:45 +0800`
- last_step / last_epoch: `367 / 10`
- best_metric: `mean_env_max_lift_m=0.0849745274`
- checkpoint: [outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth)
- checkpoint_sha256: `fdcf7e00f3d8f250686c4fb30d3527577db5eb9fbfe2e6db39be29f8b6009278`
- exit_reason: 达到批准的 367 步预算并正常退出。
- conclusion: `SUPPORTED`（A preservation 工程 gate）；Cm utility `INCONCLUSIVE`

**原因**

按 V1.10 最终计划对 control epoch-10 checkpoint 执行固定 deterministic E10，并核对相对 fresh B0 的 preservation gate。

**验证**

- [outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/eval.log) 均存在；metrics 共 367 行。
- `mean_env_max_lift_m=0.0849745274 m`，mean contact occupancy=`0.3152503455`；相对 [B0 manifest](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json) 的 ratio=`1.0009423090 / 1.3958301886`，两项均高于 `0.8`。
- `max_success_fraction=0.1875`，final success rate=`1.0`；completed episode count=`357`，mean completed episode return=`6.9445565779`。这些单侧指标不证明 Cm utility。
- manifest 为 `COMPLETED`、`last_step=367`、`last_epoch=10`、`gate_passed=true`；GPU5 退出后空闲。

**保护与回滚**

- 未修改训练或评估协议、checkpoint、baseline、旧 output；运行证据保留在独立 ignored 目录。

## 2026-09-16 21:34:22 +0800 — V1.10.1 critic-Cm B-E10 启动

- activity_id: `ACT-20260916-213422-CMRESIDUAL-V1101-CRITIC-CM-E10`
- timestamp: `2026-09-16 21:34:22 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: [V1.10 最终计划](../plan/V1.10.md) 已获确认，A/B T10 与 A-E10 工程门禁均通过。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `759e732d9c359c883c49c2b912009e357ed74bc0`
- worktree_dirty: `true`（已记录 A 阶段活动及无关 ObjectInteractionCm 用户日志改动）
- scope: GPU5、64 env、seed42、367 steps、deterministic mean-action；显式加载 critic-Cm epoch-10 checkpoint，与同一 B0 比较 preservation gate。
- run_id: `cmresidual_v110_critic_cm_e10_20260916_213422`
- run_status: `STARTED`
- output: [outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422) — `PENDING`
- manifest: [run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/run_manifest.json) — `PENDING`
- config: [config.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/config.json) — `PENDING`
- metrics: [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/metrics.jsonl) — `PENDING`
- log: [eval.log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/eval.log) — `PENDING`
- checkpoint: [outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth)
- checkpoint_sha256: `1a12b955cd0cefcecfde3560df235beb0408ec7340784fa18250359a54776cb1`
- baseline: [outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json)
- conclusion: `INCONCLUSIVE`（运行中）

**命令**

`CUDA_VISIBLE_DEVICES=5 /home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/eval_residual_stability.py --variant critic_cm --modification-version V1.10.1 --gpu 5 --seed 42 --num-envs 64 --steps 367 --checkpoint outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth --baseline-manifest outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json --run-id cmresidual_v110_critic_cm_e10_20260916_213422 --activity-id ACT-20260916-213422-CMRESIDUAL-V1101-CRITIC-CM-E10`

**原因**

A-E10 双 preservation gate 已通过，按最终计划固定顺序进入 B-E10；不改变 seed、GPU、输入或仿真协议。

**验证**

- A-E10 `COMPLETED`，367 行 metrics，lift/contact ratio=`1.0009423090 / 1.3958301886`；GPU5 已空闲。
- B checkpoint epoch/frame=`10/20480`、finite、reload diff=`0`、sigma 冻结且 SHA256 与记录一致。

**保护与回滚**

- 不改代码、配置、数据、A/B0 output 或无关 ObjectInteractionCm 日志；新运行目录独立保存。

## 2026-09-16 22:12:08 +0800 — V1.10.1 critic-Cm B-E10 完成

- activity_id: `ACT-20260916-213422-CMRESIDUAL-V1101-CRITIC-CM-E10`
- timestamp: `2026-09-16 22:12:08 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`operation`
- task_mode: `run-only/operation`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: [V1.10 最终计划](../plan/V1.10.md) 固定的 B-E10 阶段，A-E10 双 preservation gate 已通过。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `759e732d9c359c883c49c2b912009e357ed74bc0`
- worktree_dirty: `true`（本次 activity 与无关 ObjectInteractionCm 用户日志差异）
- run_id: `cmresidual_v110_critic_cm_e10_20260916_213422`
- run_status: `COMPLETED`
- completed_at: `2026-09-16 22:11:51 +0800`
- last_step / last_epoch: `367 / 10`
- best_metric: `mean_env_max_lift_m=0.0850966126`
- checkpoint: [outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth)
- checkpoint_sha256: `1a12b955cd0cefcecfde3560df235beb0408ec7340784fa18250359a54776cb1`
- exit_reason: 达到批准的 367 步预算并正常退出。
- conclusion: `SUPPORTED`（B preservation 工程 gate）；Cm utility `INCONCLUSIVE`

**原因**

按 V1.10 最终计划对 critic-Cm epoch-10 checkpoint 执行固定 deterministic E10，并与同一 B0 及已完成 A-E10 比较。

**验证**

- [outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/eval.log) 均存在；metrics 共 367 行。
- B `mean_env_max_lift_m=0.0850966126 m`、mean contact occupancy=`0.1974199628`；相对 [B0 manifest](../../../../../outputs/CmResidual/cmresidual_v110_b0_cm_path_20260916_170651/run_manifest.json) 的 ratio=`1.0023803901 / 0.8741140109`，两项均高于 `0.8`。
- B `max_success_fraction=0.34375`，final success rate=`1.0`，completed episode count=`294`，mean completed episode return=`6.9683459470`。
- 与 [A-E10 manifest](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/run_manifest.json) 的 `protocol` 完全相同；B−A 的 lift=`+0.0001220852 m`、contact=`−0.1178303827`，相对差=`+0.1437% / −37.3768%`。两侧 367 行 observation/target finite 均为 true。
- manifest 为 `COMPLETED`、`last_step=367`、`last_epoch=10`、`gate_passed=true`；GPU5 退出后空闲。

**保护与回滚**

- V1.10 固定顺序 B0→A/B T10→A/B E10 已完成；未启动 T20、多 seed 或额外调参，未改代码、配置、数据、旧 output 或无关 ObjectInteractionCm 日志。
- 新运行目录及其 manifest 保留作审计；本次只新增 CmResidual activity/experiment/README 记录，可从 Git diff 回退文档记录而不删除实验产物。

## 2026-09-16 22:13:46 +0800 — V1.10.1 E10 配对结果归档

- activity_id: `ACT-20260916-221346-CMRESIDUAL-V1101-E10-ARCHIVE`
- timestamp: `2026-09-16 22:13:46 +0800`
- modification_version: `V1.10.1`
- operation_category: `experiment`、`documentation`
- task_mode: `run-only/operation`（运行终态文档归档）
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户已确认 [V1.10 最终计划](../plan/V1.10.md) 并要求继续原实验；本次归档其 A/B E10 终态。
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `759e732d9c359c883c49c2b912009e357ed74bc0`
- worktree_dirty: `true`（本次三份 CmResidual 文档及未触碰的 ObjectInteractionCm 用户日志改动）
- scope: 仅归档 V1.10 A/B E10 运行、工程 preservation 与单 seed 结论边界；不修改科研协议或实验产物。
- run_id: `cmresidual_v110_control_e10_20260916_205028`、`cmresidual_v110_critic_cm_e10_20260916_213422`
- run_status: A/B 均为 `COMPLETED`
- last_step / last_epoch: A/B 均为 `367 / 10`
- best_metric: A/B `mean_env_max_lift_m=0.0849745274 / 0.0850966126 m`
- exit_reason: A/B 均达到批准的 367 步预算并正常退出。
- conclusion: `SUPPORTED`（A/B preservation 工程 gate）；Cm utility `INCONCLUSIVE`

**文件**

- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — A/B E10 启动、终态、命令、版本、checkpoint 与产物入口。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 配对表、训练动态诊断、假设、证据与结论边界。
- [src/task/CmResidual/docs/README.md](../README.md) — 当前状态更新为 E10 已完成。

**原因**

E10 两侧均完成；需要使 Task 入口与实验记录反映真实终态，避免仍显示“E10 未运行”。

**验证**

- A：[outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v110_control_e10_20260916_205028/eval.log)、[epoch-10 checkpoint](../../../../../outputs/CmResidual/cmresidual_v110_control_t10_20260916_183934/CmResidualV19ControlT10/nn/last_CmResidualSafeCriticControl_ep_10_rew__7.02_.pth)。
- B：[outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/run_manifest.json)、[config.json](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/config.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_e10_20260916_213422/eval.log)、[epoch-10 checkpoint](../../../../../outputs/CmResidual/cmresidual_v110_critic_cm_t10_20260916_192729/CmResidualV19CriticCmT10/nn/last_CmResidualSafeCriticCm_ep_10_rew__6.92_.pth)。
- A/B manifest 均为 `COMPLETED`、367 steps、epoch 10，`protocol` 与 B0 完全一致，两侧 367 行 observation/target finite 均为 true；GPU5 已空闲。
- A/B lift/contact ratio=`1.000942/1.395830` 与 `1.002380/0.874114`，双 80% gate 均通过；B−A 的 contact=`−0.117830`、lift=`+0.000122 m`。
- 交接前运行 `git diff --check` 与 Task-local `audit_diff.py --worktree --scope-prefix src/task/CmResidual --check-links`。

**保护与回滚**

- 未触碰 `src/task/ObjectInteractionCm/docs/logs/activity_log.md`、指导/plan、代码、配置、数据、cache、旧运行或 checkpoint；没有启动 T20、多 seed 或 DexYCB 新实验。
- 回滚入口：仅还原本次三份 CmResidual 文档差异；保留两次 ignored E10 运行目录作为可复核证据。

## 2026-09-17 16:17:43 +0800 — Cmv2 actor / DexYCB 单轨迹训练接口诊断

- activity_id: `ACT-20260917-161743-CMRESIDUAL-CMV2-DEXYCB-INTERFACE-DIAG`
- timestamp: `2026-09-17 16:17:43 +0800`
- modification_version: `V1.11.2`（只读诊断沿用当前 Task 指针；新研究实施版本待指导与计划确认）
- operation_category: `diagnostic`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-approved`（用户要求先对齐远端 Cmv2 与残差 actor 接口）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `683b5d236dc8d8349620e34952c64afd9c892da3`
- worktree_dirty: `true`（已存在 ObjectInteractionCm 用户日志改动及根级 fetch 诊断记录）
- scope: 只读比较 `origin/oyx` 的 `ObjectInteractionCmv2V13Model` 与本地 CmResidual/DexYCB 合同；不修改实现、配置、数据或训练状态。
- conclusion: `SUPPORTED`（接口差异已定位）；actor-Cm 效果与 DexYCB 训练结果 `INCONCLUSIVE`

**原因**

用户要求冻结 base、只训练 residual、先使用一条 DexYCB 轨迹，并表示稍后提供 Cmv2 checkpoint。实现前需要确认输入、输出、数据资格和 checkpoint 语义。

**验证**

- 远端 `origin/oyx:src/task/ObjectInteractionCmv2/model.py` 的 V1.3 类输出 `cm_tokens [B,16,32]`、`token_anchors`、`token_normals`、`token_mass`、`token_mask`；checkpoint 标识为 `v1_3_rigid_only`，输入必需 object/hand points、normals、hand flow，可选 `delta_time_s`。
- [src/task/CmResidual/tools/run_ppo_stability.py](../../tools/run_ppo_stability.py) 只支持旧 V1.8/V1.9/V1.10 变体；[当前网络](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py) 可设 actor/critic 输入宽度，但现有旧 Cm 上下文是 563-D。
- [当前任务实现](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) 读取旧 `ObjectInteractionCmModel`，拼接 563-D token/anchor/effect；[DexYCB base 配置](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBBase.yaml) 关闭 Cm、仅 1442-D observation。
- [当前 DexYCB reference manifest](../../../../../data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806/manifest.json) 为 72 帧、`split=evaluation`、`training_eligible=false`；[V1.11 指导](../指导/V1.11.md) 明确禁止 residual PPO，故不得直接在当前产物上启动正式训练。

**保护与回滚**

- 未 merge 远端、未加载尚未提供的 checkpoint、未覆盖现有 DexYCB 评估产物；仅本条 Task activity 可按 Git diff 回退。

## 2026-09-17 17:21:55 +0800 — V1.12.1 action-conditioned Cmv2 接口迁回 oyx

- activity_id: `ACT-20260917-172155-CMRESIDUAL-V121-ACTION-EFFECT`
- timestamp: `2026-09-17 17:21:55 +0800`
- modification_version: `V1.12.1`
- operation_category: `architecture`、`code`、`diagnostic`、`documentation`
- task_mode: `change`
- change_level: `L2`
- approval: `user-approved`
- approval_basis: 用户新增 V1.12 action-conditioned 指导后明确回复“继续”，随后确认将实现迁回 `oyx`；真实 checkpoint、物理候选排序和 PPO 仍受 V1.12 最终计划门禁约束。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `e444b1d0cce192dc9a310f0dcfd7bb67eea8f67c`（`origin/oyx` 快进后）
- worktree_dirty: `true`（保留根级和 ObjectInteractionCm 活动日志的既有/合并后本地改动；本条只登记 CmResidual 实现）
- scope: 将已验证的 action→controller/FK→nominal hand sweep→冻结 Cmv2 effect evaluator 接口迁回远端 `oyx` 基线；旧 OI-Cm、base-only、zero-residual 和 residual PPO 路径保持兼容。
- run_id: `none`
- run_status: `NOT_STARTED`
- conclusion: `SUPPORTED`（接口工程与静态合同）；科研效果 `INCONCLUSIVE`

**文件**

- [V1.12 用户指导](../指导/V1.12.md) — 用户提供的研究方向，保留为未提交工作树文件。
- [V1.12 最终计划](../plan/V1.12.md) — 本次实现范围、接口、门禁和回滚边界。
- [cm_v2_adapter.py](../../cm_v2_adapter.py) — 冻结 Cmv2 V1.3 structured effect adapter、严格 schema/config/SHA 校验与 640-D context 合同。
- [cm_v2_action_evaluator.py](../../cm_v2_action_evaluator.py) — nominal controller/FK hand-sweep、候选效应预测、finite/valid-mask 与排序指标。
- [test_cmv2_action_evaluator.py](../../tests/test_cmv2_action_evaluator.py)、[test_cmv2_adapter.py](../../tests/test_cmv2_adapter.py) — Task-local 合同测试。
- [eval_dexycb_cmv2_action_effect.py](../../tools/eval_dexycb_cmv2_action_effect.py) — synthetic/真实 checkpoint 诊断入口；本轮未加载真实 checkpoint。
- [CmResidualDexYCBCmv2ActionEval.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualDexYCBCmv2ActionEval.yaml) — 1442-D diagnostic opt-in 配置，未提供 PPO 训练入口。
- [task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[tasks/__init__.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py) — vendor task 的显式 evaluator 接入与 Hydra alias。
- [README.md](../README.md)、[current_versions.yaml](../../../../../docs/current_versions.yaml) — V1.12.1 入口和版本指针。

**原因**

原实现基于 `origin/oyx=e444b1d` 的独立工作树完成，用户明确要求不要交付独立分支，因此在保护本地日志和指导文档后，将同一组显式接口迁回 `oyx`；action evaluator 保持 Cmv2 在候选评估侧，不把 latent/context 拼接到 PPO observation。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_cmv2_action_evaluator.py src/task/CmResidual/tests/test_cmv2_adapter.py src/task/CmResidual/tests/test_reference_contract.py -q -k 'cmv2 or zero_residual_preserves_dexplore_physical_targets'`：`9 passed, 21 deselected, 1 warning`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile ...`：adapter、evaluator、CLI 和 vendor task 均通过。
- CLI `--help` 通过；`--synthetic --candidates 2` 输出 `finite=true`，candidate count `2`，effect score 有限。
- 远端合并：`oyx` 从 `683b5d2` 快进到 `e444b1d`；stash 恢复时两个活动日志仅为双方新增内容，已保留两边记录并去除冲突标记，未覆盖用户日志。
- 交接前运行 `audit_diff.py --log src/task/CmResidual/docs/logs/activity_log.md --worktree --scope-prefix src/task/CmResidual --check-links`：通过，本地链接 `12` 个可导航；本条不产生训练 run、metrics、train.log 或 checkpoint。

**保护与回滚**

- 未加载用户尚未提供的 Cmv2 checkpoint，未运行 GPU 物理候选排序、DexYCB PPO 或样本内评估；未改变 reference split、坐标系、单位、GT、reward 或 base policy。
- 旧 OI-Cm 和 base-only 配置保持原路径；回滚入口为本条实现文件、配置、测试、Task README/版本指针及本活动记录的显式 Git diff，远端 `e444b1d` 基线和独立工作树仍保留。

## 2026-09-17 18:17:57 +0800 — 导入权重的 Cmv2 checkpoint 身份诊断

- timestamp: `2026-09-17 18:17:57 +0800`
- activity_id: `ACT-20260917-181757-CMRESIDUAL-V1121-CHECKPOINT-DIAGNOSTIC`
- modification_version: `V1.12.1`
- operation_category: `diagnostic`、`documentation`
- task_mode: `read-only/diagnostic`（追加本条证据记录）
- change_level: `L0`
- approval: `auto`（用户要求继续检查已补全的权重；只读校验和证据记录）
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（已有根级、ObjectInteractionCm 日志和 V1.12 指导未提交改动）
- scope: 只读校验用户导入的权重身份及 V1.12 真实 checkpoint Gate；未启动训练、物理评估或修改 checkpoint。

**文件与证据**

- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/best.pt](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/best.pt) — 1,096,720,131 bytes；SHA256 `e5c1822547c958a8dbb282dec647733e83ea16ebb4e08db390f3cb5057aa0898`。
- [src/task/CmResidual/docs/plan/V1.12.md](../plan/V1.12.md) — 真实 checkpoint Gate 要求 Cmv2 V1.3 严格加载；本条记录校验结果。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 本次诊断与停止依据。

**原因**

先前同路径文件缺少 zip 中央目录；用户现已补全权重，需要重新判断是否可进入 V1.12 物理评估和后续训练。

**验证**

- `sha256sum third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/best.pt` 得到上述 SHA256；`unzip -t` 全部通过，文件容器完整。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -c 'import torch; ... torch.load(..., map_location="cpu", weights_only=False)'` 可读取 payload，但 `architecture_version=None`，`config.name=correspondence_ptv3_v2_finetune_day_cosine_restart`，`model.class_path=src.task.correspondence_ptv3_v2.model.StaticHOCPTv3V2`，`epoch=50`、`step=113500`。
- `FrozenCmv2Adapter` 对上述固定 SHA 严格加载时抛出 `ValueError: Cmv2 checkpoint architecture_version mismatch`。本地 `outputs/ObjectInteractionCmv2`、Task 资产目录及该 vendor 目录未发现其他 Cmv2 checkpoint。
- 结论：`INVALID_IMPLEMENTATION`（权重身份不匹配 Cmv2 V1.3 输入合同）；`run_id=none`，`run_status=NOT_STARTED`。真实 checkpoint Gate 未通过，nominal sweep、counterfactual physics 和 PPO 均未运行；科研效果仍为 `INCONCLUSIVE`。

**保护与回滚**

只追加本条活动记录；删除本条即可回滚文档变化。原权重、其他用户改动、旧运行及 checkpoint 均未触碰。

## 2026-09-17 18:28:33 +0800 — Cmv2 latest.pt 身份与合成输入门禁诊断

- timestamp: `2026-09-17 18:28:33 +0800`
- activity_id: `ACT-20260917-182833-CMRESIDUAL-V1122-LATEST-DIAGNOSTIC`
- modification_version: `V1.12.2`
- operation_category: `diagnostic`、`documentation`
- task_mode: `read-only/diagnostic`（追加证据与版本指针）
- change_level: `L0`
- approval: `auto`（用户要求检查指定 `latest.pt`；只读检查及记录）
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留既有根级、ObjectInteractionCm 日志与 V1.12 指导改动）
- scope: 核对指定 Cmv2 checkpoint 的 SHA、结构、严格加载和合成输入有限前向；未执行真实 Inspire 几何或物理候选评估。

**文件与证据**

- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/latest.pt](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/latest.pt) — 3,741,125 bytes，SHA256 `371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591`。
- [src/task/CmResidual/docs/plan/V1.12.md](../plan/V1.12.md) — 分阶段 checkpoint、nominal sweep 与 physics Gate。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — CmResidual 指针更新至 `V1.12.2`。
- [src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 本次诊断记录。

**原因**

先前 `best.pt` 是 PTv3 权重；用户指定新导入的 `latest.pt`，需核对真实模型身份与前向接口。

**验证**

- `sha256sum` 与 `unzip -t`：上述 SHA 固定，容器内容完整。
- `torch.load(..., map_location="cpu", weights_only=False)`：payload 的 `architecture_version=v1_3_rigid_only`，`step=4088`，`epoch=4`，`model` 有 39 项；无 `config` 字段。
- `FrozenCmv2Adapter(..., expected_sha256, "cpu")`：严格 state_dict 加载通过，参数全部冻结。
- `eval_dexycb_cmv2_action_effect.py --checkpoint .../latest.pt --checkpoint-sha256 371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591 --device cpu --candidates 2`：合成几何输出 `finite=true`，两个候选分数均有限。
- 结论：checkpoint 身份、严格加载与合成前向的工程诊断为 `SUPPORTED`；checkpoint 未内嵌配置，非参数超参仍需与原训练配置/manifest 核对，真实 Inspire geometry、nominal sweep parity 与 counterfactual physics Gate 尚未执行，科研效果 `INCONCLUSIVE`。`run_id=none`，`run_status=NOT_STARTED`。

**保护与回滚**

仅追加本条并更新 Task 版本指针；旧 `best.pt`、新 `latest.pt`、代码、数据、旧运行与用户改动均未触碰。删除本条并恢复指针即可回滚文档变化。

## 2026-09-17 18:42:28 +0800 — V1.12 Cmv2 真实几何与物理候选 Gate

- timestamp: `2026-09-17 18:42:28 +0800`
- activity_id: `ACT-20260917-190400-CMRESIDUAL-V1124-CMV2-PHYSICS`
- modification_version: `V1.12.4`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`、`documentation`
- task_mode: `run-only/operation`（为执行已定稿 V1.12 Gate，补齐其独立物理运行入口的最小实现）
- change_level: `L2`（Task-local 候选采样/物理评估接口）；物理运行遵循已批准的 V1.12 L3 Gate
- approval: `user-approved`
- approval_basis: V1.12 最终计划已明确批准 K=8 counterfactual physics Gate；用户在 checkpoint 身份诊断后明确回复“你继续吧”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留用户已有根级/ObjectInteractionCm 日志及 V1.12 指导差异）
- scope: 在空闲 GPU5 上验证真实 Inspire/DexYCB 状态的 controller→FK→冻结 Cmv2→单步物理 K=8 Gate；不启动 PPO，不改变 reference、GT、reward、base policy、资产、坐标或 checkpoint。
- run_ids: `cmresidual_v112_cmv2_action_effect_20260917_190000`、`cmresidual_v112_cmv2_action_effect_20260917_190200`、`cmresidual_v112_cmv2_action_effect_20260917_190400`
- run_status: 前两次 `COMPLETED`（Gate 不通过）；最后一次 `FAILED`（可行候选重采样停止）
- conclusion: `INVALID_IMPLEMENTATION`（当前仿真状态无法形成 V1.12 的有效候选/interaction gate）；科研结论 `INCONCLUSIVE`

**文件与运行**

- [src/task/CmResidual/tools/eval_dexycb_cmv2_physics.py](../../tools/eval_dexycb_cmv2_physics.py) — 新增独立 K=8 物理 Gate：固定 GPU5/seed42、53 步 base warmup、零 residual + 可行随机 residual、严格 SHA/manifest 和不覆盖输出。
- [src/task/CmResidual/docs/README.md](../README.md) — 更新 Task 当前状态至 V1.12.4 与物理 Gate 停止结论。
- [src/task/CmResidual/docs/指导/V1.12.md](../指导/V1.12.md) — 用户已有的未提交 V1.12 研究指导；本次只按其最终边界执行，未修改内容。
- [outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190000](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190000/) — 首次运行；[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190000/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190000/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190000/eval.log)。
- [outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200/) — 已完成的 K=8 可复核结果；[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200/eval.log)。
- [outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190400](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190400/) — 严格可行候选重采样失败的终态；[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190400/run_manifest.json)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190400/eval.log)。

**原因**

已定稿 V1.12 要求在进入任何 PPO 前，以真实 checkpoint、nominal action sweep 与 K=8 物理执行验证 action-conditioned Cmv2 是否获得可用的交互/ranking 信号。先前 CLI 只覆盖合成输入，故以最小 Task-local 运行入口执行该既有 Gate；候选越界时必须排除而非裁剪。

**验证**

- 真实 reference 几何静态检查：frame 0 的最小 hand–object surface distance=`0.306138 m`，16 个 token 全空；接触最强的 reference frame 53 距离=`0.000173 m`、token 数=`13/16`，说明冻结 Cmv2 与 FK/几何本身可产生有限、非空的真实几何输出，但首帧不具交互。
- GPU5 物理 Gate 在 53 步 frozen-base warmup 后运行。首次 K=8 有 `5/8` 候选在约束内，因其余候选饱和不作为有效排序证据；运行入口随后改为 rejection sampling，避免将裁剪后的候选伪装为可行 action。
- 第二次 K=8 的重采样在每个环境的当前状态得到 8 个候选，但执行后 token min/max 仍为 `0/0`、contact occupancy 均为 `0`、tip distance 为 `0.424–2.184 m`；尽管表面 top-1 相同、Spearman=`0.404762`，无 interaction token 时 Cmv2 仅输出全局先验，该数值不构成 ranking 支持。
- 进一步按各环境实际 warmup state 验证动作可行性，最后一次在 `4096` 次采样中不能得到 8 个无饱和非零 residual；零 residual 保持唯一允许的候选，故停止而不缩小尺度、裁剪、改变 reward 或换序列。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmResidual/tools/eval_dexycb_cmv2_physics.py` 通过；三次命令均严格校验 Cmv2 SHA `371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591`，生成独立 manifest/log，且无训练 checkpoint。

**保护与回滚**

- 未运行 residual PPO、未更新训练资格、未修改 DExplore/Cmv2 权重、共享 `src/base`、旧 OI-Cm 路径或用户已有改动。删除新增物理 Gate 工具、还原版本指针和本条活动记录即可回滚代码/文档；三个 `outputs/` 运行目录保留为不可覆盖的诊断证据。

## 2026-09-17 18:50:06 +0800 — V1.12 physical Gate 停止原因诊断

- timestamp: `2026-09-17 18:50:06 +0800`
- activity_id: `ACT-20260917-185006-CMRESIDUAL-V1125-PHYSICS-DIAG`
- modification_version: `V1.12.5`
- operation_category: `diagnostic`、`documentation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户明确要求诊断 V1.12 Gate 未通过原因；本次只读运行定点状态检查。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留用户已有根级/ObjectInteractionCm 日志、V1.12 指导及本 Task 的未提交 Gate 实现）
- scope: 比较 V1.11.2 base-only 证据与 V1.12 已解析配置，并在 GPU5 定点复现 53-step zero-residual warmup 的状态/目标边界；不修改代码、配置、数据、权重或训练状态。
- run_id: `none`
- run_status: `COMPLETED`（短时、无落盘的诊断命令）
- conclusion: `SUPPORTED`（停止原因已定位）；Cmv2 ranking、residual PPO 与科研效果 `INCONCLUSIVE`

**文件与证据**

- [src/task/CmResidual/docs/README.md](../README.md) — 更新 V1.12.5 的诊断状态；[src/task/CmResidual/docs/指导/V1.12.md](../指导/V1.12.md) 为用户已有指导，本次未改。
- [src/task/CmResidual/tools/eval_dexycb_cmv2_physics.py](../../tools/eval_dexycb_cmv2_physics.py) — 先前 V1.12 Gate 的未提交 Task-local 工具；本次只读诊断未改。
- [outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/run_manifest.json) — V1.11.2 frozen-base 对照。
- [outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1112_dexycb_base_gpu5_20260916_182728/metrics.jsonl) — step 53 的既有 base-only 行为证据。
- [outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v112_cmv2_action_effect_20260917_190200/run_manifest.json) — V1.12 K=8 的输入、配置与终态。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py) — zero residual 保留 DExplore unclamped base target，而非零 residual 必须经过 joint/mimic 安全约束。
- [src/task/CmResidual/docs/plan/V1.12.md](../plan/V1.12.md) — 固定状态、K=8 可行动作及失败停止边界。

**原因**

澄清 V1.12 K=8 Gate 的失败是 Cmv2 输入接线问题、冻结 base 行为问题，还是候选执行协议未能构造严格 counterfactual；该结论决定是否可进入后续计划协商。

**验证**

- V1.11.2 已通过 reset/finite/zero-residual 的工程 Gate，但其 step 53 `contact_occupancy=0`、`tip_distance_mean_m=1.668312 m`、`reference_root_position_error_m=1.446333 m`；因此 Cmv2 运行看到的无接触并非新 evaluator 回归。
- 当前 GPU5/4-env/seed42 的只读复现同样到 `reference_index=53`：实际 hand–object 最小 surface distance 为 `[3.430841, 0.260098, 0.272403, 0.263671] m`，tip distance 为 `[3.609586, 0.353069, 0.376766, 0.475268] m`，root error 为 `[2.374451, 0.392007, 0.484377, 0.704440] m`；与静态 reference frame 53 的 `0.000173 m` 接触距离不同。
- 同一复现中，四个 env 的 frozen-base target 越界 DOF 数为 `[0,1,0,0]`，最大越界 `0.096375 rad`。由于 zero-residual 为 release compatibility 保留 unclamped target，任何非零 residual 进入安全路径时会触发 clamp/saturation；这解释了 K=8 可行动作重采样的停止。
- V1.12 运行把 8 个候选放入 8 个独立 GPU PhysX env。warmup 后各 env 状态已显著分叉，故它们不是“同一状态逐个 action”的 counterfactual；仅以并行 slot 的 Spearman/top-1 不能作 ranking 结论。
- 解析配置比较只差 task alias、Cmv2 evaluator opt-in、reference profile 和 `numEnvs=4→8`；DExplore checkpoint、reference/source SHA、资产、物理参数、30 Hz 和 reward 不变。无代码/配置修改；诊断命令正常退出。

**保护与回滚**

- 未启动 PPO、未改变研究变量或覆写旧 run。仅本条、Task README 和版本指针为 L0 诊断记录，可独立回滚；此前 V1.12 物理 Gate 工具及输出保持原状。
## 2026-09-17 20:36:13 +0800 — V1.13 GRAB 重定向确定性 base 与无接触 Gate

- timestamp: `2026-09-17 20:36:13 +0800`
- activity_id: `ACT-20260917-203900-CMRESIDUAL-V1131`
- modification_version: `V1.13.1`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task 局部 base target、reference 与 observation 合同；评估运行按批准的 Gate）
- approval: `user-approved`
- approval_basis: 用户确认先用 GRAB 现成重定向数据训练、DexYCB 暂不使用，并回复“是的，就按照你想的来”；[V1.13 最终计划](../plan/V1.13.md) 定义先做确定性 base/无接触 Gate。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留先前用户和其他任务的未提交改动）
- scope: 新增 GRAB 确定性 base 路线；沿用旧 DExplore 路线供复现，不改 GRAB/DexYCB 数据、GT、split、权重、共享 `src/base` 或旧 run。
- run_id: `cmresidual_v113_grab_nominal_20260917_2039`
- run_status: `COMPLETED`（CPU、1 env、366 steps）；此前 `cmresidual_v113_grab_nominal_20260917_2036` 为终止步指标读取修复前的独立运行，不用于结论。
- last_step: `366`；best_metric: 不适用（确定性评估，无 checkpoint 选择）；best_checkpoint/latest_checkpoint: 不存在。
- conclusion: 工程接线 `SUPPORTED`；无接触追踪是否足以进入交互 PPO 为 `INCONCLUSIVE`，未启动 PPO。

**文件与证据**

- [src/task/CmResidual/docs/plan/V1.13.md](../plan/V1.13.md) — 最终计划；[src/task/CmResidual/docs/指导/V1.13.md](../指导/V1.13.md) 为用户指导。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py) — GRAB 手腕位姿加手指 q 转 native 目标，校验 FK、manifest 与 finite。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py) — 新路线绝对目标叠加 residual，按参考 manifest 应用 mimic 系数。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py)、[third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabRetargeted.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabRetargeted.yaml) — 独立的 68-D GRAB 路线，跳过 DExplore teacher。
- [src/task/CmResidual/tools/eval_grab_retargeted_base.py](../../tools/eval_grab_retargeted_base.py)、[src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — 无接触 Gate 与 GRAB mimic 合同测试。
- [src/task/CmResidual/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 状态、版本和实验解释。
- [outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/) — [config.json](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/eval.log)。
- [outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2036](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2036/) — 首次独立运行，仅作工具终止步读取审计。

**原因**

V1.12 冻结 DExplore base 在参考接触帧已明显漂移。用户按 V1.13 改为直接使用 GRAB 重定向轨迹作为名义 base；该参考的 `q_native_ref[:6]` 全零，必须结合 `wrist_pose_world_ref` 的 URDF 逆解。其小指 mimic 系数 `1.18` 也不同于旧 DExplore 映射的 `1.05`，故新路线按 manifest 实现，旧路线不变。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_reference_contract.py -q`：`23 passed`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile` 新增/修改的四个 Python 入口：通过。
- `eval_grab_retargeted_base.py --run-id cmresidual_v113_grab_nominal_20260917_2039 --activity-id ACT-20260917-203900-CMRESIDUAL-V1131`：366 步 CPU 运行正常退出；全程 finite、无接触、零 residual 饱和，末帧后正常 reset。
- 367 帧手腕 URDF 逆解的最大 FK 回代矩阵误差 `8.71e-08`。手腕位置误差平均/p95/最大 `0.01120/0.02924/0.06276 m`，指尖 `0.01395/0.03775/0.07101 m`。按 step 对齐的原 GRAB 接触标记 296 帧上，指尖误差平均/p95/最大 `0.01286/0.03703/0.04245 m`（V1.13.2 复核修正）；高分位误差超过 Cmv2 `0.02 m` interaction radius，不据此放行 PPO。单环境无接触 smoke 不支持抓取、残差或 Cm 效果结论。

**保护与回滚**

只回滚 V1.13 新增 Task 局部实现、配置、测试与本次记录；旧 DExplore config/权重、GRAB/DexYCB 数据和旧运行保持原状。两个新输出目录作为诊断证据保留，不覆盖。
## 2026-09-17 20:46:17 +0800 — V1.13.2 GRAB 无接触腕部滞后诊断与训练门禁复核

- timestamp: `2026-09-17 20:46:17 +0800`
- activity_id: `ACT-20260917-204549-CMRESIDUAL-V1132`
- modification_version: `V1.13.2`
- operation_category: `diagnostic`、`experiment`、`operation`、`documentation`
- task_mode: `change`（Task 局部诊断字段），随后 `run-only/operation`（同一 V1.13 无接触协议）；未改变控制器或研究变量。
- change_level: `L0`（诊断字段和记录；无控制行为变化）
- approval: `auto`
- approval_basis: 用户要求继续并询问能否训练；V1.13 最终计划规定无接触门禁不足时先诊断控制器与参考时序。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留此前 V1.13.1 及用户、其他 Task 的未提交改动）
- scope: 同一 CPU/1 env/seed42/366-step 无接触协议增加实际与参考腕部位置记录，复核接触帧 step 对齐和训练前门禁；不改 PD、clock、reference、reward、checkpoint、数据或旧 run。
- run_id: `cmresidual_v1132_grab_lag_20260917_2046`
- run_status: `COMPLETED`
- last_step: `366`；best_metric: 不适用（确定性诊断）；best_checkpoint/latest_checkpoint: 不存在。
- conclusion: 两帧位置滞后的诊断证据 `SUPPORTED`；可训练性与控制器修正效果 `INCONCLUSIVE`，PPO `NOT_STARTED`。

**文件与证据**

- [src/task/CmResidual/tools/eval_grab_retargeted_base.py](../../tools/eval_grab_retargeted_base.py) — 逐步指标增记实际/参考腕部世界位置，终止步使用 `terminal_wrist_pose` 避免自动 reset 污染。
- [outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/) — [config.json](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v1132_grab_lag_20260917_2046/eval.log)。
- [outputs/CmResidual/cmresidual_v113_grab_lag_20260917_2043](../../../../../outputs/CmResidual/cmresidual_v113_grab_lag_20260917_2043/) — 先前同协议运行的 manifest 被工具默认值写成 `V1.13.1`，故作为 provenance 错误的审计证据保留，不用作本次规范结果；修正后的两次逐步指标 SHA256 一致。
- [src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 对齐方法、滞后曲线和结论边界；[src/task/CmResidual/docs/plan/V1.13.md](../plan/V1.13.md) — 下一阶段低层控制修订提议，明确标记 `draft`、未获批准。
- [src/task/CmResidual/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml) — 更新当前诊断状态和 Task 版本指针。

**原因**

V1.13.1 接触标记帧指尖误差已接近或超过 Cmv2 interaction radius，不能把无接触运行正常退出当作 PPO 准入；需要判断偏差是否主要体现为固定时钟下低层跟踪滞后。同时发现先前接触帧统计把 step 1..366 错配到参考 0..365，故复核并修正旧记录的数字，不改原始指标文件。

**验证**

- `eval_grab_retargeted_base.py --run-id cmresidual_v1132_grab_lag_20260917_2046 --activity-id ACT-20260917-204549-CMRESIDUAL-V1132 --modification-version V1.13.2`：366 步正常退出，全程 finite、无手部接触、零 residual 饱和，终止步读取与同帧腕部误差 parity 最大绝对差约 `4.47e-09 m`。
- 接触标记对齐后 296 帧：同帧腕部位置误差 mean/p95=`0.01000/0.02762 m`；与两帧前参考对比 mean/p95=`0.00264/0.00740 m`。腕部误差与相邻参考位移 Pearson 相关约 `0.898`。接触帧腕部误差超过 `0.015 m` 占 `27.03%`，指尖误差超过 `0.020 m` 占 `26.69%`；这支持“约两帧滞后”的诊断，不证明增益修正或残差训练有效。
- 只读复核命令读取既有 [V1.13.1 metrics](../../../../../outputs/CmResidual/cmresidual_v113_grab_nominal_20260917_2039/metrics.jsonl) 和 GRAB source tensor；按 step 对齐后更正接触帧指尖 mean/p95/max 为 `0.01286/0.03703/0.04245 m`。训练门禁未放行，未启动 PPO。

**保护与回滚**

诊断字段、计划修订草稿和新活动/实验记录可单独撤销；两个 V1.13 运行目录、其他 Task 用户改动、GRAB/DexYCB 数据、旧 DExplore 路径与 checkpoint 均未覆盖。新运行目录保留为可复核证据。
## 2026-09-17 20:57:25 +0800 — V1.13.3 GPU5 腕部 PD 增益对照未通过门禁

- timestamp: `2026-09-17 20:57:25 +0800`
- activity_id: `ACT-20260917-205100-CMRESIDUAL-V1133`
- modification_version: `V1.13.3`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task 局部控制器物理参数化），GPU5 物理评估按批准的 `L3` Gate 执行。
- approval: `user-approved`
- approval_basis: 用户在 [V1.13 计划](../plan/V1.13.md) 的 GPU5 增益对照方案及后续阶段说明后回复“可以”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留此前 V1.13、其他 Task 与用户未提交改动）
- scope: 仅给新 GRAB retargeted 任务显式腕部 position drive gains，依预设三组在空闲 GPU5 比较无接触追踪；旧 DExplore drive 固定值、手指 drive、参考、时钟、reward、数据、checkpoint 不变。
- run_ids: `cmresidual_v1133_gpu5_kp200_kd20_20260917_2051`、`cmresidual_v1133_gpu5_kp300_kd30_20260917_2056`、`cmresidual_v1133_gpu5_kp400_kd40_20260917_2057`
- run_status: 三组均 `COMPLETED`，每组 `last_step=366`；best_metric、best_checkpoint、latest_checkpoint 不适用（确定性零 residual 评估）。
- conclusion: 对预定三组中任一满足接触帧追踪门禁的假设为 `REFUTED`；低层速度前馈、PPO 与科研效果 `INCONCLUSIVE`，PPO `NOT_STARTED`。

**文件与证据**

- [src/task/CmResidual/docs/plan/V1.13.md](../plan/V1.13.md) — 用户批准的 GPU5 增益对照修订；[src/task/CmResidual/docs/指导/V1.13.md](../指导/V1.13.md) 为用户研究指导。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabRetargeted.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabRetargeted.yaml) — 新路线腕部 gains 可配置，缺省 `200/20`；旧路线保持硬编码原值。
- [src/task/CmResidual/tools/eval_grab_retargeted_base.py](../../tools/eval_grab_retargeted_base.py) — 独立 GPU/CPU 与无接触/物体开启运行协议、预设三组 gains、manifest 和逐步指标。
- [src/task/CmResidual/docs/README.md](../README.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) — 当前状态、版本与结果解释。
- `200/20`：[运行目录](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/)、[config.json](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp200_kd20_20260917_2051/eval.log)。
- `300/30`：[运行目录](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/)、[config.json](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp300_kd30_20260917_2056/eval.log)。
- `400/40`：[运行目录](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/)、[config.json](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/metrics.jsonl)、[eval.log](../../../../../outputs/CmResidual/cmresidual_v1133_gpu5_kp400_kd40_20260917_2057/eval.log)。

**原因**

V1.13.2 显示腕部实际位置约滞后两帧；用户批准先在固定 reference clock 和同一 GPU 物理协议下验证调节现有 position drive gains 是否可消除这一滞后，不改变研究目标或 residual 语义。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile` 修改的 Task/评估 Python：通过；`/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_reference_contract.py -q`：`23 passed`；`git diff --check`：通过。
- 三组均在 GPU5、GPU pipeline、seed42、1 env、canonical `num_subscenes=4` 完成 366 步且产生有效 manifest/config/metrics/log；运行配置分别回读 `200/20`、`300/30`、`400/40`，无接触且 finite，零 residual target 差异约 `1.19e-07` 浮点精度。
- 接触标记帧腕部/指尖 p95：`200/20=0.027618/0.037040 m`、`300/30=0.026923/0.036556 m`、`400/40=0.026527/0.036484 m`；三组最佳腕部位置对齐均滞后两帧。没有一组达到预定 `0.015/0.020 m` 门禁，故按计划停止，未运行物体开启 Gate 或 PPO。三组单 seed 的结果不能说明其他控制方案或泛化效果。

**保护与回滚**

回滚仅移除新路线可配置 gains 与评估工具增量、恢复本次文档；旧 DExplore 的 `200/20`、手指 `100/10`、旧运行及用户改动不触碰。三个 GPU5 运行目录保留为独立诊断证据。

## 2026-09-17 21:14:36 +0800 — V1.13.4 GRAB 探索性 PPO 启动

- timestamp: `2026-09-17 21:14:36 +0800`
- activity_id: `ACT-20260917-211422-CMRESIDUAL-V1134`
- modification_version: `V1.13.4`
- operation_category: `code`、`experiment`、`operation`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task 局部训练配置和运行工具）；GPU5 PPO 为 `L3`。
- approval: `user-approved`
- approval_basis: 用户审阅 [V1.13 PPO 替代方案](../plan/V1.13.md) 后回复“确定”，批准越过先前失败的追踪门禁启动探索性 PPO。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `7d750d61e3cfe79461f5d10d70e55de33a17f168`
- worktree_dirty: `true`（保留先前 V1.13 及其他 Task 未提交改动）
- scope: GRAB 固定 retargeted base、400/40 腕部增益、68-D observation、18-D residual、GPU5、64 env、seed42 的 2+8 epoch 探索性 PPO；旧 DExplore、数据、reward、共享 base 不变。
- run_id: `cmresidual_v1134_grab_ppo_20260917_2115`
- run_status: `FAILED`（运行于 `2026-09-17 21:15:17 +0800` 终止；2+8 epochs 完成，但探索性门禁失败）
- completed_at: `2026-09-17 21:15:17 +0800`；原运行证据回收与日志核对于 `2026-09-17 21:16:24 +0800` 完成。
- last_step: `20480` env-steps；last_epoch: `10`。
- best_metric: 不适用（无完成 episode，rl_games 最优 reward 为 `-inf`）；best_checkpoint: 不存在；latest_checkpoint: [epoch-10 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/stage10/CmResidualGrabRetargetedPPO_stage10/nn/last_CmResidualGrabRetargetedPPO_ep_10_rew_-inf.pth)。
- conclusion: `INVALID_IMPLEMENTATION`（本次 10-epoch 探索协议不足以完成 366-step episode，且末轮饱和超限）；PPO 接线与 checkpoint 工程证据局部 `SUPPORTED`；科研效果 `INCONCLUSIVE`。

**文件与证据**

- [src/task/CmResidual/docs/plan/V1.13.md](../plan/V1.13.md)、[third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabRetargetedPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabRetargetedPPO.yaml)、[src/task/CmResidual/tools/run_grab_retargeted_ppo.py](../../tools/run_grab_retargeted_ppo.py) — 批准协议、训练配置与带 manifest 的运行入口。
- [src/task/CmResidual/docs/README.md](../README.md)、[src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md)、[docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/CmResidual/docs/logs/activity_log.md](activity_log.md) — 当前状态、实验解释、版本指针与唯一运行时间线。
- [outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/) — [config.json](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/metrics.jsonl)、[train.log](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/train.log)、[checkpoint_validation.json](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/checkpoint_validation.json)、[epoch-2 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/stage2/CmResidualGrabRetargetedPPO_stage2/nn/last_CmResidualGrabRetargetedPPO_ep_2_rew_-inf.pth)、[epoch-10 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/stage10/CmResidualGrabRetargetedPPO_stage10/nn/last_CmResidualGrabRetargetedPPO_ep_10_rew_-inf.pth)。

**原因**

用户明确选择停止控制器对照并开始 PPO；因原追踪门禁失败，本轮限定为探索性接线与可学习性检查，不宣称 residual 或 Cmv2 的科研效果。

**验证**

- 启动前 `py_compile` 和 `git diff --check` 通过；Hydra 解析为 68-D observation、64 env，模型零均值、固定 sigma 约 0.1，actor/critic 均为 68-D。
- GPU5、64 env、seed42、horizon32 完成 `2+8=10` epochs、`20480` env-steps；第 2 epoch 后 checkpoint 从零训练且通过 reload，续跑到 epoch 10。10 行 [metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/metrics.jsonl) 均 finite；epoch 10 residual saturation=`0.107638888`，超过工程停止阈值 `0.10`，因此 `run_status=FAILED`、不追加训练。
- 每环境 `10×32=320` 步，少于完整 reference 的 `366` 步；[train.log](../../../../../outputs/CmResidual/cmresidual_v1134_grab_ppo_20260917_2115/train.log) 明确报告 `Max epochs reached before any env terminated at least once`，两个 checkpoint 的 reward 标签为 `-inf`。所以本次预算设计不能观察完整 episode 或成功率，不能解释为策略已学会/未学会。
- epoch-10 checkpoint 模型、optimizer、action 均 finite，确定性 reload action 最大差 `0`，固定 sigma 约 `0.1`。启动器在饱和检查时先报错，随后只读回收 stage10 TensorBoard 和 checkpoint 证据，补齐原运行的 metrics/manifest/validation；未重复训练或改写 checkpoint。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmResidual/tools/run_grab_retargeted_ppo.py`、`git diff --check`：通过；`/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_reference_contract.py -q`：`23 passed`。

**保护与回滚**

仅撤销本次 PPO 配置、runner 与记录即可回滚实现；新运行目录保留为证据，旧 DExplore、已有运行和用户改动不触碰。

## 2026-09-17 21:39:45 +0800 — V1.14.1 真实物体 zero-residual reference 基线启动

- timestamp: `2026-09-17 21:39:45 +0800`
- activity_id: `ACT-20260917-213945-CMRESIDUAL-V1141-PHYSICAL-ZERO`
- modification_version: `V1.14.1`
- operation_category: `code`、`experiment`、`operation`、`diagnostic`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task-local object transition 指标和定向测试）；GPU5 物理基线为 `L3`。
- approval: `user-approved`
- approval_basis: 用户确认 [V1.14 最终计划](../plan/V1.14.md) 并授权按 reference 步间统计冻结 reward、课程预算和停止阈值后执行。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `80f04145ad48bd7db875b20116b6d2fc3e10863e`
- worktree_dirty: `true`（保留 V1.13、根级、ObjectInteractionCm 及用户已有未提交改动；本次只触及下列 CmResidual 文件与版本指针）。
- scope: 真实 GRAB object、400/40 wrist drive、zero residual、GPU5/1 env/seed42/366 steps；记录 object 当前 pose、SO(3) rotation error 和 local one-step transition error，不改 reward、window reset、Cmv2、reference、资产或共享 base。
- run_id: `cmresidual_v1141_grab_physical_zero_20260917_2139`
- run_status: `COMPLETED`；于 `2026-09-17 21:40:53 +0800` 正常结束。
- initial_checkpoint: `null`；best_metric/checkpoint: 不适用；last_step: `366`；last_epoch: `null`。
- conclusion: `SUPPORTED`（运行/索引/zero-residual/object-transition 指标工程接线）；object reference tracking、residual 效果和抓取效果 `INCONCLUSIVE`。

**文件与原因**

- [src/task/CmResidual/docs/plan/V1.14.md](../plan/V1.14.md) — 已定稿的 reference-transition reward、window curriculum、预算和停止边界。
- [src/task/CmResidual/tools/eval_grab_retargeted_base.py](../../tools/eval_grab_retargeted_base.py) — physical 协议新增实际/reference object pose、SO(3) pose error 与相邻 local transition error，终止步使用 terminal object pose 避免 reset 污染。
- [src/task/CmResidual/tests/test_grab_retargeted_eval.py](../../tests/test_grab_retargeted_eval.py) — 验证 xyzw pose matrix、局部 translation 和旋转测地误差。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/CmResidual/docs/README.md](../README.md) — 指向本次当前 Task 版本和正在进行的 Gate。

V1.13.4 的 PPO 未完成任何 366-step episode，且 reward 不包含 object reference tracking；本 Gate 先取得真实 object、固定索引和 zero-residual 下的可审计基线，作为是否实现新 reward/window 的工程前置条件。

**启动前验证与回滚**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmResidual/tools/eval_grab_retargeted_base.py`：通过；`/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_grab_retargeted_eval.py src/task/CmResidual/tests/test_reference_contract.py -q`：`25 passed`（6 条现有 Hydra compatibility warnings）；`git diff --check`：通过。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/)、[config.json](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/metrics.jsonl) 与 [eval.log](../../../../../outputs/CmResidual/cmresidual_v1141_grab_physical_zero_20260917_2139/eval.log) 已生成。366 步全程 finite、zero residual target delta 最大 `1.19e-07`、无异常 reset，故 A 的工程硬门通过。
- actual-vs-reference object position/rotation 的 mean/p95/max 为 `0.047775/0.164753/0.221184 m` 与 `0.268102/0.543927/0.599684 rad`；local one-step transition translation/rotation error 的 mean/p95/max 为 `0.004155/0.015792/0.020499 m` 与 `0.018703/0.051554/0.170197 rad`。这些偏差不是准确 nominal tracking 的证据；但 A 未设置其为阻断阈值，按 V1.14 仅允许进入受控 B 实现，不允许宣称抓取或 residual 改善。
- 回滚只撤销本次 V1.14.1 工具/测试、版本指针、README 和本活动条目；新 run 目录一旦创建保留为证据，不删除旧 outputs、数据、checkpoint 或用户改动。

## 2026-09-17 21:46:56 +0800 — V1.14.2 reference-transition reward 与 64-step PPO 启动

- timestamp: `2026-09-17 21:46:56 +0800`
- activity_id: `ACT-20260917-214656-CMRESIDUAL-V1142-W64-PPO`
- modification_version: `V1.14.2`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（reward、reference reset/window、Task 注册和训练合同）；GPU5 PPO 为 `L3`。
- approval: `user-approved`
- approval_basis: 用户确认 [V1.14 最终计划](../plan/V1.14.md) 并授权冻结的 reward、课程、预算与停止阈值。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `80f04145ad48bd7db875b20116b6d2fc3e10863e`
- worktree_dirty: `true`（本次 CmResidual/vendor 配置、registry 和记录与既有用户差异并存）。
- scope: random-uniform start `k∈[0,302]`、64-step window、`q/dq/object` 同 index reset；冻结 pose/transition tracking reward；GPU5/64 env/seed42/horizon64 的 2+8 epoch stage，saturation `>0.05` 停止。
- run_id: `cmresidual_v1142_reftrack_w64_20260917_2147`
- run_status: `FAILED`；于 `2026-09-17 21:48:29 +0800` 终止，未启动 continuation。
- initial_checkpoint: `null`；last_step: `8192`；last_epoch: `2`；best_metric: 不适用；best_checkpoint: 不存在；latest_checkpoint: [epoch-2 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/smoke/CmResidualGrabReferenceTransition_smoke/nn/last_CmResidualGrabReferenceTransitionPPO_ep_2_rew__-1168.13_.pth)。
- conclusion: `INVALID_IMPLEMENTATION`（epoch-1 saturation 超过冻结阈值）；reset/reward/checkpoint 工程接线局部 `SUPPORTED`；reference tracking、residual utility 和抓取效果 `INCONCLUSIVE`。

**文件与验证**

- [src/task/CmResidual/docs/plan/V1.14.md](../plan/V1.14.md) — 最终 reward/课程/预算边界。
- [reference_provider.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py)、[task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — indexed reset、SO(3)/local transition metrics 与 opt-in reward；legacy lift/distance 路径保持。
- [third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py) — 注册 V1.14 独立 task name，不改变既有 CmResidual task aliases。
- [CmResidualGrabReferenceTransition.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceTransition.yaml)、[CmResidualGrabReferenceTransitionPPO.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabReferenceTransitionPPO.yaml)、[run_grab_reference_transition_ppo.py](../../tools/run_grab_reference_transition_ppo.py) — 专用配置与 manifest/checkpoint runner。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/)、[config.json](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/config.json)、[run_manifest.json](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/run_manifest.json)、[metrics.jsonl](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/metrics.jsonl) 与 [train.log](../../../../../outputs/CmResidual/cmresidual_v1142_reftrack_w64_20260917_2147/train.log)；最终 `checkpoint_validation.json` 未生成，因为 saturation Gate 在 epoch-2 checkpoint reload 后、写最终 validation 前停止。
- smoke 完成 epoch 1/2：saturation=`0.053819/0.048611`、residual RMS=`0.105158/0.106061`；epoch 1 已超过 `0.05` 硬阈值，故 manifest 终态为 `FAILED`，不续跑 epoch 3–10，不进入 128/366 window。epoch-2 checkpoint 可由 runner 重新加载并通过 finite/deterministic action 门；无 `best_checkpoint`。tracking 指标从 TensorBoard 读取：position=`0.066881/0.181973 m`、rotation=`0.535478/0.701575 rad`、transition translation=`0.003643/0.006629 m`、transition rotation=`0.017725/0.041125 rad`（epoch 1/2）。

**原因**

V1.13.4 的 lift/distance PPO 在每 env 仅覆盖 320 步，且没有 object reference tracking 目标。V1.14.2 将 reset 的 robot/object state 与同一随机 reference index 对齐，并冻结 reference pose/transition reward；本次硬阈值停止防止在 residual 已触及 joint limit 的情况下继续把训练曲线解释为效果。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_grab_retargeted_eval.py src/task/CmResidual/tests/test_reference_contract.py -q`：`26 passed`；GPU5/1-env/3-step smoke 从 start index 78 到 81，obs/reward/transition finite；`py_compile`、Hydra 配置合同和 `git diff --check` 通过。
- [src/task/CmResidual/tests/test_reference_contract.py](../../tests/test_reference_contract.py) — indexed provider reset 单测；[src/task/CmResidual/tools/eval_grab_retargeted_base.py](../../tools/eval_grab_retargeted_base.py) — A 阶段 physical object transition 指标。
- [docs/current_versions.yaml](../../../../../docs/current_versions.yaml)、[src/task/CmResidual/docs/README.md](../README.md)、[src/task/CmResidual/docs/logs/experiment_log.md](experiment_log.md) 与本 [activity_log.md](activity_log.md) — V1.14.2 指针、终态和科研结论边界。

**保护边界**

- [docs/logs/activity_log.md](../../../../../docs/logs/activity_log.md) 与 [src/task/ObjectInteractionCm/docs/logs/activity_log.md](../../../ObjectInteractionCm/docs/logs/activity_log.md) 在本次工作开始前已处于 dirty 状态；未读取、修改或归入 V1.14.2。旧 V1.13 未提交计划/指导、DexYCB evaluator、旧 outputs/checkpoint 和共享 `src/base/` 同样未触碰。

## 2026-09-17 21:58:37 +0800 — V1.14.3 saturation 分量与 margin 诊断

- timestamp: `2026-09-17 21:58:37 +0800`
- activity_id: `ACT-20260917-215837-CMRESIDUAL-V1143-SATURATION-DIAG`
- modification_version: `V1.14.3`
- operation_category: `diagnostic`、`operation`、`documentation`
- task_mode: `read-only/diagnostic`
- change_level: `L0`
- approval: `user-approved`
- approval_basis: 用户明确要求“诊断一下”。
- skills_used: `research-experiment-workflow`
- branch: `oyx`
- base_commit: `80f04145ad48bd7db875b20116b6d2fc3e10863e`
- worktree_dirty: `true`（只读诊断；未改代码、配置、数据或 checkpoint）。
- scope: GPU5、64 random starts、63 steps、固定 seed42 和 `sigma=0.1` Gaussian action；不 optimizer update、不加载/改写 checkpoint 训练状态。
- run_id: `cmresidual_v1143_saturation_diag_20260917_2200`
- run_status: `COMPLETED`；last_step: `4032` env-steps；best_metric/checkpoint: 不适用。
- conclusion: `SUPPORTED`（局部 saturation/margin 诊断）；scale 归因、修复效果和科研效果 `INCONCLUSIVE`。

**证据、原因与验证**

- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1143_saturation_diag_20260917_2200/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1143_saturation_diag_20260917_2200/run_manifest.json)、[summary](../../../../../outputs/CmResidual/cmresidual_v1143_saturation_diag_20260917_2200/summary.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1143_saturation_diag_20260917_2200/metrics.jsonl)。总体 saturation=`4.147%`，其中 native DOF 8/14/15 为 `41.57%/21.65%/11.43%`，其他 15 个维度均为零。
- 三个分量的 mean requested absolute residual 均约 `0.005 rad`，而 base margin 的最小值均为 `0`（mean margin 分别 `0.0378/0.1631/0.0672 rad`）。这说明瞬时越界由 reference nominal target 落在 joint boundary 后、同方向 residual 触发；不是 wrist translation/rotation 的 global scale 造成。
- 此回放用固定初始 exploration sigma，而非失败 run 的原始 stochastic action trace；它解释结构性来源但不能精确复现训练的 `5.3819%`。缩小所有 scale 会同时削弱未饱和的 15 个维度，不能作为本诊断后的自动修复。
- [V1.14 最终计划](../plan/V1.14.md)、[task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[reference_provider.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py)、[CmResidualGrabReferenceTransition.yaml](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceTransition.yaml)、[runner](../../tools/run_grab_reference_transition_ppo.py)、[tests](../../tests/test_reference_contract.py)、[A evaluator](../../tools/eval_grab_retargeted_base.py)、[experiment log](experiment_log.md)、[README](../README.md)、[current_versions](../../../../../docs/current_versions.yaml) 仅作 V1.14.1/2 已实现边界的导航；本诊断未修改它们。

## 2026-09-17 22:12:25 +0800 — V1.14.4 可行动作 authority 映射与 64-step 重跑

- timestamp: `2026-09-17 22:12:25 +0800`
- activity_id: `ACT-20260917-221500-CMRESIDUAL-V1144-FEASIBLE`
- modification_version: `V1.14.4`
- operation_category: `code`、`diagnostic`、`experiment`、`operation`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task-local action mapping、诊断指标、runner 与定向测试）；GPU5 smoke/PPO 为 `L3`。
- approval: `user-approved`
- approval_basis: 用户审阅 [V1.14 最终计划](../plan/V1.14.md) 中 V1.14.3 后局部修订并明确回复“你继续吧”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `80f04145ad48bd7db875b20116b6d2fc3e10863e`
- worktree_dirty: `true`（保留先前根级、ObjectInteractionCm、V1.13 及用户已有未提交改动；未覆盖、reset 或纳入本次实现）。
- scope: 不改变 reference、reward、18-D action、configured scale、joint limits、mimic、PPO 超参数、Cmv2 或共享 `src/base`；仅将 reference-boundary 的 outward residual 映射到实际 source/mimic 可行余量，并新增 authority-limited 诊断。
- run_id: `cmresidual_v1144_feasible_authority_smoke_20260917_2215`，`run_status: COMPLETED`；GPU5/1 env/seed42、3 zero + 16 fixed-nonzero steps，`last_step=19`、无 checkpoint，target saturation 最大 `0`。
- run_id: `cmresidual_v1144_reftrack_feasible_w64_20260917_2215`，`run_status: COMPLETED`；GPU5/64 env/seed42/64-step、2+8 epoch，`last_step=40960`、`last_epoch=10`；latest/best checkpoint 为 [epoch-10](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/continuation/CmResidualGrabReferenceTransition_continuation/nn/last_CmResidualGrabReferenceTransitionPPO_ep_10_rew__-2736.07_.pth)。
- conclusion: mapping、smoke、PPO saturation gate 与 checkpoint reload 工程 `SUPPORTED`；reference tracking、residual utility 和抓取效果 `INCONCLUSIVE`。

**原因**

V1.14.3 的证据表明 saturation 来自少数 nominal reference target 已在 joint boundary 的外向 residual；全局缩小 scale 会无差别削弱其余 15 个分量，故按已批准计划改为局部可行 authority。

**文件**

- [V1.14 最终计划](../plan/V1.14.md) — 将经批准的局部修订标为最终版：不缩小 global scale，而在 base target 的剩余可行区间内逐方向收缩 authority。
- [action_mapping.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/action_mapping.py) — 按真实 native source DOF（而非 `MIMIC_SOURCE` 的独立分量位置）回推 mimic bounds；zero action 不标记为 authority-limited，outward 非零 action 在 boundary 处安全收缩且仍保持 mimic coupling。
- [task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 仅为 reference path 导出 `residual_authority_limited_ratio`；legacy physical residual path 保持原行为。
- [tests](../../tests/test_reference_contract.py) — 覆盖 boundary outward/inward、zero parity、mimic 以及 saturation=0。
- [runner](../../tools/run_grab_reference_transition_ppo.py) — 接收 `--modification-version`，使新 manifest 记录实际 V1.14.4；未改变训练超参数或 stop gate。`py_compile` 与 `git diff --check` 均通过。
- [smoke 目录](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/)、[smoke manifest](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/run_manifest.json)、[smoke summary](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/summary.json)、[smoke metrics](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/metrics.jsonl) 与 [smoke log](../../../../../outputs/CmResidual/cmresidual_v1144_feasible_authority_smoke_20260917_2215/smoke.log)：zero action authority-limited=`0`，fixed nonzero action 最高 authority-limited ratio=`0.1111111`，全程 finite、saturation=`0`。
- [PPO 目录](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/)、[config](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/config.json)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v1144_reftrack_feasible_w64_20260917_2215/checkpoint_validation.json) 与上述 checkpoint：epoch 1–10 的 saturation 均为 `0`，model/optimizer/action finite，reload action diff=`0`，到达锁定 `2+8` budget。
- [experiment log](experiment_log.md)、[README](../README.md)、[current_versions](../../../../../docs/current_versions.yaml) — 记录终态和 V1.14.4 指针。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_reference_contract.py src/task/CmResidual/tests/test_grab_retargeted_eval.py -q`：`27 passed`（6 条既有 Hydra compatibility warnings）；`py_compile`、`git diff --check` 均通过。

**保护与回滚**

回滚只撤销本次 action mapping、Task 指标、runner 参数、定向测试和 V1.14.4 文档/指针；新 outputs 保留为可审计证据。不删除或修改历史 V1.14.1–3 outputs、reference、checkpoint、数据、共享 base 或既有用户改动。

## 2026-09-17 22:40:00 +0800 — V1.14.5 冻结 Cmv2 action evaluator 与 CmBuffer

- timestamp: `2026-09-17 22:40:00 +0800`
- activity_id: `ACT-20260917-224000-CMRESIDUAL-V1145-CMBUFFER`
- modification_version: `V1.14.5`
- operation_category: `code`、`diagnostic`、`operation`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task-local evaluator/buffer/config/runner/test）；GPU5 smoke 为 `L3`。
- approval: `user-approved`
- approval_basis: 用户明确指定 action-conditioned Cmv2 不进 actor/critic/更新、CmBuffer 不存点云并回复“可以，然后开始吧”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `80f04145ad48bd7db875b20116b6d2fc3e10863e`
- worktree_dirty: `true`（提交时只暂存本次 CmResidual/V1.14 路径；根级、ObjectInteractionCm、V1.13 及用户已有差异保持不触碰）。
- scope: PPO actor/critic 保持 68-D；Cmv2 仅对实际 executed residual 的 nominal controller/FK sweep 推理。CmBuffer 每 rank 写 compressed shard，仅含可重建 state/action/effect label，不含 point/normals/flow；Cmv2 `latest.pt` 保持 eval/`requires_grad=False`，不入 reward 或 optimizer。
- run_id: `cmresidual_v1145_cmbuffer_smoke_20260917_2240`
- run_status: `COMPLETED`；GPU5/1 env/seed42，`last_step=1`，无 checkpoint。
- conclusion: buffer/evaluator 工程接线 `SUPPORTED`；Cmv2 accuracy、候选排序、微调资格、PPO/research effect `INCONCLUSIVE`。

**原因**

V1.14 的 current PPO 已完成 no-Cm baseline；用户要求按指导中的 action-conditioned effect 接口积累可审计的真实执行 transition，而不是把历史 token context 拼入 actor/critic，也不在没有标签覆盖/诊断前自动微调 Cmv2。

**文件**

- [V1.14 最终计划](../plan/V1.14.md) — 新增最终 V1.14.5 合同、冻结边界、buffer schema、验证和三卡训练前置条件。
- [cm_buffer.py](../../cm_buffer.py) — per-rank compressed NPZ shard/manifest；字段 batch-shape 一致性、finite、schema 与点云禁令受检。
- [task.py](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — pre-physics 保存 executed action prediction/pre-state，post-physics 写 PhysX local effect label；不改 PPO observation/reward。
- [CmBuffer task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceTransitionCmBuffer.yaml)、[CmBuffer PPO config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabReferenceTransitionCmBufferPPO.yaml)、[registry](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py)、[runner](../../tools/run_grab_reference_transition_ppo.py) — 隔离 Cmv2 buffer task，runner 以 `--cm-buffer` 选择该 task；旧 V1.14.4 config/runner 默认保持 no-Cm。
- [buffer test](../../tests/test_cm_buffer.py) — 无点云 chunk round-trip 与 schema；[README](../README.md)、[experiment log](experiment_log.md)、[current versions](../../../../../docs/current_versions.yaml) — 状态与版本入口。
- [smoke 目录](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/run_manifest.json)、[summary](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/summary.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/config.json)、[buffer manifest](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/cm_buffer/rank_000/manifest.json) 与 [chunk](../../../../../outputs/CmResidual/cmresidual_v1145_cmbuffer_smoke_20260917_2240/cm_buffer/rank_000/chunk_000000.npz)：1 条 finite sample、saturation=`0`，必需状态/metadata 存在且不含 point/normals/flow。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_cm_buffer.py -q`：`1 passed`；Hydra config contract、`py_compile` 与 `git diff --check` 通过。
- GPU5 smoke 使用真实 pinned Cmv2 checkpoint；该 smoke 只证明输入/输出、label 时序与存储合同，不能把单样本 prediction error 解释为模型可微调或 policy 效果。

**保护与回滚**

回滚仅删除 V1.14.5 的 CmBuffer module/config/registry/runner option/test 和文档记录；生成 shard 保留为证据。三卡分布式 capacity probe 与正式训练尚未启动，需在 rank-to-GPU、每卡容量和有限总 env-step/人工停止记录确定后单列 operation。

## 2026-09-18 00:10:06 +0800 — V1.15 Cmv2 effect-conditioned actor

- timestamp: `2026-09-18 00:10:06 +0800`
- activity_id: `ACT-20260918-110000-CMRESIDUAL-V115-ACTOR`
- modification_version: `V1.15`
- operation_category: `architecture`、`code`、`experiment`、`operation`、`documentation`
- task_mode: `change`，随后 `run-only/operation`
- change_level: `L2`（Task-local actor observation、PPO builder、config/runner/test）；GPU5 smoke 为 `L3`。
- approval: `user-approved`
- approval_basis: 用户确认 V1.15 方案“就按照你想的那样，开始吧”；此前明确 Cmv2 不进入 critic，CmBuffer 不存点云且只积累可重建必要状态。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `a4b3d447d4c0ce0f4e51a97eee08edc612352f57`
- worktree_dirty: `true`（只暂存本次 V1.15 明列路径；根 activity、ObjectInteractionCm activity、V1.12–14 指导、V1.13 plan 与诊断工具保持不触碰）。
- scope: actor raw observation 为 `[base68, token16x40, normalized predicted6, reference6, error6]` 共 `726-D`；actor 内 token `40→128`、单层 4-head masked self-attention 与 effect-conditioned attention pooling 后为 `214-D`。critic 只切 `68-D` prefix；冻结 Cmv2 保持 inference/eval/`requires_grad=False`，不进 reward、critic、optimizer 或 buffer 在线更新。actor 使用 action 前的 zero residual nominal sweep；CmBuffer 仍独立标记实际 executed residual 的 pre/post PhysX transition。
- run_id: `cmresidual_v115_cmv2_actor_smoke_20260918_1100`
- run_status: `COMPLETED`；GPU5、seed42、1 env、8 step、1 PPO epoch，`last_step=8`、`last_epoch=1`，最近 checkpoint 已生成。
- best_metric: `checkpoint_reload_action_max_abs_diff=0.0`；checkpoint 为 `last_CmResidualGrabReferenceTransitionCmv2ActorPPO_ep_1_rew_-inf.pth`。
- conclusion: actor/critic/Cmv2/CmBuffer 工程合同 `SUPPORTED`；attention utility、effect prediction accuracy、Cmv2 微调资格、PPO tracking 与抓取科研效果 `INCONCLUSIVE`。

**文件**

- [V1.15 最终计划](../plan/V1.15.md)、[V1.15 用户指导](../指导/V1.15.md)、[README](../README.md)、[current versions](../../../../../docs/current_versions.yaml) — 版本、批准边界和当前状态入口。
- [adapter](../../cm_v2_adapter.py)、[action evaluator](../../cm_v2_action_evaluator.py)、[Task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 规范 token、action 前 base-effect actor transport，以及实际 residual buffer label 的语义分离。
- [PPO builder](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_network_builder.py)、[train registry](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/train.py)、[task registry](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py) — masked attention actor 与 68-D critic 的显式非对称合同。
- [task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceTransitionCmv2Actor.yaml)、[PPO config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabReferenceTransitionCmv2ActorPPO.yaml)、[runner](../../tools/run_grab_reference_transition_ppo.py)、[contract helper](../../tools/run_ppo_stability.py)、[network contract test](../../tests/test_reference_contract.py)、[evaluator test](../../tests/test_cmv2_action_evaluator.py)、[experiment log](experiment_log.md) — opt-in V1.15 config 和可复现的 `--cmv2-actor --smoke` 入口。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/run_manifest.json)、[resolved config](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/checkpoint_validation.json)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/smoke/CmResidualGrabReferenceTransition_smoke/nn/last_CmResidualGrabReferenceTransitionCmv2ActorPPO_ep_1_rew_-inf.pth)、[CmBuffer manifest](../../../../../outputs/CmResidual/cmresidual_v115_cmv2_actor_smoke_20260918_1100/cm_buffer/rank_000/manifest.json) — GPU5 smoke 证据。

**原因**

V1.14.5 已证明 executed-action Cmv2 buffer 能无点云记录，但未让 Cmv2 参与策略。V1.15 按用户指导把其限定为 actor 的 nominal-base effect context，并将 critic/reward/frozen checkpoint/update 边界保留不变，避免将 action 后预测或未来物理状态泄漏到 actor observation。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_cmv2_adapter.py src/task/CmResidual/tests/test_cmv2_action_evaluator.py src/task/CmResidual/tests/test_reference_contract.py src/task/CmResidual/tests/test_cm_buffer.py -q -k 'cmv2 or v115 or cm_buffer'`：`10 passed, 25 deselected`；覆盖 16×40 encoding、evaluator、all-invalid token fallback、726→214 actor/68 critic、Hydra config 与 buffer schema。
- GPU5 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_grab_reference_transition_ppo.py --cmv2-actor --smoke --gpu 5 --window-length 64 --modification-version V1.15 --activity-id ACT-20260918-110000-CMRESIDUAL-V115-ACTOR --run-id cmresidual_v115_cmv2_actor_smoke_20260918_1100`：完成；checkpoint/optimizer/action finite、reload diff=`0.0`、sigma=`0.1`、saturation=`0.0`；CmBuffer 有 8 个 finite shard，字段无 `point`/`normal`/`flow`。
- `py_compile` 与 `git diff --check` 通过。smoke 在 episode 完成前到达 1 epoch，reward 为 `-inf` 是该短预算预期现象，不能用于训练效果判断。

**保护与回滚**

回滚只撤销本条的 V1.15 adapter/evaluator/Task/network/config/registry/runner/test/docs 差异；保留 V1.14 代码、reference、Cmv2 checkpoint、已有 outputs 与用户未提交路径。生成的 V1.15 output/CmBuffer 是审计证据，不纳入 Git。GPU0/1/2 分布式训练仍未启动：GPU2 被其他用户占用，且 V1.15 plan 要求先单列容量/rank-output/人工停止方案后才能启动。

## 2026-09-18 00:39:00 +0800 — V1.16 GPU nominal Cmv2、transition-only buffer 与 DDP runner

- timestamp: 2026-09-18 00:39:00 +0800
- activity_id: ACT-20260918-003000-CMRESIDUAL-V116-GPU
- modification_version: V1.16
- operation_category: architecture、code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L2（nominal sweep 实现、buffer schema、base-only RMS）；GPU5 smoke 与三卡 capacity/formal operation 为 L3。
- approval: user-approved
- approval_basis: 用户提供 [V1.16 指导](../指导/V1.16.md)，审阅 [V1.16 最终计划](../plan/V1.16.md) 后明确回复“确定”。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: a1b5073ea3e17965d7b7f9ab365ad0c3c08fc77c
- worktree_dirty: true（只暂存本次 V1.16 明列路径；根 activity、ObjectInteractionCm activity、V1.12–16 用户指导、V1.13 plan 与诊断工具保持不触碰）。
- scope: V1.15 actor raw/effective 726→214-D 和 68-D critic 不变。actor nominal Cmv2 改为 actual GPU QUERY_LINKS 到 next GPU reference link pose 的 sweep，禁止该热路径 CPU/NumPy FK 或 legacy action evaluator。新 buffer schema 是 transition-only v2 的固定 16 字段；不预测 executed residual Cmv2，GPU staging 到 shard 时才批量 CPU copy。V1.16 model 仅为 base68 启用 RunningMeanStd；token/effect 不做全局 RMS。
- run_id: cmresidual_v116_gpu_nominal_smoke_20260918_0030
- run_status: COMPLETED；GPU5、seed42、1 env、8 step、1 PPO epoch，last_step=8、last_epoch=1。
- best_metric: checkpoint_reload_action_max_abs_diff=0.0；checkpoint 为 last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_1_rew_-inf.pth。
- conclusion: GPU nominal sweep、base68-only RMS、transition-only staging buffer 与 single-rank PPO wiring 工程合同 SUPPORTED；DDP capacity、attention utility、effect accuracy、PPO tracking、Cmv2 微调资格与抓取效果 INCONCLUSIVE。

**文件**

- [V1.16 最终计划](../plan/V1.16.md)、[V1.16 用户指导](../指导/V1.16.md)、[README](../README.md)、[current versions](../../../../../docs/current_versions.yaml)、[experiment log](experiment_log.md) — 版本、边界、运行与结论入口。
- [Cmv2 adapter](../../cm_v2_adapter.py)、[transition buffer](../../cm_buffer.py)、[Task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — actor 跳过 legacy flattened context、GPU nominal sweep、v2 field whitelist/staging 与 actual transition label。
- [Cmv2 model wrapper](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_models.py)、[train registry](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/train.py)、[task registry](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py) — 仅 base68 的 RMS 与 V1.16 model/task 注册。
- [V1.16 task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceTransitionCmv2ActorV116.yaml)、[V1.16 PPO config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabReferenceTransitionCmv2ActorV116PPO.yaml)、[GPU5 smoke runner](../../tools/run_grab_reference_transition_ppo.py)、[DDP runner](../../tools/run_cmv2_actor_distributed.py)、[model contract helper](../../tools/run_ppo_stability.py) — V1.16 opt-in 路径、manual-stop formal mode 与受控 capacity launcher。
- [buffer tests](../../tests/test_cm_buffer.py)、[adapter tests](../../tests/test_cmv2_adapter.py)、[reference/model/runner tests](../../tests/test_reference_contract.py) — v2 whitelist/staging、no-context adapter、base-only RMS、GPU-sweep static contract 和 GPU0/1/2 launcher 参数。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/run_manifest.json)、[resolved config](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/checkpoint_validation.json)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/smoke/CmResidualGrabReferenceTransition_smoke/nn/last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_1_rew_-inf.pth)、[v2 buffer manifest](../../../../../outputs/CmResidual/cmresidual_v116_gpu_nominal_smoke_20260918_0030/cm_buffer/rank_000/manifest.json) — GPU5 工程证据。

**原因**

V1.15 每个 actor observation 经 legacy evaluator 触发 GPU→CPU→NumPy FK；同时 buffer 为实际 action 再执行一次 Cmv2。V1.16 用 pinned reference GPU link poses 消除 actor hot path 的 FK/host transfer，并将 Cmv2 调用限制为 actor nominal-base 一次；buffer 只保存不可替代 transition，后续可从 pin 的 asset/reference/state 离线重建 executed-action Cmv2 诊断。

**验证**

- graspenv pytest 命令运行 CmBuffer、Cmv2 adapter/evaluator、reference contract：14 passed, 25 deselected；py_compile、DDP runner --help 与 git diff --check 通过。
- GPU5 smoke 命令使用 --cmv2-actor --v116 --smoke：完成；checkpoint/optimizer/action finite、reload diff=0.0、sigma=0.1、saturation=0.0。v2 manifest 有 8 个 finite sample、16 个白名单字段、无 point/normal/flow；日志没有 legacy build_nominal_link_poses 的 CPU FK warning。
- GPU0/1/2 仅完成只读 preflight：当前 memory used 约为 2553/776/23215 MiB，GPU2 超出 DDP launcher 4096 MiB gate，故未启动 3×128 capacity probe 或正式训练。

**保护与回滚**

回滚只撤销 V1.16 adapter/buffer/Task/model/config/registry/runner/test/docs 差异；V1.14/V1.15 config、legacy evaluator、buffer schema、checkpoint、reference、outputs 和用户未提交路径保留。DDP runner 在 GPU2 空闲前拒绝启动；capacity probe 只允许 3×128 的单 epoch 工程测量，3×256 或 manual-stop formal run 需以 probe 的显存/throughput/三 rank manifest 为依据另行启动。

## 2026-09-18 00:51:18 +0800 — V1.16.1 DDP bootstrap 修复与 3×128 capacity probe 启动

- timestamp: 2026-09-18 00:51:18 +0800
- activity_id: ACT-20260918-005118-CMRESIDUAL-V1161-DDP
- modification_version: V1.16.1
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L1（Task-local launcher/lifecycle 修复）；3×128 capacity probe 为 L3。
- approval: user-approved
- approval_basis: 用户明确要求只修正 torchrun bootstrap 与 CmBuffer graceful shutdown，再实际运行 3×128 一轮 capacity probe；并明确禁止继续改变 actor/Cm/reward/critic 结构。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: eb16ce8c8a3b48caee5dadb31d9c9489243a3e40
- worktree_dirty: true（保留进入本次工作前的根 activity、ObjectInteractionCm activity、用户指导/plan 草稿与独立工具；只暂存本条明列路径）。
- scope: DDP runner 改由 active Python 的 `-m torch.distributed.run` 执行真实 Python bootstrap，不再把 Python executable 误传为 training script；Task 的 atexit cleanup 调用 CmTransitionBuffer.close() flush 未满 staging shard。保持 V1.16 的 actor 726→214、critic base68、frozen Cmv2、nominal GPU sweep、reward、buffer 固定 16 字段和 3×128 单 epoch预算不变。
- run_id: cmresidual_v1161_ddp_3x128_capacity_20260918_005118
- run_status: STOPPED；GPU capacity preflight 在创建 run manifest 或启动 torchrun 前拒绝执行，故没有运行目录、metrics、train log 或 checkpoint。
- last_step: N/A；last_epoch: N/A；best_metric: N/A；checkpoint: N/A。
- exit_reason: GPU2 使用 `17457 MiB`，超过 `--max-used-mib=4096` 的安全门限；GPU0/1/2 分别为 `2553/776/17457 MiB`。未启动任何 worker 或仿真进程。
- conclusion: INCONCLUSIVE（启动前代码验证通过；capacity 运行尚未形成证据）。

**文件**

- [V1.16 最终计划](../plan/V1.16.md)、[DDP runner](../../tools/run_cmv2_actor_distributed.py)、[DDP bootstrap](../../tools/run_cmv2_actor_bootstrap.py) — 使用真实 `.py` training script 与当前 conda Python 的分布式 launcher。
- [Task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[CmBuffer](../../cm_buffer.py) — 正常解释器退出或 Ctrl-C 时 flush 已追加、尚未满 shard 的 transition-only staging 数据。
- [buffer tests](../../tests/test_cm_buffer.py)、[runner contract tests](../../tests/test_reference_contract.py)、[README](../README.md)、[current versions](../../../../../docs/current_versions.yaml) — tail flush、bootstrap command 与状态入口。

**原因**

V1.16 的 launcher 将 Python executable 放在 torchrun 的 training-script 位置，而且此环境没有 PATH 中的 `torchrun` executable；两者都会阻止真实 DDP 启动。CmBuffer 的 episode-end flush 不覆盖在 episode 中间发生的正常退出。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile src/task/CmResidual/tools/run_cmv2_actor_distributed.py src/task/CmResidual/tools/run_cmv2_actor_bootstrap.py src/task/CmResidual/cm_buffer.py third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_cm_buffer.py src/task/CmResidual/tests/test_reference_contract.py -q -k 'cm_buffer or v116'`：`7 passed, 26 deselected`；覆盖 partial staging close 和真实 bootstrap script 合同。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m torch.distributed.run --help`：通过。此环境没有 PATH 中的 `torchrun` executable，故显式使用等价且可审计的 module 入口。
- capacity 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_cmv2_actor_distributed.py --gpus 0,1,2 --envs-per-rank 128 --capacity-probe --activity-id ACT-20260918-005118-CMRESIDUAL-V1161-DDP --modification-version V1.16.1 --run-id cmresidual_v1161_ddp_3x128_capacity_20260918_005118`。仅在每张 GPU 使用显存不超过 4096 MiB 时启动；要求三份 rank buffer manifest、finite checkpoint/action/optimizer，结论仅为工程 capacity。
- `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`：GPU0/1/2 为 `2553/776/17457 MiB`，GPU2 不满足 gate；未执行上述 capacity 命令，避免占用或干扰现有 GPU2 任务。

**保护与回滚**

回滚仅移除 bootstrap、runner command 变化、Task close hook、定向测试和本条记录；V1.16 算法、schema、训练变量、Cmv2 checkpoint、输入 reference 与已有 outputs 不变。正常 Python/KeyboardInterrupt cleanup 可落盘 partial shard；SIGKILL、进程崩溃或断电不承诺 tail recovery。

## 2026-09-18 01:03:56 +0800 — V1.16.2 GPU0/1/3 3×128 capacity probe 启动

- timestamp: 2026-09-18 01:03:56 +0800
- activity_id: ACT-20260918-010356-CMRESIDUAL-V1162-DDP
- modification_version: V1.16.2
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L1（Task-local physical GPU allowlist）；3×128 capacity probe 为 L3。
- approval: user-approved
- approval_basis: 用户明确指定后续全量训练使用 GPU 0、1、3，确认 256 env/rank 尚未经验证后，要求先实际测量显存。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 9fc85052007e03b546da61826985cb889d9ba92e
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户指导/plan 草稿与独立工具；只暂存本条明列路径）。
- scope: 只将 V1.16 DDP physical GPU allowlist 从 0/1/2 改为用户指定的 0/1/3；CUDA_VISIBLE_DEVICES 会将其重映射为各 rank 的 local cuda:0/1/2。运行 3 rank × 128 env、1 epoch capacity probe，记录每 rank buffer、checkpoint 与 finite；本轮未采样运行中显存峰值，故不启动 256 env/rank 或无 epoch 上限的正式训练。
- run_id: cmresidual_v1162_ddp_3x128_capacity_20260918_010356
- run_status: COMPLETED；last_step=12288、last_epoch=1；best_metric=`checkpoint_reload_action_max_abs_diff=0.0`。
- conclusion: 3×128 DDP/rank mapping/buffer/checkpoint 工程合同 SUPPORTED；运行中物理显存峰值、3×256 capacity 与任何科研效果 INCONCLUSIVE。

**文件**

- [V1.16 最终计划](../plan/V1.16.md)、[DDP runner](../../tools/run_cmv2_actor_distributed.py)、[runner contract tests](../../tests/test_reference_contract.py) — 用户指定的 GPU allowlist 与 rank-local 映射。
- [README](../README.md)、[current versions](../../../../../docs/current_versions.yaml) — V1.16.2 状态入口。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/checkpoint_validation.json)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/train/CmResidualGrabReferenceTransitionCmv2ActorV116_ddp/nn/last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_1_rew_-inf.pth)、[rank 0 buffer](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/cm_buffer/rank_000/manifest.json)、[rank 1 buffer](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/cm_buffer/rank_001/manifest.json)、[rank 2 buffer](../../../../../outputs/CmResidual/cmresidual_v1162_ddp_3x128_capacity_20260918_010356/cm_buffer/rank_002/manifest.json) — 首轮 3×128 DDP 工程证据。

**原因**

GPU2 正在被其他工作占用；GPU0/1/3 的预检显存均低于 4096 MiB。256 env/rank 没有实测证据，按最终计划先以 128 env/rank 收集容量数据，避免 OOM 或影响其他任务。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_reference_contract.py -q -k 'v116'`：`4 passed, 26 deselected`；`py_compile` 与 runner `--help` 通过。
- capacity 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_cmv2_actor_distributed.py --gpus 0,1,3 --envs-per-rank 128 --capacity-probe --activity-id ACT-20260918-010356-CMRESIDUAL-V1162-DDP --modification-version V1.16.2 --run-id cmresidual_v1162_ddp_3x128_capacity_20260918_010356`。仅作为工程容量测试；三 rank buffer、finite checkpoint/action/optimizer 和显存证据通过才允许讨论 3×256。
- 命令于 2026-09-18 01:05:44 +0800 正常完成：每 rank 的 transition-only buffer 各有 4096 sample；checkpoint、optimizer 与 action finite，reload diff=0。日志记录 global/local rank=0/0、1/1、2/2，且 local cuda:2 由 CUDA_VISIBLE_DEVICES 映射到物理 GPU3。runner 仅存 preflight memory，未采样峰值，因此补做 V1.16.3。

**保护与回滚**

回滚只恢复 0/1/2 allowlist 和本次 V1.16.2 文档/测试差异；不改 actor/Cmv2/reward/critic、数据、checkpoint、buffer schema、已有 output 或用户未提交文件。

## 2026-09-18 01:07:09 +0800 — V1.16.3 3×128 DDP 运行中显存峰值测量

- timestamp: 2026-09-18 01:07:09 +0800
- activity_id: ACT-20260918-010709-CMRESIDUAL-V1163-MEMPEAK
- modification_version: V1.16.3
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L1（Task-local capacity telemetry）；3×128 probe 为 L3。
- approval: user-approved
- approval_basis: 用户明确要求先实际测量显存，且首轮 3×128 已证明 DDP 接线但未保留运行中显存峰值。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 3567a4387b67f25f408700addf022bae020a9ea7
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户指导/plan 草稿与独立工具；只暂存本条明列路径）。
- scope: runner 在 torchrun 子进程存活期间每 0.5 秒以 nvidia-smi 合并物理 GPU0/1/3 的 memory.used 峰值，并写入 run manifest；复跑相同 3 rank × 128 env、1 epoch budget。采样失败不打断 PPO worker；不改算法/数据/模型合同，不启动 256 env/rank 或正式训练。
- run_id: cmresidual_v1163_ddp_3x128_mempeak_20260918_010709
- run_status: FAILED；PPO worker 正常退出并保存 checkpoint/buffer，但 runner 后处理在 checkpoint validation 前发生 AttributeError。
- last_step: unavailable（runner validation 未执行）；last_epoch: 1（worker log）；best_metric/checkpoint validation: unavailable。
- exit_reason: `AttributeError: 'int' object has no attribute 'returncode'`；`Popen.wait()` 返回 int，runner 错误读取 `.returncode`。
- conclusion: peak telemetry 与 worker rollout evidence SUPPORTED；本次 capacity run 整体为 INVALID_IMPLEMENTATION，不能作为 3×256 或正式训练批准依据。

**文件**

- [DDP runner](../../tools/run_cmv2_actor_distributed.py)、[runner contract tests](../../tests/test_reference_contract.py)、[README](../README.md)、[current versions](../../../../../docs/current_versions.yaml) — memory peak schema、定向测试与 V1.16.3 状态入口。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/train.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/train/CmResidualGrabReferenceTransitionCmv2ActorV116_ddp/nn/last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_1_rew_-inf.pth)、[rank 0 buffer](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/cm_buffer/rank_000/manifest.json)、[rank 1 buffer](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/cm_buffer/rank_001/manifest.json)、[rank 2 buffer](../../../../../outputs/CmResidual/cmresidual_v1163_ddp_3x128_mempeak_20260918_010709/cm_buffer/rank_002/manifest.json) — peak sampling 和 worker 产物。

**原因**

首轮执行已证明 3×128 可完成，但执行结束后 nvidia-smi 无法回溯峰值；必须在 worker 存活期内采样，才能用证据而非猜测判断 256 env/rank 的下一步。

**验证**

- runner 在 0.5 秒采样间隔下记录 `gpu_used_mib_peak={0:16440,1:14137,3:15358}`，monitor error 为 null。相对 preflight `2553/776/1997 MiB` 的增量约为 `13887/13361/13361 MiB`；这不构成 256 env/rank 的线性外推或安全许可。
- PPO worker log 显示三个 rank 完成 broadcast/epoch 1/checkpoint；三个 transition-only buffer manifest 均存在。由于 runner 收尾 AttributeError，该 run 不进行 checkpoint finite/reload 或 metrics validation。
- capacity 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_cmv2_actor_distributed.py --gpus 0,1,3 --envs-per-rank 128 --capacity-probe --activity-id ACT-20260918-010709-CMRESIDUAL-V1163-MEMPEAK --modification-version V1.16.3 --run-id cmresidual_v1163_ddp_3x128_mempeak_20260918_010709`。

**保护与回滚**

回滚仅删除 runner peak-monitor 与本条 V1.16.3 文档/测试差异；首轮 V1.16.2 output 保留为证据，训练、数据、buffer schema 和用户未提交路径不受影响。

## 2026-09-18 01:09:32 +0800 — V1.16.4 修复峰值 probe 收尾并复跑

- timestamp: 2026-09-18 01:09:32 +0800
- activity_id: ACT-20260918-010932-CMRESIDUAL-V1164-MEMPEAK
- modification_version: V1.16.4
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L1（Task-local runner 收尾修复）；3×128 probe 为 L3。
- approval: user-approved
- approval_basis: 用户要求实测显存；V1.16.3 已发现 runner 后处理缺陷，修复并复跑同一受限 budget 是该授权范围内的必要验证。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 405ecb379f399376c02981421ff8f5cd188134f5
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户指导/plan 草稿与独立工具；只暂存本条明列路径）。
- scope: 将 Popen return-code 判断修正为整数比较，保留 0.5 秒 physical GPU peak telemetry；复跑相同 physical GPU0/1/3、3 rank×128 env、1 epoch capacity budget，以完成 rank-buffer、metrics、checkpoint finite/reload 与 peak-memory 的整体验证。不尝试 256 env/rank 或正式训练。
- run_id: cmresidual_v1164_ddp_3x128_mempeak_20260918_010932
- run_status: COMPLETED；last_step=12288、last_epoch=1；best_metric=`checkpoint_reload_action_max_abs_diff=0.0`。
- conclusion: 3×128 DDP capacity/wiring/buffer/checkpoint 工程合同 SUPPORTED；3×256 capacity、正式训练收敛与任何科研效果 INCONCLUSIVE。

**文件**

- [DDP runner](../../tools/run_cmv2_actor_distributed.py)、[runner contract tests](../../tests/test_reference_contract.py)、[README](../README.md)、[experiment log](experiment_log.md)、[current versions](../../../../../docs/current_versions.yaml) — return-code 修复、peak telemetry 与 V1.16.4 状态。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/train.log)、[checkpoint validation](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/checkpoint_validation.json)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/train/CmResidualGrabReferenceTransitionCmv2ActorV116_ddp/nn/last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_1_rew_-inf.pth)、[rank 0 buffer](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/cm_buffer/rank_000/manifest.json)、[rank 1 buffer](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/cm_buffer/rank_001/manifest.json)、[rank 2 buffer](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/cm_buffer/rank_002/manifest.json) — 完整的 3×128 capacity 证据。

**原因**

V1.16.3 的训练 worker 返回 0，但 Popen.wait() 的 int 被误当作带 `.returncode` 的对象；必须修正后才能可信地验证 checkpoint 与 metrics。已观测峰值接近 14–16 GiB，直接倍增 env 前必须先拥有有效的 3×128 terminal evidence。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/CmResidual/tests/test_reference_contract.py -q -k 'v116'`：`4 passed, 26 deselected`；`py_compile`、diff 与 activity link audit 通过。
- capacity 命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_cmv2_actor_distributed.py --gpus 0,1,3 --envs-per-rank 128 --capacity-probe --activity-id ACT-20260918-010932-CMRESIDUAL-V1164-MEMPEAK --modification-version V1.16.4 --run-id cmresidual_v1164_ddp_3x128_mempeak_20260918_010932`。
- 完成时间 2026-09-18 01:11:09 +0800；peak physical memory 为 GPU0/1/3=`16440/14137/15358 MiB`，采样错误为 null。`metrics.jsonl` 有 epoch 1，三个 v2 buffer manifest 各有 4096 sample；checkpoint/optimizer/action finite，reload diff=0，sigma=`0.1`。总 throughput 约 `1069 FPS`；epoch 1 前无 episode 终结导致 reward `-inf`，不用于策略质量判断。

**保护与回滚**

回滚仅恢复 runner 的 Popen return-code 判断并移除 V1.16.4 文档/测试差异；V1.16.2/V1.16.3 outputs、算法、数据、buffer schema 和用户未提交路径保留。

## 2026-09-18 01:14:37 +0800 — V1.16.5 GPU0/1/3 3×128 正式 PPO 长训启动

- timestamp: 2026-09-18 01:14:37 +0800
- activity_id: ACT-20260918-011437-CMRESIDUAL-V1165-FORMAL
- modification_version: V1.16.5
- operation_category: experiment、operation、documentation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户在 V1.16.4 3×128 capacity probe 后明确指示“用这个配置开启长训”；此前已明确不加入 epoch 停止条件。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 23ebec1c62e39442ec42d6e92288c0452d13ef03
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户指导/plan 草稿与独立工具；不覆盖或纳入它们）。
- scope: physical GPU0/1/3，经 CUDA_VISIBLE_DEVICES 映射为 3 rank local cuda:0/1/2；每 rank 128 env、总 384 env，seed42、pinned GRAB reference/source/Cmv2、V1.16 GPU nominal actor、transition-only v2 CmBuffer、base68-only RMS、reference-transition reward 和所有 action/critic 合同不变。无 `--capacity-probe`、无显式 `--max-epochs` override；保留配置 `max_epochs=100000000`，由用户人工停止。
- run_id: cmresidual_v1165_ddp_3x128_formal_20260918_011437
- run_status: STARTED；launcher 将创建 manifest、config、metrics、train log、checkpoint 与三个 rank buffer manifest 后登记。
- conclusion: INCONCLUSIVE（正式训练刚启动；容量 probe 只支持工程可运行，不构成科研效果结论）。

**文件**

- [V1.16 最终计划](../plan/V1.16.md)、[DDP runner](../../tools/run_cmv2_actor_distributed.py)、[V1.16 task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceTransitionCmv2ActorV116.yaml)、[V1.16 PPO config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabReferenceTransitionCmv2ActorV116PPO.yaml) — 已批准的正式运行合同与入口。
- [V1.16.4 capacity probe](../../../../../outputs/CmResidual/cmresidual_v1164_ddp_3x128_mempeak_20260918_010932/run_manifest.json)、[README](../README.md)、[experiment log](experiment_log.md)、[current versions](../../../../../docs/current_versions.yaml) — 3×128 容量证据、状态与结论入口。

**原因**

V1.16.4 已在相同物理卡和 env 数下完成一 epoch、三 rank buffer、checkpoint finite/reload 和峰值显存验证；用户已在该容量前提下批准进入无固定 epoch budget 的正式训练。

**验证**

- 启动前 `nvidia-smi`：physical GPU0/1/3 memory used=`2553/776/1997 MiB`，均低于 runner 的 `4096 MiB` gate。
- 正式命令：`/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/CmResidual/tools/run_cmv2_actor_distributed.py --gpus 0,1,3 --envs-per-rank 128 --activity-id ACT-20260918-011437-CMRESIDUAL-V1165-FORMAL --modification-version V1.16.5 --run-id cmresidual_v1165_ddp_3x128_formal_20260918_011437`。
- 监控边界：每 0.5 秒记录 physical peak memory；活动记录在用户停止或训练终态时补充 last_step/epoch、metrics/train log、checkpoint、rank buffer、退出原因与科研结论。容量/smoke 不能代替效果结论。

**保护与回滚**

训练输出仅写入新的 `outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/`，不覆盖基线或已有 probe。用户要求停止时向 torchrun 发送中断，并由 Task close/atexit flush 未满 CmBuffer shard；数据、checkpoint、代码、research variable 和用户未提交路径不改写。

## 2026-09-18 10:07:36 +0800 — V1.16.5 正式训练进度与学习效果只读诊断

- timestamp: 2026-09-18 10:07:36 +0800
- activity_id: ACT-20260918-100736-CMRESIDUAL-V1165-DIAGNOSTIC
- modification_version: V1.16.5
- operation_category: diagnostic、operation、documentation
- task_mode: read-only/diagnostic（仅追加本活动证据）
- change_level: L0
- approval: auto（用户明确要求查看训练状态及是否有效学习）
- approval_basis: 本次用户请求；既有正式训练仍按用户批准的人工停止边界运行。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 80b6255e273b7763fef7042fb3bd8b5b648a6138
- worktree_dirty: true（保留既有用户改动；本次仅追加 CmResidual activity）
- scope: 只读检查指定会话、正式运行的进程、train log、TensorBoard scalars、CmBuffer manifest 和存储；不改变训练、checkpoint、配置、数据或研究变量。
- run_id: cmresidual_v1165_ddp_3x128_formal_20260918_011437
- run_status: RUNNING（检查时三个 worker 仍运行；已到 epoch 2995，最后完整日志 frames=36,790,272；无固定停止预算）
- conclusion: INCONCLUSIVE（尚无匹配的固定评估/对照来证明策略收益）；当前在线指标未显示持续有效学习，并有接触与追踪退化迹象。

**文件**

- [本活动记录](activity_log.md) — 追加本次诊断；其他代码、配置与输出未修改。

**原因**

用户请求续接指定会话，检查已经启动的正式训练是否在有效学习。

**证据与判断**

- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/)、[run manifest](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/config.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/train.log)、[TensorBoard scalars](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/train/CmResidualGrabReferenceTransitionCmv2ActorV116_ddp/summaries/events.out.tfevents.1789665360.server) 为本次实际证据；`metrics.jsonl` 存在但为 0 字节，不作为指标依据。
- TensorBoard `Episode/success` 按 epoch 50–299 与 2500–2990 分段均值约 `1.58%` / `1.73%`；`success_rate/iter` 是累计完成 episode 的成功率，末段约 `1.80%`，不能当作独立验证成功率。`Episode/return` 两段约 `-5697` / `-5913`，没有持续向好趋势。
- 同两段的 `reference_object_position_error_m/iter` 约 `1.213` / `1.251 m`，`reference_object_rotation_error_rad/iter` 约 `1.972` / `2.039 rad`；`contact_occupancy/iter` 约 `0.0246` / `0.0178`，而 reference contact occupancy 约 `0.408` / `0.407`。`reference_tip_position_error_m/iter` 约 `0.0147` / `0.0332 m`，也在变差。
- `residual_rms/iter` 约 `0.178` → `0.614`，但 `residual_saturation_ratio/iter=0`；这说明动作幅度增长没有转化为上述追踪和接触指标的改善，不能以数值有限、checkpoint 持续保存推断有效学习。
- [rank 0 buffer manifest](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/cm_buffer/rank_000/manifest.json)、[rank 1](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/cm_buffer/rank_001/manifest.json)、[rank 2](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/cm_buffer/rank_002/manifest.json) 在检查时各约 1226 万条。输出目录约 `37 GiB`，所在文件系统约 `889 GiB` 可用；仍需留意持续写盘。
- [epoch 2995 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/train/CmResidualGrabReferenceTransitionCmv2ActorV116_ddp/nn/last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_2995_rew_-5741.9053.pth) 是检查时最近 checkpoint；文件名中的 reward 是训练期 episode return，不等于验证集 best metric。本次不指定 best checkpoint。

**验证**

使用 `ps` 检查三个 worker；用 TensorBoard `EventAccumulator` 读取完整 scalar 历史并计算上述分段均值；核对 `train.log`、三个 buffer manifest、磁盘容量及 `metrics.jsonl` 文件大小。指标取自训练在线统计，没有固定评估集或配对零残差对照。

**保护与回滚**

未干预进程。删除本活动条目可回滚记录；运行产物和用户已有改动均保留。若要停止训练、调整奖励或训练变量，需按原人工停止约定和新计划边界执行。

## 2026-09-18 10:34:34 +0800 — V1.16.5 三卡 PPO 长训按用户指令停止

- timestamp: 2026-09-18 10:34:34 +0800
- activity_id: ACT-20260918-103434-CMRESIDUAL-V1165-STOPPED
- modification_version: V1.16.5
- operation_category: operation、experiment、documentation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确指示“先把目前的长训停止”。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 33f381a381ac1878fd61f2e7f78e99931cd652f9
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户指导/plan 草稿与独立工具；本次只追加本 Task 的终态记录）。
- scope: 向 `cmresidual_v1165_ddp_3x128_formal_20260918_011437` 的三个已确认 rank worker 发送 `SIGINT`；未删除输出、buffer、checkpoint、数据或代码。
- run_id: cmresidual_v1165_ddp_3x128_formal_20260918_011437
- run_status: STOPPED
- last_epoch: 3138
- last_step: 38,547,456 env-steps（最后一条 epoch 日志的 `frames`）
- best_metric: 不适用；训练期 `Episode/return` 不是固定评估 best metric。
- latest_checkpoint: [epoch 3138 checkpoint](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/train/CmResidualGrabReferenceTransitionCmv2ActorV116_ddp/nn/last_CmResidualGrabReferenceTransitionCmv2ActorV116PPO_ep_3138_rew_-5813.9316.pth)
- conclusion: INCONCLUSIVE；停止前的在线指标没有支持持续有效学习，且本运行没有匹配的固定评估。

**文件**

- [运行目录](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/)、[run manifest](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/config.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/train.log) — 已停止的正式训练及其可复现入口。
- [rank 0 buffer manifest](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/cm_buffer/rank_000/manifest.json)、[rank 1](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/cm_buffer/rank_001/manifest.json)、[rank 2](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/cm_buffer/rank_002/manifest.json) — 三个 rank 均已在退出时更新 manifest，各含 `12,853,504` 条 transition-only v2 sample。

**原因**

为释放 GPU0/1/3 的资源，并停止此前在线指标未显示持续收益的 CmResidual 长训。

**验证**

- 三个指定 worker（PID `3872228`、`3872232`、`3872233`）已退出；检查时 GPU0/1/3 分别回落至 `2553/776/1997 MiB` 的既有占用，GPU 使用率为 `0`。
- [train log](../../../../../outputs/CmResidual/cmresidual_v1165_ddp_3x128_formal_20260918_011437/train.log) 最后一条完整 epoch 为 `3138/100000000`、`frames=38,547,456`；其后三个 rank 都记录 `KeyboardInterrupt`，是本次人工 `SIGINT` 的预期结果。
- 最新 checkpoint 存在且非空；三个 buffer manifest 的 mtime 晚于中断，表明 close/atexit flush 已完成。`metrics.jsonl` 仍为 0 字节；训练曲线以 TensorBoard event 和 train log 为准。

**保护与回滚**

停止是可逆的运行状态操作：可从上述 epoch 3138 checkpoint 单独协商恢复，但不自动恢复。现有输出约 38 GiB，未删除或覆盖。

## 2026-09-18 10:56:32 +0800 — V1.17 外部 DExplore GRAB teacher 单卡容量 smoke 完成

- timestamp: 2026-09-18 10:56:32 +0800
- activity_id: ACT-20260918-110300-CMRESIDUAL-V117-DEXPLORE
- modification_version: V1.17
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认用已存在的 RL rollout 灵巧手轨迹训练 DExplore；确认输入应为含实际物体状态的 `inspire_rl_object_dexplore`，并要求开始训练。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 33f381a381ac1878fd61f2e7f78e99931cd652f9
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户已有 V1.12–V1.18 指导/plan 草稿和独立工具；本次只新增 V1.17 启动器、测试、文档与本活动）。
- scope: 不改写外部 `/home2/wyy/oyx_ws/dexplore`（commit `c31f57f186409ce5f0de47ced2d347abffe45d06`、clean）；以其原始 DExplore teacher 在物理 GPU5 运行新的单卡容量 smoke。
- run_id: dexplore_grab_teacher_v117_smoke_20260918_110300
- run_status: COMPLETED
- last_epoch: 2
- last_step: 64 rollout steps × 4 env（由固定 smoke 合同）；训练日志报告 epoch 1/2 的 mean reward 分别为 `2.95` / `2.96`。
- best_metric: 不适用；这是容量 smoke，没有固定评估协议。
- latest_checkpoint: [GRAB checkpoint](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110300/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth)
- conclusion: SUPPORTED（工程）：外部训练代码、660 条 RL rollout 参考、GPU PhysX、1442-D observation、18-D action、64 步 rollout、PPO 更新、checkpoint 与 TensorBoard event 已形成完整链路；不据此推断 teacher 收敛或抓取成功。

**原因**

用户要求停止无有效学习证据的 CmResidual 长训，并直接训练外部 DExplore 的 GRAB teacher；输入根经核对后改为保留 policy 实际手/物体状态的 `inspire_rl_object_dexplore`。

**修改范围**

- [版本指针](../../../../../docs/current_versions.yaml)、[Task README](../README.md) — 切换 CmResidual 当前版本至 V1.17 并导航本次运行。
- [V1.17 指导](../指导/V1.17.md)、[V1.17 最终计划](../plan/V1.17.md)、[启动器](../../tools/run_dexplore_grab_teacher.py)、[定向测试](../../tests/test_dexplore_teacher_launcher.py) — 本次新增或更新的受控范围。

**文件与运行合同**

- [V1.17 指导](../指导/V1.17.md)、[V1.17 最终计划](../plan/V1.17.md)、[启动器](../../tools/run_dexplore_grab_teacher.py)、[定向测试](../../tests/test_dexplore_teacher_launcher.py) — 运行范围、输入/输出保护和 CLI 合同。
- [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110300/)、[run manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110300/run_manifest.json)、[config](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110300/config.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110300/train.log)、[TensorBoard event](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110300/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/summaries/events.out.tfevents.1789700178.server) — 实际运行证据。
- 输入是 [RL rollout manifest](../../../../../data/processed_data/inspire_rl_object_dexplore/manifest.json)：660 条 `(T,598)` tensor，已有 `inspire.pth` rollout 的 18-DOF 与实际物体 pose；本次训练 `initial_checkpoint=null`，没有续训。

**实现与排障**

- 启动器在每个新 output 下创建仅包含 660 条 sequence 的 `motion_input/` 符号链接视图。源根中的 `_logs/` 不再被外部 loader 当作样本；不复制、修改或删除 GRAB tensor。
- 外部代码要求 `rl-games==1.1.4`，共享 `graspenv` 实际为 `1.6.5` 且缺少 `Runner.model_builder`。已在 `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117` 创建隔离环境并安装 `1.1.4`，未改动共享环境。
- 外部实现部分张量固定使用逻辑 `cuda:0`。启动器设 `CUDA_VISIBLE_DEVICES=5`，因此逻辑 `cuda:0` 对应 physical GPU5；manifest 记录了此映射。
- 先前两次失败运行均保留： [rl-games API 失败](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_105700/)、[源根 `_logs/` 扫描失败](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_105900/)；第三次在 source 设备硬编码处失败的输出也保留为 [设备映射失败证据](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_smoke_20260918_110100/)。三者均为 `INVALID_IMPLEMENTATION`，未产生可用 checkpoint。

**验证**

- `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m py_compile src/task/CmResidual/tools/run_dexplore_grab_teacher.py`
- `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py`：`1 passed`。
- 启动前预检验证 source clean、资产、660 个 finite float32 `(T,598)` tensor、GPU5 仅 `6 MiB` 占用和 `rl-games==1.1.4`；成功日志确认 `Device count 1`、`Physics Device: cuda:0`、`num_motions: 4`、`num_obs: 1442`、`num_actions: 18`、checkpoint/event 存在且非空。

**保护与回滚**

删除本版本 launcher、测试、V1.17 文档/README/版本指针和本活动条目可回滚代码记录；全部 smoke output、隔离环境、外部 checkout、660 条输入数据和 CmResidual V1.16.5 停止产物均保留。正式长训需要独立最终计划，明确 GPU 并行规模、总 iteration/环境步预算、周期性 checkpoint 与固定评估协议。

## 2026-09-18 11:19:06 +0800 — V1.17 单轨迹 DExplore GPU5 稳定 env 容量测量完成

- timestamp: 2026-09-18 11:19:06 +0800
- activity_id: ACT-20260918-113000-CMRESIDUAL-V117-DEXPLORE-CAPACITY
- modification_version: V1.17
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后 run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确指定只训练 CmResidual 已使用的轨迹，并要求立即测量稳定 env 数。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: d643401a9d2576675cd00eac1debd15ffdc9ff75
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户已有 V1.12–V1.18 指导/plan 草稿和独立工具；本次仅更新 V1.17 launcher、测试、README/指导/计划与本活动）。
- scope: 外部 DExplore 源仍为 commit `c31f57f186409ce5f0de47ced2d347abffe45d06`、clean；physical GPU5 上仅运行 `s1_airplane_lift` 的单卡容量 probe，不改外部源码、CmResidual 合同或输入 tensor。
- run_id: `dexplore_grab_teacher_v117_single_airplane_env64_20260918_111700`、`dexplore_grab_teacher_v117_single_airplane_env128_20260918_111900`、`dexplore_grab_teacher_v117_single_airplane_env256_20260918_112100`、`dexplore_grab_teacher_v117_single_airplane_env512_20260918_112300`、`dexplore_grab_teacher_v117_single_airplane_env1024_20260918_112500`、`dexplore_grab_teacher_v117_single_airplane_env2048_20260918_112700`、`dexplore_grab_teacher_v117_single_airplane_env4096_20260918_113000`
- run_status: COMPLETED（64–2048 env）；FAILED（4096 env，CUDA OOM）；稳定容量为 2048 env（仅就两 epoch / 一次 PPO iteration 的工程稳定性）。
- last_epoch: 2（各成功档位）；last_step: 每个成功档位固定 64 rollout steps / env。
- best_metric: 不适用；容量 probe 没有固定评估协议。
- latest_checkpoint: [2048 env GRAB checkpoint](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env2048_20260918_112700/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth)
- conclusion: SUPPORTED（工程）：GPU5 的单一 `s1_airplane_lift` 可在 2048 env 完成 PPO rollout、更新、checkpoint 与 event；4096 env 在 rollout buffer 展平时 OOM。该结果不证明长训稳定、收敛或抓取效果。

**原因**

原 660 轨迹容量 smoke 只会在 4 env 时加载前四条轨迹。用户要求改为 CmResidual 使用的 `s1_airplane_lift`，使全部并行环境重复采样同一参考，并实测可用 env 上限。

**修改范围**

- [启动器](../../tools/run_dexplore_grab_teacher.py)、[定向测试](../../tests/test_dexplore_teacher_launcher.py)、[Task README](../README.md)、[V1.17 指导](../指导/V1.17.md)、[V1.17 最终计划](../plan/V1.17.md) — 将输入收窄到 `s1_airplane_lift`，新增显式 `--num-envs` 与 GPU5 峰值显存采样。

**结果与证据**

| env | run_status | GPU peak | PPO total fps | 证据 |
| ---: | --- | ---: | ---: | --- |
| 64 | COMPLETED | 18,181 MiB | 665.7 | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env64_20260918_111700/run_manifest.json) |
| 128 | COMPLETED | 18,297 MiB | 1,250.7 | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env128_20260918_111900/run_manifest.json) |
| 256 | COMPLETED | 18,617 MiB | 1,919.1 | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env256_20260918_112100/run_manifest.json) |
| 512 | COMPLETED | 19,093 MiB | 2,710.3 | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env512_20260918_112300/run_manifest.json) |
| 1024 | COMPLETED | 20,227 MiB | 3,748.0 | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env1024_20260918_112500/run_manifest.json) |
| 2048 | COMPLETED | 22,583 MiB | 4,505.7 | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env2048_20260918_112700/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env2048_20260918_112700/train.log)、[TensorBoard event](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env2048_20260918_112700/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/summaries/events.out.tfevents.1789701431.server) |
| 4096 | FAILED | 23,373 MiB | — | [manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env4096_20260918_113000/run_manifest.json)、[OOM log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v117_single_airplane_env4096_20260918_113000/train.log) |

- 2048 env 日志明确 `num_motions: 1`、`num_envs: 2048`、`num_obs: 1442`、`num_actions: 18`，第二个 PPO epoch 的 mean reward `5.19`，并生成非空 checkpoint/event。
- 4096 env 已完成环境创建且峰值达 23,373 MiB，随后 `experience_buffer` 展平请求额外 1.41 GiB 时 OOM；GPU5 退出后恢复至 6 MiB，未遗留训练进程。

**验证**

- `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m py_compile src/task/CmResidual/tools/run_dexplore_grab_teacher.py`
- `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py`：`1 passed`。
- 每一成功档位都有独立 config、manifest、train log、checkpoint 和 event；失败档保留 manifest/train log，不覆盖任何成功输出。

**保护与回滚**

2048 是本次短 probe 的最后通过档位，正式长训宜保留显存余量而非直接视作长时间稳定保证。删除本次 launcher/测试/文档/活动差异可回滚代码记录；所有 capacity output、隔离环境、外部 checkout、输入和 CmResidual 既有产物均保留。

## 2026-09-18 11:33:34 +0800 — V1.17 DExplore Horovod 双卡依赖探测失败

- timestamp: 2026-09-18 11:33:34 +0800
- activity_id: ACT-20260918-113334-CMRESIDUAL-V117-HOROVOD-PROBE
- modification_version: V1.17
- operation_category: operation、diagnostic、documentation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确询问并授权补齐双卡 DExplore 所需环境。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 3fccb5141d2edf2e6cba2e93a050b3a5d1c359d5
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户已有 V1.12–V1.18 指导/plan 草稿和独立工具；本次仅追加 CmResidual activity）。
- scope: 仅尝试在 `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117` 安装 GPU/NCCL Horovod；不改外部 DExplore 源、共享 `graspenv`、训练输入或运行输出。
- run_id: dexplore_horovod_install_probe_20260918_113334
- run_status: FAILED
- conclusion: INVALID_IMPLEMENTATION（当前运行时组合）：`horovod==0.28.1` 已识别 MPI、CUDA、NCCL 与 PyTorch，但其 PyTorch C++ 扩展不能编译到当前 `torch==2.4.1+cu121`；没有安装 Horovod，也没有启动双卡训练。

**原因**

外部 DExplore 原始多卡路径在 `multi_gpu=True` 时直接导入 `horovod.torch`；用户希望以双卡承载单卡无法容纳的并行环境。

**证据与保护**

- [V1.17 最终计划](../plan/V1.17.md) — 当前单轨迹与容量合同；本次只检查其双卡所需运行依赖。
- 当前隔离环境保留 `rl-games==1.1.4`；为构建探测新增局部 `nvidia-cuda-nvcc-cu12==12.1.105` 与局部 NCCL 兼容链接。共享环境没有改动，`horovod` 仍不存在。
- 初次构建缺少 pip NCCL 的无版本 `libnccl.so` 链接；局部兼容链接后 CMake 已找到 MPI、CUDA 12.4、NCCL 与 PyTorch，但在 Horovod 0.28.1 的 `cuda_util.cc` 对 Torch 2.4 API 编译时失败。
- 中断后确认没有遗留 Horovod pip、CMake 或编译进程。可删除隔离环境内的 `nvidia-cuda-nvcc-cu12` 和 `third_party/nccl/` 兼容目录回滚本次依赖探测；未删除，因为后续可在与 DExplore README 声明的 Torch 2.2.2 匹配的独立环境中复用。

**验证**

- `pip show horovod` 确认该包未安装；`pip show nvidia-cuda-nvcc-cu12` 确认仅该隔离环境含版本 `12.1.105`。
- `ps` 检查确认不存在本次 Horovod pip、CMake 或编译子进程。

## 2026-09-18 12:06:42 +0800 — V1.18 reference PPO 与 frozen-Cmv2 planner 实现

- timestamp: 2026-09-18 12:06:42 +0800
- activity_id: ACT-20260918-120642-CMRESIDUAL-V118-IMPLEMENT
- modification_version: V1.18
- operation_category: architecture、code、experiment、documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户在 V1.18 草案形成后明确指示“先开始做 V1.18”。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 1d3cbc263b0b8c527c398e1e96d1d1c0b8fa0a93
- worktree_dirty: true（保留根 activity、ObjectInteractionCm activity、用户已有 V1.12–V1.18 指导与独立工具；本次只纳入下列 V1.18 路径。）
- scope: 新建 opt-in V1.18 task/PPO/config/launcher；既有 V1.16 actor、legacy source `frame()`、residual authority、transition-only v2 schema 和历史 outputs 不改写。
- run_id: 无（实现阶段尚未启动 smoke）
- run_status: NOT_STARTED
- conclusion: INCONCLUSIVE

**文件**

- [V1.18 最终计划](../plan/V1.18.md)、[版本指针](../../../../../docs/current_versions.yaml)、[Task README](../README.md) — 固定 721-D 字段、reward/termination、K=8 planner 与阶段边界。
- [GPU FK/planner](../../v118_planner.py)、[retargeted provider](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/reference_provider.py)、[V1.18 task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 从 retargeted fields 重建 428-D transport，构造 `+1/+16` 1442-D observation，并以一次 active `[B*K,...]` Cmv2 forward 产出 detached teacher。
- [V1.18 PPO agent](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/v118_agent.py)、[runner 注册](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/train.py)、[task registry](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/__init__.py) — 在 rollout buffer 固定存储 teacher action/weight，并只对 actor mean 加 distillation loss。
- [task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceV118.yaml)、[PPO config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualGrabReferenceV118PPO.yaml)、[启动器](../../tools/run_v118_reference_ppo.py)、[定向测试](../../tests/test_v118_planner.py) — 128 env/rank、64-step smoke、配置预检与可审计运行入口。

**原因**

落实用户指定的 reference-conditioned PPO 主架构：future reference 提供任务意图，Cmv2 只作为冻结的局部动作后果 teacher，避免 V1.16 将 Cmv2 token/effect 直接拼入 actor observation 的路径。

**验证**

- `PYTHONPATH=third_party/IsaacGymEnvs:. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_v118_planner.py`：`2 passed`；Torch FK 相对 pinned CPU URDF FK 最大元素误差 `2.38e-7`。
- `PYTHONPATH=third_party/IsaacGymEnvs:. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_reference_contract.py src/task/CmResidual/tests/test_cmv2_action_evaluator.py`：`35 passed`。
- `train.py --cfg job task=CmResidualGrabReferenceV118 train=CmResidualGrabReferenceV118PPO` 解析为 1442-D observation、18-D action、`cm_planner_continuous`、K=8、`cm_distill_coef=0.1`；`git diff --check` 通过。

**保护与回滚**

删除上述 V1.18 增量并将版本指针恢复 V1.17 即可回滚；不删除 reference、Cmv2 checkpoint、legacy source、V1.16 code/buffer 或任一历史 output。下一个 gate 是独立的 128 env、64 horizon 实现 smoke；成功只说明工程链路，不作学习效果结论。

## 2026-09-18 12:08:30 +0800 — V1.18 128-env 实现 smoke 初始化失败

- timestamp: 2026-09-18 12:08:30 +0800
- activity_id: ACT-20260918-120830-CMRESIDUAL-V118-SMOKE
- modification_version: V1.18
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户已批准 V1.18 最终计划并明确指示开始实施；该 smoke 是计划中 128 env、64 horizon 的首个工程 gate。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: a21cadfb972bee99b25e20cd5ff77536b4ecfc12
- worktree_dirty: true（仅本条 STARTED 运行记录尚未提交；根/ObjectInteractionCm activity 和用户已有未跟踪文档/工具继续保护。）
- scope: physical GPU5，单 rank×128 env，window/horizon=64，K=8，`lambda_cm=0.1`；未启动正式 Stage A/B/C 长训。
- run_id: cmresidual_v118_smoke_20260918_120830
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v118_smoke_20260918_120830/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_20260918_120830/run_manifest.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v118_smoke_20260918_120830/train.log)。
- last_step: 0
- exit_reason: V1.18 曾错误进入 legacy OI-Cm fallback，Hydra 工作目录被解释为 checkpoint 路径；未构造 task 或执行 PPO rollout。
- conclusion: INVALID_IMPLEMENTATION

**修复**

将 V1.18 加入 lazy frozen-Cmv2 分支，禁止它加载无关 OI-Cm；该增量会先经定向测试和提交，再用新的 output 重试同一 128-env smoke。

## 2026-09-18 12:09:30 +0800 — V1.18 修正 frozen-Cmv2 lazy 初始化

- timestamp: 2026-09-18 12:09:30 +0800
- activity_id: ACT-20260918-120930-CMRESIDUAL-V118-LAZY-CMV2
- modification_version: V1.18
- operation_category: code、documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户已批准 V1.18 实施；这是首个 smoke 的确定性初始化缺陷修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: a21cadf51f86e9c6f8dabb6d06068e4a54a86cca
- worktree_dirty: true（仅本次修复与运行终态记录未提交；其他用户路径继续保护。）
- scope: 只修正 V1.18 的 adapter 选择；legacy OI-Cm、V1.16 actor path 和 checkpoint 不改写。
- run_id: cmresidual_v118_smoke_20260918_120830
- run_status: FAILED（上述历史 smoke）；修复后将另建 run_id 重试。
- conclusion: INCONCLUSIVE

**文件**

- [V1.18 task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[失败 manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_20260918_120830/run_manifest.json) — 将 `useV118Planner` 归入 lazy frozen-Cmv2 分支，避免无关 OI-Cm checkpoint load。

**原因**

V1.18 已要求 Cmv2 是 frozen planner，但初始化条件仅覆盖 V1.16 actor/evaluator，导致 V1.18 误触旧 OI-Cm fallback。

**验证**

`py_compile` 通过；`pytest -q src/task/CmResidual/tests/test_v118_planner.py src/task/CmResidual/tests/test_reference_contract.py` 为 `32 passed`。GPU5 空闲（6 MiB），新的 128-env smoke 仍待启动。

**保护与回滚**

删除该条件中的 `or self.use_v118_planner` 即可回到提交 `a21cadf`；失败运行目录保留，不覆盖。

## 2026-09-18 12:10:30 +0800 — V1.18 128-env 实现 smoke 重试在 PPO agent 初始化失败

- timestamp: 2026-09-18 12:10:30 +0800
- activity_id: ACT-20260918-121030-CMRESIDUAL-V118-SMOKE-RETRY
- modification_version: V1.18
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: V1.18 最终计划的首个工程 smoke；重试只包含已通过定向验证的 lazy-initialization 修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: dc1bf2db068c5256a7fb34cd41fa022837f137dd
- worktree_dirty: true（仅本条 STARTED 记录未提交；用户现有改动继续保护。）
- scope: physical GPU5，单 rank×128 env，window/horizon=64，K=8，`lambda_cm=0.1`；未启动正式 Stage A/B/C 长训。
- run_id: cmresidual_v118_smoke_retry_20260918_121030
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry_20260918_121030/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry_20260918_121030/run_manifest.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry_20260918_121030/train.log)。
- last_step: 0
- exit_reason: vendored `CommonAgent` 仍读取已废弃的 `seq_len`，而当前 rl-games 提供的是 `seq_length`；task 已构造但未 rollout。
- conclusion: INVALID_IMPLEMENTATION

## 2026-09-18 12:11:30 +0800 — V1.18 对齐 rl-games sequence length 名称

- timestamp: 2026-09-18 12:11:30 +0800
- activity_id: ACT-20260918-121130-CMRESIDUAL-V118-RLGAMES-COMPAT
- modification_version: V1.18
- operation_category: code、documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户已批准 V1.18 实施；这是 retry 运行揭示的确定性 rl-games API 兼容修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: dc1bf2db068c5256a7fb34cd41fa022837f137dd
- worktree_dirty: true（只含本次修复和最新运行终态记录；用户改动未纳入。）
- scope: 仅在 V1.18 专用 agent 中将当前 rl-games `seq_length` 映射到 vendored dataset 所需的 `seq_len`；共享 CommonAgent 不改。
- run_id: cmresidual_v118_smoke_retry_20260918_121030
- run_status: FAILED（上述 retry）；修复后将另建 output 重试。
- conclusion: INCONCLUSIVE

**文件**

- [V1.18 PPO agent](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/v118_agent.py)、[失败 manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry_20260918_121030/run_manifest.json) — 在专用 agent 的配置加载阶段建立版本兼容 alias。

**原因**

当前 rl-games 将公共字段命名为 `seq_length`，而该仓库旧 CommonAgent/AMPDataset 仍读取 `seq_len`；这阻止 agent 完成初始化。

**验证**

`py_compile` 通过；`pytest -q src/task/CmResidual/tests/test_v118_planner.py` 为 `2 passed`。GPU5 仍为 6 MiB，新的 128-env smoke 待启动。

**保护与回滚**

删除该 alias 即可回到提交 `dc1bf2d`；两个失败 smoke 的输出均保留。

## 2026-09-18 12:12:30 +0800 — V1.18 128-env 实现 smoke 第二次重试在 planner mask 形状失败

- timestamp: 2026-09-18 12:12:30 +0800
- activity_id: ACT-20260918-121230-CMRESIDUAL-V118-SMOKE-RETRY2
- modification_version: V1.18
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: V1.18 最终计划的首个工程 smoke；仅包含已验证的 V1.18 专用 agent compatibility 修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 2cab285aa43f7ef7b93b481814e4a39ff5244c6a
- worktree_dirty: true（仅本条 STARTED 记录未提交；用户现有改动继续保护。）
- scope: physical GPU5，单 rank×128 env，window/horizon=64，K=8，`lambda_cm=0.1`；未启动正式 Stage A/B/C 长训。
- run_id: cmresidual_v118_smoke_retry2_20260918_121230
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry2_20260918_121230/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry2_20260918_121230/run_manifest.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry2_20260918_121230/train.log)。
- last_step: 0
- exit_reason: 已进入 V1.18 planner，但扁平 `[B*K,18]` `applied_delta` 未恢复为 `[B,K,18]`，feasibility mask 相与时报维度不匹配。
- conclusion: INVALID_IMPLEMENTATION

## 2026-09-18 12:13:30 +0800 — V1.18 修正 planner candidate shape 恢复

- timestamp: 2026-09-18 12:13:30 +0800
- activity_id: ACT-20260918-121330-CMRESIDUAL-V118-PLANNER-SHAPE
- modification_version: V1.18
- operation_category: code、documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户已批准 V1.18 实施；这是实际 planner rollout 揭示的局部 shape 缺陷修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 2cab285aa43f7ef7b93b481814e4a39ff5244c6a
- worktree_dirty: true（只含本次修复与运行终态记录；用户改动未纳入。）
- scope: 仅将已有 `compose_reference_residual` 输出的 `applied_delta` 按 candidate 恢复形状；candidate 语义、authority、K=8 和 Cmv2 均不变。
- run_id: cmresidual_v118_smoke_retry2_20260918_121230
- run_status: FAILED（上述 retry）；修复后将另建 output 重试。
- conclusion: INCONCLUSIVE

**文件**

- [GPU planner](../../v118_planner.py)、[失败 manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry2_20260918_121230/run_manifest.json) — 统一 `targets` 与 `applied_delta` 的 `[B,K,18]` candidate contract。

**原因**

flat Cmv2 batch 前，retargeted residual mapping 仍返回 `[B*K,18]` 辅助张量；planner 只恢复了 target，遗漏了 feasibility 所需 applied delta。

**验证**

`py_compile` 通过；`pytest -q src/task/CmResidual/tests/test_v118_planner.py` 为 `2 passed`。GPU5 仍为 6 MiB，新的 128-env smoke 待启动。

**保护与回滚**

删除该 `view(count, K, 18)` 即可回到提交 `2cab285`；三个失败 smoke 输出均保留。

## 2026-09-18 12:14:30 +0800 — V1.18 128-env 实现 smoke 第三次重试在 Cmv2 active batch OOM

- timestamp: 2026-09-18 12:14:30 +0800
- activity_id: ACT-20260918-121430-CMRESIDUAL-V118-SMOKE-RETRY3
- modification_version: V1.18
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: V1.18 最终计划的首个工程 smoke；仅包含已验证的 planner shape 修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: a8278243dbfb148835489a8be3fbf7ebaf26bc8b
- worktree_dirty: true（仅本条 STARTED 记录未提交；用户现有改动继续保护。）
- scope: physical GPU5，单 rank×128 env，window/horizon=64，K=8，`lambda_cm=0.1`；未启动正式 Stage A/B/C 长训。
- run_id: cmresidual_v118_smoke_retry3_20260918_121430
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry3_20260918_121430/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry3_20260918_121430/run_manifest.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry3_20260918_121430/train.log)。
- last_step: 0
- exit_reason: task/agent/planner 已构造，首个 all-active `[128*8]` frozen-Cmv2 interaction graph batch 额外申请 `2.84 GiB`，GPU5 可用 `2.22 GiB`，CUDA OOM。
- conclusion: INVALID_IMPLEMENTATION

## 2026-09-18 12:15:30 +0800 — V1.18 限定并轮转 Cmv2 active planner batch

- timestamp: 2026-09-18 12:15:30 +0800
- activity_id: ACT-20260918-121530-CMRESIDUAL-V118-ACTIVE-CAP
- modification_version: V1.18
- operation_category: code、experiment、documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户批准 V1.18 实施；该资源边界由已保留的 128-env smoke OOM 直接测得。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: a8278243dbfb148835489a8be3fbf7ebaf26bc8b
- worktree_dirty: true（只含本次修复和运行终态记录；用户改动未纳入。）
- scope: 维持每 rank 128 env、K=8 和 single Cmv2 forward；仅将同时进入 frozen Cmv2 graph 的 active env 配置化限为 16，并以确定性游标轮转。
- run_id: cmresidual_v118_smoke_retry3_20260918_121430
- run_status: FAILED（上述 retry）；修复后将另建 output 重试。
- conclusion: INCONCLUSIVE

**文件**

- [GPU planner](../../v118_planner.py)、[V1.18 task](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py)、[task config](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceV118.yaml)、[定向测试](../../tests/test_v118_planner.py)、[V1.18 最终计划](../plan/V1.18.md)、[失败 manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry3_20260918_121430/run_manifest.json) — 新增 `maxActiveEnvs=16` 的旋转 active gate。

**原因**

Cmv2 local interaction graph 对 candidate batch 的显存需求使 all-active `128×8` 无法在 24 GiB GPU 运行；该上限保持 PPO 的 128 env rollout 和每个被选 env 的 K=8 one-batch planner 语义。

**验证**

`py_compile` 通过；`pytest -q src/task/CmResidual/tests/test_v118_planner.py` 为 `2 passed`；Hydra 解析确认 `numObservations=1442`、`maxActiveEnvs=16` 和 `cm_planner_continuous`。新的 128-env smoke 待启动。

**保护与回滚**

设置 `maxActiveEnvs=128` 可恢复 all-active 行为，但会复现已记录的 OOM；历史失败 output 不删除。

## 2026-09-18 12:28:16 +0800 — V1.18 128-env 实现 smoke 第四次重试在 active planner batch OOM

- timestamp: 2026-09-18 12:28:16 +0800
- activity_id: ACT-20260918-122816-CMRESIDUAL-V118-SMOKE-RETRY4
- modification_version: V1.18
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: V1.18 最终计划定义的首个 128-env、64-horizon rollout/update 工程 gate；本次只包含已提交的 active-batch cap 修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 55cc3802c9db3c970a7bf8dc514f8e55eaa0ccd1
- worktree_dirty: true（根级/ObjectInteractionCm activity 及用户已有未跟踪文档/工具保持不变；本次只会更新本条运行终态。）
- scope: physical GPU5，单 rank×128 env，window/horizon=64，K=8，`lambda_cm=0.1`，single Cmv2 active `[16*8]` batch；未启动正式 Stage A/B/C 长训。
- run_id: cmresidual_v118_smoke_retry4_20260918_122816
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry4_20260918_122816/)、[config](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry4_20260918_122816/config.json)、[manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry4_20260918_122816/run_manifest.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry4_20260918_122816/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry4_20260918_122816/train.log)、[buffer manifest](../../../../../outputs/CmResidual/cmresidual_v118_smoke_retry4_20260918_122816/cm_buffer/rank_000/manifest.json)。
- last_step: 0
- exit_reason: task/agent/planner 已构造；首个 active `[16*8]` frozen-Cmv2 interaction graph batch 额外申请 `2.00 GiB`，GPU5 仅余 `829.81 MiB`，CUDA OOM。
- best_metric: 不适用；checkpoint: 未生成。
- conclusion: INVALID_IMPLEMENTATION

**原因**

在固定的 128-env rollout 中，Isaac Gym/PhysX 与 task/agent 已占用 GPU5 的绝大部分显存；当前计划的 16 个 active environment 所需 Cmv2 local-interaction graph 无法在剩余容量内构造。

**验证**

resolved config 确认 1442-D observation、18-D action、K=8、`maxActiveEnvs=16` 和 `cm_planner_continuous`；训练日志显示 task、agent、planner 和 rank-000 transition-only buffer 均已构造，失败发生在首个 pre-action frozen-Cmv2 forward，未执行 PPO rollout/update。

**后续闸门**

当前最终计划冻结 `maxActiveEnvs=16`。继续将其降为更小的 active batch 会改变每个 rollout 的 teacher 覆盖率，必须经用户确认并更新最终计划后才能进行；本次不启动任何正式 Stage A/B/C 长训。

## 2026-09-18 13:11:19 +0800 — V1.17.1 固定 DexPlore 2048-env 正式训练合同

- timestamp: 2026-09-18 13:11:19 +0800
- activity_id: ACT-20260918-131119-CMRESIDUAL-V1171-DEXPLORE-FORMAL-CONTRACT
- modification_version: V1.17.1
- operation_category: code、experiment、operation、documentation
- task_mode: change
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确选择 2048 env，并授权 Agent 自主确定正式训练的预算、checkpoint 节奏与停止条件。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 7f3374e565ca151815286f3ef2f6f958a59e2ab2
- worktree_dirty: true（根级/ObjectInteractionCm activity 及用户已有未跟踪文档/工具保持不变；本次只纳入 V1.17 launcher、测试、计划和本条记录。）
- scope: 外部 DexPlore checkout、GRAB tensor、1442-D observation、18-D action、reward、物理、single trajectory、seed42、GPU5 与无 Horovod 不变；新 formal launcher 固定 2048 env、horizon64、minibatch256、从零初始化，运行约 20.05M env-steps。
- run_id: dexplore_grab_teacher_v1171_formal_env2048_20260918_131119
- run_status: NOT_STARTED
- conclusion: INCONCLUSIVE

**文件**

- [V1.17 最终计划](../plan/V1.17.md)、[启动器](../../tools/run_dexplore_grab_teacher.py)、[定向测试](../../tests/test_dexplore_teacher_launcher.py) — 新增 formal mode：`max_iterations=152`（实际 153 epoch）、每 38 epoch 保存 checkpoint，并将生成的训练 YAML 固定在独立运行目录。

**原因**

2048 env 已完成完整 PPO smoke，4096 env 在 rollout-buffer flatten OOM；用户选择已验证的 2048 env，并授权设置可审计的正式预算。外部运行实现以 `epoch_num > max_epochs` 才停止，故 152 的参数对应实际 153 个 PPO epoch/20,054,016 env-steps。

**验证**

- `PYTHONPATH=. /home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m py_compile src/task/CmResidual/tools/run_dexplore_grab_teacher.py` 通过。
- `PYTHONPATH=. /home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py`：`3 passed`。
- formal dry-run 通过：输入 source/asset/runtime 合同、`rl-games==1.1.4`、GPU5 预检 `6 MiB`、2048 env、64 horizon、256 minibatch、152 max iterations 均已解析。

**保护与回滚**

仅移除本条 formal mode、测试和计划追加即可回滚代码/记录；新的运行目录、外部 DexPlore checkout、隔离环境、输入 tensor、V1.17 capacity smoke 与其他用户改动均不覆盖或删除。

## 2026-09-18 13:12:15 +0800 — V1.17.1 DexPlore 2048-env 正式 teacher 训练运行中

- timestamp: 2026-09-18 13:12:15 +0800
- activity_id: ACT-20260918-131215-CMRESIDUAL-V1171-DEXPLORE-FORMAL
- modification_version: V1.17.1
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确选择 2048 env，并授权 Agent 确定正式训练预算和周期 checkpoint；合同已在 V1.17 最终计划追加并提交。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 3fde972e25bc6694433f701b249a21c2776c5ab1
- worktree_dirty: true（仅本条运行状态记录未提交；根级/ObjectInteractionCm activity 和用户已有未跟踪文档/工具保持不变。）
- scope: physical GPU5，single-rank 2048 env，`s1_airplane_lift`，horizon64，minibatch256，seed42，从零初始化；`max_iterations=152`，实际目标 153 epoch/20,054,016 env-steps；每 38 epoch checkpoint；不使用 Horovod、capacity-smoke checkpoint 或外部源码修改。
- run_id: dexplore_grab_teacher_v1171_formal_env2048_20260918_131215
- run_status: FAILED
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/)、[config](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/config.json)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/run_manifest.json)、[train config](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train_config.yaml)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train.log)。
- conclusion: INCONCLUSIVE

**原因**

按已定稿的 V1.17.1 合同启动外部 DExplore teacher；2048 env 是 GPU5 上实际通过 PPO update 的最大容量档位，正式训练仍从零初始化。

**验证**

子进程仍存活；GPU5 显存约 `22.6 GiB` 且持续有计算利用率，manifest 已写入完整 source SHA、外部 commit、运行时、命令与生成的 train config。外部 stdout 是文件块缓冲，首个 epoch 指标尚未刷入日志；当前未见 traceback、OOM 或 non-finite 证据。

## 2026-09-18 14:21:40 +0800 — V1.17.2 DexPlore 单卡正式训练按用户指令停止

- timestamp: 2026-09-18 14:21:40 +0800
- activity_id: ACT-20260918-142140-CMRESIDUAL-V1172-DEXPLORE-STOP
- modification_version: V1.17.2
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求“先把目前的 dexplore 训练停了”，并要求改在 physical GPU 0/1/3 进行同步多卡 DexPlore 训练。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: ebfb44fe15eb226c651a47767a389ee2e52a3a07
- worktree_dirty: true（根级/ObjectInteractionCm activity、用户已有未跟踪指导/工具和 V1.19 plan 草案保持不变；本条只更新既有 V1.17 运行的终态。）
- scope: 仅向 `dexplore_grab_teacher_v1171_formal_env2048_20260918_131215` 的 process group 发送 `SIGINT`；不删除 output、checkpoint、TensorBoard event、外部 checkout 或输入数据，也不停止其他 GPU 任务。
- run_id: dexplore_grab_teacher_v1171_formal_env2048_20260918_131215
- run_status: STOPPED
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train.log)、[best checkpoint](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth)、[latest checkpoint](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000114.pth)。
- last_epoch: 123；last_step: 16,121,856；best_metric: 日志中未单独导出；best_checkpoint: `GRAB.pth`；latest_checkpoint: `GRAB_00000114.pth`。
- conclusion: INCONCLUSIVE

**原因**

原定 153 epochs 的 single-rank GPU5 formal teacher 已完成至少 123 个 epoch；用户要求停止该运行并迁移到 GPU0/1/3 的同步多卡版本。该人工停止不代表 OOM、non-finite 或科研假设失败。

**验证**

`kill -INT -- -389105` 后 launcher 和其 DExplore child 均已退出；日志最后一段为 `KeyboardInterrupt`，来自用户指定的 SIGINT。checkpoint 保留 epoch 38/76/114，TensorBoard event 存在。多卡新运行尚未启动：现有 DexPlore 代码仅以 Horovod 实现同步多卡，而现有 `graspenv` 三卡环境为 `torch.distributed` 且没有 Horovod；不会将三张卡启动为相互独立的三个 PPO 训练冒充 DDP。

## 2026-09-18 15:20:11 +0800 — V1.17.2 三卡 Horovod launcher 与合同

- timestamp: 2026-09-18 15:20:11 +0800
- activity_id: ACT-20260918-152011-CMRESIDUAL-V1172-DEXPLORE-HOROVOD-LAUNCHER
- modification_version: V1.17.2
- operation_category: code、experiment、documentation
- task_mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户明确要求在 physical GPU 0/1/3 跑 DExplore，并在停止既有单卡训练后继续；[`V1.17`](../plan/V1.17.md) 的 V1.17.2 追加合同已据此定稿。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: ebfb44fe15eb226c651a47767a389ee2e52a3a07
- scope: [`run_dexplore_grab_teacher.py`](../../tools/run_dexplore_grab_teacher.py) 新增 opt-in `--launcher horovod`，固定 physical GPU 0/1/3、`horovodrun -np 3 -H localhost:3`、每 rank 2048 env、rank-local CUDA 映射、isolated CUDA/NCCL runtime 与三卡 manifest；[`test_dexplore_teacher_launcher.py`](../../tests/test_dexplore_teacher_launcher.py) 覆盖启动命令和三卡 formal/smoke 预算。单卡默认模式、数据、观测、动作、奖励、物理、seed、外部 DExplore checkout 与既有输出均未修改。
- runtime: 新建且不纳入 Git 的 `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117_hvd`，Python 3.8、Torch `2.0.1+cu118`、`rl-games==1.1.4`、Horovod `0.28.1`、MPI/CUDA/NCCL；Torch `2.2.2+cu121` 无法编译 Horovod 0.28.1 的 PyTorch C++ 扩展，故按兼容性证据替换。
- validation: `graspenv/bin/python -m py_compile src/task/CmResidual/tools/run_dexplore_grab_teacher.py`；`graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py`（5 passed）；`CUDA_VISIBLE_DEVICES=0,1,3 horovodrun -np 3 -H localhost:3 ... hvd.allreduce_(..., op=hvd.Sum)` 三 rank 分别 `rank/local_rank/device=0/0/0, 1/1/1, 2/2/2` 且均得到 `sum=6.0`。
- rollback: `git revert` 本次 launcher 提交即可恢复单卡-only launcher；隔离 runtime 与 output 不属于版本库且未影响共享环境。
- conclusion: INCONCLUSIVE（工程通信和启动合同已通过；尚未运行 DExplore 三卡 capacity smoke，不能解释为 teacher 收敛或抓取效果。）

**原因**

外部 DExplore 的同步多卡实现是 Horovod；既有 Ref2Dex `torch.distributed` 环境无法成为该入口的替代。用户指定使用 GPU 0/1/3，且 2048 env 是每 rank 的既定容量，因而新增显式、可审计的 Horovod launcher，而不改变单卡路径或外部源码。

**验证**

静态编译与 5 个 launcher 合同测试通过；在 `CUDA_VISIBLE_DEVICES=0,1,3` 下的真实 NCCL `Sum` all-reduce 由三 rank 全部返回 6.0。该证据只证明通信与启动合同，正式 DExplore smoke 仍是下一道门。

## 2026-09-18 15:22:00 +0800 — V1.17.2 三卡 DExplore capacity preflight 未启动

- timestamp: 2026-09-18 15:22:00 +0800
- activity_id: ACT-20260918-152145-CMRESIDUAL-V1172-DEXPLORE-HOROVOD-SMOKE
- modification_version: V1.17.2
- operation_category: diagnostic、operation
- task_mode: run-only/operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.17.2 最终合同的 capacity/synchronization smoke 前置门；用户要求在指定三卡继续运行 DExplore。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: a46a83024f0560f0ff6f17d3b59c76815a63a0c4
- scope: 仅执行 [`run_dexplore_grab_teacher.py`](../../tools/run_dexplore_grab_teacher.py) 的 Horovod dry-run preflight，核验 [`V1.17`](../plan/V1.17.md) 的三卡容量门；未创建训练 output，也未修改外部 DExplore、数据、配置或其他任务。
- run_id: dexplore_grab_teacher_v1172_hvd3x2048_smoke_20260918_152145
- run_status: NOT_STARTED
- command: `dexplore_v117_hvd/bin/python src/task/CmResidual/tools/run_dexplore_grab_teacher.py --launcher horovod --mode smoke --num-envs 2048 --dry-run`。
- output: PENDING（dry-run 未创建 output/manifest/train.log；容量门未通过前不得创建正式 smoke 目录）。
- evidence: Horovod 三 rank NCCL Sum all-reduce 已通过；随后 launcher preflight 读取 physical GPU 0/1/3 为 `2553/6901/1997 MiB`，GPU1 `80%` 利用率，超过保守 4096 MiB 共存门限；全机 GPU2/4/5/6/7 也分别有 `6543/6550/6131/6130/6394 MiB` 且 `70%/77%/91%/90%/81%` 利用率。
- conclusion: INCONCLUSIVE（环境和同步实现已验证；资源容量不足导致 smoke 未启动，不是 DExplore、NCCL、数据或科研假设失败。）

**原因**

每 rank 2048 env 的 DExplore 容量已知会占用大部分 24GiB 卡，且现有任务正在高利用率运行；抢占或与其混跑会使结果不可靠，并可能使无关任务 OOM。用户尚未授权停止这些其他运行，也未授权改为少于三 rank 或改变每 rank 2048 env 合同。

**验证**

preflight 在创建输出前以非零状态退出，且 `nvidia-smi` 与 launcher 报告一致；没有启动 Isaac Gym、没有写 checkpoint/训练日志、没有发送信号给任何现有任务。

## 2026-09-18 15:26:21 +0800 — V1.17.3 GPU0/3 双卡 Horovod 执行合同

- timestamp: 2026-09-18 15:26:21 +0800
- activity_id: ACT-20260918-152621-CMRESIDUAL-V1173-DEXPLORE-2GPU-LAUNCHER
- modification_version: V1.17.3
- operation_category: code、experiment、operation、documentation
- task_mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户明确要求“用0和3这两张卡跑，graspenv的其他进程停一下”。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: 86f66ed6dd6f76e4830a8420706df421dff2b663
- scope: 向 GPU0 上的 `1419735/1433137/1790780` InteractionTransfer viewer 与 GPU3 上的 `2035553/2436140/2678096` CmDecoderv2 viewer/rollout 服务发送 SIGINT；它们均为 `graspenv` 进程且均退出。[`V1.17`](../plan/V1.17.md) 追加 V1.17.3，[`run_dexplore_grab_teacher.py`](../../tools/run_dexplore_grab_teacher.py) 固定 Horovod physical GPU `(0,3)`、world size 2，[`test_dexplore_teacher_launcher.py`](../../tests/test_dexplore_teacher_launcher.py) 同步双卡预算。未停止 GPU1/2/4/5/6/7 进程，未触碰非 `graspenv` 进程、外部 DExplore、数据、观测、动作、奖励、物理或旧 output。
- runtime_contract: `CUDA_VISIBLE_DEVICES=0,3`、`horovodrun -np 2 -H localhost:2`、每 rank 2048 env、总 4096 env；smoke `262,144` steps，formal 最多 `40,108,032` steps；从零初始化。
- validation: `graspenv/bin/python -m py_compile src/task/CmResidual/tools/run_dexplore_grab_teacher.py`；`graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py`（5 passed）；真实双 rank NCCL Sum all-reduce：rank/local_rank/device `0/0/0` 与 `1/1/1` 均得到 `sum=3.0`；SIGINT 后 GPU0、GPU3 均降为 2MiB。
- rollback: `git revert` 本次 launcher 提交可恢复三卡 `(0,1,3)` 定义；被停止的 viewer/rollout 服务没有删除产物，可由其原始命令重新启动。
- conclusion: INCONCLUSIVE（双卡通信与资源释放已验证；DExplore capacity smoke 尚未启动，不能作为训练效果结论。）

**原因**

三卡 preflight 因 GPU1 的高负载未启动；用户改为两卡并明确授权仅停止 GPU0、GPU3 的 `graspenv` 进程。world size 是同步训练的运行合同，故同时更新计划、启动器、测试和预算，避免将双卡运行误记为三卡。

**验证**

两个目标 GPU 已无残留的计算进程；双卡 collective 在指定的逻辑设备上通过。下一步是使用新的 run_id 运行完整 DExplore 1-iteration smoke，检查 Isaac Gym 初始化、PPO 迭代、checkpoint 与 TensorBoard 产物。

## 2026-09-18 15:27:00 +0800 — V1.17.3 双卡 DExplore capacity smoke 已启动

- timestamp: 2026-09-18 15:27:00 +0800
- activity_id: ACT-20260918-152700-CMRESIDUAL-V1173-DEXPLORE-HOROVOD-SMOKE
- modification_version: V1.17.3
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L2
- approval: user-approved
- approval_basis: 用户授权 GPU0/3 双卡执行；[`V1.17`](../plan/V1.17.md) V1.17.3 双卡最终合同。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: fbca3ff6d1266108dc3bca49c5230099098a47f1
- scope: `CUDA_VISIBLE_DEVICES=0,3`、Horovod 2 rank、每 rank 2048 env、horizon 64、minibatch 256、seed42、max_iterations 1、从零初始化；不恢复旧 checkpoint，不改外部源码/数据/任务合同。
- run_id: dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152700
- run_status: RUNNING
- command: `dexplore_v117_hvd/bin/python src/task/CmResidual/tools/run_dexplore_grab_teacher.py --launcher horovod --mode smoke --num-envs 2048`。
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152700/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152700/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152700/train.log)。
- exit_code: 1；last_epoch: N/A；last_step: N/A；checkpoint: N/A；tensorboard: N/A。
- conclusion: INVALID_IMPLEMENTATION

**原因**

双卡 preflight、NCCL collective 和输入合同均已通过；本运行是 formal 前唯一允许的 capacity/synchronization smoke。

**验证**

启动前 dry-run 返回通过，GPU0/3 各 2MiB；两个 Horovod rank 随后在 `dexplore/run.py` import 阶段报 `ModuleNotFoundError: No module named 'isaacgym'` 并以 exit code 1 退出，GPU 峰值均为 4MiB，故不是 OOM。隔离环境随后以 `pip install --no-deps -e /home2/wyy/isaac-gym/isaacgym/python` 补齐 Isaac Gym；该命令先被中止以阻止 resolver 将 Torch 升级到 2.4.1，最终确认 Torch 仍为 2.0.1+cu118，Isaac Gym import 成功。

## 2026-09-18 15:29:00 +0800 — V1.17.3 双卡 DExplore capacity smoke 重试已启动

- timestamp: 2026-09-18 15:29:00 +0800
- activity_id: ACT-20260918-152900-CMRESIDUAL-V1173-DEXPLORE-HOROVOD-SMOKE-RETRY
- modification_version: V1.17.3
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L2
- approval: user-approved
- approval_basis: 与同版本 V1.17.3 final contract 相同；前一 run 因隔离环境缺 Isaac Gym 而无效，修复该环境依赖后以新目录重试。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: fbca3ff6d1266108dc3bca49c5230099098a47f1
- scope: 保持 GPU0/3、2 Horovod rank、每 rank 2048 env、horizon 64、minibatch 256、seed42、max_iterations 1 与从零初始化；唯一环境变化是隔离 runtime 内的本地 Isaac Gym editable 安装，不改外部源码、数据或研究合同。
- run_id: dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152900
- run_status: FAILED
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152900/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152900/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_152900/train.log)。
- exit_code: 1；last_epoch: N/A；last_step: N/A；checkpoint: N/A；tensorboard: N/A；gpu_peak_mib: GPU0/3 均 4MiB。
- conclusion: INVALID_IMPLEMENTATION

**原因**

前一 smoke 未执行训练，失效原因是可修复的隔离环境依赖缺失；新 run_id 防止覆盖该失败证据。

**验证**

Isaac Gym/gymtorch 成功加载后，两个 rank 在 DExplore `base_dexplore_task.py` import 阶段均报 `ModuleNotFoundError: No module named 'trimesh'` 并 exit 1；没有创建 checkpoint/TensorBoard，故不是 OOM。隔离环境随后仅以 `--no-deps` 安装 `trimesh==4.10.1`、`scipy==1.10.1`、`termcolor==2.4.0`，正常入口 `import run` 通过。

## 2026-09-18 15:31:00 +0800 — V1.17.3 双卡 DExplore capacity smoke 第三次启动

- timestamp: 2026-09-18 15:31:00 +0800
- activity_id: ACT-20260918-153100-CMRESIDUAL-V1173-DEXPLORE-HOROVOD-SMOKE-RETRY2
- modification_version: V1.17.3
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.17.3 final contract；此前两次无效运行均在训练前因隔离环境依赖缺失退出，重试不改变研究变量。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: fbca3ff6d1266108dc3bca49c5230099098a47f1
- scope: 固定 GPU0/3、2 rank、每 rank 2048 env、horizon64、minibatch256、seed42、max_iterations1、从零初始化；环境仅补齐 Isaac Gym、trimesh/scipy/termcolor，Torch 2.0.1+cu118、Horovod0.28.1 与外部源码不变。
- run_id: dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_153100
- run_status: FAILED
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_153100/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_153100/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1173_hvd2x2048_smoke_20260918_153100/train.log)。
- exit_code: 1；last_epoch: N/A；last_step: N/A；checkpoint: N/A；tensorboard: N/A；gpu_peak_mib: GPU0 `16974`、GPU3 `16645`。
- conclusion: INVALID_IMPLEMENTATION

**原因**

常规 DExplore `run.py` 导入链已完整通过；该 run 是在完成运行时依赖闭包后的第一轮真实容量 smoke。

**验证**

`import run` 已成功加载 DExplore、Isaac Gym、gymtorch 与 rlgpu。两个 rank 均完成 PhysX 初始化并进入 motion load；rank1 的 `obj_rot` 在 `cuda:1`，而未传 device 的 Isaac Gym `to_torch(object_points)` 固定生成于 `cuda:0`，在 `quat_rotate` 报跨设备 RuntimeError。没有 checkpoint/TensorBoard，故为外部多卡实现缺陷而非 OOM。

## 2026-09-18 15:33:00 +0800 — V1.17.4 rank-local Isaac Gym bootstrap

- timestamp: 2026-09-18 15:33:00 +0800
- activity_id: ACT-20260918-153300-CMRESIDUAL-V1174-DEXPLORE-RANK-BOOTSTRAP
- modification_version: V1.17.4
- operation_category: code、experiment、documentation
- task_mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户要求继续 GPU0/3 双卡 DExplore；[`V1.17`](../plan/V1.17.md) V1.17.4 最终追加合同将已实证的 rank1 默认设备错配限定为 Task-local bootstrap 修复。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: fbca3ff6d1266108dc3bca49c5230099098a47f1
- scope: 新增 [`dexplore_horovod_rank_bootstrap.py`](../../tools/dexplore_horovod_rank_bootstrap.py)，在导入外部 DExplore 前将 Isaac Gym `to_torch` 的未传入 device 默认值映射至 `cuda:${HOROVOD_LOCAL_RANK}`；[`run_dexplore_grab_teacher.py`](../../tools/run_dexplore_grab_teacher.py) 仅在 Horovod 命令中调用此 bootstrap；[`test_dexplore_teacher_launcher.py`](../../tests/test_dexplore_teacher_launcher.py) 验证该命令入口。显式 device、单卡命令、外部 DExplore/Isaac Gym 源码、数据和研究合同未修改。
- validation: `graspenv/bin/python -m py_compile` 两个 launcher 文件通过；`graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py`（5 passed）。第三次 smoke 的 rank1 失败堆栈精确指向 `to_torch(object_points)` 默认 `cuda:0` 与 rank1 PhysX `cuda:1` 的冲突；修复后需以新 run_id 重新运行真正 smoke。
- rollback: `git revert` 本次 bootstrap 提交会恢复 Horovod 直接执行外部 `dexplore/run.py`；此前失败 output 均保留。
- conclusion: INCONCLUSIVE（设备错配原因已支持并以最小 shim 修复；尚未证明 PPO smoke 可完成。）

**原因**

外部 DExplore 的 Horovod 分支将 PhysX/rl device 设为 rank-local，却保留 Isaac Gym helper 的全局 `cuda:0` 默认值；这在 rank0 不可见、只在 rank1 暴露。修改外部 checkout 会污染参考实现，因此采用本仓库 bootstrap。

**验证**

bootstrap 只改变省略 `device` 的调用；任何 `device=...` 参数继续原样传递。双卡 collective、数据合同、显存容量及完整 DExplore import 已分别通过；下一 run 将验证环境构造、PPO epoch、checkpoint 与 TensorBoard。

## 2026-09-18 15:35:00 +0800 — V1.17.4 双卡 DExplore capacity smoke 已启动

- timestamp: 2026-09-18 15:35:00 +0800
- activity_id: ACT-20260918-153500-CMRESIDUAL-V1174-DEXPLORE-HOROVOD-SMOKE
- modification_version: V1.17.4
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L2
- approval: user-approved
- approval_basis: [`V1.17`](../plan/V1.17.md) V1.17.4 final contract；用户授权继续 GPU0/3 双卡运行。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: c42b481c97614c70e6b0d211ed38e845017934e5
- scope: [`run_dexplore_grab_teacher.py`](../../tools/run_dexplore_grab_teacher.py) 的 bootstrap 路径已解析为绝对路径；CUDA_VISIBLE_DEVICES=0,3，2 Horovod rank、每 rank 2048 env、horizon64、minibatch256、seed42、max_iterations1，从零初始化；仅 Horovod rank bootstrap 改写 Isaac Gym 未显式 device 默认值，外部源码/数据/任务合同不变。
- run_id: dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153500
- run_status: FAILED
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153500/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153500/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153500/train.log)。
- exit_code: 2；last_epoch: N/A；last_step: N/A；checkpoint: N/A；tensorboard: N/A；gpu_peak_mib: GPU0/3 均 4MiB。
- conclusion: INVALID_IMPLEMENTATION

**原因**

V1.17.4 rank-local bootstrap 已经通过静态、命令合同与相关设备错配诊断；本 run 是验证该最小修复的首个真实 capacity smoke。

**验证**

dry-run 通过并显示 Horovod 在两 rank 上执行 Task-local bootstrap；实际执行时因 bootstrap 是相对路径而外部 DExplore cwd 不同，两个 rank 均报找不到该文件并 exit 2，未初始化 GPU。路径已改为绝对路径。

## 2026-09-18 15:37:00 +0800 — V1.17.4 双卡 DExplore capacity smoke 路径修复后重试

- timestamp: 2026-09-18 15:37:00 +0800
- activity_id: ACT-20260918-153700-CMRESIDUAL-V1174-DEXPLORE-HOROVOD-SMOKE-RETRY
- modification_version: V1.17.4
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L2
- approval: user-approved
- approval_basis: V1.17.4 final contract；仅修复前一 smoke 的 Task-local bootstrap 路径，无研究变量变化。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: c42b481c97614c70e6b0d211ed38e845017934e5
- scope: GPU0/3、2 rank、每 rank2048 env、horizon64、minibatch256、seed42、max_iterations1；使用绝对 [`dexplore_horovod_rank_bootstrap.py`](../../tools/dexplore_horovod_rank_bootstrap.py) 路径，不改外部源码或数据。
- run_id: dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153700
- run_status: RUNNING
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153700/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153700/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1174_hvd2x2048_smoke_20260918_153700/train.log)（PENDING）。
- conclusion: INCONCLUSIVE

**原因**

上一 run 仅路径解析失败；绝对路径使 Horovod 子进程无论 cwd 为何都能执行 Ref2Dex bootstrap。

**验证**

launcher 静态检查与 5 个合同测试通过；运行终态待本次 launcher 退出后更新。

## 2026-09-18 17:16:00 +0800 — V1.17.5 双卡 DExplore 同步退出修复与 smoke 启动

- timestamp: 2026-09-18 17:16:00 +0800
- activity_id: ACT-20260918-171600-CMRESIDUAL-V1175-DEXPLORE-HOROVOD-SMOKE
- modification_version: V1.17.5
- operation_category: code、experiment、operation、documentation
- task_mode: change，随后进入 run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认继续 GPU0/3 双卡、从 `GRAB_00000114.pth` checkpoint 续训、额外训练 1000 epoch，并授权修复后直接启动大规模训练；本条先执行 smoke 门禁。
- skills_used: research-change-control、research-experiment-workflow
- branch: oyx
- base_commit: ebefee4e41916173658de57625fa412a963e7e74
- worktree_dirty: true（保留既有根级/ObjectInteractionCm activity、未跟踪指导/plan/工具；本次相关差异限定在 V1.17 plan、DExplore launcher/bootstrap/test 与本 activity。）
- scope: [`V1.17 最终计划`](../plan/V1.17.md) 追加 V1.17.5；[`run_dexplore_grab_teacher.py`](../../tools/run_dexplore_grab_teacher.py) 增加 resume checkpoint provenance 与额外 epoch 预算；[`dexplore_horovod_rank_bootstrap.py`](../../tools/dexplore_horovod_rank_bootstrap.py) 在 Task-local bootstrap 中补 Horovod `should_exit` 同步退出；[`test_dexplore_teacher_launcher.py`](../../tests/test_dexplore_teacher_launcher.py) 增加续训预算与 bootstrap 合同测试。外部 DExplore checkout、Isaac Gym 安装、数据、观测、动作、reward、物理和旧 outputs 不修改。
- run_id: dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600
- run_status: COMPLETED
- command: `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117_hvd/bin/python src/task/CmResidual/tools/run_dexplore_grab_teacher.py --run-id dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600 --activity-id ACT-20260918-171600-CMRESIDUAL-V1175-DEXPLORE-HOROVOD-SMOKE --modification-version V1.17.5 --launcher horovod --mode smoke --num-envs 2048`
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600/train.log)、[checkpoint](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth)、[TensorBoard event](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_smoke_20260918_171600/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/summaries/events.out.tfevents.1789722983.server)。
- completed_at: 2026-09-18 17:19:33 +0800
- exit_code: 0；last_epoch: 2；checkpoint: `GRAB.pth`；tensorboard: `events.out.tfevents.1789722983.server`；gpu_peak_mib: GPU0/3 均 23011MiB。
- conclusion: INCONCLUSIVE

**原因**

V1.17.4 最新 smoke 已完成 PPO epoch 和 checkpoint 写出，但 rank0 达到 `MAX EPOCHS NUM` 后独自退出，rank1 继续 backward/allreduce，导致 Horovod shutdown。rl_games 原版包含 `should_exit` broadcast，外部 DExplore 的 fork 丢失该同步退出逻辑。

**验证**

`dexplore_v117_hvd/bin/python -m py_compile` 通过；`graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_teacher_launcher.py` 为 `7 passed`；`git diff --check` 通过；Horovod dry-run 显示 GPU0/3 预检各 2MiB，resume formal 预算为 `max_iterations=1113`、额外 `262,144,000` env-steps。真实 smoke 完成两 rank 初始化、epoch1/2 PPO、checkpoint 和 TensorBoard event 写出；日志没有 `Horovod has been shut down`、traceback、OOM 或 non-finite，manifest 为 `COMPLETED`。

## 2026-09-18 17:20:00 +0800 — V1.17.5 双卡 DExplore checkpoint 续训正式启动

- timestamp: 2026-09-18 17:20:00 +0800
- activity_id: ACT-20260918-172000-CMRESIDUAL-V1175-DEXPLORE-HOROVOD-RESUME1000
- modification_version: V1.17.5
- operation_category: experiment、operation
- task_mode: run-only/operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认继续 GPU0/3 双卡、从 checkpoint 续训并额外跑 1000 epoch；本 run 已由 `ACT-20260918-171600-CMRESIDUAL-V1175-DEXPLORE-HOROVOD-SMOKE` 的 `COMPLETED` smoke 解锁。
- skills_used: research-experiment-workflow
- branch: oyx
- base_commit: ebefee4e41916173658de57625fa412a963e7e74
- worktree_dirty: true（包含本次 V1.17.5 代码/计划/activity 差异与既有无关 dirty 文件；运行产物不纳入 Git。）
- scope: GPU0/3、2 Horovod rank、每 rank2048 env、horizon64、minibatch256、seed42、`s1_airplane_lift`；从 [单卡 latest checkpoint](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000114.pth) 的 epoch114 续训，额外 1000 epoch，目标 final epoch 1114，`max_iterations=1113`，目标 `262,144,000` environment steps。外部 DExplore checkout、Isaac Gym、输入数据、观测、动作、reward 和物理不修改。
- run_id: dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000
- run_status: RUNNING
- command: `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117_hvd/bin/python src/task/CmResidual/tools/run_dexplore_grab_teacher.py --run-id dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000 --activity-id ACT-20260918-172000-CMRESIDUAL-V1175-DEXPLORE-HOROVOD-RESUME1000 --modification-version V1.17.5 --launcher horovod --mode formal --num-envs 2048 --resume-checkpoint outputs/Dexplore/dexplore_grab_teacher_v1171_formal_env2048_20260918_131215/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000114.pth --extra-epochs 1000`
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/)（PENDING）、[config](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/config.json)（PENDING）、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/run_manifest.json)（PENDING）、[train config](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train_config.yaml)（PENDING）、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train.log)（PENDING）。
- conclusion: INCONCLUSIVE

**原因**

正式续训只在双卡 smoke 通过后启动；该 run 是优化/teacher 训练，不产生抓取成功或泛化科学结论。

**验证**

启动前 GPU0/3 均为 2MiB、利用率 0%；输入 checkpoint 存在且 dry-run 已确认 resume formal 预算。

## 2026-09-18 18:03:23 +0800 — V1.19 流式 exact planner 与 Stage-A bypass 实现

- timestamp: 2026-09-18 18:03:23 +0800
- activity_id: ACT-20260918-180323-CMRESIDUAL-V119-STREAMING-PLANNER
- modification_version: V1.19
- operation_category: code、documentation
- task_mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户明确更正为按 [`指导/V1.19.md`](../指导/V1.19.md) 修改；[`V1.19 最终计划`](../plan/V1.19.md) 已定稿。
- skills_used: research-change-control
- branch: oyx
- base_commit: ebefee4e41916173658de57625fa412a963e7e74
- worktree_dirty: true（保留既有 V1.17.5 DExplore launcher/bootstrap/test/activity 差异、根级/ObjectInteractionCm activity、未跟踪指导/plan/工具；本次 V1.19 差异限定在下列 scope。）
- scope: [`ObjectInteractionCmv2 model`](../../../ObjectInteractionCmv2/model.py) 增加可选 object-query edge chunk；[`FrozenCmv2Adapter`](../../cm_v2_adapter.py) 增加 planner 专用 `predict_effect_only`；[`V1.18/V1.19 planner`](../../v118_planner.py) 将 active-env 轮转截断替换为全 active env microbatch；[`V1.18 task bridge`](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) 和 [`task config`](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/cfg/task/CmResidualGrabReferenceV118.yaml) 传入 `plannerEnvMicrobatch=16`、`interactionObjectChunk=32`；[`V1.18 agent`](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/v118_agent.py) 在 `cm_distill_coef==0` 时跳过 planner；[`Cmv2 tests`](../../../ObjectInteractionCmv2/tests/test_v1_3_spatial.py)、[`adapter tests`](../../tests/test_cmv2_adapter.py)、[`planner tests`](../../tests/test_v118_planner.py) 补 V1.19 回归；[`current_versions`](../../../../../docs/current_versions.yaml) 指向 CmResidual `V1.19`。
- run_id: N/A
- run_status: N/A（未启动新的 GPU smoke 或训练；正在运行的 V1.17.5 DExplore 正式续训未被停止或改写。）
- output: N/A
- conclusion: SUPPORTED（工程实现与 CPU 定向验证支持 V1.19 调度合同；尚未形成 GPU smoke 或学习效果结论。）

**原因**

V1.18 retry4 的 OOM 来自一次性对 active env × K candidate 构造 Cmv2 local interaction edge tensor；继续降低 `maxActiveEnvs` 会改变 teacher 覆盖。V1.19 改为 exact 数学不变的执行调度：planner 覆盖所有 active env，但按 env microbatch 串行；Cmv2 内部按 object-query chunk 串行 edge MLP；Stage A 在 `lambda_cm=0` 时不再构造 teacher。

**验证**

`/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile` 覆盖 model、adapter、planner、agent、task bridge：通过；`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/ObjectInteractionCmv2/tests/test_v1_3_spatial.py src/task/CmResidual/tests/test_cmv2_adapter.py src/task/CmResidual/tests/test_v118_planner.py`：`12 passed`；OmegaConf 单文件字段检查确认 observation=1442、K=8、`plannerEnvMicrobatch=16`、`interactionObjectChunk=32` 且无 `maxActiveEnvs`；`git diff --check`：通过。默认系统 `python` 因版本过旧无法解析项目类型标注，验证改用项目环境 `graspenv`。

## 2026-09-18 18:23:42 +0800 — V1.19 GPU2 128-env streaming planner smoke 未通过

- timestamp: 2026-09-18 18:23:42 +0800
- activity_id: ACT-20260918-182342-CMRESIDUAL-V119-GPU-SMOKE
- modification_version: V1.19
- operation_category: code、experiment、operation、documentation
- task_mode: run-only/operation，期间为修复 agent 兼容性短暂切换到 change
- change_level: L2
- approval: user-approved
- approval_basis: 用户要求“试一下 GPU”；[`V1.19 最终计划`](../plan/V1.19.md) 已授权 128 env、64 horizon、1 PPO update 的 GPU 工程 smoke。GPU0/3 正在运行 V1.17.5 DExplore，未触碰；GPU2 是当时最空闲卡，但第二次启动时已有另一用户进程约 3.5 GiB 占用。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: ebefee4e41916173658de57625fa412a963e7e74
- worktree_dirty: true（包含既有 V1.17.5 差异、V1.19 代码/文档差异和本次 agent 兼容性修复；outputs 不纳入 Git。）
- scope: physical GPU2，单 rank×128 env，window/horizon=64，K=8，`lambda_cm=0.10`，`plannerEnvMicrobatch=16`，`interactionObjectChunk=32`，1 epoch；[`V1.18/V1.19 agent`](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/learning/v118_agent.py) 增加 `infos["terminate"]` 缺失时回退到 `dones` 的 bootstrap mask；[`planner tests`](../../tests/test_v118_planner.py) 补静态回归。Cmv2 checkpoint、planner teacher 数学、reward、reference、buffer schema、训练数据、DExplore 运行和旧 outputs 不修改。
- run_id: cmresidual_v119_streaming_smoke_gpu2_20260918_182051
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_20260918_182051/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_20260918_182051/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_20260918_182051/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_20260918_182051/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_20260918_182051/train.log)、[buffer manifest](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_20260918_182051/cm_buffer/rank_000/manifest.json)。
- exit_reason: `KeyError: infos lacks terminate in V118PlannerAgent.play_steps after first rollout/env_step`；last_epoch: N/A；last_step: 0；checkpoint: N/A；metrics: empty。
- conclusion: INVALID_IMPLEMENTATION

- run_id: cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239
- run_status: FAILED
- output: [运行目录](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239/)、[manifest](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239/config.json)、[metrics](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239/metrics.jsonl)、[train log](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239/train.log)、[buffer manifest](../../../../../outputs/CmResidual/cmresidual_v119_streaming_smoke_gpu2_retry_20260918_182239/cm_buffer/rank_000/manifest.json)。
- exit_reason: PhysX 在 `env_step/post_physics_step` 报 CUDA illegal memory access；日志先出现 `PxgCudaDeviceMemoryAllocator fail to allocate memory 1681915904 bytes`，发生在共享 GPU2 条件下。last_epoch: N/A；last_step: 0；checkpoint: N/A；metrics: empty。
- conclusion: INVALID_IMPLEMENTATION

**原因**

第一次 GPU2 smoke 已经构造 task/agent/planner 并写出一个 transition-only buffer shard，未复现 V1.18 retry4 的 frozen-Cmv2 edge tensor OOM；但专用 agent 假定 `infos["terminate"]` 必然存在，当前 task wrapper 未提供该 key。修复后第二次运行越过该点，但 PhysX GPU simulation 在共享 GPU2 上因约 1.68 GiB allocator failure 进入 illegal memory access，未完成 PPO epoch。

**验证**

`/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile third_party/IsaacGymEnvs/isaacgymenvs/learning/v118_agent.py src/task/CmResidual/tests/test_v118_planner.py`：通过；`PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_v118_planner.py`：`5 passed`；两次 GPU smoke 的 resolved config 均确认 observation=1442、action=18、K=8、microbatch=16、object chunk=32、transition-only buffer。该证据支持 V1.19 已不在首个 planner Cmv2 forward 处触发旧 2 GiB edge OOM，但完整 128-env GPU smoke 仍未通过，科研效果仍为 `INCONCLUSIVE`。

## 2026-09-18 18:27:20 +0800 — V1.17.5 DExplore 双卡续训状态诊断

- timestamp: 2026-09-18 18:27:20 +0800
- activity_id: ACT-20260918-182720-CMRESIDUAL-V1175-DEXPLORE-STATUS
- modification_version: V1.17.5
- operation_category: diagnostic、operation
- task_mode: read-only/diagnostic
- change_level: L0
- approval: user-requested
- approval_basis: 用户询问当前 DExplore 训练状态以及物体是否被抓起。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: ebefee4e41916173658de57625fa412a963e7e74
- scope: 只读检查 [`run_manifest.json`](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/run_manifest.json)、[`train.log`](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train.log)、[`TensorBoard event`](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/summaries/events.out.tfevents.1789723261.server) 和 checkpoint 目录；未停止、重启或修改训练进程、checkpoint、外部 DExplore 源码、输入数据或运行配置。
- run_id: dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000
- run_status: RUNNING
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/)、[manifest](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/run_manifest.json)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train.log)、[checkpoint dir](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/)。
- last_epoch: 165；last_logged_mean_reward: 89.49；best_logged_mean_reward: 91.23 at epoch 138；last20_mean_reward: 84.417；latest_checkpoint: [`GRAB_00000152.pth`](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000152.pth)；latest_checkpoint_epoch: 152。
- conclusion: INCONCLUSIVE

**原因**

双卡 Horovod rank0/rank1 进程仍存活，GPU0/3 各约 23 GiB 占用并有计算利用率。训练从 epoch 115 续跑到至少 epoch 165，reward 从 24.68 快速升至 80–90 区间，高于启动 checkpoint 附近，但日志和 TensorBoard 只记录 `mean_rewards`、`episode_lengths`、loss/usage，没有 `success`、`lift`、object height 或视频指标。

**验证**

`ps` 确认 launcher、mpirun 和两个 rank 仍在运行；`nvidia-smi` 确认 GPU0/3 持续占用；日志解析得到 51 个 epoch 记录，最近 10 个为 86.87、81.74、81.80、85.15、85.39、84.63、87.00、86.24、88.43、89.49。TensorBoard 标量标签为 reward、episode length、loss 和 usage；DExplore reward 函数是 HOI imitation reward，包含 object tracking/contact/interaction graph，但当前运行没有直接输出“物体被抓起”的独立判据。因此当前只能说优化表现较好，不能宣称物体已经被抓起。

## 2026-09-18 20:00:35 +0800 — V1.17.5 DExplore epoch228 策略 rollout 可视化诊断

- timestamp: 2026-09-18 20:00:35 +0800
- activity_id: ACT-20260918-200035-CMRESIDUAL-V1175-DEXPLORE-VISUAL-E228
- modification_version: V1.17.5
- operation_category: diagnostic、operation
- task_mode: run-only/operation + read-only/diagnostic
- change_level: L0
- approval: user-requested
- approval_basis: 用户要求查看当前 DExplore 训练并可视化；本次只导出/分析 checkpoint，不停止、重启或修改正在运行的双卡训练。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 4871eed0ad939311dd04dc9c4a27634304a28a9e
- worktree_dirty: true（包含既有 V1.17.5/V1.19 代码、计划、activity 差异和本次输出 manifest；训练 checkpoint 与输出不纳入 Git。）
- scope: 从正在运行的 [DExplore 双卡续训目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/) 取当前最新已保存 checkpoint [`GRAB_00000228.pth`](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000228.pth)，在 GPU2 上做 1-env deterministic test/export；未修改训练进程、checkpoint、外部 DExplore 源码、输入数据、观测、动作或 reward。
- run_id: dexplore_v1175_visual_epoch228_20260918_195618
- parent_run_id: dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000
- run_status: COMPLETED
- command: `CUDA_VISIBLE_DEVICES=2 /home2/wyy/oyx_ws/.runtime_envs/dexplore_v117_hvd/bin/python dexplore/run.py --task Dexplore_Inspire --cfg_env dexplore/data/cfg/inspire.yaml --cfg_train dexplore/data/cfg/train/rlg/inspire.yaml --motion_file /home2/wyy/oyx_ws/Ref2Dex/outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/motion_input --output_path /home2/wyy/oyx_ws/Ref2Dex/outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/play --test --checkpoint /home2/wyy/oyx_ws/Ref2Dex/outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000228.pth --headless --sim_device cuda:0 --rl_device cuda:0 --graphics_device_id 0 --num_envs 1 --episode_length 432 --export_rl --export_output_dir /home2/wyy/oyx_ws/Ref2Dex/outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/rl_export`
- output: [可视化目录](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/)、[manifest](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/run_manifest.json)、[export log](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/export.log)、[rollout metrics](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/rollout_metrics.json)、[object z/lift plot](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/object_z_lift_compare.png)、[RL export tensor](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/rl_export/s1_airplane_lift/interaction_hand_inspire.pt)、[reference preview video](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/preview/s1_airplane_lift_human_object.mp4)、[qpos compare plot](../../../../../outputs/Dexplore/dexplore_v1175_visual_epoch228_20260918_195618/preview/s1_airplane_lift_qpos_compare.png)。
- training_status_at_check: parent run 仍为 RUNNING；日志最新 `epoch_num:238 mean_rewards:[86.72]`，当前已记录 124 个续训 epoch，最佳 logged mean reward 为 95.68 at epoch 223；最新已保存 checkpoint 仍为 epoch 228。
- rollout_metrics: 432 frames；RL object z start=0.9182605743、max=0.9182605743、end=0.9168922901；RL max lift from start=0.0m；reference max lift from start=0.2814320326m；object xyz RMSE vs reference=0.067584008；lift >2cm/>5cm/>10cm/>15cm/>20cm 的帧数均为 0。
- conclusion: REFUTED（仅针对“epoch228 确定性策略 rollout 已把物体抓/抬起来”这一命题；正在训练的后续未保存 checkpoint 仍需等新 checkpoint 再导出验证。）

**原因**

reward 曲线升高不能直接证明抓起，因为当前 DExplore 日志没有 success/lift/object-height 标量。本次导出的策略轨迹显示物体高度基本保持在初始高度附近，而参考轨迹有约 28.1cm 抬升；因此 epoch228 策略尚未学会把物体抬起。

**验证**

export 日志显示 checkpoint 成功加载并导出 1 条 RL rollout；`rollout_metrics.json` 与 `object_z_lift_compare.png` 均显示 RL 物体最大相对抬升为 0.0m，所有大于 2cm 的 lift 帧数为 0。GPU0/3 上原双卡训练进程仍存活，本次可视化使用 GPU2。

## 2026-09-18 20:28:37 +0800 — V1.17.5 DExplore reward 趋势复核

- timestamp: 2026-09-18 20:28:37 +0800
- activity_id: ACT-20260918-202837-CMRESIDUAL-V1175-DEXPLORE-REWARD-TREND
- modification_version: V1.17.5
- operation_category: diagnostic、operation
- task_mode: read-only/diagnostic
- change_level: L0
- approval: user-requested
- approval_basis: 用户询问 reward 是否仍在增长；本次只读解析训练日志和进程状态。
- skills_used: research-experiment-workflow、research-change-control
- branch: oyx
- base_commit: 4871eed0ad939311dd04dc9c4a27634304a28a9e
- scope: 只读检查 [train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train.log) 和 checkpoint 目录；未停止、重启或修改训练进程、checkpoint、外部 DExplore 源码、输入数据或运行配置。
- run_id: dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000
- run_status: RUNNING
- output: [运行目录](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/)、[train log](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train.log)、[checkpoint dir](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/)。
- last_epoch: 261；last_logged_mean_reward: 83.90；best_logged_mean_reward: 95.68 at epoch 223；last5_mean_reward: 85.438；last10_mean_reward: 87.032；last20_mean_reward: 87.270；last50_mean_reward: 89.154；last30_linear_slope: -0.184 reward/epoch；latest_checkpoint: [`GRAB_00000228.pth`](../../../../../outputs/Dexplore/dexplore_grab_teacher_v1175_hvd2x2048_resume114_extra1000_20260918_172000/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000228.pth)。
- conclusion: INCONCLUSIVE（reward 优化未持续创新高，近期表现为平台/轻微回落；reward 趋势本身仍不能证明抓取成功。）

**原因**

用户询问 reward 是否仍在增长；该判断需要用最新训练日志比较最近窗口、峰值和当前值，而不是只看单个 epoch。

**验证**

截至 20:28:31，训练进程仍存活；日志从 epoch115 记录到 epoch261。最近三个 10-epoch 窗口均值分别为 90.277（epoch232–241）、87.509（epoch242–251）、87.032（epoch252–261），最近 30 epoch 线性斜率为 -0.184 reward/epoch。
