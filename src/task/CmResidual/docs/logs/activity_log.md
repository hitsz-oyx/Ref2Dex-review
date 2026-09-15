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
