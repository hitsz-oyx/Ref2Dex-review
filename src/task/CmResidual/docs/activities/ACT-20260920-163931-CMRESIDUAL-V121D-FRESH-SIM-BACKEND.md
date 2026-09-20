# V1.21d native-init fresh-simulator branch backend

- timestamp: 2026-09-20 16:39:31 +0800
- activity_id: ACT-20260920-163931-CMRESIDUAL-V121D-FRESH-SIM-BACKEND
- work_version: V1.21
- mode: change
- change_level: L2（Task-local branch backend、physics schema 与验证合同）
- approval: user-approved
- approval_basis: 用户于 2026-09-20 审阅 V1.21d 草案后明确要求“定稿实现”；不包含任何真实 GPU/PhysX run
- branch: ai/cmresidual/v121-cm-actor
- base_commit: 4b7bd0a

## Scope

按 V1.21d 最终计划实现 runtime-neutral 的 native-init fresh-sim protocol：每个 state 必须由
factory 创建一个新 9-env simulator，从 DExplore `StateInit.Start` 原生初态开始，先校验 collector
初态身份，再重放非空 executed-action prefix。candidate 前公开状态 parity 通过后，才允许 Gate C
的 8+1 branch 或 Gate B 的 9-way all-duplicate branch；无论通过或失败均销毁 simulator。

同时实现 12-state Gate A 的 phase/prefix 分位选样、64-state Gate B 的 9 replicas/36 pairs
校准、V1.21d physics provenance schema，以及需要显式 GPU 和 `--execute` 的 Gate 0 runner。

## Changed

- [v121d_fresh_sim.py](../../v121d_fresh_sim.py)：fresh simulator 生命周期、native-initial 和
  pre-candidate parity、prefix done/reset hard stop、8+1/9-way branch mapping、Gate A selection、
  Gate B pairwise calibration。
- [v121d_artifacts.py](../../v121d_artifacts.py)：版本化 backend/schema、lifecycle/parity/duplicate
  字段验证，并显式拒绝把旧 tensor-restore records 当作 V1.21d 证据。
- [run_v121d_native_init_smoke.py](../../tools/run_v121d_native_init_smoke.py) 与
  [v121d_native_init_smoke_bootstrap.py](../../tools/v121d_native_init_smoke_bootstrap.py)：先用单 env
  原生 collector 生成 episode artifact，销毁 source sim，再以 fresh 9-env task 重放的 Gate 0 入口；
  manifest 固定 V1.21d plan/backend 和输入 SHA。
- [test_v121d_fresh_sim.py](../../tests/test_v121d_fresh_sim.py)：覆盖生命周期、初态/prefix failure、
  done/reset stop、8+1、all-duplicate、Gate A/B 和 schema 拒读。
- V1.21d plan 由 draft 原地定稿，并更新 Task README 和 Activity index。

## Protected

保持 V1.21c 的 DExplore checkpoint、frozen Cmv2、1442-D observation、18-D native action、
`H_ref=6`、reference clock、candidate RNG、canonical IG、score、phase quota、bootstrap、科学 gate、
坐标、单位、GT、split 和 checkpoint 解释不变。旧 V1.21c tensor-restore 入口作为失败历史保留；
未修改 vendor、`src/base`、数据、cache、checkpoint、既有 outputs 或用户指导，未运行 PhysX/PPO。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121d_fresh_sim.py src/task/CmResidual/tests/test_v121c_ranking.py`：`25 passed`。
- 四个新增 runtime/runner 文件的 `python3 -m py_compile`：通过。
- `python3 tools/verify.py --changed`：通过。
- `git diff --check`：通过。
- 全 `src/task/CmResidual/tests/` 在 collection 阶段因环境缺少既有可选依赖 `smplx` 停止；受影响的
  三个旧测试为 `test_cmv2_action_evaluator.py`、`test_dexycb_base.py`、`test_v118_planner.py`，没有
  测试执行。本 Activity 未安装依赖、未修改共享导入链。

## Scientific conclusion and next step

静态合同和 fake-runtime 接线为工程 `SUPPORTED`；未执行真实 simulator，因此 native-init 是否能解决
此前 9-env q/dq 分叉仍为 `INCONCLUSIVE`，Cm ranking 科学结论也保持 `INCONCLUSIVE`。下一步只有在
用户另行确认 GPU、预算和 run manifest 后，才运行独立 V1.21d Gate 0；通过后再分别审批 collector、
Gate A、Gate B 和 Gate C。

## Rollback

回退本 Activity 对应提交即可移除 V1.21d Task-local backend、schema、runner 和测试，并把计划/README
恢复到实现前状态；不得删除或覆盖 V1.21c 失败证据、用户指导、vendor、数据、cache、checkpoint 或 outputs。
