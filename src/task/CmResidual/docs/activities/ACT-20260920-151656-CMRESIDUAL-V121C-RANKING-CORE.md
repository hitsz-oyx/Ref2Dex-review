# V1.21c native-action ranking core

- timestamp: 2026-09-20 15:16:56 +0800
- activity_id: ACT-20260920-151656-CMRESIDUAL-V121C-RANKING-CORE
- work_version: V1.21
- mode: change
- change_level: L2 (Task-local)
- approval: user-approved
- approval_basis: 用户在 V1.21c 计划定稿后明确要求“你实现吧”。
- branch: ai/cmresidual/v121-cm-actor
- base_commit: a530ad4
- git_commit: 3abc6a74e65fa587ac9fb503927d1005509c5913

## Scope

按 [V1.21c 最终计划](../plan/V1.21c.md) 实现 Task-local 的 P=1 native-action
ranking 合同。没有启动正式 PhysX collection/ranking run、PPO、P=3、异步 worker、Cm
微调或长训。

## Changed

- 新增 [v121c_ranking.py](../../v121c_ranking.py)：固定 candidate RNG、state identity、
  active/phase quota、DExplore 256-point canonical IG、current-object-local right
  composition、Cm score、Cm/PhysX pairwise/top-1、duplicate-anchor calibration、
  episode-block bootstrap。
- 新增 [v121c_artifacts.py](../../v121c_artifacts.py)：state/physics/episode replay
  schema 校验、非覆盖 JSON manifest 写入和输入文件 SHA256。
- 新增 [eval_v121c_ranking.py](../../tools/eval_v121c_ranking.py)：只读 validated replay
  `.npz` 的离线 metrics runner，写入新 run 目录、manifest、metrics 和 replay snapshots。
- 新增 [test_v121c_ranking.py](../../tests/test_v121c_ranking.py)：13 项纯合同测试，覆盖
  candidate 可复现性、native mapping、IG/pose、validity、quota、parity、bootstrap、
  schema 和 runner smoke。
- 将已确认的 [plan/V1.21c.md](../plan/V1.21c.md) 定稿为 FINAL；旧计划和用户指导保持只读。

## Protected

保持 DExplore 1442-D observation、(T,598) motion source、18-D native action、H_ref=6、
坐标/单位/GT/split/checkpoint 解释和 Cmv2 frozen/no-grad 语义不变；未修改
`third_party/`、`src/base/`、数据、cache、checkpoint、既有 output、旧计划或用户指导。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py`: 13 passed。
- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py src/task/CmResidual/tests/test_dexplore_cm_geometry.py src/task/CmResidual/tests/test_cmv2_adapter.py`: 19 passed。
- `python3 -m py_compile`：新增 4 个 Python 文件通过。
- `python3 tools/verify.py --changed`: `VERIFY PASS`（统一入口的 Python 3 执行）。
- `python tools/verify.py --changed`: 环境 `python` 指向 Python 2，因 `tools/verify.py`
  未声明 UTF-8 而在非 ASCII 注释处 SyntaxError；不将此结果声称为通过。
- 计划相对链接、trailing-whitespace 和 untracked 新代码审计通过。

## Scientific conclusion and next step

本 Activity 只证明实现合同、schema、统计和离线 runner 的工程接线；没有 PhysX ranking
证据，因此 Cm 的科学结论仍为 `INCONCLUSIVE`。下一步需单独确认 GPU0/官方 DExplore
runtime 和输入资产 SHA，完成 collector 的 episode-prefix replay、64-state duplicate
calibration 与 run manifest 后，再申请独立的 V1.21c ranking run。

## Rollback

回退提交 `3abc6a74e65fa587ac9fb503927d1005509c591` 即可移除本次 Task-local 实现和计划；
不删除、不覆盖用户指导、vendor、数据、cache、checkpoint、既有 outputs 或历史记录。
