# V1.21e snapshot-restore parity implementation

- timestamp: 2026-09-20T21:19:08+08:00
- activity_id: ACT-20260920-211908-CMRESIDUAL-V121E-SNAPSHOT-PARITY
- work_version: V1.21
- base_commit: `02e9862`
- branch: `ai/cmresidual/v121e-snapshot-restore`
- mode: change
- change_level: L2
- approval: 用户确认 [V1.21e FINAL plan](../plan/V1.21e.md) 并批准 GPU3 小规模运行

## Scope

实现 opt-in 的 collector snapshot、独立 fresh-process prefix/restore duplicate producer、canonical
shared-baseline physics score 与 offline parity gate。固定 16 states、`6/6/4` phase quota、actor mean
action、30 Hz two-substep、既有 object hard ceiling 和 duplicate-relative metric limits。默认 V1.21c
collector 输出保持不变；旧 artifact、V1.21d 证据和 vendor 均未修改。

## Verification and conclusion

- `python3 -m py_compile`：V1.21e modules 与 collector bootstrap 通过；
- `python3 -m pytest -q src/task/CmResidual/tests/test_v121e_snapshot.py src/task/CmResidual/tests/test_v121c_ranking.py`：`37 passed`；
- `git diff --check`：通过。

当前只完成工程接线，scientific conclusion 为 `INCONCLUSIVE`。正式 GPU3 run 将使用已提交的干净
commit 和独立 run_id；终态另写 Activity/experiment card。回滚为 revert 本 Activity 对应实现提交，
不删除任何运行产物。
