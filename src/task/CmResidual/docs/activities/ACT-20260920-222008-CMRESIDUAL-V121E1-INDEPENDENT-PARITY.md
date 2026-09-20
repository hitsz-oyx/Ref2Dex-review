# V1.21e.1 independent single-env snapshot parity implementation

- timestamp: 2026-09-20T22:20:08+08:00
- activity_id: ACT-20260920-222008-CMRESIDUAL-V121E1-INDEPENDENT-PARITY
- work_version: V1.21
- base_commit: `a1d4dfcce9d85335ed3248476cb4498392c308a5`
- branch: `ai/cmresidual/v121e-snapshot-restore`
- mode: change
- change_level: L2
- approval: 用户批准 [V1.21e.1 FINAL 微补充](../plan/V1.21e.1.md) 的 Task-local 修复与 GPU3
  6-state/24-process 小规模运行

## Scope

将 V1.21e parity 从同一 simulator 的 two-env duplicate 改为每 state 四个独立 fresh subprocess，
每个 subprocess 只有一个 env：A1/A2 replay 相同 episode prefix，B1/B2 direct restore 相同 collector
snapshot。只读复用正式 V1.21e run 的 frozen snapshot/episode artifacts，不重新采集或改写源文件；
样本缩为 deterministic `2 moving / 2 contact / 2 precontact`。

GPU tensor restore 删除 setter 后、下一次 simulate 前的 refresh/observation 重算；pre-action pose、IG、
raw observation 和 score baseline 只读 collector canonical snapshot。producer 记录 snapshot、setter
input、bookkeeping 与 action identity，并 hard-gate setter input 的 exact copy。V1.21c initial restore 的
同类顺序一并修正。旧 V1.21d/V1.21e outputs 与结论均未改写。

## Verification and rollback

- DExplore runtime：`41 passed`（`test_v121e_snapshot.py` + `test_v121c_ranking.py`）；
- `python3 -m py_compile`：四个修改模块通过；
- `python3 tools/verify.py --changed`：`VERIFY PASS`；
- `git diff --check`：通过。

当前只完成工程合同与测试，scientific conclusion 为 `INCONCLUSIVE`。正式 GPU3 run 使用提交后的
干净 commit 和新 run_id，终态另写 Activity/experiment card。回滚为 revert 本 Activity 对应的
Task-local implementation commit；不删除任何旧运行产物。
