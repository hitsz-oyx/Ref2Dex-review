# V1.21c GPU3 prefix-replay parity smoke failed

- timestamp: 2026-09-20T16:11:39+08:00
- activity_id: ACT-20260920-161139-CMRESIDUAL-V121C-PREFIX-PARITY-FAILED
- work_version: V1.21
- mode: change → run → change
- change_level: L1 Task-local runner；实际 PhysX smoke 按已批准的 L3 运行门禁执行
- approval: user-approved
- approval_basis: 用户在恢复会话中明确指定 GPU3 并要求继续；V1.21c FINAL plan 已批准实现，正式
  512-state ranking、PPO 和长训仍未授权
- branch: ai/cmresidual/v121-cm-actor
- base_commit: 2e256f72d418178070c8f6342eb2de6a0624c4a8
- implementation_commits: b2ba0a7, fe64dae, de2a87b, 2a9c23e, 50971e8, 2e256f7

## Scope

新增 V1.21c Task-local smoke launcher/bootstrap，在 physical GPU3 创建 9 个官方 DExplore Inspire
环境，从一个保存的 episode initial state 恢复，执行一个 recorded native-action prefix step，在
candidate step 前检查 q/dq、actor roots 和 task indices 的 `1e-5` parity；仅在通过后才允许执行
8 candidates 加 candidate-0 duplicate。launcher 在启动前固定输入 SHA、git commit、GPU、seed、
protocol、config 和 run manifest；未加载或更新 PPO/Cmv2 参数。

## Changed

- 新增 `tools/run_v121c_prefix_smoke.py`：GPU3/input preflight、独立 output/config/manifest/log、失败
  终态与不可覆盖 run_id。
- 新增 `tools/v121c_prefix_smoke_bootstrap.py`：官方 DExplore task bootstrap、episode artifact、真实
  prefix replay、9-env branch、duplicate ceiling 和分量级 parity diagnostics。
- 扩展 `tests/test_v121c_ranking.py`：固定 9-env/seed/device 命令合同和直接执行入口。

## Runs and terminal evidence

统一命令形式：

```text
python3 src/task/CmResidual/tools/run_v121c_prefix_smoke.py --run-id <run_id>
```

- `cmresidual_v121c_prefix_duplicate_smoke_gpu3_20260920_1600`: `FAILED`；Isaac Gym binding 路径
  未进入 `PYTHONPATH`，未创建 simulator。
- `..._1605`: `FAILED`；Isaac Gym/Torch 导入顺序错误，未创建 simulator。
- `..._1610`: `FAILED`；motion root 未转发到 DExplore cfg，未创建 simulator。
- `..._1615`: `FAILED`；首次真实 GPU PhysX prefix 后 pre-branch parity 失败。
- `..._1620`: `FAILED`；分量诊断确认 q position/velocity 最大误差分别为
  `0.0070017576 rad` / `0.6476358175 rad/s`；actor root position 最大误差约 `1.99e-6 m`，
  task indices 完全一致。
- `..._1630`: `FAILED`；将 synthetic prefix 改为官方 mapping 下的 native hold action后，q
  position/velocity 误差仍为 `0.0111185312 rad` / `0.5535187721 rad/s`；root position 最大误差
  仍约 `1.99e-6 m`，task indices 完全一致。

最终 run output：
`outputs/CmResidual/cmresidual_v121c_prefix_duplicate_smoke_gpu3_20260920_1630/`；manifest、config、
`logs/eval.log`、episode artifact、pre-candidate snapshot 和 `smoke_result.json` 均保留。run_status 为
`FAILED`，最后完成 `1` 个 prefix control step，candidate step 未执行；best metric、checkpoint 和
`metrics.jsonl` 均不适用。GPU3 运行后回落到约 `6 MiB`，无遗留进程。

## Protected

没有放宽 `1e-5` branch parity 或 duplicate hard ceiling；保持 DExplore 1442-D observation、598-D
motion source、18-D native action、H_ref=6、坐标/单位、GT、split、physics config、checkpoint 解释和
frozen Cmv2 不变。未修改 vendor、`src/base`、数据、cache、checkpoint 或既有 outputs；每次失败使用
新 run_id，历史失败产物未删除或覆盖。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py`: `17 passed`。
- launcher dry-run：固定 checkpoint/Cmv2/data/URDF/mesh SHA 与 GPU3，preflight 通过。
- `python3 -m py_compile`、`git diff --check`：通过。
- 每次实现提交前 `python3 tools/verify.py --changed`: `VERIFY PASS`。
- 最终真实 smoke：`FAILED`，原因是 pre-candidate q/dq parity 超过计划阈值；不是 OOM、输入 SHA
  不匹配或 root/task-index parity 问题。

## Scientific conclusion and blocker

run_status 为 `FAILED`，protocol conclusion 为 `INVALID_IMPLEMENTATION`；科学结论保持
`INCONCLUSIVE`。当前 9-env GPU PhysX prefix branch 不能满足 V1.21c 的公开 q/dq parity，故按
FINAL plan 停止，未执行 candidates、duplicate anchor、512-state collection 或 ranking。后续必须先
形成并批准新的 branch-replay 实现方案；不能通过放宽阈值、忽略 q/dq 或直接 clone 中途 state 绕过。

## Rollback

代码回滚入口为本 Activity 列出的六个 Task-local 提交；运行产物作为失败证据保留，不删除。回滚不触碰
用户指导、FINAL plan、vendor、数据、cache、checkpoint 或历史 outputs。
