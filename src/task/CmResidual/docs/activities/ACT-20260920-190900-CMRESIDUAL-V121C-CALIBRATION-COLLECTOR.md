# V1.21c calibration collector implementation

- timestamp: 2026-09-20T19:09:00+08:00
- activity_id: ACT-20260920-190900-CMRESIDUAL-V121C-CALIBRATION-COLLECTOR
- work_version: V1.21
- git_commit: pending at record creation
- base_commit: `10b0147115ff45908a6d461c5b8bba8051201c1f`
- branch: `ai/cmresidual/v121-cm-actor`
- mode: change
- change_level: L1（实现已批准 V1.21c calibration collection 合同，不改变研究语义）
- approval: covered by user-approved V1.21c plan and explicit continuation request

## Changed

新增 Task-local 两阶段 calibration collection 接线：

- episode bootstrap 通过官方 DExplore player/model/checkpoint 生成 Gaussian sampled native actions；
- 每个 episode 使用一个单独进程和 `seed=5909+episode_id`，关闭 early termination，固定 Start init；
- 保存 initial DOF/root/task indices、完整 executed-action/done/reference/progress history；
- pre-action state 使用 1442-D raw observation 和 executed-action prefix hash 生成 canonical state_id；
- active/phase 严格使用 5 个 contact bodies、5 个 tips、256-point object surface、`H_ref=6`
  以及 V1.21c 的 force/distance/motion 阈值；
- 外层按 episode 顺序收集，按 state_id 确定性选择 `24 moving + 24 contact + 16 precontact`，
  不跨 bucket 补位，最多 64 episodes；
- manifest 固定代码、指导/plan SHA、checkpoint/input/URDF/mesh SHA、seed、GPU、30 Hz/two-substep、
  surface seed、阈值、命令模板和输出入口。

这一步只收集 calibration state/prefix，不执行 9-env duplicate replay，也不冻结 `epsilon_PhysX`。

## Protected

observation、native action mapping、reward、GT、reference clock、坐标、单位、split、checkpoint、
30 Hz physics 合同、duplicate ceiling 与 epsilon 公式均未改变。没有修改 vendor、`src/base`、数据、
cache、checkpoint 或旧 outputs。

## Verification

- `python3 -m pytest -q src/task/CmResidual/tests/test_v121c_ranking.py`：`21 passed`；
- 两个新增入口 `python3 -m py_compile`：通过；
- `git diff --check`：通过；
- GPU0 preflight：`22119 / 24576 MiB` used，超过固定 `1024 MiB` capacity gate；按 plan 未启动 run、
  未创建 run output，也未自动换卡。

## Next step and rollback

下一步需要 GPU0 释放，或由用户明确批准改变计划中的默认 collector GPU。资源确认后先运行 collection，
检查 `24/24/16` 配额，再实现/执行 9-env duplicate calibration。回滚本 Activity 对应提交即可移除
collector 工具和测试，不影响已通过的 Gate 0 或 30 Hz 修复。
