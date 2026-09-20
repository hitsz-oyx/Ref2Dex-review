# V1.21b DExplore–Cmv2 geometry bridge

- timestamp: 2026-09-20 00:00:00 +0800
- activity_id: ACT-20260920-000000-CMRESIDUAL-V121B-GEOMETRY-BRIDGE
- work_version: V1.21
- mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 V1.21b 最终计划并继续执行第 3 阶段。
- branch: ai/cmresidual/v121-cm-actor
- base_commit: de795e0

## Scope

落实 [V1.21b](../plan/V1.21b.md) 第 3 阶段的纯 Torch geometry bridge；不接入 PPO、
不改 vendor DExplore、不给 Cmv2 传梯度，也不启动 Cm-on run。

## Contract evidence and changed

- 冻结 Cmv2 checkpoint meta 固定 `coordinate_frame=object_pose_t`、1024 object points、
  1538 hand points、16 tokens、`dt=1/30`；训练缓存包含 `inspire_rl` stream，不是只由
  MANO 组成。
- DExplore 使用的 `inspire_hand_right.urdf` 与现有 Cmv2/Inspire chain 的 SHA256 完全相同
  (`7d0023…80f58`)；DExplore config 固定 `ballSize=1.0`、30 Hz。
- 新增 [bridge](../../dexplore_cm_geometry.py)：从 DExplore pre-action native 18-DoF 和
  Isaac root `[xyz,xyzw,...]` 生成 world-space 1024 object / 1538 hand points+normals，并以
  pinned URDF FK 将原生 DExplore action candidate 转成 nominal one-step hand flow。
- 映射逐项复制 released `Dexplore_Inspire._action_to_pd_targets`：wrist delta、finger
  `[0,1]` range mapping 和六个 Inspire coupling 均保持。全零 policy action 对应 finger
  range midpoint，而不是 hold-current；bridge 不错误地把它归零。

## Verification

- 新增 [geometry tests](../../tests/test_dexplore_cm_geometry.py)：PD/coupling、xyzw pose、
  pinned airplane/Inspire URDF 的 1024/1538 shape、finite、unit normals、相同 candidate 的
  deterministic flow。
- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q ...`：21 passed
  （geometry、DDP、Cmv2 adapter、V1.18 planner suites）。
- `py_compile` 与 `tools/verify.py --changed` 在提交前执行。

## Protected and next step

保护 V1.20e 的 598-D input、18-D action、reward、termination、physics、DDP contract、
checkpoint 解释和 vendor source。bridge 只消费 pre-action state；尚不加载 Cmv2 checkpoint，
不改变 actor/critic/optimizer 或随机数流。

下一步需要单独实现 Task-local DExplore agent hook：只在非零 Cm 系数时懒加载 frozen Cmv2，
由该 bridge 产生 detached teacher action/weight；Cm-off 行为必须逐字段保持当前已通过的容量
run。完成 hook contract tests 后才可申请 Cm-on engineering smoke。

## Rollback

回退本 Activity 对应提交即可移除 bridge 和测试；不影响 vendor、数据、checkpoint 或 outputs。
