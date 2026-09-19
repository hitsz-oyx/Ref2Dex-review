# V1.21b DExplore Cm-off parity bootstrap

- timestamp: 2026-09-19 22:00:00 +0800
- activity_id: ACT-20260919-220000-CMRESIDUAL-V121B-CMOFF-PARITY
- work_version: V1.21
- mode: change
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 V1.21b 草案并指示继续。
- branch: ai/cmresidual/v121-cm-actor
- base_commit: 380b13ffc0915df6e6a4c13888a5faa8c32a35a2

## Scope

落实 [V1.21b](../plan/V1.21b.md) 第 1 阶段：为 vendor DExplore 的既有
Task-local DDP launcher 增加明确的 Cm-off bootstrap。它是随后 Cm runner 的同一
入口身份，但本次只允许 `cm_distill_coef=0`，不启动训练且不实现 geometry bridge。

## Changed

- 新增 [Cm-off bootstrap](../../tools/dexplore_cm_off_rank_bootstrap.py)：要求显式
  `--cm-distill-coef 0`，进入 DExplore 前拒绝任何预导入 Cm/Cmv2 module，随后委托
  既有 DDP rank bootstrap。
- [DDP launcher](../../tools/run_dexplore_v120_ddp.py) 增加可选 task-local rank
  bootstrap、零 Cm 系数和 work-version metadata；默认 V1.20 bootstrap/命令不变。
- [定向测试](../../tests/test_dexplore_v120_ddp_compat.py) 覆盖非零拒绝、零系数
  passthrough 与 Cm-off bootstrap command；V1.21b 计划由用户确认定稿。

## Protected

未修改 `third_party/DExplore`、其 reward/observation/action/physics、V1.20e 输入数据、
DDP facade 的默认路径、checkpoint、数据、cache 或任何 output。Cm-off 分支不导入、
构造或更新 Cmv2。

## Verification

- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q src/task/CmResidual/tests/test_dexplore_v120_ddp_compat.py src/task/CmResidual/tests/test_v118_planner.py`：15 passed。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m py_compile ...`：通过。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python tools/verify.py --changed`：`VERIFY PASS`。

## Result and next step

工程实现为 `SUPPORTED`，尚无训练或科学结论。下一步是在独立 output 中以此 bootstrap
运行 V1.21b 第 2 阶段的小规模 Cm-off DDP engineering smoke；通过后才评估 4 x 2048
容量门与 geometry bridge。

## Rollback

回退本 Activity 同一提交中的 Task-local bootstrap、launcher 参数和测试即可；不删除
或改写任何 DExplore vendor 文件、数据、checkpoint 或 outputs。
