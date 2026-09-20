# V1.21c revised multi-env noise Gate 0

- timestamp: 2026-09-20T18:53:41+08:00
- activity_id: ACT-20260920-185500-CMRESIDUAL-V121C-NOISE-GATE0
- work_version: V1.21
- git_commit: `8e738999cc11515d33c5e1d33ef1e1b950abd966`
- base_commit: `8e738999cc11515d33c5e1d33ef1e1b950abd966`
- branch: `ai/cmresidual/v121-cm-actor`
- scope: corrected 30 Hz 1-state/9-env prefix + candidate duplicate engineering smoke
- approval: user-approved
- run_id: `cmresidual_v121c_noise_gate0_gpu3_20260920_1900`
- run_status: COMPLETED
- conclusion: N/A
- scientific_conclusion: INCONCLUSIVE

## Fixed run contract

- physical GPU3 / logical cuda:0，preflight `6 MiB` used；
- seed 5909，`s1_airplane_lift`，9 env，1 prefix action + 1 candidate action；
- 每个 native action 为 30 Hz control step，包含两个 `1/60 s` PhysX substeps；
- `PCG64(5909)` 将 `[0,0,1,2,3,4,5,6,7]` 置换到 env slots；
- numeric parity 只记录诊断；finite、task/reference indices 与 duplicate object ceiling 是 hard gate；
- object duplicate ceilings：position `5e-4 m`，rotation `5e-3 rad`。

输入 checkpoint、manifest、motion tensor、URDF、mesh、DExplore entrypoint 与 Isaac Gym binding
均在 run manifest 固定路径和 SHA256；代码提交和计划 SHA 也已固定。

## Result

- env candidate mapping：`[3,0,6,1,2,4,7,0,5]`；candidate 0 位于 env 1 与 env 7；
- task/reference indices：9 env 全部为 `[0,0,0,1]`，hard gate 通过；
- duplicate object position divergence：`1.5955379240040202e-06 m`；
- duplicate object rotation divergence：`5.95300923449941e-07 rad`；
- duplicate DOF max absolute divergence：`0.0`；
- duplicate hard ceiling：通过；
- numeric parity diagnostic：`parity_valid=false`，公开 float state 最大差 `0.2536234558`；
  其中 DOF position 最大差 `0.0175827742 rad`、DOF velocity 最大差 `0.2536234558 rad/s`。

这个 smoke 支持修订后的 prefix/30 Hz/candidate permutation/duplicate hard-gate 接线可运行；它不证明
Cm ranking、策略效果或 PhysX exact clone，也不表示底层多环境数值分叉已解决。没有启动 collector、
64-state calibration、512-state ranking 或 PPO。

## Evidence

- output：`outputs/CmResidual/cmresidual_v121c_noise_gate0_gpu3_20260920_1900/`
- manifest：`outputs/CmResidual/cmresidual_v121c_noise_gate0_gpu3_20260920_1900/run_manifest.json`
- config：`outputs/CmResidual/cmresidual_v121c_noise_gate0_gpu3_20260920_1900/config.json`
- result：`outputs/CmResidual/cmresidual_v121c_noise_gate0_gpu3_20260920_1900/replay_smoke/smoke_result.json`
- log：`outputs/CmResidual/cmresidual_v121c_noise_gate0_gpu3_20260920_1900/logs/eval.log`
- checkpoint：无（engineering smoke）
- metrics.jsonl：无（单次 smoke 以 `smoke_result.json` 为结构化指标）

## Next step and rollback

下一门禁是实现并执行独立的 64-state duplicate-noise calibration，冻结 `epsilon_PhysX`；在它通过前
不得进入 512-state ranking。运行产物保留为审计证据；代码回滚入口为提交 `8e73899` 之前的状态，
但不回滚 `8cacd30` 的 30 Hz 修复。
