# V1.21d corrected duplicate calibration failed the hard ceiling

- timestamp: 2026-09-20T20:55:42+08:00
- activity_id: ACT-20260920-205542-CMRESIDUAL-V121D-CALIBRATION-FAILED
- work_version: V1.21
- git_commit: `5369ffc4408752ee5d48f64c943e983818b6ded7`
- branch: `ai/cmresidual/v121-cm-actor`
- mode: run
- change_level: L3
- approval: 用户于 2026-09-20 在确认 V1.21d 独立 GPU3 运行闸门后明确回复“可以”

## Run identity

- run_id: `cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050`
- run_status: `FAILED`
- conclusion: `INVALID_IMPLEMENTATION`
- scientific_conclusion: `INCONCLUSIVE`
- command: `python3 src/task/CmResidual/tools/run_v121c_duplicate_calibration.py --run-id cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050 --collection outputs/CmResidual/cmresidual_v121c_calibration_collection_gpu3_seedfix_20260920_1950 --gpu 3`
- output: `outputs/CmResidual/cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050/`
- manifest: `outputs/CmResidual/cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050/run_manifest.json`
- config: `outputs/CmResidual/cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050/config.json`
- metrics: `outputs/CmResidual/cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050/metrics.jsonl`（0 bytes；episode hard gate 在 outer metrics 写入前终止）
- log: `outputs/CmResidual/cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050/logs/eval.log`
- episode evidence: `outputs/CmResidual/cmresidual_v121d_duplicate_calibration_gpu3_20260920_2050/episodes/0000/records.jsonl`、`summary.json`
- checkpoint: none
- resource: physical GPU3 / logical `cuda:0`，preflight `6 MiB / 24576 MiB`

## Protocol and result

run 固定到 V1.21d corrected contract：pinned `dexplore_v117` runtime、30 Hz/two-substep、
PCG64 candidate-env permutation、shared canonical collected baseline 消去、object goal `t+6`、IG reference
`t+1`，并消费已通过 provenance hard gate 的 frozen `24/24/16` collection。

首个 source episode（episode 0 / seed 5909）的 11 个 selected states 完成 replay；task/reference indices
`11/11` 一致，numeric parity `0/11`，duplicate hard ceiling `0/11` 通过。观测范围：

```text
duplicate object position: min 0.0023741 m, median 1.8701711 m, max 6.4800906 m
duplicate object rotation: min 0.0472257 rad, median 2.6682627 rad, max 3.1346872 rad
duplicate DOF max abs:     min 0.0097215, median 0.0470690, max 1.2029319 rad
duplicate score delta:     min -160211.046875, median -7095.688965, max 15409.394531
```

即使最小 object position/rotation divergence 也分别超过 `5e-4 m / 5e-3 rad` hard ceiling。bootstrap
写出 episode records/summary 后抛出 `duplicate hard ceiling failed`；outer runner 正确写入
`FAILED / INVALID_IMPLEMENTATION` 并停止，没有处理其余 episodes。

## Evidence boundary and next step

本 run 消除了 V1.21d 列出的已知 score-clock、runtime 与 collection-validation confound 后，当前
long-prefix 9-env protocol 仍不能提供可接受的 duplicate noise。因此不能冻结 `epsilon_PhysX`，不得进入
512-state ranking，也不允许调小 epsilon、放宽 ceiling、换 seed、删除失败 state 或只选短 prefix。

该结果是当前 counterfactual replay protocol 的 `INVALID_IMPLEMENTATION`，不是 Cmv2 ranking hypothesis
的 `REFUTED`；Cm action-ranking 科学结论保持 `INCONCLUSIVE`。若继续该研究方向，需要新的指导/计划
定义不同的物理 GT 协议或 candidate validation 方法。

## Protected and rollback

未修改代码、数据、cache、checkpoint、vendor、frozen collection、旧 run 或任何研究阈值；GPU3 已释放。
运行产物保留为失败证据且不提交。本文档回滚只撤回 Activity/README 记录，不删除 run output。
