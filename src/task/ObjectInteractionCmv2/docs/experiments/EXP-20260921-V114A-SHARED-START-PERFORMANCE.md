# EXP-20260921-V114A-SHARED-START-PERFORMANCE

- task：ObjectInteractionCmv2
- work_version：`V1.14`
- git_commit：`225c143`
- run_id：`cmv2_v114a_shared_start_gpu3_20260921T2306`
- run_status：`COMPLETED`
- conclusion：`SUPPORTED`

## 假设与口径

在不减少手点数量的前提下，把 `B=1,K=8,N=1024,H=4096` 的 candidate start KNN 从 K 次降为一次，
可使 endpoint 相对 duplicated-start 和历史 tuned endpoint 均至少加速 `1.5x`，且 shared endpoint +
32D encode/candidate 中位数不超过 `28 ms`。随机权重、FP32、TF32 关闭、GPU3、chunk `1024/1024`；
本实验只检验工程延迟与显存，不检验预测质量。

## 证据

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| duplicated-start endpoint median | `33.316 ms` | 对照 |
| shared-start endpoint median | `19.958 ms` | — |
| duplicated/shared speedup | `1.669x` | `>=1.5x` |
| historical `32.926 ms` / shared speedup | `1.650x` | `>=1.5x` |
| shared endpoint + 32D model | `25.851 ms` | `<=28 ms` |
| 32D candidate peak allocated | `158.746 MiB` | 不高于 128D |

运行证据位于 `research/v114_narrow_candidate_benchmark/output/cmv2_v114a_shared_start_gpu3_20260921T2306/`：
`run_manifest.json`、`config.json`、`metrics.jsonl`、`run.log` 和 `summary.json`。

## 结论边界

四个自动门槛全部通过，因此“共享 start KNN 能在 H=4096 的八 candidate 工程路径上达到计划延迟门槛”得到支持。
该结论不证明真实五组输入的接触覆盖、训练收敛、预测质量或 rollout 帧率；真实 cache 当前未挂载，覆盖审计仍是训练前门禁。
