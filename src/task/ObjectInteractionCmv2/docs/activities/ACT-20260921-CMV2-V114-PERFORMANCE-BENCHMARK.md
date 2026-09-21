# V1.14 Candidate 性能 Benchmark

- timestamp：`2026-09-21T22:01:19+08:00`
- activity_id：`ACT-20260921-CMV2-V114-PERFORMANCE-BENCHMARK`
- Task / work_version：`ObjectInteractionCmv2` / `V1.14`
- git_commit：默认 chunk run `2d8373ef8658bb96df10bb568d6aa7ee2717c15a`；tuned chunk run `5a7382987564ad2554f643467295f5eadfba768f`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope / impact：随机初始化 synthetic GPU 性能 benchmark；不训练、不读取 cache、不改变研究语义；`L1`。
- approval：用户在 V1.14 实现交接后明确要求继续；V1.14 FINAL plan 已固定性能协议与门槛。
- run_status：两个 run 均 `COMPLETED`。
- scientific conclusion：`REFUTED`；32D 未满足 V1.14 的全部性能门槛。该结论不涉及预测精度。

## Runs 与证据

共同合同为 GPU3 RTX 3090、FP32、TF32 off、seed 42、随机初始化、`B=1,K=8,N=1024,E=32`、
endpoint `H=4096`、30 次预热、100 次模型计时和 10 次 endpoint 计时。

1. `cmv2_v114_narrow_candidate_benchmark_20260921T2200CST`
   - output：`src/task/ObjectInteractionCmv2/research/v114_narrow_candidate_benchmark/output/cmv2_v114_narrow_candidate_benchmark_20260921T2200CST`
   - chunks：object `128`、hand `256`
   - 128D→32D：local `5.7134→5.5020 ms`，candidate `9.1884→9.1284 ms`，峰值显存
     `83.97→45.57 MiB`；endpoint `63.4752 ms`；估算总时间 `73.5524→73.5334 ms`。
   - manifest、config、metrics、log 与 `summary.json` 均完整，结论 `REFUTED`。

2. `cmv2_v114_narrow_candidate_chunk1024_benchmark_20260921T2204CST`
   - output：`src/task/ObjectInteractionCmv2/research/v114_narrow_candidate_benchmark/output/cmv2_v114_narrow_candidate_chunk1024_benchmark_20260921T2204CST`
   - chunks：object `1024`、hand `1024`
   - 128D→32D：local `4.6746→2.4484 ms`（`1.91x`），candidate `6.2218→4.4160 ms`
     （`1.41x`），encode+candidate `7.0333→5.4922 ms`（`1.28x`），峰值显存
     `447.84→159.07 MiB`；endpoint `32.9257 ms`；估算总时间 `39.9590→38.4179 ms`（`1.04x`）。
   - 只有显存门槛通过；local 3x、candidate 2x 和含 endpoint 1.5x 门槛均失败，结论 `REFUTED`。

## 操作结论与回滚

chunk 调整不改变数学语义，并使 32D 的估算总时间相对默认 chunk 约加速 `1.91x`；但大 chunk 会把
32D candidate 峰值显存从约 `45.57 MiB` 提高到 `159.07 MiB`，不能直接套用于大训练 batch。
两次运行均无 checkpoint。回滚只需不采用 planner-oriented chunk；输出为忽略的独立证据目录，不覆盖旧运行。
