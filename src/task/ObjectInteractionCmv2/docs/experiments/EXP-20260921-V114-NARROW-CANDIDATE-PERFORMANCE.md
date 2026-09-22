# V1.14 窄交互 Candidate 性能

- work_version：`V1.14`
- hypothesis：在相同 candidate-sharing 接口下，32D 相对 128D 可使 local interaction 至少加速 3x、
  完整 candidate forward 至少加速 2x、包含 endpoint 构建的顺序总时间至少加速 1.5x，且峰值显存不增加。
- protocol：见 [V1.14 FINAL plan](../plan/V1.14.md) 与
  [benchmark 定义](../../research/v114_narrow_candidate_benchmark/experiment.yaml)。
- runs：
  - `cmv2_v114_narrow_candidate_benchmark_20260921T2200CST`，commit `2d8373e`，默认 chunks `128/256`；
  - `cmv2_v114_narrow_candidate_chunk1024_benchmark_20260921T2204CST`，commit `5a73829`，tuned chunks `1024/1024`。
- conclusion：`REFUTED`。

## Evidence

默认 chunk 下，32D 几乎不改变延迟，只降低参数量和显存。算子 profile 显示降宽确实减少 GEMM 时间，但
小 object chunk 产生的大量 `cat`、逐元素、reduction 和 kernel launch 固定成本掩盖了收益。

tuned chunk 恢复了部分降宽收益：local interaction `1.91x`、candidate forward `1.41x`、
encode+candidate `1.28x`。但 endpoint union-rerank 仍需 `32.93 ms`，远高于 32D 模型的 `5.49 ms`；
因此含 endpoint 的顺序总时间只提升 `1.04x`。32D 的主要已证实价值是显存下降，而不是计划预期的整体延迟倍数。

chunk 本身是当前最有效且不改变研究语义的工程变量：32D 的 endpoint+model 估算从默认约 `73.53 ms`
降至 tuned 约 `38.42 ms`，约 `1.91x`。这不能归因于 32D 架构。

## Limitations

- 输入和模型权重均为 synthetic/random，不提供预测质量证据；
- 仅测单张 RTX 3090、FP32、固定形状；
- endpoint+model 是两个独立 CUDA-event 中位数之和，不包含更上游动作生成与环境仿真；
- 大 chunk 增加峰值显存，不能未经校准用于训练 batch；
- 尚未测试 fused edge head、低分辨率 hand proxy 或改变 endpoint 邻域定义。

按 V1.14 停止条件，本结果不支持直接进入短程或正式训练。后续若要继续追求总延迟，需要先协商是否把
研究问题转向 endpoint 构建或 fused edge 表达；前者可能改变邻域合同，后者需要新的受控架构补充。
