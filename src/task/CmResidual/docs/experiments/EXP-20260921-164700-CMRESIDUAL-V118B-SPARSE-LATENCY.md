# V1.18b sparse local-interaction latency

- experiment_id: EXP-20260921-164700-CMRESIDUAL-V118B-SPARSE-LATENCY
- timestamp: 2026-09-21T16:47:00+08:00
- work_version: V1.18b
- git_commit: `90b61ca705200c569ea0ffc8711997d13f8a77b2`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（tested implementations provide reproducible speedup）

## Hypothesis and protocol

假设：保持 2 cm 内有效 top-32 edge 与 frozen-Cmv2 输出等价时，merge-topk、conservative swept-link AABB
broad phase 和 radius-valid sparse edge MLP 能在 V1.18a 的同一 K=8 active state 上提供可重复延迟收益。
两个独立 GPU6 runs 固定 state/candidate SHA，每个 variant 20 warm-up + 100 CUDA Events，并用循环移位交错顺序
消除固定测量位置 confound。

## Conclusion and limits

工程 parity 得到支持：三个变体在两个 runs 中的四个 rollout 行为字段误差均为 0。但性能假设被本实现反驳：
merge-topk 两次总延迟均回退约 26%；boolean-index/scatter sparse edge 两次均回退；link-AABB 只在第一次改善
13.33%，第二次回退 0.69%，且 peak allocated 从 162.21 MiB 增至 256.64 MiB。因此没有变体满足“可重复收益”，
生产路径保持 legacy。

结论只覆盖当前 PyTorch eager 实现、单个 pinned active state 和 GPU6。它不反驳 fused CUDA/Triton top-k、
固定容量的无动态 `nonzero` broad phase、预构建 BVH/voxel grid，或在多 state 分布上的收益；也不能外推到训练
质量、整轮 FPS 或其它 GPU。下一步若继续，应避免 Python/dynamic ragged tensor 路线，先评估 fused kernel 或
重新设计低维 interaction representation。
