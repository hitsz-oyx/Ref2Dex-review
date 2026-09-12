# 跨时间配对覆盖

按 [V1.1.10 最终计划](../../docs/plan/v1.1.md) 检查 V1.1.9 的4254个 val active-only窗口。
仅放开同 parent 内源起始时间，保留原始 MANO 轨迹、K4、30 Hz、每步逐点效果差与接触门槛；不加载 decoder、不训练、不写正式 cache。

```bash
CUDA_VISIBLE_DEVICES=3 OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.cross_hand_pair_coverage.run --run-id <unique_run_id> --smoke
```

smoke 后去掉 --smoke 运行全部30 parent。输出位于忽略的 output/run_id：所有严格候选边、固定排序选择、逐parent指标、几何描述、配置/来源、manifest、summary和图。
几何描述保留原始/实际 pose、canonical点和压缩接触mask，用于独立检查，不是新的训练缓存。

选择最小平均归一化效果差；平局优先时间接近再起点较小。窗口数、唯一供体数、一对一最大窗口匹配容量及两流 K+1帧均不重叠的贪心配对数分别报告。
贪心数是保守下界，不是最优解或独立物理试验数。输出只描述当前val候选库可用性，不能证明迁移，也不能转为训练样本。
