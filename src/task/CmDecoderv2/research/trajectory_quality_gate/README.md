# 轨迹质量标记与跨手候选交叉检查

按 [V1.1.12计划](../../docs/plan/v1.1.md) 实现。`gates.quality_flags` 输入K4每步效果EPE/RMS、K+1中心/旋转误差，分别返回effect、pose与combined标记。
effect保持1mm/25%阈值；pose20为20mm/15deg，pose40、pose80是固定敏感性档位。后三者是工程诊断值，未经物理成功率验证。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.trajectory_quality_gate.run --run-id <unique_id>
```

输出 `window_metrics.jsonl`、`repair_queue.csv`、`candidate_manifest.jsonl`、配置/来源摘要与coverage图；reason bit 1/2/4分别表示effect/position/rotation失败。
候选清单含原始MANO和目标Inspire的parent、cache/raw起点，`diagnostic_only=true`且split=val，不是正式训练index。
参考跟踪差与实际跨手不兼容是不同命题。保留反例并重算各条件下双流不重叠贪心配对，不把跟踪标记自动应用于正式样本。
