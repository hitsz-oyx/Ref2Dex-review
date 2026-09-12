# 物体轨迹来源与跟踪审计

对应 [V1.1.11最终计划](../../docs/plan/v1.1.md)。只读比较MANO parent、dexplore几何参考和RL实际导出。
native四元数约定用于参考/实际pose跟踪；旧cache转置约定仅用于与前两轮一致的逐点effect比较。两种约定显式分开，不修正GT。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m src.task.CmDecoderv2.research.object_tracking_audit.run --run-id <unique_id> --smoke
```

两序列smoke后去掉--smoke。lag扫描固定共同支持，按时间前半选择、后半评估；负lag表示实际滞后参考。均值偏移残差只作解释，不是新的对齐GT。
输出包含三段逐步误差、native跟踪误差、lag矩阵、输入/外部源码摘要及日志。分段EPE不具有可加性；结果不代表物理任务成功率。
