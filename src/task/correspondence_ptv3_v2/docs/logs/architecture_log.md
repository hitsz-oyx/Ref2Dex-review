# correspondence_ptv3_v2 architecture log

## 当前架构入口

详细 pipeline 保留在任务文档 [`../架构.md`](../架构.md)。本日志记录该架构最近的跨数据集约定：

1. Stage 3 输出提供 clean full object pool 和 clean hand geometry。
2. Dataset 可对 full pool 应用共享 object SE(3)，将整个 pool 交给 runner。
3. Runner 用 256 个 hand proxy 评分 full pool，取 near 384 + global 128 个 object points。

## OakInk 无 MANO runtime 路径

OakInk 样本没有 MANO 参数，所以 `apply_hand_perturb=false` 时不进行 MANO forward。如果仍要 runtime resampling，必须提供 `meta.stored_hand_proxy_indices_path`；该 JSON 版本化一组直接索引存储 hand points 的空间 FPS proxy。Runner 会校验索引数量、范围和唯一性，缺失时 fail-fast。
