# ObjectInteractionCm 文档入口

- [`指导/V1.1.md`](指导/V1.1.md)：用户维护的 V1.1 研究指导与讨论原文；
- [`plan/V1.1.md`](plan/V1.1.md)：V1.1 执行计划（final）；
- [`architecture/V1.1.md`](architecture/V1.1.md)：V1.1 架构快照；
- [`plan/V1.2.5.md`](plan/V1.2.5.md)：修正 DExplore 实际物体轨迹、KNN=16 的数据重建与 Cm 重训计划（final）；
- [`指导/V1.3.md`](指导/V1.3.md)：V1.3 全量离线 KNN cache 的用户指导（final）；
- [`plan/V1.3.md`](plan/V1.3.md)：V1.3 全量离线 KNN cache、loader/model 合同与验证计划（final）；
- [`plan/V1.2.6.md`](plan/V1.2.6.md)：MANO/Inspire 2048 点 Region 加权采样 PLY 预览计划（final）；
- [`research/hand_region_sampling/README.md`](../research/hand_region_sampling/README.md)：Region 加权采样预览实验；
- [`research/oakink2_segment_visualizer/README.md`](../research/oakink2_segment_visualizer/README.md)：OakInk2 active-tool v1.1 原生双手/物体切分查看器；
- [`research/oakink2_temporal_segmentation/README.md`](../research/oakink2_temporal_segmentation/README.md)：V1.4.20 运动优先、2 cm 双侧边界搜索 pilot；
- [`research/grab_stride_visualization/README.md`](../research/grab_stride_visualization/README.md)：当前 V1.4 GRAB 双手 cache 的 1–10 stride 跨帧点流查看器；
- [`research/inspire_fk_repair/README.md`](../research/inspire_fk_repair/README.md)：V1.4.6 Inspire surface 双重 FK 修复、双数据集 pilot 与人工复核入口；
- [`logs/activity_log.md`](logs/activity_log.md)：Task 活动时间线（产生实际实现、诊断或运行事件后维护）；
- [`logs/experiment_log.md`](logs/experiment_log.md)：正式训练、评估结果与科研结论。
- [`activities/README.md`](activities/README.md)：Task 活动记录索引。
- [`logs/repo_memory.md`](logs/repo_memory.md)：可迁移的 Task 事实与用户固定工具偏好。
- [`research/paired_grab_visualization/README.md`](../research/paired_grab_visualization/README.md)：同序列 GRAB MANO / Inspire 只读配对可视化。
- [`research/geometric_q_surface_preview/README.md`](../research/geometric_q_surface_preview/README.md)：复用 DExplore geometric native q 的右手 MANO 2048 / Inspire 10135 点配对预览。

本 Task 与旧 `Cm` 独立；数据/cache、checkpoint 和运行产物仍按仓库目录规范放置，不复制到本目录。
