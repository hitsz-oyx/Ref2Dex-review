# Inspire surface FK 修复 pilot

本实验验证 V1.4.6 对 bilateral Inspire surface 双重 FK 的修复。它只导出一条 GRAB 和一条 ARCTIC
序列到独立 `output/<run_id>/`，生成可视化 index、几何诊断报告和 run manifest；不会覆盖 V1.4 v2
全量 cache，也不会自动启动全量重导、KNN 或训练。

默认样例：

- GRAB：`s1/airplane_fly_1`；
- ARCTIC：`s01/box_use_01`。

运行入口为 `run.py`。pilot 通过后，使用 `src.task.ObjectInteractionCm.visualize_grab` 读取输出的
`index.json`，在轨迹下拉框中切换两个数据源。

启动查看器时显式传入 `--modification-version V1.4.6`，并为 `--output` 使用尚不存在的独立运行目录。
`--future-delta 10` 表示叠加 10 个 cache 帧后的点云；30 Hz 下为 0.333 秒。界面中调为 0 可先检查
单时刻手形，调为 1–10 可比较跨帧跨度。灰/橙分别为当前物体/双手，绿/紫为未来物体/双手。

pilot 报告的 bbox、连通分量和法向检查只验证双重 FK 散裂修复；它不证明重定向接触保真、
左右手资产正确性或训练 stride 合理。当前沿用 exporter 已有的左手使用右手 surface 资产回退，
且 cache 未保存 viewer 所需的 qpos，因此本轮显示点云；这些边界需与散裂修复分开审阅。
