# OakInk2 双手 Inspire 几何试转

按 [V1.4 plan §7](../../docs/plan/V1.4.md) 验证 NAS 的 OakInk2 标注能否重建 MANO 双手并运行左右 Inspire position retargeting。此处产物为诊断数据，不是 OI-Cm 训练 cache。

入口为 `run.py`，在 `graspenv` 下运行 `python run.py --help` 查看参数。命令、参数、环境版本、输入及模型 hash 会保存到新的 `output/<run_id>/`；默认从排序后的序列首、中、尾抽取三条，每条均匀选 64 个 mocap 帧。选取原始帧，不插值或重新定义时间轴。`--sequence` 可指定单条序列；`--frames 0` 可保留该序列全部 mocap 帧。

`python plot.py <run_dir>` 从 NPZ 的 link FK 和对应 URDF visual mesh 生成 `geometry_review.png`，叠加 MANO、物体及整段指尖误差。图中只选一个接近物体的帧作检查，完整转换不按该选择过滤。运行记录与终态见 [Task activity](../../docs/logs/activity_log.md)。

重建严格沿用官方 `oakink2_preview/launch/viz/seg_3d.py`：manotorch quaternion wxyz、`center_idx=0`、`flat_hand_mean=True`，重建后加 `lh/rh__tsl`。因此手腕世界位置应等于标注平移。保存官方 manotorch 的 21 点顺序和 tip 顶点约定，不能混用其他库的 tip 索引。

左右手和所有物体均保留原始 OakInk2 世界坐标（米）；优化器使用各自官方 Inspire offline 配置，objective/limits 不改，不做单侧坐标旋转。输出保存 native qpos 及关节名称、各 link 的 FK、MANO 顶点、逐对象 pose 和固定 canonical mesh 顶点采样，便于复核手物对齐。

报告区分重建一致性与 retarget 指尖拟合误差。成功生成有限 qpos 不等于无穿透、接触保持或训练效果成立。多物体如何形成 OI-Cm 样本、正式采样频率和数据划分不在本次试转中定义。
