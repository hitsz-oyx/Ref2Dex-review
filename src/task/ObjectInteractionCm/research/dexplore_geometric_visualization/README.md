# Dexplore 既有 geometric 轨迹查看

复用仓库外 `dexplore/data_processing/visualize_inspire_trajectory.py`，只读显示已导出的
Inspire native 18D 轨迹和对应物体网格，不运行 converter、RL、数据重导或训练。

每次运行的具体输入、启动命令、端口和运行日志保存在独立 `output/<run_id>/`。
以 producer 的 manifest 和 geometric 目录区分数据来源；文件名本身不能区分 geometric/RL。
若没有同源 InterAct `human.npz`，不启用原查看器的 MANO 叠加，也不冒用其他坐标系的点云。

异机归档按 DExplore 规则由 1335 条筛为 660 条；当前机器没有该归档，只有同一 converter 生成、
manifest 明确记录为 656 条的本机集合。运行和汇报必须保留这个数量差异，不能将本机集合称为 660。
页面直接渲染由 `373:391` 驱动的完整 URDF visual mesh，并以 `198:205` 更新 geometric reference
object；它不读取或生成 10135 点 surface cache。
