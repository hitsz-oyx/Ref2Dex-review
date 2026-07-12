# =============================================================================
# correspondence_ptv3 包初始化模块
# =============================================================================
# 本文件是 `src.task.correspondence_ptv3` 包的初始化文件。
# 该包基于 PointTransformerV3 (PTv3) 序列化注意力机制，构建了一个
# 面向"静态手-物体对应关系 (Static Hand-Object Correspondence)"的
# 训练/评估任务。
#
# 主要功能：
#   1. 将 CorrespondencePTV3Runner 暴露到包级别，方便外部直接通过
#      `src.task.correspondence_ptv3.CorrespondencePTV3Runner` 导入使用；
#   2. 通过 __all__ 显式声明对外公开的符号，便于 from-import 风格的引用；
#   3. 保持包结构与 `correspondence_v1` 等其他对应关系任务一致。
#
# 注意：具体的 Runner 实现见 runner.py，配置见 config.py，
#      数据加载见 dataset.py，模型定义见 model.py。
# =============================================================================

# 导入本任务对应的训练/评估运行器（Runner）。
# Runner 是整个训练/评估流程的核心入口：负责构建数据加载器、
# 搭建模型、计算损失、记录指标、保存检查点等所有流程性工作。
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner

# 显式声明本包对外导出的符号列表。
# 当用户执行 `from src.task.correspondence_ptv3 import *` 时，
# 只会导入 __all__ 中列出的名字，这是 Python 包设计的最佳实践。
__all__ = ["CorrespondencePTV3Runner"]
