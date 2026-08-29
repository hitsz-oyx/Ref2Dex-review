---
name: modular-component-runtime
description: 设计或修改任务无关的 Component、Artifact、Contract 和 Pipeline 系统，提供显式接口与安全组合。
metadata:
  short-description: 构建可组合的科研组件
---

# 模块化组件运行时

新增或修改可复用组件、manifest、Artifact 合同、Pipeline 组合、适配器或 registry 行为时使用本 Skill。

## 抽象边界

- `Component` 是可执行扩展点；数据适配器、模型、求解器、评估器和导出器都可以实现它。
- `Artifact` 表示值或持久化引用；`Contract` 描述类型、shape、dtype、schema、单位、坐标系和时间语义。
- 用 capabilities 和 tags 做发现。不要把 encoder、decoder、robot 或 dataset 等领域词汇做成框架一级分类。
- registry 的发现必须无副作用；只有显式的验证或执行命令才导入、实例化 entrypoint。

## 兼容性

- manifest 是组件的公共声明。保持已有 ID 和语义稳定；输入、输出、坐标系、GT、缓存 schema 或 checkpoint 解释改变时创建新版本。
- 一次性辅助函数或只改超参数的实验不要新建 Component；公共合同不变时优先新增配置变体。
- 首次迁移时用 adapter 包装已有 task runner/model，不要移动或重写它们。训练生命周期与推理组合可以保持分离。

## 安全

- 执行前校验合同，对不兼容的元数据或缺失输入快速失败。
- 执行必须显式给出配置、checkpoint、资源和输出位置。
- 发现和 dry-run 命令不得触发真实训练、数据加载、DDP 或外部副作用。
- 组件改动跨越科研语义或共享框架边界时，先使用 change-control Skill 并取得用户批准。
