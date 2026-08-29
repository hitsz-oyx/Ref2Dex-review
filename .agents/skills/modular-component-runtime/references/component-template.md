# Component 模板

本模板只用于需要被 registry、Pipeline 或其他 Task 外部发现的组件。单一实现、只在一个文件内使用的普通模块不需要 manifest。

## `component.yaml`

```yaml
api_version: component/v1

id: <domain.capability>
version: 1.0.0
status: experimental  # experimental / active / deprecated / archived

entrypoint: <python.module>:<ClassName>

capabilities:
  - execute  # execute / fit / predict / solve / evaluate / export / visualize

tags: []

inputs:
  <name>:
    type: tensor
    shape: [B, ...]
    dtype: float32
    unit: null
    coordinate: null
    optional: false

outputs:
  <name>:
    type: tensor
    shape: [B, ...]
    dtype: float32
    unit: null
    coordinate: null
    optional: false
```

## Python 接口

```python
from src.base import Component


class ExampleComponent(Component):
    """一个具有显式输入输出合同的可执行组件。"""

    def execute(self, inputs, context=None):
        self.validate_inputs(inputs)
        outputs = self._run(inputs, context=context)
        self.validate_outputs(outputs)
        return outputs
```

## 版本与兼容

- 只改内部实现且输入输出合同不变：保留同一 `id/version`，或按项目规则递增 patch 版本。
- 新增可选输入、输出或能力：使用兼容的 minor 版本，并保留旧调用路径。
- 改变 shape、dtype、单位、坐标系、GT、缓存 schema 或 checkpoint 解释：创建新 major 版本，不要静默重定向旧配置。
- 旧版本进入 `deprecated` 或 `archived` 后，历史实验通过锁定的 Git commit/tag 复现；主分支不永久携带所有历史实现。

## 内部多个实现

多个紧密相关实现可以放在同一 Python 文件中，由显式 `mode` 或 strategy 选择。只有当实现需要独立发现、版本、资源、生命周期或跨任务复用时，才拆成多个外部 Component。

```yaml
mode: aggregate  # aggregate / additive / mixture
```

避免用大量相互耦合的布尔开关表达互斥模式。
