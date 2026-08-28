# 通用组件目录

这里存放可被 registry 发现的 `component.yaml`。组件的实现可以继续位于
`src/task/<Task>/` 或其他 Python 包中，manifest 只声明稳定的公共接口，不要求
立刻搬迁现有代码。

最小组件需要声明：

- `id`、`version`、`entrypoint`；
- 一个或多个 `capabilities`；
- 输入/输出 `ports` 及其类型合同；
- `status`（`experimental`、`reference`、`active` 或 `deprecated`）。

查看和校验组件：

```bash
python tools/researchctl.py list components
python tools/researchctl.py check components/examples/identity/component.yaml components
```

`decoder`、`encoder`、`MANO` 等词只能作为 tags 或领域约束，不是框架的固定
一级分类。

当前只读接入的 Ref2Dex manifest：

- `ref2dex.correspondence.ptv3_v2`
- `ref2dex.cm.v1`
- `ref2dex.cmdecoder.pointflow.inspire_f1`

这些 manifest 只建立 registry 索引和接口说明，不会改变原有 Task 的训练入口。
