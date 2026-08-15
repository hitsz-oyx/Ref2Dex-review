# InteractionTransfer 研究日志

## 实验：V0 最小 forward gate

**假设**
当前静态 hand-object relation 调制 hand action，可以产生严格 zero-preserving 的动态 interaction message，并由 object-only decoder 输出 object flow。

**改动**
新建独立 `InteractionTransfer` 包：edge builder、冻结静态编码器、relation encoder、无 bias action/message path、attention aggregator、object-only effect decoder 和 one-step dataset reader。

**结果**
合成张量 forward 通过 shape/finite-value 检查；零 hand flow 时 edge message 最大绝对值为 0。真实 cache 训练尚未启动，故不报告 EPE。

**决策**
保留 V0 实现，下一步再以真实 GRAB cache 与 direct forward baseline 比较。

## 实验：V0.1 action intervention

**假设**
固定当前 `(O,H)` 后，GT、反向和 shuffle action 应产生非零且不同的动态 message；zero action 必须严格为零。

**改动**
接入本地 vendored DenseToken/PTv3 checkpoint loader（默认使用 `src/task/Cm/densetoken_ckpt/best.pt`，不依赖其他 Task 运行时 import）；修正 object projection 为无 bias，新增 GT/zero/reverse/shuffle intervention 诊断。诊断脚本自动优先使用 CUDA；无 CUDA 时才回退 synthetic encoder。

**结果**
synthetic intervention：`message_norm(gt)=0.0109645`、`zero=0`、`reverse=0.0107316`、`shuffle=0.0099526`；四种 effect 均可计算，GT 相对 zero 的 effect 差异为 `1.0365e-4`。

**决策**
保留。zero-preserving 和 action sensitivity gate 通过；已在 `graspenv` 的 RTX 3090 上用真实 DenseToken/PTv3 完成 `(1,512,1538)` forward，输出 shape 为 `(1,512,3)`，zero-message 最大绝对值为 `0.0`。

## 实验：V0.2 数据与 DenseToken fidelity 修正

**假设**
只有在 DenseToken 输入特征、transition 筛选和点数与既有 one-step baseline 对齐后，InteractionTransfer 的 forward/intervention 结果才具备可比较性。

**改动**
补齐 opposite-cloud centroid projection 特征；将 pretrained edge feature 与 contact probability 接入 relation encoder；dataset 改为 512 object / 1538 hand，并筛选 right-active、left-inactive 的 one-step transition；修复 shuffle 为 hand-point 维度置乱；训练脚本自动选择 CUDA 并支持 `--device`。

**结果**
`graspenv` + CUDA 诊断通过：object flow shape `(1,512,3)`，valid edges `5868`，zero-message `0.0`。GT/reverse/shuffle message norm 分别为 `0.0891573/0.0904426/0.0893798`；相对 zero 的 effect delta 分别为 `2.8672e-4/2.8384e-4/2.7575e-4`。

**决策**
保留。数据控制变量和 DenseToken relation prior 已接入；真实 GRAB EPE 对比待提供 cache 路径后运行。
