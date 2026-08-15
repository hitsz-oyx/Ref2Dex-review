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

## 实验：真实 cache 路径核查与单样本 forward

**观察**
仓库已有真实 GRAB cache，并非缺失。PointWorldWAM overfit 配置使用 `data/processed_data/interaction_dynamics_v1/data/grab`；Cm/Stage4 使用 `data/processed_data/stage4/data/grab`。

**结果**
前者包含 20 个序列；InteractionTransfer 在 `s1/airplane_fly_1` 上按 V0.2 筛选得到有效 transition，读取为 `512 object / 1538 hand`。`graspenv` CUDA forward 输出 shape 为 `(1,512,3)`，全为有限值，有效 edge 数 `127`。

**决策**
保留。此前“待提供 cache 路径”的结论撤回；下一步可以直接进行真实 cache 的短训和 direct baseline EPE 对比。

## 实验：V0.3 DenseToken fidelity 与公平 transition 对齐

**假设**
补齐 DenseToken 的 opposite-cloud centroid projection，并复用 PointWorld one-step 的数据校验与运动排序后，InteractionTransfer 的输入特征和 transition 定义才完全一致；模型应继续满足 zero-preserving，并能在真实 cache 上下降训练 loss。

**改动**
在冻结 DenseToken 编码器中计算 opposite cloud centroid-direction·normal，替换原先恒为零的第三个几何特征。数据集加入 `source_fps=120`、`raw_frame_id` 间隔为 4 的校验，按 object motion 降序排序，并复用 PointWorld 的固定点索引。

**结果**
`graspenv` + RTX 3090 forward 通过：输出 `(1,512,3)`，有效 edge `5868`，`zero-message=0.0`。GT/reverse/shuffle message norm 为 `0.0890855/0.0903455/0.0892916`，对应 effect delta 为 `2.8514e-4/2.8261e-4/2.7422e-4`。真实 `s1/airplane_fly_1` cache 单序列 10 步 CUDA 短训 loss 从 `0.0561577` 降至 `0.0168040`，无 NaN/Inf。

**决策**
保留 V0.3 修改；实现 gate 和短训 gate 均通过。完整多序列训练及 direct baseline EPE 留待下一实验。

## 实验：V0.4 正式结构与数据入口固化

**假设**
将 V0 forward 的信息约束正式固化后，当前模型应保持 `ΔH=0 => C_m=0`，且 effect decoder 只从 object points、normals 和聚合后的 object-side field 预测 flow。

**改动**
InteractionTransfer dataset 内置确定性 `fixed_point_indices`，移除对 PointWorldWAM 的运行时依赖；V0.4 固定 one-step `gap=1`，不接受其他 gap。指导文档同步纳入版本管理。

**结果**
`graspenv` + RTX 3090 forward 通过：输出 `(1,512,3)`，有效 edge `5868`，`zero-message=0.0`。真实 `s1/airplane_fly_1` cache CUDA 短训 5 步 loss 从 `0.0485412` 降至 `0.0265517`，无 NaN/Inf。

**决策**
保留 V0.4 固化；正式结构 gate 和短训 gate 通过。完整多序列 EPE 与 direct baseline 对比仍属于后续实验。
