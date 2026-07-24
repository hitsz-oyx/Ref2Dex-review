# CmAction 框架

## 1. 问题定义

CmAction 的目标是在不访问未来物体几何的条件下，从当前的手—物交互状态和手部运动中提取一组紧凑的动作表征 `C_m`，并用它预测物体表面点在短时间后的位移。

对一个时间对 `(t, t+Δ)`，模型输入为当前物体 `O_t`、当前手 `H_t`、手部位移 `ΔH_t` 和 wrist 的相对位姿；监督目标为物体点流 `F_t^o`。因此学习问题可以写为：

```text
(O_t, H_t, ΔH_t, ΔT_wrist)  →  C_m  →  F̂_t^o
```

其中 `C_m` 不是单个全局向量，而是 `K` 个手部动作 slot。当前默认 `K=16`，每个 slot 的维度为 256。设计意图是让这些 slot 在手表面上形成可解释的软区域，并把局部接触、手指运动和整体 wrist 运动压缩为对物体运动有用的动作信息。

本阶段只研究 hand-side 的 `C_m`，不包含 object-side effect token `C_p`。

## 2. 数据表示与信息边界

Stage 4 直接由原始 GRAB 序列生成。每个样本对应一个当前帧和一个未来帧，所有几何均表示在当前时刻的 hand-root 坐标系中。若 `T_{world←hand,t}` 表示当前手根到世界坐标的刚体变换，则：

```text
x_t^h       = R_tᵀ (x_t^world - p_t)
x_{t+Δ}^{h} = R_tᵀ (x_{t+Δ}^world - p_t)
F_t^o       = x_{t+Δ}^{h} - x_t^h
ΔH_t        = H_{t+Δ}^{h} - H_t^{h}
```

这样，手流和物体流在同一参考系内，既保留手指关节运动，也保留 wrist 的整体运动。`wrist_delta` 另外保存 `T_{hand_t←hand_{t+Δ}}`，使模型可以显式使用 wrist 的 SE(3) 变化。

一个 Stage 4 文件保存一条序列、一个手侧。训练时主要使用：

| 字段 | 含义 |
| --- | --- |
| `obj_points`, `obj_normals` | 当前物体的 4096 个稳定表面点及法向 | 
| `hand_points`, `hand_normals` | 当前手的 1538 个 MANO face-center 点及法向 |
| `hand_flow`, `wrist_delta` | 当前到未来的手部运动信息 |
| `obj_flow_gt` | 物体点流监督 |
| `obj_candidate_mask_5cm` | 当前帧距手小于 5cm 的物体候选点 |

训练和模型前向**不读取未来物体点**。`obj_flow_gt` 只由数据生成脚本计算，并只在 loss、评估和可视化中使用。因而模型不能通过未来物体位姿直接恢复目标答案。

Stage 4 schema 1.1 还可选保存 `hand_root_pose_world` 与 `next_hand_root_pose_world`。它们只用于可视化中的 hand/world 坐标切换，不参与模型输入、监督或训练。

## 3. 冻结的当前交互编码器

CmAction 复用已经训练好的 `correspondence_ptv3_v2` dense-token 模型。该编码器以当前 `(O_t, H_t)` 为输入，输出：

```text
Z_t^o ∈ R^(N_o×D)     当前物体点 token
Z_t^h ∈ R^(N_h×D)     当前手点 token
c_t^h ∈ [0,1]^(N_h)  当前手点的冻结 contact 概率
```

其中运行时从 5cm 候选中稳定采样 `N_o=512` 个物体点，手点数为 `N_h=1538`。该编码器在 Cm 训练中保持 `eval` 和 `no_grad`：Cm checkpoint 只保存新增的 Slot Attention 与 flow decoder 参数，不重复保存 dense encoder 权重。

冻结的编码器提供当前接触状态与局部几何语义；时序预测能力则由后续 `C_m` 分支学习。

## 4. 从手部动作场到 C_m

对每个手点 `j`，先构造完整的局部动作描述：

```text
u_j = φ_h([z_j^h, y_j, n_j, Δy_j, c_j^h, vec(ΔT_wrist)])
```

这里 `y_j`、`n_j`、`Δy_j` 分别是手点位置、法向和手流；`c_j^h` 是冻结 contact 概率；`vec(ΔT_wrist)` 取相对变换前 3 行 4 列并复制到所有手点。拼接输入维度为 `D+22`，经 pointwise MLP 映射到 256 维动作特征 `u_j`。

随后 Slot Attention 在所有手点上执行 3 次迭代。对于 slot `k` 和手点 `j`，模型先得到归属概率：

```text
A_kj = softmax_k(q_kᵀ key(u_j))
```

`A` 在 slot 维度归一化，因此每一个手点的概率和为 1。为形成每个 slot 的聚合权重，再定义：

```text
W_kj = A_kj / Σ_j A_kj
```

`W` 在手点维度归一化。Slot Attention 使用 `W` 汇聚特征并更新 slot，最终得到：

```text
C_m = {c_1, …, c_K},   c_k ∈ R^256
```

同一组 `W` 还将 slot 投影回手表面：

```text
m_k       = Σ_j W_kj y_j                 soft anchor 位置
v_k       = Σ_j W_kj Δy_j                slot 平均手流
n_k       = Normalize(Σ_j W_kj n_j)      slot 平均法向
```

因此每个 slot 同时具有 token 表征、软空间锚点和局部运动摘要。这里没有 hard Top-K anchor、activity head 或 activity/diversity loss；slot 的分化完全由下游 object-flow 监督驱动。

## 5. Object-flow 解码器

物体分支先对每个当前物体点构造局部上下文：

```text
r_i = φ_o([z_i^o, x_i, n_i])
```

随后将每个物体点与每个动作 slot 配对。第 `i` 个物体点到第 `k` 个 slot 的输入包括：

```text
[r_i, c_k, x_i - m_k, v_k, n_k]
```

flow-edge MLP 为每条物体点—slot 边输出一个标量 logit `e_ik` 与一个候选位移 `f_ik`。在 slot 维度归一化后：

```text
α_ik = softmax_k(e_ik)
F̂_i^o = Σ_k α_ik f_ik
```

最后使用 object valid mask 将无候选的占位点置零。该结构保证 object token 只在手侧信息被压缩成 `C_m` 后才进入解码器；不存在从 object side 回流到 Slot Attention 的捷径。

## 6. 训练目标与评估

训练仅对 valid object point 使用 masked Smooth-L1：

```text
L_flow = (1 / Σ_i mask_i) Σ_i mask_i · SmoothL1(F̂_i^o, F_i^{o*})
```

优化器只更新 hand motion MLP、Slot Attention、object context MLP 和 flow-edge decoder。常规设置使用 AdamW、余弦学习率、3% warmup 和梯度裁剪；验证按 sequence 划分，而不是把同一条序列的不同时间对随机分到训练与验证中。

主评估量为 masked `flow_mse`、`flow_mae`、预测/GT flow norm。由于有效 object point 的数目及运动幅度随样本变化，这些标量应与可视化和分组分析一起解释，而不宜作为唯一结论。

## 7. 可解释性诊断

CmAction 不把低 loss 自动等同于获得了动态动作表征。训练过程额外记录以下不参与反传的诊断量：

| 指标 | 含义 |
| --- | --- |
| `slot_assignment_entropy` | 一个手点在 16 个 slot 间的平均归属熵；过高可能表示 slot 未分化 |
| `slot_weight_overlap` | 不同 `W_k` 的余弦重叠；过高可能表示多个 slot 聚合相同区域 |
| `decoder_slot_usage/slot_XX` | decoder 对每个 slot 的平均注意力 `q_k` |
| `decoder_slot_usage_entropy/max` | decoder 是否只依赖极少数 slot |

可视化中，`S` 显示每个手点的 `argmax_k A_kj` 与 soft anchor，用于观察同一 MANO 点是否随动作和接触发生 slot 重分配。`--assignment-report` 可以在整条序列上统计重分配比例与 decoder usage。

此外，trajectory rollout 通过把前一步预测的物体状态送入下一步模型来观察多步漂移。严格 rollout 要求前一 pair 的 future frame 正好是后一 pair 的 current frame，例如 `t→t+3→t+6`。当前正式训练数据的 `pair_stride=3, pair_hop=4` 不满足该条件，因此只能在 `hop=1` 的可链式 Stage 4 文件上做严格多步诊断；不能用 GT object state 填补缺失帧。

## 8. 当前框架的结论边界

当前架构能够检验两件事：第一，手部动作信息经 `K=16` 个 slot 压缩后能否支持短时物体点流预测；第二，这些 slot 是否在手表面形成稳定且被 decoder 实际使用的分组。

它尚不能仅凭 flow 指标证明 slot 就是高层动作 primitive。若分配长期固定在 MANO 的相同解剖区域，模型仍可能依靠 slot 内 feature 预测物体流。后续研究应结合跨动作的 assignment 变化、接触区域、decoder usage 和消融实验，区分动态 functional grouping 与固定 anatomical segmentation。

`C_p` 的后续设计应以当前 `C_m` 为输入，通过 correspondence 投影到 object side，再压缩为物体效应表征；不应让未来物体几何反向成为 `C_m` 的输入。
