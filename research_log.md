# 研究日志

## 实验：InteractionDynamics V1 数据链路

**假设**
现有 GRAB raw adapter 已包含构造固定 8 步 chunk 所需的全部几何和根位姿，不需要依赖 Stage4 pair 文件。

**观察到的失败 / 现象**
旧 Cm cache 没有写入 object root pose，并且只在当前 hand frame 中构造 endpoint motion。

**诊断**
V1 需要独立的 sequence schema，并在运行时分别构造 current-object frame 和 current-hand frame 表示。

**改动**
新增 V1 配置、raw cache builder、固定 horizon=8 的 dominant-hand wrapper 和 runtime chunk Dataset；cache 新增 `obj_root_pose_world`。

**结果**
静态编译和配置加载通过。合成轨迹验证了 8 步累计手/物位移、稳定 object index、tensor shape、坐标往返变换以及 normal 不受平移影响。`graspenv` 中成功重建并缓存真实 GRAB 序列。

**决策**
保留。

## 实验：V1 模型结构与梯度链路

**假设**
以 object patch 为索引的 cross-attention 能建立 action → interaction → effect 路径，并保留 World、Action 和 Effect 各模块的梯度。

**改动**
实现 deterministic FPS/KNN、联合 World Transformer、复用 hand patch index 的 Action Encoder、Action-World fusion、无 object residual 的 canonicalization、temporal effect query、厘米内部尺度 masked MSE、轨迹指标和 collapse diagnostics。

**结果**
Uni3D-S DeepSpeed checkpoint 成功加载 22,105,474 个参数，shape mismatch 为 0；新增的 150,144 个参数仅为 normal/entity embedding。真实 FrozenDenseToken + Uni3D-S 的全尺寸 batch 完成 forward/backward，输出为 `[1,128,384]`、`[1,64,384]`、`[1,64,384]`、`[1,8,512,3]`，峰值显存 845.7 MiB。World、Action、canonicalization 和 Effect 均有梯度，DenseToken 保持冻结。

**决策**
保留。

## 实验：真实数据单步训练闭环

**假设**
真实 cache、FrozenDenseToken、Uni3D-S、AMP loss 和双学习率 optimizer 能通过共享 Runner 完成训练。

**观察到的失败 / 现象**
首次 bfloat16 训练在 frozen spconv kernel 中失败，原因是 autocast 把 PTv3 输入改成了不受支持的 dtype。

**诊断**
FrozenDenseToken 的 checkpoint contract 是 FP32，不应继承新 Runner 的 bfloat16 autocast。

**改动**
仅在 FrozenDenseToken 周围禁用 autocast，其余可训练 V1 模块继续使用 bfloat16。

**结果**
真实 optimizer step 成功，耗时约 2 秒，loss 0.006959，Object ADE 1.444 mm。

**决策**
保留 FP32 frozen-adapter 边界。

## 实验：原始 V1 的 32-chunk overfit

**假设**
仅使用 effect supervision，V1 bottleneck 可以记住一个小型 contact-active sequence 子集。

**改动**
固定 `toruslarge_lift` 的 32 个 chunk，以 batch size 8 训练 200 steps，不增加 loss 或输入。

**结果**
Zero-flow ADE/FDE 为 42.53/76.98 mm；step-200 为 21.40/35.46 mm，改善 49.7%/53.9%，但没有充分 overfit。Interaction token variance 为 7.09e-6，pairwise cosine 约为 1。逐层诊断显示 world-object variance/cosine 为 0.399/0.713，action-context 为 0.0112/0.9952，interaction 为 4.27e-6/0.999996。Canonical attention entropy 为 4.15866，接近 64-way 均匀分布上限 4.15888。

**诊断**
主要 collapse 发生在 residual-free object canonicalization，而不是 pretrained world-object tokens。

**决策**
结论不充分：effect 优于 zero flow，但 representation-collapse 条件失败。

## 实验：Canonical query scale=10

**假设**
Canonicalization collapse 是 attention logits 长期处在近均匀 softmax 区间导致的。

**改动**
仅将 canonicalizer 的 normalized query 放大 10 倍，重复相同的 200-step overfit。

**结果**
Interaction variance 从 7.09e-6 增加到 2.55e-3，但 ADE 21.40 → 21.38 mm，FDE 35.46 → 34.58 mm，基本没有实际收益。

**决策**
撤回。Sharper attention 能改变 token diversity，但不能解释 overfit plateau。

## 实验：Effect Decoder 的 token 使用干预

**假设**
Effect Decoder 可能只使用 sample-level mean interaction，而忽略 token-level spatial variation。

**改动**
在 baseline step-200 checkpoint 上分别将 64 个 interaction tokens 替换成样本内均值、跨样本交换或全部清零，不重新训练。

**结果**
相对原预测的 RMSE 变化分别为：均值替换 0.0048 mm、跨样本交换 0.260 mm、清零 14.56 mm。

**诊断**
Decoder 强依赖 interaction feature，但几乎只使用其 sample-level mean，没有利用 object-patch-indexed 结构。

**决策**
原始 V1 虽然优于 zero flow，但没有达到 object-indexed representation 目标。

## 实验：Effect spatial routing bias

**假设**
用空间 bias 将 effect query 显式路由到附近 object patch，可以迫使 Decoder 使用局部 interaction。

**改动**
加入 sigma=5 cm、仅影响 attention routing 的 Gaussian bias；没有把 object geometry 加入 interaction value 或 flow head。

**结果**
ADE 21.40 → 21.39 mm，FDE 35.46 → 41.40 mm，interaction variance/cosine 仍为约 8.52e-6/1。

**诊断**
上游 values 已经几乎相同，空间路由无法恢复局部 interaction。

**决策**
撤回。

## 实验：Action-context local gain

**假设**
Canonicalization 前，全局 action component 在数值上淹没了 patch-local residual。

**改动**
将 action context 分解为 mean 和 local residual，使用：

```text
action_context = mean + 10 × local_residual
```

没有增加输入、辅助 loss 或 object residual。

**结果**
相同 32 个 chunk 训练 400 steps 后，epoch ADE/FDE 达到 6.71/12.43 mm，相比 zero flow 改善 84.2%/83.9%；最后一个 batch 为 5.39/7.74 mm。最终 interaction variance/cosine 为 0.204/0.793，不再存在严重 token collapse。

**决策**
保留 `action_local_gain=10` 作为 V1 默认值。单序列 controlled overfit 验收通过。

**下一步**
建立 sequence-disjoint 80/10/10 split，并在 full GRAB 训练前运行小规模多序列实验。
