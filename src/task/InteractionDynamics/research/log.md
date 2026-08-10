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

## 实验：10 序列小规模泛化验证

**假设**
通过单序列 overfit 的表示能够在少量多序列数据上学习，并在未见序列上优于 zero-flow，同时不发生严重 token collapse。

**观察到的失败 / 现象**
原配置的最佳指标名写成 `val/object_ade_mm`，而 Runner 实际产出 `val/object/ade_mm`；该问题只在存在验证集时触发。全局 `max_samples` 还会按路径排序截断，使小规模训练偏向靠前序列。

**诊断**
指标键是配置拼写错误；多序列采样需要按 sequence directory 单独限制，而不能对排序后的全体样本直接截断。

**改动**
缓存 10 条 GRAB 序列，生成 3412 个 dominant-hand chunk，以序列隔离方式划分 8/1/1；每条序列最多取 128 个 chunk。修正最佳指标名，并增加 `max_samples_per_sequence`。使用 batch size 8、AdamW、5 epochs，共训练 640 steps。

**结果**
训练 ADE：35.53 → 19.54 mm。验证 zero-flow ADE/FDE 为 42.94/66.76 mm，模型为 32.07/46.03 mm，分别改善 25.3%/31.1%。测试 zero-flow 为 30.98/54.62 mm，模型为 25.19/44.81 mm，分别改善 18.7%/18.0%。验证 interaction variance/cosine 为 0.1108/0.8874；测试为 0.0367/0.9627。

**决策**
保留。小规模 sequence-disjoint 泛化成立，模型在两个未见序列上均优于 zero-flow；测试序列 token 多样性弱于验证序列，需要在扩大数据前继续观察，而不是据此增加结构。

**下一步**
先实现 representation/attention 导出并检查不同序列的 interaction token 与注意力，再决定是否扩大到完整 GRAB。

## 实验：未见序列的表征与注意力导出

**假设**
逐样本导出 interaction tokens 和三类 attention，可以确认测试序列较高的 token cosine 是否同时伴随更均匀的 object-to-action attention。

**改动**
实现 `InteractionDynamics/extract.py`，按 `seq_id/side/raw_frame_id.npz` 独立保存 V1 规定的 patch center、action/interaction tokens、预测与真值 object effect，以及三类 attention。对小规模实验的验证集和测试集各导出前 8 个样本。

**结果**
全部字段和 shape 与 V1 schema 一致；单样本压缩文件约 5.6 MiB。验证样本 interaction variance/cosine/attention entropy 为 0.0645/0.9345/4.0201；测试样本为 0.0231/0.9765/4.0819。64-way 均匀 attention 的熵上限为 4.1589。

**诊断**
测试序列的 interaction tokens 确实更相似，并伴随更接近均匀分布的 object-to-action attention。当前只有单个测试 sequence 的 8 个相邻片段，证据不足以归因于模型结构，也可能来自动作或接触模式本身。

**决策**
保留导出工具；不增加 regularization 或新结构。

**下一步**
扩充 sequence-disjoint 验证/测试序列后重复同一诊断，区分序列特性与系统性 attention bottleneck。

## 实验：small10 训练强度与均匀时间采样

**假设**
原 small10 只训练 5 epochs 且每序列取最前 128 个片段，训练不足和时间覆盖偏差共同限制了泛化结论。

**改动**
先保持旧采样从 5 epochs 续训到总计 20 epochs；再将 `max_samples_per_sequence` 改为在全部合法片段上均匀取样，从头训练相同 20 epochs。

**结果**
旧采样第 20 轮 train ADE 为 10.87 mm，最佳验证 ADE 为 30.78 mm，最佳 checkpoint 测试 ADE 为 29.31 mm，说明继续训练形成明确 train/val gap。均匀采样把 `toruslarge_lift` 覆盖从前段扩展到 current frame 37–847；最佳验证 ADE 进一步降至 27.62 mm。均匀测试集 zero-flow ADE/FDE 为 45.99/79.83 mm，模型为 46.01/81.19 mm，表明旧测试前段采样明显低估了后段难度。

**决策**
保留均匀时间采样，并将 small10 默认训练改为 20 epochs。原 5-epoch、前 128 帧结果不能作为稳定泛化结论。

## 实验：post-gain interaction token 干预

**假设**
即使 `action_local_gain=10` 提高了 token 方差，Effect Decoder 仍可能只使用样本级全局 interaction。

**改动**
在均匀采样 small10 最佳 checkpoint 上比较 normal、mean-token replacement、token shuffle 和 cross-sample replacement。

**结果**
验证 ADE 分别为 27.62/27.34/27.62/30.03 mm；测试为 46.01/46.48/46.01/47.17 mm。mean-token 几乎不损失性能，shuffle 与 normal 数值相同，cross-sample 只产生有限退化。

**诊断**
Decoder 使用样本级 interaction 信息，但没有建立有效的 object-indexed token-level dependency。仅观察 variance/cosine 不足以证明局部表征被使用。

**决策**
保留 `intervention.py` 作为后续结构验收工具；当前 object-flow supervision 下的 global shortcut 得到直接确认。

## 实验：small20 的 12/4/4 序列泛化

**假设**
增加验证和测试序列可以区分单条 sequence 特性与系统性泛化问题。

**改动**
扩展到 20 条 GRAB 序列，使用 sequence-disjoint 12/4/4 划分和均匀时间采样；训练集 1536、验证集 362、测试集 512 个片段，训练 20 epochs。

**结果**
train ADE 收敛到 14.38 mm。zero-flow 验证/测试 ADE 为 49.05/45.82 mm；最佳 checkpoint 为 46.93/44.80 mm，仅改善 4.3%/2.2%。最佳 checkpoint 的验证/测试 interaction cosine 为 0.9965/0.9987，接近完全 collapse；继续训练后验证 ADE 恶化至 52.61 mm。

**决策**
结论明确：训练可以收敛，但现有结构对未见 sequence 的提升很弱，并存在系统性 global-token shortcut。下一步应修改 canonicalization 或 supervision，而不是继续单纯增加 epoch。

## 实验：Uni3D 联合单位球输入

**假设**
pretrained Uni3D 的单位球坐标分布与当前米制输入不一致，可能导致多序列泛化弱。

**改动**
仅对送入 Uni3D backbone 的 hand+object 联合副本使用共享中心和尺度归一化，其余 action、contact 和 effect 几何保持米制；在 small20 上训练 5 epochs。

**结果**
单位球版本最佳验证 ADE 为 49.47 mm；原始米制版本相同早期阶段最佳为 46.93 mm。interaction cosine 仍约 0.99。

**决策**
撤回。坐标分布 shift 不是当前主要瓶颈，下一实验转向 canonicalization / action-preserve supervision。

## 实验：V3 hand/object patch 直接监督

**假设**
参考 Fast-WAM 的逐 token 直接目标，在 `action_context_tokens` 上重建 hand patch motion、在 `interaction_tokens` 上重建 object patch effect，可以约束中间表征并让 Effect Decoder 使用 object-indexed 局部信息。

**观察到的失败 / 现象**
small20 仅使用 dense effect loss 时，最佳验证/测试 improvement 很弱，interaction cosine 为 0.9965/0.9987，mean 与 shuffle intervention 几乎不改变预测。

**改动**
增加两个线性 head，分别预测 `[8,64,3]` 的 hand/object patch 平均位移；Runner 使用 Uni3D 的稳定 KNN index pooling GT，三项厘米制 MSE 权重均为 1。完整 future object displacement 只作为 Runner target，不进入模型 forward。

**结果**
真实 batch 的初始 effect/action/patch-effect loss 为 19.02/20.60/19.65，量级相当且一步更新成功。small20 训练 5 epochs 后最佳 checkpoint 位于 step 384，验证 ADE 为 48.82 mm，弱于原 baseline 的 46.93 mm；验证 interaction cosine 仍接近 1。最佳 checkpoint 的 normal/mean/shuffle/cross-sample ADE 为 48.82/48.81/48.82/49.09 mm。

**诊断**
两个辅助 head 能优化，但它们是主 Effect Decoder 之外的旁路；模型仍可让 patch head拟合以全局刚体运动为主的平均 target，同时让 dense decoder 继续忽略局部 token。直接监督存在不等于主预测路径建立了局部依赖。

**决策**
保留 V3 三层监督实现作为明确对照，但该实验未通过局部表征验收，不扩大训练规模，也不调辅助 loss 权重。

**下一步**
最小化地把 object-patch prediction 与 dense effect prediction 结构性耦合，再用同一 small20 与 intervention 验收。

## 实验：V4 patch-routed Effect Decoder

**假设**
把 `interaction token → patch motion` 串入 dense effect 主链，并让每个 effect point 只读取最近 object patch 的 token，可以消除全局 attention 与 absolute xyz 的刚体捷径，使 shuffle interaction tokens 显著破坏预测。

**改动**
移除 Effect Decoder 的全局 cross-attention。effect point 根据当前 object-frame 坐标硬路由到最近 Uni3D object patch；预测为对应 patch motion 加局部 residual，residual 仅输入对应 `C_j`、point-to-patch 相对坐标、normal 和时间编码。干预工具同步重算被干预 token 的 patch motion，避免保留原始旁路结果。

**结果**
真实 batch forward/backward 和 5-epoch small20 训练均成功。最佳 checkpoint 位于 step 384。验证 ADE 为 49.45 mm，测试 ADE 为 42.94 mm；旧仅 effect baseline 为 46.93/44.80 mm，表现一升一降。验证 normal/mean/shuffle/cross-sample ADE 为 49.45/49.40/49.42/49.76 mm，shuffle 相对 normal 变化 -0.05%；测试为 42.94/42.93/42.95/43.74 mm，shuffle 相对变化 +0.03%。两者均远低于预设 +10% 验收线。

**诊断**
局部路由本身已经进入主链，但上游 interaction cosine 仍接近 1；当所有 `C_j` 和 patch-motion prediction 近似相同时，硬路由仍等价于读取全局刚体状态。结构耦合无法单独创造 object effect 中不存在或很弱的局部辨识信号。

**决策**
保留 V4 实现和负结果作为强结构对照，但判定未通过 object-indexed representation 验收；不继续调辅助 loss 权重或扩大训练。

**下一步**
重新审视刚体 object effect 是否支持 64-token object-indexed 表示；若继续该表示，需要引入能区分相同 SE(3) 下不同 human mechanism 的目标。

## 实验：canonicalization 定位与 patch target 能量分解

**假设**
如果 action-context 本身保留局部差异而 canonicalizer 将其抹平，空间局部 canonicalization 可能恢复 object-indexed interaction；同时 patch target 的空间能量占比决定这种表示是否真的受到数据监督。

**观察到的失败 / 现象**
V4 已将 patch prediction 接入 dense 主链，但 shuffle 仍几乎不影响 ADE。

**改动**
先导出 V4 最佳 checkpoint 的 16 个验证样本，比较 action-context、interaction 和 attention。随后以 5 cm Gaussian 权重按 object-hand 相对位置聚合 action tokens，训练相同 small20 5 epochs。最后增加 `target_diagnostics.py`，将 hand/object patch motion 分解为跨 patch 均值和 patch-local residual 能量。

**结果**
原 V4 action-context variance/cosine 为 0.592/0.814，interaction 为 0.000406/0.99959，canonical attention entropy 为 4.044，确认坍缩发生在 canonicalizer。空间 canonicalization 初始验证 interaction variance/cosine 改善到 0.0276/0.9720，但最佳 checkpoint 验证/测试 ADE 为 51.31/43.23 mm；shuffle 相对 normal 仅退化 0.07%/0.85%，仍未通过 10% 验收，结构改动已撤回。target 分解中，验证 hand/object local energy ratio 为 1.16%/1.01%，测试为 1.90%/3.30%；全局共同运动占 96.70–98.99%。

**诊断**
原 canonicalizer 的确会平均 action token，但这不是唯一根因。GRAB 刚体 patch motion target 的绝大多数能量本来就在全局共同分量，raw patch MSE 自然鼓励 64 个 token 给出相同预测。人为制造 token 差异不能让 decoder 使用数据中缺乏辨识力的局部信息。

**决策**
撤回 spatial canonicalization，保留 target 诊断工具。停止继续调 canonicalizer、loss 权重或 diversity 正则。

**下一步**
将表示拆分为全局 object motion/SE(3) 与 hand-patch mechanism；object-patch 表示只在有 local residual/contact 等信息性目标时使用。

## 实验：V5 hand-object 相对轨迹先验统计

**假设**
对 hand patch 与当前最近 object patch 的共同运动作差后，法向/切向相对运动应比 raw patch flow 含有更强的跨局部差异，因而更适合定义 interaction mechanism。

**观察到的失败 / 现象**
V4 的 hand/object raw patch target 只有约 1%–3% 局部能量，无法辨识相同刚体效果背后的不同手部作用方式。

**改动**
Dataset 增加仅供 Runner 监督与诊断使用的 current-object-frame future hand displacement，不加入模型 forward。新增只读 `relative_diagnostics.py`：复用 Uni3D patch index，为每个 hand patch 固定匹配当前最近 object patch，统计未来 8 步累计距离变化、法向/切向相对运动，以及 2 cm 接触状态转移。

**结果**
验证集法向/切向相对运动的跨边局部能量比例为 33.11%/22.16%，测试集为 41.58%/37.49%；相比 raw hand/object patch flow 的 1.16%/1.01%（验证）和 1.90%/3.30%（测试）明显更强。距离变化局部能量仅为 2.92%/3.20%。2 cm 阈值下验证/测试接触占比为 9.45%/17.86%，接触建立或解除只占全部 edge-step 的 0.98%/1.26%。切向累计相对运动中位数为 0.58/0.59 cm。

**诊断**
共同 global transport 被相对运动有效抵消，`v_n/v_t` 确实提供了 raw flow 缺失的局部辨识信号。距离变化仍主要受样本级共同分量控制；硬 contact transition 很稀疏，第一版若以分类为主会面临严重类别不均衡。

**决策**
保留相对轨迹诊断和数据 target，判定 V5 值得进入最小训练原型。训练时优先直接监督法向/切向相对运动，距离只作辅助，不先加入 contact classification。

**下一步**
构造 hand→object 局部 edge token 和 `v_n/v_t` 预测头，在 small20 上比较 target 拟合、token 局部差异与 effect consistency。

## 实验：V5 最近邻 edge token 直接监督

**假设**
将 hand patch 的 action-context、当前最近 object patch token 和相对几何融合成 edge token，并直接监督未来累计法向/切向相对运动，可以学习不坍缩且能跨序列预测 mechanism 的表示。

**改动**
新增 64 个 hand→nearest-object edge token 和 `[8,64,4]` 的 `[v_n,v_t]` 预测头。V5 配置保留 effect consistency，关闭 raw hand/object patch 辅助损失。先监督全部最近邻边，再按 V5 建议仅监督当前距离小于 5 cm 的近场边；不加入稀疏 contact classification，也不调 loss 权重。

**结果**
全部 edge 实验位于 `outputs/interactiondynamics/interaction_dynamics_20260810_001729`。5 epochs 后最佳验证/测试 relative RMSE 为 1.389/1.178 cm，差于 zero predictor 的 1.370/1.110 cm；effect ADE 为 49.96/45.31 mm。

5 cm edge 实验位于 `outputs/interactiondynamics/interaction_dynamics_20260810_002606`，最佳 checkpoint 为 step 768。验证/测试有效 edge 占 38.66%/56.35%；relative RMSE 为 0.999/0.874 cm，仍差于 masked zero 的 0.968/0.841 cm。effect ADE 为 48.77/44.10 mm。edge token variance/cosine 为验证 0.125/0.871、测试 0.155/0.840，未发生坍缩。

**诊断**
排除远距离 edge 可降低 target 绝对误差，却不能使相对轨迹预测优于零基线。当前失败不是 token collapse，也不是单纯由远处非交互手部区域污染；更可能来自固定 current-frame 最近邻无法描述未来 correspondence、累计轨迹 MSE 受重尾样本影响，或 small20 跨序列 mechanism 本身难以从当前输入泛化。

**决策**
保留 edge-token 和 5 cm mask 实现作为 V5 明确负对照，不接入 Effect Decoder，不继续调 loss 权重或扩大数据规模。

**下一步**
先分解近场 target 的逐步增量、重尾程度和 correspondence 稳定性；只有诊断支持时，再最小测试逐帧匹配或 robust relative loss。

## 实验：V6 correspondence、mask 与逐步增量

**假设**
V5 打不过零预测可能来自固定 current-frame nearest 在未来失效、current-near mask 漏掉接触建立，以及累计 target 的重尾；逐步增量可能是最小可预测 target。

**改动**
Dataset 增加仅供 Runner/诊断使用的 future object normals。新增 `correspondence_diagnostics.py`，使用稳定 patch point index 重建未来 patch centers，逐帧计算 nearest object patch；比较 fixed/dynamic correspondence、current-near/chunk-active mask，以及 fixed cumulative、fixed increment、dynamic increment 的分布。随后只改变一个变量，在相同 small20、5 cm current mask、MSE 和网络上训练 `increment_fixed` 5 epochs。

**结果**
验证/测试分别有 51.57%/53.83% edge 至少换过一次最近 patch，fixed 逐帧匹配率为 68.29%/66.11%，平均切换 0.83/0.88 次。current-near mask 占 38.66%/56.35%，chunk-active 占 43.89%/60.28%，即漏掉 active edges 的 11.91%/6.51%。

逐步增量把 current-near RMS 从累计 target 的 1.331/1.179 cm 降至 0.352/0.300 cm，但重尾比 `q99/q50` 仍为 19.53/22.21。dynamic increment 的 RMS 为 0.354/0.299 cm，与 fixed increment 几乎相同。

训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_084655`，最佳 effect checkpoint 位于 step 192。验证/测试 relative RMSE 为 0.275/0.240 cm，仍差于 zero predictor 的 0.247/0.206 cm；effect ADE 为 48.62/43.51 mm。各 epoch 验证 relative RMSE 均未优于 zero。

dominant-hand summary 的 8,488 个候选窗口中保留 5,770（67.98%）；主要排除 1,836 个双方不活跃窗口（21.63%）和 841 个双手协作窗口（9.91%）。

**诊断**
固定 correspondence 确实经常失效，静态 mask 也会漏掉接触建立，但它们不是当前误差分布的唯一主因：动态/固定增量统计近似，而单独改成增量仍不优于 zero。直接加入“全量”还会引入大量双方不活跃窗口；真正可能补充 mechanism 多样性的是 bilateral-active 子集。

**决策**
保留诊断工具、future normals target 和 increment 配置作为负对照。不继续做仅替换 dynamic nearest 的 5-epoch 低信息量训练，也不调 relative loss 权重。

**下一步**
重新设计 interaction-change 与 bilateral-active 数据采样，并先建立条件均值等强基线；目标在新采样上可预测后，再训练 mechanism representation。

## 实验：V7 时空 object-centric interaction field 与全局 SE(3)

**假设**
V5/V6 的困难来自把 interaction 压缩为单帧最近邻 edge 并直接拟合高噪声局部 target。保留完整时间 action token，以连续 Gaussian 场将已知手部轨迹投影到物体表面，再显式预测全局刚体 SE(3)，应更符合 GRAB 刚体 effect 的结构并改善未见序列泛化。

**观察到的失败 / 现象**
V4 的 object patch flow 约 97%–99% 为全局共同运动；V5/V6 的局部相对目标均未优于零预测。继续训练 nearest edge 或调相对损失权重的信息量较低。

**改动**
Action Encoder 输出 `[B,8,64,D]` temporal tokens。对每个时间步和 object patch，使用 5 cm Gaussian 对 64 个 hand patches 连续加权，融合 action、object token、相对几何、法向和手部法向/切向增量，形成时空 interaction field。全局池化后预测 8 个逐步 SE(3) increment；旋转由 axis-angle 转换并逐步复合，dense point flow 由刚体变换解析生成。训练只使用厘米制 translation MSE 与 rotation-matrix MSE（权重 1/10），关闭旧 effect/patch/relative losses。增加 field 空间/时间方差、描述子方差、接近密度和 soft-weight entropy 指标。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_102612`，5 epochs / 1920 steps，最佳 checkpoint 位于 step 1152。验证 ADE 为 46.70 mm，zero-flow 为 49.05 mm，改善 2.35 mm（4.8%）。测试 ADE/FDE 为 42.08/73.09 mm，zero-flow ADE 为 45.82 mm，改善 3.75 mm（8.2%）；此前 V4/V6 测试 ADE 为 42.94/43.51 mm。测试 SE(3) translation error 为 0.953 cm，rotation error 为 2.572°。测试 field 空间/时间 feature variance 为 0.0411/0.0171，descriptor variance 为 0.00340，soft hand-weight entropy 为 3.621（64-way 上限 4.159），说明场在空间和时间上都不是常量。训练 batch ADE 可达到约 10 mm。实现与坐标链路聚焦测试共 15 项全部通过。

**诊断**
显式刚体归纳偏置避免了逐点 decoder 学习一个几乎纯全局的 flow，同时连续 splatting 保留了完整动作时序并消除了 hard nearest correspondence。验证和测试均优于各自 zero-flow，方向成立；但当前尚不能区分收益主要来自 SE(3) head，还是来自局部时空 field。

**决策**
保留 V7 路径，作为下一阶段主实验方向。旧 V1–V6 分支只保留作对照，不据此继续调 nearest-edge target。

**下一步**
做 action shuffle、移除显式 descriptor、global-action-only 三个最小消融，确认 V7 是否真实利用局部时空交互，再考虑扩大数据。

## 实验：V8 full-resolution local-edge sanity check

**假设**
V6 的 64-patch 表示可能在交互关系形成前丢失指尖和接触边界等稀疏高频信号。若保留全部 1538 个 hand surface points，并只连接每点最近的 4 个 object points，逐点 dense interaction representation 应明显优于 patch-edge 和 zero baseline。

**观察到的失败 / 现象**
V6 patch-edge 的验证/测试增量 relative RMSE 为 0.275/0.240 cm，差于 zero 0.247/0.206 cm；但不能排除 FPS/KNN patch pooling 已破坏接触几何。

**改动**
新增不加载 Uni3D/DenseToken 的 `DenseEdgeInteractionModel`。对完整 1538×4096 当前几何计算 KNN，每个 hand point 保留 4 条边；以逐步 hand position、相对位置/距离、hand/object normals、hand increment 及其法向/切向分解构造 edge，经 64 维 MLP 和距离 soft pooling 得到 `[B,8,1538,64]` dense interaction set。Runner 使用 future object 仅构造逐点固定对应的增量 relative target，并同时报告 zero 与 hand-motion-only 基线。尚未加入 bottleneck `C`。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_110406`，5 epochs / 1920 steps，耗时 2 分 36 秒。所有 epoch 验证结果均未超过 zero；最佳 checkpoint 为 step 384。最佳验证 relative RMSE 为 0.2470 cm，zero 为 0.2465 cm，hand-only 为 0.5971 cm；测试为 0.2031/0.2027/0.5507 cm。验证/测试 5 cm 内有效 hand-point 比例为 60.76%/77.85%。dense token feature variance 从验证 epoch 1 的 0.000894 增至 epoch 5 的 0.001322，但误差同步恶化。相关聚焦测试通过。

**诊断**
全量 hand points 和更高近场覆盖率没有让相对增量跨序列变得可预测，说明 64-patch 过早压缩不是 V6 失败的主要原因。hand-only 基线远差于 zero，也说明 target 不能靠直接复制已知 hand action 解决。继续增加 K、edge width 或 learned compressor 只会在尚未成立的 `E_dense` 上增加复杂度。

**决策**
保留轻量 V8 分支和负结果作为 dense-first 对照；暂不实现 `E_dense → C` reconstruction/compressor，也不调 K 或网络宽度。V7 的显式 SE(3) effect 路径仍是当前主方向。

**下一步**
优先完成 V7 action/descriptor/global-only 消融以定位 8.2% test ADE 改善来源；只有找到比 zero 更可预测的 interaction-specific target 后，再重启 dense bottleneck 研究。

## 实验：V9 global-action-only 对 V7 局部场归因

**假设**
若 V7 的 object-indexed spatiotemporal field 确实编码了不可替代的局部 interaction，移除空间 field 与显式 `[rho,a_n,a_t]` descriptor、只保留全局 temporal action 和 object context 后，验证与测试 ADE 应明显退化。

**观察到的失败 / 现象**
V7 测试 ADE 比 zero-flow 改善 8.2%，但 V8 已证明提高局部几何分辨率不能改善 relative target。尚不清楚 V7 收益来自 local field，还是来自更匹配刚体数据的 SE(3) head 与全局 motion correlation。

**改动**
增加 `field_ablation`。`global_action_only` 保留与 V7 相同的 Action Encoder、world cross-attention、object tokens、SE(3) head、loss 和 small20 配置；仅将 temporal action 在 64 个 hand patches 上取均值并投影，再复制到全部 object positions，同时把 field descriptor 清零。送入 SE(3) head 的 spatial feature variance 和 descriptor variance 因而严格为 0。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_112110`，5 epochs / 1920 steps，最佳 checkpoint 位于 step 1536。各轮验证 ADE 为 48.29、46.20、46.26、45.41、45.68 mm；最佳验证 ADE/FDE 为 45.41/77.51 mm，优于 full-field 的 46.70 mm。最佳 checkpoint 测试 ADE/FDE 为 42.29/73.38 mm，full-field 为 42.08/73.09 mm，仅退化 0.21/0.29 mm（ADE 约 0.5%）。测试 translation/rotation error 为 0.957 cm/2.628°，full-field 为 0.953 cm/2.572°。global-only 测试 field spatial variance 和 descriptor variance 均为 0。

**诊断**
局部场被完全消除后，验证反而改善，测试仅有远小于跨实验波动的轻微退化；因此没有证据表明 V7 使用了不可替代的 object-indexed local interaction。V7 的有效部分是逐步 SE(3) 刚体归纳偏置，以及包含 world context 的全局 temporal action conditioning。原 8.2% test 改善不能作为 mechanism latent 已成立的证据。

**决策**
保留 `global_action_only` 作为新的强 effect baseline，并保留 full-field 作对照；降低 full-field 的研究优先级。暂不做 descriptor-only 调权、dense compressor 或增加 field 分辨率。

**下一步**
先设计 interaction-rich sampling 或能区分“相同全局 hand transport、不同局部接触机制”的对照任务。只有局部模型能稳定超过 global-action-only 时，再研究可压缩的 interaction bottleneck `C`。

## 实验：V9 global C 监督的 MANO self-inverse

**假设**
即使 V9 的 C 主要表示全局 temporal action，它仍可能作为可微监督，从当前 MANO 手初始化恢复未来 8 帧动作，从而先打通 human self-inverse 到后续 robot FK 所需的优化链路。

**改动**
实现 `inverse_mano.py`。脚本根据 cache 的 `source_raw_file/raw_frame_id/side` 回到 GRAB raw 参数，加载对应 subject v_template 的左右手 MANO。冻结 V9 encoder，以原始 human future action 的 `[8,128]` global C 为 teacher；将未来 MANO pose、global orientation、translation 全部初始化为当前帧，优化 C MSE 加时间平滑先验。GT future hand points 不进入 loss，只计算 ADE/FDE。

**结果**
test sample 0、200 steps、Adam lr 0.003、smooth weight 0.01 时，C RMSE 从 0.2377 降至 0.00953；hand-point ADE 从静止基线 27.52 mm 降至 10.88 mm，FDE 从 50.43 mm 降至 10.63 mm。产物为 `output/InteractionDynamics/mano_inverse/test_000_smooth.npz`。高学习率 0.03、smooth weight 1e-4 的对照虽然将 C RMSE 降到 0.118，但 ADE/FDE 恶化到 82.97/252.04 mm。

**诊断**
C→MANO 的梯度链和 raw/cache 帧对应均已打通。C 监督含有可恢复动作的信息，但 inverse 明显欠约束：弱先验允许错误 MANO 轨迹匹配更低的 C loss。合理的时间先验不是可选鲁棒性，而是决定重建是否可信的必要约束。

**决策**
保留 MANO self-inverse 工具，并将稳定的 lr 0.003、smooth weight 0.01 设为默认。当前只通过单样本 sanity check，不宣称 split-level 泛化。

**下一步**
批量运行多个 test chunks，报告成功率和 ADE/FDE 分布；增加 C-only、C+smooth 和 oracle hand trajectory 三组对照后，再接 Robot FK。

## 实验：V10 Action token 的 global/local linear probe

**假设**
V9 只证明 object-indexed field 不必要，并不能证明 64 个 hand-patch Action tokens 已经坍缩。若 Action Encoder 仍保留手指局部动作，冻结表示上的线性 probe 应能在未见序列上预测去除全局均值后的 patch-local hand flow，并优于 zero predictor。

**改动**
新增 `action_probe.py`。在冻结 V9 best checkpoint 上一次性提取 small20 train/val/test 的 `action_context_tokens` 和稳定 KNN patch flow，将 target 精确分解为每步 global mean 与 zero-mean local residual。分别拟合 full、global、local 三个 affine ridge probe；ridge 从 0.001–100 中仅按 validation RMSE 选择，随后固定评估 test。主 encoder 不更新。

**结果**
Action token 的 train spatial variance 为 2.370，跨 patch pairwise cosine 为 0.721，未发生 collapse。global probe 选择 ridge 1，训练/验证/测试 RMSE 为 0.601/1.556/1.666 cm，对应相对 zero 改善 85.2%/67.0%/60.4%。local residual probe 选择 ridge 100，RMSE 为 0.297/0.427/0.452 cm，相对 zero 改善 44.9%/16.4%/22.8%。full-flow probe 选择 ridge 100，验证改善 20.1%，测试只改善 3.1%。结果保存在 `output/InteractionDynamics/action_probe/v9_small20.json`。

**诊断**
Action Encoder 明显保留全局运动，也保留较弱但可跨 sequence 线性读出的 patch-local residual。因此 V9 的结论应限定为“SE(3) effect head 不需要空间 field”，不能扩展为“Action representation 已全局坍缩”。full-flow 指标被大尺度 global motion 主导，无法单独衡量局部信息保存程度。

**决策**
保留 probe 工具。当前 checkpoint 已通过 local-information sanity check，不立即打开旧 `action_loss_weight=1`；旧 head 重建 raw flow 的目标不够针对性。若后续训练 action-information loss，应拆分 global mean head 和 local residual head，并分别报告相对 zero 的改善。

**下一步**
在 MANO self-inverse 中比较只监督 V9 global C 与同时监督 patch-level Action tokens，检查 local Action 信息能否进一步降低手指局部重建误差；这比直接重训旧 raw action head更接近 RL prior 需求。

## 实验：V11 object-centric body action field 辅助监督

**假设**
由全局时序 `C` 和 object tokens 重建不依赖 MANO 身份的 `[rho,d,v_n^local,v_t^local]` 场，可以保护局部执行体动作信息，同时不损害 SE(3) effect 泛化。

**观察到的失败 / 现象**
V9 的 Action local probe 在 test 上优于零预测 22.8%，但局部信息明显弱于 global 信息；直接重建 MANO 又会引入 human-specific 语义。

**改动**
保持 V9 global-action-only 与 SE(3) 路径不变。新增 `C + object world token → 6D field` MLP。target 使用完整 1538 点 future hand surface：5 cm Gaussian density、最近距离，以及将累计 hand displacement 减去全手均值后按 object normal 分解的局部法向/切向运动。三组 loss 等权，均不使用 future object motion、MANO joint、finger ID 或 hand point identity。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_134059`。先训练 5 epochs / 1920 steps，再从 latest checkpoint 续训到总计 20 epochs / 7680 steps；实测耗时分别为 10:02 和 30:53，总计 40:55。按 object ADE 选择的 checkpoint 仍是 epoch 1：验证 48.42 mm，test ADE/FDE 45.71/79.40 mm。

local-motion 验证 RMSE 从 epoch 1 的 0.321 cm 下降，在 epoch 15 达到全程最好 0.242 cm，但仍差于 zero 0.223 cm；epoch 20 为 0.253 cm。使用 epoch 15 checkpoint 评估 test，density 为 0.136（zero 0.311）、distance 为 2.083 cm（zero 5.882 cm）、local-motion 为 0.277 cm（zero 0.243 cm），object ADE/FDE 为 44.41/79.28 mm。长训练比 epoch 1 的 test ADE 45.71 mm 有恢复，但仍弱于 V9 的 42.29/73.38 mm。

epoch 1 Action token spatial variance/cosine 为 1.030/0.760，local probe test 相对 zero 改善 34.0%。epoch 15 为 1.718/0.674，local probe 验证/test 改善 21.5%/26.1%，仍高于 V10 的 16.4%/22.8%，但优势明显缩小；global probe test 改善回到 60.5%，与 V10 的 60.4% 相同。

**诊断**
5 epochs 时训练尚未收敛，不能据此断言最终负优化；20-epoch 结果确认 local target 继续改善到 epoch 15，但随后平台波动，始终未超过 zero。辅助监督确实改变了 Action Encoder，并略微增加其线性可读局部信息；decoder 仍主要学会 density/distance 这类静态几何。object effect 长训练只部分恢复且始终弱于 V9，因此现有证据支持“固定 small20 下存在目标冲突”，但不外推为该监督在更大数据上必然负优化。

**决策**
保留 V11 实现和配置作为部分正向、整体未通过的对照。结论以完整 20 epochs 为准，不再称 5-epoch 结果已经收敛。当前不调 loss 权重：V11 的 field-zero 与 effect 两项验收仍失败，简单降权不能解决 target 可预测性。

**下一步**
先对 local-motion target 做 conditional-mean / 邻近区域 mask 诊断，确认远离物体处的 Gaussian 归一化是否把微弱、无意义的运动放大；只有 target 本身存在可超过零的强基线后，再测试近场 masked local loss 或与 SE(3) 梯度解耦。

## 实验：V9 SE(3) controlled overfit 与 Cm 误差口径对齐

**假设**
small20 上约 30–45 mm 的 train/test ADE 可能来自 SE(3) 坐标链或模型容量错误；若是如此，V9 在与原 V1 相同的 32 chunks、400 steps 下也无法充分过拟合。

**观察到的失败 / 现象**
V9 small20 训练 5 epochs 的 train ADE 为 29.03 mm，V11 为 43.60 mm；而 Cm full-GRAB 实验最终 train EPE 为 8.31 mm，容易被理解为 InteractionDynamics 连训练集也学不好。

**改动**
不改代码。使用 V9 `global_action_only`、SE(3) translation/rotation loss 和原 overfit 数据设置，固定 32 chunks、batch size 4，训练 50 epochs / 400 steps；无验证集和辅助 interaction loss。

**结果**
产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_140302`。约 200 steps 时单 batch ADE 已为 2.2–2.5 mm；400 steps 的 epoch ADE/FDE 为 1.277/1.598 mm，最后 batch 为 1.378/1.511 mm，translation error 为 0.040 cm、rotation error 为 0.444°。V9 因而能够比原 V1 的 6.71/12.43 mm 更充分地记住相同规模数据。

Cm 的可比 full-GRAB 结果使用 1068 条 train sequences、20 epochs / 226780 steps，最终 train EPE 为 8.31 mm；验证 stride 1–10 的 endpoint EPE 为 2.64–18.45 mm，平均 10.40 mm。它还只从当前手部 5 cm 内采 object points，并以校准尺度的 vector Huber 直接预测单个 endpoint flow。InteractionDynamics small20 只有 12 条 train sequences、5 epochs / 1920 steps，ADE 是未来 8 个累计时刻的平均，并由 8 个增量 SE(3) 复合得到。

**诊断**
SE(3) 解析链、尺度和容量没有阻止拟合；“无法过拟合”的判断来自把 small20 多序列训练均值当成 controlled overfit，并与 Cm 的大数据长训练 endpoint EPE 直接比较。剩余差距主要是未见 sequence 泛化：当前 V9 将 64 个 action patches 全局平均为 `C`，会丢掉 Cm 直接使用完整 1538 点 endpoint hand flow 与近场 object geometry 时保留的接触机制；同时逐步 SE(3) 的小误差会沿 8 步累计。V11 在 small20 上观察到目标冲突，但不是基础路径学不动。

**决策**
不修改 ADE 或 SE(3) 坐标实现，也不以扩大 decoder 容量为下一步。后续比较必须同时报告 controlled-overfit ADE、zero-relative improvement、逐步 EPE，并在相同 sequence 数与训练预算下对齐 Cm/InteractionDynamics；模型研究优先恢复近场局部 action 条件，而不是继续只用 global mean `C`。

## 实验：V12-A wrist 与局部关节动作分解监督

**假设**
现有 ActionToken 已包含手指局部动作，只是累计 raw flow 被 wrist 刚体运动主导；将监督拆成相邻 wrist SE(3) 和各帧 wrist frame 下的局部关节流，应同时超过各自零预测基线。

**观察到的失败 / 现象**
V10 local linear probe 只有 22.8% 改善，V11 的 object-centric local-motion 解码在 20 epochs 后仍差于零预测。现有 `action_reconstruction` 只预测累计 patch mean flow，且 V9/V11 配置将其权重设为零。

**改动**
Dataset 新增相邻 `hand_root_increment_pose_gt`，并把每帧完整 MANO surface 分别变换到自己的 wrist frame 后做稳定顶点差分，得到 `hand_articulation_increment_gt`。不修改 ActionEncoder 输入；从 `[B,8,64,D]` temporal ActionToken 逐 patch 解码 articulated flow，并从 patch 均值解码 root translation 与 rotation。三条 loss 独立配置和记录，物体 SE(3) 主路径保持 V9 global-action-only。

**结果**
运行 `outputs/interactiondynamics/interaction_dynamics_20260810_171704`，small20 共 5 epochs / 1920 steps，耗时 10:06。局部关节流验证 RMSE 从 epoch 1 的 0.0928 cm 降到 epoch 4 最佳 0.0586 cm，但仍差于 zero 0.0393 cm；epoch 5 回升到 0.0987 cm。最佳 checkpoint 的 test 局部流为 0.0599 cm，zero 为 0.0354 cm。wrist translation 验证/test 为 0.2648/0.2555 cm，明显优于 zero 0.6892/0.6637 cm；test wrist rotation error 为 2.75°。test object ADE/FDE 为 42.72/74.35 mm，略弱于 V9 的 42.29/73.38 mm。

**诊断**
目标分解是有效的：root motion 的可预测性与 local articulation 的失败被清楚隔离。现有 encoder 直接读取当前姿态和相对当前帧的 8 步累计位移，确实更容易编码全局刚体运动；只增加正确 local loss 能持续降低误差，但在相同训练预算下仍无法跨序列胜过极强的近零动作基线。结果支持 V12 文档提出的 static pose 与相邻 pose-pair 编码动机。

**决策**
保留 V12-A 作为诊断基线，不继续仅调整当前 cumulative encoder 的辅助 loss 权重。下一实验进入 V12-B/C：显式编码每帧 canonical static surface，再由相邻 pose representation 构造 incremental ActionToken。

## 实验：V12-B/C static pose-pair incremental ActionToken

**假设**
将每帧手表面放入各自 wrist frame，先编码 static local/global PoseToken，再由相邻 pose pair 构造 ActionToken，可以避免 cumulative flow 的 root-motion 主导，并使局部关节流跨序列优于零预测。

**观察到的失败 / 现象**
V12-A 的 wrist motion 可泛化，但 articulated flow test 0.0599 cm 仍差于 zero 0.0354 cm，说明仅给旧 cumulative encoder 增加正确监督不够。

**改动**
Dataset 输出当前加未来共 9 帧 wrist-local stable MANO surface 和 8 个 wrist SE(3) increment。共享 static encoder 在当前 World hand 的稳定 KNN 上产生 64 个 PoseToken 和 global mean code；pair encoder 融合前后 PoseToken、差分、global codes、wrist increment，以及相邻 patch 点差的 mean/max/min，再经 temporal Transformer 得到 8 步 ActionToken。static head 重建 64 个 wrist-local patch centers。由于 articulation target RMS 仅约 0.01–0.05 cm，decoder 使用零初始化并将该 loss 权重设为 100；static loss 权重为 0.01。

**结果**
最终 controlled overfit 为 `outputs/interactiondynamics/interaction_dynamics_20260810_174654`：32 chunks、400 steps、1:52。epoch 50 articulation 为 0.00990 cm，优于 zero 0.01197 cm（改善 17.3%）；static patch-center 为 2.910 cm，优于 centered baseline 2.969 cm；object ADE/FDE 为 3.11/4.15 mm。此前随机初始化或 raw-MSE 同权会让小幅 articulation 梯度被淹没，均未通过 overfit，已撤回。

small20 产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_174924`，5 epochs / 1920 steps，耗时 10:03。按 validation articulation 选择 epoch 4：validation 为 0.01330 cm，zero 0.03930 cm，改善 66.2%；test 为 0.01173 cm，zero 0.03537 cm，改善 66.8%。test wrist translation 为 0.2698 cm，zero 0.6637 cm，改善 59.4%；rotation error 2.17°。validation static patch-center 为 2.763 cm，略优于 centered baseline 2.800 cm；test 为 3.184 cm，差于 baseline 2.943 cm。test object ADE/FDE 为 45.07/78.57 mm，弱于 V9 的 42.29/73.38 mm。

**诊断**
V12 的主要表示假设成立：相邻 wrist-frame surface motion 能形成跨 sequence 可读的详细 ActionToken，且比 V12-A 有大幅改善。static pose 在 validation 通过、test 失败，符合不同 subject morphology 混入 wrist-local surface 的风险。object effect 没同步改善，是因为当前仍使用 `global_action_only`，将 64 个详细 ActionToken 池化后送入 SE(3)；interaction tokens 的 test cosine 仍为 0.9993。

**决策**
保留 V12-B/C 作为新的 ActionEncoder 主候选。下一步不再调整 action reconstruction，而是分开处理两个问题：用 neutral deformation 或 MANO PCA pose 消除 static code 的 morphology；让 interaction/effect 显式消费 local incremental ActionToken，并用 object ADE 与 intervention 验证。

## 实验：V13 Local ActionToken 到 interaction slots

**假设**
V12 已学到可泛化的局部关节动作；用 object-query cross-attention 建立 object-indexed interaction，再压缩为 8 个 embodiment-independent slots，可以让 object SE(3) 真正依赖 local ActionToken identity。

**观察到的失败 / 现象**
V12 的 object effect 仍先对 64 个 ActionToken 取全局均值。V13 首版虽然产生 `[B,8,8,128]` slots，但 SE(3) 对 slots 再取均值；small20 test normal/mean/shuffle/cross-sample ADE 为 43.965/43.939/43.964/44.381 mm，mean 和 shuffle 几乎无影响，未通过核心 intervention。

**诊断**
仅引入 slot 形状不等于 decoder 使用局部结构。首版 object→action 可以退化为集合汇聚，slot→SE(3) 的硬均值又进一步削弱局部配对。需要显式提供 object-hand 空间对应，并移除最终硬均值。

**改动**
保留 V12 PosePairActionEncoder 与全部 action supervision。object token 查询带 object/hand patch 位置编码的 ActionToken，并加入 5 cm 距离 attention bias；64 个 object-indexed interaction tokens 由 8 个 learnable slots 聚合，再由单个 learned effect query 读取 slots 作为 SE(3) 条件。intervention 在进入 local fusion 前执行 normal、patch mean、patch reverse shuffle 和跨 sample 替换。

**结果**
修正版 controlled overfit 为 `outputs/interactiondynamics/interaction_dynamics_20260810_194117`：32 chunks、400 steps、1:58，object ADE 2.896 mm。small20 为 `outputs/interactiondynamics/interaction_dynamics_20260810_194350`：5 epochs / 1920 steps、10:27，最佳 epoch 5 验证 ADE/FDE 44.358/76.638 mm，articulation 0.0170 cm（zero 0.0393）。test normal ADE/FDE 为 43.584/75.845 mm；mean 43.732、shuffle 43.602、cross-sample 44.013 mm，相对 normal 分别退化 0.148、0.018、0.429 mm。

**决策**
保留 V13 结构作为下一步原型，但判定核心验收未通过：normal 虽优于 mean/shuffle，shuffle 差距远不够明显，尚不能证明 local ActionToken identity 真正影响 object effect。当前不进入 V14 multi-embodiment，也不修改已经成功的 V12 ActionToken。

**下一步**
导出并检查两级 attention 的空间分布；设计不会被 object/slot 集合池化抵消的局部 effect 读取方式，再以相同四组 intervention 验收。
