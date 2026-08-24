## 2026-08-21 — ARCTIC MANO 小子集采用 axis-angle45 5mm 标定并做三条件评估

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`

**未指定点**

用户只要求先导出部分带 MANO 的 ARCTIC 数据并评估，没有指定导出规模、hand noise 表示和条件矩阵。

**实际选择**

复用已有 min11 物体清单，每个物体取一个 `s01` 左手文件，共 11 文件 / 4175 帧；从 raw `.mano.npy` 重新生成 Stage 2/3 到 NAS。ARCTIC 使用 axis-angle45，不转换成 PCA；沿用 ARCTIC 9mm geometry calibration，并按 `5/9` 缩放到 5mm RMS。评估 `object_only`、`hand_only`、`hand_object` 三个条件，关闭 runtime resampling。

**其他合理选择**

1. 直接把 ARCTIC axis-angle45 转换为 GRAB PCA24 后加噪声。
2. 只做 hand-only，不做 joint condition。
3. 导出多个 subject 或全量 ARCTIC。

**选择理由**

axis-angle45 是 ARCTIC 原生 MANO 表示，避免引入 PCA basis/mean 的额外转换；min11 已覆盖全部 object 类别且控制评测成本；三条件能区分 hand-only 学习和 object/compound exposure。

**对结果的影响**

结果适合判断方向和暴露分布问题，不代表跨 subject 或全量 ARCTIC 结论；5mm RMS 与 GRAB PCA5mm 是几何尺度对齐，不是参数分布完全相同。

**可逆性**

完全可逆；NAS 输出使用独立版本目录，不覆盖旧 Stage 3。

**建议用户确认**

否

## 2026-08-20 — V1 object-macro 通过按 object 单独评估再取平均

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`

**未指定点**

V1 要求报告 object-macro，但现有 evaluator 只直接输出全局 micro，没有对象维度的汇总接口。

**实际选择**

先用确定性分层子集覆盖全部 object / subject / action，再为每个 object 单独建立软链子集，分别跑 pure GRAB 与 GRAB+ContactPose，最后对 11 个 object 的结果取算术平均作为 object-macro。

**其他合理选择**

1. 修改 evaluator 增加对象维度的汇总输出。
2. 只报告 micro，不补 object-macro。
3. 用 object 频率加权平均近似 macro。

**选择理由**

不改 evaluator 可以保持评测协议稳定，且 per-object 重跑得到的 macro 口径直接、可复查。算术平均符合 V1 的 object-macro 语义。

**对结果的影响**

object-macro 与 micro 会略有差异，但都来自同一套协议和同一批 object 子集，便于判断是否存在单一高频 object 偏置。

**可逆性**

完全可逆

**建议用户确认**
否

## 2026-08-20 — ARCTIC 时间受限评测采用 subject/object/action 确定性分层

- branch: `feature/hand-pca-perturbation`
- post-commit: `25262bc`

**未指定点**

用户要求在全量 ARCTIC 或覆盖全部物体类别的分层子集上评估，但未指定时间受限子集的每层采样规则。

**实际选择**

全量评估优先；只有 GPU 时间受限时才使用分层子集。分层按 `(subject, object)` 覆盖全部可用组合，每层优先各取字典序最小的 `grab` 和 `use` paired sequence，并保留实际存在的左右手文件。

**其他合理选择**

1. 每个物体类别固定相同帧数，不强制覆盖所有 subject。
2. 按全量 ARCTIC 的类别频率等比例抽样。
3. 完全随机抽取后用 seed 固定结果。

**选择理由**

当前旧子集因字典序截取而全部落在 `box_*`。显式覆盖 subject、object 和 action 能直接消除该类别偏置；确定性字典序规则易复现，且不依赖额外随机状态。

**对结果的影响**

分层结果更适合做类别 macro 比较，但其自然类别频率与全量 ARCTIC 不同，不能代替全量 frame-weighted micro 指标。正式结论仍优先使用全量评估。

**可逆性**

完全可逆

**建议用户确认**

否

## 2026-08-20 — 三域首轮混训采用独立 Dataset 与等比例 batch sampler

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`

**未指定点**

用户指定了三个域各占 1/3，但没有指定是把文件拼到一个目录后加权采样，还是保留三个 Dataset 并在 batch 级别控制比例；也没有指定首轮 batch size 和验证 loader 的组织方式。

**实际选择**

新增 `DomainConcatDataset` 与 `DomainBalancedSampler`：每个本地 batch 固定使用三个域相同数量的样本，DDP 时按 rank 切分每个域的采样流。首轮配置使用 batch size 48（每域 16），保留统一 object pose perturbation，关闭 MANO 重建/手部扰动、runtime object resampling 和 scale augmentation。每个域单独构造 clean/perturbed validation loader。

**其他合理选择**

1. 将三个目录软链接到一个目录，使用普通随机 sampler；
2. 使用带 replacement 的 PyTorch WeightedRandomSampler；
3. 训练只保留一个混合 validation loader，再由 batch 内 dataset id 统计域指标。

**选择理由**

batch 级等比例采样能直接保证每个 optimizer step 的域贡献，且不会被数据集 frame 数量支配；独立 validation loader 能暴露某一域退化，不需要修改 evaluator 的指标语义。首轮不引入自适应 loss/gradient balancing，避免把采样问题和优化调节问题混为一个实验。

**对结果的影响**

相比普通 concat sampler，小域会被重复循环，大域不会按原始数量占主导；因此训练预算应按 optimizer step 和每域曝光量记录。batch size 48 在单卡时 global batch 为 48，在双卡时为 96，后续与纯 GRAB 做严格比较时需显式记录这一点。

**可逆性**

完全可逆；删除 `data.domain_paths` 即回到原有单域 loader。

**建议用户确认**

否；这是对“1/3 domain sampling、不开 runtime、不加 scale”的直接工程实现。

## 2026-08-19 — 5 mm no-PCA runtime 训练以 5 mm 配置为底，仅关闭 hand perturb

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`

**未指定点**

新 run 应该在 5 mm 配置基础上保留哪些字段，尤其是 `exclusive_hand_object_perturb`、`runtime_resample_object` 和训练超参数是否都保持不变。

**实际选择**

以 `full_grab_50ep_geometry_5mm_exclusive_ddp2.yaml` 为底，只把 `meta.apply_hand_perturb` 设为 `false`，其余训练预算、数据路径、object perturb 与 runtime resampling 全部沿用 5 mm。

**其他合理选择**

1. 额外把 `hand_perturb_prob` 设为 `0.0`。
2. 直接复制旧 no-PCA 配置，再手动补回 5 mm 的 runtime sampling 语义。

**选择理由**

用户要求“配置和这版 5mm 一致”，因此最保守的做法是只关闭 PCA 扰动本身，避免把训练 recipe 一起改掉，保证和 5 mm 的对照关系清晰。

**对结果的影响**

如果结果变化，基本可以归因到“是否施加 hand perturb”这一项，而不是 batch、LR、epoch 或 runtime 采样预算。

**可逆性**

完全可逆

**建议用户确认**

否
## 2026-08-21 — H50/O50 首轮先使用 GPU 0 单卡启动

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`

**未指定点**

用户确认了 H50/O50 互斥比例，但未指定必须等待空闲 GPU、共享已有 GPU，还是先用单卡启动。

**实际选择**

在不终止其他任务的前提下，先用物理 GPU 0 单卡启动 H50/O50，覆盖 `train.distributed.enable=false`，保留 `train.max_steps=454000`。

**其他合理选择**

- 等待 GPU 3 空闲后按原 DDP2 运行；
- 与 GPU 3 上的其他任务共享显存启动 DDP2；
- 先只创建配置，不启动训练。

**选择理由**

GPU 1/2/3 当时均有活动进程；共享 GPU 3 可能引入显存和吞吐干扰，终止其他任务也不符合当前协作约束。GPU 0 剩余显存足够完成单卡启动，因此先取得 H50/O50 的可运行性和中期曲线。

**对结果的影响**

单卡 global batch 从原 DDP2 的 32 降为 16。虽然 optimizer step 上限仍为 454000，但每步样本数减半，不能把该 run 解释为严格 matched-sample budget；后续正式比较应在空闲两卡上复跑或按样本数重新对齐。

**可逆性**

完全可逆；配置不改变旧 checkpoint，可停止后在 DDP2 重新开始。

**建议用户确认**

否；待 GPU 资源允许时自动改用 DDP2 复跑即可。
## 2026-08-24 — HOCap 首版按序列/物体实例/手拆分为外部 Stage 3

- scope: task:correspondence_ptv3_v2
- anchor: working tree / 2026-08-24

**未指定点**

用户要求直接开始 HOCap 转换验证，但未指定一个 HOCap 序列中多个物体实例如何映射到当前“一样本一个物体、一只手”的 Stage 3 契约。

**实际选择**

每个 HOCap 序列的每个 `object_id` 和每只有效手单独写一个 Stage 3 NPZ；物体使用 `cleaned_mesh_10000.obj` 的固定 4096 表面采样，手使用 HOCap 的 MANO PCA45、`flat_hand_mean=false` 和标准 1538 面拓扑；接触距离由 hand-root 坐标中的几何最近邻派生。HOCap 只作为外部测试数据，不加入当前 mixed 训练。

**选择理由与影响**

不把同一序列的多个物体错误拼成一个 correspondence 目标，并保留对象 ID 维度以便后续按物体宏平均；该接触目标是几何派生量，不应解释为 HOCap 人工接触标注。

**可逆性 / 是否需要用户确认**

输出使用独立 NAS 版本目录，完全可逆；后续可以改为按交互物体筛选或保留完整多物体场景，不覆盖原始 HOCap 数据。
