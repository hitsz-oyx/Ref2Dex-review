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
