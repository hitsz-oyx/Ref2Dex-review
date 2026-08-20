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
