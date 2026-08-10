# Posetoken 研究日志

## 实验：PoseToken V1 synthetic MANO controlled overfit

**假设**
固定 morphology 的 root-local MANO surface 可以压缩为 64 个 local PoseTokens 和一个紧凑 global PoseToken，并分别重建局部与完整表面。

**观察到的失败 / 现象**
原始 Point MLP + max pool 在 400 steps 时 dense/global 只改善 37.7%/30.0%；提高到 2000 steps 和 lr `1e-3` 后 global 改善 95.7%，但 dense 只有 73.2%。4000 steps 时没有显式 atlas identity 的 dense 仍只有 77.0%，未达到 80% 验收线。

**诊断**
global bottleneck 容量足够；local decoder 共享于全部 patches，而 canonical identity 经 point MLP 和 max pool 后不易稳定保留。该问题影响局部重建，不是 root frame、单位或 correspondence 错误。

**改动**
使用 beta=0 的有界 local/wide/sparse MANO PCA pose mixture。canonical surface 固定构建 64×32 atlas；local token 加入 learnable patch atlas embedding。local decoder 重建 patch 32 点 deformation，128D global token 独立重建完整 1538 点 deformation。

**结果**
产物为 `outputs/posetoken/pose_token_20260810_204439`。32 poses、4000 steps、1:18。dense RMSE `0.1062 cm`，zero `0.8402 cm`，改善 `87.2%`；global RMSE `0.0248 cm`，zero `1.2345 cm`，改善 `98.0%`；local EPE `1.288 mm`。root SE(3) invariance 和 canonical patch identity 测试通过。

**决策**
保留 synthetic MANO、固定 atlas、local/global 双 decoder 和 atlas embedding；controlled overfit 通过。

## 实验：PoseToken V1 未见 synthetic pose 泛化

**假设**
在固定 beta=0 下，PoseToken 能泛化到独立随机种子的未见 MANO PCA poses，并同时超过 local/global canonical zero baseline。

**改动**
使用 1024/256/256 个独立 train/val/test poses，batch 64，lr `1e-3`，训练 100 epochs / 1600 steps。没有 object、trajectory、contact、augmentation 或额外正则。

**结果**
产物为 `outputs/posetoken/pose_token_20260810_204643`，耗时 0:39。最佳 checkpoint test dense RMSE `0.2204 cm`，zero `0.9435 cm`，改善 `76.6%`；global RMSE `0.4846 cm`，zero `1.4152 cm`，改善 `65.8%`；local EPE `2.381 mm`。global variance/cosine/L2 为 `0.739/0.062/8.891`，没有 collapse。

从 256 个 test poses 导出的二维 PCA 随第一个 MANO pose 分量连续变化，未观察到单点 collapse；encoder-only 权重和 PCA 产物分别导出到 `output/Posetoken/research/pose_encoder_v1.pt` 与 `global_pose_pca.{npz,png}`。

**决策**
V1 generalization 验收通过：未见 pose 的 local/global 两项均明显优于 zero。当前结论只覆盖右手、固定 morphology 的 synthetic MANO 分布。

**下一步**
V2 随机 beta，并以相同 theta、beta=0 surface 为 target，直接验证 morphology normalization；再做 GRAB/ARCTIC OOD 验证。

## 实验：V2 synthetic morphology normalization

**假设**
输入 `MANO(theta,beta)`、监督同姿态 `MANO(theta,0)`，可以保留 pose 区分并降低 PoseToken 对 morphology 的敏感性。

**观察到的失败 / 现象**
从随机初始化训练 32 个成对样本并加入 0.01 consistency 时，4000 steps 后 dense 只改善 77.4%，global 只改善 18.2%，global norm/cosine 达到 163/0.97，接近 collapse。V1 设计本来就是 V2 的预训练阶段，因此不应绕过 V1 随机开始。

**改动**
数据集增加 beta 标准差 0.75 的随机 morphology input 和 beta=0 target；同一 theta 可生成两个 beta 供 val/test 诊断。V2 从 V1 small best checkpoint 初始化，使用原 local/global reconstruction loss 微调 100 epochs / 1600 steps。尝试的 0.01 consistency 只让 global/local morphology ratio 从 13.30/17.21% 变为 12.96/16.99%，同时 global reconstruction 变差，已撤回。

**结果**
保留实验为 `outputs/posetoken/pose_token_20260810_205909`，耗时 0:41。test dense RMSE `0.1741 cm`、zero `0.9681 cm`，改善 82.0%；global RMSE `0.4840 cm`、zero `1.4502 cm`，改善 66.6%；EPE `2.109 mm`。输入 surface 的同姿态跨 beta / 不同姿态 RMSE 比为 20.5%；V2 global/local token ratio 为 13.3%/17.2%。

直接把未微调 V1 checkpoint 放到同一 V2 test 上，EPE 为 `3.402 mm`，global/local ratio 为 16.5%/16.5%。V2 明显改善 normalized surface reconstruction 和 global morphology ratio，但 local ratio 小幅变差，不能声称 local morphology invariance 同步提升。

**决策**
保留 V1 初始化 + beta=0 reconstruction target；撤回显式 consistency loss。V2 的 global normalization 通过，local normalization 结论不充分。

**下一步**
先在 GRAB/ARCTIC 上做真实姿态 OOD reconstruction；再针对 local token 比较 subject-neutral atlas 输入或从 token 中线性 probe beta，避免仅凭 synthetic pair ratio 下结论。
