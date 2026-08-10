# Actiontoken 研究日志

## 实验：ActionToken V1 controlled overfit

**假设**
Frozen PoseToken pair 可以编码完整局部 articulation motion，并由 `[B,8,64,384]` ActionToken 重建每 patch 32 点 adjacent flow。

**数据**
32 条 synthetic MANO trajectories，每条 9 帧，beta=0，root 固定。PCA pose 速度使用 decay 0.9、noise std 0.035 的平滑过程；稳定使用 canonical 64×32 atlas。

**观察到的失败 / 现象**
首版普通 pair MLP + temporal Transformer 在 4000 steps 的 train/val 改善为 60.5%/约 66%，static RMSE `0.0142 cm`、reverse residual `0.0283 cm`，方向语义没有学稳。加入 encoder 反对称后 train/val 提升到 77.7%/74.1%，但 flow decoder bias 仍破坏 static/reverse。

**改动**
PoseEncoder 完全冻结；DynamicActionEncoder 只读取相邻 local/global PoseTokens。pair MLP、temporal Transformer 和 flow decoder 均取 forward/reverse 的反对称分量，从结构上保证 static flow 为零且 sequence swap 变号。未增加 antisymmetry loss。

**结果**
保留实验为 `outputs/actiontoken/action_token_20260810_213438`。使用 V2 PoseEncoder 训练 6000 steps / 5:48，最佳 step 5808 的 val RMSE 为 `0.00865 cm`，zero `0.03576 cm`，改善 `75.1%`，EPE `0.0991 mm`；slow/medium/fast RMSE 为 `0.00261/0.00999/0.02378 cm`。static/reverse residual 均为 0。最后 train epoch 改善约 76%，未达到 80% controlled-overfit 门槛。

使用 V1 fixed-morphology encoder 的对照为 `outputs/actiontoken/action_token_20260810_214059`，4000 steps / 3:55，最佳 val 改善 `74.0%`，没有优于 V2 encoder。

**诊断**
坐标、单位、稳定 correspondence 和动作方向约束均已通过；绝对 EPE 已很小，但 zero baseline 也只有 `0.0358 cm`。延长 4000→6000 steps 与切换 V1/V2 PoseEncoder 都没有跨过 80%，剩余问题更可能是 frozen static PoseToken pair 对 patch 内 dense flow 的可读上限，或 temporal readout 对局部细节的损失。

**决策**
保留独立 task、smooth generator、冻结 PoseEncoder、dense decoder 和解析反对称结构；判定 ActionToken V1 controlled-overfit 尚未通过。按指导停止 full generalization、RootToken 和 InteractionDynamics 回接。

**下一步**
冻结相邻 PoseToken features 做无 temporal 的 local-flow probe，并与直接 surface-pair oracle 对齐，定位信息瓶颈发生在 PoseEncoder 还是 DynamicActionEncoder。

## 实验：V1.1 PoseToken 与时序读出归因对照

**假设**
原完整模型约 75% 的改善上限可能来自 frozen PoseToken 信息损失，也可能来自 temporal Transformer 对单 transition 局部细节的破坏。用同一数据分别移除 temporal、增强 pair readout、直接读取 surface 和解码两帧后作差可以区分二者。

**改动**
新增 `probes.py`，固定随机种子 42，在与 V1 相同的 32 条、9 帧 synthetic MANO trajectories 上运行四组对照。A 使用 local/global PoseToken pair 和两层 MLP，不经过 temporal；B 只使用 local PoseToken pair 和更宽 MLP；C 使用真实 surface pair 作为 oracle；D 使用完整 PoseToken checkpoint 分别重建两帧形变后直接作差。A–C 保持解析奇对称读出。

**结果**
A 训练 4000 steps，RMSE `0.00199 cm`、EPE `0.02772 mm`、相对 zero 改善 `94.54%`。B 训练 4000 steps，RMSE `0.00259 cm`、EPE `0.03192 mm`、改善 `92.90%`。C 在 4000 steps 改善 `90.00%`，延长到 20000 steps 后 RMSE `0.000824 cm`、EPE `0.01118 mm`、改善 `97.74%`，说明 4000-step oracle 尚未充分优化。D 不训练，RMSE `0.00637 cm`、EPE `0.07304 mm`、改善 `82.55%`。作为比较，原 temporal ActionToken 最佳改善为 `75.14%`。

**诊断**
只读取 PoseToken pair 的 A/B 均显著超过原模型，并接近充分训练的 surface oracle；直接 PoseToken 解码差分也已超过 80%。因此 PoseToken 压缩不是当前主要信息瓶颈，约 18–20 个百分点的差距主要由 temporal readout 路径引入。C 的短程结果不能作为信息上界，延长训练后才接近预期 oracle。

**决策**
保留 frozen PoseToken、pair readout 和解析反对称约束；下一实验优先把 temporal 分支改为 pair flow 上的小残差修正，避免覆盖已被 A/B 证明可读的局部运动。暂不扩大数据、不做 RootToken、不回接 InteractionDynamics。

## 实验：ActionToken V2 纯局部 point-flow 表示

**假设**
稳定 correspondence 的 root-local dense surface flow 可以由不含 PoseToken 和 temporal context 的 `64×384` ActionToken 压缩，并在未见 pose pairs 上保留局部运动；32-pair controlled overfit 应超过 90% zero-baseline improvement。

**观察到的失败 / 现象**
V1 temporal 模型只改善约 75%，V1.1 已证明去除时序读出后 PoseToken pair 可达到 92.9%–94.5%。因此继续修补 trajectory encoder 会混淆动作与状态，改为直接验证纯 motion bottleneck。

**诊断**
MANO 1538 个 face centers 具有稳定 correspondence，动作点流可由两帧各自在 wrist frame 下的 surface 解析相减获得；网络无需重新学习 correspondence 或 subtraction。

**改动**
保留全部 V1 路径，新增 independent pose-pair generator、small/medium/large 与 sparse PCA motion mixture、`FlowActionEncoder` 和 V2 Runner。每 patch 的 32×3 cm flow 经 `96→256→384` odd encoder，再由 odd decoder 重建；唯一 loss 为 dense flow MSE。新增 overfit/small 配置和 V2 encoder-only 导出 schema。

**结果**
32 transitions 实验 `outputs/actiontoken/action_token_20260810_232906` 训练 4000 steps / 44 秒，最终 val RMSE `0.00190 cm`、zero `0.04376 cm`、改善 `94.85%`、EPE `0.0192 mm`。4096/1024/1024 实验 `outputs/actiontoken/action_token_20260810_233002` 训练 3200 steps / 27 秒；独立 test seed 的 RMSE `0.003059 cm`、zero `0.065341 cm`、改善 `95.286%`、EPE `0.02757 mm`，slow/medium/fast RMSE 为 `0.000878/0.003193/0.008553 cm`。两组 static/reverse token 与 flow 残差均为 0。

**决策**
保留 V2，纯局部 motion compression 假设通过 controlled overfit 和 unseen-pair generalization。导出 `output/Actiontoken/research/action_encoder_v2_flow.pt`，不提交权重；不再给 ActionEncoder 增加 Transformer、Pose conditioning 或 interaction 信息。

**下一步**
大规模预训练前改为 parameter-only、batch 内 MANO forward；随后把 RootToken 作为独立模块，并用 V2 checkpoint 重做 InteractionDynamics articulation intervention。
