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
