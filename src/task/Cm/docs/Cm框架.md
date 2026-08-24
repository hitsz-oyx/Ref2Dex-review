# CmAction 当前框架

## 问题与信息边界

CmAction 从当前手—物状态与**未来手的相对运动**预测当前物体采样点的短时点流。一个样本以当前帧 `t` 的 hand-root 坐标系表示：

```text
(O_t, H_t, ΔH_t) → C_m → F̂_obj,t
```

其中 `ΔH_t = H_{t+s}^{hand_t} - H_t^{hand_t}`，而监督为
`F_obj,t = O_{t+s}^{hand_t} - O_t^{hand_t}`。未来物体点只用于在 Dataset
中构造 `obj_flow_gt`，绝不进入 DenseToken 或 Cm 模型。当前实现也**不输入**
`wrist_delta`、未来 hand token 或未来 object token。

缓存采用一份 `shared.npz`（物体状态）及每只手一份 `left.npz` / `right.npz`
（手状态）。训练按运行时 stride 采样；候选物点始终来自当前帧的
`obj_candidate_mask_5cm`，最多 512 点。`active_only=true` 仅过滤候选为空的
current frame。

## 模型

冻结 DenseToken/PTv3 只编码当前 `(O_t, H_t)`，输出 object token、hand token
及 hand contact 概率。它固定为 `eval/no_grad`，Cm checkpoint 不保存其权重。

每个手点的 Cm 输入是：

```text
[z_hand, hand_position × geometry_input_scale,
 hand_normal, hand_flow × hand_flow_input_scale, dense_hand_contact]
```

因此 hand motion MLP 的输入维度为 `dense_token_dim + 10`。Slot Attention 将
1538 个手点聚为 `K=16` 个 256 维 `C_m` slot，并由同一组 slot weights 产生
soft hand anchor 位置与法向。

每个 slot 通过 Hard-Concrete 的 nonzero probability 做 hard gate：训练时使用
概率参与 soft routing，评估时只保留超过 `slot_threshold` 的 slot；若全部低于
阈值，选择 logit 最大的一个 slot 作为 deterministic fallback。因此这是
**hard-gate Slot Attention**，不是 null-expert、Top-K anchor 或 activity head。

object decoder 对每个 object point 与每个保留 slot 构造
`[object_context, cm_token, object_to_anchor, anchor_normal]`，输出候选 flow 与
routing logit，再对 slot 加权求和。无效 object 位置的输出强制为零。

## 三个 scale 与训练目标

三个 scale 职责独立：

| 字段 | 用途 |
| --- | --- |
| `geometry_input_scale` | Cm head 内的 object/hand 坐标输入缩放 |
| `hand_flow_input_scale` | Cm head 内的 hand flow 输入缩放 |
| `object_flow_target_scale` | flow decoder target/loss 归一化 |

模型只暴露米制预测：`pred_obj_flow = pred_obj_flow_scaled / object_flow_target_scale`。
Runner 再将预测和 GT 同乘 `object_flow_target_scale`，以米制 `flow_smooth_l1_beta`
对应的 scaled vector Huber loss 训练。该乘除位于同一计算图，故不会缩小 decoder
梯度。旧 `internal_point_flow_scale` 仅作临时兼容并会告警；新配置不得使用它。

正式 scale 只能从 train split 通过 `python -m src.task.Cm.tools.data.compute_flow_scale`
计算。metadata 保存 stride、active-only、点数和权重规则；正式配置设置
`require_flow_calibration: true`，缺失这些 metadata 会拒绝启动。启动训练还会与
config 交叉检查，并验证 `flow_target_rms_m * object_flow_target_scale ≈ 1`。

## 评估与诊断

主质量指标是点加权 EPE：`flow/epe_mm`。每个 validation/test stride 先聚合所有
有效点的 residual、GT norm 与 prediction norm，再计算 `relative_epe` 和
`norm_ratio`；因此结果不依赖 batch 划分。checkpoint 选择使用
`val/mean_stride_epe_mm`。

每个 stride 的完整指标写入 JSONL；W&B 仅显示汇总和 stride 1/5/10 的 EPE，避免
曲线过多。P90 不再作为指标，因为 batch P90 的平均不是真实全局分位数。

硬门控诊断包括 `slot/expected_active_mean`、`slot/hard_active_mean`、
`slot/fallback_ratio`、`slot/effective_branch_count`、
`slot/global_top1_usage` 与 `slot/per_sample_top1_usage`。这些不是额外监督；唯一
的辅助损失是可选 L0 风格 slot count、置信度和 active-overlap 项。

Cm 仍只定义 hand-side 原因表征 `C_m`。未来的 `C_p` 应由 `C_m` 通过当前
correspondence 投影到 object side 形成，不能用未来物体几何反向构造 `C_m`。
