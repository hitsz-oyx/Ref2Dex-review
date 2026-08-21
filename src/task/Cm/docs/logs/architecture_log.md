# Cm 任务架构记录

## V1.2 主 Pipeline

```text
GRAB/ARCTIC Stage4
  → 4096-point object correspondence cache
  → ragged 5cm candidate + uint32 B=4 sampling bank
  → sequence-disjoint object-v2 Dataset
  → current object/hand → Frozen DenseToken
  → hard-gated Slot Attention (K=16) → C_m
  → object point-flow decoder → weighted EPE evaluator
```

单个 sample 始终只有一侧 1538 个 hand points，不拼双手。未来 hand flow 进入 action branch；未来 object 只用于构造 `obj_flow_gt`，禁止 future leakage。模型和 loss 不区分 GRAB/ARCTIC，`dataset_id` 只用于分数据集统计。

V1.2.1 在这条主线上的实现修正包括：sequence 级固定 split、train-only flow calibration、`_MmapSequenceDataset` 的共享 epoch、`_cache` 的 LRU 打开上限，以及联合 `grab/` + `arctic/` root 的统一识别；随后又补上了 `grab/` 与 `arctic/` 各自单独 root 的同口径对照配置。模型语义仍保持 no-gate + time condition，`use_slot_gate=false`，`use_time_condition=true`，不引入新的输入模态。

## 当前模型张量规格

以下规格对应当前 `src/task/Cm/densetoken_ckpt/best.pt`（DenseToken 维度 `D=96`）和默认 `num_cm_tokens=K=16`。`C` 表示 Cm 隐维度：主 baseline 使用 `C=256`，GRAB gate+cm64 候选使用 `C=64`；`B` 表示训练 batch size，`N_o=512`，`N_h=1538`。

### 1. Dataset / DenseToken 输入

| 张量 | 形状 | 语义 / 单位 |
| --- | --- | --- |
| `obj_points` | `[B,N_o,3]` | 当前物体点，hand-root 局部坐标，m |
| `obj_normals` | `[B,N_o,3]` | 当前物体法向 |
| `hand_points` | `[B,N_h,3]` | 当前单侧手点，hand-root 局部坐标，m |
| `hand_normals` | `[B,N_h,3]` | 当前单侧手法向 |
| `hand_flow` | `[B,N_h,3]` | 未来手点相对当前帧的 flow，仍在当前 hand-root frame，m |
| `obj_flow_gt` | `[B,N_o,3]` | 未来物体点监督，只有 Dataset 构造 GT 时使用，m |
| `obj_valid_mask` | `[B,N_o]` | 当前 5 cm candidate 中的有效采样点 |
| `delta_time_s` | `[B]` | `stride / 30`，秒 |

DenseToken 只接收当前帧几何：将 object 与 hand 拼为 `[B,N_o+N_h,3]=[B,2050,3]`，对应 normals 和 valid mask 也是 `[B,2050,3]`、`[B,2050]`。冻结 PTv3 输出：

```text
z_obj             [B, N_o, D] = [B, 512, 96]
z_hand            [B, N_h, D] = [B, 1538, 96]
dense_hand_contact[B, N_h]    = [B, 1538]
```

DenseToken 在 `eval/no_grad` 下运行；`hand_flow`、未来 object 几何和 `obj_flow_gt` 不进入该模块。

### 2. Hand action branch / Slot Attention

每个 hand point 的输入拼接为：

```text
[z_hand(D), hand_points(3), hand_normals(3), hand_flow(3), contact(1)]
→ [B, N_h, D+10] = [B, 1538, 106]
→ hand_motion_encoder
→ u_hand [B, N_h, C]
```

Slot Attention 使用 `K=16` 个可学习 slot，迭代 `slot_iters=3` 次：

```text
slot_init                         [B, K, C]
attention logits                  [B, K, N_h]
cm_assignment                     [B, K, N_h]  # 沿 K softmax，每个 hand point 的 slot 分配和为 1
cm_slot_weights                   [B, K, N_h]  # 每个 slot 再沿 N_h 归一化，用于 hand anchor
cm_tokens / C_m                   [B, K, C]
cm_anchor_pos                     [B, K, 3]
cm_anchor_normal                  [B, K, 3]
```

当 `use_time_condition=true` 时，`delta_time_s` 经 `MLP(1→C)` 得到 `[B,C]`，加到每个 `cm_tokens` 和 object context；它不改变 slot 数量或 gate 结构。

### 3. Gate / routing

```text
slot_gate_head(cm_tokens)         [B, K, C] → [B, K, 1] → slot_gate_logits [B, K]
Hard-Concrete nonzero probability  [B, K]
hard_slot_mask                    [B, K] bool
```

`slot_gate_head` 的参数在 K 个 slot 间共享，输入仍是每个 slot 自己的 `cm_token`。当前 `CmFlowHead` 直接使用 Hard-Concrete 的 analytic nonzero probability 和确定性阈值；没有调用随机采样分支。训练时使用 soft routing 与 hard routing 的 straight-through 组合；评估时使用同一概率的确定性阈值。若所有 slot 都低于阈值，则按最大 gate logit 保留一个 fallback slot。`gate_warmup_enabled` 时，前 5 个 epoch 强制全 slot，随后 5 个 epoch 将 threshold 从 0 ramp 到配置值（默认 0.85），并同步 ramp slot count loss 权重。

### 4. Object flow decoder

object context 的输入为：

```text
[z_obj(D), obj_points(3), obj_normals(3)]
→ [B, N_o, D+6] = [B, 512, 102]
→ object_context_encoder
→ obj_context [B, N_o, C]
```

对每个 object point 和每个 slot 构造 edge 输入：

```text
[obj_context(C), cm_token(C), obj_to_anchor(3), anchor_normal(3)]
→ [B, N_o, K, 2C+6]
→ edge_backbone
→ edge_features [B, N_o, K, C/2]
```

两个 head 分别输出：

```text
dynamic_logits       [B, N_o, K]
dynamic_candidate_flow_scaled [B, N_o, K, 3]
edge_weight          [B, N_o, K]  # 沿 K 归一化
pred_obj_flow_scaled [B, N_o, 3]  # 沿 K 加权求和
pred_obj_flow        [B, N_o, 3]  # 除以 object_flow_target_scale，恢复米制
```

无效 object 点的最终预测强制为零。`C=256` 时 edge 输入维度为 518、feature 维度为 128；`C=64` 时分别为 134、32。

### 5. Loss / 诊断张量

Runner 对有效 object 点计算 scaled vector Smooth-L1 flow loss；slot 辅助项包括 L0 风格 count loss、max-probability confidence loss，以及可选 active-overlap loss。`decoder_slot_usage` 为 `[B,K]`，用于统计 `effective_branch_count`、top-1 usage 和 slot 熵；这些诊断量默认不自动成为监督目标。

## 数据与评估合同

输入为 `obj_points/obj_normals [512,3]`、`hand_points/hand_normals [1538,3]`、`hand_flow [1538,3]`、valid mask 和 `delta_time_s`；输出为米制 `pred_obj_flow [512,3]`。stride 为 1--10，train/val/test 按 sequence 互斥。主指标为全 loader 点加权 EPE，checkpoint 使用 `val/mean_stride_epe_mm`；另报 zero-flow、action shuffle/reverse 和 slot gate 诊断。

固定 split 通过 `data.split_json_path` 指向的 `splits.json` 生成，训练和校准都只看 train split；这是为了让 V1.2.1 的对照实验不混入序列级泄漏。联合 root 的 metadata 在 `CmObjectV2Dataset` 读取时自动合并到 loader 返回的元信息里。

## Scale 与缓存

`geometry_input_scale`、`hand_flow_input_scale`、`object_flow_target_scale` 职责分离；正式 scale 仅由 train split calibration 产生并在启动时校验。V1.1.2 的 geometry/candidate/sampling cache 保留，约 517GB 的全量 DenseToken bank 已撤回，正式路径在线 Frozen DenseToken。
