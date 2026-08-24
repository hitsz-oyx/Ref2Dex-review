# Cm 任务架构记录

- scope: task:Cm
- last_updated: 2026-08-23
- last_verified: 2026-08-23
- related: [当前状态](status_log.md)、[接手记忆](repo_memory.md)、[实验记录](experiment_log.md)、[V1.2.1 指导](../指导/V1.2.1.md)

## V1.2 主 Pipeline

```text
GRAB/ARCTIC Stage4（GRAB 必须使用 subject-specific v_template）
  → 4096-point object correspondence cache
  → ragged 5cm candidate + uint32 B=4 sampling bank
  → sequence-disjoint object-v2 Dataset
  → current object/hand → Frozen DenseToken
  → hard-gated Slot Attention (K=16) → C_m
  → object point-flow decoder → weighted EPE evaluator
```

单个 sample 始终只有一侧 1538 个 hand points，不拼双手。未来 hand flow 进入 action branch；未来 object 只用于构造 `obj_flow_gt`，禁止 future leakage。模型和 loss 不区分 GRAB/ARCTIC，`dataset_id` 只用于分数据集统计。

V1.2.1 在这条主线上的实现修正包括：sequence 级固定 split、train-only flow calibration、`_MmapSequenceDataset` 的共享 epoch、`_cache` 的 LRU 打开上限，以及联合 `grab/` + `arctic/` root 的统一识别；随后又补上了 `grab/` 与 `arctic/` 各自单独 root 的同口径对照配置。模型语义仍保持 no-gate + time condition，`use_slot_gate=false`，`use_time_condition=true`，不引入新的输入模态。

当前有效的新版 mixed 数据入口使用轻量链接 cache：新根目录保留真实的 `grab/<subject>/<sequence>/` 与 `arctic/<subject>/<sequence>/` 层级，每条 sequence 的 `shared/left/right` 目录分别软链接到修复版 GRAB 和既有 ARCTIC。这样既不复制约 115G 数据，也保持 `_sequence_dirs()`、split 路径边界检查和 mmap Dataset 的既有合同。旧 `cm_object_v2` 仍供已经启动的历史 run 读取，但其中 GRAB 使用过平均 MANO template，不得作为新版正式训练数据。

## 当前模型张量规格

以下规格对应当前 `src/task/Cm/densetoken_ckpt/best.pt`（DenseToken 维度 `D=96`）和默认 `num_cm_tokens=K=16`。`C` 表示 Cm 隐维度：主 baseline 使用 `C=256`，历史 GRAB gate 候选使用 `C=64`，geometry-only object ablation 使用 `C=32`；`B` 表示训练 batch size，`N_o=512`，`N_h=1538`。

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

`use_object_context` 是物体侧 DenseToken/context 的显式兼容开关。`true` 保留历史路径：

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

无效 object 点的最终预测强制为零。`C=256` 时历史 context edge 输入维度为 518、feature 维度为 128；`C=64` 时分别为 134、32。

当 `use_object_context=false` 时，`z_obj` 和 `object_context_encoder` 完全不进入 object-flow decoder。物体侧只保留原始几何与 Cm-relative 几何：

```text
[obj_points(3), obj_normals(3), cm_token(C), obj_to_anchor(3), anchor_normal(3)]
→ [B, N_o, K, C+12]
→ edge_backbone
```

此路径仍允许 object point 通过坐标和法向被定位，但当前交互 context 必须经 hand-side `z_hand/dense_hand_contact → C_m` 瓶颈传递；改变 `z_obj` 不得改变预测。`C=32` 时 edge 输入维度为 44。`use_time_condition=false` 时 `delta_time_s` 可以保留在数据兼容字段中，但既不加到 `cm_tokens`，也不进入 object-flow decoder。

### 5. Loss / 诊断张量

Runner 对有效 object 点计算 scaled vector Smooth-L1 flow loss；slot 辅助项包括 L0 风格 count loss、max-probability confidence loss，以及可选 active-overlap loss。`decoder_slot_usage` 为 `[B,K]`，用于统计 `effective_branch_count`、top-1 usage 和 slot 熵；这些诊断量默认不自动成为监督目标。

## 数据与评估合同

输入为 `obj_points/obj_normals [512,3]`、`hand_points/hand_normals [1538,3]`、`hand_flow [1538,3]`、valid mask 和 `delta_time_s`；输出为米制 `pred_obj_flow [512,3]`。stride 为 1--10，train/val/test 按 sequence 互斥。主指标为全 loader 点加权 EPE，checkpoint 使用 `val/mean_stride_epe_mm`；另报 zero-flow、action shuffle/reverse 和 slot gate 诊断。

固定 split 通过 `data.split_json_path` 指向的 `splits.json` 生成，训练和校准都只看 train split；这是为了让 V1.2.1 的对照实验不混入序列级泄漏。联合 root 的 metadata 在 `CmObjectV2Dataset` 读取时自动合并到 loader 返回的元信息里。

训练 stride 从整数区间 `1..10` 中确定性采样；`val_strides=[1,5,10]` 只是验证报告使用的三个固定切片，不代表训练只使用三种 stride。

DexYCB 跨数据集评估复用同一 Stage4 sequence schema。其序列 world frame 是 identity-extrinsic reference/master camera，`pose_y` 为 `xyzw + translation`，`pose_m` 与物体位于同一 frame。`pose_m[3:48]` 先按完整 MANO PCA basis 展开为 45 维 axis-angle，MANO 使用 `flat_hand_mean=false`。只支持右手 capture；全零 MANO 前缀不写入 cache，内部缺标注禁止压缩。

## 当前数据路径

| 角色 | 路径 | 状态 / 语义 |
| --- | --- | --- |
| 新版 mixed 逻辑 cache | `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820` | 当前正确入口；修复版 GRAB + 既有 ARCTIC，链接树本身约 24M |
| 新版 GRAB 实体 cache | `data/processed_data/cm_object_v2_subject_template_20260820/grab` | 1335 sequences / 2670 hand streams；subject-specific template |
| 复用 ARCTIC 实体 cache | `data/processed_data/cm_object_v2/arctic` | 301 sequences；本次无需重建 |
| 旧 mixed cache | `data/processed_data/cm_object_v2` | 历史 run 仍在读取；旧 GRAB 受平均 MANO template 问题影响 |
| 新版 mixed split | `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/splits_seed42/splits.json` | sequence 级、按 dataset 分层，train/val/test=`1308/164/164` |
| 新版 mixed calibration | `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/metadata.json` | seed42 train-only、stride 1--10 等权，RMS=`0.07328625889337191 m` |
| 新版 mixed E1 statistics | `data/processed_data/cm_object_v2_grab_arctic_subject_template_20260820/object_v2_statistics.json` | GRAB=`327798`、ARCTIC=`334248` samples |
| 新版训练配置 | `src/task/Cm/configs/object_v2_grab_arctic_subject_template_20260820.yaml` | C=256 正式训练入口，训练中 |
| DexYCB Stage4 cache | `data/processed_data/stage4/data/dexycb` | 修复版 subject-10/right；50 sequences / 2853 frames / active-frame ratio 80.30% |
| DexYCB 正式评测 split | `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json` | test 含全部 50 条 DexYCB；train 仅为 loader 所需的 1 条 GRAB 占位，不参与评测 |
| DexYCB C=256 评测配置 | `src/task/Cm/configs/eval_dexycb_subject10_c256_fixed_20260822.yaml` | 固定 stride 1/5/10、batch 32、沿用训练 checkpoint scale |

## Scale 与缓存

`geometry_input_scale`、`hand_flow_input_scale`、`object_flow_target_scale` 职责分离；正式 scale 仅由 train split calibration 产生并在启动时校验。新版 mixed 配置的 `flow_target_rms_m=0.07328625889337191`、`object_flow_target_scale=13.645122770626802`，必须与同根 `metadata.json` 精确一致。V1.1.2 的 geometry/candidate/sampling cache 保留，约 517GB 的全量 DenseToken bank 已撤回，正式路径在线 Frozen DenseToken。

## HRDexDB fine-tune 扩展

`dataset_hrdexdb.py` 读取统一的 `cmdecoder_layered_v4` geometry cache：`hand_points_world [T,1538,3]`、稳定 `obj_points_pool_world [T,4096,3]`、对应 normals、每帧 `obj_candidate_mask_5cm [T,4096]`、`wrist_pose_world [T,4,4]`、`frame_time` 和可选 `source_frame_id`。每个样本在线随机选择 stride `1..10`，先从当前手的 5cm candidate 中采样 512 个物点，再把当前/未来端点都变换到当前 `hand_root_t`，生成 `hand_flow`、`obj_flow_gt` 与真实 `delta_time_s`；MANO 与各机器人手型必须在 geometry 构建阶段归一化到同一 1538 点合同。旧的仅有 512 个全表面物点、没有 candidate mask 的 smoke cache 不满足 Cm fine-tune 合同。`embodiment_id` 只作诊断字段，HRDexDB 内部不再额外分层。

该 geometry cache 不包含 DenseToken 特征。HRDexDB fine-tune 时 `data.use_dense_cache` 必须为 `false`，模型在线执行 DenseToken；因此即使 `meta.freeze_dense_encoder=false`，梯度仍能从 Cm loss 回到 DenseToken。任何包含 `cached_z_obj/cached_z_hand/cached_hand_contact` 的 dense feature bank 都禁止用于该阶段。

`BalancedCmMixture` 将 GRAB、ARCTIC、HRDexDB 三个外部 source bucket 按 `1/3,1/3,1/3` 概率抽样，保持 Cm 的 object-flow 语义；验证和测试按 `source × stride∈{1,5,10}` 建立独立 loader，并额外汇总 `val/mean_stride_epe_mm` 供 checkpoint 选择。微调固定沿用 base 的 `object_flow_target_scale`，不重新 calibration；该 scale 只作为 loss 数值归一化，不改变米制指标。`configs/hrdexdb_finetune_cm64.yaml` 是方案 A 的入口：从 base C=64 checkpoint 恢复后将 `meta.freeze_dense_encoder=false`，DenseToken 与 Cm head 联合训练。冻结阶段 checkpoint 不保存 DenseToken；可训练阶段 checkpoint 保存该前缀以支持恢复。
