# corresponse_v1: Static Hand-Object Correspondence

本文档描述 `corresponse_v1` 任务（Static Hand-Object Correspondence）的完整张量流向、数据约定、模型结构、损失函数与训练/评估流程。

---

## 1. 任务目标

输入单帧单手的统一点云（`No_train + Nh_train`），预测：

| 输出 | 形状 | 含义 |
|------|------|------|
| `pred_obj_contact` | `[B, No_train]` | 每个 object 点的接触概率 |
| `pred_obj_to_hand_cano` | `[B, No_train, 3]` | 每个 object 点对应的 canonical hand 位置 |
| `pred_obj_to_hand_finger` | `[B, No_train, num_fingers]` | 对应 hand finger 分类 logits |
| `pred_obj_to_hand_region` | `[B, No_train, num_regions]` | 对应 hand region 分类 logits |
| `pred_cross_contact` | `[B, Nh_train, K_cross]` | 每条 hand-to-object cross edge 的接触概率 |

默认：
- `No_train = 512`，`Nh_train = 1538`，`N_total = 2050`
- `K_obj_local = 16`，`K_hand_local = 16`，`K_cross = 32`
- `hidden_dim = 128`，`num_local_layers = 3`，`num_cross_layers = 2`

---

## 2. 数据约定（Stage 3 训练样本）

字段顺序、坐标系约定严格遵循 `docs/架构.md` 5.4 节。

### 2.1 输入字段

| 字段 | 形状 | 含义 |
|------|------|------|
| `points` | `[N_total, 3]` | object frame 下的统一点云，前 `No_train` 是 object，后面是 hand |
| `normals` | `[N_total, 3]` | object frame 下的法向 |
| `point_type_id` | `[N_total]` | 0 = object, 1 = hand |
| `point_valid_mask` | `[N_total]` | object padding 点为 False |
| `finger_id` | `[N_total]` | hand 手指 ID（0=palm, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky），object 点为 -1 |
| `hand_region_id` | `[N_total]` | hand 局部区域 ID（0=palm, 1=fingertip, 2=finger_pad, 3=finger_side, 4=finger_back, 5=hand_back），object 点为 -1 |
| `hand_cano_points` | `[N_total, 3]` | canonical MANO face-center 坐标，object 点为 0 |
| `obj_local_knn_idx` | `[No_train, K_obj_local]` | object-object KNN 邻居 id |
| `obj_local_knn_valid_mask` | `[No_train, K_obj_local]` | object KNN 有效 mask |
| `hand_local_knn_idx` | `[Nh_train, K_hand_local]` | hand-hand KNN 邻居 id |
| `hand_local_knn_valid_mask` | `[Nh_train, K_hand_local]` | hand KNN 有效 mask |
| `hand_to_obj_knn_idx` | `[Nh_train, K_cross]` | hand-to-object KNN 邻居 id |
| `hand_to_obj_knn_valid_mask` | `[Nh_train, K_cross]` | cross KNN 有效 mask |

### 2.2 监督字段

| 字段 | 形状 | 含义 |
|------|------|------|
| `obj_contact_label` | `[No_train]` | soft contact label ∈ [0, 1] |
| `obj_to_hand_cano_points` | `[No_train, 3]` | canonical hand 对应坐标 |
| `obj_to_hand_finger_id` | `[No_train]` | 对应 finger id，padding 点为 -1 |
| `obj_to_hand_region_id` | `[No_train]` | 对应 region id，padding 点为 -1 |
| `obj_label_valid_mask` | `[No_train]` | object label 有效 mask |

> 注意：所有 KNN 字段只保存 id，几何量（delta、dist、signed_dist、normal_dot）在 forward 中实时计算。

---

## 3. 整体张量流向

```
[Dataset / .npz]
    │
    ▼  (可选：rigid augmentation)
[Batch Dict]  ── B, N_total=2050
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. Input Embedding                                            │
│    - Object Embedding: xyz(3)+normal(3)+type_emb(16) → 128    │
│    - Hand Embedding:   xyz(3)+normal(3)+type_emb(16)         │
│                        +finger_emb(16)+region_emb(16)        │
│                        +cano_xyz(3) → 128                     │
└──────────────────────────────────────────────────────────────┘
    │
    │  z_obj: [B, No, 128]       z_hand: [B, Nh, 128]
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Local Encoders (×3 EdgeConv)                               │
│    message_ij = MLP([z_i, z_j − z_i, rel_xyz_ij])            │
│    z_i = z_i + mean_aggregate(message_ij)                    │
│                                                              │
│    - Object Local: KNN = obj_local_knn_idx (K=16)            │
│    - Hand Local:   KNN = hand_local_knn_idx (K=16)           │
└──────────────────────────────────────────────────────────────┘
    │
    │  z_obj: [B, No, 128]       z_hand: [B, Nh, 128]
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Cross Interaction (×2 Hand→Object)                         │
│    实时计算每条 cross edge 的几何量:                           │
│      delta = hand_xyz − obj_xyz             [Nh, K, 3]        │
│      dist  = ||delta||                      [Nh, K, 1]        │
│      signed_dist = dot(obj_normal, delta)   [Nh, K, 1]        │
│      normal_dot = dot(obj_normal, hand_normal) [Nh, K, 1]    │
│                                                              │
│    msg_ji = MLP([z_obj_i, z_hand_j, edge_geo])                │
│    z_obj_i = z_obj_i + scatter_mean(msg_ji)                   │
│    （只更新 object token，hand token 不更新）                  │
└──────────────────────────────────────────────────────────────┘
    │
    │  z_obj: [B, No, 128]
    ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Output Heads                                               │
│    - contact_head:  Linear(128→64→1)  → pred_obj_contact     │
│    - cano_head:     Linear(128→64→3)  → pred_obj_to_hand_cano│
│    - finger_head:   Linear(128→64→6)  → pred_obj_to_hand_finger│
│    - region_head:   Linear(128→64→6)  → pred_obj_to_hand_region│
│    - cross_edge_head: Linear(128*2+6→64→1)                   │
│        (z_hand, z_obj_neighbors, edge_geo) → pred_cross_contact│
└──────────────────────────────────────────────────────────────┘
    │
    ▼
[5 个 pred 字典]
```

---

## 4. Forward 详细张量形状

设 batch 维度为 `B`，以下省略 B。

### 4.1 拆分

```python
# 来自 batch
points        : [B, N_total, 3] = [B, 2050, 3]
normals       : [B, N_total, 3]
finger_id     : [B, N_total]
hand_region_id: [B, N_total]
hand_cano     : [B, N_total, 3]

# 拆分
obj_points   = points[:, :No]                 # [B, 512, 3]
hand_points  = points[:, No:]                 # [B, 1538, 3]
obj_normals  = normals[:, :No]                # [B, 512, 3]
hand_normals = normals[:, No:]                # [B, 1538, 3]

hand_finger_id  = finger_id[:, No:]           # [B, 1538]
hand_region_id  = hand_region_id[:, No:]      # [B, 1538]
hand_cano_only  = hand_cano[:, No:]           # [B, 1538, 3]
```

### 4.2 Input Embedding

```python
# Object
obj_input = [obj_points, obj_normals, type_emb(obj)]   # [B, 512, 22]
z_obj = obj_embed_mlp(obj_input)                        # [B, 512, 128]

# Hand
hand_input = [hand_points, hand_normals, type_emb(hand),
              finger_emb, region_emb, hand_cano_only]  # [B, 1538, 57]
z_hand = hand_embed_mlp(hand_input)                     # [B, 1538, 128]
```

### 4.3 Local Encoders (×3)

每一层对每个 query 点 `i`：

```python
# 1. gather neighbors
z_neigh = z[knn_idx]                                   # [N, K, D]
rel_xyz = points[knn_idx] - points                      # [N, K, 3]

# 2. edge MLP
edge_in = [z_query, z_neigh − z_query, rel_xyz]        # [N, K, D*2+3]
msg = edge_mlp(edge_in)                                # [N, K, D]

# 3. mean aggregate over valid neighbors
z = z + mean(msg, dim=K)
```

输出：
- `z_obj: [B, 512, 128]`
- `z_hand: [B, 1538, 128]`

### 4.4 Cross Interaction (×2)

对每条 `(hand j, obj i)` cross edge：

```python
# 1. gather object neighbor features
z_obj_neigh = z_obj[hand_to_obj_knn_idx]              # [B, 1538, 32, 128]

# 2. compute geometric features (实时)
delta       = hand_points - obj_points[knn_idx]        # [B, 1538, 32, 3]
dist        = norm(delta)                              # [B, 1538, 32, 1]
signed_dist = dot(obj_normals[knn_idx], delta)         # [B, 1538, 32, 1]
normal_dot  = dot(obj_normals[knn_idx], hand_normals)  # [B, 1538, 32, 1]

# 3. message
edge_in = [z_obj_neigh, z_hand_expanded, edge_geo]    # [B, 1538, 32, D*2+6]
msg = cross_mlp(edge_in)                               # [B, 1538, 32, D]

# 4. scatter mean back to object
z_obj[i] = z_obj[i] + mean over j that connects to i
```

输出：`z_obj: [B, 512, 128]`（hand token 保持不变）

### 4.5 Output Heads

```python
pred_obj_contact         = contact_head(z_obj)               # [B, 512]
pred_obj_to_hand_cano    = cano_head(z_obj)                  # [B, 512, 3]
pred_obj_to_hand_finger  = finger_head(z_obj)                # [B, 512, 6]
pred_obj_to_hand_region  = region_head(z_obj)                # [B, 512, 6]
pred_cross_contact       = cross_edge_head(...)              # [B, 1538, 32]
```

---

## 5. 损失函数

`runner.py::_compute_losses` 计算 5 个加权 loss：

| Loss | 公式 | 监督 | 权重 |
|------|------|------|------|
| `obj_contact` | `BCEWithLogits(pred, label, weight=valid_mask)` | soft contact | 1.0 |
| `obj_cano` | `SmoothL1(pred, target).sum(-1) * valid` | canonical hand 坐标 | 5.0 |
| `obj_finger` | `CrossEntropy(pred, target) * valid` | finger id | 0.25 |
| `obj_region` | `CrossEntropy(pred, target) * valid` | region id | 0.25 |
| `cross_edge_contact` | `BCEWithLogits(pred, dynamic_label, weight=valid_mask)` | 动态生成的 soft edge label | 1.0 |

**动态 edge label**（在 `_compute_dynamic_edge_labels` 中）：

对每条 cross edge `(hand j, obj i)`，根据当前距离 `d = ||hand_xyz[j] − obj_xyz[i]||`：

```
if d ≤ d_pos:   label = 1.0
elif d < d_neg: label = (1 − (d − d_pos)/(d_neg − d_pos)) ** gamma
else:           label = 0.0
```

默认：`d_pos=0.005, d_neg=0.03, gamma=2.0`。

总 loss：

```
loss = 1.0 * obj_contact
     + 5.0 * obj_cano
     + 0.25 * obj_finger
     + 0.25 * obj_region
     + 1.0 * cross_edge_contact
```

---

## 6. 数据增强

仅允许刚体变换（KNN id 在增强后仍有效）：

1. 随机旋转：绕随机轴 `axis ~ U(S^2)`，角度 `θ ~ U(0, rotation_range)`，Rodrigues 公式
2. 随机平移：每维 `t_i ~ U(-translation_range, +translation_range)`
3. 均匀缩放：`s ~ U(scale_min, scale_max)`

> 注意：hand_cano_points 也需要随旋转同步旋转（平移、缩放不影响 canonical 坐标）。

---

## 7. 训练 & 评估

### 7.1 训练

```bash
python -m src.task.corresponse_v1.train \
    --data processed_data/generated/train_corr_static/grab \
    --output-dir outputs/train/corresponse_v1 \
    --device cuda:0
```

可选 `--set` 覆盖：

```bash
python -m src.task.corresponse_v1.train \
    --set train.lr=1e-4 \
    --set train.epochs=200 \
    --set data.batch_size=4
```

### 7.2 评估

```bash
python -m src.task.corresponse_v1.eval \
    --checkpoint outputs/train/corresponse_v1/checkpoints/best.pt \
    --device cuda:0
```

---

## 8. 目录结构

```
src/task/corresponse_v1/
├── __init__.py        # 导出 CorrResponseRunner
├── config.py          # Config (TaskConfig): meta / model / data / train / wandb
├── model.py           # StaticHOCNet, EdgeConvBlock, CrossGraphBlock
├── runner.py          # CorrResponseRunner: step / _compute_losses / _compute_dynamic_edge_labels
├── dataset.py         # CorrStaticDataset + make_dataloaders
├── train.py           # 训练入口
├── eval.py            # 评估入口
└── README.md          # 本文件
```

`src/utils/correspondence.py` 提供：

- `gather_knn_features(features, knn_idx, knn_valid_mask)` — 通用 KNN gather
- `compute_local_edge_features(points, knn_idx, knn_valid_mask)` — 局部相对位移
- `compute_cross_edge_features(...)` — cross edge 几何量（delta, dist, signed_dist, normal_dot）
- `soft_contact_label(dist, d_pos, d_neg, gamma)` — soft contact label 生成
