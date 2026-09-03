# Ref2Dex 架构记录

- scope: root
- last_updated: 2026-08-25
- last_verified: 2026-08-25
- related: [接手记忆](repo_memory.md)、[修改记录](modification_log.md)、[项目总览](../项目总览.md)

## 1. 研究目标与总体架构

Ref2Dex 的目标不是直接把一套手的关节角回归成另一套手的关节角，而是先学习与手的形态相对解耦的“手—物交互动作”表示，再针对每一种目标手将该表示解码为目标手运动。整体分为三个串联但训练边界清晰的阶段：

```text
参考手/物体序列
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ correspondence_ptv3_v2                                      │
│ 当前手物几何 → PTv3 → z_obj / z_hand / contact prior         │
│ 训练目标：在物体位姿扰动下恢复 clean 接触/对应关系            │
└─────────────────────────────────────────────────────────────┘
      │ 冻结 DenseToken checkpoint
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Cm                                                          │
│ DenseToken + 当前手部几何 + endpoint hand_flow              │
│       → Hand Motion Encoder → Slot Attention → C_m           │
│       → Object Flow Head（训练 Cm 的辅助/主任务）             │
└─────────────────────────────────────────────────────────────┘
      │ 冻结 Cm checkpoint；在线或 mmap 提取 C_m
      ▼
┌─────────────────────────────────────────────────────────────┐
│ CmDecoder（每一种目标手单独一套配置、数据和 checkpoint）      │
│ C_m + 目标手当前状态 → 目标手 point flow 或 q/wrist motion    │
│       → 固定 correspondence + URDF FK + 可微拟合             │
│       → 目标手下一时刻关节和腕部姿态                          │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
目标手轨迹 / 重定向结果
```

共享的数据适配、坐标变换和时序 cache 位于 `process/`，通用训练和评估基础设施位于 `src/base/`。三个研究阶段的实现分别位于 `src/task/correspondence_ptv3_v2/`、`src/task/Cm/` 和 `src/task/CmDecoder/`。

前两阶段的表示不包含目标机器人的 URDF、关节角或 mimic-joint 规则；目标手的运动学只在 CmDecoder 阶段引入。因此，同一个 Cm checkpoint 可以作为多个目标手 decoder 的共同输入，但不同目标手的 decoder 不能直接互换。当前仓库已经打通 GRAB → Inspire F1 的无配对重定向 smoke test；该测试证明数据流和接口连通，不代表完整的重定向精度结论。

## 2. 全局坐标、时间和数据约定

### 2.1 坐标系

- 人手数据（correspondence 和 Cm）默认使用当前帧 `hand_root` 坐标系：

  ```text
  x_hand_root = R_hand_root^T (x_world - t_hand_root)
  ```

- OakInk 当前规范导出使用官方 MANO root quaternion 同时移除手、物体的 camera/world root rotation，并携带可供训练端重建的 axis-angle45 MANO 参数；旧版只减 wrist 的 camera-frame 数据不满足该跨数据集坐标合同。

- CmDecoder 的机器人 task cache 使用当前机器人 wrist 坐标系；目标点和 `hand_flow` 都表达在当前 wrist frame，避免把目标腕部刚体运动错误地从 flow 中消掉。
- 法向只做旋转，不施加平移；所有 flow、平移和 point loss 的单位为米，旋转向量和关节角的单位为弧度。

### 2.2 当前帧与未来帧

一个训练样本通常由 `(t, t+stride)` 构成。`hand_flow` 是固定点 correspondence 下的：

```text
hand_flow = hand_points_(t+stride) - hand_points_t
```

它在当前帧坐标系中表达。Cm 使用这个 endpoint flow 作为动作条件；未来物体几何和 `obj_flow_gt` 只用于 Cm 的监督，不能进入 DenseToken 或 Cm 的当前几何编码分支。

### 2.3 共同的点数和有效掩码

当前主 checkpoint 的合同为：完整物体点池 `4096`，运行时物体点 `512`，单侧手点 `1538`。物体候选不足 `512` 时用 padding 补齐，并由 `obj_valid_mask` 忽略无效位置；手点不做这种 padding。跨帧监督依赖稳定的点顺序/面采样，否则 point flow 和后续 FK 拟合没有明确语义。

HRDexDB 的共享 geometry layer 必须遵守同一合同：保留稳定的 `[T,4096,3]` 物体池和 `[T,4096]` 的当前手 5 cm candidate mask，Cm loader 再在线采样 `[512,3]`。仅有全表面 `[T,512,3]` 且没有 candidate mask 的 decoder smoke cache 不可直接作为 Cm 训练输入；CmDecoder 的 legacy task layer 仍可保留独立的 512 点字段。

HRDexDB 的仓库内规范原始数据入口为 `dataset/HRDexDB/v0_nonvideo`，配套机器人资产和读取 helper 位于同级 `assets/`、`hrdexdb_contact_heatmaps/`；整个 `dataset/HRDexDB/` 是机器本地数据，不纳入 Git。

correspondence_ptv3_v2 的 HRDexDB robot 路径采用独立 URDF/FK q-space：固定
link/barycentric 手点绑定，hand q joints 加受限噪声后回到 `base_link` hand-root；
不把 robot qpos 转成 MANO 字段，也不将 arm/base 扰动与当前 hand-root corruption
混合。七域 mixed loader 通过样本级 contract dispatch 同时容纳 MANO 与 robot，三种
机器人按 FK 标定使用不同 q-space 噪声乘数。

minimal allhands archive 的 Stage3 适配由 `process/HRDexDB/correspondence_minimal_adapter.py` 统一处理四种手型：human 使用 MANO reconstruction，三类机器人使用对应 URDF/FK。输出仍使用统一 4096 object pool、1538 hand points 和 hand-root schema。

## 3. 阶段一：correspondence_ptv3_v2 / DenseToken

### 3.1 数据生成和运行时采样

数据流为：

```text
raw dataset
  → 数据集适配（Stage 2）
  → hand-root 规范化与 clean 几何统计（Stage 3 v2）
  → CorrStaticDatasetV2
  → 确定性采样 + 物体位姿扰动
  → PTv3
```

Stage 3 每帧保留 `4096` 个稳定物体表面点和 `1538` 个 MANO face-center 手点，并预计算：

```text
obj_candidate_mask_5cm [T,4096]
hand_to_obj_min_dist   [T,1538]
```

每个运行时样本从 5 cm candidate 中确定性抽取最多 `512` 个物点。训练集按 epoch 改变采样，验证集固定 epoch 0；采样种子由 sequence、side、frame、epoch 等共同决定。

训练时只扰动物体点和法向，不扰动手：

```text
输入：  (O_tilde, H)
监督：  clean (O, H) 计算的连续接触 target
```

连续接触 target 不是二分类标签，而是：

```text
y(d) = clamp(1 - d / 0.02, 0, 1)
```

其中 `d` 是 object-hand 点对距离，`0.02 m` 是接触半径。

### 3.2 PTv3 输入特征

对每个点构造 11D runtime feature：

```text
[xyz(3), entity_one_hot(2), normal(3),
 distance_to_opposite_nearest(1),
 dot(direction_to_nearest, normal)(1),
 dot(direction_to_opposite_centroid, normal)(1)]
```

物体点的 opposite set 是有效手点，手点的 opposite set 是有效物点。物体和手使用同一个 PTv3 backbone，而不是两个独立 encoder：

```text
coord  = concat(object_points, hand_points)       [B,2050,3]
feat   = concat(object_11d, hand_11d)              [B,2050,11]
tokens = PTv3(feat, coord, point_valid_mask)
z_obj  = tokens[:, :512]
z_hand = tokens[:, 512:]
```

当前 DenseToken checkpoint 的 token 维度为 `D=96`，因此：

```text
z_obj  [B,512,96]
z_hand [B,1538,96]
```

### 3.3 监督 head 与下游接口

模型有一个共享的 object-hand cross-edge head 和一个可选的 dense hand-contact head：

- `random128`：每个有效物点均匀抽取 128 个手点，使用 shared edge MLP 计算连续接触概率；这是主要 checkpoint 选择流。
- `contact-aux`：按接触强度分层采样弱/中/强/极强接触及 hard negative，复用同一个 edge head，只改变边的采样分布。
- `dense hand contact`：根据完整 4096 物体点池的最短距离，输出 `[B,1538]` 的逐手点 contact prior。

下游 `FrozenDenseTokenEncoder` 加载该 checkpoint 后，只执行 PTv3 的 `encode_points` 和 hand-contact head，得到 `z_obj`、`z_hand`、contact prior。它始终处于 `eval/no_grad`，不读取 `hand_flow`、未来物体或 `obj_flow_gt`。

详细实现见 [`correspondence_ptv3_v2/docs/架构.md`](../../src/task/correspondence_ptv3_v2/docs/架构.md)。

## 4. 阶段二：Cm / 通用交互动作表征

### 4.1 Cm 输入和冻结边界

Cm 的一个 sample 包含：

| 张量 | 形状 | 用途 |
|---|---:|---|
| `obj_points` / `obj_normals` | `[B,512,3]` | 当前物体几何 |
| `hand_points` / `hand_normals` | `[B,1538,3]` | 当前单侧手几何 |
| `hand_flow` | `[B,1538,3]` | 当前到未来的手部动作条件 |
| `obj_flow_gt` | `[B,512,3]` | 仅用于 object-flow 监督 |
| `obj_valid_mask` | `[B,512]` | 忽略 padding 物点 |
| `delta_time_s` | `[B]` | 可选的 stride 时间条件 |

进入 Cm 时，DenseToken encoder 已经冻结；Cm checkpoint 只保存可训练的 motion head、slot attention、gate 和 object-flow head。训练 Cm 时更新这些 head，不能把 DenseToken/PTv3 参数带入 optimizer。

### 4.2 Hand action branch：从逐点运动到 slot

每个手点拼接 DenseToken、几何、运动和 contact prior：

```text
[z_hand(96), hand_xyz(3), hand_normal(3), hand_flow(3), contact(1)]
      [B,1538,106]
          │
          ▼
hand_motion_encoder（MLP/GELU）
          │
          ▼
u_hand [B,1538,C]
          │
          ▼
Slot Attention（K=16，默认迭代 3 次）
```

Slot Attention 产生：

```text
cm_assignment  [B,16,1538]  # 沿 slot 维归一化，每个手点分配和为 1
cm_slot_weights [B,16,1538]  # 沿手点维归一化，用于 anchor
cm_tokens      [B,16,C]
```

`cm_tokens` 是 Cm 的核心通用表征：每个 slot 可以理解为当前交互动作场中的一个局部/动态分支，但 slot index 没有预先绑定“拇指”“食指”等固定语义。

可选的 `delta_time_s` 经过 `MLP(1→C)` 后加到每个 slot 和 object context；它改变时间条件，不改变 slot 数量。

### 4.3 Slot gate 和 routing

主模型可用共享的 gate head 对每个 slot 计算 Hard-Concrete nonzero probability：

```text
cm_tokens [B,16,C]
  → slot_gate_head
  → slot_gate_logits / slot_nonzero_prob [B,16]
  → hard_slot_mask [B,16]
```

训练时 soft routing 与 hard routing 使用 straight-through 组合；评估时使用确定性阈值。如果所有 slot 都被阈值过滤，保留 gate logit 最大的 fallback slot。关闭 gate 的配置则让全部 slot 参与普通 softmax。gate 只改变 slot 路由和稀疏性，不改变 `cm_tokens` 的 shape。

### 4.4 Object-flow branch

物体 context 只使用当前物体几何和物体 DenseToken：

```text
[z_obj(96), obj_xyz(3), obj_normal(3)]
      [B,512,102]
          │
          ▼
object_context_encoder → obj_context [B,512,C]
```

每个物点和每个 slot 建立一条 edge：

```text
[obj_context(C), cm_token(C), obj_to_anchor(3), anchor_normal(3)]
      [B,512,16,2C+6]
          │
          ▼
edge_backbone → edge_features [B,512,16,C/2]
          ├─ dynamic_logits [B,512,16]
          └─ candidate_flow [B,512,16,3]
                   │ slot routing 加权求和
                   ▼
pred_obj_flow [B,512,3]（米）
```

默认 `C=256` 时 edge 输入为 `518` 维，edge feature 为 `128` 维；`C=64` 时对应为 `134` 和 `32` 维。无效物点的最终预测强制为零。

Cm 的训练 loss 主要是有效物点上的 scaled vector Smooth-L1 object-flow loss，并可附加 slot count、confidence 和 active-overlap 等辅助项。主评估指标是全 loader 点加权 EPE；slot 使用率、有效分支数、top-1 usage 和 slot entropy 是诊断量，不是默认监督目标。

### 4.5 Cm checkpoint 的使用方式

训练完成后，Cm checkpoint 与 DenseToken checkpoint 组成一个不可随意拆分的 encoder 组合：

```text
当前参考手/物体几何 + hand_flow
    → Frozen DenseToken
    → Frozen CmFlowModel
    → cm_tokens [B,16,256]
```

`cm_tokens` 可以在线计算，也可以通过 `build_cm_cache.py` 写成 `[N,16,256]` 的 mmap 文件。预提取 cache 必须记录 DenseToken/Cm checkpoint hash、manifest 和 sampling/schema 版本；更换任一上游 checkpoint 后必须重建或重新校验 cache。

详细实现见 [`Cm/docs/logs/architecture_log.md`](../../src/task/Cm/docs/logs/architecture_log.md)。

## 5. 阶段三：CmDecoder / 目标手专用解码器

### 5.1 为什么每种手需要单独 decoder

Cm 表示的是手—物交互动作，不直接表示某个机器人应该转动哪一个关节。不同目标手有不同的：

- URDF 链、关节数量、joint limit 和 mimic joint 规则；
- 表面点采样方式和 `1538` 个点到 link-local 点的绑定；
- 当前 wrist frame、腕部自由度和动作幅度分布；
- 可用的 q / wrist supervision 和拟合超参数。

因此每种目标手使用独立的 target-specific cache、配置和 decoder checkpoint；它们共享的是冻结的 DenseToken/Cm 表征接口，而不是 decoder 权重。

### 5.2 目标手 cache 和样本

当前 Inspire F1 路径由 HRDexDB 的 q、C2R 和物体姿态恢复 world geometry，再转换到当前 wrist frame：

```text
robot q / object pose / mesh
      → URDF FK + C2R
      → world hand/object points and normals
      → current-wrist task pair (t, t+stride)
```

分层 cache 为：

```text
geometry/
  world hand/object points, normals, q_full, wrist_pose_world, frame time
task/
  hand-root_t(or current-wrist) points, normals, hand_flow,
  q_t, q_next, wrist_delta, source_frame_id
cm/
  optional cm_tokens.npy [N,16,256]
```

逐点路径的当前有效版本是 3 Hz `v2` cache：`hand_flow_frame=current_wrist`。基础 30 Hz cache 使用 `stride=1`；3 Hz 诊断通常使用 `stride=10`。cache manifest 记录 schema、源数据 hash、实现 hash、采样配置和 checkpoint hash，字段不一致时应重建或 fail-fast。

### 5.3 Frozen Cm 在 decoder 中的两种来源

有配对训练时，目标手自己的 `(obj_t, hand_t, hand_flow)` 送入冻结 Cm，得到与目标手训练样本对应的 `C_m`；这使 decoder 学会“目标手当前状态 + 交互动作表示 → 目标手下一状态”。

跨形态重定向时，参考手序列负责生成 `C_m`，目标手只提供当前状态：

```text
参考手 (source geometry, source hand_flow)
      → Frozen DenseToken + Frozen Cm
      → source cm_tokens

目标手当前 (target geometry, target wrist/q)
      + source cm_tokens
      → target-specific CmDecoder
```

当前 Cm 需要 endpoint `hand_flow` 才能编码动作，因此实时部署还需要参考侧的短时动作估计或相邻帧差分；仅有单张静态参考图像时，不能直接得到完整的 Cm 动作条件。

### 5.4 整体 q/wrist Decoder（baseline）

`CmDecoderModel` 支持 `qt_cm`、`qt_only` 和 `cm_only` 三种对照。主 `qt_cm` 路径为：

```text
cm_tokens [B,16,256] → flatten [B,4096]
q_t        [B,6]
             │ concat
             ▼
         [B,4102] → LayerNorm
             → Linear(4102,512) + GELU
             → Linear(512,256) + GELU
             → Linear(256,6 或 12)
```

输出前 6 维为 `q_next` 或 `delta_q`（由配置决定）。启用 wrist motion 时，后 6 维为：

```text
pred_wrist_delta_translation [B,3]  # 米，当前 wrist frame
pred_wrist_delta_rotvec      [B,3]  # 弧度，当前 wrist frame
```

该 baseline 直接学习参数空间映射，结构简单，但不同的 q/wrist 分解可能产生相近的手部点运动，因此需要同时检查点重建误差，不能只看 q MAE。

### 5.5 逐手点 Cm Decoder（当前重定向主路径）

`CmPointFlowModel` 的网络路径不读取 `q_t`。它先用目标手当前几何经过冻结 DenseToken，得到：

```text
z_hand       [B,1538,96]
hand_contact [B,1538]
```

每个手点构造 point context：

```text
[z_hand(96), hand_xyz(3), hand_normal(3), contact(1)]
       [B,1538,103]
            │
            ▼
hand_context_encoder → [B,1538,C]
```

再将每个手点与每个冻结 Cm slot 建 edge：

```text
hand_context [B,1538,C] × cm_tokens [B,16,C]
      → edge input [B,1538,16,2C]
      → edge_backbone [B,1538,16,C/2]
          ├─ edge_logit_head → routing logits [B,1538,16]
          └─ edge_flow_head  → candidate flow [B,1538,16,3]
      → 沿 slot softmax routing
      → 加权得到 pred_hand_flow [B,1538,3]
```

当前 `C=256` 时 edge 输入是 `512` 维、hidden 是 `128` 维。point-flow 训练使用固定 correspondence 的米制 Smooth-L1：

```text
target_hand_points = hand_points + hand_flow
pred_hand_points   = hand_points + pred_hand_flow
loss               = SmoothL1(pred_hand_points, target_hand_points)
```

checkpoint 主要按 `val/hand_flow/epe_mm` 选择。只有 point-flow decoder 的参数更新，DenseToken 和 Cm 全部保持冻结。

### 5.6 Point flow 到目标手 q/wrist 的后处理

点流本身不一定对应唯一的 q/wrist 分解，因此使用 `q_optimizer.py` 做运动学后处理：

1. 使用目标手 URDF、静态 `hand_point_link_index [1538]` 和 `hand_points_local [1538,3]` 建立固定点绑定。
2. 以当前 `q_t`、零相对腕平移/旋转为优化初值。
3. 用预测点作为目标：`target = current_hand_points + pred_hand_flow`。
4. 对 6 个独立 Inspire F1 手指关节和当前 wrist 的 6D 相对 SE(3) 联合优化；mimic joints 按 URDF multiplier 展开，q 每步 clamp 到 joint limit。
5. 通过可微 FK 生成预测手点，最小化点误差，并加入 q prior 和 wrist prior。
6. 得到 `q_next` 和 `wrist_delta`，更新下一帧目标手状态。

向量化 FK 的实际路径为：

```text
q → link_tf [B,L,4,4]
  → gather(point_link_index) [B,1538,4,4]
  → homogeneous matmul(hand_points_local)
  → predicted hand points [B,1538,3]
```

这个拟合只在 decoder 训练后的推理/评估阶段使用，不参与 point-flow checkpoint 的反向训练；未来 arm FK 不参与目标手重建。

## 6. 端到端重定向的逐帧执行顺序

以 GRAB → Inspire F1 为例，当前无配对路径的逻辑是：

```text
初始化：目标机器人 q0 = joint-limit 中点，腕部放到物体中心外侧的确定姿态

对每个参考帧 t：
  1. 读取参考手 t→t+1 的 hand_flow 和手物当前几何
  2. 参考手经过 Frozen DenseToken + Frozen Cm，得到 source cm_tokens
  3. 读取目标机器人当前 q、wrist pose 和物体点云
  4. 用目标机器人当前几何经过 Frozen DenseToken，得到 target z_hand/contact
  5. 将 target geometry + source cm_tokens 输入目标手 CmDecoder
  6. 预测目标手 1538 个点的 hand_flow
  7. 用固定 point binding + 可微 FK 拟合 q_next 和相对 wrist SE(3)
  8. 更新机器人 q/wrist，输出目标手世界坐标点和下一帧状态
```

参考手的点云不被当作目标机器人的 q 监督；它只产生动作条件 `C_m`。目标机器人点云负责提供目标形态、当前接触状态和解码坐标系。这样才是“共享动作表示、目标手专用运动学”的重定向结构。

## 7. 训练、评估和可视化边界

训练必须按以下顺序执行：

1. 独立训练/验证 correspondence checkpoint，确认 `val_perturbed` 接触恢复有效。
2. 固定 DenseToken，训练 Cm；train/val/test 按 sequence 互斥，flow scale 和统计量只从 train split 生成。
3. 固定 DenseToken + Cm，为每一种目标手建立独立 cache 并训练其 CmDecoder；目标手数据按 episode/object 做无泄漏划分。
4. 最后运行 point-flow → FK/q fitting 的目标手评估和跨形态重定向 smoke/full test。

评估不能只看一个指标：

- correspondence：`pseudo_recovery_brier`、missed/fake contact recovery、strong-contact MAE；
- Cm：点加权 object-flow EPE、zero-flow improvement、stride 分项和 slot usage 诊断；
- CmDecoder：point EPE/RMSE、q MAE、wrist 平移/旋转误差、identity baseline 对比；
- 重定向：目标手轨迹连续性、点云与物体关系、q/joint-limit 合法性和可视化结果。

可视化入口：

- `src/task/correspondence_ptv3_v2/visualize.py`：接触/对应关系；
- `src/task/Cm/visualize.py` 与 evaluator：object flow、Cm slot、anchor；
- `src/task/CmDecoder/viewer.py`：当前手、GT 手和预测手；
- `src/task/CmDecoder/grab_retarget.py`：导出 GRAB → Inspire F1 轨迹和 PNG。

## 8. 必须保持的不变量和 fail-fast 条件

- DenseToken 固定接收 `512` 个物点、`1538` 个单侧手点；点数、法向、坐标系和有效掩码必须匹配 checkpoint。
- DenseToken 只能编码当前几何；future geometry、`hand_flow` 和 `obj_flow_gt` 不得泄漏到 PTv3 当前状态分支。
- Cm 的 slot 数量/维度由 checkpoint 决定；当前主合同是 `[B,16,256]`，C=64 只能与匹配配置和 cache 一起使用。
- Cm token cache 必须绑定 DenseToken checkpoint、Cm checkpoint、manifest、schema 和采样配置 hash，禁止静默混用。
- CmDecoder 必须使用匹配目标手的 URDF、点绑定、joint limits、mimic 规则和 wrist frame；不同目标手 decoder 不能直接共享输出层。
- 3 Hz wrist-aware v2 cache 的 `hand_flow_frame` 必须为 `current_wrist`；历史 v1 不能用于联合 wrist+q 点监督。
- 训练/验证/测试 split 必须 sequence/episode/object-disjoint；calibration、统计量和 checkpoint 选择不能读取 test。
- 缺失 subject-specific MANO template、非连续帧、错误坐标声明、错误 cache 版本或不一致 calibration 必须直接报错，不能静默回退。

各阶段的完整实现事实分别见 [`correspondence_ptv3_v2/docs/架构.md`](../../src/task/correspondence_ptv3_v2/docs/架构.md)、[`Cm/docs/logs/architecture_log.md`](../../src/task/Cm/docs/logs/architecture_log.md) 和 [`CmDecoder/docs/logs/architecture_log.md`](../../src/task/CmDecoder/docs/logs/architecture_log.md)。
