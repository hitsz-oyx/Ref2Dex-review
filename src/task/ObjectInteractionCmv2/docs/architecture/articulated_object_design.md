# ObjectInteractionCmv2 铰接对象架构设计

- 状态：`V1.5.1 首阶段实现中 / loader、graph 与 analytic FK 已实现 / 真实 cache FK replay 未通过，训练未批准`
- 目的：把当前 V1.3 的 single-root rigid flow 路径扩展为“root 刚体运动 + 多 link joint motion + 解析 FK flow”。
- 适用数据：GRAB 等刚体对象退化为单 link；ARCTIC 当前的双 link、单 revolute joint 对象为第一实现目标。
- 非目标：本文件不批准 cache 重写、训练或 checkpoint 转换；V1.5 的代码边界由同版本 final plan 冻结，
  真实 cache 的 FK replay 仍是训练前阻断闸门。

## 1. 总览：从手-物局部交互到结构化多 link flow

模型仍只从当前时刻的物体/手几何和手的未来短时运动取得交互证据；它不会读取 future object geometry、future joint angle 或 GT flow 作为输入。区别在于，物体不再被假定为一个刚体：每个点先被分组到 rigid link，link 表征经 joint graph 传递，再预测 root 与 joint 的最小运动变量，最后以 FK 重建每个点的 future flow。

```text
current object points/normals + current hand points/normals + hand flow
                 │
                 ▼
      local swept KNN32 interaction encoder
                 │ per-object-point geometry/contact feature
                 ▼
  link-wise geometry pool + link-wise contact pool       current q / joint geometry
                 │                     │                          │
                 └──────────────┬──────┴──────────────┬───────────┘
                                ▼                     ▼
                    link nodes Z[B,L,D]      joint edges E[B,J,D]
                                │                     │
                                └──── articulation graph ────┘
                                               │
                                               ▼
                   root head Δξ_root[B,6] + joint head Δq[B,J]
                                               │
                                               ▼
                masked analytic FK, indexed by object link id
                                               │
                                               ▼
                         obj_flow_pred[B,N,3]
```

第一版的实参为 `N=1024`、`H=4096`（MANO）或 `20270`（Inspire）、`L<=2`、`J<=1`、feature width `D=128`、grounded token count `K=16`、graph message-passing rounds `M=2`。`L/J/H` 都是 batch-padded 维度，不能把 padding 或任意 link ID 当作语义。

## 2. 坐标、时间与 FK 约定

所有长度单位为 `m`，时间来自 cache 的 `frame_time`，训练 stride 仍由 runtime transition 决定。以下以列向量记号说明；实现可沿用 PyTorch 的 row-vector 计算，但必须有等价的单元测试。

- `T^w_{r,t}[B,4,4]`：**root-to-world** pose；ARCTIC 使用现有 `obj_root_pose_world`，不能再使用恒等的 `obj_pose_world` 作为其 root supervision。
- `p_t^r[N,3]`：在当前 root frame 的 object point；loader 由 `T_w_r,t` 把 world point 转入此 frame。
- `link_id[N]`：点所属 rigid link。它只用于 gather/segment pooling/FK selection，禁止送入 embedding。
- `q_t[J]`：当前 joint coordinate；只作为输入状态。`q_{t+\Delta}`、`Δq_gt` 和 future root pose 只用于监督。
- `ΔT_root,t→t+Δ = (T^w_{r,t})^{-1} T^w_{r,t+Δ}`：future root 在当前 root frame 的相对变换；其 6D Lie-algebra 参数是 `Δξ_root_gt[B,6]`。

对 ARCTIC 第一阶段，cache producer 已使用的运动合同是：link 0 为底部，link 1 为顶部；在 producer 的
row-vector 约定下，top 在 canonical root frame 围绕 `axis=(0,0,-1)`、`origin=(0,0,0)` 作 revolute rotation，
再接受 root SE(3)。该符号已由 `process/ARCTIC/raw.py` 的 `R_arti` 与真实 cache replay 验证，并冻结在
[`configs/active/arctic_articulation_v1_5.json`](../../configs/active/arctic_articulation_v1_5.json)；不能只靠
object 名称猜测。泛化接口允许每 joint 提供 axis/origin/type/limits。

令 `A_j(q)` 为 joint `j` 的齐次变换，`L(i)` 为 point `i` 的 link。预测 future point（表达在 current root frame）为：

```text
q̂_next = q_t + Δq̂
T̂_i = ΔT̂_root · FK_link(L(i), q̂_next) · FK_link(L(i), q_t)^(-1)
p̂_future_i = T̂_i · p_t_i
obj_flow_pred_i = p̂_future_i - p_t_i
```

对 rigid fallback，`L=1,J=0,FK_link=I`，故该式精确退化为现有 root SE(3) flow。对 ARCTIC 的 root link，joint relative transform 也为 identity；只有 top link 接受 `Δq̂`。

## 3. 端到端张量规格与数据流

### 3.1 Loader 输出

下表的 `input` 表示可进入前向；`target` 表示只能在 loss/metric 使用。`B` 是 batch size，`N=1024`，`H=max(H_b)`，`L=max(num_links_b)`，`J=max(num_joints_b)`。

| tensor | shape / dtype | role | 来源与约束 |
| --- | --- | --- | --- |
| `obj_points` | `[B,N,3] float32` | input | current root frame；从 4096 pool 的确定性 sample |
| `obj_normals` | `[B,N,3] float32` | input | 与 `obj_points` 同行对应、单位法向 |
| `obj_link_id` | `[B,N] int64` | structural input | range `[0,num_links_b)`，只用于分组/FK gather |
| `num_links` | `[B] int64` | structural input | rigid=1，ARCTIC first stage=2 |
| `joint_parent/child` | `[B,J] int64` | structural input | link indices，padding 为 `-1` 并由 mask 屏蔽 |
| `joint_type` | `[B,J] int64` | structural input | first stage only `revolute`；编码 type，不编码 object/link identity |
| `joint_axis_root` | `[B,J,3] float32` | structural input | unit vector；必须 finite，padding 无效 |
| `joint_origin_root` | `[B,J,3] float32` | structural input | current root frame、单位 m |
| `joint_q_t` | `[B,J] float32` | input | current frame；不得用 future q 替代 |
| `joint_limits` | `[B,J,2] float32` | structural input | `(min,max)` rad；无已验证 limit 时显式 mask，不臆造 |
| `joint_valid_mask` | `[B,J] bool` | input mask | padding=false；rigid sample 全 false |
| `hand_points/normals` | `[B,H,3] float32` | input | MANO/Inspire high-res stream；仅 batch padding |
| `hand_flow` | `[B,H,3] float32` | input | current→future hand displacement，仍是 swept KNN evidence |
| `hand_valid_mask` | `[B,H] bool` | input mask | 必须屏蔽 padding |
| `delta_time_s` | `[B] float32` | input | current→future真实时间差 |
| `obj_flow_gt` | `[B,N,3] float32` | target | current root frame的 observed future point flow |
| `delta_xi_root_gt` | `[B,6] float32` | target | 从 `obj_root_pose_world` 计算 |
| `delta_q_gt` | `[B,J] float32` | target | `q_future-q_t`，按 joint type/周期性定义严格处理 |

Loader 在任何一项不满足以下条件时 fail-fast：part ID 越界、point correspondence 不一致、root pose 非 SE(3)、axis 非单位、future frame 不递增、joint mask/padding 不一致、FK replay 与已缓存 `obj_flow_gt` 超容差。不能用 identity root pose 代替 ARCTIC 的已存 `obj_root_pose_world`。

### 3.2 前向张量流

```text
obj_points[B,N,3], obj_normals[B,N,3]
       │  GeometryEncoder
       ▼
g[B,N,D]

g + obj/hand geometry + hand_flow + masks
       │  swept LocalInteractionEncoder (KNN=32, radius=0.02 m)
       ▼
c[B,N,D], active[B,N], edge_indices[B,N,32], edge_valid[B,N,32]

(g,c,active,obj_link_id,num_links)
       │  masked segment pools; no learned link-ID embedding
       ▼
G_link[B,L,D], I_link[B,L,D], contact_stats[B,L,C], link_valid[B,L]

(G_link,I_link,contact_stats,joint_type,axis,origin,q_t,limits,joint_valid)
       │  node/edge encoder + M=2 graph message-passing rounds
       ▼
Z_link[B,L,D], Z_joint[B,J,D]
       │
       ├─ RootHead(pool(Z_link,link_valid)) ───────────────► Δξ_root_hat[B,6]
       └─ JointHead(Z_joint,joint_valid) ──────────────────► Δq_hat[B,J]
                                                               │
(obj_points,obj_link_id,q_t,Δξ_root_hat,Δq_hat,graph masks)  │
       └──────────────────────────── AnalyticFK ──────────────┘
                                      ▼
                      obj_flow_pred[B,N,3], point_future_pred[B,N,3]
```

`G_link` 对每个 link 的全部 object geometry 聚合；`I_link` 仅对 `active=true` 的局部接触特征聚合。二者分开是为了避免小接触区被长 link 的无接触点稀释。空接触 link 仍产生 finite geometry node，contact component 以零值和显式 `has_contact` 表示。

## 4. 模块规格

### 4.1 保留的局部交互编码器

`LocalInteractionEncoder` 维持当前 swept KNN32、2 cm 半径与 `hand_valid_mask` 语义。对每个 object point，它从相对位置、current/end swept distance、object/hand normal、normal/tangent hand flow 与真实 `delta_time_s` 构造 edge feature，输出 `c[B,N,D]`。这部分不需要 object link ID，因而可在 link permutation 下保持等变。

### 4.2 Link-wise pool 与 grounded tokens

对每个 batch item、每个有效 link `l`：

```text
G_l = masked_mean({g_i | link_id_i=l})
I_l = masked_contact_pool({c_i | link_id_i=l, active_i=true})
s_l = [active_fraction, mean_normal_hand_velocity, mean_tangent_hand_velocity]
z_l^0 = MLP_link([G_l, I_l, s_l]) ∈ R^D
```

保留 `K=16` grounded tokens，但 token 必须携带 `[anchor_position, anchor_normal, link_id_for_grouping, mass]`。在 graph 输入时按 link 聚合 token，而不是把 token 序号解释为固定语义。token padding 由 mask 屏蔽。

### 4.3 Joint edge 与 articulation graph

Joint edge 的几何状态为：

```text
e_j^0 = MLP_joint([
  one_hot(joint_type_j), joint_axis_j, joint_origin_j / feature_scale_m,
  normalized(q_t_j, joint_limits_j), optional_limit_mask
]) ∈ R^D
```

第一版将父/子 node 与 edge 做双向 gated message passing 两轮：

```text
m_parent→child = φ_m(z_parent, z_child, e_j)
m_child→parent = φ_m(z_child, z_parent, e_j)
z_l^(r+1) = φ_u(z_l^r, Σ incoming_m)
e_j^(r+1) = φ_e(e_j^r, z_parent^(r+1), z_child^(r+1))
```

所有 reduce 均以 `link_valid/joint_valid` masked；没有 joint 的 GRAB 样本跳过 edge update。节点或 edge 的排列变化，只会对应地置换输出，不能改变数值语义。

### 4.4 Structured effect heads 与 analytic FK

- **RootHead**：聚合有效 `Z_link`，输出 `Δξ_root_hat[B,6]`，由稳定 Rodrigues/exponential map 成 `ΔT_root_hat[B,4,4]`。
- **JointHead**：从 `Z_joint[B,J,D]` 输出 `Δq_hat[B,J]`；无效 joint 强制为零。revolute 的输出可在训练时依 `joint_valid` mask 并按记录的 limit 作受控 clamp，但不能把 clamp 当作修复错误 GT 的手段。
- **AnalyticFK**：计算每个有效 link 的 `T_link(q_t)` 与 `T_link(q_t+Δq_hat)`，按 `obj_link_id` gather 到 `[B,N,4,4]`，再计算 point flow。它不包含可学习 residual；首版若 FK 无法 replay cache GT，必须先修 schema/坐标，不以 residual 掩盖。

## 5. 监督、指标与训练边界

训练总损失是受配置声明的结构化组合：

```text
L = λ_flow L_smoothL1(obj_flow_pred, obj_flow_gt)
  + λ_root_t L_huber(Δt_hat, Δt_gt)
  + λ_root_R L_geodesic(ΔR_hat, ΔR_gt)
  + λ_joint L_joint(Δq_hat, Δq_gt; joint_valid)
  + λ_limit L_limit(q_t + Δq_hat, joint_limits; valid limit only)
```

初版不使用任意 flow residual。所有 `λ`、angle wrapping、limit policy、root-fixed mask 和 speed/scale 正规化都必须在未来 config/manifest 固定；本设计不预设数值以免将未经批准的实验变量伪装为架构事实。

除现有 EPE、prediction/GT flow magnitude、angle 外，必须额外报告：

- root translation / rotation error；
- `Δq` MAE（degree）和关节 limit violation；
- per-link EPE（root link、moving link）与 ARCTIC articulated-only pooled EPE；
- cache-FK replay max/mean error；
- rigid fallback GRAB 的 EPE，确保 `L=1,J=0` 没有退化。

旧 V1.4.4 checkpoint 的 `v1_3_rigid_only` state dict 与本模型不兼容；不得自动加载或把它解释为 articulated baseline。可以将其作为独立 rigid baseline 评估，但新的训练必须产生独立 checkpoint/run manifest。

## 6. 数据合同与 batch 行为

| domain | links / joints | required state | FK branch |
| --- | --- | --- | --- |
| GRAB rigid | `L=1, J=0` | root pose；`obj_link_id=0` | root SE(3) only |
| ARCTIC first stage | `L=2, J=1` | part ID、root pose、q、revolute axis/origin/type | root SE(3) + top-link revolute FK |
| future arbitrary articulated | `L>=1, J>=0` | validated kinematic tree and per-link correspondence | generic tree FK |

同一 batch 允许 GRAB 与 ARCTIC 混合：`obj_link_id` 仍为 `[B,N]`，link/joint tensors pad 到 batch max，所有 pool/graph/FK/head/loss 均以 mask 处理。手的 `H` padding 与物体的 `L/J` padding 是独立概念，不能复用 mask。

ARCTIC source cache 已有 `obj_part_id`、`obj_articulation`、`obj_root_pose_world`；但目标 V1.5 runtime 合同要求显式、可审计的 joint type/axis/origin/limits 来源。若现有 manifest 无法证明该来源，先生成独立 metadata sidecar 或新 cache schema，而非把隐含 preprocessing 假设散落在 model 中。

## 7. 实施前必须验证的张量不变量

1. **FK replay**：使用 GT root pose 与 `q_t/q_future` 的解析 FK，在每个 ARCTIC sequence 上重建 cached future points；先冻结 max/mean tolerance，再训练。
2. **Part consistency**：`obj_part_id` 固定对应 canonical point identity，所有 sampled point 的 part 不随帧漂移。
3. **Root semantics**：ARCTIC 使用 `obj_root_pose_world`；不得使用为兼容旧 rigid loader 填写的 identity `obj_pose_world`。
4. **No leakage**：前向只能读取 `q_t`、current root pose 和 hand future flow；`q_future`/future root/object point 只进入 target/FK replay。
5. **Permutation check**：交换 link 编号并同步 parent/child、point link ID 与 joint graph 后，输出 flow 按同一物理对象保持不变。
6. **Rigid reduction**：`L=1,J=0` 的 analytic FK 必须与 root SE(3) reference 实现逐点一致。
7. **Mixed-batch masking**：无效 hand/link/joint padding 不得产生 edge、message、loss 或 non-finite value。

## 8. 分阶段落地与保护边界

```text
Phase A  schema audit + exact FK replay (no model training)
Phase B  loader/collate + analytic FK + synthetic two-link tests
Phase C  link pools + graph + root/joint heads + gradient/checkpoint smoke
Phase D  independent GRAB-rigid vs ARCTIC-articulated benchmark
```

每阶段应单独拥有 final plan、run ID、manifest 和活动记录。任何 FK replay、坐标、单位或 point correspondence 失败都阻断下一阶段。不得修改旧 V1.4.4 checkpoint、旧结果、GRAB/ARCTIC split、cache queue 或公共 `src/base`；不以 residual、重采样、删除 high-motion frame 或替换 GT 使结果“可训练”。

## 9. 当前状态与决策入口

V1.5.1 已实现 Task-local loader/collate、two-link graph、root/joint head 与 masked analytic FK，并以 synthetic
mixed GRAB/ARCTIC batch 完成 forward/backward smoke。现有 ARCTIC cache 尚未持久化经验证的 per-object
axis/origin metadata；现已由 producer 源码恢复并冻结为 `axis=(0,0,-1), origin=(0,0,0)`。在 `phone_use_01`
四个真实 transition 上，GT FK replay 的 mean error 为 `0.001529/0.000111/0.000130/0.000054 mm`，但全 cache
replay、loss 权重、joint limits、训练预算及旧 checkpoint baseline protocol 仍需后续计划冻结；当前实现不得用于训练。
