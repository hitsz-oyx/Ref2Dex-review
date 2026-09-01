# CmDecoder 张量架构

- scope: task:CmDecoder
- last_updated: 2026-09-01
- last_verified: 2026-09-01
- related: [当前状态](status_log.md), [实验](experiment_log.md), [接手记忆](repo_memory.md)

## 当前入口边界（V1.2.12）

- CmDecoder 的模型、Runner、Dataset 和配置直接位于 `src/task/CmDecoder/`；Task-local `components/`、`data/` 和 `registry/` 已撤出。
- 外部数据与 cache 使用根级 `data/processed_data/`、`dataset/` 和 `outputs/`；Task 内不保留数据软链接或路径清单。
- 下方历史架构描述保留用于复现，不代表当前存在的 Component/registry 入口。

## 数据与坐标

### HRDexDB 多手型 cache builder

`build_cache.py` 按 episode 路径的首级目录自动选择数据适配器，支持：

- `human`：读取固定拓扑 MANO OBJ（1538 个三角面），以 `mano_params/*.json` 的 MANO wrist joint/global orientation 作为当前 hand-root；不伪造机器人关节监督。
- `allegro_v5`：读取 16 维 Allegro hand state 与 `allegro_v5/xarm_allegro_v5.urdf`。
- `inspire_dftp`：读取 DFTP hand state 与 `xarm_inspire_DFTP.urdf`。
- `inspire_f1`：保持原有 F1 state、URDF 和 cache 语义。

四类数据均优先使用 `object_6d_pose_v2/<kind>/<object>_<scene>.npz`，回退 v1/episode 内 pose；对象 mesh 来自 `assets/mesh_v2`（回退 `assets/mesh`）。几何层输出 `[T,1538,3]` 手点、legacy 512 点 task 物体字段，以及供 Cm 使用的稳定 `[T,4096,3]` object pool、法向和 `[T,4096]` 当前手 5cm candidate mask；任务层仍在当前 wrist frame 形成对应点 `hand_flow`。这样 Cm 与 CmDecoder 可以共享一次几何/FK 计算，但 DenseToken feature 不进入该 cache。

机器人 `q_full` 保留原始维度（arm 6 + hand 6/16）；MANO 的 `q_full` 为零占位并标记 `q_semantics=unavailable_mano`。混合手型的 point-flow dataset 将非 Inspire q 映射为固定六维零张量，因为当前 flat point decoder 不读取 q；任何 q decoder 训练必须按手型过滤，禁止把该占位当作 MANO/Allegro 的 Inspire 关节 GT。

每个样本由同一 episode 的 `(t, t+stride)` 组成：基础 cache 使用30 Hz的 `stride=1`，当前动作诊断还使用3 Hz的 `stride=10`。arm q 只用于完整 FK 恢复当前腕部坐标系；模型不预测未来 arm q，而以当前腕为原点预测当前腕到目标腕的相对 SE(3)，并预测6个 Inspire F1手指关节角。

```text
HRDexDB robot q (12) + hand/object geometry
  └─ URDF FK + C2R → world geometry
  └─ current wrist pose → hand-root_t geometry for Cm
```

固定面采样保证相邻帧 correspondence：

### 分层 cache

训练前由 `build_cache.py` 独立构建，DDP 训练进程只读 cache：

```text
geometry/
  world hand/object points + normals
  full q + wrist pose + frame time
task/
  hand-root_t inputs + hand flow + q_t/q_next
```

每层包含 manifest；`schema_version + source SHA256 + implementation SHA256 + sampling config` 一致时复用，否则重建。新增任务字段可写入 task sidecar，无需重新计算 geometry FK。task cache 还保存真实 `delta_time_s`、source frame mapping、连续 30 Hz mask 和 `q_delta_abs_max`。

object pose 优先读取官方根级 `object_6d_pose_v2/inspire_f1/<object>_<scene>.npz`，缺失时回退 `object_6d_pose_v1/...`，最后才兼容 episode 内旧逐帧 TXT。compact NPZ 必须包含从 `frame_0` 开始的连续键，每项为 `float32 [4,4]`。

| Tensor | Shape | 含义 |
|---|---:|---|
| `hand_points` | `[B, 1538, 3]` | 当前帧手部采样点，`hand-root_t` |
| `hand_normals` | `[B, 1538, 3]` | 当前帧手部采样法向 |
| `hand_flow` | `[B, 1538, 3]` | `hand_points_(t+1) - hand_points_t`，均在 `hand-root_t` |
| `obj_points` | `[B, 512, 3]` | 当前物体采样点，`hand-root_t` |
| `obj_normals` | `[B, 512, 3]` | 当前物体采样法向 |
| `obj_valid_mask` | `[B, 512]` | 物体点有效掩码 |
| `q_t` | `[B, 6]` | 当前手指关节角，弧度 |
| `q_next` | `[B, 6]` | 下一帧手指关节角监督，弧度 |
| `wrist_delta_translation` | `[B, 3]` | 当前腕到目标腕的相对平移，米，表达在 `hand-root_t` |
| `wrist_delta_rotvec` | `[B, 3]` | 当前腕到目标腕的相对旋转 axis-angle，弧度，表达在 `hand-root_t` |

腕部监督由 task cache 同级的30 Hz geometry sidecar 中 `wrist_pose_world` 与 `source_frame_id` 按 horizon pair 只读派生，不重写现有 task cache，因此不破坏已经导出的 Cm token sidecar manifest。

> 版本边界：3 Hz `v2` 是 wrist-aware 点监督的当前有效版本，目标点使用 `_to_frame(hand_target, wrist_current)`，manifest 声明 `hand_flow_frame=current_wrist`；dataset 按配置 fail-fast 校验该字段。`v1` 曾误用 `wrist_target`，会消除腕部刚体运动，仅保留用于历史 finger-only 实验复现，不能用于联合 wrist+q 的纯点监督。

## Frozen Cm encoder

当前 checkpoint 为 GRAB-only、16-slot Cm：

```text
obj_points + obj_normals + obj_valid_mask
hand_points + hand_normals + hand_flow
        │
        └─ frozen CmFlowModel（`torch.no_grad()`）
                └─ cm_tokens: [B, 16, 256]
```

Cm 及其 DenseToken/backbone 参数全部冻结，梯度不回传。EXP-011 使用 object-v2 GRAB+ARCTIC checkpoint；旧 EXP-010 使用早期 GRAB checkpoint，二者的 token sidecar 不可混用。

可选将 Cm 输出预提取为 `cm/cm_tokens.npy`（`[N,16,256]`）并 mmap 读取；`build_cm_cache.py` 使用 checkpoint hash 校验其有效性。`flow_mode=shuffled` 时强制在线编码，避免错误复用 normal-flow token。

## Decoder

原有整体 q Decoder 保留为 baseline；`decoder_input` 支持三种对照：`qt_cm`（baseline 主模型）、`qt_only`、`cm_only`。

```text
cm_tokens [B,16,256] ── flatten ── [B,4096]
q_t       [B,6]
             │ concat
             ▼
        [B,4102] → LayerNorm
             ▼
        Linear(4102,512) → GELU
             ▼
        Linear(512,256) → GELU
             ▼
        Linear(256,12)
             ▼
        pred_delta_q [B,6]
        pred_wrist_delta_translation [B,3]
        pred_wrist_delta_rotvec [B,3]
             │
             └─ q_t + pred_delta_q
                        ▼
                 pred_q_next [B,6]
```

### 逐手点 Cm Decoder

新增 `CmPointFlowModel`，神经网络路径不读取 `q_t`。Frozen DenseToken 由当前几何产生逐手点 `z_hand [B,1538,96]` 和 contact prior；当前手点、法向、DenseToken 特征和 contact 形成 point context。每个手点分别与16个 frozen Cm slot 构造 edge，预测 slot candidate flow 与 routing weight：

```text
[z_hand(96), hand_points(3), hand_normals(3), contact(1)]
        → hand_context [B,1538,C]

hand_context × cm_tokens [B,16,C]
        → edge [B,1538,16,2C]
        → candidate_flow [B,1538,16,3]
        → slot softmax routing [B,1538,16]
        → pred_hand_flow [B,1538,3]
```

训练目标为 cache 的对应点 `hand_flow`，使用米制 Smooth-L1；模型选择指标为 `val/hand_flow/epe_mm`。`q_t` 不进入上述网络，只在训练之外的 q fitting 使用。

启用 `predict_wrist_motion` 的新 baseline 同时回归上述12维相对运动；未保存该字段的历史 checkpoint 仍按原6维输出严格加载，但其预测腕部只能视为单位变换，不能作为腕部感知 baseline。

### Point flow → wrist SE(3) + q 后处理

`q_optimizer.py` 用现有 Inspire F1 URDF 实现可微 FK。每个 cache sample 先在 `q_t` 处把1538个当前点精确反绑到对应 hand link，随后以 `q_t`、零平移和零旋转为初值，联合优化6维独立手指 q 与当前腕到目标腕的相对 SE(3)；mimic joints 按 URDF multiplier 展开，且每步将 q clamp 到 joint limit。目标为拟合 `hand_points + pred_hand_flow`，另有可配置的小 q prior 和 wrist prior。该优化不参与训练，也不影响 point-flow checkpoint 选模。未来 arm FK 不参与目标手重建。

## 训练目标

修复后的 wrist-aware cache 将默认使用纯对应点几何监督：

```text
pred_q_next + pred_relative_wrist
        ↓ fixed-correspondence differentiable hand FK
pred_hand_points [B,1538,3]

target_hand_points = hand_points + hand_flow
loss = SmoothL1(pred_hand_points, target_hand_points; beta=0.01 m)
```

每个 batch 先在 `q_t` 将当前 cache 点精确反绑到对应 URDF link，再以预测 q 做 FK 并施加预测相对 wrist SE(3)。默认 `q_loss_weight=0`、`wrist_translation_loss_weight=0`、`wrist_rotation_loss_weight=0`、`baseline_point_loss_weight=1`；原三类参数 loss 仍计算和记录，可通过配置恢复。点 loss 使用原始米制坐标，checkpoint 按 `val/hand_points/epe_mm` 选取；同时报告 q MAE、wrist 参数误差、point EPE/RMSE 和 identity-hand point EPE。只有 Decoder 参数更新。

该 loss 的训练开销显著高于直接参数 loss：每个 batch 都要执行上述反绑、预测 q 的可微 FK，并对1538个点应用相对腕变换后反传。当前实测约 `115—140 ms/step`，直接 q/translation/rotation loss 约 `15 ms/step`；后续可通过缓存 point-to-link/local-point 绑定和向量化 link transform 优化，但不得改变对应点语义。

历史配置默认 `baseline_point_loss_weight=0`、`q_loss_weight=1`，因此旧 checkpoint 的训练/评估语义保持兼容。旧 checkpoint 未保存 `prediction_target` 时仍按历史 direct-q 语义加载。

## 训练与评估边界

当前20-episode诊断沿用历史 episode-random manifest（16/2/2），只适合小规模 checkpoint 对照。全量数据按用户指定采用 object-disjoint：同一 object 的全部 episode 只能属于 train、val、test 中一个 split。30 Hz geometry 层保留576个 episode；3 Hz task 层会排除没有任何连续 stride=10 pair 的 episode，当前保留568个，但不得重新随机划分，其余 episode 继承原 object split。

## 可视化

`python -m src.task.CmDecoder.viewer --object <name> --scene <id>` 默认加载历史全量 object-disjoint 3 Hz `qt_cm` checkpoint；`--decoder-checkpoint` 可覆盖。viewer 从 checkpoint 自带 manifest 读取对应 cache pair，并提供统一的可视化模式、Pair/Rollout step、当前/GT/预测 mesh、手/物体采样点和 GT flow 控件。`--rollout-trajectory <file.npz>` 只额外启用 `Closed-loop rollout` 模式，不替换基础 UI；未显式传入轨迹时该模式显示为灰色禁用。界面图例固定约定：蓝色当前手（rollout 中为起始预测）、绿色 GT、橙色预测、灰色物体点、浅灰手点、品红 GT flow 箭头；mesh 为半透明表面。腕部感知 baseline 直接显示预测相对 SE(3) 与 `pred_q_next`；逐点 checkpoint 会先预测 hand flow，再调用上述联合 wrist+q fitting。GT 使用数据中的真实相对腕变换，预测手使用预测/优化的相对腕变换；当前 arm FK 只定义当前坐标系，未来 arm FK 不参与。历史6维 baseline 兼容加载时预测腕变换为单位变换。

## Cache binding 与向量化点变换（2026-08-23）

v4 geometry cache 现在为每个 episode 额外保存静态点绑定：

```text
hand_point_link_index [1538] int16
hand_points_local     [1538,3] float32，米，link-local
```

`hand_point_link_index` 使用 manifest 中的稳定 URDF 全 link order；绑定由
URDF visual origin/scale、固定 surface face/barycentric sampling 和 sample seed
决定，不随 frame 或 horizon 改变。v2 horizon task cache 携带同一静态字段，避免
训练 Dataset 访问 geometry sidecar 时产生额外 pair 映射。

旧 cache 可用 `migrate_point_bindings.py` 补字段；迁移复用已有
`q_full/hand_points_world/wrist_pose_world`，只对每个 episode 做首帧绑定验证，
不重算逐帧几何。Dataset 以 `[B,1538]` / `[B,1538,3]` collate 后检查 batch
内绑定一致，再压回静态 `[1538]` / `[1538,3]`。

预测 q 的 FK 仍需对每个 batch 可微执行；但各 link 的点变换已改为：

```text
link_tf [B,L,4,4]
  → gather(hand_point_link_index) [B,1538,4,4]
  → homogeneous matmul(hand_points_local) [B,1538,3]
```

因此训练不再执行当前 `q_t` 的 FK、逐 link 逆矩阵和点反绑。真实 cache
smoke 与旧路径最大差约 `1.2e-7 m`，CPU batch=32 点重建约 `2.7×` 加速。

## GRAB 30Hz 无配对重定向 smoke test

`src/task/CmDecoder/grab_retarget.py` 使用 subject-template GRAB 的 `right/`
序列，固定 `stride=1`（相邻缓存帧，`delta_time_s=1/30 s`）。GRAB 手点、法向
和 `hand_flow` 只用于 Frozen Cm 产生 `[1,16,C]` tokens；机器人分支使用
Inspire F1 当前 q 的 `[1,1538,3]` 点和法向，以及物体在当前机器人腕系的
`[1,512,3]` 点云。机器人初始 q0 为六个独立关节 limit 中点，腕部位姿为物体
点云中心外侧 `approach_distance=0.12 m`，z 轴指向物体中心。每帧通过
point-flow→固定对应点 q fitting 得到 q 和相对 wrist SE(3)，再更新机器人腕位姿。

输出 NPZ/PNG 仅用于无配对 sanity check，不构成 GRAB→Inspire F1 的真实 q 监督。
`--start-frame` 用于选择同一序列内的起始缓存帧；评估必须确认所选窗口的
`obj_valid_mask` 非空，禁止把 candidate 为空时的 padding 物体点当作有效输入。

## GRAB→ARCTIC MANO 跨域简化评估

`mano_grab_point_config.py` 使用 combined object-v2 split 中过滤出的 GRAB
sequence 训练 `CmPointFlowModel`。输入仍为当前手点、法向、物体点和冻结 Cm
tokens，监督为 GRAB 当前 wrist frame 下的下一帧 MANO `hand_flow`；q/URDF 不参与。

`arctic_mano_eval.py` 在 ARCTIC 上完全不读取 GRAB 帧。每个 ARCTIC 当前帧的真实
`hand_flow` 只用于生成 source action token，decoder 的 target geometry 使用当前
ARCTIC MANO 点。teacher-forced 模式用真实当前点；rollout 模式把预测的下一点从当前
wrist frame 变换到下一帧 ARCTIC wrist frame，再继续预测。rollout 使用记录的 ARCTIC
wrist pose 处理坐标系变化，并暂时复用对应帧真实法向，因此这是 MANO 点流跨域探针，
不是完整的机器人重定向或无 GT 自主动作生成。
