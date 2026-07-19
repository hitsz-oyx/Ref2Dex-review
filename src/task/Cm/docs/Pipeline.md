# CmAction：GRAB Raw → Stage 4 → Cm → Object Point Flow

## 范围与命名

本任务的物理目录暂时保持为 `src/task/Cm`，以保留现有的 dense-token checkpoint 路径；任务的完整语义名称统一写为 **CmAction**。本阶段只定义和训练 hand-centered `C_m`，不实现 `C_p`。

`C_m` 是一组锚在手表面的时序动作 token：它描述手的哪里在运动、如何运动、当前与物体的交互上下文是什么。它的监督目标是预测当前物体点的下一时刻位移。`C_p` 将来应当从 `C_m` 投影到物体表面后产生，而不是在本阶段平行地从未来物体几何中编码。

## 信息边界

每个训练样本只允许 Cm encoder 看到：

```text
O_t, H_t, ΔH_t, ΔT_wrist(t→t+1), dense-token(O_t, H_t)
```

未来物体 `O_{t+1}` 不会落盘到 Stage 4；它只在生成时变成监督字段 `obj_flow_gt = O_{t+1}^{h_t} - O_t^{h_t}`。因此 Cm 不可能通过未来物体形状或位姿直接读取 point flow。

当前 dense-token checkpoint 是：

```text
src/task/Cm/densetoken_ckpt/best.pt
```

它是 `correspondence_ptv3_v2` 的 hand-root 模型，输入为 512 个 object point 和 1538 个 MANO face-center hand point。CmAction 完全复用它保存的配置、512 点采样种子和 PTv3 权重；默认冻结它，只训练 temporal Cm bottleneck 和 flow decoder。

## Stage 4：直接从原始 GRAB 生成

Stage 4 不依赖 Stage 2 或 Stage 3。它调用 `process.GRAB.raw.GRABRawAdapter` 重建 MANO、稳定采样同一物体的 4096 个表面点，再直接写入连续帧对。与 Stage 2 不同，它不按 5cm 删除帧，因为动作时序必须连续保留；5cm 仅作为与 dense-token checkpoint 对齐的运行时 object candidate mask。

对每个时间对 `(t, t+Δ)`，所有点都变换到 **t 时刻** hand-root：

```text
x_t        = R_h,t^T (x_t^world     - p_h,t)
x_{t+Δ}^{h_t} = R_h,t^T (x_{t+Δ}^world - p_h,t)
F_obj      = x_{t+Δ}^{h_t} - x_t
F_hand     = y_{t+Δ}^{h_t} - y_t
ΔT_wrist   = (T_world_from_hand,t)^-1 T_world_from_hand,t+Δ
```

这个定义保留 wrist 的全局平移/旋转与手指关节运动，且 object flow 与 hand flow 的坐标系一致。不要对每一帧分别 hand-root 规范化后直接相减。

一个 Stage 4 NPZ 是一条 sequence、一个 side；关键字段如下。

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| `obj_points`, `obj_normals` | `P × 4096 × 3` | `O_t`；dense-token 的当前 object context |
| `hand_points`, `hand_normals` | `P × 1538 × 3` | `H_t`；dense-token/current hand context |
| `hand_flow` | `P × 1538 × 3` | `H_{t+Δ}^{h_t} - H_t`，Cm 的动作输入 |
| `wrist_delta` | `P × 4 × 4` | `T_hand_t_from_hand_t+Δ`，显式 wrist SE(3) |
| `obj_flow_gt` | `P × 4096 × 3` | 唯一 future-object 监督，不作为模型输入 |
| `obj_candidate_mask_5cm` | `P × 4096` | 对齐旧 checkpoint 的 512 point 采样候选 |
| `obj_to_hand_min_dist`, `hand_to_obj_min_dist` | `P × 4096` / `P × 1538` | 当前帧接触、active-pair 和诊断统计 |
| `raw_frame_id`, `next_raw_frame_id` | `P` | 可复现采样与时间索引 |

其中 `P = ceil((T - pair_stride) / pair_hop)`；`pair_stride` 是 future frame 的偏移，`pair_hop` 是相邻样本 current frame 的偏移。默认 `ds_rate=1`、`pair_stride=3`、`pair_hop=1`，因此使用所有 GRAB 原始帧并构造重叠窗口：`1→4、2→5、3→6…`。GRAB 是 120Hz，所以 point-flow 的时间间隔是 `3 / 120 = 25ms`。并使用当前 5cm candidate mask 完全复现 dense-token 的 `epoch=0` 512 点采样。

### 生成命令

单条序列 smoke 数据：

```bash
PYTHONPATH=. python -m process.stage4.prepare_cm \
  --seq s1/bowl_pass_1 \
  --side right \
  --ds-rate 1 \
  --pair-stride 3 \
  --pair-hop 1 \
  --device cuda \
  --output-root processed_data/generated/stage4/grab_cm_raw_stride3_smoke
```

批量运行可传入 GRAB manifest：

```bash
PYTHONPATH=. python -m process.stage4.prepare_cm \
  --manifest tmp/manifests/grab_subset_100.csv \
  --side both \
  --ds-rate 1 \
  --pair-stride 3 \
  --pair-hop 1 \
  --device cuda \
  --output-root processed_data/generated/stage4/grab_cm_raw_stride3
```

缺少某一侧手轨迹的 GRAB sequence 会跳过该 side；它不是错误。默认不镜像左手，因为现有 dense-token checkpoint 的 full-GRAB 训练配置也没有镜像；只有明确决定改为右手统一约定时才使用 `--mirror-left-to-right`，且 Stage 4 和后续 Cm 训练必须保持一致。

## CmAction 模型

```text
(O_t, H_t) -- frozen PTv3 --> (Z_t^o, Z_t^h)
(Z_t^h, y_h, n_h, ΔH_t, contact_h, vec(ΔT_wrist)) -- pointwise MLP --> U_t^h
U_t^h -- Slot Attention --> C_m ∈ R^(K×256)
(Z_t^o, x_o, n_o, C_m, soft hand anchors) -- geometric flow-edge decoder --> F_hat_obj,t
```

当前版本保留完整手部动作场：`Z_t^h` 提供当前交互状态，`y_h/n_h` 提供
局部几何，`ΔH_t` 提供局部运动，冻结 dense hand-contact 概率提供接触先验，
`ΔT_wrist` 提供整体 wrist SE(3)。Slot Attention 是唯一替换项：它取代旧的
scene cross-attention、activity head 和 hard Top-K，而不改动 object-side
几何 flow decoder。

默认 `K=16`、Slot Attention 迭代 3 次。导出字段为：

```text
cm_tokens          # [K, 256] hand-side temporal action slots
cm_anchor_pos      # [K, 3] soft hand-region center
cm_anchor_normal   # [K, 3] normalized soft hand-region normal
cm_hand_flow       # [K, 3] soft hand-region mean local displacement
cm_assignment      # [K, 1538], point-to-slot assignment; sum over K is 1
cm_slot_weights    # [K, 1538], per-slot aggregation weights; sum over hand points is 1
```

`cm_anchor_pos/cm_hand_flow/cm_anchor_normal` 都由 `cm_slot_weights` 对手点软
聚合而来；`cm_assignment` 描述每个手点在各 slot 间的归属。不存在 top-K hard
anchor 或 activity loss。object flow decoder
保留每个 object-point 到每个 slot 的相对位置、slot 平均手流和 slot 法向，
并以 masked Smooth-L1 训练；没有可采样 object candidate 的帧会被 mask，
不会制造假的 zero-object 输入。

训练只优化 flow loss；另记录 `slot_assignment_entropy`（手点在 slot 间的分配熵）与
`slot_weight_overlap`（不同 slot 聚合权重的余弦重叠），用于发现 slot collapse，二者均不参与反传。

这与早期的 `top-K anchor + activity head` Cm head 参数结构不兼容；必须从
新的 CmAction run 训练，不能恢复旧 Cm head checkpoint。冻结的
DenseToken/PTv3 checkpoint 不受影响。

## BaseRunner 训练、评估与导出

CmAction 现在与 `correspondence_ptv3_v2` 一样使用 `TaskConfig + BaseRunner + CheckpointManager`：标准 checkpoint 保存有效 config、Stage 4 metadata、optimizer、scheduler 和新的 Cm head。冻结的 dense-token 权重不重复写入 Cm checkpoint；模型按 `meta.dense_checkpoint` 重载它。

训练时 DenseToken checkpoint 始终为 `eval/no_grad`，避免 Cm 训练破坏已经验证过的 static correspondence。常规训练：

```bash
PYTHONPATH=. python -m src.task.Cm.train \
  --config src/task/Cm/configs/baseline.yaml \
  --data processed_data/generated/stage4/grab_cm_raw_stride3 \
  --output-dir outputs/train/cm_grab_v0 \
  --device cuda
```

Stage 4 保留完整动作序列；而 `data.active_only=true`（默认）只让具有 5cm object candidate 的 pair 进入第一阶段 flow loss，避免将没有 object token 的非接触 pair 当作伪零监督。验证/诊断时可覆写为 `data.active_only=false`。

固定两条 active pair 的 overfit 配置：

```bash
PYTHONPATH=. python -m src.task.Cm.train \
  --config src/task/Cm/configs/overfit.yaml \
  --device cuda
```

该配置设定 `data.min_object_flow_norm=0.035`（candidate point 平均流速至少 35mm），避开物体几乎静止的接触 pair；它检验的是 point-flow 拟合而非零流 baseline。

常规评估直接从 checkpoint 读取训练时保存的 config：

```bash
PYTHONPATH=. python -m src.task.Cm.eval \
  --checkpoint outputs/train/cm_grab_v0/<run>/checkpoints/best.pt \
  --device cuda
```

导出 Cm 与其预测点流：

```bash
PYTHONPATH=. python -m src.task.Cm.extract \
  --stage4-input processed_data/generated/stage4/grab_cm_raw_stride3_smoke \
  --checkpoint outputs/train/cm_action_overfit_raw_stride3/<run>/checkpoints/latest.pt \
  --output-root outputs/cm_tokens/grab_cm_raw_stride3_smoke \
  --device cuda
```

输出 NPZ 会保存每个 pair 的 sampled object identity、valid mask、`pred_obj_flow`、`obj_flow_gt`、Slot Attention 的所有 Cm 字段，并记录该文件的 masked `flow_mse`。这使后续的可视化、ablation 和 `C_p = ObjectProjection(C_m, correspondence, scene)` 能不重跑 PTv3。

`--start-pair` 与 `--max-pairs` 可用于只导出某一段序列（例如先检查实际进入 5cm candidate 区间后的 pair）。

## 交互式 point-flow 可视化

可视化直接从 Cm checkpoint 预测 flow，不依赖预导出的 token 文件。当前 object point 是灰色、GT future point/line 是绿色、预测 future point/line 是红色；在 Both 模式中三者同时显示，便于检查位移方向与大小。

```bash
PYTHONPATH=. python -m src.task.Cm.visualize \
  --checkpoint outputs/train/cm_action_overfit_raw_stride3/<run>/checkpoints/latest.pt \
  --input processed_data/generated/stage4/grab_cm_raw_stride3_smoke/s1/bowl_pass_1_right.npz \
  --device cuda
```

按键：`A/D`（或左右方向键）切 pair；`G` 循环 `GT → Pred → Both`；`L` 开关稀疏 flow line；`H` 开关 future-hand context；`R` 重置相机。可先加 `--check-only` 检查预测 shape、有效 object 数和 flow MSE，不创建窗口。默认 `--pair` 是 active-pair 过滤后的索引；加入 `--include-inactive` 后它才对应完整 Stage 4 pair 顺序。

## 先验验证项

在扩大训练前应完成以下验证：

1. Stage 4 中 `obj_flow_gt` 与 `hand_flow` 使用同一个 `hand_root_t`，而不是每帧各自 root。
2. 以同一 raw frame id 重跑，512 selected object indices 与 dense-token 旧数据的 epoch-0 采样一致。
3. 打乱 `hand_flow` 或 `wrist_delta` 后 flow 指标应显著变差；否则 Cm 可能没有使用动作信息。
4. `cm_slot_weights` 的每个 slot 应形成有限的手部区域；其 `cm_hand_flow` 应能区分运动手指、稳定抓持与非接触运动，而不是所有 slot 均匀覆盖整只手。`cm_assignment` 则应在每个 hand point 上形成有区分度的 slot 归属。
5. 任何 future object 字段都不能进入 `FrozenDenseTokenEncoder` 或 `CmFlowModel`；`obj_flow_gt` 只在 loss/eval 中读取。

## Cp 的预留边界

本代码不创建 `C_p`。下一阶段应冻结或 stop-gradient 当前 `C_m`，利用当前 correspondence 将 hand token 投影到 object point，再从 `(Z_t^o, C_m, F_hat_obj)` 压缩 object-side effect token。应对齐两者预测的 object effect，而不要把 `C_m` 与 `C_p` embedding 强行做成同一特征。
