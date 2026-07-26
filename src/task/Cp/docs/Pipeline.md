# Cp：从物体任务先验到人手动作的当前实现流程

本文描述 `src/task/Cp` 当前可执行的最小闭环实现，以及 Stage 5 数据的生成、训练顺序和已完成的验证。设计目标来自 [框架.md](框架.md)：从完整 human demo 中抽取物体任务先验 `C_p`，再结合当前手状态生成动作先验 `C_m`，最后预测未来手部点流。

## 1. 总体数据流

```text
完整 human demo（当前/未来物体与当前/未来手）
        │
        ├─ Stage 5：clean object effect
        │       ├─ object contact field c_gt
        │       ├─ object flow F_o_gt
        │       ├─ current hand H_t
        │       └─ hand flow F_m_gt
        │
        ├─ Stage B：G_p_gt → Cp → [c_hat, F_o_hat]
        │
        └─ Stage C：Cp + H_t → C_m_hat → frozen D_m → F_m_hat
                                                │
                                                └─ F_m_hat ≈ F_m_gt
```

坐标系始终是当前时刻的 hand-root frame（`hand_root_t`）。未来手和未来物体会先变换到当前手根坐标系，再计算 point flow；因此所有流都有明确物理含义。

## 2. Stage 5：GRAB 原始数据到 Cp 样本

实现：`process/stage5/prepare_cp.py`。

该脚本复用 `process.stage4.prepare_cm.build_stage4_sequence` 的 MANO 重建、固定物体表面采样和 current-hand-frame 变换，只将输出 schema 扩展成 Cp 所需形式：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `obj_points`, `obj_normals` | `T x 4096 x 3` | 当前物体的 clean surface pool |
| `obj_flow_gt` | `T x 4096 x 3` | 当前物体点到未来物体点的流 |
| `obj_contact_gt` | `T x 4096` | 连续接触 target，`clip(1 - d/0.02, 0, 1)` |
| `obj_candidate_mask_5cm` | `T x 4096` | 运行时 512 点的候选掩码 |
| `hand_points`, `hand_normals` | `T x 1538 x 3` | 当前 MANO face-center surface |
| `hand_flow` | `T x 1538 x 3` | 当前手点到未来手点的流 |
| `raw_frame_id`, `next_raw_frame_id`, `time_delta_sec` | `T` | temporal-pair identity |

输出目录固定为：

```text
processed_data/generated/stage5/<subject>/<sequence>_<side>.npz
```

当前 GRAB 的实际布局是：

```text
dataset/GRAB/data/grab/<subject>/*.npz
dataset/GRAB/data/tools/object_meshes/contact_meshes/*.ply
```

因此需要把 `--grab-root` 指向 `dataset/GRAB/data`，例如：

```bash
PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m process.stage5.prepare_cp \
  --grab-root dataset/GRAB/data \
  --seq s1/cylindersmall_pass_1 \
  --side right \
  --pair-stride 3 --pair-hop 4 \
  --output-root processed_data/generated/stage5 \
  --device cuda
```

运行前需要存在 MANO `MANO_LEFT.pkl`/`MANO_RIGHT.pkl`（当前在 `dataset/arctic/data/body_models/mano`），且 `graspenv` 需要 `smplx` 和 `chumpy`。

## 3. 运行时数据集

实现：`src/task/Cp/dataset.py` 中的 `Stage5CpDataset`。

每个 item 是一个 temporal pair：

1. 以 `sequence id + side + raw frame id + seed` 生成稳定随机种子；
2. 从完整 4096 点池的 `obj_candidate_mask_5cm` 中抽取至多 512 个物体点；
3. 候选不足时使用 invalid padding，`obj_valid_mask` 保证 object effect loss 忽略 padding；
4. 手点始终是完整 1538 个 MANO face-center 点。

训练/验证切分通过 `src.base.make_file_split_dataloaders` 按 sequence 文件分组，不随机拆分同一 sequence 的重叠 temporal pairs。

## 4. 模型实现

实现：`src/task/Cp/model.py`，主类为 `CpHumanClosureModel`。所有阶段都在同一个 module 内，Runner 用 `meta.stage` 选择训练目标。

### 4.1 Stage A：旧 Cm 空间到无泄漏 hand-flow decoder

```text
旧 Cm checkpoint（warm start）
      → frozen hand_motion_encoder + SlotAttention → C_m
      → trainable Cm hand decoder(C_m, z_hand, H_t)
      → F_m_hat
```

旧 checkpoint 中的 `hand_motion_encoder`、`slot_attention`、object-flow decoder 被兼容加载后冻结；只随机初始化并训练新增的 `hand_context_encoder`、`hand_token_score`、`hand_articulation_decoder` 和 `wrist_decoder`。hand decoder 的输入中不含 future hand、GT `hand_flow` 或 `wrist_delta`。训练完成的完整 Cm checkpoint（包括 decoder）将在 Stage C 中整体冻结。

### 4.2 Stage B：object effect 到 Cp

```text
[O_t position, O_t normal, c_gt, F_o_gt]
      → cp_encoder → SlotAttention(8) → C_p
      → Cp effect decoder(C_p, O_t position, O_t normal)
      → [contact logit, F_o_hat]
```

effect decoder 为每个 object-point / Cp-slot 先预测共享 effect feature，再由独立的 mixture、contact 和 flow heads 解码。slot softmax 加权聚合避免对所有 slots 的简单平均；独立 head 避免 contact BCE 与毫米级 flow 在最终投影层发生梯度冲突。object flow 以 `object_flow_scale_m=0.01` 归一化后计算 Smooth L1。contact 的优化项与 BCE 梯度相同，但记录并最小化 excess BCE：`BCE(logit, c_gt) - H(c_gt)`；其中 `H(c_gt)` 是软标签给出的不可约理论下界，因此 excess 为 0 表示达到最优。Stage-C 加载并冻结这些 Cp effect 模块，使 `C_p` 的 task-effect 语义固定。

### 4.3 Stage C：真正的人类闭环

```text
frozen DenseToken z_hand + current hand state E_state(H_t)
      ├─ query C_p via cross attention
      └─ SlotAttention(16)
              ↓
          C_m_hat
              ↓
     frozen hand decoder(C_m_hat, H_t)
              ↓
           F_m_hat
```

Stage-C 仅优化独立的 `closure_hand_state`、`cp_to_hand` 和 `closure_cm_slots`。Cm decoder 使用完全不同且冻结的 current-state encoder。loss 复用 Cm 的 wrist/articulation 分解：

```text
L_wrist + L_weighted_articulation + 0.2 L_global_articulation
```

不直接比较 `C_m_hat` 和 teacher slot，因为 slot index 存在置换自由度。

## 5. BaseRunner 覆写点

`CpHumanClosureRunner` 继承共享 `src.base.BaseRunner`，只覆写：

| 方法 | 任务职责 |
| --- | --- |
| `make_dataloaders` | 创建 Stage 5 train/val loader |
| `build_model` | Stage C 时加载并冻结完整 Cm hand decoder 与 Cp effect checkpoint |
| `step` | 根据 `meta.stage` 计算阶段损失与指标 |

因此 optimizer、AMP、checkpoint、distributed、日志和 overfit mode 均使用 `src/base` 的 公共实现。

## 6. 训练顺序

先训练 A 和 B，之后才允许训练 C：

```bash
# A: legacy Cm warm start -> train only Cm->Fm decoder
PYTHONPATH=. python -m src.task.Cm.train \
  --config src/task/Cm/configs/hand_decoder_warm_start.yaml \
  --data processed_data/generated/stage5 \
  --output-dir outputs/train/cm_hand_decoder

# B: object effect -> Cp
PYTHONPATH=. python -m src.task.Cp.train \
  --config src/task/Cp/configs/baseline.yaml \
  --set meta.stage=cp_effect \
  --data processed_data/generated/stage5 \
  --output-dir outputs/train/cp_effect

# C: frozen complete Cm decoder + frozen B + trainable Cp -> Cm generator
PYTHONPATH=. python -m src.task.Cp.train \
  --config src/task/Cp/configs/baseline.yaml \
  --set meta.stage=closed_loop \
  --set meta.cm_hand_checkpoint=outputs/train/cm_hand_decoder/<run>/checkpoints/latest.pt \
  --set meta.cp_effect_checkpoint=outputs/train/cp_effect/<run>/checkpoints/latest.pt \
  --data processed_data/generated/stage5 \
  --output-dir outputs/train/cp_closed_loop
```

通常 Stage B/C 应设置 `data.active_only=true`，避免完全没有 5cm object candidate 的 pair 对 object effect 训练造成无效 loss。

## 7. 可视化

实现：`src/task/Cp/visualize.py`。

颜色约定：灰色为当前手，绿色为 GT future hand，红色为预测 future hand。支持 Open3D 交互窗口和静态 PNG：

```bash
DISPLAY=localhost:10.0 PYTHONPATH=. python -m src.task.Cp.visualize \
  --checkpoint <closed_loop_checkpoint> \
  --input processed_data/generated/stage5/s1/cylindersmall_pass_1_right.npz \
  --pair 27 --device cuda \
  --save outputs/train/cp_closed_loop/closure.png
```

`--check-only` 只做 checkpoint 和一次前向兼容性检查，不开启窗口。`--save` 可在无窗口环境输出静态图。

交互窗口采用与 `Cm/visualize.py` 相同的深色背景、GT 绿色、预测红色和稀疏流线表示。按键如下：

| 按键 | 操作 |
| --- | --- |
| `A` / `D`（或方向键） | 上一 / 下一 temporal pair |
| `V` | Human → Object flow → Both |
| `G` | 物体流 GT → Pred → Both |
| `C` | 当前物体点颜色 Off → GT contact → Pred contact |
| `L` | 开关稀疏的当前→未来物体流线段 |
| `H` | 开关 GT / predicted future hand |
| `R` | 重置相机 |

## 8. 已完成验证

数据验证使用 `s1/cylindersmall_pass_1` right hand：

```text
166 temporal pairs
93 active pairs (至少一个 5cm object candidate)
minimum object-to-hand distance: 0.075 mm
```

已进行单样本过拟合诊断；新的 Cp effect 使用有真实物体运动的 pair 42：

| 阶段 | 指标 | 结果 |
| --- | --- | --- |
| A：Cm decoder warm start | hand-flow EPE | 12.946 mm → **0.347 mm** |
| A：Cm decoder warm start | wrist translation / rotation | **0.048 mm / 0.028°** |
| B：旧共享 effect head | object-flow EPE | **未通过**（17.700 mm；已废弃） |
| B：新独立 effect heads（标准训练入口） | object-flow EPE | **0.410 mm**（pair 42，500 step） |
| B：新独立 effect heads | contact BCE 理论下界 | **0.3641** |
| B：新独立 effect heads | excess contact BCE | **0.2263** |
| C：真实 `Cp → Cm → Fm` 闭环 | hand-flow EPE | 6.653 mm → **0.346 mm** |

上述旧 B/C 结果来自共享 effect head，现已确认 Cp object-flow 未收敛，**不可作为通过结果或用于后续闭环训练**。新的独立 effect heads 已在真实移动 pair 42（GT 平均物体位移 35.3 mm）上通过标准训练入口完成 500-step 诊断：object-flow EPE 为 **0.410 mm**。该 pair 的软接触标签熵下界为 **0.3641**，当前 BCE 为 **0.5904**，故 excess contact BCE 为 **0.2263**；这正是还可优化的接触预测误差。仍需使用新架构训练完整 Stage B，再重新训练 Stage C；此前的闭环静态图只保留为历史记录。

## 9. 当前边界与后续工作

当前实现已完成真实 Cm 接口、warm-start、冻结边界和单样本闭环验证；以下仍待完成：

1. 多 sequence 的正式训练与 sequence-held-out 验证；
2. `H-only`、zero-`C_p`、shuffle-`C_p` 三项必要 ablation；
3. fingertip EPE、direction cosine、magnitude ratio 的完整报告；
4. `C_m_hat -> object flow` effect-cycle 评估；
