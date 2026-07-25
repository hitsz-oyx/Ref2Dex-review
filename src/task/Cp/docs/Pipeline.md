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

### 4.1 Stage A：Cm teacher 到无泄漏 hand-flow decoder

```text
[H_t position, H_t normal, F_m_gt, zero]
      → teacher_encoder → SlotAttention(16) → C_m_teacher
      → hand decoder(C_m_teacher, H_t position, H_t normal)
      → F_m_hat_teacher
```

`hand_flow` 只允许进入 `teacher_encoder` 以构造 Stage-A teacher token；`hand_decoder` 的输入中不含 future hand、`hand_flow` 或 `wrist_delta`。训练完成后，Stage-C 仅加载并冻结 `hand_decoder`。

### 4.2 Stage B：object effect 到 Cp

```text
[O_t position, O_t normal, c_gt, F_o_gt]
      → cp_encoder → SlotAttention(8) → C_p
      → Cp effect decoder(C_p, O_t position, O_t normal)
      → [contact logit, F_o_hat]
```

使用有效 object point 的 Smooth L1 object-flow loss 与 BCE contact loss。Stage-C 加载并冻结 `cp_encoder`、`cp_slots`、`cp_decode`，使 `C_p` 的 task-effect 语义固定。

### 4.3 Stage C：真正的人类闭环

```text
current hand state E_state(H_t)
      ├─ query C_p via cross attention
      └─ SlotAttention(16)
              ↓
          C_m_hat
              ↓
     frozen hand decoder(C_m_hat, H_t)
              ↓
           F_m_hat
```

Stage-C 仅优化 `hand_state`、`cp_to_hand` 和 `cm_slots`。loss 为：

```text
SmoothL1(F_m_hat, F_m_gt)
```

不直接比较 `C_m_hat` 和 `C_m_teacher`，因为 slot index 存在置换自由度。

## 5. BaseRunner 覆写点

`CpHumanClosureRunner` 继承共享 `src.base.BaseRunner`，只覆写：

| 方法 | 任务职责 |
| --- | --- |
| `make_dataloaders` | 创建 Stage 5 train/val loader |
| `build_model` | Stage C 时加载并冻结 Stage A/B checkpoint 子模块 |
| `step` | 根据 `meta.stage` 计算阶段损失与指标 |

因此 optimizer、AMP、checkpoint、distributed、日志和 overfit mode 均使用 `src/base` 的公共实现。

## 6. 训练顺序

先训练 A 和 B，之后才允许训练 C：

```bash
# A: Cm teacher -> hand flow decoder
PYTHONPATH=. python -m src.task.Cp.train \
  --config src/task/Cp/configs/baseline.yaml \
  --set meta.stage=hand_decoder \
  --data processed_data/generated/stage5 \
  --output-dir outputs/train/cp_hand_decoder

# B: object effect -> Cp
PYTHONPATH=. python -m src.task.Cp.train \
  --config src/task/Cp/configs/baseline.yaml \
  --set meta.stage=cp_effect \
  --data processed_data/generated/stage5 \
  --output-dir outputs/train/cp_effect

# C: frozen A/B + trainable Cp -> Cm generator
PYTHONPATH=. python -m src.task.Cp.train \
  --config src/task/Cp/configs/baseline.yaml \
  --set meta.stage=closed_loop \
  --set meta.hand_decoder_checkpoint=outputs/train/cp_hand_decoder/<run>/checkpoints/latest.pt \
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

## 8. 已完成验证

数据验证使用 `s1/cylindersmall_pass_1` right hand：

```text
166 temporal pairs
93 active pairs (至少一个 5cm object candidate)
minimum object-to-hand distance: 0.075 mm
```

对固定的一个 active pair（pair 27）进行了 200-step 单样本过拟合：

| 阶段 | 指标 | 结果 |
| --- | --- | --- |
| Stage A | hand-flow EPE | 206.3 mm → 1.32 mm |
| Stage B | contact BCE | 0.585 → 6.4e-4 |
| Stage B | object-flow EPE | 203.6 mm → 14.5 mm |
| Stage C | hand-flow EPE | 124.1 mm → 1.316 mm |
| Stage C | hand-flow P90 EPE | 2.146 mm |

对应静态可视化在：`outputs/train/cp_overfit_closed/closure_overfit.png`。红色预测未来手与绿色 GT 未来手已基本重合，说明当前代码路径、冻结 checkpoint 加载和闭环梯度均可运行。

## 9. 当前边界与后续工作

当前实现的目标是首先跑通人类闭环。它已验证数据、三阶段 checkpoint 链路、冻结语义、过拟合和可视化，但尚未完成：

1. 多 sequence 的正式训练与 sequence-held-out 验证；
2. `H-only`、zero-`C_p`、shuffle-`C_p` 三项必要 ablation；
3. fingertip EPE、direction cosine、magnitude ratio 的完整报告；
4. `C_m_hat -> object flow` effect-cycle 评估；
5. 将现有 `Cm`/DenseToken checkpoint 作为外部 frozen teacher 的兼容接入。
