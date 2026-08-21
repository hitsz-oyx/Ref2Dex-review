# CmDecoder 张量架构

## 数据与坐标

每个样本是 30 Hz 的相邻帧 `(t, t+1)`。arm q 只用于完整 FK 恢复腕部位姿；Decoder 不预测 arm，目标始终是 6 个 Inspire F1 手指关节角。

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

## Frozen Cm encoder

当前 checkpoint 为 GRAB-only、16-slot Cm：

```text
obj_points + obj_normals + obj_valid_mask
hand_points + hand_normals + hand_flow
        │
        └─ frozen CmFlowModel（`torch.no_grad()`）
                └─ cm_tokens: [B, 16, 256]
```

Cm 及其 DenseToken/backbone 参数全部冻结，梯度不回传。

可选将 Cm 输出预提取为 `cm/cm_tokens.npy`（`[N,16,256]`）并 mmap 读取；`build_cm_cache.py` 使用 checkpoint hash 校验其有效性。`flow_mode=shuffled` 时强制在线编码，避免错误复用 normal-flow token。

## Decoder

`decoder_input` 支持三种对照：`qt_cm`（主模型）、`qt_only`、`cm_only`。主模型使用 `qt_cm`。

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
        Linear(256,6)
             ▼
        pred_q_next [B,6]
```

## 训练目标

```text
loss = SmoothL1(pred_q_next, q_next)
```

记录 `MAE`（弧度、角度）和 `RMSE`（弧度）；只有 Decoder 参数更新。
