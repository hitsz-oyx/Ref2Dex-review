# PointWorldWAM 研究日志

## 实验：V0 E0-E3 PointWorld GRAB forward overfit

**假设**

PointWorld small-droid 的 PTv3 interaction backbone 经新的纯几何 adapter 微调后，能够从
GT MANO hand point tracks 预测 GRAB 刚体 object tracks；有效模型的 GT-hand 结果应显著
优于 static/shuffled hand。

**观察到的失败 / 现象**

仓库起初没有 `third_party/PointWorld`，本地 Stage2 目录也没有有效 sequence。第一次训练
还发现按随机单 batch 最低 loss 保存 `best.pt` 会选到比 final 更差的模型；第一版薄适配
也漏掉了官方 `BaseModel.forward` 中对 normalized displacement 的反归一化。

**诊断**

官方 `DynamicsPredictor` 可绕过 RGB-D/DINO scene encoder 独立使用。现有
`interaction_dynamics_v1` cache 由相同 GRAB raw geometry 构建，保留了稳定 point ID、
world 坐标和 meter 单位，可用于 V0。checkpoint 应用全体 overfit 窗口的 GT-hand ADE
选择，不能使用单 batch loss。predictor 输出必须先用 DROID per-timestep statistics 恢复
为 meter；修正前的训练数字已废弃，不用于结论。

**改动**

- 以 submodule 接入 PointWorld commit `05484826dfef74cbe278a3974179a5a16705d35d`；
- 从 small-droid checkpoint partial-load 448 个 PTv3 backbone、共 464 个匹配张量；
- 构造 11 帧、1024 object points、256 hand points 的固定 ID 窗口；
- 使用 normal/dist-to-hand 与 normal/velocity/acceleration adapter；
- 使用官方 DROID per-timestep output statistics 将 predictor displacement 反归一化到 meter；
- 对两个高运动 sequence 的 64 个窗口训练 200 step motion-weighted Huber；
- 增加 GT/static/shuffled、ADE/FDE/translation/rotation 诊断和轨迹图；
- checkpoint 改为每 25 step 用全 64 窗口 GT ADE 选择，并额外保存 last。

**结果**

数据 sanity：object/hand shape 分别为 `[11,1024,3]`、`[11,256,3]`，ID 唯一，
normal norm 均值为 1，全部 finite；示例窗口 object endpoint motion 均值为 `0.5048 m`。

best checkpoint 的统一复算：

| 条件 | ADE (mm) | FDE (mm) | translation (mm) | rotation (deg) |
|---|---:|---:|---:|---:|
| pretrained init + GT | 140.15 | 278.22 | 277.13 | 49.97 |
| trained + GT | 67.04 | 125.19 | 118.32 | 79.86 |
| trained + static | 142.56 | 272.73 | 270.97 | 75.93 |
| trained + shuffled | 156.55 | 293.28 | 290.60 | 69.21 |

trained GT ADE 相对 init 改善 `52.17%`，相对 static/shuffled 分别改善 `52.98%`、
`57.18%`。可视化显示预测整体运动方向随 hand/object 变化，但终点和 rotation 仍有明显误差；
GT rotation error 也没有优于 static/shuffled。

**决策**

保留。E0、E1 和 E3 通过，证明 forward adapter 使用了 hand trajectory；E2 只部分通过，
尚未达到“基本记住”尤其是 rotation 的 Go 标准。按 V0 不进入 inverse branch。

**下一步**

仅用一个 sequence 延长 overfit，并加 scratch PTv3 对比；若 rotation 仍不收敛，再检查
feature/rigid target 表达，而不是扩展 inverse 或统一 WAM。

## 实验：V0 long-overfit 收敛、预训练消融与 held-out gate

**假设**

200 step 尚不足以判断 F0。若 PointWorld interaction prior 有效，延长训练后 pretrained
应比 scratch 更快、更低地收敛；只有 overfit、GT/static/shuffled sensitivity 和未训练
sequence 刚体运动都通过，才进入 inverse。

**观察到的失败 / 现象**

3000 step 时 pretrained 单 sequence 已平台化，但 pretrained 双 sequence和两个 scratch
实验的 best 仍接近末端。200-step 模型的 rotation error 很高，无法判断是欠训练还是结构问题。

**诊断**

采用 pretrained/scratch × 单/双 sequence 四路并行。未平台化实验从 best checkpoint 续训；
pretrained 双 sequence 到 5000 step，scratch 单 sequence 到 5000 step，scratch 双
sequence 到 7000 step。用全 64 窗口 GT ADE 选 best，并冻结主模型测试 motion 排名后续的
两个未训练 sequence。

**改动**

- 增加 `model.pretrained` scratch 开关和 `train.py --resume`；
- 增加 3000-step 四卡矩阵、续训、统一 best 复算与收敛曲线脚本；
- dataset 增加 `sequence_offset`，使 held-out sequence 与训练 sequence 明确分离；
- 所有实验保持 1024 object points、256 hand points、64 windows 与同一 loss/学习率。

**结果**

best checkpoint 统一复算：

| 实验 | 总 step | best step | ADE (mm) | FDE (mm) | translation (mm) | rotation (deg) |
|---|---:|---:|---:|---:|---:|---:|
| pretrained，双 sequence | 5000 | 4750 | 16.23 | 28.60 | 24.34 | 26.58 |
| scratch，双 sequence | 7000 | 7000 | 30.83 | 55.43 | 52.18 | 25.86 |
| pretrained，单 sequence | 3000 | 2750 | 13.48 | 23.50 | 19.41 | 21.14 |
| scratch，单 sequence | 5000 | 4500 | 24.84 | 42.76 | 38.31 | 24.03 |

pretrained 双 sequence 的 GT ADE 比 static/shuffled 低 `88.35%/89.32%`；单 sequence
低 `85.20%/86.82%`。pretrained ADE 约为对应 scratch 的一半，预训练迁移对点轨迹明确有效。
主实验最近四次评估 ADE 范围为 `2.58 mm`，best 后回升，视为平台化。

冻结 pretrained 双 sequence best，在未训练的 `apple_lift/hammer_use_3` 上：

| 条件 | ADE (mm) | FDE (mm) | translation (mm) | rotation (deg) |
|---|---:|---:|---:|---:|
| GT hand | 58.37 | 113.52 | 104.38 | 64.27 |
| static hand | 154.87 | 306.96 | 305.24 | 49.29 |
| shuffled hand | 185.16 | 358.15 | 354.55 | 61.11 |

GT ADE 仍比 static/shuffled 低 `62.31%/68.48%`，说明未训练 sequence 上没有完全忽略
hand action；但 rotation error 为 `64.27°`，甚至差于 static，刚体旋转泛化失败。

**决策**

保留 long-overfit 与 pretrained transfer 结论，但 F0 未通过完整 Go。点轨迹/平移 overfit
和 action sensitivity 已成立，held-out rotation 未成立。按用户条件停止，不实现 inverse。

**下一步**

若继续 F0，应加入显式 rigid SE(3) target 或刚体一致性/rotation loss，并进行真正的
sequence-disjoint 训练；继续单纯增加 step 已无依据。

## 实验：V0.1 geometry-enhanced forward overfit

**假设**

在完全相同的 2 sequences、64 windows、1024 object points、256 hand points、11 帧、
loss 和 optimizer 下，同时加入 xyz feature、RPE、5 mm grid 并关闭 DropPath/order shuffle，
应把训练集 ADE 从 `16.23 mm` 降到 `<5 mm`，rotation 降到 `5–10°`。

**观察到的失败 / 现象**

PointWorld release 的 `SerializedAttention` 仅支持 FlashAttention，并在 `enable_rpe=True`
时直接报错，不能仅靠切换官方 builder 参数打开 RPE。

**诊断**

PTv3 已提供 `attention_cls` 注入点，因此在 task 层实现 non-flash RPE attention，复用上游
padding、serialization 和 relative-position helper，可保持 DynamicsPredictor 及 checkpoint
key 不变。训练后又核查了 T=11、首帧零 displacement、per-timestep 反归一化与 absolute
track 恢复路径，未发现 shape/时间轴错位。预测非刚体残差为 `4.88 mm`，但刚体投影后 ADE
仍为 `14.00 mm`，说明剩余误差主要不是非刚体畸变，而是整体 SE(3) 姿态没有拟合准确。

**改动**

- scene adapter 输入改为 object xyz + normal + 11 帧 dist-to-hand，共 17D；
- hand adapter 输入改为 hand xyz + normal + velocity + acceleration，共 12D；
- grid 从 15 mm 改为 5 mm，DropPath 设为 0，关闭 order shuffle；
- 新增 task-local non-flash RPE attention/builder，上游 PointWorld submodule 不修改；
- 加载 448 个原有 PTv3 backbone（共 464 个 checkpoint 张量），22 个 RPE table 随机初始化，
  非 RPE 的预训练骨干漏载为 0；
- 按指导训练 5000 step，并用 best checkpoint 统一复算 GT/static/shuffled 和刚体投影。

**结果**

best 在 step 4750。两次统一复算存在约 `0.04 mm` 的 CUDA 数值波动，以下记录最后一次：

| Model | ADE (mm) | FDE (mm) | translation (mm) | rotation (deg) |
|---|---:|---:|---:|---:|
| V0 original（指导记录） | 16.23 | 28.60 | 24.34 | 26.58 |
| V0.1 xyz+RPE+5mm | 14.76 | 25.80 | 22.40 | 19.97 |

V0.1 相对 V0 四项分别改善约 `9.1%/9.8%/8.0%/24.9%`。V0.1 static/shuffled ADE
为 `140.92/149.30 mm`，GT 分别低 `89.53%/90.11%`。轨迹可视化显示中末帧整体姿态
仍有明显偏差。模型有正向改进，但 ADE 仍大于 10 mm，未通过 `<5 mm` strict gate；rotation
也未进入 `5–10°`。

**决策**

保留 geometry 实现和结果，但判定 No-Go。本轮不进入 inverse branch。

**下一步**

优先检验 DROID output statistics 与 GRAB displacement 的 domain mismatch，并考虑显式 rigid
SE(3) target/head；不再把增加训练步数作为主要方案。
