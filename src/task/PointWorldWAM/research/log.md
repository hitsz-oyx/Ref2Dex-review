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

## 实验：V0.2A 局部 one-step point-flow capacity

**假设**

把 11 帧联合预测改为 `(P_t,H_t,ΔH_t) -> ΔP_t`，删除 DROID output normalization，
并将随机初始化 3D head/new modules/pretrained backbone 的学习率拆为 `3e-4/1e-4/1e-5` 后，
64 个干净右手 transition 应能过拟合到 1–2 mm，同时明显优于 static/shuffled action。

**观察到的失败 / 现象**

旧数据按 object motion 选择窗口并固定 right hand，没有排除 left-hand involvement；旧
DynamicsPredictor 还固定输出未来 10 帧并依赖 DROID per-timestep statistics，无法直接回答
单步 GRAB metric flow capacity。

**诊断**

`phone_call_1/scissors_use_2` 分别有 411/241 个 transition 满足相邻两帧都 right 5 cm
candidate 非空且 left candidate 为空；raw frame gap 恒为 4，对应 `dt=1/30 s`。从两者按
object motion 选 64 个 transition，其 zero-flow error 为 `37.86 mm`，不是近静态伪任务。
checkpoint 成功迁移 448 个 PTv3 backbone 和 4 个 FiLM 参数；非 RPE 预训练漏载为 0。

**改动**

- 新增 clean right-hand transition dataset，返回 current geometry、hand/object flow 与真实 dt；
- object feature 为 xyz/normal/current dist2hand 共 7D，hand feature 为 xyz/normal/flow 共 9D；
- 保留 V0.1 的 5 mm grid、RPE、DropPath=0、固定 serialization order；
- 新增随机初始化的 128→128→3 pointwise head，直接输出 meter object flow；
- 不构建/加载旧 dynamics/log-var head，不读取 DROID statistics；
- 增加三级 LR、best checkpoint、续训与 GT/static/shuffled/zero-flow 诊断；
- 从 5000-step best 续训到 10000 step，确认后期平台。

**结果**

best 在 step 9500：

| 条件 | point error (mm) | translation (mm) | rotation (deg) |
|---|---:|---:|---:|
| GT hand flow | 5.12 | 4.27 | 3.28 |
| static action | 24.34 | 24.13 | 4.64 |
| shuffled action | 19.22 | 18.76 | 3.85 |
| zero-flow baseline | 37.86 | 37.54 | 8.58 |

GT point error 比 static/shuffled/zero-flow 分别低 `78.9%/73.3%/86.5%`，action
sensitivity 明确。GT target 的刚体残差为 `4.6e-8 m`，确认 stable point ID/target 正确；
预测非刚体残差为 `2.27 mm`，严格刚体投影后 point error 为 `4.54 mm`，说明非刚体畸变只
解释部分误差，整体 SE(3) 仍是主要剩余项。逐 transition error 中位数/P90/max 为
`4.81/7.76/9.22 mm`，与 motion 相关系数 `-0.09`。

**决策**

保留。局部 temporal formulation 和 action conditioning 有效，但绝对误差未通过 1–2 mm
capacity gate。按 V0.2 停止，不运行 V0.2B、V0.3 rollout 或 inverse。

**下一步**

先检验 rigid-consistent head/target parameterization；若 GT one-step 仍不能精确拟合，再加入
previous object flow 判断 state aliasing，不直接扩展 variable span。

## 实验：V1 双手 MANO action Flow Matching overfit

**假设**

给定当前 object、previous object flow、左右 MANO 和 desired object flow，把 normalized noisy
relative MANO action 可微还原成 noisy hand point flow，再用共享 PTv3 和独立左右 FM head，
应能在 128 个 transition 上生成优于 identity 的下一帧双手 surface；通过后再放开全量。

**观察到的失败 / 现象**

默认 Python 没有 `smplx`，但 `graspenv` 同时具备 smplx、PyTorch 2.4.1 和完整 PointWorld
CUDA 依赖。旧 one-step dataset 是 right-only 且下采样到 256 hand points，不满足 V1。

**诊断**

新增 dataset 使用 `current>=1`、gap=1、current 帧 left/right 至少一手 active 的全部合法集合，
不按 motion 排序；debug 的 128 个样本用均匀索引覆盖 20 sequences，完整集合为 6652。
从 raw GRAB 读取每手 global orientation、24D PCA pose、translation、betas，并注入 s1 的左右
v_template。完整 128 条 debug transition 上，zero action 重建 current surface、GT action 重建
next surface 的逐样本平均误差均值约 `0.000056–0.000058 mm`，最差 `0.000340 mm`，排除了
MANO/action target 错位。

**改动**

- 新增双手 WAM dataset，保留 1024 object 与左右各 1538 MANO face centers；
- relative action 使用 `Δx`、`log(R_t^T R_{t+1})` 和 24D PCA delta，每手 30D；
- 新增可微 action compose/MANO surface 和左右独立 action normalization；
- 双手 forward 使用 object xyz/normal/distL/distR/previous flow 与左右 xyz/normal/flow；
- WAM object token 使用 current/previous/desired flow，hand token 使用 noisy candidate flow；
- 4100 spatial tokens 共享预训练 PTv3，加入三类 type embedding、noise-time embedding、
  noisy action embedding与独立左右 FM head；
- 训练只使用左右 FM MSE，推理使用 10-step Euler；
- 增加 translation/rotation/PCA/surface、identity/mean/random baseline 和 forward consistency。

**结果**

MANO action chain 的数值一致性通过。双手 forward 5000-step best：

| 条件 | object point error (mm) |
|---|---:|
| GT hand flow | 10.09 |
| static hand flow | 10.17 |
| shuffled hand flow | 10.20 |
| zero object flow | 6.53 |

forward 没有学到 action，effect consistency 不能作为可信指标。

WAM 从 5000-step best 续训到 10000 step，fixed-noise FM loss 从 step 500 的 `3.94` 降至
`1.77`。best 10-step generation：

| 指标 | Left | Right |
|---|---:|---:|
| translation (mm) | 6.19 | 8.12 |
| rotation (deg) | 1.22 | 1.94 |
| PCA pose RMSE | 0.0446 | 0.1033 |
| hand surface (mm) | 6.61 | 8.68 |
| identity surface (mm) | 5.40 | 7.98 |
| action-mean surface (mm) | 5.50 | 8.03 |
| random surface (mm) | 10.97 | 13.70 |

双手平均生成 surface error 为 `7.65 mm`。它比 random prior 好，但仍比 identity 差；可视化
显示生成手型正常，主要误差是下一帧整体位置/姿态。生成/GT hand 经不合格 forward 得到的
effect error 为 `7.53/7.42 mm`，差异无解释力。

**决策**

保留 V1 完整实现与 128-transition 证据，但判定 debug overfit No-Go。不放开 6652-transition
全量，不添加 effect loss，也不进入 unified WAM。

**下一步**

先做 action-space direct regression/conditional memorization 上界，并检查 Euler flow path 是否是
当前主要瓶颈；只有生成 surface 明确优于 identity 后再恢复 full training。
