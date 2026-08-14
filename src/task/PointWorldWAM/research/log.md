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
