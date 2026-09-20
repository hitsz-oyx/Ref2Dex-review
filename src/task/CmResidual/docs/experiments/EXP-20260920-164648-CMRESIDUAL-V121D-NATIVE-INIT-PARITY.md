# V1.21d native-init 对 9-env one-step parity 的诊断

- work_version: V1.21
- conclusion: REFUTED
- related_activity: [ACT-20260920-164648-CMRESIDUAL-V121D-GATE0-FAILED](../activities/ACT-20260920-164648-CMRESIDUAL-V121D-GATE0-FAILED.md)
- run_id: cmresidual_v121d_native_init_gate0_gpu3_20260920_1645

## Hypothesis and protocol

窄假设：V1.21c 的 q/dq 分叉主要来自 simulator 创建后只恢复公开 root/DOF tensor；如果先用
单环境 DExplore `StateInit.Start` 采集 episode artifact，再销毁 source simulator，并在一个全新的
9-env simulator 中用相同 seed/motion/config 原生初始化和重放同一个 native hold prefix step，9 个
env 会满足固定 `1e-5` public-state parity。

运行固定 git commit `4607422dcc840a1e258133231aafb5641da2bd69`、GPU3、seed 5909、
`s1_airplane_lift`、1-step prefix。初态与 prefix 后均比较 q/dq、actor roots 和 task/reference
indices；parity 失败则不执行 candidate/duplicate。

## Evidence

原生初态成功复现 collector artifact：最大公开状态误差 `4.1723251e-7`，q/dq 误差均为零，indices
完全相同。这支持 native-init 接线和 source/validation 身份一致。

一个相同 prefix step 后，9 env 的最大 q/dq 差异为 `0.0111185312 rad` 和
`0.5535187721 rad/s`，显著超过 `1e-5`。这两个值与旧 tensor-restore 最终 smoke 基本逐值相同；
root position/quaternion/twist 也超过 parity 门槛。simulator 生命周期完整，未见 OOM 或输入 SHA 漂移。

## Conclusion and limits

“tensor restore 是该 one-step 分叉主因、native init + fresh sim 可恢复 9-env parity”的窄假设为
`REFUTED`。当前更符合“同一个多环境 PhysX simulator 中的跨 env one-step 分叉”这一解释，但这是
基于单 seed、单 state、单 GPU 的推断；尚未用彼此隔离的单环境 simulator 直接验证，不能断言具体
PhysX 隐藏机制。

本实验没有执行 candidates、Cm prediction、ranking 或 PPO，因此 Cm ranking、抓取效果、策略收敛和
泛化仍为 `INCONCLUSIVE`。
