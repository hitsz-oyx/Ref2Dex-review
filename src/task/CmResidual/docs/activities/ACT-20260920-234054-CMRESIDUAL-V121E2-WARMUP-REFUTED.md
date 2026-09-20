# V1.21e.2 short warm-up parity terminal state

- timestamp: 2026-09-20T23:40:54+08:00
- activity_id: ACT-20260920-234054-CMRESIDUAL-V121E2-WARMUP-REFUTED
- work_version: V1.21
- git_commit: `281d440316b217884b1054e0505c0ccf18ce9991`
- branch: `ai/cmresidual/v121e-snapshot-restore`
- mode: run
- change_level: L3（正式 GPU PhysX 小规模运行）
- approval: 用户批准 [V1.21e.2 FINAL 微补充](../plan/V1.21e.2.md) 与 GPU3 运行
- run_id: `cmresidual_v121e2_warmup_gpu3_20260920_2317`
- run_status: `COMPLETED`
- conclusion: `REFUTED`
- scientific_conclusion: `REFUTED`（仅限 `warmup<=8 insufficient`）

## Operation and provenance

命令：

```text
/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python \
  src/task/CmResidual/tools/run_v121e2_warmup_sweep.py \
  --run-id cmresidual_v121e2_warmup_gpu3_20260920_2317 --gpu 3
```

GPU3 preflight 为 `2 MiB`。运行只读复用 V1.21e.1 的 6-state selection（SHA256
`53d36c9fb86a9b12ede0fa55841f0311c689d640d92f3c36544d12ee7bb289da`）及两个 frozen episodes。
episode action SHA 分别为 `78a964eb…fcdc4feb`、`786118b7…aa3e5d9b`，均为 431 actions。

两个独立 generation passes 从 episode initial state 完整 replay，在目标 `t-8/t-4/t-2/t-1/t`
物化 30 个唯一 snapshots。聚合文件 SHA256 为
`20b0756763d46da220ec86dddfc10efc20084f93a71f231d42d820df1c06fbee`，权限为 `0444`；每个
parity arm 后复核 SHA 未变化。generation 不计入 72 arms。

## Evidence and terminal result

输出目录：`outputs/CmResidual/cmresidual_v121e2_warmup_gpu3_20260920_2317/`。实际完成 72/72 fresh
single-env arms、6/6 state metrics，phase composition 为 `2/2/2`。所有 full 与 warm-up duplicate 的
position 差均为 `0`，rotation 最大舍入分别为 `2.98e-8 / 5.16e-8 rad`；setter、identity、finite、
snapshot immutability 和 duplicate hard ceiling 全部通过，6/6 states 均为 `implementation_valid=true`。

各 L 在 6/6 states 上均未通过 candidate 前后完整 gates：

| L | pre position min/median/max (m) | post position min/median/max (m) | post q min/median/max (rad) |
| --- | --- | --- | --- |
| 0 | `0 / 0 / 0` | `1.04e-7 / 0.04537 / 0.13327` | `0.20953 / 0.25158 / 0.31739` |
| 1 | `9.11e-8 / 0.05693 / 0.18135` | `9.40e-8 / 0.06685 / 0.32359` | `0.14856 / 0.17862 / 0.26288` |
| 2 | `9.90e-8 / 0.08724 / 0.26562` | `9.54e-8 / 0.11512 / 0.38030` | `0.09956 / 0.18922 / 0.24818` |
| 4 | `8.37e-8 / 0.05932 / 0.11353` | `7.60e-8 / 0.06305 / 0.13150` | `0.14711 / 0.17615 / 0.24016` |
| 8 | `8.56e-8 / 0.15325 / 0.70656` | `8.04e-8 / 0.15041 / 0.81269` | `0.06315 / 0.21887 / 0.27468` |

`L=0` 的 candidate 前 public state 与 full target 精确一致，但同一 candidate 后仍显著分叉。增加到
`L=1/2/4/8` 没有形成收敛或稳定通过后缀；post rotation 最大值依次为
`1.01879/2.46313/3.00093/1.67326/2.70682 rad`，post IG 最大差依次为
`0.06799/0.20581/0.28250/0.16595/0.70038`。因此 `minimal_supported_L=null`，有效且有界的结论是：
在本 6-state 协议下，最多 8 个 warm-up control steps 不足以恢复 full-prefix one-step parity。

这不反驳 `L=16/32` 或 snapshot/warm-up 的所有变体，也不识别缺失的 PhysX hidden state。未修改 Cm，
未运行 512-state ranking、PPO 或 PhysX private-state reverse engineering。无 step/epoch、best metric 或
checkpoint。回滚入口为 revert implementation commit `281d440`；运行产物保留且不纳入 Git。
