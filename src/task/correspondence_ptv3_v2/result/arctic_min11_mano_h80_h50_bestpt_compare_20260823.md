# ARCTIC min11 MANO：5 mm H80/O20 与 H50/O50 best.pt 对比

> **失效说明（2026-08-23）**：评估端当时未覆盖 checkpoint 内的 `hand_perturb_prob`，因此 H80 的 hand 条件实际按 0.8 gate、H50 按 0.5 gate，hand-only / hand+object 输入不一致。下方 object-only 结果仍有效；涉及 hand 的结果与据此形成的 exposure 结论标记为 `INVALID_IMPLEMENTATION`，不得继续用于公平排名。评估器已修复，统一 100% 手扰动后的 10 mm 结果见 [`arctic_min11_mano_protocol_e_10mm_compare_20260823.md`](arctic_min11_mano_protocol_e_10mm_compare_20260823.md)。

## 协议

- 数据：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`，11 个物体、11 个 s01 左手文件、4175 帧。
- 两条 run 均使用各自按 GRAB `val_clean/cross_edge_random_qfl` 选出的 `best.pt`。
- H80/O20：step 399520 / epoch 44；H50/O50：step 236067 / epoch 13。
- `object_only`：object 10°/10 mm，hand clean。
- `hand_only`：ARCTIC axis-angle45 hand noise，目标 5 mm RMS，object clean。
- `hand_object`：同时施加上述 hand 与 object noise，评测时关闭 exclusive gate。
- batch size 16、num_workers 0、runtime object resampling 关闭；所有条件复用同一 evaluator 和 Stage 3 数据。

QFL 越低越好，`pseudo_recovery_brier` 越高越好。本文以合并 fake-contact 与 missed-contact changed edges 的 `pseudo_recovery_brier` 作为总体 recovery 主指标；projection 和两类分解指标仅保留在原始 JSON 中用于诊断。

## 核心结果

| 条件 | 指标 | H80/O20 best | H50/O50 best | H50 相对变化 |
| --- | --- | ---: | ---: | ---: |
| object-only | clean random QFL | **0.00003749** | 0.00008600 | +129.4%（更差） |
| object-only | perturbed random QFL | 0.00030386 | **0.00025914** | -14.7%（更好） |
| object-only | `pseudo_recovery_brier` | 0.2787 | **0.4119** | +0.1332 |
| hand-only | perturbed random QFL | **0.00005050** | 0.00009518 | +88.5%（更差） |
| hand-only | `pseudo_recovery_brier` | **0.2122** | -0.0074 | -0.2197 |
| hand+object | perturbed random QFL | 0.00031991 | **0.00026396** | -17.5%（更好） |
| hand+object | `pseudo_recovery_brier` | 0.2693 | **0.4116** | +0.1423 |

## 解释

H50/O50 明显改善了 object-only 和 hand+object：两种条件的 perturbed QFL 均降低，综合 `pseudo_recovery_brier` 分别提高 0.1332 和 0.1423。这支持“原 H80/O20 的 object exposure 不足”是其 object/joint recovery 较弱的重要原因。

代价同样明确：H50/O50 的 clean QFL 约为 H80/O20 的 2.29 倍，hand-only perturbed QFL 约为 1.88 倍，hand-only `pseudo_recovery_brier` 从 0.2122 降到 -0.0074。当前结果不是全面提升，而是从 hand/clean 一侧向 object/joint robustness 一侧移动。

两条 best checkpoint 的训练预算并不匹配：H80/O20 为双卡 global batch 32、step 399520；H50/O50 为单卡 global batch 16、step 236067。因此结果可用于判断当前 checkpoint 的行为和 exposure 方向，但不能作为严格单变量因果消融。

## 产物

- JSON / log：`output/research/arctic_min11_mano_bestpt_h80_h50_20260823/`
- evaluator：`src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py`
