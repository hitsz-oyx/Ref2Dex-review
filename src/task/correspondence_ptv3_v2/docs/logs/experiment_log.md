# Experiment log

## 当前状态

已完成 OakInk 坐标和无 MANO runtime sampling 的实现 gate，正式训练进行中；同时已完成 GRAB 5 mm PCA 互斥扰动版与两条 9 mm baseline 的当前 `best.pt` 对比。5 mm 版在 correspondence QFL 上更好，但 recovery 指标整体落后于 9 mm rebuilt baseline；和旧 no-PCA legacy run 比，clean QFL 更好，但 perturbed / recovery 指标更差；后续 no-PCA 默认对照已切到 8 月中旬的 `old1797_compact_repro`，它和当前 5 mm 语境更接近。当前 GRAB “5 mm 配置 + runtime sampling + no-PCA hand perturb”训练目录为 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614`；该 run 原进程停在日志 step 36320 / epoch 4 附近，但可用 `latest.pt` 只保存到 step 27240 / epoch 3，2026-08-20 已从该 checkpoint 放入 tmux `grab5mm_no_pca_runtime_resume_20260820` 继续两卡训练，W&B run id 为 `zeqrhxyk`。V1 指导下的 ARCTIC 外部评估也已完成：分层子集覆盖 11 个 object / 5 个 subject，共 179 条序列、93,968 帧；pure GRAB 在 micro 和 object-macro 上都略优于 GRAB+ContactPose，差距幅度有限，结果页已写入 `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.md`。

2026-08-21 已完成一个受控的 ARCTIC min11 快速筛查：11 个物体各取一个确定性文件、约 4175 帧；noPCA 与 5mm 均关闭 hand/PCA perturb 和 runtime resampling，只开启相同的 10°/10 mm object perturb。5mm 在 clean 拟合上更好，但 noPCA 的 recovery 明显更好，且 perturbed correspondence QFL 略好。由于两条 checkpoint 的训练步数不同且子集只有 s01，该证据仅作方向性判断；完整 hand+object 公平对比仍未完成。

2026-08-21 已从 ARCTIC raw `.mano.npy` 重导出带 MANO 的 min11 Stage 3 到 NAS，并完成 noPCA / 5mm 的 object-only、hand-only、hand+object 三条件评估。5mm 明显改善 hand-only，但在 object-only 和 joint recovery 上仍低于 noPCA；这说明当前互斥训练配方的 object/compound exposure 仍不足。

三域等比例混训 run `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_20260820_141453` 已停止。`train.log` 和 `metrics.jsonl` 的最后有效记录均为 step 27980 / epoch 2；`latest.pt`、`best.pt` 和 step checkpoint 均只保存到 step 22748 / epoch 1。日志无 Python traceback、CUDA OOM、kernel OOM、SIGTERM/SIGKILL 或保存 `.tmp` 残留，当前也没有对应 tmux 会话和训练进程。因此只能确认它是在 checkpoint 周期之间被外部停止或会话消失，不能从本地日志判断具体是哪个终止动作。

已按用户确认建立 H50/O50 互斥配置：`src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_h50_o50_exclusive_ddp2.yaml`。其 corruption gate 为 hand-only 50%、object-only 50%、compound 0%；训练启动记录见 EXP-010。

2026-08-23 完成协议 E：在同一 MANO min11 上使用统一 100% 的 10 mm hand perturb，并新增 ΔQFL。noPCA 的实际退化量最小但最终 QFL 较差；H80 best 的 hand-only 最终质量最好；H50 best 的三条件等权 perturbed QFL 最好。核验中发现旧协议 D evaluator 继承 checkpoint 的 `hand_perturb_prob`，因此旧 hand-only / hand+object 比较为 `INVALID_IMPLEMENTATION`，object-only 仍有效。

2026-08-24 已用纯 GRAB、训练时关闭手部扰动的 noPCA latest checkpoint 完成 HOCap subject_1 首次外部评估。28 个序列/物体文件共 23,896 个帧样本全部通过；clean random QFL 为 0.00016607，固定 10°/10 mm object perturb 后为 0.00020392，recovery Brier 为 0.5267。该结果只覆盖一个 subject，且接触 GT 为几何派生，因此先作为可运行基线，不作模型优劣定论。

2026-08-24 已完成 OakInk true-hand-root 全量重导出。新产物使用官方 root quaternion 同时规范化手和物体，并携带可由训练端 `smplx.MANO` 重建的 axis-angle45 参数；2596 个文件 / 252,172 帧通过 loader、MANO 重建、接触距离和四视角一致性校验。572 个视角组因官方下载缺少 object mesh 跳过。

## EXP-013 — OakInk true-hand-root 全量重导出与 MANO 重建校验

### 日期

2026-08-24

### 目的

替换 2026-08-18 只做 wrist 平移中心化的 OakInk Stage 3，使 OakInk 与 GRAB/ContactPose 的严格 rotation-canonicalized hand-root 合同一致，并保留后续 MANO 手姿态扰动所需参数。

### 转换协议

- 数据源：OakInk-Image 官方 `hand_v`、`hand_j`、`obj_transf`、`general_info` 和 object mesh；
- root pose：使用 `general_info.hand_anno.hand_pose[0]` 的 `[w,x,y,z]` quaternion 与 `hand_tsl` 构造 `T_world_root`，再由 `cam_extr @ T_world_root` 得到 camera-space root；
- 几何：camera-space 手顶点、物体点和法向统一乘 root pose 的逆变换；778 MANO vertices 通过标准拓扑转换为 1538 face centers/normals；
- MANO：保存 axis-angle45 local pose、global orientation、betas、template 和 `hand_root_pose`；
- translation：OakInk `hand_tsl` 是 wrist joint 世界位置，导出时减去 shape-dependent MANO wrist template offset，得到 `smplx.MANO.transl`；
- 输出不覆盖旧目录，旧 wrist-centered 版本保留用于历史实验复现。

### 结果

| 项目 | 结果 |
| --- | ---: |
| 输入视角组 | 3168 |
| 成功 NPZ | 2596 |
| 总帧数 | 252,172 |
| 缺 object mesh 跳过组 | 572 |
| Stage 3 体积 | 约 17 GB |
| 抽样 MANO 重建 RMS | 0.000040–0.000069 mm |
| 抽样 MANO 重建最大误差 | ≤ 0.000131 mm |
| 接触距离缓存最大误差 | ≤ 1.50×10⁻⁸ m |
| 四视角 hand RMS | ≤ 0.000030 mm |
| 四视角 object RMS | ≤ 0.000069 mm |

### 实现核验

Verdict: PASS

- 2596 个 `.npz` 均完成原子 `mano_transl` 修复，没有 `.repairing.npz` 残留；
- `CorrStaticDatasetV2(use_mano_reconstruction=true, apply_hand_perturb=true, coordinate_frame=hand_root)` 严格加载通过；
- 首、中、末及跨目录抽样的字段、有限值、root rotation 正交性、KD-tree 距离缓存和训练端 MANO forward 均通过；
- 同一物理帧四个 camera view 在 hand-root 中重合，证明 camera rotation 已被消除；
- 汇总记录 `schema_version=2.0.0`，能力门槛由完整 MANO 字段而非版本字符串决定。

### 结论状态

**SUPPORTED**（数据转换与训练接口的工程/几何结论，不代表训练效果）

### 限制与下一步

- 缺 mesh 的 572 个视角无法从当前本地官方下载内容补出物体监督；
- 历史配置仍指向旧目录，下一次启动训练前必须显式切换到新目录；
- 是否把 OakInk 纳入手扰动训练是后续训练设计选择，本次仅保证数据能力，不自动改变已有实验协议。

### 证据

- `research/oakink_conversion/convert_oakink_pilot.py`
- `research/oakink_conversion/repair_oakink_mano_transl.py`
- NAS 输出汇总 `oakink_pilot_stats.json`

## EXP-012 — 纯 GRAB noPCA 在 HOCap subject_1 的外部测试

### 日期

2026-08-24

### 目的

确认训练时没有加入手部扰动的纯 GRAB checkpoint 能否直接消费新转换的 HOCap Stage 3，并建立后续与三域 mixed checkpoint 对照的首条外部基线。

### 协议

- checkpoint：`outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614/checkpoints/latest.pt`，step 317800 / epoch 35；checkpoint 配置中 `apply_hand_perturb=false`；
- 数据：NAS `processed_data/stage3/hocap_subject1_annotation_v1`，28 个序列/物体文件、23,896 个帧样本；
- clean：直接使用存储的 clean hand points，不重建 MANO；
- perturbed：手扰动关闭，物体固定启用 10° rotation / 10 mm translation；
- 两条流均关闭 runtime object resampling，batch size 16，在物理 GPU 3 上评估。

### 实现核验

Verdict: PASS

- 先用 `G18_1_left.npz` 的 676 帧完成 smoke，再运行全量；
- 全量 JSON 记录 `hand_perturb=false`、`hand_perturb_probability=0.0`、`runtime_resample_object=false`；
- 28 个文件、23,896 个帧样本全部完成 clean / perturbed 推理，无 schema、坐标系、CUDA 或有限值错误。

### 结果

| Metric | clean | object perturbed |
| --- | ---: | ---: |
| random QFL ↓ | 0.00016607 | 0.00020392 |
| random MAE ↓ | 0.00702896 | 0.00731687 |
| random nonzero MAE ↓ | 0.03092725 | 0.03283193 |
| contact auxiliary QFL ↓ | 0.00526155 | 0.00657405 |
| hand contact QFL ↓ | 0.00042121 | 0.00046103 |
| recovery Brier ↑ | — | 0.5267 |
| recovery projection ↑ | — | 0.4541 |

object perturb 使 random QFL 增加 0.00003785，约为 clean 的 +22.8%；random MAE 增加约 4.1%。

### 解释

该 checkpoint 能直接迁移到 HOCap 的 hand-root 几何表示，且在 object perturb 后仍输出有效的 recovery 指标，说明新数据导出与现有评估链路在工程上兼容。当前没有同协议的 HOCap checkpoint 横向对照，不能仅凭绝对值判断泛化是否足够好。

### 结论状态

**INCONCLUSIVE**

工程兼容性已确认，但科研结论仍受 subject_1 单主体、几何派生接触 GT、micro 聚合以及缺少 matched baseline 限制。

### 下一步

三域 mixed 获得可用 checkpoint 后，使用完全相同的 HOCap object-only 协议比较；若要形成正式 benchmark，再增加 subject 覆盖并补 sequence/object-macro 汇总。

### 证据

- `output/research/hocap_subject1_grab_nopca_20260824/full_object_only.json`
- smoke：`output/research/hocap_subject1_grab_nopca_20260824/smoke_object_only.json`

## EXP-011 — 协议 E：10 mm MANO min11 三条件评估

### 日期

2026-08-23

### 假设

5 mm hand perturb 的伪几何能量较小时，比例型 `pseudo_recovery_brier` 可能放大很小的绝对误差。把 hand 几何扰动提高到 10 mm，并同时报告 `ΔQFL = perturbed QFL - clean QFL`，应能区分相对恢复效率、实际退化量和最终绝对质量。

### 协议

- 数据：ARCTIC MANO min11，11 文件 / 4,175 帧 / s01 / 11 objects；
- checkpoint：noPCA latest、H80 latest、H80 best、H50 best，以及追加的历史 GRAB+ContactPose 两域 mixed latest；
- object-only：clean hand + object 10°/10 mm；
- hand-only：hand axis-angle45 10 mm RMS + clean object；
- hand+object：同时施加上述两种扰动；
- hand 条件固定 `hand_perturb_prob=1.0`，runtime resampling 关闭。

### 实现核验

首次运行后发现 evaluator 会继承 checkpoint 的训练时 `hand_perturb_prob`：noPCA/H80 为 0.8，H50 为 0.5，导致 changed-edge 支持量不同。修复为评估端统一 1.0 后覆盖重跑最初 8 个 hand 条件；追加历史两域模型后，五个 checkpoint 的 hand-only changed-edge fraction 均为 0.003200，hand+object 均为 0.004462。全部输出均为 4,175 帧且无错误。

该缺陷同时影响旧协议 D 的 hand-only / hand+object 公平比较，因此 EXP-009 与 EXP-010 中依赖这些条件的历史结论标记为 `INVALID_IMPLEMENTATION`；object-only 数字不受影响。

### 结果

| checkpoint | Balanced perturbed QFL ↓ | Balanced ΔQFL ↓ | Balanced recovery Brier ↑ |
| --- | ---: | ---: | ---: |
| noPCA latest | 0.00027900 | **+0.00005226** | **0.4339** |
| H80 latest | 0.00025684 | +0.00021759 | 0.3053 |
| H80 best | 0.00025872 | +0.00022123 | 0.2981 |
| H50 best | **0.00023450** | +0.00014843 | 0.3755 |
| GRAB+ContactPose latest | 0.00097292 | **-0.00000792** | 0.1672 |

Hand-only 分项：noPCA / H80 latest / H80 best / H50 best 的 recovery 分别为 0.2295 / 0.3598 / 0.3621 / 0.3117，ΔQFL 分别为 +0.00002624 / +0.00006907 / +0.00006986 / +0.00006339。

### 解释

noPCA 在 10 mm 下没有出现旧 5 mm 表中的巨大负 recovery，且绝对退化最小，支持“小分母会放大 recovery 观感”的假设。但 noPCA 的 clean 基线较差，所以最终 perturbed QFL 仍是 hand-only 四者最高；小 ΔQFL 不能等价为最好模型。H80 best 的 hand-only 最终质量和 recovery 最好，H50 best 则取得最低的三条件等权 perturbed QFL。

joint recovery 仍可能由 object changed-edge 能量主导，不能把 noPCA 的高 joint recovery 解读为良好的 hand robustness。

追加的历史 GRAB+ContactPose 两域 mixed 在三个条件上的 perturbed QFL 分别为 0.00098546 / 0.00096150 / 0.00097180，Balanced recovery 为 0.1672，整体明显落后于另外四条新路线。它的 hand-only 和 joint ΔQFL 为负，但这是相对约 0.000981 的高 clean 误差略有下降，不能解释成最佳鲁棒性。

### 结论状态

SUPPORTED

### 限制与下一步

- min11 只有 s01，结果只作机制筛查；
- checkpoint 选择与训练预算不匹配；
- 若需要严谨比较 5 mm 与 10 mm，应使用修复后的 evaluator 重跑 5 mm，而不是与旧协议 D 数字直接比较。

### 证据

- 结果：`src/task/correspondence_ptv3_v2/result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md`
- JSON / log：`output/research/arctic_min11_mano_protocol_e_10mm_20260823/`
- evaluator：`src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py`
- 历史两域 checkpoint：`outputs/train/correspondence_ptv3_v2_old1797_grab_contactpose_full_gpu3_20260816_110237/checkpoints/latest.pt`

## EXP-010 — GRAB 5 mm H50/O50 互斥扰动

### 日期

2026-08-21

### 对应指导

当前 H50/O50 对照方案；ARCTIC 外部评估使用 `src/task/correspondence_ptv3_v2/docs/指导/V1.md` 的统一协议。

### 假设

相较当前 H80/O20/no-compound 配方，H50/O50 能在保持 hand-noise recovery 能力的同时增加 object-only exposure；如果 object-only 与 joint ARCTIC recovery 的下降主要来自 object exposure 不足，H50/O50 应缩小与 noPCA 的 object-only 差距，而 hand-only 不应完全退化。

### Baseline

- 5 mm H80/O20：`src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_exclusive_ddp2.yaml`
- noPCA H0/O100：`src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_no_pca_runtime_ddp2.yaml`
- 训练从随机初始化开始，保持 5 mm hand geometry noise、optimizer、loss 和 object perturb 标准不变。

### 本次修改

- `hand_perturb_prob=0.5`、`exclusive_hand_object_perturb=true`；
- `apply_obj_perturb=true`、`obj_perturb_prob=1.0`；
- 因互斥 gate，实际为 50% hand-only、50% object-only、0% hand+object；
- 不加入 scale augmentation，不引入新的 runtime sampling 机制。

### 实现审查

Verdict: PASS

- gate 先以 0.5 概率决定 hand perturb，再在 hand 分支关闭 object perturb；
- object 分支保留 object perturb；
- 配置继承既有 5 mm H80/O20 recipe，未改模型、loss 或 evaluator。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
/home/wbcd/miniconda3/envs/graspenv/bin/python \
-m src.task.correspondence_ptv3_v2.train \
--config src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_h50_o50_exclusive_ddp2.yaml \
--set train.distributed.enable=false train.max_steps=454000
```

### 结果

> 2026-08-23 更正：下表 hand-only / hand+object 因 evaluator 分别继承 H80=0.8、H50=0.5 的手扰动概率而输入不一致，标记为 `INVALID_IMPLEMENTATION`；object-only 仍有效。修复后的统一评估见 EXP-011。

训练已于 2026-08-22 完成 454000 step。按 GRAB `val_clean/cross_edge_random_qfl` 选择的 `best.pt` 位于 step 236067 / epoch 13。2026-08-23 使用与旧 H80/O20 完全相同的 ARCTIC MANO min11 三条件协议，重新评估两条 run 各自的 `best.pt`；旧 H80/O20 best 位于 step 399520 / epoch 44。

| 条件 | 指标 | H80/O20 best | H50/O50 best |
| --- | --- | ---: | ---: |
| object-only | clean random QFL | **0.00003749** | 0.00008600 |
| object-only | perturbed random QFL | 0.00030386 | **0.00025914** |
| object-only | `pseudo_recovery_brier` | 0.2787 | **0.4119** |
| hand-only | perturbed random QFL | **0.00005050** | 0.00009518 |
| hand-only | `pseudo_recovery_brier` | **0.2122** | -0.0074 |
| hand+object | perturbed random QFL | 0.00031991 | **0.00026396** |
| hand+object | `pseudo_recovery_brier` | 0.2693 | **0.4116** |

### 关键观察

H50/O50 的 object-only 与 hand+object 综合 `pseudo_recovery_brier` 分别比 H80/O20 提高 0.1332 和 0.1423，perturbed QFL 也分别改善 14.7% 和 17.5%；但 hand-only `pseudo_recovery_brier` 从 0.2122 降到 -0.0074，hand-only perturbed QFL 变差 88.5%，clean QFL 约变为 2.29 倍。

### 解释

结果支持 object exposure 不足确实是 H80/O20 object/joint recovery 较弱的重要原因；把比例改为 H50/O50 后，object-only 与未在训练中出现的 compound 条件均明显改善。但 hand-only 与 clean 拟合同步退化，说明当前变化是 robustness trade-off，而不是全面提升。

两条 best checkpoint 的 optimizer step 和 global batch 不匹配：H80/O20 为双卡 global batch 32、step 399520；H50/O50 为单卡 global batch 16、step 236067。因此这里只能确认当前 checkpoint 行为和 exposure 方向，不能写成严格 matched-sample 因果消融。

### 结论状态

INCONCLUSIVE

### 决策

保留 H50/O50 作为独立 matched recipe；不覆盖旧 H80/O20 或 noPCA checkpoint。

### 下一步

- 若目标偏 object/joint robustness，H50/O50 是更好的候选；若更重视 clean/hand-only，H80/O20 当前更好。
- 做严格结论前，应按相同 global sample exposure 或相同 global batch 重跑 matched 对照。
- 若要寻找折中点，可单独测试 H60/O40 或 H70/O30，不应把本结果解释为比例越均等越好。

### 证据

- 配置：`src/task/correspondence_ptv3_v2/configs/full_grab_50ep_geometry_5mm_h50_o50_exclusive_ddp2.yaml`
- 训练输出：`outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_h50_o50_exclusive_ddp2_20260821_073040`
- tmux：`grab5mm_h50_o50_20260821_gpu0`
- W&B：`gi854jvw`（本地 `wandb/run-20260821_073221-gi854jvw`）
- best.pt 对比结果：`src/task/correspondence_ptv3_v2/result/arctic_min11_mano_h80_h50_bestpt_compare_20260823.md`
- 评测 JSON / log：`output/research/arctic_min11_mano_bestpt_h80_h50_20260823/`

## EXP-009 — 带 MANO 的 ARCTIC min11 三条件鲁棒性评估

> 2026-08-23 更正：本实验的 hand-only / hand+object 继承了 checkpoint 内的 `hand_perturb_prob=0.8`，没有实现“所有帧统一施加手扰动”的协议语义，标记为 `INVALID_IMPLEMENTATION`；object-only 仍有效。修复后的统一 10 mm 评估见 EXP-011。

### 日期

2026-08-21

### 对应指导

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

### 假设

如果 5mm hand perturb 训练确实学到了 hand-noise recovery，那么在同一 ARCTIC MANO 输入上，5mm 应优于未训练 hand perturb 的 noPCA；如果互斥门控减少了 object/compound exposure，则 5mm 在 object-only 或 hand+object 上可能仍然落后。

### Baseline

- noPCA: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614/checkpoints/latest.pt`，step 317800。
- 5mm: `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/checkpoints/latest.pt`，step 454000。
- 数据：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`。

### 本次修改

- 从 ARCTIC raw MANO `.mano.npy` 重建 11 个 Stage 3 文件到 NAS。
- evaluator 增加 `object_only`、`hand_only`、`hand_object` 条件开关；默认 object-only 行为保持不变。
- ARCTIC axis-angle45 hand noise 使用 9mm calibration 缩放到 5mm RMS。

### 实现审查

Verdict: PASS

- 11 个输出文件均含 `mano_pose (T,45)`、`mano_global_orient`、`mano_transl`、`mano_betas`。
- 所有文件标记 `mano_pose_repr=axis_angle`、`mano_use_pca=false`。
- 三种条件使用同一 4175 帧 NAS 数据，runtime resampling 关闭。
- 训练步数和 subject 覆盖不匹配，结果只作方向性比较。

### 实验命令

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python \
  -m src.task.correspondence_ptv3_v2.research.contactpose_checkpoint_compare.evaluate \
  --checkpoint <checkpoint> \
  --test-root /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1 \
  --output output/research/arctic_min11_mano_v1/<name>.json \
  --condition {object_only|hand_only|hand_object} \
  --hand-target-rms-mm 5.0 --device cuda:0 --batch-size 16 --num-workers 0
```

### 结果

| 条件 | noPCA recovery Brier | 5mm recovery Brier |
| --- | ---: | ---: |
| object-only | **0.5355** | 0.2929 |
| hand-only | -0.5066 | **0.2112** |
| hand+object | **0.5356** | 0.2813 |

### 关键观察

5mm 的 hand-only recovery 从 noPCA 的负值提升到正值，说明 hand perturb 训练路径确实有效；但它没有转化为 object-only 或 hand+object recovery 优势。clean QFL 三种条件都由 5mm 明显更好。

### 解释

当前 5mm exclusive 配方约为 H80/O20/no-compound，noPCA 约为 H0/O100。这个 exposure 差异足以解释 object-only 和 joint 的排序，但不能排除 recovery 指标与 clean 拟合之间的行为差异。

### 结论状态

**INCONCLUSIVE**

局部证据支持“5mm hand perturb 训练改善 hand-only recovery”；但由于训练步数和 exposure 配方未严格匹配，当前实验不足以对整体 hand/object/joint robustness 做因果结论。

### 决策

保留当前两条 checkpoint；不把 5mm exclusive 作为全面替代 noPCA。下一轮加入 H80/O100 compound 对照。

### 下一步

- 用共同 optimizer step 做 H0/O100、H80/O20、H80/O100 对照。
- 继续使用 object-only、hand-only、hand+object 三条件和 absolute changed-edge error。

### 证据

- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_compare_20260821.md`
- `src/task/correspondence_ptv3_v2/result/arctic_min11_mano_v1_manifest.json`
- `output/research/arctic_min11_mano_v1/*.json`

## EXP-008 — ARCTIC min11 上 noPCA 与 5mm 的 matched object-only 快速筛查

### 日期

2026-08-21

### 对应指导

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

### 假设

在统一的 object-only corruption 下，5mm 训练是否能改善 object perturb recovery；使用覆盖全部物体类别的极小分层集先判断方向，可避免直接支付全量 ARCTIC 的计算成本。

### Baseline

- noPCA checkpoint: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614/checkpoints/latest.pt`，step 317800。
- 5mm checkpoint: `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/checkpoints/latest.pt`，step 454000。
- 子集：`src/task/correspondence_ptv3_v2/result/arctic_min11_v1_manifest.json`。

### 本次修改

- 不修改模型、loss 或 evaluator。
- 对两条 checkpoint 使用相同 stored clean hand / object-only protocol。
- 每个物体仅保留一个确定性 ARCTIC 文件，以控制评测时间。

### 实现审查

Verdict: PASS

- 11/11 物体类别均覆盖。
- 两个 checkpoint 的 hand/PCA perturb、runtime resampling、object perturb probability 均由 evaluator 统一设置。
- 结果 JSON 已保存到 `output/research/arctic_min11_v1/`。
- 限制：仅 `s01`，且训练步数不匹配，因此不作为最终 causal comparison。

### 实验命令

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python \
  -m src.task.correspondence_ptv3_v2.research.contactpose_checkpoint_compare.evaluate \
  --checkpoint <checkpoint> \
  --test-root /tmp/arctic_eval_min11_v1 \
  --output output/research/arctic_min11_v1/<name>.json \
  --device cuda:0 --batch-size 16 --num-workers 0
```

### 结果

| 指标 | noPCA | 5mm |
| --- | ---: | ---: |
| clean random QFL | 0.0007073 | **0.0001345** |
| perturbed random QFL | **0.0008409** | 0.0009378 |
| perturbed recovery Brier | **0.5760** | 0.3404 |
| perturbed recovery projection | **0.4967** | 0.2447 |

### 关键观察

5mm 的 clean 拟合明显优于 noPCA，但 recovery 明显更差；perturbed correspondence QFL 也略差。这说明 recovery 与 clean correspondence QFL 不是同一个轴，不能只看 clean 拟合指标。

### 解释

该 protocol 对 object-only robustness 是公平的，不能用来回答 hand-only 或 hand+object robustness。5mm 的优势还可能混入更长训练预算（454k vs 317.8k）和 checkpoint 选择差异。

### 结论状态

**INCONCLUSIVE**

方向上不支持“当前 5mm 互斥配置提升 object recovery”；当前规模、subject 覆盖和训练预算仍不足以确认更广泛的因果结论。

### 决策

保留该快速筛查，不把它升级为最终 benchmark；下一轮优先记录并对齐实际 perturb exposure，在相同 PCA 强度下补 hand-only / object-only / hand+object 评测。

### 下一步

- 用同一 object-only protocol 对齐到共同训练 step 或各自 best checkpoint。
- 再抽一个跨 `s01/s02/s04/s05` 的小型版本，保持每物体相同样本数。
- 若要回答完整鲁棒性，再补 hand-only 与 hand+object matched evaluation。

### 证据

- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_compare_20260821.md`
- `src/task/correspondence_ptv3_v2/result/arctic_min11_v1_manifest.json`
- `output/research/arctic_min11_v1/nopca.json`
- `output/research/arctic_min11_v1/5mm.json`

2026-08-21 已停止上述 no-PCA runtime 续训，保留 run 目录和 checkpoint：`outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614`。停止前最新日志为 step 331300 / epoch 37，`latest.pt` 保存到 step 317800 / epoch 35，`best.pt` 仍为 clean QFL 最优的早期 checkpoint。停止原因是转向验证“5 mm exclusive hand/object gate 是否使 object perturb 暴露不足”的独立对照，不再继续消耗 GPU。

2026-08-20 这条 GRAB 续跑又中断了，根因不是模型报错，而是根分区 `/` 已满到 100%，当前只剩约 3.1 GiB 可用；训练日志停在 step 36320 / epoch 4 附近，checkpoint 目录里留下了未完成的 `.tmp` 文件，说明中断发生在保存权重过程中。已把 `dataset/GRAB/data` 切到 NAS 软链并释放本地空间，随后又用 tmux `grab5mm_no_pca_runtime_resume_20260820` 从 `latest.pt` 重新拉起。

正式 OakInk-only 训练已于 2026-08-18 启动：单卡物理 GPU 3（训练进程 `cuda:0`）、50 epoch、batch 16、无手部 PCA 扰动、stored-hand runtime proxy。Run 目录为 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_oakink_50ep_no_hand_perturb_runtime_20260818_145747`，原 W&B run id 为 `d74pw3bu`。该训练在 2026-08-19 08:03 UTC 前后停在日志 step 255300、epoch 19，未留下 Python traceback；由于 `latest.pt` 仅保存到 step 211410、epoch 15，2026-08-19 08:53 UTC 已从该 checkpoint 放入 tmux `oakink_no_hand_runtime_resume_20260819` 继续单卡 GPU 3 训练，新 W&B run id 为 `umayygiz`。

OakInk、纯 GRAB、GRAB+ContactPose 三个 checkpoint 的同协议 ARCTIC 子集对比已完成。纯 GRAB / GRAB+ContactPose 复用已有历史结果，只补测 OakInk；统一协议为 20 个文件、12,389 帧，stored clean hand points，关闭 MANO/PCA hand perturb 与 runtime object resampling，并分别测试 clean 和固定 10° rotation / 10 mm translation object perturb。三方结果已汇总到 `src/task/correspondence_ptv3_v2/result/` 的时间戳文件，当前结论仅适用于该子集，不能外推到全量 ARCTIC。

## 2026-08-20 — 三模型 ARCTIC 子集统一对比（已完成）

### 目标

比较 OakInk-only、纯 GRAB、GRAB+ContactPose 联合训练三个 checkpoint 在相同 ARCTIC 子集上的 correspondence 拟合和 object-perturb recovery 指标，控制评测时间，同时保留可复查的 checkpoint、数据子集和协议记录。

### 当前口径

- ARCTIC 子集：`tmp/arctic_eval_subset_20260817`，20 个文件、12,389 帧。
- 输入：`stored_clean_hand_points`。
- 手部扰动：关闭。
- runtime object resampling：关闭。
- object perturb：固定启用，10° rotation、10 mm translation。
- 纯 GRAB 与 GRAB+ContactPose：复用已有结果 `output/research/contactpose_checkpoint_compare/arctic_pure_20260817_132348.json` 和 `arctic_mixed_20260817_132348.json`。
- OakInk：使用 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_oakink_50ep_no_hand_perturb_runtime_20260818_145747/checkpoints/best.pt`，已完成同协议评测。

### 完成情况

三方的 clean/perturbed random QFL、random MAE、contact auxiliary QFL、recovery Brier 和 recovery projection 已汇总到 `src/task/correspondence_ptv3_v2/result/` 下的时间戳文件。结果中标注了纯 GRAB / GRAB+ContactPose 来自历史 ARCTIC 子集日志，OakInk 为本次补测；三者 checkpoint 训练协议并非完全同源，因此结论仅作方向性比较。

### 结果摘要

- 纯 GRAB 的 correspondence 拟合最好，GRAB+ContactPose 次之，OakInk 最弱。
- recovery 指标同样是纯 GRAB 最好，GRAB+ContactPose 略弱，OakInk 最差。
- 结果文件已落盘，见 `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` 与 `.json`。

## EXP-005 — OakInk / GRAB / GRAB+ContactPose ARCTIC 子集统一对比

### 日期

2026-08-20

### 对应指导

`src/task/correspondence_ptv3_v2/docs/架构.md`

### 假设

在同一 ARCTIC 子集上复用纯 GRAB / GRAB+ContactPose 的历史评测，只补测 OakInk，可以用较低成本判断三套 checkpoint 在相同数据分布下的 correspondence 拟合和 object perturb recovery 差异；如果 OakInk 的训练协议差异过大，则结果只能作为方向性参考。

### Baseline

- 纯 GRAB：`output/research/contactpose_checkpoint_compare/arctic_pure_20260817_132348.json`
- GRAB+ContactPose：`output/research/contactpose_checkpoint_compare/arctic_mixed_20260817_132348.json`
- OakInk：`outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_oakink_50ep_no_hand_perturb_runtime_20260818_145747/checkpoints/best.pt`

### 本次修改

- 复用已有纯 GRAB / GRAB+ContactPose ARCTIC 子集结果。
- 补测 OakInk checkpoint 在同一 ARCTIC 子集上的 clean 与固定 10° / 10 mm object perturb。
- 生成带时间戳的 Markdown/JSON 汇总结果。

### 实现审查

Verdict: PASS

- 使用同一 ARCTIC 子集 `tmp/arctic_eval_subset_20260817`。
- 三方均关闭 MANO/PCA hand perturb 与 runtime object resampling。
- recovery 指标按任务架构定义为越大越好，旧结果页中的相反标注已修正。
- OakInk 仅补测一条 checkpoint，未引入新的评测协议。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python \
  -m src.task.correspondence_ptv3_v2.research.contactpose_checkpoint_compare.evaluate \
  --checkpoint outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_oakink_50ep_no_hand_perturb_runtime_20260818_145747/checkpoints/best.pt \
  --test-root tmp/arctic_eval_subset_20260817 \
  --output /tmp/oakink_arctic_eval_20260820.json \
  --device cuda:0 --batch-size 16 --num-workers 0
```

### 结果

| Metric | OakInk | GRAB | GRAB+ContactPose |
| --- | ---: | ---: | ---: |
| clean random QFL | 0.0004892 | 0.0001431 | 0.0001669 |
| perturbed random QFL | 0.0007355 | 0.0002136 | 0.0002670 |
| clean random MAE | 0.026147 | 0.002711 | 0.004029 |
| perturbed random MAE | 0.026676 | 0.003699 | 0.005797 |
| clean contact auxiliary QFL | 0.010313 | 0.004611 | 0.004878 |
| perturbed contact auxiliary QFL | 0.018312 | 0.006532 | 0.007280 |
| perturbed recovery Brier | 0.5361 | 0.8383 | 0.8000 |
| perturbed recovery projection | 0.4246 | 0.8120 | 0.7728 |
| fake-contact recovery Brier | 0.6692 | 0.9066 | 0.8484 |
| missed-contact recovery Brier | 0.4005 | 0.7687 | 0.7507 |

### 关键观察

纯 GRAB 在 correspondence 和 recovery 两侧都最好，GRAB+ContactPose 居中，OakInk 最弱。OakInk 的 recovery 仍高于随机基线，但与另外两者差距明显。

### 解释

该对比说明在这个 ARCTIC 子集上，OakInk checkpoint 与 GRAB 系列 checkpoint 不是同量级的结果。由于训练数据表示和协议不完全同源，不能把差异直接解释成数据量、联合训练或单一扰动策略的效果。

### 结论状态

**INCONCLUSIVE**

对比有效，但三条 checkpoint 的训练协议差异过大，不能据此做单变量归因。

### 决策

以后若再比较，优先用同数据版本、同训练预算的对照 run；当前以纯 GRAB 作为 ARCTIC 上的参考基线。

### 下一步

如需更强结论，再跑全量 ARCTIC 或补一个严格匹配训练协议的 OakInk/GRAB 对照。

### 证据

- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md`
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.json`
- `/tmp/oakink_arctic_eval_20260820.json`
- `output/research/contactpose_checkpoint_compare/arctic_pure_20260817_132348.json`
- `output/research/contactpose_checkpoint_compare/arctic_mixed_20260817_132348.json`

## EXP-006 — V1 ARCTIC 分层外部评估与 object-macro 汇总

### 日期

2026-08-20

### 对应指导

`src/task/correspondence_ptv3_v2/docs/指导/V1.md`

### 假设

如果把 ARCTIC 评估从旧的 box 偏置子集改成覆盖全部 object / subject / action 的确定性分层子集，那么纯 GRAB 与 GRAB+ContactPose 的对比会更接近 V1 的 held-out 域外比较；若 GRAB+ContactPose 真有稳定退化，object-macro 也应该和 micro 一样维持同向排序。

### Baseline

- pure GRAB：`outputs/train/correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130/checkpoints/latest.pt`
- GRAB+ContactPose：`outputs/train/correspondence_ptv3_v2_old1797_grab_contactpose_full_gpu3_20260816_110237/checkpoints/latest.pt`
- 分层子集：`/tmp/arctic_eval_stratified_v1`

### 本次修改

- 新增确定性分层子集构造脚本。
- 新增 object 级软链子集构造脚本。
- 在同一 evaluator 下跑 pure GRAB / GRAB+ContactPose 的 micro 结果。
- 对 11 个 object 单独评估并取算术平均，得到 object-macro。

### 实现审查

Verdict: PASS

- 覆盖全部 11 个 object、5 个 subject。
- 维持固定的手输入、hand perturb、runtime resampling 和 object perturb 协议。
- 没有改 evaluator、模型或 loss。
- object-macro 来自 per-object 重跑后的算术平均，可复查。

### 实验命令

```bash
PYTHONPATH=. /home/wbcd/miniconda3/envs/graspenv/bin/python \
  -m src.task.correspondence_ptv3_v2.research.contactpose_checkpoint_compare.evaluate \
  --checkpoint outputs/train/correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130/checkpoints/latest.pt \
  --test-root /tmp/arctic_eval_stratified_v1 \
  --output output/research/arctic_v1/stratified/pure_grab.json \
  --device cuda:0 --batch-size 16 --num-workers 0
```

### 结果

| 指标 | pure GRAB micro | GRAB+ContactPose micro | pure GRAB object-macro | GRAB+ContactPose object-macro |
| --- | ---: | ---: | ---: | ---: |
| clean random QFL | 0.0002657 | 0.0002645 | 0.0002789 | 0.0002750 |
| clean random MAE | 0.0037520 | 0.0047859 | 0.0038794 | 0.0048947 |
| clean contact QFL | 0.0057336 | 0.0055567 | 0.0059288 | 0.0056918 |
| perturbed random QFL | 0.0003378 | 0.0003621 | 0.0003500 | 0.0003733 |
| perturbed random MAE | 0.0046071 | 0.0061020 | 0.0047317 | 0.0062039 |
| perturbed contact QFL | 0.0075595 | 0.0078594 | 0.0077316 | 0.0080088 |
| perturbed recovery Brier | 0.7981 | 0.7775 | 0.8102 | 0.7880 |
| perturbed recovery projection | 0.7761 | 0.7452 | 0.7852 | 0.7536 |
| fake-contact recovery Brier | 0.8821 | 0.8397 | 0.8918 | 0.8495 |
| missed-contact recovery Brier | 0.7101 | 0.7125 | 0.7239 | 0.7233 |

### 关键观察

纯 GRAB 在 micro 和 object-macro 上都略优于 GRAB+ContactPose。差距不大，但排序一致；object-macro 没有推翻 micro 结论。

### 解释

这说明旧的 box 子集偏置不是唯一证据来源后，GRAB 仍然保持轻微优势。GRAB+ContactPose 没有在这条 held-out ARCTIC 子集上形成稳定增益。

### 结论状态

**INCONCLUSIVE**

V1 指导下的 held-out ARCTIC 对比有效，但两条 checkpoint 的训练预算仍不完全相同，因此只能说明当前协议下 pure GRAB 略优，不能进一步做严格单变量归因。

### 决策

把 `pure GRAB` 视为当前 ARCTIC held-out 的参考基线；如果后续要再比较，优先补同预算的对照，而不是直接改 evaluator。

### 下一步

若需要更强结论，再补同训练预算、同数据协议的 GRAB+ContactPose 或 OakInk 对照。

### 证据

- `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.md`
- `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.json`
- `output/research/arctic_v1/arctic_v1_summary.json`
- `output/research/arctic_v1/stratified/manifest.json`
- `output/research/arctic_v1/objectwise/*/*.json`

## EXP-001 — OakInk wrist 坐标与 stored-hand runtime proxy 验证

### 日期

2026-08-18

### 对应指导

当前 task 架构：`src/task/correspondence_ptv3_v2/docs/架构.md`

### 假设

OakInk 的 hand/object 在同一 camera frame 中减去 wrist 可以保持 clean contact 语义；同时使用存储 hand points 的固定 spatial proxy 应能近似现有 MANO proxy 的 runtime near-pool 分布。

### Baseline

- commit: `d5510e0`
- config: `src/task/correspondence_ptv3_v2/configs/hand_heatmap.yaml`
- checkpoint: 无（实现 gate）

### 本次修改

- OakInk 多视角 wrist/camera/world 验证脚本。
- `use_mano_reconstruction=false` + `runtime_resample_object=true` 的 stored-hand proxy 路径。
- 版本化 OakInk FPS-256 proxy 索引与候选配置。

### 实现审查

Verdict: PASS

- 无 MANO 字段时不执行 MANO forward。
- 缺 proxy 配置时 fail-fast。
- runtime 仍从 full 4096 pool 选择 near 384 + global 128。
- OakInk 样本没有数值非法或重复选择。

### 实验命令

```bash
PYTHONPATH=. python -m src.task.correspondence_ptv3_v2.research.oakink_conversion.validate_wrist_centering \
  --oakink-root /mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk \
  --groups 8 \
  --output output/research/oakink_wrist_centering/metrics.json
```

### 结果

| Metric | Result |
| --- | ---: |
| wrist-only centered hand cross-view RMS | 163.14 mm mean |
| inverse-extrinsic world-centered hand RMS | 0.000080 mm mean |
| cross-view contact-distance MAE | 0.000035 mm mean |
| stored/MANO proxy top-1024 Jaccard | 0.948 mean |
| PTv3 smoke loss | 0.206954 |

### 关键观察

只减 wrist 不会破坏手物接触距离，但不会消除 camera rotation。OakInk-only 监督可用；与严格 rotation-canonicalized hand-root 数据混训前需单独评估旋转分布差异。

### 解释

这个 gate 支持“保留 OakInk clean contact 语义”和“无 MANO 依赖保留 runtime sampling”，但没有提供 OakInk 训练后的泛化性结论。

### 结论状态

**SUPPORTED**（仅针对实现和几何语义 gate，不是训练效果结论）

### 决策

保留 stored-hand proxy 路径，在正式训练前不回退到关闭 runtime sampling。

### 下一步

确认正式训练的 epoch/step 、GPU 配额和是否需要与 GRAB 保持相同优化预算，然后启动 OakInk-only 训练。

### 证据

- `output/research/oakink_wrist_centering/metrics.json`
- `src/task/correspondence_ptv3_v2/calibration/oakink_stored_hand_fps256.json`
- `src/task/correspondence_ptv3_v2/configs/oakink_50ep_no_hand_perturb_runtime.yaml`

## EXP-002 — GRAB 5 mm 互斥扰动版与两条 9 mm best.pt 对比

### 日期

2026-08-19

### 对应指导

`src/task/correspondence_ptv3_v2/docs/架构.md`

### 假设

将 GRAB 的手扰动从 9 mm 降到 5 mm，并改成手 / 物体互斥扰动，可能会让 correspondence 更容易拟合，同时保持或改善扰动恢复能力；如果 recovery 指标明显下降，说明更弱的手噪声不足以支撑 perturbation recovery。

### Baseline

- 当前 5 mm run: `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/checkpoints/best.pt`
- 主对照 9 mm rebuilt run: `outputs/correspondence_ptv3_v2/grab_full_v21_geometry9mm_ddp2_rebuiltdata_newrun_20260815_112700/checkpoints/best.pt`
- 参考 9 mm old run: `outputs/correspondence_ptv3_v2/grab_full_v21_geometry9mm_ddp2_newrun_20260810_113610/checkpoints/best.pt`

### 本次修改

- 无代码改动。
- 直接从三条 run 的 `metrics.jsonl` 中提取各自当前 `best.pt` 对应的验证记录。

### 实现审查

Verdict: PASS

- 三条 run 都使用 `val_clean/cross_edge_random_qfl` 作为 `best.pt` 选择标准。
- 对比只使用已有验证日志，没有引入新的评测协议或额外测试集。
- `pseudo_recovery_*` 指标按任务架构定义为“越大越好”。

### 实验命令

```bash
/home/wbcd/miniconda3/envs/graspenv/bin/python - <<'PY'
...读取三条 metrics.jsonl 并选出 val_clean/cross_edge_random_qfl 最优记录...
PY
```

### 结果

| Metric | 5 mm best.pt | 9 mm rebuilt best.pt | 9 mm old best.pt |
| --- | ---: | ---: | ---: |
| best step / epoch | 208840 / 23 | 581088 / 32 | 399498 / 22 |
| val_clean/cross_edge_random_qfl | 0.0003275 | 0.0011168 | 0.0013895 |
| val_perturbed/cross_edge_random_qfl | 0.0007359 | 0.0015789 | 0.0016656 |
| val_perturbed/pseudo_recovery_brier | 0.3449 | 0.5472 | 0.5724 |
| val_perturbed/pseudo_recovery_projection | 0.3396 | 0.4177 | 0.4505 |
| val_perturbed/pseudo_fake_contact_recovery_brier | 0.2439 | 0.6357 | 0.7345 |
| val_perturbed/pseudo_missed_contact_recovery_brier | 0.4545 | 0.4503 | 0.3947 |

### 关键观察

5 mm 版在 clean / perturbed QFL 上明显更好，说明 correspondence 拟合更强；但 `pseudo_recovery_*` 三个主恢复指标都低于 9 mm rebuilt baseline，尤其 fake-contact recovery 掉得最明显。`missed-contact` recovery 仅和 rebuilt 9 mm 接近，但没有超过它。

### 解释

这更像是“更容易拟合主监督，但 perturbation recovery 变弱”的折中，而不是整体性能单调提升。换句话说，5 mm + 互斥门控把模型往 correspondence 端推得更稳，但没有把 recovery 端一起推上去。

### 结论状态

**INCONCLUSIVE**

当前证据支持 5 mm 版在 correspondence 端更好，但不足以把它判成整体优于 9 mm rebuilt baseline；恢复指标的下降是实质性的。

### 决策

5 mm 版本可以继续作为“偏 correspondence”的候选 run 观察，但当前不把它当作 9 mm rebuilt baseline 的全面替代。

### 下一步

如果后续还要继续比，优先补一个固定外部测试集，或者至少等 5 mm 训练到更稳定的后期再看 recovery 指标是否追平 9 mm rebuilt。

### 证据

- `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/metrics.jsonl`
- `outputs/correspondence_ptv3_v2/grab_full_v21_geometry9mm_ddp2_rebuiltdata_newrun_20260815_112700/metrics.jsonl`
- `outputs/correspondence_ptv3_v2/grab_full_v21_geometry9mm_ddp2_newrun_20260810_113610/metrics.jsonl`

## EXP-003 — GRAB 5 mm 互斥扰动版与旧 no-PCA legacy run 的方向性对比

### 日期

2026-08-19

### 对应指导

`src/task/correspondence_ptv3_v2/docs/架构.md`

### 假设

如果把 5 mm 互斥扰动版和仓库里唯一明确的 no-PCA legacy GRAB run 对比，5 mm 版应当至少在 clean correspondence 上不差，但 recovery 是否更好需要实测。

### Baseline

- 当前 5 mm run: `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/checkpoints/best.pt`
- no-PCA legacy run: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_50ep_hand_heatmap_stage3data_20260805_041125/checkpoints/best.pt`

### 本次修改

- 无代码改动。
- 直接从两条 run 的 `metrics.jsonl` 中提取各自当前 `best.pt` 对应的验证记录。

### 实现审查

Verdict: PASS

- 对比只使用已有验证日志，没有引入新的评测协议或额外测试集。
- 该对比是方向性的，不是严格 apples-to-apples：legacy run 仍是旧 stage3data/v20 语境，当前 run 是 v21 语境。

### 实验命令

```bash
/home/wbcd/miniconda3/envs/graspenv/bin/python - <<'PY'
...读取两条 metrics.jsonl 并选出 val_clean/cross_edge_random_qfl 最优记录...
PY
```

### 结果

| Metric | 5 mm best.pt | old no-PCA legacy best.pt |
| --- | ---: | ---: |
| best step / epoch | 208840 / 23 | 9080 / 4 |
| val_clean/cross_edge_random_qfl | 0.0003275 | 0.0004991 |
| val_perturbed/cross_edge_random_qfl | 0.0007359 | 0.0006805 |
| val_perturbed/pseudo_recovery_brier | 0.3449 | 0.6845 |
| val_perturbed/pseudo_recovery_projection | 0.3396 | 0.6045 |
| val_perturbed/pseudo_fake_contact_recovery_brier | 0.2439 | 0.7294 |
| val_perturbed/pseudo_missed_contact_recovery_brier | 0.4545 | 0.6457 |

### 关键观察

5 mm 版的 clean QFL 更好，但 perturbed QFL 略差；在所有 `pseudo_recovery_*` 指标上，5 mm 版都明显低于旧 no-PCA legacy run。也就是说，旧 no-PCA 版本更偏向 recovery，5 mm 版本更偏向 clean correspondence。

### 解释

这组结果更像是“把模型从 recovery 端挪到 correspondence 端”。如果只看 clean 拟合，5 mm 版更强；如果看扰动恢复，旧 no-PCA legacy 版更强。

### 结论状态

**INCONCLUSIVE**

由于 baseline 属于旧 stage3data/v20 语境，不能把这组结果当作严格同协议结论；但方向上已经说明 5 mm 版不是对 no-PCA legacy 的全面提升。

### 决策

不把旧 no-PCA legacy run 视为当前 v21 训练的主 baseline，只保留为方向性参考。

### 下一步

如果要做严格比较，需要补一个同数据版本、同训练预算的 v21 no-PCA 对照 run。

### 证据

- `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/metrics.jsonl`
- `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_50ep_hand_heatmap_stage3data_20260805_041125/metrics.jsonl`

## EXP-004 — GRAB 5 mm 互斥扰动版与 `finetune_day_cosine_restart` 复现对比

### 日期

2026-08-19

### 对应指导

`src/task/correspondence_ptv3_v2/docs/架构.md`

### 假设

如果 5 mm 互斥扰动版能复现你说的那条“几天前、与 `finetune_day_cosine_restart` 保持一致”的 no-PCA 参考，至少在 clean / perturbed QFL 上不应明显更差。

### Baseline

- 当前 5 mm run: `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/checkpoints/best.pt`
- 原始 no-PCA 参考: `outputs/train/correspondence_ptv3_v2_finetune_day_cosine_restart/checkpoints/best.pt`
- 几天前复现 run: `outputs/train/correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130/checkpoints/best.pt`

### 本次修改

- 无代码改动。
- 直接从三条 run 的 `metrics.jsonl` 提取 `best.pt` 对应记录。

### 实现审查

Verdict: PASS

- `finetune_day_cosine_restart` 和 `old1797_compact_repro` 都是同一条 cosine_restart 语境下的对照。
- 这两条 no-PCA 参考没有 `pseudo_recovery_*` 记录，因此只比较可直接对齐的 QFL 指标。

### 实验命令

```bash
/home/wbcd/miniconda3/envs/graspenv/bin/python - <<'PY'
...读取三条 metrics.jsonl 并选出 val_clean/cross_edge_random_qfl 最优记录...
PY
```

### 结果

| Metric | 5 mm best.pt | finetune_day no-PCA best.pt | repro best.pt |
| --- | ---: | ---: | ---: |
| best step / epoch | 208840 / 23 | 113500 / 50 | 127120 / 56 |
| val_clean/cross_edge_random_qfl | 0.0003275 | 0.0002550 | 0.0002668 |
| val_perturbed/cross_edge_random_qfl | 0.0007359 | 0.0003530 | 0.0003654 |

### 关键观察

5 mm 版在这两条 no-PCA 参考上都更差，而且是 clean / perturbed 两项都更差。相对原始 `finetune_day_cosine_restart`，clean QFL 约差 28%，perturbed QFL 约差 108%；相对几天前的复现 run，clean 约差 23%，perturbed 约差 102%。

### 解释

这说明当前 5 mm PCA 互斥扰动版没有把 no-PCA 那条路线复现出来，反而把模型推向了另一种折中：clean correspondence 仍然能学，但扰动后的稳定性没跟上。

### 结论状态

**REFUTED**

“5 mm 互斥扰动版能复现 `finetune_day_cosine_restart` 这条 no-PCA 参考”的假设被当前 best.pt 指标否定。

### 决策

把 `finetune_day_cosine_restart` 视为当前问题下的 canonical no-PCA 参考；`old1797_compact_repro` 只作为复现检查。

### 下一步

如果还想继续追这个方向，先补一个同版本、同预算的 v21 no-PCA 对照，再判断 5 mm 改动到底带来的是收益还是协议漂移。

### 证据

- `outputs/train/correspondence_ptv3_v2_finetune_day_cosine_restart/metrics.jsonl`
- `outputs/train/correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130/metrics.jsonl`
- `outputs/correspondence_ptv3_v2/grab5mm_exclusive_ddp2_retry_20260817_20260817_233019/metrics.jsonl`

## 2026-08-19 — 训练吞吐与四数据集规模核验

这是运行监测和数据规模统计，不构成新的训练效果实验结论。

### 训练状态

| Run | GPU | 进度 | 日志 ETA | 吞吐 |
| --- | --- | ---: | ---: | ---: |
| GRAB 5 mm exclusive DDP | physical 1, 2 | 343700 / 454000 (75.70%) | 7.83 h | 125.2 samples/s（全局） |
| OakInk no-hand-perturb runtime | physical 3 | 173800 / 704700 (24.66%) | 32.47 h | 72.7 samples/s |

两条 run 的 metrics mtime 分别为 2026-08-19 02:30:38 UTC（GRAB）与 02:30:30 UTC（OakInk）。当时 GPU 2/3 的利用率均为 100%/78%（瞬时采样），训练未停滞。

### 统一统计口径

- 一个 `.npz` 是一个序列段；其中第 0 维的每一帧对应一个训练样本。
- 4096 个物体点、1538 个手点是单个样本内部几何，不额外扩大样本数。
- 帧数通过读取全部 NPZ 内 `.npy` 数组头获得，不解压几何数组；磁盘量为当前 Stage-3 文件总量。

| 数据集 / 版本 | NPZ 序列段 | 帧样本 | 平均帧/段 | Stage-3 磁盘量 |
| --- | ---: | ---: | ---: | ---: |
| GRAB v2.1 | 1963 | 327808 | 167.0 | 37.03 GiB |
| OakInk mano-face hand-root 导出 | 2596 | 252172 | 97.1 | 16.06 GiB |
| ContactPose use_stage3_v2 | 885 | 403452 | 455.9 | 28.71 GiB |
| ARCTIC hand-root v2 | 297 | 161181 | 542.7 | 21.84 GiB |

### OakInk 耗时解释

- OakInk 已使用启动索引 cache：`output/cache/oakink_stage3_corr_index.json` 包含 2596 文件、252172 帧。其 `perf/data_wait_ms=0.19 ms`，仅约 step 时间的 0.09%，因此并非未缓存或 NAS 读取造成主要瓶颈。该索引只避免启动时扫描/计数，并不预计算每步的 runtime sampling。
- OakInk 禁用 MANO/PCA 后单步为约 220.2 ms，快于 GRAB 的约 255.7 ms，说明确实省掉了 MANO 重建和手扰动计算。
- 但 OakInk 仍保留 full-pool runtime object resampling：每样本从 4096 物点相对 256 hand proxy 计算 near pool，选取 near 384 + global 128 点；PTv3 前后向和这部分计算仍是主耗时。
- OakInk 是单卡全局 batch 16，GRAB 是两卡全局 batch 32。实际每 epoch 的有效样本/step 分别为 OakInk 225504/14094、GRAB 290560/9080。因此 OakInk 原始帧数虽仅为 GRAB 的 76.9%，每 epoch optimizer step 却是 GRAB 的 1.55 倍；这才是 50 epoch wall time 更长的主因。

### 相同协议下的 step 规模参考

| 数据集 | 已知或估算有效训练样本/epoch | 单卡 global batch 16 的 step/epoch | 50 epoch step |
| --- | ---: | ---: | ---: |
| OakInk（实际） | 225504 | 14094 | 704700 |
| GRAB（实际；DDP global batch 32） | 290560 | 9080 | 454000 |
| ContactPose（按 10% sequence validation 近似估算） | ~363k | ~22694 | ~1.13M |
| ARCTIC（按 10% sequence validation 近似估算） | ~145k | ~9066 | ~453k |

ContactPose 与 ARCTIC 的最后三列仅是按当前 batch/validation 比例的规模估算；因 sequence-level split 和 `drop_last`，正式 run 应以启动后的实际 `steps_per_epoch` 为准。

## EXP-007 — 三域等比例从头混训 baseline

### 日期

2026-08-20

### 对应指导

当前三域混训方案；V1 ARCTIC 外部评估作为启动前依据。

### 假设

在不引入 runtime object resampling 和 scale augmentation 的条件下，按 GRAB、ContactPose、OakInk 各 1/3 的 domain-balanced sampling 训练，可以避免原始 frame 数量导致的域偏置，并提供可解释的三域互补 baseline。

### Baseline

- commit: `working tree`
- config: `src/task/correspondence_ptv3_v2/configs/mixed_grab_contactpose_oakink_equal.yaml`
- checkpoint: random initialization；无 resume

### 本次修改

- 独立构造 GRAB / ContactPose / OakInk Dataset；
- 每个 local batch 固定 `16/16/16`；
- `runtime_resample_object=false`；
- 无 scale augmentation；
- `use_mano_reconstruction=false`、`apply_hand_perturb=false`；
- 保留统一 object rotation/translation perturbation；
- optimizer budget 临时以历史 mixed 的 `154670` optimizer steps 对齐。

### 实现审查

Verdict: PASS（截至 step 20）

关键检查：
- 三域 batch 比例为 1/3；
- 无 runtime full-pool 字段；
- 无 MANO / non-MANO collate 冲突；
- DDP world size 2，物理 GPU 0、3；
- `grad_clipped=0`，首个记录的 `grad_norm=0.825555`；
- 无 checkpoint resume。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=0,3 PYTHONPATH=. \
python -m torch.distributed.run --standalone --master_port=29504 \
  --nproc_per_node=2 --module src.task.correspondence_ptv3_v2.train \
  --config src/task/correspondence_ptv3_v2/configs/mixed_grab_contactpose_oakink_equal.yaml \
  --set train.max_steps=154670 --distributed
```

### 结果

旧 run 后续在日志 step 27980 中断，最近完整 checkpoint 为 step 22748 / epoch 1；未发现 Python traceback、CUDA OOM 或 kernel OOM。2026-08-22 05:02 UTC 已在物理 GPU 1、2 上从该 checkpoint 恢复到独立输出目录，并创建新的 W&B run。恢复后已确认 step 22760 至 22820 连续写出，尚无恢复后的新 validation checkpoint。

| Metric | Value |
|---|---:|
| world size | 2 |
| global batch | 96 |
| total steps | 154670 |
| first logged grad norm | 0.825555 |
| first logged loss | 0.152940 |

### 关键观察

等比例 sampler 使每 epoch 约 24985 steps，因此配置中的 100 epoch 若不加 max_steps 会展开到约 227 万 steps；本次已在首个 optimizer step 前改用 154670-step 上限。

### 解释

当前尚不能判断三域混合是否提升 ARCTIC。需要等待至少一个完整 validation 周期，并分别查看三个训练域与 ARCTIC 的指标。

### 结论状态

INCONCLUSIVE

### 2026-08-22 续训状态

- physical GPU: 1、2；DDP world size 2；global batch 96；
- resume checkpoint: 旧 run `latest.pt`，step 22748 / epoch 1；
- output: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_resume_gpu12_20260822_050212`；
- W&B run ID: `d5yvor4z`（新 run，不续接旧 run ID）；
- total optimizer-step budget: 154670；
- 启动初期 NAS data wait 偏高，短窗口吞吐不作为稳定 ETA 依据。

### 决策

保留当前 run；不再修改采样比例、runtime 或 scale 设置。若出现明显域间 loss/gradient 差异，先记录诊断，再单独设计 loss balancing 消融。

### 下一步

- 等待首个 validation checkpoint；
- 检查每域 `val_clean` / `val_perturbed` 指标；
- 训练完成后按 V1 方案评估全量或全类别 ARCTIC；
- 统计三域曝光量、梯度范数和 clipping ratio。

### 证据

- run: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_20260820_141453`
- tmux: `mixed_grab_contactpose_oakink_equal_20260820_v1`
- W&B: `https://wandb.ai/hitsz-oyx/ref2dex/runs/div3m72u`
- resumed run: `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_resume_gpu12_20260822_050212`
- resumed tmux: `mixed_equal_resume_20260822_gpu12`
- resumed W&B: `https://wandb.ai/hitsz-oyx/ref2dex/runs/d5yvor4z`
## EXP-014 — 各数据集相邻帧手部运动幅度诊断（2026-08-30）

- **假设**：在不额外跳帧的情况下，可用相邻 Stage 3 帧的手点位移估计数据集的自然运动尺度，并据此同步扰动幅度。
- **方法**：对每个 NPZ 的相邻存储帧计算 1538 个手点欧氏位移的 frame-mean 与 frame-RMS；有 `obj_root_pose_world` 的域额外转换到 object frame。单位为 mm/相邻存储帧。
- **结论**：object-frame RMS 的均值约为 HRDex human 3.04、Inspire DFTP 2.75、Inspire F1 2.52、Allegro V5 3.39、GRAB 6.19 mm。GRAB Stage 3 的 `raw_frame_id` 间隔恒为 4，故其值对应预处理后的 4-frame 间隔；ContactPose 导出序列内手点/MANO/root 全部静止；OakInk 有 24.1% 非连续 raw id，直接相邻统计会被跳变污染。
- **状态**：SUPPORTED（作为运动尺度诊断；不作为原始视频 stride=1 的严格测量）。

### Temporal stride 对比

- Inspire F1 object-frame hand RMS 均值：stride=1/2/3 分别为 `2.515/4.291/5.933 mm`；对应中位数为 `1.305/2.213/3.034 mm`。
- GRAB 当前 Stage 3 的 raw frame 间隔恒为 4，object-frame hand RMS 均值/中位数为 `6.190/3.733 mm`。
- 判断：F1 stride=2 是合理的中间折中，但若目标是匹配 GRAB 的单帧幅度，stride=3 更接近；最终 stride 仍需结合两套数据的真实采样频率确定，不能只按位移倍数选择。

## EXP-015 — object-centered GRAB + Inspire F1 checkpoint 的 ARCTIC min11 快速外部评估（2026-08-31）

### 目的

快速检查当前验证最佳 checkpoint 在未参与训练的 ARCTIC 数据上的迁移效果，先使用已有 11 物体子集，避免直接等待全量逐帧评估。

### 协议

- checkpoint：`outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_grab_inspire_f1_object_centered_root_pose_20260829_135110/checkpoints/best.pt`，实际保存 step `148008` / epoch `28`；
- 数据：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`，11 个序列、4175 帧；
- 坐标：输入树为 hand-root，评估器使用 `hand_root_pose` 与 `obj_root_pose_world` 惰性转换到 object frame；
- 条件：分别评估 `object_only`、`hand_only`、`hand_object`；hand 条件使用 axis-angle45、5 mm RMS，object 条件使用 10° rotation + 10 mm translation，runtime object resampling 均关闭；
- batch size 16，GPU 1，单进程。

### 结果

| condition | clean QFL | perturbed QFL | ΔQFL | perturbed MAE | recovery Brier | recovery projection |
|---|---:|---:|---:|---:|---:|---:|
| object-only | 0.00014361 | 0.00027877 | +0.00013516 | 0.007418 | 0.4344 | 0.3490 |
| hand-only | 0.00014367 | 0.00015647 | +0.00001280 | 0.006849 | -0.0562 | 0.3990 |
| hand+object | 0.00014367 | 0.00028649 | +0.00014283 | 0.007303 | 0.4342 | 0.3445 |

object-only 与 hand+object 的 random QFL 分别增加 `0.00013516`（约 `94.1%`）和 `0.00014283`（约 `99.4%`）；hand-only 的 QFL 仅增加 `0.00001280`（约 `8.9%`），但其 recovery Brier 为 `-0.0562`，说明在该外部 hand 扰动协议下恢复质量不理想。三种条件均能完成推理。

### 实现核验与限制

- 外部 evaluator 发现并修复了混合 checkpoint 的 `domain_paths`、mixed hand/robot contract 和 object-frame 转换兼容问题；训练权重和训练配置未修改。
- 先启动的全量 `arctic_full_mano_v21_20260823` 评估因 NAS 逐帧 I/O 过慢在约 1 小时后主动中止，无结果文件；本 EXP 仅采用 min11 快速子集。
- 子集覆盖 11 个物体但仅为历史 min11 规模，不能替代全量或按 object-macro 的正式 benchmark，也没有同协议 baseline 横向对照。

### 结论状态

**INCONCLUSIVE**（工程迁移和方向性结果可用，但不足以形成 ARCTIC 泛化结论）

### 证据

- `output/research/arctic_v1/stratified/object_centered_best_object_only_20260831.json`
- `output/research/arctic_v1/stratified_object_centered_best_object_only_20260831.log`

## EXP-016 — object-centered GRAB + Inspire F1 checkpoint 的 HOCap subject_1 全量外部评估（2026-08-31）

### 目的

验证当前 object-centered checkpoint 在 HOCap 未参与训练的数据上的迁移能力，并与既有纯 GRAB HOCap baseline 使用相同的 object-only 协议对照。

### 协议

- checkpoint：当前 `best.pt`，step `148008` / epoch `28`；
- 数据：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hocap_subject1_annotation_v1`，28 个序列、23896 帧；
- HOCap 原始 Stage 3 为 hand-root，使用 `hand_root_pose` 与 `obj_root_pose_world` 无损转换到 object frame；
- clean 使用存储的 clean hand points；perturbed 固定 object 10° rotation + 10 mm translation；
- runtime object resampling 关闭；batch size 16，GPU 1，单进程；关闭训练期交互帧过滤，覆盖全部帧。

### 结果

| Metric | clean | object perturbed |
|---|---:|---:|
| `cross_edge_random_qfl` | 0.00007205 | 0.00015032 |
| `cross_edge_random_mae` | 0.002261 | 0.002686 |
| `pseudo_recovery_brier` | — | 0.4886 |
| `pseudo_recovery_projection` | — | 0.4080 |
| `pseudo_changed_edge_fraction` | 0 | 0.002767 |

object perturb 使 random QFL 增加 `0.00007826`（约 `108.6%` 相对 clean）。相较既有纯 GRAB noPCA baseline（clean `0.00016607`、perturbed `0.00020392`、recovery Brier `0.5267`），当前 checkpoint 的 clean / perturbed QFL 分别降低约 `56.6%` / `26.3%`，但 recovery Brier 低 `0.0381`。

### 实现核验与限制

- 外部评估器已修复混合 checkpoint 的 `domain_paths`、mixed hand/robot contract、object-frame 转换和训练期交互帧过滤继承问题；训练权重与训练配置未修改。
- 结果覆盖 subject_1 的 28 个序列，但 HOCap 接触 GT 为几何派生，且仍只有单一 subject；与既有 baseline 的训练预算、坐标协议和模型结构并非完全 matched，因此只作方向性外部证据。

### 结论状态

**INCONCLUSIVE**（工程兼容性和方向性泛化结果已确认，不能据此形成跨 subject 正式结论）

### 证据

- `output/research/hocap_subject1_object_centered_20260831/object_centered_best_object_only_full.json`
- 既有 baseline：`output/research/hocap_subject1_grab_nopca_20260824/full_object_only.json`

## EXP-017 — 七域混合 best checkpoint 的 HOCap 全量外部评估（2026-09-01）

### 目的

验证完成七域混合续训后的最佳 checkpoint 在 HOCap 完整 subject_1 数据上的迁移表现。

### 协议

- checkpoint：七域混合 `best.pt`，step `34907` / epoch `1`；
- 数据：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/hocap_subject1_annotation_v1`，28 个序列、共 23896 帧；
- 条件：沿用既有 HOCap 全量 `object_only` 协议，clean 使用存储手点，perturbed 使用 10° object rotation + 10 mm translation；runtime object resampling 关闭；
- batch size 16，GPU 1，单进程，关闭训练期交互帧过滤。

### 结果

| Metric | clean | object perturbed |
|---|---:|---:|
| `cross_edge_random_qfl` | 0.00007355 | 0.00014111 |
| `cross_edge_random_mae` | 0.004540 | 0.005136 |
| `pseudo_recovery_brier` | — | 0.5913 |
| `pseudo_recovery_projection` | — | 0.5026 |

object perturb 使 random QFL 增加 `0.00006757`（约 `91.9%`）。与上一版 object-centered GRAB + Inspire F1 checkpoint 的 HOCap 结果（clean `0.00007205`、perturbed `0.00015032`、recovery Brier `0.4886`）相比，七域版 clean 基本持平，perturbed QFL 略低，但 recovery Brier / projection 较差。该比较仅作方向性参考，两个 checkpoint 的坐标系、训练域和训练协议不同。

### 实现核验与限制

- 全量 object-only 推理正常完成，覆盖 HOCap 当前可用的全部 28 个序列 / 23896 帧。
- 尝试 `hand_only` / `hand_object` 时发现现有 HOCap Stage 3 将 axis-angle45 数据错误标记为 `mano_use_pca=True`，评测器因此缺少匹配的手噪声 profile；这两种条件本次未产出结果，不将失败当作模型结论。
- HOCap 当前只有 subject_1，接触 GT 为几何派生，仍不足以形成正式跨主体 benchmark。

### 结论状态

**INCONCLUSIVE**（全量 object-only 工程评测完成，科研泛化结论受数据与协议限制）

### 证据

- `output/research/hocap_subject1_seven_domain_20260901/object_only.json`
- `output/research/hocap_subject1_object_centered_20260831/object_centered_best_object_only_full.json`

## EXP-018 — HOCap PCA45 profile 修正、subject_1 重导出与三条件评测（2026-09-01）

### 目的

修复 HOCap 手部扰动评测因缺少 PCA45 几何噪声 profile 而无法运行的问题，并在重导出后的完整 subject_1 数据上完成七域 checkpoint 的三条件评测。

### 实现修正

- 核验原始 `poses_m.npy` 与 `process/HOCap/stage3_export.py`：HOCap 手部参数为 MANO PCA45，原有 `mano_use_pca=True` / `mano_num_pca_comps=45` / `mano_pose_repr="pca"` 标记是正确的，不将其错误改写为 axis-angle；
- 使用完整 subject_1 Stage 3 样本标定 `hocap_pca45_target_9mm.json`，并让外部 evaluator 对 HOCap 自动选择 `hocap` 数据集 ID 与该 profile；
- 使用包含 cv2/PyYAML 的 `hocopt` 环境重新导出 subject_1，7 个序列、28 个文件、23896 帧，`failed=0`；重导出前后抽查文件的几何与 MANO 字段完全一致。

### 协议

- checkpoint：七域混合 `best.pt`，step `34907` / epoch `1`；
- HOCap：`processed_data/stage3/hocap_subject1_annotation_v1`，完整 28 序列 / 23896 帧；
- `object_only`：10° object rotation + 10 mm translation；
- `hand_only`：HOCap PCA45 几何标定噪声，目标 9 mm RMS；
- `hand_object`：同时启用上述 hand/object 扰动；runtime object resampling 关闭；batch size 16，GPU 1，单进程。

### 结果

| condition | clean QFL | perturbed QFL | perturbed MAE | recovery Brier | recovery projection |
|---|---:|---:|---:|---:|---:|
| object-only | 0.00007366 | 0.00014232 | 0.005145 | 0.5907 | 0.5015 |
| hand-only | 0.00007365 | 0.00015818 | 0.004488 | 0.4739 | 0.3962 |
| hand+object | 0.00007365 | 0.00024166 | 0.004871 | 0.4617 | 0.3552 |

### 结论状态

**INCONCLUSIVE**（工程链路已修复并完成全量 subject_1 评测；HOCap 仍只有一个主体，接触 GT 为几何派生，结果仅作方向性证据）

### 证据

- `src/task/correspondence_ptv3_v2/calibration/hocap_pca45_target_9mm.json`
- `output/research/hocap_subject1_seven_domain_reexport_20260901/{object_only,hand_only,hand_object}.json`
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/HOCap/logs/subject_1_stage3_reexport_20260901.log`
