# Experiment log

## 当前状态

已完成 OakInk 坐标和无 MANO runtime sampling 的实现 gate，正式训练进行中；同时已完成 GRAB 5 mm PCA 互斥扰动版与两条 9 mm baseline 的当前 `best.pt` 对比。5 mm 版在 correspondence QFL 上更好，但 recovery 指标整体落后于 9 mm rebuilt baseline；和旧 no-PCA legacy run 比，clean QFL 更好，但 perturbed / recovery 指标更差；后续 no-PCA 默认对照已切到 8 月中旬的 `old1797_compact_repro`，它和当前 5 mm 语境更接近。当前 GRAB “5 mm 配置 + runtime sampling + no-PCA hand perturb”训练目录为 `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_full_grab_geometry_5mm_no_pca_runtime_ddp2_20260819_124614`；该 run 原进程停在日志 step 36320 / epoch 4 附近，但可用 `latest.pt` 只保存到 step 27240 / epoch 3，2026-08-20 已从该 checkpoint 放入 tmux `grab5mm_no_pca_runtime_resume_20260820` 继续两卡训练，W&B run id 为 `zeqrhxyk`。V1 指导下的 ARCTIC 外部评估也已完成：分层子集覆盖 11 个 object / 5 个 subject，共 179 条序列、93,968 帧；pure GRAB 在 micro 和 object-macro 上都略优于 GRAB+ContactPose，差距幅度有限，结果页已写入 `src/task/correspondence_ptv3_v2/result/arctic_grab_grabcontactpose_compare_20260820_090000.md`。

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
