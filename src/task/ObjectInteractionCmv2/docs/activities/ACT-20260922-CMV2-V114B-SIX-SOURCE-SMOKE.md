# ACT-20260922-CMV2-V114B-SIX-SOURCE-SMOKE

- timestamp：`2026-09-22T09:14:20+08:00`
- activity_id：`ACT-20260922-CMV2-V114B-SIX-SOURCE-SMOKE`
- work_version：`V1.14`
- git_commit：`a7ad08b8f1583778d0f13f6d6f4b0e1ebbe333ab`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope：最低成本真实数据六组接口 smoke（`GRAB/ARCTIC/OakInk2 × MANO/Inspire`）
- level：`run`；bounded smoke，不是正式训练
- approval：用户于 2026-09-22 明确要求一次覆盖、采用最低成本并以通过为目标

## 运行与产物

- run_id：`cmv2_v114b_six_source_smoke_20260922_a7ad08b`
- run_status：`COMPLETED`
- 命令入口：`python -m src.task.ObjectInteractionCmv2.train_part_se3_v114_ddp launch --smoke-steps 1`
- 输出：`outputs/objectinteractioncmv2/cmv2_v114b_six_source_smoke_20260922_a7ad08b/`
- 最后 step/epoch：`1/1`
- best metric：`72.63413365681964`；仅为随机初始化 smoke 的运行记录，不解释为模型效果
- checkpoint：`latest.pt`、`best.pt`
- 证据：`run_manifest.json`、`metrics.jsonl`、`train.log`

输入物化为 smoke-only 小缓存：GRAB 与 ARCTIC 各 2 条 8-frame 序列，OakInk2 为 2 个 8-frame official-30 Hz
片段；总目录约 90 MB，位于
`data/processed_data/ObjectInteractionCmv2/v114b/six_source_smoke/`。active split 的 train/val 对六个 group 均各含
1 条记录。该目录不得替代正式 cache，也不得作为正式训练数据合同。

## 结果与验证

- 单个训练 batch 的 `batch_counts_per_rank` 明确为六组各 `1`，证明一次 batch 同时覆盖全部 group。
- 六组均完成 validation stride `1/2/3`，每个 `(group,stride)` 读取 4 个样本；没有缺组、异常退出或 non-finite
  指标。
- 实现提交前：定向测试 `8 passed`，完整 Task 测试 `78 passed`，`python tools/verify.py --changed` 为
  `VERIFY PASS`，Python compile 与 `git diff --check` 通过。
- 本轮为工程接线 smoke；scientific conclusion：`N/A`。loss 与 best metric 不支持模型质量、跨域泛化或收敛结论。

## 改变、保护与回滚

- 新增 bounded 六组输入/config/metadata helper；MANO producer 仅在显式 `--allow-partial` 时接受小 assignment；
  ARCTIC Inspire 输出保留既有 part/articulation/root-pose 合同。
- 保持模型架构、GT、坐标、单位、loss、stride、KNN32、2 cm 阈值、checkpoint 解释与每手 2048 点不变。
- 删除本轮 smoke 产物即可回滚运行状态；代码回滚入口为提交 `a7ad08b` 的父提交。不得因此删除 raw 数据、正式
  cache 或其他 run。
