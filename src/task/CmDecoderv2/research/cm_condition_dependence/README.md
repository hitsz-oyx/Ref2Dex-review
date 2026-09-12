# 冻结 decoder 的 Cm 条件依赖

检验同一个 Inspire 当前状态下，正确 Cm 是否比另一真实动作的完整 Cm 更有用；先单步，再 16 步手状态
递归。输入交换包含整个 K=4 window 的 tokens、anchor position/normal，保留时间顺序与对应关系。
这是同源条件依赖诊断，后续跨手交换实验以此为前置证据。

源参考与目标动作在本实验中来自同一 Inspire 序列；输入 source hand-flow 合法包含未来参考运动。
“GT 不输入 decoder”具体指没有额外传入未来 target q/wrist/手点字段，不能理解成不使用未来参考的在线预测。

执行范围见 [src/task/CmDecoderv2/docs/plan/v1.1.md](../../docs/plan/v1.1.md) 的 V1.1.8 增补。

```bash
CUDA_VISIBLE_DEVICES=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONPATH=. \
  /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoderv2.research.cm_condition_dependence.run \
  --run-id <unique_run_id> --activity-id <activity_id> --smoke
```

`--smoke` 使用最前两条有 active window 的 val sequence，仍运行完整对照和短递归，用于工程验证。
正式运行去掉 `--smoke`，使用全部 30 条 Inspire val sequence 的原 active-only 窗口。

- `correct`：原 Cm；`identity`：保持当前手状态；`zero_all`：所有 tokens/anchors 为零，是 OOD 辅助消融。
- `swap`：同一序列、起点至少相差 20 帧的随机完整有效窗口。
- `matched_swap`：在上述候选中，当前 wrist 平移差 <=30 mm、旋转差 <=30°、q RMS 差 <=0.25 rad、
  active fraction 差 <=0.1、flow RMS 比 [0.5,2]，且输入手流均值向量有至少规定幅度的变化。
  精确定义固定在 `diagnostics.MATCH_LIMITS`。无匹配时不放宽阈值，不用未来目标或预测误差筛选。
- 递归 donor 是连续 16 步片段，不逐步重抽；每个条件只在起点得到一次 GT target-hand state。
  各步物体状态仍来自参考，不能解释为物理闭环。
- `penalty_vs_correct_mm > 0` 表示该条件在相同接收样本上比正确 Cm 更差；
  `gain_over_identity_mm > 0` 表示优于保持当前手。每个对照按自身共同样本报告对应 correct/identity，
  不同覆盖率的绝对 EPE 不直接作差。CI 按 sequence 重采样 2000 次；donor 与 recipient 属于同一 cluster。
- zero_all 变差不单独证明 Cm 可迁移；swap 无变化也不能证明完全忽略 Cm，需要结合输入/输出变化和覆盖率。
  val 曾用于 checkpoint 选择，不能用本实验报告正式独立测试泛化。

运行目录保存 `config.json`、`metadata.json`、`run_manifest.json`、`run.log`、`cm_bank.npz`、`donors.npz`、
`metrics.jsonl`、`rollout_metrics.jsonl`、q/wrist 预测 NPZ、`dependence_summary.json` 与 `dependence.png`。
模型/输入摘要与 forward parity 自动核验，GT cache/full-native-FK correspondence 逐 batch 抽一行检查；
原生仿真从动关节可能不满足 decoder 的固定 mimic 关系，额外报告 6 维 GT FK 重建残差，不能把它称为
优化后的误差下界。默认单 GPU、
30 分钟、8 GiB GPU allocation 和 1 GiB 输出预算。运行终态与解释见
[src/task/CmDecoderv2/docs/logs/activity_log.md](../../docs/logs/activity_log.md)。

复核已有运行：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m src.task.CmDecoderv2.research.cm_condition_dependence.verify <run_directory>
```

复核会重建 donor、配对 summary，并用独立 NumPy FK 对分条件样本重算点误差，输出 `verification.json`。
生成 summary 的 `INCONCLUSIVE` 是保守默认值；正式分命题解释见 Task experiment log，不从一个消融数值
自动推断跨手成功。
