# ObjectInteractionCm AI 修改记录

## 2026-09-03 — 新增 ObjectInteractionCm best 的 t-SNE 诊断

- change_level: L1（Task 内离线诊断脚本与研究产物）
- approval: auto（用户明确指定 checkpoint 并要求绘图）
- branch: working tree
- post-commit: 未提交；训练进程未停止
- scope: task 内部 / latent visualization

**文件 / 产物**

- `src/task/ObjectInteractionCm/research_tsne.py` — 使用 ObjectInteractionCm runner、test index 和双手数据契约，生成 pooled dataset/stride/hand-RMS/object-RMS 图及自然 all-stride hand-RMS 图。
- `output/research/objectinteractioncm_tsne_best_20260903/` — 正式图、`tsne.npz` 和 `metadata.json`。

**验证**

- graspenv Python 编译通过；smoke 和正式运行均成功；正式样本 1040 条，GRAB 统计 stride 1..10，Inspire-F1 统计偶数 stride 2..20。
