# Ref2Dex 仓库常识

## 运行环境

- Python：`/home2/wyy/miniconda3/envs/graspenv/bin/python`（3.8.20）。
- PyTorch：2.4.1+cu121；主要 GPU 假设为 NVIDIA RTX 3090（约 24 GB）。
- GRAB：`dataset/GRAB`；ARCTIC：`data/raw_data/ARCTIC/arctic`，raw sequences 位于其下的 `data/arctic_data/data/raw_seqs`。
- 派生 cache：`data/processed_data/`；训练/诊断产物：`output/exp/`、`output/research/`、`outputs/`，均不提交。
- 测试：`/home2/wyy/miniconda3/bin/conda run -n graspenv python -m pytest ...`。

## 文档入口

- 全局结构：[`architecture_log.md`](architecture_log.md)
- 全局环境与路径：本文档
- 全局自主决策：[`decision_log.md`](decision_log.md)
- 全局修改记录：[`modification_log.md`](modification_log.md)
- Cm 任务入口：[`src/task/Cm/docs/logs/repo_notes_log.md`](../../src/task/Cm/docs/logs/repo_notes_log.md)

## 共享 GRAB 数据资产约定

- `process/GRAB/raw.py` 被 Cm、Stage2、GeneOH 和 InteractionDynamics 共用。GRAB 的 sequence root 与 `tools/subject_meshes` 可能分别位于 `dataset/GRAB/data/grab` 和 `dataset/GRAB/data/tools`，调用方不得假设所有相对 asset 都直接位于传入 root 下。
- 正式 Cm/V2 Stage4 通过 `require_subject_vtemplate=True` 强制使用 subject-specific MANO template；缺失 template 会显式报错，避免生成语义错误的 cache。旧流程如确需兼容可显式使用 `--allow-default-mano`。
