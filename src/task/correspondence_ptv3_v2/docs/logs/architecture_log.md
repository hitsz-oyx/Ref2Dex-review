# correspondence_ptv3_v2 architecture log

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-08-23
- last_verified: 2026-08-23
- related: [当前状态](status_log.md) / [实验](experiment_log.md)

## 当前架构入口

详细 pipeline 保留在任务文档 [`../架构.md`](../架构.md)。本日志记录该架构最近的跨数据集约定：

1. Stage 3 输出提供 clean full object pool 和 clean hand geometry。
2. Dataset 可对 full pool 应用共享 object SE(3)，将整个 pool 交给 runner。
3. Runner 用 256 个 hand proxy 评分 full pool，取 near 384 + global 128 个 object points。

## OakInk 无 MANO runtime 路径

OakInk 样本没有 MANO 参数，所以 `apply_hand_perturb=false` 时不进行 MANO forward。如果仍要 runtime resampling，必须提供 `meta.stored_hand_proxy_indices_path`；该 JSON 版本化一组直接索引存储 hand points 的空间 FPS proxy。Runner 会校验索引数量、范围和唯一性，缺失时 fail-fast。

## 三域等比例混训路径

`configs/mixed_grab_contactpose_oakink_equal.yaml` 使用 `data.domain_paths` 为
GRAB、ContactPose、OakInk 分别构造 `CorrStaticDatasetV2`，再由
`DomainConcatDataset` 合并索引。`DomainBalancedSampler` 在每个本地 batch
中固定抽取三个域相同数量的样本，并在 DDP 下按 rank 切分每个域的采样流；它
不会按三个数据集的 frame 数量加权，也不会把较小域静默耗尽后改变比例，而是
按确定性 seed 循环采样。

该首轮 baseline 显式设置 `use_mano_reconstruction=false`、
`apply_hand_perturb=false` 和 `runtime_resample_object=false`。因此三个域共享
存储的 clean hand/object 几何和普通 512 点 object sampling；object pose
rotation/translation perturbation 仍可按统一配置开启。每个域拥有独立的
`val_clean/<domain>/` 与 `val_perturbed/<domain>/` loader，checkpoint 选择指标
不应只依赖混合平均值。

## MANO 扰动评估不变量

`research/contactpose_checkpoint_compare/evaluate.py` 的 hand-only / hand+object 条件必须显式固定 `hand_perturb_prob=1.0`，不得继承 checkpoint 的 H80/H50 训练门控比例。评估条件表示所有样本采用同一 corruption stream；训练 exposure 只能作为被比较的模型属性，不能改变评估输入分布。

历史 v2.0 checkpoint 的 config 可能没有 `meta.mano_model_dir`；评估器在 hand 条件下使用当前 Task `Config.meta.mano_model_dir` 作为只读兼容默认值，不修改 checkpoint，也不改变模型权重或协议扰动。

协议 E 使用 ARCTIC axis-angle45 的 10 mm RMS hand perturb；object 扰动保持 rotation std 10° / translation std 10 mm。绝对退化定义为同条件 `perturbed random QFL - clean random QFL`，并与 perturbed QFL、`pseudo_recovery_brier` 并列报告。
