# =============================================================================
# correspondence_ptv3 任务配置模块
# =============================================================================
# 本文件定义了 correspondence_ptv3 任务所使用的所有配置项，继承自
# `src.base.TaskConfig`。该任务在数据格式、训练入口等方面与
# `correspondence_v1` 保持一致，主要差异在于把编码主干网络替换成了
# 基于 PointTransformerV3 (PTv3) 序列化注意力思想的实现。
#
# 配置分为以下几个内部类（嵌套命名空间）：
#   - meta     : 模型与训练相关的超参数（点云规模、PTv3 主干结构、损失权重等）
#   - model    : 模型类路径与类型名
#   - data     : 数据加载相关的配置（路径、批量大小、子进程数等）
#   - train    : 训练流程相关的配置（优化器、学习率、保存策略、恢复点等）
#   - wandb    : Weights & Biases 实验追踪相关配置
#
# 通过将这些配置集中在一个文件内，可以让实验者无需修改代码即可调整
# 几乎所有可调参数，并通过命令行 `--set key=value` 的方式覆盖。
# =============================================================================

# 启用 Python 3.7+ 的 PEP 563 风格的延迟类型注解求值，提升导入性能并
# 允许在类型注解中引用尚未定义的类型（例如本文件中的嵌套类）。
from __future__ import annotations

# 用于把 vendored third_party 路径写成仓库内相对固定的位置。
from pathlib import Path

# 导入基础任务配置类 `TaskConfig`。
# 我们的 `Config` 类直接继承自它，复用其通用字段并在此基础上扩展本任务
# 特有的配置项。
from src.base import TaskConfig


ROOT = Path(__file__).resolve().parents[3]


class Config(TaskConfig):
    """correspondence_ptv3 任务的顶层配置类。

    继承自 `TaskConfig`，添加了本任务专有的字段：
      - `name`           : 任务名（与包名保持一致）
      - `runner_class`   : 训练/评估运行器的全限定类路径，
                           外部框架会根据该字符串动态导入对应 Runner。
    """

    # 任务名称，与目录名一致，供日志/检查点/W&B 等模块识别任务身份。
    name = "correspondence_ptv3"
    # Runner 类的全限定路径，框架在创建任务实例时会动态 import 该类。
    runner_class = "src.task.correspondence_ptv3.runner.CorrespondencePTV3Runner"

    class meta(TaskConfig.meta):
        """模型结构、数据预处理、损失权重等“元参数”配置。"""

        # ---- 基础几何规模配置 -----------------------------------------------
        # 物体点云中保留的点数（由数据预处理阶段统一采样得到）。
        num_obj_points: int = 512
        # Stage 3 文件中保存的完整物体候选池点数；训练时再采样到 512。
        num_obj_pool: int = 4096
        # 静态手部点云中保留的点数（合并 6 根手指/手掌表面采样）。
        num_hand_points: int = 1538
        # Stage 3 中保存的 clean GT 邻居数（历史字段，当前数据为 32）。
        k_cross: int = 32
        # runtime context 邻域大小：只用于 cross-attention / token 形成。
        k_ctx: int = 32
        # runtime logit 邻域大小上限。当前默认采用“2cm 内正样本 + 按正样本数
        # 配平的 >4cm 负样本”，总长度不足时 padding。
        k_logit: int = 64
        # 兼容旧配置字段。balanced logit 采样默认不再使用固定的半难负样本配额。
        k_logit_hard_neg: int = 16
        # runtime context 半径，单位 meter。
        ctx_radius: float = 0.04
        # runtime logit 正样本统计半径，单位 meter。
        logit_pos_radius: float = 0.02
        # runtime 远负样本的最小半径，单位 meter。
        logit_neg_min_radius: float = 0.04
        # 兼容旧配置字段。balanced logit 采样默认不再使用这个 6cm 上界。
        logit_neg_radius: float = 0.06

        # ---- 主干网络与输入特征 ---------------------------------------------
        # 当前共享点特征维度：
        # xyz(3) + point_type one-hot(2) + normal(3)
        # + nearest-opposite distance(1)
        # + nearest-direction·normal(1)
        # + centroid-direction·normal(1)
        point_feat_dim: int = 11
        # 是否启用 object->hand cross-attention。
        # 当前基线默认关闭，只保留为可选增强开关。
        use_cross_attn: bool = False
        # 指向 vendored PointTransformerV3 仓库的本地路径。
        # 注意：本任务并不真的使用 PTv3 仓库的全部依赖（spconv/scatter/flash），
        # 但仍通过 importlib 动态加载其中的 `model.PointTransformerV3`。
        ptv3_repo_path: str = str(ROOT / "third_party" / "PointTransformerV3")
        # PTv3 体素化时的栅格粒度，单位与点云坐标一致。
        ptv3_grid_size: float = 0.003
        # PTv3 序列化所用的多种排序方式：z-order、z-trans、hilbert、hilbert-trans。
        # 多种排序联合使用，可以缓解序列化顺序对注意力学习带来的偏差。
        ptv3_order: tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
        # 编码器各阶段下采样步长，对应 4 次下采样 (2^4=16x)。
        ptv3_stride: tuple[int, ...] = (2, 2, 2, 2)
        # 编码器 5 个阶段的 Transformer Block 深度。
        ptv3_enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2)
        # 编码器 5 个阶段的特征通道数。
        ptv3_enc_channels: tuple[int, ...] = (96, 192, 384, 384, 384)
        # 编码器 5 个阶段的注意力头数。
        ptv3_enc_num_head: tuple[int, ...] = (6, 12, 24, 24, 24)
        # 编码器 5 个阶段序列化时每个 patch 内的最大点数（用于控制注意力复杂度）。
        ptv3_enc_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024, 1024)
        # 解码器 4 个上采样阶段的 Block 深度。
        ptv3_dec_depths: tuple[int, ...] = (2, 2, 2, 2)
        # 解码器 4 个阶段的特征通道数。
        ptv3_dec_channels: tuple[int, ...] = (96, 192, 384, 384)
        # 解码器 4 个阶段的注意力头数。
        ptv3_dec_num_head: tuple[int, ...] = (6, 12, 24, 24)
        # 解码器序列化时每个 patch 内的最大点数。
        ptv3_dec_patch_size: tuple[int, ...] = (1024, 1024, 1024, 1024)
        # Transformer FFN 中间层相对于输入维度的扩展倍率。
        ptv3_mlp_ratio: float = 4.0
        # QKV 线性投影是否使用 bias。
        ptv3_qkv_bias: bool = True
        # 注意力权重的 dropout 概率。
        ptv3_attn_drop: float = 0.0
        # 注意力输出线性层的 dropout 概率。
        ptv3_proj_drop: float = 0.0
        # Stochastic Depth（DropPath）的最大丢弃率。
        ptv3_drop_path: float = 0.1
        # 是否使用 Pre-Norm 结构（True = Pre-Norm，False = Post-Norm）。
        ptv3_pre_norm: bool = True
        # 是否在不同序列化排序间随机打乱，增强模型对顺序的鲁棒性。
        ptv3_shuffle_orders: bool = True
        # 是否启用 Relative Position Encoding（相对位置编码）。
        # 在使用序列化 patch 注意力时默认关闭，靠序列化顺序隐式表达位置。
        ptv3_enable_rpe: bool = False
        # 是否尝试启用 Flash Attention（如果运行环境支持）。
        ptv3_enable_flash: bool = True
        # 是否在 attention 计算时上转（upcast）到更高精度。
        ptv3_upcast_attention: bool = False
        # 是否在 softmax 计算时上转精度，避免数值下溢。
        ptv3_upcast_softmax: bool = False

        # ---- 接触标签（contact label）的距离-软标签参数 ---------------------
        # 距离 ≤ d_pos 时，soft label 接近 1（强接触）。
        d_pos: float = 0.005
        # 距离 ≥ d_neg 时，soft label 接近 0（无接触）。
        d_neg: float = 0.03
        # 软标签过渡带的锐度（gamma 越大过渡越陡，越接近阶跃函数）。
        gamma: float = 1.0
        # 物体点是否参与对应关系（correspondence）学习的最小接触概率阈值。
        # 只有 `obj_contact_label > corr_contact_label_min` 的点会被视作
        # “正样本”参与对应关系头（如 object cano / finger / region）训练。
        corr_contact_label_min: float = 0.1
        # 训练时对手部进行随机旋转扰动的标准差（度），模拟手部姿态噪声。
        hand_rot_std_deg: float = 10.0
        # 训练时对手部进行随机平移扰动的标准差（米），与旋转配合使用。
        hand_trans_std: float = 0.01
        # 训练时对手部施加扰动的概率（1.0 = 每个样本都做扰动）。
        hand_perturb_prob: float = 1.0
        # 是否额外构建一套“固定手部扰动”的验证集。
        val_augment: bool = True
        # 固定扰动验证集的手部扰动概率。
        val_hand_perturb_prob: float = 1.0

        # ---- bin-contact 输出与可选 heads -----------------------------------
        # ContactOpt 风格的 contact probability 离散 bin 数。
        num_contact_bins: int = 10
        # 如何从 bin logits 解码回 [0, 1] 标量概率。
        # 可选: "expectation" / "argmax"
        contact_bin_decode_mode: str = "expectation"
        # 10-bin contact CE 的类别权重。这里直接写入从 ContactOpt 权重
        # 导出的均值归一化版本，保留相对比例，同时让平均权重为 1。
        contact_bin_weights: list[float] | None = [
            0.04906267471446666,
            0.124615854284691,
            0.5289112177088706,
            1.0479041499564616,
            1.4865461685891466,
            1.6614463943665032,
            1.722506187101053,
            1.5057927629354453,
            1.0416008624789648,
            0.8316137278643991,
        ]
        # 可选的外部权重路径，默认关闭；当前 baseline 直接使用上面的内嵌权重。
        contact_bin_weight_path: str | None = None
        # 手指类别数：通常 6（5 指 + 1 手掌），与数据集中 finger_id 取值一致。
        num_fingers: int = 6
        # 手掌区域数：通常 6（指尖 / 指节 / 手掌分区），由数据集 region_id 决定。
        num_regions: int = 6
        # 是否启用 finger / region 分类头。
        # 关闭时模型不输出这两个预测，对应损失项权重也置零。
        use_finger_region_head: bool = False
        # 是否启用 object canonical 对应点预测头。
        # 当前阶段默认关闭，用 contact / cross-edge supervision 优先塑造 token。
        use_cano_head: bool = False

        # ---- 损失项权重 -----------------------------------------------------
        # 物体点 10-bin contact 分类损失权重。
        loss_obj_contact_weight: float = 1.0
        # 兼容旧配置名；runner 中会优先读取新字段。
        loss_contact_weight: float = 1.0
        # 物体到手的规范化（canonical）位置回归的损失权重。
        loss_cano_weight: float = 0.0
        # 手指分类的交叉熵损失权重（仅在启用分类头时生效）。
        loss_finger_weight: float = 0.0
        # 手掌区域分类的交叉熵损失权重（仅在启用分类头时生效）。
        loss_region_weight: float = 0.0
        # 交叉边（cross edge）接触预测的损失权重。
        loss_cross_edge_weight: float = 1.0
        # 兼容旧配置字段。balanced logit 采样默认按样本数配平，通常不再依赖
        # 额外的 far-negative reweight。
        loss_cross_edge_far_weight: float = 0.5
        # cross-edge rank-k auxiliary loss 的损失权重。
        # 当前只作为可选实验项，基线默认不反传。
        loss_cross_edge_rankk_weight: float = 0.0
        # rank-k auxiliary loss 中，每个 object 点保留多少个 GT top-k anchor。
        rankk_loss_k: int = 4
        # 只有当 label_pos - label_neg 至少达到该 gap 时，才构造 ranking pair。
        rankk_label_gap: float = 0.15
        # pairwise margin ranking 的 margin。
        rankk_margin: float = 0.2
        # AUPRC 使用的二值 GT 阈值；与 rank-k loss 解耦。
        pr_label_threshold: float = 0.5

        # ---- 全局几何增强（训练时）开关与范围 -------------------------------
        # 是否在训练时对整帧点云施加随机旋转增强。
        augment_rotation: bool = True
        # 是否在训练时对整帧点云施加随机平移增强。
        augment_translation: bool = False
        # 是否在训练时对整帧点云施加随机缩放增强。
        augment_scale: bool = False
        # 随机旋转的最大角度（度），180.0 表示任意方向的随机旋转。
        rotation_range: float = 180.0
        # 随机平移的最大范围（米），各坐标独立在 [-range, range] 内采样。
        translation_range: float = 0.1
        # 随机缩放的取值区间 (min, max)，例如 (0.9, 1.1) 表示 ±10%。
        scale_range: tuple[float, float] = (0.9, 1.1)

    class model(TaskConfig.model):
        """模型实例化配置：用于框架根据 `class_path` 动态构建模型。"""

        # 模型类的全限定路径，框架会 import 该路径并实例化。
        class_path = "src.task.correspondence_ptv3.model.StaticHOCPTv3"
        # 模型类型字符串（用于日志/检查点记录），便于区分不同变体。
        type = "static_hoc_ptv3"

    class data(TaskConfig.data):
        """数据集与数据加载器相关配置。"""

        # 训练集路径：Stage 3 `train_corr_static` 目录或单个 .npz 文件。
        train_path: str = ""
        # 验证集路径：留空时会按照 `val_split` 自动从 train_path 切分。
        val_path: str = ""
        # 黑名单文件路径（.json 或行分隔的 .txt），用于排除有问题的样本。
        blacklist_path: str | None = None
        # 从训练集中切分出验证集的比例（仅在 val_path 为空时生效）。
        val_split: float = 0.1
        # 留空切分时，是否按“原始序列分组”而不是按单个 npz 文件随机切分。
        group_val_by_sequence: bool = True
        # 每个进程/每张卡上的训练批量大小。
        batch_size: int = 8
        # 每个进程/每张卡上的验证批量大小（一般可略大于训练以加速评估）。
        val_batch_size: int = 16
        # 每个进程的 DataLoader 子进程数；0 表示在主进程内加载（便于调试）。
        num_workers: int = 0
        # 训练时是否打乱样本顺序。
        shuffle: bool = True
        # 按 sequence block 打乱，避免随机帧顺序反复解包大型 Stage 3 npz。
        # 关闭后恢复 PyTorch 的全局 RandomSampler。
        sequence_locality_shuffle: bool = True
        # 训练时是否丢弃最后不足一个 batch 的样本，保证 batch 维度规整。
        drop_last: bool = True
        # 是否使用 pinned memory（GPU 训练时建议开启，加速 host→device 拷贝）。
        pin_memory: bool = True
        # DataLoader 子进程在 epoch 间是否保持常驻（需 num_workers > 0 才有效）。
        persistent_workers: bool = False
        # 每个 worker 预取的 batch 数（num_workers > 0 时生效）。
        prefetch_factor: int = 2

    class train(TaskConfig.train):
        """训练流程相关配置（优化器、学习率、保存策略等）。"""

        # 训练输出目录：检查点、配置、日志等都保存在这里。
        output_dir: str = "outputs/train/correspondence_ptv3"
        # 随机种子，保证可复现性。
        seed: int = 42
        # 训练设备：auto/cpu/cuda/cuda:0 等。
        device: str = "auto"
        # 最大训练轮次（epochs）。
        epochs: int = 100
        # 最大训练步数（steps），与 epochs 互为兜底；为 None 时只看 epochs。
        max_steps: int | None = None
        # 优化器类型，目前支持 "adamw" / "adam" / "sgd" 等。
        optimizer: str = "adamw"
        # 初始学习率。
        lr: float = 1e-4
        # AdamW 权重衰减系数。
        weight_decay: float = 1e-4
        # 学习率调度器类型，例如 "cosine" / "step" / "constant"。
        scheduler: str = "cosine"
        # 学习率 warmup 占总训练步数的比例；设定后优先于 warmup_steps。
        warmup_ratio: float | None = None
        # 学习率 warmup 的绝对步数；仅在 warmup_ratio 为 None 时生效。
        warmup_steps: int = 1000
        # 梯度裁剪的 L2 范数上限，None 表示不裁剪。
        grad_clip_norm: float | None = 1.0
        # 是否启用自动混合精度训练 (AMP)。
        amp: bool = False
        # 是否对模型调用 torch.compile（PyTorch 2.0+ 特性）。
        compile: bool = False
        # 每隔多少 step 记录一次训练日志。
        log_every_steps: int = 50
        # 按 step 间隔评估的频率，None 表示仅按 epoch 评估。
        eval_every_steps: int | None = None
        # 按 epoch 间隔评估的频率（与 eval_every_steps 同时存在时取更严格者）。
        eval_every_epochs: int = 1
        # 按 step 间隔保存检查点的频率。
        save_every_steps: int | None = None
        # 按 epoch 间隔保存检查点的频率。
        save_every_epochs: int = 10
        # 最多保留的最近检查点数量（多余的会被自动删除以节省磁盘）。
        max_to_keep: int = 5
        # 恢复训练的检查点路径，None 表示从头训练。
        resume: str | None = None
        # 早停的耐心值（连续多少轮不提升就停止），None 表示不早停。
        early_stopping_patience: int | None = None
        # 早停判定阈值（提升幅度小于该值则视为未提升）。
        early_stopping_threshold: float = 0.0
        # 用于挑选"最佳模型"的指标名（与 metrics 字典中的 key 对应）。
        metric_for_best: str = "val_clean/loss"
        # `metric_for_best` 是否越低越好（例如 val_clean/loss）。
        lower_is_better: bool = True

        class distributed(TaskConfig.train.distributed):
            """分布式训练/评估配置。"""

            # 是否启用分布式逻辑。实际多卡需要配合 torchrun 启动。
            enable: bool = False
            # 通信后端。auto=CUDA 时优先 nccl，否则 gloo。
            backend: str = "auto"
            # 初始化进程组的超时时间（分钟）。
            timeout_minutes: int = 30
            # 是否在 DDP 中广播 buffers。
            broadcast_buffers: bool = False
            # 仅当模型存在条件分支、部分参数并非每步都参与反传时才需要打开。
            find_unused_parameters: bool = False

    class wandb(TaskConfig.wandb):
        """Weights & Biases 实验追踪配置。"""

        # 是否启用 wandb 记录。
        enable: bool = True
        # wandb 项目名。
        project: str = "ref2dex"
        # wandb 团队/用户名（私有组织需要填写）。
        entity: str | None = None
        # 实验分组名（同组实验可在 UI 中合并对比）。
        group: str | None = None
        # 单次实验名。
        name: str = "correspondence_ptv3"
        # 实验标签，便于在 wandb 中过滤/检索。
        tags: list = ["correspondence", "ptv3", "static"]
        # wandb 模式：offline（本地缓存）/ online（实时上传）/ disabled（关闭）。
        mode: str = "offline"
        # 实验任务类型，便于区分 train / eval / sweep 等。
        job_type: str = "train"
        # 是否将模型检查点上传到 wandb Artifact。
        log_model: bool = False
