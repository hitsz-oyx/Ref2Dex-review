# Ref2Dex Agent 规范

本仓库 AI Agent 协作规范。涉及科研实验工作流、研究优先级、可信度检查、范围纪律等通用研究工程规范，已拆分到：

```text
.agents/skills/research-experiment/SKILL.md
```

进行科研实验、模型训练、benchmark、evaluation 或连续实验推进前，应先调用该 skill。

本文件仅保留本仓库特定的目录、文档、Git 与协作规范。

---

## 0. 需求确认与歧义处理

在开始修改代码、配置、数据处理逻辑或实验实现之前，必须先确认用户需求已经足够明确。

### 0.1 先检查需求是否存在歧义

重点检查：

- 目标是否明确；
- 修改范围是否明确；
- 输入、输出和数据语义是否明确；
- 模型、数据集、训练和评测行为是否存在多种合理解释；
- 用户是否已经指定必须保持不变的内容；
- 当前需求是否与已有代码、指导文档或科研约束存在冲突；
- 是否缺少会显著影响最终实现或实验结论的信息。

如果存在会影响以下内容的歧义：

- 研究目标；
- 模型语义；
- 数据语义；
- GT 定义；
- 数据筛选；
- 训练 / 测试划分；
- 评价指标；
- 实验公平性；
- 核心接口行为；
- 用户最终可观察到的功能；

不得自行选择一种解释后直接实现。

必须及时向用户指出具体歧义，并说明：

1. 当前不明确的是什么；
2. 为什么这个问题会影响实现或实验结论；
3. 主要有哪些合理选项；
4. 如果有推荐方案，可以说明推荐理由，但不得把推荐方案当成用户已经确认的需求。

在这些关键歧义解决之前，不开始对应的实质性修改。

如果用户的回答仍然存在新的关键歧义，应继续确认，直到当前任务已经达到可以明确执行的状态。

### 0.1.1 需求歧义与研究未知必须区分

以下情况不属于需要用户先给出答案的需求歧义：

- 用户明确要求比较多个候选方案；
- 某个设计选择本身就是当前研究问题；
- 实验目的就是判断某个假设是否成立；
- 成功阈值需要通过 baseline 或实验结果确定。

例如：

> "研究 Cm 应该使用单手还是双手"

不是要求用户先选择单手或双手，
而是要求 Agent 将其转化为可验证的研究问题。

只有「用户到底要求研究单手 / 双手问题，还是已经决定必须使用双手」这类任务含义本身不清楚的情况，才需要进一步确认。

### 0.2 不要把普通实现自由度误判为需求歧义

以下通常不需要询问用户：

- 局部变量命名；
- 普通函数拆分；
- 等价的代码组织方式；
- 不改变外部行为的轻量重构；
- 明显可以从现有代码风格确定的实现细节；
- 对结果没有实质影响且容易回退的工程选择。

这些内容可以自行选择合理方案。

如果某个自主选择虽然不需要阻塞任务，但存在其他合理方案，
且可能影响后续维护、模型行为或实验结果，
按 `AI 自主决策记录` 章节的规则记录。

### 0.3 不得通过猜测消除关键歧义

不要因为以下原因跳过需求确认：

- 「现有代码大概是这个意思」；
- 「通常大家都会这么做」；
- 「这个方案最常见」；
- 「这样实现最方便」；
- 「用户可能就是这个意思」。

如果不同解释会导致实质不同的结果，就应当询问。

### 0.4 需求确认完成后再实施

正式修改前，应能够明确回答：

- 本次要解决什么问题；
- 哪些内容需要修改；
- 哪些内容必须保持不变；
- 成功标准是什么；
- 是否存在尚未解决的关键歧义。

只有最后一项为「否」时，才开始实质性修改。

### 0.5 提问期间不要提前实施

当正在等待用户澄清关键需求时：

- 可以阅读代码、追踪执行路径、收集事实；
- 可以分析不同方案的影响；
- 可以准备候选实现思路；

但不得提前修改会受该歧义影响的代码、配置、数据或实验定义。

用户确认后，再按照确认后的需求实施。

### 0.6 三级划分原则

```text
会显著影响需求 / 科研语义的歧义
→ 必须先问

不值得打断、但存在多个合理选择
→ AI 自己决定
→ 写入 AI 自主决策记录

普通无关紧要的工程细节
→ AI 自己决定
→ 不记录
```

核心原则：

> 任何会影响需求含义、科研结论、数据 / 模型语义或用户可观察行为的关键歧义，都必须在修改前确认；普通实现自由度不需要确认。

---

## 1. 默认语言

- 项目文档、研究日志、实验记录和 Git 提交信息默认使用中文。
- 代码标识符、第三方接口名、标准缩写、命令、配置字段和必须保持兼容的 schema 字段可以保留英文。
- 用户明确指定其他语言时，以用户要求为准。

---

# 本仓库规范

## 2. AI 维护的 5 类递归文档

AI 在仓库中维护 5 类文档，统一使用**英文名 + `_log` 后缀**，全部存放在就近作用域的 `docs/logs/` 子目录中：

| 文档 | 文件名 | 内容定位 |
| --- | --- | --- |
| **repo_notes**（仓库常识） | `repo_notes_log.md` | 接手项目可能踩坑的细枝末节：Python 解释器绝对路径、关键依赖版本、数据集/缓存根路径、特殊命名约定、隐式约束、当前文档主结构等。 |
| **architecture**（项目总览 / 总体考虑） | `architecture_log.md` | 仓库级总体考虑：Pipeline 详细描述、整体架构、共享数据流、跨 task 通用约定等。**要详细写，不只是大致看一下**，所以不叫「总览」。 |
| **experiment**（实验文档） | `experiment_log.md` | 实验假设、改动、定量结果、决策、下一步。 |
| **decision**（AI 决策） | `decision_log.md` | 需求已明确情况下的非平凡自主选择。 |
| **modification**（AI 修改记录） | `modification_log.md` | 每次 AI 实际修改的代码、配置、文档（包括对其他 doc 自身的修改）。 |

5 类 doc 自身**不维护 changelog**。对 doc 自身的修改也统一写到 `modification_log.md` 中。

### 2.1 路径与作用域

5 类 doc 均遵循"作用域原则"：Task 级按需维护，根级仅维护具有全仓库意义的内容；其中**根级 `experiment_log.md` 可按需创建，不要求每个仓库初始化时建立**。

```text
任务级: src/task/<Task>/docs/logs/<name>_log.md
根级:   docs/logs/<name>_log.md
```

- 根级只放"跨 task 影响 / 全仓库级"的事实。
- 任务级放"只在该 task 内部成立"的事实。
- 当某次改动只发生在单个 task 内部时，**只更新任务级**的对应 doc，根级 doc 不必同步。
- 当改动会**影响其他 task / 公共组件 / 共享 utils / 根 AGENTS.md / 外部依赖**时，**必须同时在根级对应 doc 写一条**。

判断标准：

> 这次修改是否会让 task 外部、其他 task 或仓库全局的使用者 / 后续 AI 读到 / 调用 / 依赖到？
> 答案为「是」就在根级对应 doc 也写一条。

`experiment_log.md` 根级版本仅用于记录具有跨 task 意义、影响仓库总体研究方向或会被多个 task 依赖的实验结论；不存在跨 task 意义的实验不应在根级写。

示例（task 内部改动）：

```text
只改 src/task/Cm/...
→ 更新 src/task/Cm/docs/logs/..._log.md
→ docs/logs/..._log.md 不必动
```

示例（跨 task / 全局改动）：

```text
改了 src/base/... 或 process/... 或外部数据集 / API / 根基础设施
→ 更新对应 task 级的 ..._log.md
→ 同时更新 docs/logs/..._log.md
```

辅助文档（不属于 5 类）：

```text
src/task/<Task>/docs/
├── 约束.md            # 什么不能做 / 已知假设不能破坏
├── 指导/              # 当前研究方案，每个版本一个文件
│   ├── V1.md
│   └── V<n>.md
└── research/<实验名>/  # 诊断脚本
```

### 2.2 各 doc 职责

- **`repo_notes_log.md`**：接手项目可能踩坑的细枝末节——Python 解释器绝对路径、关键依赖版本、数据集/缓存根路径、特殊命名约定、隐式约束、当前文档主结构等。AI 在执行任务过程中可逐步补充。
- **`architecture_log.md`**：仓库级总体考虑——Pipeline 详细描述、整体架构、共享数据流、跨 task 通用约定等。要详细写，不只是大致看一下。
- **`experiment_log.md`**：实验假设、改动、定量结果、决策、下一步。
- **`decision_log.md`**：需求已明确情况下的非平凡自主选择。
- **`modification_log.md`**：每次 AI 实际修改的代码、配置、文档（包括对其他 doc 自身的修改）。

> 注意：`repo_notes_log.md` 偏向"细枝末节"和"环境/路径/约定"等不直接体现研究语义的事实；
> `architecture_log.md` 偏向"Pipeline / 架构 / 跨 task 总体考虑"等结构性内容。
> 这两类不要重复维护。

### 2.3 experiment_log（实验文档）

`experiment_log.md` 是长期科研状态和实验事实记录，不是工作日报。

基本原则：

- 顶部维护简短的当前研究状态；
- 实验历史只追加；
- 只记录会影响研究判断的新证据；
- 实现错误不得记录成科学反证；
- 正式实验必须能够追溯到配置、命令、输出、commit hash、对应指导版本。

#### 2.3.1 单个实验的固定模板（EXP）

每次正式实验在 `experiment_log.md` 中按以下模板新增一个 `## EXP-NNN` 段落（NNN 为该 task 内的全局递增编号）：

```markdown
## EXP-017 — <简短名称>

### 日期

2026-08-17

### 对应指导

`docs/指导/V2.md`

### 假设

为什么这样做可能有效，期望看到什么。

不能只写「测试 dual hand」，而要写「为什么 dual hand 可能有效」。

### Baseline

- commit: `abc1234`
- config: `configs/cm_scene_v1.yaml`
- checkpoint: `outputs/.../best.pt`

### 本次修改

- active-hand input → dual-hand input
- 其余保持不变：
  - decoder
  - loss
  - dataset split
  - training budget

### 实现审查

Verdict: PASS / FAIL

关键检查：
- 左右手坐标系正确
- inactive hand mask 正确
- 无 future leakage
- evaluator 未修改

### 实验命令

```bash
python -m src.task.Cm.train \
  --config src/task/Cm/configs/cm_dual_hand.yaml
```

### 结果

| Metric                 | Baseline | Candidate |
| ---------------------- | -------: | --------: |
| one-step EPE           |  3.21 mm |   3.05 mm |
| rollout EPE            | 12.70 mm |  11.83 mm |
| bimanual subset EPE    |  8.41 mm |   6.92 mm |
| single-hand subset EPE |  3.11 mm |   3.09 mm |

### 关键观察

提升主要集中在 bimanual subset。
single-hand subset 基本没有变化。

### 解释

结果与假设一致：另一只手提供的运动信息可能减少了双手接触场景中的状态歧义。

但当前 candidate 参数量比 baseline 多约 18%，
因此还不能把提升完全归因于 dual-hand information。

### 结论状态

**INCONCLUSIVE**

当前证据支持继续研究 dual-hand，
但不足以确认「提升来自双手信息本身」。

### 决策

不将当前模型作为最终 best。
保留实验结果和代码分支。

### 下一步

做 matched-capacity 对比：

- active-hand larger model
- dual-hand model

控制参数量后重新比较。

### 证据

- checkpoint: `outputs/...`
- metrics: `outputs/.../metrics.json`
- log: `outputs/.../train.log`
- commit: `def5678`
```

#### 2.3.2 必填字段

每个 EXP 必须包含：

- 假设
- Baseline
- 本次修改
- 实验命令
- 结果
- 关键观察
- 解释
- 结论状态
- 决策
- 下一步
- 证据位置

尤其容易缺三个：

- `假设`：必须写「为什么这样做可能有效」，不能只写做了什么。
- `结论状态`：必须明确标 `SUPPORTED / REFUTED / INCONCLUSIVE / INVALID_IMPLEMENTATION` 之一。
- `下一步`：否则下次开启 context 不知道为什么做 V3。

#### 2.3.3 结论状态

```text
SUPPORTED
当前有效证据支持假设

REFUTED
当前实验有效且满足假设预定义的反证条件

INCONCLUSIVE
实验有效，但不足以支持或反驳假设

INVALID_IMPLEMENTATION
实验实现无效，不产生结论状态
```

`REFUTED` 严格定义：

> 只有实验有效且确实满足假设预定义的反证条件时使用；单纯「指标没涨」通常优先判为 `INCONCLUSIVE`。

#### 2.3.4 实现失败 ≠ 科研失败

如果发现实现 bug 导致结果异常，必须标 `INVALID_IMPLEMENTATION`，并解释原因：

```markdown
### 结论状态

INVALID_IMPLEMENTATION

### 原因

左手 points 仍然使用 right-hand root frame，
导致双手输入语义错误。

本次结果不得用于判断 dual-hand hypothesis。
```

这能避免以后误读历史。

#### 2.3.5 何时创建新 EXP

判断标准：

> 这个操作是否产生了一条可以影响科研判断的新证据？

应该记录（✔）：

- 输入 / 表示 / 训练方式的关键改变；
- 容量、组件、损失的有效消融；
- intervention / counterfactual；
- baseline reproduction；
- 3-seed 完整验证；
- zero-flow 等 sanity intervention。

不应该记录（✘）：

- viewer 调颜色；
- 函数重命名；
- 修 typo；
- 改 CLI；
- 修 dataloader 性能；
- viewer bug 修复；
- 单纯确认代码路径能够运行。

如果只是实现修复不回答研究问题，放到最近实验的「实现备注」下，或者直接留在 Git commit：

```markdown
### 实现备注

- 修复 left-hand mask；
- 修复 cache indexing；
- 不改变实验定义。
```

### 2.4 诊断脚本与实验产物

诊断脚本继续放在 `src/task/<Task>/research/<实验名>/` 下，只放代码，不放大型产物。

```text
src/task/<Task>/research/
└── <实验名>/
    ├── diag_xxx.py
    └── ...
```

5 类 log doc 与 `research/<实验名>/` 下的诊断脚本需要纳入版本管理，不 ignore。

实验产物统一放到仓库根目录 `output/` 下：

```text
output/
├── exp/                # 训练实验输出
└── research/           # 分析诊断产物
```

`output/exp/` 与 `output/research/` 都被 gitignore 忽略；详细产物不提交，只有摘要级结论写入 `experiment_log.md`。

---

## 3. 递归项目文档

本仓库采用由总到分的递归文档结构。AI 维护的 5 类 doc 既是规范对象也是阅读入口；本节说明如何用它们按"由总到分"理解项目。

### 3.1 根级入口

仓库根目录下：

```text
docs/logs/architecture_log.md     # 仓库级总体考虑 / Pipeline / 架构
docs/logs/repo_notes_log.md       # 仓库级细枝末节 / 环境 / 路径 / 约定
docs/logs/decision_log.md         # 跨 task / 全局 AI 自主决策
docs/logs/modification_log.md     # 跨 task / 全局 AI 修改记录
```

阅读顺序上，**`architecture_log.md` 是仓库级总入口**，用于说明：

- 整体研究目标；
- 主要目录；
- 共享数据和流程；
- 主要 Task；
- 各子任务文档索引；
- 全局已经确认的事实。

仓库级 `repo_notes_log.md` 偏向"细枝末节"，承接 3.3 的运行环境小节，与 `architecture_log.md` 不重复。

### 3.2 Task 级入口

每个任务在：

```text
src/task/<Task>/docs/logs/repo_notes_log.md       # 任务级仓库常识 / 状态
src/task/<Task>/docs/logs/architecture_log.md     # 任务级 Pipeline / 架构
src/task/<Task>/docs/logs/experiment_log.md       # 任务级实验文档
src/task/<Task>/docs/logs/decision_log.md         # 任务级 AI 自主决策
src/task/<Task>/docs/logs/modification_log.md     # 任务级 AI 修改记录
```

阅读顺序上，**任务级 `repo_notes_log.md` 是 Task 入口**，至少应覆盖：

- 任务目标（**简短一句话**，不替代指导文档中的研究方案）；
- 该 task 内部的数据 / 模型 / 主要代码入口 / 运行方式 **索引**（具体细节指向 `architecture_log.md` 或专题文档）；
- 该 task 内部特有的环境差异（解释器、依赖、cache 路径、命名约定等）；
- 当前文档主结构索引（指向 `architecture_log.md` / `experiment_log.md` / `指导/` 等）；
- **当前研究近况**（1~3 句简述）：当前正在做哪个指导版本 / 哪个研究方向、上一阶段结论的大致落点、最近的修改 / 实验是否处于关键节点。

任务级 `repo_notes_log.md` **不记录**：

- 实验过程、定量结果、假设、对比、ablation → 属于 `experiment_log.md`；
- 已确认的研究结论、下一步研究方向 → 属于 `experiment_log.md`；
- 跨实验或跨实现的设计选择理由 → 属于 `decision_log.md`；
- 每次具体改动了什么文件 / 哪一步做到哪 → 属于 `modification_log.md`；
- "近况"**不要写成详细状态报告**，它只是入口的简况指针，让读者一眼知道方向；具体进度看 `modification_log.md`，具体结论看 `experiment_log.md`。

任务级 `architecture_log.md` 描述该 task 自己的 Pipeline / 架构 / 数据流。

复杂 Task 可以继续在自己的：

```text
src/task/<Task>/docs/
```

目录中拆分：

```text
约束.md
指导/                 # 当前研究方案，每个版本一个文件
其他专题文档
```

并由该 Task 的：

```text
src/task/<Task>/docs/logs/repo_notes_log.md
```

建立索引。

### 3.3 运行环境

为了后续 AI 可以直接运行仓库代码，**根级 `docs/logs/repo_notes_log.md` 必须包含 `运行环境` 小节**（不放在 `architecture_log.md` 中），至少给出：

- 推荐的 Python 解释器绝对路径（如 `~/miniconda3/envs/<env>/bin/python`）；
- 关键依赖（PyTorch / CUDA / 其他库）的版本要求；
- 主要数据集 / 缓存的根路径；
- GPU 等硬件假设；
- 必须在该环境下运行的命令或脚本入口（及其所需的额外参数，如 `--grab-root`）。

任务级 `repo_notes_log.md` 也应记录该 task 自身对环境的具体要求（例如某个 cache 只在某些路径下生成），与根级保持一致；如有差异，任务级优先级最高。

环境（解释器路径、依赖版本、数据集路径、硬件）发生变化时，应同步更新对应 `repo_notes_log.md`。

---

## 4. 开始修改 Task 前的阅读顺序

AI 开始修改某个 Task 前，应按照以下顺序理解项目：

```text
docs/logs/architecture_log.md
        ↓
docs/logs/repo_notes_log.md
        ↓
src/task/<Task>/docs/logs/repo_notes_log.md
        ↓
src/task/<Task>/docs/logs/architecture_log.md
        ↓
按任务类型判断是否需要主动读取其他 log（见 §4.1）
        ↓
其中引用且与当前任务直接相关的文档
        ↓
相关代码
```

只读取当前研究问题所需的文档和代码。文档的作用是帮助快速定位执行路径，而不是替代实际代码阅读。

不要为了开始一个局部实验而遍历整个 `docs/` 或整个仓库。

### 4.1 5 类文档的按需阅读规则

除默认入口文档外，AI 必须根据当前任务**主动判断是否需要读取**其他 log，不得仅因为入口文档没有显式引用就跳过。

按以下规则判断：

- **`repo_notes_log.md`**
  - 默认读取。
  - 用于获取环境、路径、依赖、命名约定、隐式约束和当前状态。

- **`architecture_log.md`**
  - 涉及代码结构、Pipeline、数据流、模型结构、模块接口、跨模块行为时必须读取。
  - 修改现有架构前必须读取对应作用域的 architecture log。

- **`experiment_log.md`**
  - 涉及训练、实验、benchmark、evaluation、ablation、数据集调整、指标变化、已有科研结论或继续历史实验时必须读取。
  - 在提出新的实验方案前，应先检查是否已有相同或相关实验，避免重复实验。

- **`decision_log.md`**
  - 涉及已有设计选择、替换已有实现、重新讨论某项设计，或当前实现看起来存在多个合理方案时必须读取。
  - 目的是确认过去是否已经做过相关自主决策，以及当时的理由。

- **`modification_log.md`**
  - 涉及修改已有实现、排查 regression、理解某段代码为什么存在、恢复历史行为、review 最近修改、或当前代码行为与文档不一致时必须读取。
  - 不要求普通局部开发每次完整读取整个 modification log。

读取顺序与作用域：

1. 优先读取当前 Task 级文档；
2. 涉及共享组件、跨 Task 行为或仓库级约定时，再读取根级对应文档；
3. 不相关的历史 log 不需要读取；
4. 不得为了形式完整而每次加载全部 5 类文档。

### 4.2 历史 log 的读取粒度

对于 append-only 的历史 log（`experiment_log.md` / `decision_log.md` / `modification_log.md`）：

- 默认先读取文件顶部的"当前状态 / 索引"；
- 再通过关键词、实验编号、日期、模块名定位相关历史；
- 只读取与当前问题相关的历史段落；
- 除非进行全局 review，不得默认完整加载大型历史 log。

### 4.3 缺失文档的自动创建

不存在的文档在"读取阶段"可以跳过，但**如果当前任务已经触发该类文档的维护条件，AI 应主动创建对应文档**，而不是因为文件不存在而永久跳过。

触发条件示例：

- 新 Task 第一次进行正式实验 → 创建 `experiment_log.md`；
- 第一次发生非平凡自主决策 → 创建 `decision_log.md`；
- AI 第一次实际修改该 Task → 创建 `modification_log.md`；
- Task 已形成稳定 Pipeline → 创建或补充 `architecture_log.md`；
- 出现值得后续复用的环境、路径、约定或状态信息 → 创建或补充 `repo_notes_log.md`。

禁止为了形式完整，在没有实际内容时批量创建大量空 log。

---

## 5. 文档同步原则

代码、数据 schema、运行方式或已经确认的研究结论发生变化时，应同步修改最靠近该事实的 5 类 log doc（参见第 2 节）。判断同步到哪一级、哪一类 doc 的标准：

```text
仓库级 Pipeline / 架构 / 跨 task 共享约定
-> docs/logs/architecture_log.md

仓库级细枝末节 / 环境 / 路径 / 约定
-> docs/logs/repo_notes_log.md

Task 入口 / 状态 / 该 task 的环境差异
-> src/task/<Task>/docs/logs/repo_notes_log.md

Task 自身 Pipeline / 架构 / 数据流
-> src/task/<Task>/docs/logs/architecture_log.md

有研究意义的实验
-> src/task/<Task>/docs/logs/experiment_log.md
（必要时在 docs/logs/experiment_log.md 写一条仓库级摘要）

AI 自主决策
-> 对应作用域的 decision_log.md（参见第 6 节）

本次修改记录
-> 对应作用域的 modification_log.md（参见第 7 节）
```

不要在多个文件中复制维护同一份详细信息。

文档应帮助后续 AI 自上而下理解项目：

```text
项目
-> Task
-> 专题
-> 代码
```

不要写成长篇流水账。

可以直接通过代码快速获得的实现细节，不需要大段复制到文档。

### 5.1 文档与代码不一致时

5 类 log 是项目记忆和导航信息，**不是代码事实的最终替代品**。代码和可复现实验结果才是事实源。

当文档描述与当前代码、配置、实际数据或可复现实验结果冲突时：

1. 不得为了让代码符合旧文档而直接修改代码；
2. 应先通过当前代码、配置、Git 历史或可复现实验确认真实状态；
3. 如果确认文档已经过时，应更新对应 log；
4. 如果无法判断哪个正确，应将其视为需要调查的问题；
5. 涉及科研语义时按第 0 节处理关键歧义。

原则：

> 文档用于快速建立上下文，代码和可复现实验用于验证事实。

---

## 6. AI 自主决策记录

维护对应作用域的 `decision_log.md`。

本节适用于：

> 用户需求本身已经足够明确，但实现过程中仍存在用户未指定的、
> 不改变已确认需求含义的非平凡设计或工程选择。

例如：

- 两种实现都满足已经确认的需求；
- 某个局部模型组件存在多个合理实现；
- 某个工程方案存在多个可替换选项；
- 当前选择可能影响后续扩展、性能或实验结果，但不会改变当前研究问题本身。

此类情况无需频繁打断用户。
AI 可以选择最保守、最易验证、最易回退的方案继续，但必须记录该自主决策。

每条记录至少包括：

- **锚定**：`commit` 哈希 / `branch` 名称 / 日期；
- 哪一点没有被用户明确指定；
- 本次实际选择；
- 其他合理选择；
- 为什么做出当前选择；
- 对结果可能产生的影响；
- 可逆性；
- 是否建议用户后续确认。

固定模板：

```markdown
## <YYYY-MM-DD> — <一句话摘要>

- branch: <branch>
- post-commit: <hash 或 "HEAD~N">

**未指定点**
...

**实际选择**
...

**其他合理选择**
...

**选择理由**
...

**对结果的影响**
...

**可逆性**
完全可逆 / 局部可逆 / 不可逆

**建议用户确认**
否 / 是（建议时机：...）
```

普通代码组织、命名、格式化和无科研意义的实现细节不记录。

如果决策会改变研究目标、数据/GT 语义、核心评价标准、关键 scientific invariant 或造成不可逆影响，不得仅记录后自行继续，必须按第 0 节先请求用户决策。

如果不确定性来自「用户需求本身存在多种实质不同的解释」，则不属于自主决策范围，必须按第 0 节先向用户确认。

### 就近记录原则

决策日志跟决策对象走，避免不同 task 的自主决策混在一个文件里失去可读性：

```text
Task 内的自主决策
-> src/task/<Task>/docs/logs/decision_log.md

跨 Task / 全仓库级自主决策
-> docs/logs/decision_log.md
```

---

## 7. AI 修改记录

每次 AI 实际修改仓库后，必须在 `modification_log.md` 留下一条记录，便于后续 AI 锚定「这版代码长什么样、当时为什么这么改」。

`modification_log.md` 与 `experiment_log.md`、`decision_log.md` 平级，遵循同样的就近记录原则（见第 2 节）：

```text
Task 内的修改
-> src/task/<Task>/docs/logs/modification_log.md

跨 Task / 全仓库级修改
-> docs/logs/modification_log.md
```

每条记录至少包括：

- **锚定**：`commit` 哈希 / `branch` 名称 / 日期；
- **文件路径**：被改动的文件绝对路径或仓库内路径；
- **功能**：被改动代码或配置做了什么（一句话即可）；
- **改动原因 / 出于什么考虑**：用户需求、研究假设、bug 修复、依赖更新、performance 优化等；
- **对应指导**（如有）：`src/task/<Task>/docs/指导/V<n>.md`；
- **影响范围**：仅 task 内部 / 跨 task / 全局。

可选字段（**有就写，没有可以省**）：

- **做到什么地步了（单次修改进度）**：如果本次修改本身是多步动作（例如"重构 1/3 完，正在改 2/3"），写明当前位于哪一步、剩几步、下一步是什么。用于多步改动在被中断 / 跨 session 后能被准确接续。
- **做到什么地步了（项目阶段进度）**：如果本次修改属于某个指导版本（V<n>）或研究阶段的若干步之一，写明该阶段整体进度（例如"V2 阶段 3/5 个修改完成"）。
- **下一步打算做什么**：如果本次修改的发起者已经确定接下来的动作（即使还没执行），写在这里；否则不需要为"可能的下一步"硬凑内容。这条是"已完成的下一步"指示，不是"明天计划"。

固定模板：

```markdown
## <YYYY-MM-DD> — <一句话摘要>

- branch: <branch>
- post-commit: <hash 或 "HEAD~N">
- 范围: task 内部 / 跨 task / 全局

**文件**

- `path/to/file_a.py` — <该文件做什么>
- `path/to/file_b.yaml` — <该配置做什么>

**改动原因**

<出于什么考虑 / 对应哪条用户需求 / 哪条研究假设>

**对应指导**（如有）

`docs/指导/V<n>.md`

**单次修改进度**（可选）

- 当前: <这一步做完 / 正在做 X / 阻塞在 Y>
- 剩余: <还剩 N 步，简述>
- 续接点: <下次从哪里接>

**项目阶段进度**（可选）

- 阶段: V<n> / 整体 <name>
- 进度: <n>/<total> 个修改完成
- 本次对应阶段中的第几步: <k>

**下一步打算做什么**（可选）

<下一个待执行动作，已经确定的；没有就省略>
```

不要求记录每个 diff 行，但要让未来读到这条记录的 AI 能立刻知道：

> 改了什么文件、为什么改、当时是哪次提交、对应哪版指导、影响范围到哪里；如果这次改动只是某个长动作的中间一步，也要能从记录里知道当前在哪、剩什么、下一步做什么。

如果本次修改是因为 `experiment_log.md` 或 `decision_log.md` 中的某条结论/决策，应在该条记录中显式引用（按 commit hash 或日期 + 段落标题）。

### 7.1 自递归豁免

`modification_log.md` 对自身的追加不再产生新的 modification 记录。即一次任务产生的一条 modification 记录，可以同时覆盖：

- 本次代码 / 配置修改；
- 本次其他 log 文档同步修改；
- 为记录本次任务而对 `modification_log.md` 自身进行的追加。

不得因为追加 modification log 本身而递归创建新的 modification 记录。

---

## 8. 长时间运行任务与异步等待

对于训练、仿真、编译、数据处理、模型下载、视频渲染、benchmark 以及其他长时间运行任务，禁止通过高频 polling 反复唤醒模型。

这里的 polling 指：

> Agent 为了确认后台任务是否完成，反复调用工具查询状态，而不是让工具自身长时间等待。

核心原则：

> 让工具等待，不要让模型每隔几秒重新推理一次来确认任务是否结束。

这既减少无意义的模型调用，也避免长上下文在每次轮询中被重复处理造成大量 token 消耗。

### 8.1 空 `write_stdin` polling

当 `write_stdin` 不发送任何实际内容，只用于查询后台进程状态时：

- `yield_time_ms` MUST >= `180000`；
- 不需要中间输出时优先使用 `300000`；
- 禁止每隔数秒或几十秒执行一次空 `write_stdin`。

推荐：

```text
启动任务
-> 等待 180~300 s
-> 获得输出或状态变化
-> 再决定下一步
```

避免：

```text
启动任务
-> 10 s
-> poll
-> 10 s
-> poll
-> 10 s
-> poll
-> ...
```

### 8.2 `functions.wait`

对于长时间运行的异步任务：

```text
functions.wait
```

必须使用：

```text
yield_time_ms >= 180000
```

不需要中间状态时，优先让 wait 长时间阻塞并自然返回，而不是主动进行短间隔轮询。

### 8.3 嵌套工具等待

如果外层 `functions.exec` 中存在内部长等待工具：

```text
functions.exec
└── nested tool wait
```

则外层 `@exec yield_time_ms` 必须至少比最长的内部等待多：

```text
30000 ms
```

即：

```text
outer_exec_yield_time
>=
longest_nested_wait + 30000
```

例如：

```text
nested wait: 180000 ms
outer exec:  >= 210000 ms
```

避免出现：

```text
外层 exec 先 yield
-> Agent 被重新唤醒
-> 内层工具实际上仍然在等待
-> Agent 再次进入等待
-> 产生无意义模型调用
```

### 8.4 非空 `write_stdin`

上述长等待规则只适用于纯状态 polling。

如果 `write_stdin` 正在发送实际输入，例如：

```text
"y\n"
"\n"
Ctrl-C
交互式命令输入
```

应立即发送，不要人为等待 180~300 秒。

因此需要区分：

```text
write_stdin("")
```

表示：

```text
查询状态
-> 长等待
```

而：

```text
write_stdin("实际输入")
```

表示：

```text
与进程交互
-> 立即发送
```

### 8.5 长任务默认行为

对于预期运行超过几十秒的命令：

- 尽量一次启动后长时间等待；
- 不频繁检查 stdout；
- 不因为暂时没有输出而主动高频 poll；
- 不为了展示「任务仍在运行」而反复调用模型；
- 只有中间输出会实际影响下一步研究决策时，才提前检查；
- 如果任务预计需要数分钟，应按照分钟级而不是秒级进行状态检查。

尤其适用于：

```text
训练
仿真
Isaac Sim
MuJoCo
CUDA 编译
数据预处理
模型下载
benchmark
视频渲染
大规模测试
```

---

## 9. Agent 与子任务使用原则

不要为了形式上的 Agent 化而拆分大量子任务。

简单问题优先由当前 Agent 直接完成。

只有在以下情况才考虑并行或独立子任务：

- 多个调查方向彼此独立；
- 多个模块可以并行阅读；
- 大型仓库需要明确按模块拆分；
- 独立实验可以并行运行；
- 最终需要单独 review 一个已经完成的实现。

避免：

```text
收到任务
-> planner
-> researcher
-> coder
-> tester
-> reviewer
-> 每个 Agent 再继续创建子 Agent
```

如果主 Agent 自己可以在较少步骤内完成任务，不要创建额外子 Agent。

子任务数量应以提高研究效率为目标，而不是最大化并行度。

---

## 10. 本仓库 Git 规范

用户要求提交时：

1. 提交前检查：

```bash
git status
git diff
```

2. 只纳入当前任务相关文件。
3. 不修改、覆盖、reset 或 revert 用户已有的无关改动。
4. **如果用户自己也改了一些仓库内容，`git status` / `git diff` 中会同时出现 AI 的改动和用户的改动。AI 必须先向用户确认是否一并提交用户那部分改动，不得自行决定把用户的改动一起带进同一个 commit。**
5. 默认不提交：

```text
数据缓存
训练输出
临时文件
checkpoint
大模型权重
自动生成的大型结果文件
```

除非用户明确要求。

6. **指导文档默认可以参与提交**，包括但不限于：

```text
AGENTS.md
.agents/skills/**/SKILL.md
src/task/<Task>/docs/约束.md
src/task/<Task>/docs/指导/V*.md
5 类 log doc（repo_notes / architecture / experiment / decision / modification）
```

不需要为了一次提交把它们单独排除。AI 修改这些文件后，按 10.4 确认范围，然后照常 `git add` + `git commit`。

7. Git commit message 默认使用中文。
8. Commit message 应简洁描述实际研究或实现结果，而不是泛泛描述操作。

例如优先：

```text
修正接触点时间对齐并验证 InteractionDynamics loss
```

而不是：

```text
修改代码
```
