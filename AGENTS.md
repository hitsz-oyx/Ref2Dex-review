# Ref2Dex Agent 规范

---

## 0. 需求确认与歧义处理（最重要）

在开始修改代码、配置、数据处理逻辑或实验实现之前，必须先确认用户需求已经足够明确。

重点检查：

- 目标是否明确；
- 修改范围是否明确；
- 用户是否已经指定必须保持不变的内容；
- 当前需求是否与已有代码、指导文档或科研约束存在冲突；
- 是否缺少会显著影响最终实现或实验结论的信息。

如果存在歧义不得自行选择一种解释后直接实现，必须及时向用户指出具体歧义并说明。在这些关键歧义解决之前，不开始对应的实质性修改。如果用户的回答仍然存在新的关键歧义，应继续确认，直到当前任务已经达到可以明确执行的状态。

---

## 1. 默认语言

- 项目文档、研究日志、实验记录和 Git 提交信息默认使用中文。
- 代码标识符、第三方接口名、标准缩写、命令、配置字段和必须保持兼容的 schema 字段可以保留英文。
- 用户明确指定其他语言时，以用户要求为准。

---

## 2. AI 维护的 6 类递归文档

AI 在仓库中维护 6 类语义文档，全部存放在就近作用域的 `docs/logs/` 子目录中。除 memory 外使用**英文名 + `_log` 后缀**；memory 拆成仓库记忆和机器记忆两个递归载体：

| 文档 | 文件名 | 内容定位 |
| --- | --- | --- |
| **status**（当前状态） | `status_log.md` | 当前阶段、当前指导版本、最近可靠结论、进行中的任务、阻塞点和下一步；是可覆盖更新的状态快照。 |
| **memory**（接手记忆） | `repo_memory.md` / `machine_memory.md` | 前者记录可随仓库迁移的约定、相对路径、历史遗留和兼容性事实并纳入 Git；后者记录本机解释器、GPU、绝对路径、挂载点、代理和机器特有坑，并由 Git 忽略。 |
| **architecture**（完整架构事实源） | `architecture_log.md` | 当前有效的完整技术架构：Pipeline、数据流、字段、张量规格、坐标系、模型接口、loss、评估指标和可视化约定。 |
| **experiment**（实验文档） | `experiment_log.md` | 实验假设、改动、定量结果、决策、下一步。 |
| **decision**（AI 决策） | `decision_log.md` | 需求已明确情况下的非平凡自主选择。 |
| **modification**（AI 修改记录） | `modification_log.md` | 每次 AI 实际修改的代码、配置、文档（包括对其他 doc 自身的修改）。 |

6 类 doc 自身**不维护 changelog**。对 doc 自身的修改也统一写到 `modification_log.md` 中；两种 memory 的历史条目是长期事实记忆，不替代修改日志。`repo_notes_log.md` 已废弃，禁止重新创建；其中仍有效的事实必须按机器相关性拆入两种 memory。

### 2.1 路径与作用域

6 类 doc 均遵循"作用域原则"：Task 级按需维护，根级仅维护具有全仓库意义的内容；其中**根级 `experiment_log.md`、`status_log.md`、`repo_memory.md`、`machine_memory.md` 可按需创建，不要求每个仓库初始化时建立**。

```text
任务级: src/task/<Task>/docs/logs/<name>_log.md
        src/task/<Task>/docs/logs/{repo_memory,machine_memory}.md
根级:   docs/logs/<name>_log.md
        docs/logs/{repo_memory,machine_memory}.md
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

辅助文档（不属于 6 类）：

```text
src/task/<Task>/docs/
├── 约束.md            # 什么不能做 / 已知假设不能破坏
├── 指导/              # 当前研究方案，每个版本一个文件
│   ├── V1.md
│   └── V<n>.md
└── research/<实验名>/  # 诊断脚本
```

### 2.2 统一文档外壳与各 doc 模板

六类文档使用统一的元信息外壳，但正文结构按文档职责自由组织。模板是最小起点，不要求每个 Task 机械填满所有小节。

```markdown
# <文档名称>

- scope: root / task:<Task>
- last_updated: YYYY-MM-DD
- last_verified: YYYY-MM-DD  # 适用时填写
- related: <相关文档链接>

## 当前内容或索引
...
```

- **`status_log.md`**：只描述当前状态快照。每次 AI 修改代码、配置、数据处理逻辑、实验流程或文档后，都必须检查状态是否变化；变化时在原地更新并刷新时间戳。状态应链接到相关实验、指导和证据，而不是复制它们的详细内容。

  ```markdown
  ## 当前状态

  - 当前阶段 / 指导:
  - 当前进行中:
  - 最近可靠结论:
  - 阻塞 / 风险:
  - 下一步:
  - 证据与相关文档:
  ```
- **`repo_memory.md`**：记录后续接手者需要知道、但不能直接从代码结构中可靠推出，且在不同机器上仍成立的长期事实，包括仓库相对数据/cache 入口、命名约定、历史遗留、已知坑、兼容性和已验证/已废弃的路径。能用仓库相对路径表达的内容不得改写为某台机器的绝对路径。每条记录带日期，必要时带 `last_verified`、commit 或来源。它纳入版本管理，不是实验日志、决策日志或修改日志。

  ```markdown
  ## YYYY-MM-DD — <事实摘要>

  - category: path / convention / legacy / pitfall
  - status: active / stale / deprecated
  - last_verified: YYYY-MM-DD
  - fact:
  - source / anchor:
  ```
- **`machine_memory.md`**：记录只对当前机器成立的解释器绝对路径、依赖环境、CUDA/GPU、NAS/磁盘挂载点、仓库软链目标、代理设置和机器特有故障。该文件与 `repo_memory.md` 使用同一根级/Task 级递归作用域，但必须由 `.gitignore` 忽略，不得提交，也不得成为代码或配置的唯一事实源。没有机器特有事实时不创建空文件。

  ```markdown
  # <Task> 本机记忆

  - scope: root / task:<Task>
  - last_updated: YYYY-MM-DD
  - last_verified: YYYY-MM-DD
  - git: ignored

  ## 运行环境或本机路径

  - Python:
  - CUDA / GPU:
  - 本机数据 / cache 映射:
  - 代理 / 服务限制:
  ```
- **`architecture_log.md`**：当前架构的独立、完整事实源，不依赖其他架构总文档。章节顺序可由 Task 自定，但至少覆盖 Pipeline 和阶段输入输出、数据字段与可选性、关键张量的 shape/dtype/单位/坐标系/时间语义/mask、采样和坐标变换、模型与监督/loss、训练/验证/测试边界、评估指标与聚合方式、可视化入口与限制、必须保持的 scientific/data/runtime invariant。

  ```markdown
  ## 架构总览
  ## Pipeline 与数据流
  ## 数据 / 张量契约
  ## 模型、监督与 loss
  ## 训练与评估
  ## 可视化
  ## 不变量与 fail-fast 条件
  ```
- **`experiment_log.md`**：只记录会影响科研判断的正式实验；当前阶段和运行状态放在 `status_log.md`。使用上方最小实验模板。结论状态可用 `SUPPORTED`、`REFUTED`、`INCONCLUSIVE` 或 `INVALID_IMPLEMENTATION`；实现错误不得当作科研反证。重命名、性能修复、CLI 调整或单纯确认代码能够运行，不必新建 EXP。
- **`decision_log.md`**：只记录用户需求已明确、但实现中仍存在的非平凡自主选择；普通命名、格式化和局部代码组织不需要记录。使用上方单条模板。

  ```markdown
  ## YYYY-MM-DD — <决策摘要>

  - scope:
  - anchor: branch / commit / date

  **未指定点**
  ...

  **实际选择**
  ...

  **选择理由与影响**
  ...

  **可逆性 / 是否需要用户确认**
  ...
  ```

  若选择会改变研究目标、数据/GT 语义、核心评价标准或关键 invariant，不能只记录后继续，必须先向用户确认；其他非平凡工程选择可记录后自主推进。
- **`modification_log.md`**：每次 AI 实际修改代码、配置或文档后都要追加一条记录（包括对其他 doc 自身的修改）；对 `modification_log.md` 自身的追加不再递归产生新记录。

  ```markdown
  ## YYYY-MM-DD — <修改摘要>

  - branch:
  - post-commit:
  - scope: task 内部 / 跨 task / 全局

  **文件**
  - `path/to/file` — <做了什么>

  **改动原因**
  ...
  ```

> 信息归类边界：当前进行到哪一步放 `status_log.md`；跨机器成立的相对路径、约定和历史遗留放 `repo_memory.md`；仅本机成立的绝对路径、环境和挂载映射放 `machine_memory.md`；Pipeline、schema、张量和指标放 `architecture_log.md`；科研证据放 `experiment_log.md`；自主选择理由放 `decision_log.md`；实际改动放 `modification_log.md`。同一事实只保留一个详细来源，其他文档只建立链接。

#### 2.2.1 时间戳与更新方式

- `status_log.md` 的 `last_updated` 在状态变化时覆盖更新；不要求为每个微小代码编辑追加历史记录。
- 两种 memory 的事实条目至少带日期，环境或路径尽量带 `last_verified`；过时条目标记为 `stale` 或 `deprecated`。
- `architecture_log.md` 的 `last_verified` 在架构变化或重新核验后更新；历史变更由 `modification_log.md` 追踪。
- 日期统一使用 `YYYY-MM-DD`；需要区分运行先后时使用 `YYYY-MM-DD HH:MM TZ`。

### 2.3 诊断脚本与实验产物

诊断脚本继续放在 `src/task/<Task>/research/<实验名>/` 下，只放代码，不放大型产物。

```text
src/task/<Task>/research/
└── <实验名>/
    ├── diag_xxx.py
    └── ...
```

6 类语义文档中，`repo_memory.md` 和五类 `*_log.md` 以及 `research/<实验名>/` 下的诊断脚本需要纳入版本管理；`machine_memory.md` 是唯一例外，必须 ignore。

实验产物统一放到仓库根目录 `output/` 下：

```text
output/
├── exp/                # 训练实验输出
└── research/           # 分析诊断产物
```

`output/exp/` 与 `output/research/` 都被 gitignore 忽略；详细产物不提交，只有摘要级结论写入 `experiment_log.md`。

---

## 3. 递归项目文档

本仓库采用由总到分的递归文档结构。AI 维护的 6 类 doc 既是规范对象也是阅读入口；本节说明如何用它们按"由总到分"理解项目。

### 3.1 根级入口

仓库根目录下：

```text
docs/logs/architecture_log.md     # 仓库级完整架构事实源 / Pipeline / 数据流
docs/logs/status_log.md           # 仓库级当前状态快照
docs/logs/repo_memory.md          # 仓库级可迁移约定 / 相对路径 / 历史遗留（纳入 Git）
docs/logs/machine_memory.md       # 本机环境 / 绝对路径 / 挂载映射（Git 忽略，按需）
docs/logs/experiment_log.md       # 跨 task / 全局科研实验摘要（按需）
docs/logs/decision_log.md         # 跨 task / 全局 AI 自主决策
docs/logs/modification_log.md     # 跨 task / 全局 AI 修改记录
```

阅读顺序上，**`status_log.md` 是仓库级当前入口**，`architecture_log.md` 是仓库级架构事实入口。二者分别用于说明：

- `status_log.md`：当前研究阶段、进行中的工作、阻塞点和下一步；
- `architecture_log.md`：整体研究目标、主要目录、共享数据和流程、主要 Task、各子任务文档索引及全局已确认的技术事实。

仓库级 `repo_memory.md` 只记录跨机器成立的路径约定和历史遗留，不承担当前状态或架构总览；若本机 `machine_memory.md` 存在，接手时还应读取其中的环境与挂载映射，但不得将其视为仓库公共事实。

### 3.2 Task 级入口

每个任务在：

```text
src/task/<Task>/docs/logs/status_log.md           # 任务级当前状态快照
src/task/<Task>/docs/logs/repo_memory.md          # 任务级可迁移约定 / 相对路径 / 历史遗留
src/task/<Task>/docs/logs/machine_memory.md       # 任务级本机环境 / 绝对路径（Git 忽略，按需）
src/task/<Task>/docs/logs/architecture_log.md     # 任务级 Pipeline / 架构
src/task/<Task>/docs/logs/experiment_log.md       # 任务级实验文档
src/task/<Task>/docs/logs/decision_log.md         # 任务级 AI 自主决策
src/task/<Task>/docs/logs/modification_log.md     # 任务级 AI 修改记录
```

阅读顺序上，**任务级 `status_log.md` 是 Task 当前入口**，`repo_memory.md` 是 Task 公共接手记忆入口，`machine_memory.md` 是可选的本机接手记忆入口，`architecture_log.md` 是 Task 架构事实入口。它们分别覆盖：

- `status_log.md`：当前指导版本、当前阶段、最近可靠结论、正在运行或等待的工作、阻塞点和下一步；
- `repo_memory.md`：Task 特有、跨机器成立的相对数据/cache 入口、命名约定、历史遗留和已知坑；
- `machine_memory.md`：Task 在当前机器上的解释器、依赖环境、绝对路径、GPU 和服务限制；
- `architecture_log.md`：任务完整 Pipeline、数据字段、张量规格、模型/评估/可视化接口和不变量。

任务级 `status_log.md` **不记录**：

- 实验过程、定量结果、假设、对比、ablation → 属于 `experiment_log.md`；
- 跨实验或跨实现的设计选择理由 → 属于 `decision_log.md`；
- 每次具体改动了什么文件 / 哪一步做到哪 → 属于 `modification_log.md`；
- 可迁移的相对路径、历史遗留和已知坑 → 属于 `repo_memory.md`；本机环境和绝对路径 → 属于 `machine_memory.md`。

任务级 `architecture_log.md` 是该 task 自己的完整 Pipeline / 架构 / 数据流事实源，不依赖 `docs/架构.md`。若存在同名或旧版辅助文档，必须避免把架构事实只写在那里。

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

并由该 Task 的 `status_log.md` 建立到 `architecture_log.md`、`repo_memory.md`、`experiment_log.md`、`decision_log.md`、`modification_log.md` 和指导文档的索引。`machine_memory.md` 因被忽略而不要求建立可提交链接，读取规则由本规范统一规定。

---

## 4. 开始修改 Task 前的阅读顺序

AI 开始修改某个 Task 前，应按照以下顺序理解项目：

```text
docs/logs/status_log.md
        ↓
docs/logs/architecture_log.md
        ↓
docs/logs/repo_memory.md
        ↓
docs/logs/machine_memory.md（存在时）
        ↓
src/task/<Task>/docs/logs/status_log.md
        ↓
src/task/<Task>/docs/logs/architecture_log.md
        ↓
src/task/<Task>/docs/logs/repo_memory.md
        ↓
src/task/<Task>/docs/logs/machine_memory.md（存在时）
        ↓
按任务类型判断是否需要主动读取其他 log（见 §4.1）
        ↓
其中引用且与当前任务直接相关的文档
        ↓
相关代码
```

只读取当前研究问题所需的文档和代码。文档的作用是帮助快速定位执行路径，而不是替代实际代码阅读。

不要为了开始一个局部实验而遍历整个 `docs/` 或整个仓库。

### 4.1 6 类文档的按需阅读规则

除默认入口文档外，AI 必须根据当前任务**主动判断是否需要读取**其他 log，不得仅因为入口文档没有显式引用就跳过。

按以下规则判断：

- **`status_log.md`**
  - 默认读取。
  - 用于快速获得当前阶段、进行中的工作、阻塞点和下一步；它不是历史事实源。

- **`repo_memory.md`**
  - 默认读取。
  - 用于获取跨机器成立的相对路径、命名约定、历史遗留和已知坑。

- **`machine_memory.md`**
  - 文件存在时默认读取；不存在时直接跳过。
  - 用于获取当前机器的解释器、依赖环境、GPU、绝对路径、挂载映射、代理和机器特有坑。
  - 内容不得写入提交，也不得反向固化为仓库公共配置。

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
4. 不得为了形式完整而每次加载全部 6 类文档。

### 4.2 历史 log 的读取粒度

对于 append-only 的历史文档（`repo_memory.md` / `machine_memory.md` / `experiment_log.md` / `decision_log.md` / `modification_log.md`）：

- 默认先读取 `status_log.md` 顶部的当前状态，再读取历史日志顶部的索引或当前状态摘要；
- 再通过关键词、实验编号、日期、模块名定位相关历史；
- 只读取与当前问题相关的历史段落；
- 除非进行全局 review，不得默认完整加载大型历史 log。

### 4.3 缺失文档的自动创建

不存在的文档在"读取阶段"可以跳过，但**如果当前任务已经触发该类文档的维护条件，AI 应主动创建对应文档**，而不是因为文件不存在而永久跳过。

触发条件示例：

- 新 Task 第一次进行正式实验 → 创建 `experiment_log.md`；
- 第一次发生非平凡自主决策 → 创建 `decision_log.md`；
- AI 第一次实际修改该 Task → 创建 `modification_log.md`；
- Task 已形成稳定 Pipeline → 创建或补充完整的 `architecture_log.md`；
- 出现跨机器成立的相对路径、约定或历史遗留 → 创建或补充 `repo_memory.md`；
- 出现当前机器特有的解释器、GPU、绝对路径、挂载、代理或服务问题 → 创建或补充被忽略的 `machine_memory.md`；
- Task 或仓库出现需要接手的阶段、阻塞或下一步 → 创建或补充 `status_log.md`。

禁止为了形式完整，在没有实际内容时批量创建大量空 log。

---

## 5. 文档同步原则

代码、数据 schema、运行方式或已经确认的研究结论发生变化时，应同步检查最靠近该事实的 6 类 log doc（参见第 2 节）。判断同步到哪一级、哪一类 doc 的标准：

```text
仓库级 Pipeline / 架构 / 跨 task 共享约定
-> docs/logs/architecture_log.md

仓库级当前阶段 / 阻塞 / 下一步
-> docs/logs/status_log.md

仓库级可迁移相对路径 / 约定 / 历史遗留
-> docs/logs/repo_memory.md

仓库级本机环境 / 绝对路径 / 挂载 / 代理
-> docs/logs/machine_memory.md（Git 忽略）

Task 入口 / 状态 / 该 task 的记忆差异
-> src/task/<Task>/docs/logs/status_log.md（状态）
-> src/task/<Task>/docs/logs/repo_memory.md（可迁移记忆）
-> src/task/<Task>/docs/logs/machine_memory.md（本机记忆，Git 忽略）

Task 自身 Pipeline / 架构 / 数据流
-> src/task/<Task>/docs/logs/architecture_log.md

有研究意义的实验
-> src/task/<Task>/docs/logs/experiment_log.md
（必要时在 docs/logs/experiment_log.md 写一条仓库级摘要）

AI 自主决策
-> 对应作用域的 decision_log.md（使用 §2.2 模板）

本次修改记录
-> 对应作用域的 modification_log.md（使用 §2.2 模板）
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

6 类 log 是项目记忆和导航信息，**不是代码事实的最终替代品**。代码和可复现实验结果才是事实源；其中 `architecture_log.md` 是架构文档的规范事实入口，但仍不得凌驾于实际代码和可复现实验结果。

当文档描述与当前代码、配置、实际数据或可复现实验结果冲突时：

1. 不得为了让代码符合旧文档而直接修改代码；
2. 应先通过当前代码、配置、Git 历史或可复现实验确认真实状态；
3. 如果确认文档已经过时，应更新对应 log；
4. 如果无法判断哪个正确，应将其视为需要调查的问题；
5. 涉及科研语义时按第 0 节处理关键歧义。

原则：

> 文档用于快速建立上下文，代码和可复现实验用于验证事实。

---

## 6. 长时间运行任务与异步等待

对于训练、仿真、编译、数据处理、模型下载、视频渲染、benchmark 以及其他长时间运行任务，禁止通过高频 polling 反复唤醒模型。

这里的 polling 指：

> Agent 为了确认后台任务是否完成，反复调用工具查询状态，而不是让工具自身长时间等待。

核心原则：

> 让工具等待，不要让模型每隔几秒重新推理一次来确认任务是否结束。

这既减少无意义的模型调用，也避免长上下文在每次轮询中被重复处理造成大量 token 消耗。

### 6.1 空 `write_stdin` polling

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

### 6.2 `functions.wait`

对于长时间运行的异步任务：

```text
functions.wait
```

必须使用：

```text
yield_time_ms >= 180000
```

不需要中间状态时，优先让 wait 长时间阻塞并自然返回，而不是主动进行短间隔轮询。

### 6.3 嵌套工具等待

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

### 6.4 非空 `write_stdin`

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

### 6.5 长任务默认行为

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

## 7. Agent 与子任务使用原则

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

## 8. 本仓库 Git 规范

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
受版本管理的 6 类文档（status / repo memory / architecture / experiment / decision / modification；不含被忽略的 machine memory）
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
