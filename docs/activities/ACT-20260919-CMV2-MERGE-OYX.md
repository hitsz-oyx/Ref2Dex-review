# ACT-20260919-CMV2-MERGE-OYX — 合并 oyx 稳定集成分支

timestamp: 2026-09-19T12:55:00+00:00
activity_id: ACT-20260919-CMV2-MERGE-OYX
work_version: V1.2h
base_commit: 06bab8b
branch: cmv2
scope: repository governance and stable-branch integration
change_level: L3
approval: user-approved
approval_basis: 用户明确要求 fetch 远程 oyx 仓库并尝试 merge；随后要求继续。

## Goal

将远端 `origin/oyx` 的稳定集成变更纳入当前 `cmv2` 分支，同时保留本地 Cmv2 V1.10/V1.11 工作及其运行产物边界。

## Changes

- fetch `origin/oyx` 并以 `--no-commit --no-ff` 合并提交 `c1b8d73`。
- 唯一文本冲突为 `docs/current_versions.yaml`：采用远端 `work_version` 字段合同，并保留本地较新的 `ObjectInteractionCm: V1.4.23` 与 `ObjectInteractionCmv2: V1.11.1` 指针。
- 纳入远端治理迁移、CmResidual/DExplore 及相关稳定集成变更；未修改本地 V1.11 铰接训练实现或生成数据、cache、checkpoint。

## Protected semantics

- Cmv2 当前数据语义、GRAB/ARCTIC stride 采样、训练配置和运行输出均未因 merge 改写。
- 不重启、不停止、不提交训练与 OakInk2 导出产生的 outputs、cache、checkpoint。

## Verification

- `git diff --name-only --diff-filter=U`：0 个未解决冲突。
- `git diff --cached --check -- . ':(exclude)third_party/DExplore'`：通过。
- `VERIFY_BASE=06bab8b conda run -n graspenv python tools/verify.py --changed`：失败；远端新增 CmResidual Activity/experiment 卡链接到本机不存在的 `data/` 与 `outputs/` 产物。该失败未改写远端证据记录。
- 全量 `git diff --check` 对远端新 vendor `third_party/DExplore` 报既有 CRLF/trailing-whitespace，不对 vendor 资产作格式化重写。

## Rollback

反转本次 merge commit 可恢复合并前的 `cmv2` 代码历史；运行产物未受该 commit 影响。
