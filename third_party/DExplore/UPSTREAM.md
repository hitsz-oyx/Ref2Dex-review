# DExplore vendor provenance

此目录是 Ref2Dex 的 squashed vendor snapshot；它不是 Git submodule，未携带嵌套 `.git`。

- canonical upstream: `https://github.com/NVlabs/dexplore`
- source fork used for the pinned snapshot: `https://github.com/hitsz-oyx/dexplore.git`
- pinned commit: `c31f57f186409ce5f0de47ced2d347abffe45d06`
- source commit subject: `Add GRAB Inspire retargeting and trajectory export tools`
- snapshot import: 2026-09-19; `git archive <pinned commit>` of tracked files only
- `dexplore/run.py` SHA256: `3fa7f8b516a9bdddc1a606607ff922358c287fcbf9ec04119006ae8f409df612`
- license: [LICENSE](LICENSE) (`NVIDIA License`); [NOTICE](NOTICE) and nested asset notices remain in the snapshot.
- excluded tracked LFS artifacts: `checkpoint/inspire.pth` and `checkpoint/inspire_distill.pth` (95 MiB and
  71 MiB after local LFS smudge).  They remain in the external source mirror and must be supplied by an explicit
  checkpoint path for evaluation; no checkpoint is vendored into Ref2Dex.

## Ignored runtime meshes

Upstream deliberately ignores `dexplore/data/assets/mjcf/objects/`; it is not part of this vendor's tracked-source
snapshot.  V1.20c materializes only the required airplane/table meshes byte-for-byte from the fixed raw GRAB paths
at execution time, after SHA256 verification.  They remain Git-ignored, are never symlinks, and are recorded in each
run manifest; see [V1.20c](../../src/task/CmResidual/docs/plan/V1.20c.md).

## Ref2Dex integration boundary

V1.20a uses this immutable source through a Task-local `torch.distributed` facade under
`src/task/CmResidual/tools/`; it must not patch DExplore or `rl_games` in place.  Runtime outputs
belong in repository-root `outputs/Dexplore/<run_id>/`, never in this vendor tree; `/outputs/` is
ignored as a defensive guard.

To update this snapshot, first inspect the upstream/fork diff and license, create a separately
approved vendor migration plan, re-import tracked files from an explicit commit, update this record,
and rerun the affected CmResidual smoke.  Do not run `git pull` inside this directory.
