# Audit contract

The bundled audit helper is deliberately narrow. It can verify that the latest modification
record names the files in a selected Git diff and contains a level, approval, scope, reason,
and validation section. It can also report missing or untracked paths when asked.

It cannot determine whether an AI chose the correct level, whether a scientific conclusion is
valid, or whether a user actually intended an approval. Those remain human-review decisions.

Use staged mode for handoff so unrelated unstaged work is ignored. Use worktree mode only when
the entire working-tree diff is intentionally in scope.
