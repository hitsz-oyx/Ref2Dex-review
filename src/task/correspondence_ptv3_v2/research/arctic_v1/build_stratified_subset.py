from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


NAME_RE = re.compile(
    r"^(?P<object>.+)_(?P<action>grab|use)_(?P<trial>.+)_(?P<side>left|right)\.npz$"
)


@dataclass(frozen=True)
class ArcticFile:
    subject: str
    object: str
    action: str
    trial: str
    side: str
    sequence: str
    path: str


def parse_file(path: Path, root: Path) -> ArcticFile | None:
    match = NAME_RE.match(path.name)
    if match is None:
        return None
    subject = path.relative_to(root).parts[0]
    obj = match.group("object")
    action = match.group("action")
    trial = match.group("trial")
    side = match.group("side")
    sequence = f"{subject}_{obj}_{action}_{trial}"
    return ArcticFile(
        subject=subject,
        object=obj,
        action=action,
        trial=trial,
        side=side,
        sequence=sequence,
        path=str(path),
    )


def choose_sequences(files: list[ArcticFile]) -> list[ArcticFile]:
    by_subject_object: dict[tuple[str, str], list[ArcticFile]] = defaultdict(list)
    for item in files:
        by_subject_object[(item.subject, item.object)].append(item)

    selected_sequences: set[str] = set()
    missing_actions: list[dict[str, str]] = []
    for key, items in sorted(by_subject_object.items()):
        subject, obj = key
        actions = {item.action for item in items}
        for action in ("grab", "use"):
            candidates = sorted(
                {item.sequence for item in items if item.action == action}
            )
            if candidates:
                selected_sequences.add(candidates[0])
            else:
                fallback = sorted({item.sequence for item in items})[0]
                selected_sequences.add(fallback)
                missing_actions.append(
                    {
                        "subject": subject,
                        "object": obj,
                        "missing_action": action,
                        "fallback_sequence": fallback,
                    }
                )

    return sorted(
        [item for item in files if item.sequence in selected_sequences],
        key=lambda item: (item.subject, item.object, item.action, item.trial, item.side),
    ), missing_actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic ARCTIC V1 stratified symlink subset."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", default="/tmp/arctic_eval_stratified_v1")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest).resolve()

    parsed: list[ArcticFile] = []
    skipped: list[str] = []
    for path in sorted(source_root.rglob("*.npz")):
        item = parse_file(path, source_root)
        if item is None:
            skipped.append(str(path))
            continue
        parsed.append(item)

    selected, missing_actions = choose_sequences(parsed)

    if output_root.exists() or output_root.is_symlink():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for item in selected:
        src = Path(item.path)
        rel = src.relative_to(source_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)

    objects = sorted({item.object for item in selected})
    subjects = sorted({item.subject for item in selected})
    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "selection_rule": (
            "For each (subject, object), choose the lexicographically first grab "
            "paired sequence and the lexicographically first use paired sequence; "
            "if an action is missing, fall back to the lexicographically first "
            "available sequence. Keep all existing left/right files for selected "
            "paired sequences."
        ),
        "source_npz_count": len(parsed),
        "selected_npz_count": len(selected),
        "subjects": subjects,
        "objects": objects,
        "missing_actions": missing_actions,
        "skipped_files": skipped,
        "selected_files": [asdict(item) for item in selected],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
