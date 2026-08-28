#!/usr/bin/env python3
"""通用科研组件 registry 的最小命令行入口。

示例::

    python tools/researchctl.py list components
    python tools/researchctl.py check components/example_pipeline.yaml components
    python tools/researchctl.py graph components/example_pipeline.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.base.registry import ComponentRegistry, RegistryError
from src.base.pipeline import PipelineSpec


def _registry(roots: list[str]) -> ComponentRegistry:
    return ComponentRegistry.discover(roots)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="列出已发现组件")
    list_parser.add_argument("roots", nargs="+", help="组件目录或 component.yaml")
    list_parser.add_argument("--status")
    list_parser.add_argument("--capability")
    list_parser.add_argument("--tag")

    check_parser = sub.add_parser("check", help="校验 manifest 或 pipeline")
    check_parser.add_argument("path", help="component.yaml 或 pipeline.yaml")
    check_parser.add_argument("roots", nargs="+", help="组件目录或 component.yaml")
    check_parser.add_argument("--resolve-entrypoints", action="store_true", help="显式导入并检查 manifest 入口")

    describe_parser = sub.add_parser("describe", help="显示组件的输入输出合同")
    describe_parser.add_argument("component_id")
    describe_parser.add_argument("roots", nargs="+", help="组件目录或 component.yaml")

    graph_parser = sub.add_parser("graph", help="打印 pipeline 的组件连接")
    graph_parser.add_argument("pipeline")

    run_parser = sub.add_parser("run", help="准备运行 pipeline（当前只支持 dry-run）")
    run_parser.add_argument("pipeline")
    run_parser.add_argument("roots", nargs="+", help="组件目录或 component.yaml")
    run_parser.add_argument("--dry-run", action="store_true", help="只校验并打印拓扑顺序")

    args = parser.parse_args()
    try:
        if args.command == "list":
            registry = _registry(args.roots)
            for spec in registry.list(status=args.status, capability=args.capability, tag=args.tag):
                caps = ",".join(spec.capabilities) or "-"
                tags = ",".join(spec.tags) or "-"
                print(f"{spec.id}\t{spec.version}\t{spec.status}\t{caps}\t{tags}")
            return 0
        if args.command == "check":
            path = Path(args.path)
            if path.name == "component.yaml":
                from src.base.component import load_manifest
                spec = load_manifest(path)
                if args.resolve_entrypoints:
                    from src.base.component import resolve_entrypoint
                    resolve_entrypoint(spec.entrypoint)
                print(f"OK component {spec.id} ({spec.version})")
                return 0
            issues = _registry(args.roots).check_pipeline(path)
            if args.resolve_entrypoints:
                registry = _registry(args.roots)
                for node in PipelineSpec.from_yaml(path).nodes:
                    registry.resolve_entrypoint(node.component)
            if issues:
                for issue in issues:
                    print(f"ERROR {issue}", file=sys.stderr)
                return 1
            print(f"OK pipeline {path}")
            return 0
        if args.command == "describe":
            spec = _registry(args.roots).get(args.component_id)
            print(f"{spec.id} ({spec.version}) [{spec.status}]")
            print(f"entrypoint: {spec.entrypoint}")
            print(f"capabilities: {', '.join(spec.capabilities) or '-'}")
            print("inputs:")
            for name, port in spec.inputs.items():
                print(f"  {name}: type={port.type} shape={port.shape or '-'} dtype={port.dtype or '-'} unit={port.unit or '-'}")
            print("outputs:")
            for name, port in spec.outputs.items():
                print(f"  {name}: type={port.type} shape={port.shape or '-'} dtype={port.dtype or '-'} unit={port.unit or '-'}")
            return 0
        if args.command == "graph":
            import yaml
            raw = yaml.safe_load(Path(args.pipeline).read_text(encoding="utf-8")) or {}
            for node in raw.get("nodes", []):
                print(f"{node['id']} [{node['component']}]")
            for edge in raw.get("edges", []):
                print(f"  {edge['from']} -> {edge['to']}")
            return 0
        if args.command == "run":
            if not args.dry_run:
                raise RegistryError("Pipeline execution is not enabled in the prototype; use --dry-run.")
            pipeline = PipelineSpec.from_yaml(args.pipeline)
            registry = _registry(args.roots)
            issues = registry.check_pipeline(args.pipeline)
            if issues:
                for issue in issues:
                    print(f"ERROR {issue}", file=sys.stderr)
                return 1
            print(f"DRY-RUN {pipeline.id} ({pipeline.version})")
            print("order: " + " -> ".join(pipeline.topological_order()))
            return 0
    except (OSError, ValueError, RegistryError) as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
