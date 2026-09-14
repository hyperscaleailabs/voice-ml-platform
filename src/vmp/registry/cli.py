"""`vmp registry list|show|promote`. Root from `--root` or `VMP_REGISTRY_ROOT`.

`--version` uses `dest="artifact_version"` because the top-level `vmp --version`
flag already owns `args.version`.
"""

from __future__ import annotations

import argparse
import json
import sys

from vmp.cli import register
from vmp.config import Settings
from vmp.registry.store import FileRegistry, RegistryError
from vmp.types import STAGES


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=None, help="registry directory (default: settings)")
    sub = parser.add_subparsers(dest="registry_command", required=True)
    p = sub.add_parser("list", help="list versions, newest first")
    p.add_argument("--name", default=None)
    p = sub.add_parser("show", help="print one artifact as JSON")
    p.add_argument("--name", required=True)
    p.add_argument("--version", dest="artifact_version", required=True)
    p = sub.add_parser("promote", help="move a version to a stage")
    p.add_argument("--name", required=True)
    p.add_argument("--version", dest="artifact_version", required=True)
    p.add_argument("--stage", required=True, choices=STAGES)


def _registry(args: argparse.Namespace) -> FileRegistry:
    root = args.root or Settings.from_sources().registry_root
    return FileRegistry(root)


def _run(args: argparse.Namespace) -> int:
    reg = _registry(args)
    try:
        if args.registry_command == "list":
            for a in reg.list(args.name):
                print(f"{a.name}\t{a.version}\t{a.stage}\t{a.base_model}\t{a.config_hash[:12]}")
            return 0
        if args.registry_command == "show":
            print(
                json.dumps(
                    reg.get(args.name, args.artifact_version).to_dict(), indent=2, sort_keys=True
                )
            )
            return 0
        promoted = reg.promote(args.name, args.artifact_version, args.stage)
        print(f"{promoted.name}:{promoted.version} -> {promoted.stage}")
        return 0
    except (RegistryError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def register_cli() -> None:
    register("registry", "model registry: list, show, promote", _add_args, _run)


__all__ = ["register_cli"]
