"""`vmp rag index|query|check`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp.cli import register
from vmp.config import Settings, load_config

DEFAULT_CONFIG = Path("configs") / "rag.toml"


def _index_from_args(args: argparse.Namespace):
    from vmp.rag.index import Index

    cfg = load_config(args.config) if Path(args.config).exists() else {}
    root = Path(args.root) if args.root else Settings.from_sources().rag_index_root
    index = Index.from_config(cfg, root)
    if Index.exists(root):
        index.load()
    return index


def _cmd_index(args: argparse.Namespace) -> int:
    index = _index_from_args(args)
    report = index.build(args.corpus, dry_run=args.dry_run)
    if not args.dry_run:
        report["saved_to"] = str(index.save())
    print(json.dumps(report, indent=1))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    index = _index_from_args(args)
    result = index.query(args.text, args.k)
    if args.json:
        print(json.dumps(result.to_dict(), indent=1))
    else:
        if not result.chunks:
            print("no relevant context (below relevance floor)")
        for c, s, p in zip(result.chunks, result.scores, result.paths, strict=True):
            cos = result.cosines.get(c.id, 0.0)
            print(f"{s:.4f}  cos={cos:.3f}  {p:8s}  {c.id}")
        print()
        print(result.context_pack(args.max_chars))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    index = _index_from_args(args)
    print(json.dumps(index.check(args.query, args.k), indent=1))
    return 0


def _add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="TOML config path")
    parser.add_argument("--root", default=None, help="index directory (default: settings)")
    sub = parser.add_subparsers(dest="rag_command", required=True)

    p = sub.add_parser("index", help="build or update the index from a directory")
    p.add_argument("corpus", help="directory to index")
    p.add_argument("--dry-run", action="store_true", help="report the plan only")
    p.set_defaults(_rag_run=_cmd_index)

    p = sub.add_parser("query", help="retrieve context for a question")
    p.add_argument("text")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--json", action="store_true")
    p.set_defaults(_rag_run=_cmd_query)

    p = sub.add_parser("check", help="report cosine spread rank 1 vs rank N")
    p.add_argument("--query", action="append", required=True, help="repeatable")
    p.add_argument("--k", type=int, default=20)
    p.set_defaults(_rag_run=_cmd_check)


def _run(args: argparse.Namespace) -> int:
    return int(args._rag_run(args))


def register_cli() -> None:
    register("rag", "chunk, embed, index and retrieve", _add_args, _run)


__all__ = ["register_cli"]
