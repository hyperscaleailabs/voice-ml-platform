"""`vmp data ...` subcommands."""

from __future__ import annotations

import argparse
import json

from vmp.cli import register


def _add_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="data_cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic golden text set as JSONL")
    s.add_argument("--out", required=True, help="output JSONL path")
    s.add_argument("--per-category", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--categories", default="", help="comma-separated subset")
    s.add_argument("--dry-run", action="store_true")

    q = sub.add_parser("pairs", help="build DPO preference pairs from a JSONL file")
    q.add_argument("--in", dest="inp", required=True, help="input JSONL")
    q.add_argument("--out", required=True, help="output JSONL")
    q.add_argument("--min-margin", type=float, default=0.0)
    q.add_argument("--all-pairs", action="store_true")
    q.add_argument("--dry-run", action="store_true")


def _run_synth(args: argparse.Namespace) -> int:
    from vmp.data.io import write_utterances
    from vmp.data.synthetic import CATEGORIES, generate_golden_set

    cats = [c for c in args.categories.split(",") if c] or list(CATEGORIES)
    utts = generate_golden_set(args.per_category, args.seed, cats)
    manifest = {
        "out": args.out,
        "seed": args.seed,
        "categories": cats,
        "n": len(utts),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        write_utterances(args.out, utts)
    print(json.dumps(manifest))
    return 0


def _run_pairs(args: argparse.Namespace) -> int:
    from vmp.data.corpus import PreferencePairBuilder
    from vmp.data.io import write_jsonl

    builder = PreferencePairBuilder(min_margin=args.min_margin, all_pairs=args.all_pairs)
    pairs = builder.build_from_jsonl(args.inp)
    by_source: dict[str, int] = {}
    for p in pairs:
        by_source[p.source] = by_source.get(p.source, 0) + 1
    manifest = {
        "in": args.inp,
        "out": args.out,
        "n": len(pairs),
        "by_source": by_source,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        write_jsonl(args.out, (p.to_dict() for p in pairs))
    print(json.dumps(manifest))
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.data_cmd == "synth":
        return _run_synth(args)
    if args.data_cmd == "pairs":
        return _run_pairs(args)
    return 2


def register_cli() -> None:
    register("data", "datasets: synthetic golden set, preference pairs", _add_args, _run)


__all__ = ["register_cli"]
