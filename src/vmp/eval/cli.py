"""`vmp eval` subcommands: WER, golden-set run, release gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp.cli import register
from vmp.eval.gates import evaluate_gates, load_rules
from vmp.eval.golden import GoldenSet, IdentitySTT, NoisySTT, StubTTS, score_asr
from vmp.eval.wer import align_words, corpus_wer, wer
from vmp.types import EvalResult


def _add_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="eval_command", required=True)

    p = sub.add_parser("wer", help="word error rate between reference and hypothesis")
    p.add_argument("--ref", required=True, help="reference text, or a file with one per line")
    p.add_argument("--hyp", required=True, help="hypothesis text, or a file with one per line")
    p.add_argument("--numbers", action="store_true", help="map number words to digits")

    p = sub.add_parser("golden", help="synthesise a golden set and score an STT")
    p.add_argument("--set", required=True, help="JSONL of utterances with meta.category")
    p.add_argument("--out", default=".vmp/golden", help="directory for clips")
    p.add_argument("--stt", choices=("identity", "noisy"), default="identity")
    p.add_argument("--drop-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("gate", help="evaluate release gates over metrics")
    p.add_argument("--rules", required=True, help="TOML with [[gate]] tables")
    p.add_argument("--metrics", required=True, help="JSON file: {metric: value} or EvalResult")


def _lines_or_text(value: str) -> list[str]:
    p = Path(value)
    if p.exists() and p.is_file():
        return [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [value]


def _run(args: argparse.Namespace) -> int:
    if args.eval_command == "wer":
        refs = _lines_or_text(args.ref)
        hyps = _lines_or_text(args.hyp)
        if len(refs) == 1 and len(hyps) == 1:
            a = align_words(refs[0], hyps[0], numbers=args.numbers)
            print(json.dumps({"wer": wer(refs[0], hyps[0], numbers=args.numbers), **a.to_dict()}))
        else:
            print(json.dumps(corpus_wer(refs, hyps, numbers=args.numbers), indent=2))
        return 0
    if args.eval_command == "golden":
        gs = GoldenSet.load(args.set)
        manifest = gs.build_audio(StubTTS(), args.out, dry_run=args.dry_run)
        stt = IdentitySTT() if args.stt == "identity" else NoisySTT(args.drop_rate, seed=args.seed)
        result = score_asr(stt, manifest, dry_run=args.dry_run)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.eval_command == "gate":
        rules = load_rules(args.rules)
        data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        if "metrics" in data and "name" in data:
            decision = evaluate_gates([EvalResult.from_dict(data)], rules)
        else:
            decision = evaluate_gates({k: float(v) for k, v in data.items()}, rules)
        print(json.dumps(decision.to_dict(), indent=2))
        return 0 if decision.passed else 1
    return 2


def register_cli() -> None:
    register("eval", "evaluation: WER, golden set, release gates", _add_args, _run)


__all__ = ["register_cli"]
