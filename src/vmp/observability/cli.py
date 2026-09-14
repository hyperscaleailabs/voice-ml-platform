"""`vmp obs` subcommands: summarise traces, evaluate SLOs, check drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp.cli import register
from vmp.observability.drift import DriftDetector
from vmp.observability.slo import DEFAULT_SLOS, compute_sli, evaluate_slos, slos_from_config
from vmp.observability.trace import (
    format_summary,
    group_by_turn,
    read_trace,
    summarise_turn,
    time_to_first_audio,
)


def _add_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="obs_command", required=True)

    p = sub.add_parser("summarise", help="one summary line per turn from a JSONL trace")
    p.add_argument("--trace", required=True, help="path to a JSONL trace")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")

    p = sub.add_parser("slo", help="compute SLIs from a trace and compare with SLOs")
    p.add_argument("--trace", required=True)
    p.add_argument("--config", default=None, help="TOML with [[slo]] tables")
    p.add_argument("--wer", type=float, default=None, help="golden-set WER to include")

    p = sub.add_parser("drift", help="PSI / KS between two numeric series")
    p.add_argument("--reference", required=True, help="text file, one number per line")
    p.add_argument("--current", required=True)
    p.add_argument("--psi-threshold", type=float, default=0.2)
    p.add_argument("--ks-threshold", type=float, default=0.1)


def _read_series(path: str) -> list[float]:
    return [float(s) for s in Path(path).read_text().split() if s.strip()]


def _run(args: argparse.Namespace) -> int:
    if args.obs_command == "summarise":
        rows = list(read_trace(args.trace))
        out = []
        for (session, turn), turn_rows in group_by_turn(rows).items():
            summary = summarise_turn(turn_rows)
            ttfa = time_to_first_audio(turn_rows)
            if args.json:
                out.append({"session": session, "turn": turn, "ttfa_ms": ttfa, "stages": summary})
            else:
                head = f"{session} turn {turn}"
                if ttfa is not None:
                    head += f" ttfa {ttfa:.0f}ms"
                print(f"{head}: {format_summary(summary)}")
        if args.json:
            print(json.dumps(out, indent=2))
        return 0
    if args.obs_command == "slo":
        slos = list(DEFAULT_SLOS)
        if args.config:
            from vmp.config import load_config

            slos = slos_from_config(load_config(args.config)) or slos
        sli = compute_sli(read_trace(args.trace), wer=args.wer)
        report = {"sli": sli.to_dict(), "budgets": evaluate_slos(slos, sli)}
        print(json.dumps(report, indent=2))
        return 0 if all(b["met"] in (True, None) for b in report["budgets"]) else 1
    if args.obs_command == "drift":
        det = DriftDetector(psi_threshold=args.psi_threshold, ks_threshold=args.ks_threshold)
        report = det.check(_read_series(args.reference), _read_series(args.current))
        print(json.dumps(report.to_dict(), indent=2))
        return 1 if report.drifted else 0
    return 2


def register_cli() -> None:
    register("obs", "observability: trace summaries, SLOs, drift", _add_args, _run)


__all__ = ["register_cli"]
