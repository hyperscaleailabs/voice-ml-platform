"""A release gate end to end, with nothing but the standard library installed.

Builds a small golden set inline, "synthesises" it with `StubTTS`, transcribes
it with the deterministic `NoisySTT` to get a WER, derives latency and error
SLIs from a synthetic trace, then applies `configs/gates.toml` to the combined
metrics and prints the `GateDecision`. A second, worse candidate shows the gate
failing and the no-regression check catching a drop against the baseline.

    python examples/demo_release_gate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmp.eval.gates import evaluate_gates, load_rules, no_regression
from vmp.eval.golden import GoldenSet, NoisySTT, StubTTS, score_asr, utterances_from_texts
from vmp.eval.latency import ttfa_percentiles
from vmp.observability.slo import DEFAULT_SLOS, compute_sli, evaluate_slos
from vmp.observability.trace import ListSink, Tracer
from vmp.types import EvalResult

GATES = ROOT / "configs" / "gates.toml"

# A golden set in the shape `vmp.data.synthetic` produces: (category, text).
SENTENCES = [
    ("basic", "what is the weather like today"),
    ("basic", "set a reminder for the morning"),
    ("basic", "tell me a short story about the sea"),
    ("basic", "how long does the battery last"),
    ("commands", "turn the kitchen light off"),
    ("commands", "play the next track please"),
    ("commands", "increase the volume by two steps"),
    ("asr_names", "call Alice Bergstrom at the office"),
    ("asr_names", "send a message to Nikolai Petrov"),
    ("asr_names", "add Priya Raghunathan to the meeting"),
    ("numbers", "set a timer for twenty one minutes"),
    ("numbers", "the total came to forty seven euros"),
]

# Time to first audio per turn, in milliseconds: a synthetic latency profile.
TTFA_MS = [740, 820, 880, 910, 960, 1020, 1090, 1180, 1260, 1480, 1720, 2350]


def synthetic_trace(ttfa_ms: list[int], error_turns: set[int]) -> list[dict]:
    """Trace rows for one session: one turn per `ttfa_ms` entry, some of them failing."""
    sink = ListSink()
    tracer = Tracer(sink)
    for number, ttfa in enumerate(ttfa_ms, 1):
        if number in error_turns:
            try:
                with tracer.span("turn", "release-check", number):
                    with tracer.span("stt", "release-check", number):
                        pass
                    with tracer.span("llm", "release-check", number):
                        raise TimeoutError("llm timed out")
            except TimeoutError:
                pass
            continue
        with tracer.span("turn", "release-check", number):
            with tracer.span("stt", "release-check", number):
                pass
            with tracer.span("llm", "release-check", number) as payload:
                payload["ttft_ms"] = round(ttfa * 0.25, 1)
            for seq in (1, 2):
                with tracer.span("tts", "release-check", number, seq=seq):
                    pass
                with tracer.span("playback", "release-check", number, seq=seq) as payload:
                    if seq == 1:
                        payload["response_ms"] = float(ttfa)
    return sink.rows


def asr_result(out_dir: Path, drop_rate: float, seed: int) -> EvalResult:
    golden = GoldenSet(utterances_from_texts(SENTENCES), name="golden")
    manifest = golden.build_audio(StubTTS(), out_dir, dry_run=False)
    return score_asr(NoisySTT(drop_rate=drop_rate, seed=seed), manifest)


def print_decision(title: str, decision) -> None:
    print(f"{title}: passed={decision.passed}")
    for reason in decision.reasons:
        print(f"  {reason}")


def main() -> int:
    t0 = time.perf_counter()
    rules = load_rules(GATES)
    print(f"gates from {GATES.relative_to(ROOT)}")
    for gate in rules:
        bound = f"max {gate.max}" if gate.max is not None else f"min {gate.min}"
        required = "required" if gate.required else "optional"
        print(f"  {gate.metric:<13} {bound:<12} {required:<9} {gate.description}")
    print()

    with tempfile.TemporaryDirectory(prefix="vmp-gate-") as tmp:
        root = Path(tmp)

        # -- candidate A: a good run --------------------------------------
        asr = asr_result(root / "clips-a", drop_rate=0.02, seed=11)
        print(f"golden set: {len(SENTENCES)} utterances, stt={asr.details['stt']}")
        print(
            f"  wer={asr.metrics['wer']:.4f}  cer={asr.metrics['cer']:.4f}  "
            f"exact_match={asr.metrics['exact_match']:.4f}"
        )
        for category, metrics in asr.details["per_category"].items():
            print(f"    {category:<10} n={int(metrics['n']):<3} wer={metrics['wer']:.4f}")

        rows = synthetic_trace(TTFA_MS, error_turns=set())
        sli = compute_sli(rows, wer=asr.metrics["wer"])
        latency = ttfa_percentiles(rows)
        print(f"\ntrace: {sli.n_turns} turns, {len(rows)} rows")
        print(
            f"  ttfa p50={latency['p50']:.0f} ms  p95={latency['p95']:.0f} ms  "
            f"mean={latency['mean']:.0f} ms"
        )
        print(f"  error_rate={sli.values['error_rate']:.4f}")

        for budget in evaluate_slos(DEFAULT_SLOS, sli):
            print(
                f"  slo {budget['slo']:<11} measured={budget['measured']:<9.4g} "
                f"objective={budget['objective']:<8.4g} burn={budget['burn_rate']:.2f} "
                f"met={budget['met']}"
            )

        candidate = EvalResult(
            name="candidate",
            metrics={**asr.metrics, **sli.values},
            n=asr.n,
            details={"source": "golden set + trace"},
        )
        print()
        decision = evaluate_gates([candidate], rules)
        print_decision("candidate A", decision)
        as_dict = decision.to_dict()
        print(
            "  GateDecision: "
            + json.dumps(
                {
                    "passed": as_dict["passed"],
                    "reasons": len(as_dict["reasons"]),
                    "results": [r["name"] for r in as_dict["results"]],
                }
            )
        )

        # -- candidate B: a regression ------------------------------------
        worse_asr = asr_result(root / "clips-b", drop_rate=0.25, seed=5)
        worse_rows = synthetic_trace([t + 2200 for t in TTFA_MS], error_turns={3, 7})
        worse_sli = compute_sli(worse_rows, wer=worse_asr.metrics["wer"])
        worse = EvalResult(
            name="candidate",
            metrics={**worse_asr.metrics, **worse_sli.values},
            n=worse_asr.n,
            details={"source": "golden set + trace"},
        )
        print()
        print_decision("candidate B", evaluate_gates([worse], rules))
        print()
        print_decision("candidate B vs candidate A", no_regression(worse, candidate))

    print(f"\ndone in {time.perf_counter() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
