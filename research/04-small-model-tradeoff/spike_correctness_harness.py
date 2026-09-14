"""Spike 04 — a harness that refuses to report latency without correctness.

Runs with nothing but the standard library installed. It writes a synthetic
JSONL of question / expected / answer / latency rows for two model tiers, then
scores that file with `vmp.eval.wer.exact_match` and prints one table in which
speed and correctness sit in the same row and cannot be quoted apart.

The harness (`score_rows`, `render_table`) is the point; the JSONL it scores is
synthetic and proves nothing about any model. The measured version of this
trade-off is cited in README.md.

    python research/04-small-model-tradeoff/spike_correctness_harness.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vmp.data.io import read_jsonl, write_jsonl
from vmp.data.synthetic import generate_golden_set
from vmp.eval.latency import percentile
from vmp.eval.wer import exact_match, normalise

# Two tiers, as the serving config names them. The numbers below are the
# *synthetic generator's* parameters, not measurements: a small model that
# answers quickly and is sometimes wrong, a large one that is slower and is not.
TIERS: dict[str, dict[str, Any]] = {
    "small-270m": {"median_ms": 190.0, "spread_ms": 70.0, "wrong_every": 3},
    "large-4b": {"median_ms": 4400.0, "spread_ms": 900.0, "wrong_every": 0},
}
WRONG_ANSWERS = ("forty-two", "Oslo", "Venus", "ninety", "oxygen", "twelve")
N_QUESTIONS = 18


def synthetic_runs(path: Path) -> int:
    """Write the JSONL this spike scores. Deterministic in the seeds below.

    One row per (question, tier): the question, the expected answer from the
    golden set, the answer the tier "gave", and the latency of the llm stage.
    """
    questions = generate_golden_set(
        per_category=N_QUESTIONS, seed=3, categories=("factual_short",)
    )
    rows: list[dict[str, Any]] = []
    for tier, cfg in TIERS.items():
        rng = random.Random(f"{tier}:latency")
        for i, u in enumerate(questions):
            expected = str(u.meta["answer"])
            wrong_every = int(cfg["wrong_every"])
            wrong = bool(wrong_every) and i % wrong_every == 0
            answer = WRONG_ANSWERS[i % len(WRONG_ANSWERS)] if wrong else expected
            rows.append(
                {
                    "id": f"{tier}-{i:03d}",
                    "model": tier,
                    "question": u.text,
                    "expected": expected,
                    "answer": f"It's {answer}.",
                    "llm_ms": round(
                        max(1.0, rng.gauss(float(cfg["median_ms"]), float(cfg["spread_ms"]))), 1
                    ),
                }
            )
    return write_jsonl(path, rows)


def score_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-model correctness and latency. `exact_match` is the package's own.

    `numbers=True` maps number words to digits before comparing, so "sixty" and
    "60" are the same answer. The answer text is compared as a whole utterance,
    which is why the expected answer is embedded in a spoken sentence.
    """
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    out: dict[str, dict[str, float]] = {}
    for model, model_rows in by_model.items():
        correct = 0
        for row in model_rows:
            expected, answer = str(row["expected"]), str(row["answer"])
            # The expected answer is a phrase inside the reply; match on containment
            # of the normalised tokens, and fall back to whole-utterance equality.
            correct += int(
                exact_match(expected, answer, numbers=True)
                or _contains(expected, answer)
            )
        latencies = [float(row["llm_ms"]) for row in model_rows]
        n = len(model_rows)
        out[model] = {
            "n": float(n),
            "correct": float(correct),
            "accuracy": correct / n,
            "wrong": float(n - correct),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "mean_ms": sum(latencies) / n,
        }
    return out


def _contains(expected: str, answer: str) -> bool:
    """Normalised token-sequence containment, using the package's normaliser."""
    exp = normalise(expected, numbers=True)
    hyp = normalise(answer, numbers=True)
    if not exp or len(exp) > len(hyp):
        return False
    return any(hyp[i : i + len(exp)] == exp for i in range(len(hyp) - len(exp) + 1))


def render_table(scores: dict[str, dict[str, float]]) -> str:
    """One row per model. Latency and correctness in the same row, always."""
    header = (
        f"  {'model':<12} {'n':>3} {'correct':>10} {'accuracy':>9} "
        f"{'p50 ms':>9} {'p95 ms':>9} {'mean ms':>9}"
    )
    lines = [header, "  " + "-" * (len(header) - 2)]
    for model in sorted(scores, key=lambda m: scores[m]["p50_ms"]):
        s = scores[model]
        correct = f"{int(s['correct'])}/{int(s['n'])}"
        lines.append(
            f"  {model:<12} {int(s['n']):>3} {correct:>10} "
            f"{s['accuracy'] * 100:>8.1f}% {s['p50_ms']:>9.1f} {s['p95_ms']:>9.1f} "
            f"{s['mean_ms']:>9.1f}"
        )
    return "\n".join(lines)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tier_runs.jsonl"
        written = synthetic_runs(path)

        print("=" * 78)
        print("1. synthetic run file")
        print("=" * 78)
        distinct = len({str(r["question"]) for r in read_jsonl(path)})
        print(f"  rows            {written}  ({N_QUESTIONS} questions x {len(TIERS)} tiers)")
        print(f"  distinct        {distinct} questions; the generator samples the "
              "factual_short")
        print("                  templates with replacement, so questions repeat")
        print("  fields          id, model, question, expected, answer, llm_ms")
        print("  questions       vmp.data.synthetic factual_short, seed 3")
        print("  answers         generated: the large tier is always right, the small tier")
        print("                  is wrong on every third question, by construction")

        rows = list(read_jsonl(path))
        scores = score_rows(rows)

        print()
        print("=" * 78)
        print("2. latency against correctness (SYNTHETIC)")
        print("=" * 78)
        print(render_table(scores))

        fast, slow = sorted(scores, key=lambda m: scores[m]["p50_ms"])
        speedup = scores[slow]["p50_ms"] / scores[fast]["p50_ms"]
        lost = scores[slow]["accuracy"] - scores[fast]["accuracy"]
        print()
        print(f"  {fast} is {speedup:.1f}x faster at p50 and {lost * 100:.1f} points less")
        print("  accurate on the same questions. Neither half of that sentence is")
        print("  publishable on its own, which is the whole claim.")

        print()
        print("=" * 78)
        print("3. what a real measurement looked like")
        print("=" * 78)
        print("  gemma3:270m vs gemma3 4B on 6 factual questions: llm median 192 ms vs")
        print("  4,427 ms; 2/6 wrong vs 0/6 (alpha-core, cycle 5, 2026-09-12,")
        print("  `notebook_optimized.ipynb`). Six questions is a small set and the")
        print("  correctness figure carries that caveat wherever it is quoted.")
        print()
        print("  The table above is this harness running on a file it invented. It")
        print("  measures no model. This repository has not run the comparison; see the")
        print("  Outcome section of README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
