"""Evaluation: WER/CER, golden set, judges, preference win-rate, latency, release gates.

Standard library only. `jiwer` is an optional cross-check imported lazily.
"""

from __future__ import annotations

from vmp.eval.gates import Gate, evaluate_gates, load_rules, no_regression
from vmp.eval.golden import GoldenSet, IdentitySTT, NoisySTT, StubTTS
from vmp.eval.judge import ExactMatchJudge, Judge, LLMJudge, RubricJudge
from vmp.eval.latency import percentile, stage_percentiles, ttfa_percentiles
from vmp.eval.preference import dpo_margins, win_rate
from vmp.eval.wer import (
    Alignment,
    align_words,
    cer,
    corpus_wer,
    exact_match,
    normalise,
    wer,
)

__all__ = [
    "Alignment",
    "ExactMatchJudge",
    "Gate",
    "GoldenSet",
    "IdentitySTT",
    "Judge",
    "LLMJudge",
    "NoisySTT",
    "RubricJudge",
    "StubTTS",
    "align_words",
    "cer",
    "corpus_wer",
    "dpo_margins",
    "evaluate_gates",
    "exact_match",
    "load_rules",
    "no_regression",
    "normalise",
    "percentile",
    "stage_percentiles",
    "ttfa_percentiles",
    "wer",
    "win_rate",
]
