"""Release gates: metric thresholds a candidate must satisfy before promotion.

A rule is a dict like `{"metric": "wer", "max": 0.05}` or
`{"metric": "exact_match", "min": 0.9}`. Rules load from TOML `[[gate]]` tables.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vmp.types import EvalResult, GateDecision


@dataclass(frozen=True)
class Gate:
    """One threshold rule. At least one of `min` / `max` must be set."""

    metric: str
    min: float | None = None
    max: float | None = None
    required: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError(f"gate {self.metric!r} needs min or max")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Gate:
        return cls(
            metric=str(data["metric"]),
            min=float(data["min"]) if data.get("min") is not None else None,
            max=float(data["max"]) if data.get("max") is not None else None,
            required=bool(data.get("required", True)),
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "min": self.min,
            "max": self.max,
            "required": self.required,
            "description": self.description,
        }

    def check(self, value: float | None) -> tuple[bool, str]:
        if value is None:
            if self.required:
                return False, f"{self.metric}: missing (required)"
            return True, f"{self.metric}: missing (optional, skipped)"
        if self.max is not None and value > self.max:
            return False, f"{self.metric}: {value:.4g} > max {self.max:.4g}"
        if self.min is not None and value < self.min:
            return False, f"{self.metric}: {value:.4g} < min {self.min:.4g}"
        bounds = []
        if self.min is not None:
            bounds.append(f">= {self.min:.4g}")
        if self.max is not None:
            bounds.append(f"<= {self.max:.4g}")
        return True, f"{self.metric}: {value:.4g} ok ({', '.join(bounds)})"


def collect_metrics(results: Iterable[EvalResult] | dict[str, float]) -> dict[str, float]:
    """Flatten results into one metric dict.

    Each metric is available bare (`wer`) and prefixed by result name
    (`golden_asr.wer`); a later result overrides an earlier bare key.
    """
    if isinstance(results, dict):
        return {k: float(v) for k, v in results.items()}
    out: dict[str, float] = {}
    for r in results:
        for k, v in r.metrics.items():
            out[k] = float(v)
            out[f"{r.name}.{k}"] = float(v)
    return out


def evaluate_gates(
    results: Sequence[EvalResult] | dict[str, float],
    rules: Iterable[Gate | dict[str, Any]],
) -> GateDecision:
    """Apply every rule; the decision passes only when all rules pass."""
    metrics = collect_metrics(results)
    gates = [g if isinstance(g, Gate) else Gate.from_dict(g) for g in rules]
    reasons: list[str] = []
    passed = True
    for g in gates:
        ok, reason = g.check(metrics.get(g.metric))
        reasons.append(("PASS " if ok else "FAIL ") + reason)
        passed = passed and ok
    if not gates:
        reasons.append("no gates configured")
    result_list = list(results) if not isinstance(results, dict) else []
    return GateDecision(passed=passed, reasons=reasons, results=result_list)


def load_rules(source: str | Path) -> list[Gate]:
    """Load `[[gate]]` tables from a TOML path or TOML text."""
    text: str
    p = Path(str(source))
    text = p.read_text(encoding="utf-8") if p.suffix == ".toml" and p.exists() else str(source)
    data = tomllib.loads(text)
    return [Gate.from_dict(item) for item in data.get("gate", [])]


def no_regression(
    candidate: EvalResult | dict[str, float],
    baseline: EvalResult | dict[str, float],
    tolerance: float = 0.0,
    *,
    lower_is_better: Iterable[str] = ("wer", "cer", "error_rate", "ttfa_p50_ms", "ttfa_p95_ms"),
    metrics: Iterable[str] | None = None,
) -> GateDecision:
    """Fail when any shared metric moves in the worse direction by more than `tolerance`.

    `tolerance` is absolute. Metrics in `lower_is_better` regress when they
    increase; every other metric regresses when it decreases.
    """
    c = candidate.metrics if isinstance(candidate, EvalResult) else candidate
    b = baseline.metrics if isinstance(baseline, EvalResult) else baseline
    lower = set(lower_is_better)
    names = list(metrics) if metrics is not None else sorted(set(c) & set(b))
    reasons: list[str] = []
    passed = True
    for name in names:
        if name not in c or name not in b:
            reasons.append(f"SKIP {name}: not in both results")
            continue
        cv, bv = float(c[name]), float(b[name])
        delta = cv - bv
        worse = delta > tolerance if name in lower else delta < -tolerance
        tag = "FAIL" if worse else "PASS"
        reasons.append(
            f"{tag} {name}: baseline {bv:.4g} -> candidate {cv:.4g} (delta {delta:+.4g})"
        )
        passed = passed and not worse
    if not names:
        reasons.append("no shared metrics to compare")
    results = [r for r in (candidate, baseline) if isinstance(r, EvalResult)]
    return GateDecision(passed=passed, reasons=reasons, results=results)


__all__ = ["Gate", "collect_metrics", "evaluate_gates", "load_rules", "no_regression"]
