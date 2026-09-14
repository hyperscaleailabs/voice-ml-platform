"""Drift detectors: population stability index and two-sample Kolmogorov-Smirnov.

Pure Python and deterministic. Works on any numeric series: feature values,
per-item WER, per-turn latency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


def _edges(expected: Sequence[float], actual: Sequence[float], bins: int) -> list[float]:
    lo = min(min(expected), min(actual))
    hi = max(max(expected), max(actual))
    if hi == lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    return [lo + i * width for i in range(bins + 1)]


def _histogram(values: Sequence[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) - 1)
    last = len(counts) - 1
    lo, hi = edges[0], edges[-1]
    width = (hi - lo) / len(counts)
    for v in values:
        idx = int((v - lo) / width) if width > 0 else 0
        idx = min(max(idx, 0), last)
        counts[idx] += 1
    n = len(values)
    return [c / n for c in counts] if n else [0.0] * len(counts)


def psi(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
    *,
    epsilon: float = 1e-4,
) -> float:
    """Population stability index over equal-width bins spanning both samples.

    `sum((a - e) * ln(a / e))` with proportions floored at `epsilon`. Identical
    distributions give 0. Common reading: < 0.1 stable, 0.1-0.2 moderate, > 0.2 shifted.
    """
    if not expected or not actual:
        raise ValueError("psi needs non-empty samples")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    edges = _edges(expected, actual, bins)
    e = _histogram(expected, edges)
    a = _histogram(actual, edges)
    total = 0.0
    for ei, ai in zip(e, a, strict=True):
        ei = max(ei, epsilon)
        ai = max(ai, epsilon)
        total += (ai - ei) * math.log(ai / ei)
    return total


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample KS statistic: the largest gap between the empirical CDFs."""
    if not a or not b:
        raise ValueError("ks_statistic needs non-empty samples")
    xs = sorted(a)
    ys = sorted(b)
    i = j = 0
    n, m = len(xs), len(ys)
    d = 0.0
    while i < n and j < m:
        v = xs[i] if xs[i] <= ys[j] else ys[j]
        while i < n and xs[i] <= v:
            i += 1
        while j < m and ys[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass(frozen=True)
class DriftReport:
    """Outcome of one drift check. `drifted` is True when any threshold is exceeded."""

    name: str
    n_reference: int
    n_current: int
    psi: float
    ks: float
    mean_reference: float
    mean_current: float
    mean_shift: float
    psi_threshold: float
    ks_threshold: float
    drifted: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DriftDetector:
    """Checks a current series against a reference series with fixed thresholds."""

    psi_threshold: float = 0.2
    ks_threshold: float = 0.1
    bins: int = 10

    def check(
        self,
        reference: Sequence[float],
        current: Sequence[float],
        name: str = "feature",
    ) -> DriftReport:
        p = psi(reference, current, self.bins)
        k = ks_statistic(reference, current)
        mr, mc = _mean(reference), _mean(current)
        reasons: list[str] = []
        if p > self.psi_threshold:
            reasons.append(f"{name}: psi {p:.4f} > {self.psi_threshold}")
        if k > self.ks_threshold:
            reasons.append(f"{name}: ks {k:.4f} > {self.ks_threshold}")
        return DriftReport(
            name=name,
            n_reference=len(reference),
            n_current=len(current),
            psi=p,
            ks=k,
            mean_reference=mr,
            mean_current=mc,
            mean_shift=mc - mr,
            psi_threshold=self.psi_threshold,
            ks_threshold=self.ks_threshold,
            drifted=bool(reasons),
            reasons=reasons,
        )

    def check_many(
        self,
        reference: dict[str, Sequence[float]],
        current: dict[str, Sequence[float]],
    ) -> dict[str, DriftReport]:
        """Check every feature present in both dicts."""
        return {
            k: self.check(reference[k], current[k], name=k)
            for k in sorted(reference)
            if k in current
        }


__all__ = ["DriftDetector", "DriftReport", "ks_statistic", "psi"]
