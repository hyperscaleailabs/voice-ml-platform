"""Service level objectives for a voice agent.

An `SLO` names an indicator (`ttfa_p95_ms`, `error_rate`, `wer`, ...) and the
bound it must stay within over a window. `compute_sli` derives indicator values
from trace rows; `error_budget` and `burn_rate` compare them to the objective.

The default objectives below are targets for this platform, not measurements.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from vmp.observability.trace import group_by_turn, time_to_first_audio


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (`p` in 0..100) of a non-empty sequence."""
    if not values:
        raise ValueError("percentile of empty sequence")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


@dataclass(frozen=True)
class SLO:
    """`indicator` must be `comparator` `objective` over `window` (e.g. "30d")."""

    name: str
    indicator: str
    objective: float
    window: str = "30d"
    comparator: str = "<="
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def met(self, value: float) -> bool:
        if self.comparator == "<=":
            return value <= self.objective
        if self.comparator == ">=":
            return value >= self.objective
        raise ValueError(f"unknown comparator: {self.comparator}")


@dataclass
class SLI:
    """Indicator values computed from evidence (trace rows, eval results)."""

    values: dict[str, float] = field(default_factory=dict)
    n_turns: int = 0
    sources: dict[str, str] = field(default_factory=dict)

    def get(self, indicator: str) -> float | None:
        return self.values.get(indicator)

    def to_dict(self) -> dict[str, Any]:
        return {"values": dict(self.values), "n_turns": self.n_turns, "sources": dict(self.sources)}


# Targets for this platform. They are objectives to design against, not measured
# results; measurements come from `compute_sli` on real traces.
DEFAULT_SLOS: tuple[SLO, ...] = (
    SLO(
        "ttfa_p95",
        "ttfa_p95_ms",
        3000.0,
        "30d",
        "<=",
        "target: p95 time to first audio at or below 3000 ms",
    ),
    SLO("error_rate", "error_rate", 0.01, "30d", "<=", "target: at most 1% of turns error"),
    SLO("wer", "wer", 0.05, "30d", "<=", "target: golden-set WER at or below 5%"),
)


def compute_sli(
    rows: Iterable[dict[str, Any]],
    *,
    wer: float | None = None,
) -> SLI:
    """Derive `ttfa_p50_ms`, `ttfa_p95_ms`, `error_rate` (and `wer` if given) from rows.

    A turn errors when any of its `.end` rows carries `payload.error`, or when its
    `turn.start` has no matching `turn.end`.
    """
    row_list = list(rows)
    groups = group_by_turn(row_list)
    ttfas: list[float] = []
    errors = 0
    for turn_rows in groups.values():
        ttfa = time_to_first_audio(turn_rows)
        if ttfa is not None:
            ttfas.append(ttfa)
        failed = any(
            str(r.get("event", "")).endswith(".end") and (r.get("payload") or {}).get("error")
            for r in turn_rows
        )
        started = any(r.get("event") == "turn.start" for r in turn_rows)
        ended = any(r.get("event") == "turn.end" for r in turn_rows)
        if failed or (started and not ended):
            errors += 1
    sli = SLI(n_turns=len(groups))
    if ttfas:
        sli.values["ttfa_p50_ms"] = percentile(ttfas, 50)
        sli.values["ttfa_p95_ms"] = percentile(ttfas, 95)
        sli.sources["ttfa_p50_ms"] = "trace: playback.end response_ms"
        sli.sources["ttfa_p95_ms"] = "trace: playback.end response_ms"
    if groups:
        sli.values["error_rate"] = errors / len(groups)
        sli.sources["error_rate"] = "trace: turns with payload.error or missing turn.end"
    if wer is not None:
        sli.values["wer"] = float(wer)
        sli.sources["wer"] = "eval: golden set"
    return sli


def burn_rate(slo: SLO, value: float) -> float:
    """How fast the budget is used: measured / objective for `<=` objectives.

    1.0 means exactly on target; 2.0 means the budget is consumed twice as fast
    as allowed. For `>=` objectives the ratio is inverted: (1 - value) / (1 - objective).
    """
    if slo.comparator == "<=":
        if slo.objective == 0:
            return math.inf if value > 0 else 0.0
        return value / slo.objective
    if slo.comparator == ">=":
        allowed = 1.0 - slo.objective
        if allowed <= 0:
            return math.inf if value < slo.objective else 0.0
        return (1.0 - value) / allowed
    raise ValueError(f"unknown comparator: {slo.comparator}")


def error_budget(slo: SLO, sli: SLI | float, *, elapsed_fraction: float = 1.0) -> dict[str, Any]:
    """Budget report for one SLO: `burn_rate`, `consumed`, `remaining`, `met`.

    `consumed = burn_rate * elapsed_fraction` (fraction of the window elapsed);
    `remaining = 1 - consumed`, which goes negative when the budget is exhausted.
    """
    value = sli.get(slo.indicator) if isinstance(sli, SLI) else sli
    if value is None:
        return {
            "slo": slo.name,
            "indicator": slo.indicator,
            "objective": slo.objective,
            "measured": None,
            "burn_rate": None,
            "consumed": None,
            "remaining": None,
            "met": None,
            "note": "indicator not measured",
        }
    rate = burn_rate(slo, float(value))
    consumed = rate * elapsed_fraction
    return {
        "slo": slo.name,
        "indicator": slo.indicator,
        "objective": slo.objective,
        "measured": float(value),
        "burn_rate": rate,
        "consumed": consumed,
        "remaining": 1.0 - consumed,
        "met": slo.met(float(value)),
    }


def evaluate_slos(slos: Iterable[SLO], sli: SLI) -> list[dict[str, Any]]:
    return [error_budget(s, sli) for s in slos]


_INDICATOR_EXPR = {
    "ttfa_p95_ms": "histogram_quantile(0.95, sum(rate(vmp_ttfa_ms_bucket[{w}])) by (le))",
    "ttfa_p50_ms": "histogram_quantile(0.50, sum(rate(vmp_ttfa_ms_bucket[{w}])) by (le))",
    "error_rate": "sum(rate(vmp_errors_total[{w}])) / sum(rate(vmp_turn_total[{w}]))",
    "wer": 'max(vmp_wer{{set="golden"}})',
}


def _expr(slo: SLO, window: str) -> str:
    template = _INDICATOR_EXPR.get(slo.indicator, slo.indicator)
    return template.format(w=window)


def to_prometheus_rules_yaml(
    slos: Iterable[SLO],
    *,
    group: str = "vmp-slo",
    rate_window: str = "5m",
    for_duration: str = "10m",
) -> str:
    """Render alerting rules as YAML text without a YAML library.

    One alert per SLO, named `VmpSlo<Name>` (CamelCase), firing when the
    indicator breaks its objective for `for_duration`.
    """
    lines = ["groups:", f"  - name: {group}", "    rules:"]
    for slo in slos:
        camel = "".join(part.capitalize() for part in slo.name.replace("-", "_").split("_"))
        op = ">" if slo.comparator == "<=" else "<"
        expr = f"({_expr(slo, rate_window)}) {op} {slo.objective}"
        lines.extend(
            [
                f"      - alert: VmpSlo{camel}",
                f'        expr: "{expr}"',
                f"        for: {for_duration}",
                "        labels:",
                "          severity: page",
                f"          slo: {slo.name}",
                "        annotations:",
                f'          summary: "{slo.indicator} outside objective ({slo.comparator} '
                f'{slo.objective}, window {slo.window})"',
            ]
        )
        if slo.description:
            lines.append(f'          description: "{slo.description}"')
    return "\n".join(lines) + "\n"


def slos_from_config(data: dict[str, Any]) -> list[SLO]:
    """Build SLOs from a TOML dict shaped like `configs/slo.toml`."""
    out: list[SLO] = []
    for item in data.get("slo", []):
        out.append(
            SLO(
                name=str(item["name"]),
                indicator=str(item["indicator"]),
                objective=float(item["objective"]),
                window=str(item.get("window", "30d")),
                comparator=str(item.get("comparator", "<=")),
                description=str(item.get("description", "")),
            )
        )
    return out


__all__ = [
    "DEFAULT_SLOS",
    "SLI",
    "SLO",
    "burn_rate",
    "compute_sli",
    "error_budget",
    "evaluate_slos",
    "percentile",
    "slos_from_config",
    "to_prometheus_rules_yaml",
]
