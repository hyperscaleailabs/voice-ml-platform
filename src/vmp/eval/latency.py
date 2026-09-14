"""Latency percentiles from trace rows, per stage and for time-to-first-audio."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from vmp.observability.slo import percentile
from vmp.observability.trace import group_by_turn, time_to_first_audio


def stage_percentiles(
    rows: Iterable[dict[str, Any]],
    ps: Sequence[float] = (50, 95),
) -> dict[str, dict[str, float]]:
    """`{stage: {"p50": .., "p95": .., "n": .., "mean": ..}}` from `<stage>.end` rows."""
    per_stage: dict[str, list[float]] = {}
    for row in rows:
        event = str(row.get("event", ""))
        if not event.endswith(".end"):
            continue
        per_stage.setdefault(event[: -len(".end")], []).append(float(row.get("ms") or 0.0))
    out: dict[str, dict[str, float]] = {}
    for stage, values in per_stage.items():
        entry = {f"p{int(p) if float(p).is_integer() else p}": percentile(values, p) for p in ps}
        entry["n"] = float(len(values))
        entry["mean"] = sum(values) / len(values)
        entry["max"] = max(values)
        out[stage] = entry
    return out


def ttfa_percentiles(
    rows: Iterable[dict[str, Any]],
    ps: Sequence[float] = (50, 95),
) -> dict[str, float]:
    """Percentiles of `playback.end` `response_ms` across turns. Empty when none."""
    values = [
        v
        for turn_rows in group_by_turn(list(rows)).values()
        if (v := time_to_first_audio(turn_rows)) is not None
    ]
    if not values:
        return {}
    out = {f"p{int(p) if float(p).is_integer() else p}": percentile(values, p) for p in ps}
    out["n"] = float(len(values))
    out["mean"] = sum(values) / len(values)
    return out


__all__ = ["percentile", "stage_percentiles", "ttfa_percentiles"]
