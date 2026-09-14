"""Offline store: historical feature rows with a point-in-time correct join.

The join answers "what did we know about entity E at time T?" and nothing
newer: for each request it picks the latest row whose `event_ts <= T` and
which is still inside the view's TTL at T. Rows from the future are never
visible, which is what keeps training features consistent with serving.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vmp.data.io import read_jsonl, write_jsonl
from vmp.features.views import FeatureView
from vmp.types import FeatureRow

Request = tuple[str, float]  # (entity_id, event_ts)


@runtime_checkable
class OfflineStore(Protocol):
    def write(self, view: FeatureView, rows: Iterable[FeatureRow]) -> int: ...

    def read(
        self, view: FeatureView, start_ts: float | None = None, end_ts: float | None = None
    ) -> list[FeatureRow]: ...

    def point_in_time(self, view: FeatureView, requests: Sequence[Request]) -> list[FeatureRow]: ...


def point_in_time_join(
    rows: Iterable[FeatureRow],
    requests: Sequence[Request],
    ttl_s: float | None,
    feature_names: Sequence[str] | None = None,
) -> list[FeatureRow]:
    """For each `(entity_id, event_ts)` return the freshest eligible row's values.

    Eligible: `row.event_ts <= event_ts` and, when `ttl_s` is set,
    `event_ts - row.event_ts <= ttl_s`. Missing -> all-None values. The output
    row's `event_ts` is the request time, so it can be joined back to a label.
    """
    by_entity: dict[str, list[FeatureRow]] = defaultdict(list)
    for r in rows:
        by_entity[r.entity_id].append(r)
    keys: dict[str, list[float]] = {}
    for e, rs in by_entity.items():
        rs.sort(key=lambda r: r.event_ts)
        keys[e] = [r.event_ts for r in rs]

    out: list[FeatureRow] = []
    for entity_id, event_ts in requests:
        empty: dict[str, Any] = (
            dict.fromkeys(feature_names) if feature_names is not None else {}
        )
        rs = by_entity.get(entity_id)
        if not rs:
            out.append(FeatureRow(entity_id, event_ts, empty))
            continue
        i = bisect.bisect_right(keys[entity_id], event_ts) - 1
        if i < 0:
            out.append(FeatureRow(entity_id, event_ts, empty))
            continue
        cand = rs[i]
        if ttl_s is not None and event_ts - cand.event_ts > ttl_s:
            out.append(FeatureRow(entity_id, event_ts, empty))
            continue
        values = dict(empty)
        values.update(cand.values)
        if feature_names is not None:
            values = {k: values.get(k) for k in feature_names}
        out.append(FeatureRow(entity_id, event_ts, values))
    return out


def latest_per_entity(rows: Iterable[FeatureRow]) -> dict[str, FeatureRow]:
    """The most recent row per entity. Ties keep the later one in iteration order."""
    latest: dict[str, FeatureRow] = {}
    for r in rows:
        cur = latest.get(r.entity_id)
        if cur is None or r.event_ts >= cur.event_ts:
            latest[r.entity_id] = r
    return latest


def _in_window(r: FeatureRow, start_ts: float | None, end_ts: float | None) -> bool:
    if start_ts is not None and r.event_ts < start_ts:
        return False
    return not (end_ts is not None and r.event_ts > end_ts)


class JsonlOfflineStore:
    """One append-only JSONL file per view under `root`. Reference implementation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, view: FeatureView) -> Path:
        return self.root / f"{view.name}.jsonl"

    def write(self, view: FeatureView, rows: Iterable[FeatureRow]) -> int:
        return write_jsonl(self.path(view), (r.to_dict() for r in rows), append=True)

    def read(
        self, view: FeatureView, start_ts: float | None = None, end_ts: float | None = None
    ) -> list[FeatureRow]:
        p = self.path(view)
        if not p.exists():
            return []
        rows = (FeatureRow.from_dict(d) for d in read_jsonl(p))
        return [r for r in rows if _in_window(r, start_ts, end_ts)]

    def point_in_time(self, view: FeatureView, requests: Sequence[Request]) -> list[FeatureRow]:
        return point_in_time_join(self.read(view), requests, view.ttl_s, view.feature_names)


class ParquetOfflineStore:
    """One Parquet file per view. Imports `pyarrow` inside the methods, never at import."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, view: FeatureView) -> Path:
        return self.root / f"{view.name}.parquet"

    @staticmethod
    def _pa():
        try:
            import pyarrow as pa  # type: ignore[import-not-found]
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "ParquetOfflineStore needs pyarrow: pip install 'voice-ml-platform[features]'"
            ) from e
        return pa, pq

    def write(self, view: FeatureView, rows: Iterable[FeatureRow]) -> int:
        pa, pq = self._pa()
        existing = self.read(view) if self.path(view).exists() else []
        all_rows = [*existing, *rows]
        records = [
            {"entity_id": r.entity_id, "event_ts": r.event_ts, **r.values} for r in all_rows
        ]
        self.root.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), self.path(view))
        return len(all_rows) - len(existing)

    def read(
        self, view: FeatureView, start_ts: float | None = None, end_ts: float | None = None
    ) -> list[FeatureRow]:
        _, pq = self._pa()
        p = self.path(view)
        if not p.exists():
            return []
        out: list[FeatureRow] = []
        for rec in pq.read_table(p).to_pylist():
            r = FeatureRow(
                entity_id=rec.pop("entity_id"),
                event_ts=float(rec.pop("event_ts")),
                values={k: rec.get(k) for k in view.feature_names},
            )
            if _in_window(r, start_ts, end_ts):
                out.append(r)
        return out

    def point_in_time(self, view: FeatureView, requests: Sequence[Request]) -> list[FeatureRow]:
        return point_in_time_join(self.read(view), requests, view.ttl_s, view.feature_names)


__all__ = [
    "JsonlOfflineStore",
    "OfflineStore",
    "ParquetOfflineStore",
    "Request",
    "latest_per_entity",
    "point_in_time_join",
]
