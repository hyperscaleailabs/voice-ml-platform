"""Materialise a view from the offline store into the online store.

For a time window, take the latest row per entity and write it online with the
view's TTL. Returns a manifest so a scheduler can log what happened. `dry_run`
reads and plans but writes nothing.
"""

from __future__ import annotations

import time
from typing import Any

from vmp.features.offline import OfflineStore, latest_per_entity
from vmp.features.online import OnlineStore
from vmp.features.views import FeatureView


def materialize(
    view: FeatureView,
    offline: OfflineStore,
    online: OnlineStore | None,
    start_ts: float | None = None,
    end_ts: float | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy the freshest row per entity in `[start_ts, end_ts]` to the online store."""
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError(f"end_ts {end_ts} is before start_ts {start_ts}")
    if online is None and not dry_run:
        raise ValueError("online store is required unless dry_run=True")
    t0 = time.perf_counter()
    rows = offline.read(view, start_ts, end_ts)
    latest = latest_per_entity(rows)
    written = 0
    if not dry_run:
        assert online is not None
        for entity_id, row in latest.items():
            values = {k: row.values.get(k) for k in view.feature_names}
            online.put(view.name, entity_id, values, row.event_ts, view.ttl_s)
            written += 1
    return {
        "view": view.name,
        "entity": view.entity,
        "features": list(view.feature_names),
        "ttl_s": view.ttl_s,
        "window": {"start_ts": start_ts, "end_ts": end_ts},
        "rows_read": len(rows),
        "entities": len(latest),
        "entities_written": written,
        "dry_run": dry_run,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


__all__ = ["materialize"]
