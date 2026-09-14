"""Online store: the current feature values per entity, with TTL expiry.

Serving reads from here on every turn, so the interface is small: put, get,
get_many, delete. Values older than the view's TTL are treated as absent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

Values = dict[str, Any]


@runtime_checkable
class OnlineStore(Protocol):
    def put(
        self, view: str, entity_id: str, values: Values, ts: float, ttl_s: float | None = None
    ) -> None: ...

    def get(self, view: str, entity_id: str, now: float | None = None) -> Values | None: ...

    def get_many(
        self, view: str, entity_ids: Sequence[str], now: float | None = None
    ) -> dict[str, Values | None]: ...

    def delete(self, view: str, entity_id: str) -> bool: ...


class InMemoryOnlineStore:
    """Dict-backed store. `clock` is injectable so tests control time."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._data: dict[tuple[str, str], tuple[float, float | None, Values]] = {}

    def put(
        self, view: str, entity_id: str, values: Values, ts: float, ttl_s: float | None = None
    ) -> None:
        self._data[(view, entity_id)] = (ts, ttl_s, dict(values))

    def get(self, view: str, entity_id: str, now: float | None = None) -> Values | None:
        item = self._data.get((view, entity_id))
        if item is None:
            return None
        ts, ttl_s, values = item
        t = self._clock() if now is None else now
        if ttl_s is not None and t - ts > ttl_s:
            return None
        return dict(values)

    def get_many(
        self, view: str, entity_ids: Sequence[str], now: float | None = None
    ) -> dict[str, Values | None]:
        return {e: self.get(view, e, now) for e in entity_ids}

    def delete(self, view: str, entity_id: str) -> bool:
        return self._data.pop((view, entity_id), None) is not None

    def timestamp(self, view: str, entity_id: str) -> float | None:
        item = self._data.get((view, entity_id))
        return None if item is None else item[0]

    def entities(self, view: str) -> list[str]:
        return sorted(e for v, e in self._data if v == view)

    def __len__(self) -> int:
        return len(self._data)


class RedisOnlineStore:
    """Redis-backed store. Key `vmp:features:<view>:<entity>`, JSON value, `EX` = ttl.

    Imports `redis` inside `__init__`, never at module import.
    """

    def __init__(self, url: str = "redis://localhost:6379/0", prefix: str = "vmp:features") -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "RedisOnlineStore needs redis: pip install 'voice-ml-platform[features]'"
            ) from e
        self._r = redis.Redis.from_url(url)
        self.prefix = prefix

    def _key(self, view: str, entity_id: str) -> str:
        return f"{self.prefix}:{view}:{entity_id}"

    def put(
        self, view: str, entity_id: str, values: Values, ts: float, ttl_s: float | None = None
    ) -> None:
        payload = json.dumps({"ts": ts, "values": values})
        ex = int(ttl_s) if ttl_s else None
        self._r.set(self._key(view, entity_id), payload, ex=ex)

    def get(self, view: str, entity_id: str, now: float | None = None) -> Values | None:
        raw = self._r.get(self._key(view, entity_id))
        if raw is None:
            return None
        return json.loads(raw)["values"]

    def get_many(
        self, view: str, entity_ids: Sequence[str], now: float | None = None
    ) -> dict[str, Values | None]:
        raws = self._r.mget([self._key(view, e) for e in entity_ids])
        return {
            e: (None if raw is None else json.loads(raw)["values"])
            for e, raw in zip(entity_ids, raws, strict=True)
        }

    def delete(self, view: str, entity_id: str) -> bool:
        return bool(self._r.delete(self._key(view, entity_id)))


__all__ = ["InMemoryOnlineStore", "OnlineStore", "RedisOnlineStore", "Values"]
