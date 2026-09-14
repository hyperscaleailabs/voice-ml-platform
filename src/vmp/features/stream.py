"""Streaming feature updates from stage traces.

Trace events are dicts with keys `ts, session, turn, event, span, seq, ms,
payload` (DESIGN.md). `SessionFeatureUpdater` consumes them in order and keeps
rolling per-session aggregates that match `session_features`:

- `turn_count`: number of `turn.end` events
- `avg_user_utterance_s`: mean of `listen.end` `ms` / 1000
- `avg_response_ms`: mean of `playback.end` `payload["response_ms"]`
- `last_intent`: latest `llm.end` `payload["intent"]`
- `accent_profile`: latest `stt.end` `payload["accent"]`

The in-process path is the reference; `KafkaSource` is the same loop fed from a
topic and imports `confluent_kafka` only when constructed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vmp.data.io import read_jsonl
from vmp.features.online import OnlineStore
from vmp.features.views import SESSION_FEATURES, FeatureView
from vmp.types import FeatureRow

Event = dict[str, Any]


@runtime_checkable
class EventSource(Protocol):
    def __iter__(self) -> Iterator[Event]: ...


class ListSource:
    def __init__(self, events: Iterable[Event]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[Event]:
        return iter(self._events)


class JsonlSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[Event]:
        return read_jsonl(self.path)


class KafkaSource:
    """Iterate JSON messages from a Kafka topic. Imports `confluent_kafka` in `__init__`."""

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "vmp-features",
        poll_timeout_s: float = 1.0,
        max_messages: int | None = None,
    ) -> None:
        try:
            from confluent_kafka import Consumer  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "KafkaSource needs confluent-kafka: pip install confluent-kafka"
            ) from e
        self._consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe([topic])
        self.poll_timeout_s = poll_timeout_s
        self.max_messages = max_messages

    def __iter__(self) -> Iterator[Event]:  # pragma: no cover - needs a broker
        import json

        n = 0
        while self.max_messages is None or n < self.max_messages:
            msg = self._consumer.poll(self.poll_timeout_s)
            if msg is None:
                continue
            if msg.error():
                raise RuntimeError(str(msg.error()))
            n += 1
            yield json.loads(msg.value())


@dataclass
class SessionState:
    """Running sums for one session. Means are computed on read, so updates are O(1)."""

    turn_count: int = 0
    listen_n: int = 0
    listen_ms_sum: float = 0.0
    response_n: int = 0
    response_ms_sum: float = 0.0
    last_intent: str | None = None
    accent_profile: str | None = None
    last_ts: float = 0.0

    def features(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "avg_user_utterance_s": (
                self.listen_ms_sum / self.listen_n / 1000.0 if self.listen_n else None
            ),
            "avg_response_ms": (
                self.response_ms_sum / self.response_n if self.response_n else None
            ),
            "last_intent": self.last_intent,
            "accent_profile": self.accent_profile,
        }


@dataclass
class SessionFeatureUpdater:
    """Fold trace events into `session_features` and push each turn's snapshot online."""

    online: OnlineStore | None = None
    view: FeatureView = SESSION_FEATURES
    states: dict[str, SessionState] = field(default_factory=dict)
    events_seen: int = 0
    events_used: int = 0
    rows: list[FeatureRow] = field(default_factory=list)

    def consume(self, event: Event) -> dict[str, Any] | None:
        """Apply one event. Returns the session's features when a turn completes, else None."""
        self.events_seen += 1
        session = event.get("session")
        name = event.get("event")
        if not session or not isinstance(name, str):
            return None
        st = self.states.setdefault(session, SessionState())
        payload = event.get("payload") or {}
        ts = float(event.get("ts") or 0.0)
        st.last_ts = max(st.last_ts, ts)
        used = True
        if name == "listen.end" and event.get("ms") is not None:
            st.listen_n += 1
            st.listen_ms_sum += float(event["ms"])
        elif name == "playback.end" and payload.get("response_ms") is not None:
            st.response_n += 1
            st.response_ms_sum += float(payload["response_ms"])
        elif name == "llm.end" and payload.get("intent") is not None:
            st.last_intent = str(payload["intent"])
        elif name == "stt.end" and payload.get("accent") is not None:
            st.accent_profile = str(payload["accent"])
        elif name == "turn.end":
            st.turn_count += 1
            feats = st.features()
            self.events_used += 1
            self.rows.append(FeatureRow(session, st.last_ts, dict(feats)))
            if self.online is not None:
                self.online.put(self.view.name, session, feats, st.last_ts, self.view.ttl_s)
            return feats
        else:
            used = False
        if used:
            self.events_used += 1
        return None

    def run(self, source: Iterable[Event]) -> dict[str, Any]:
        """Consume every event. Returns counts."""
        for ev in source:
            self.consume(ev)
        return {
            "view": self.view.name,
            "events_seen": self.events_seen,
            "events_used": self.events_used,
            "sessions": len(self.states),
            "turns": sum(s.turn_count for s in self.states.values()),
            "rows": len(self.rows),
        }

    def features(self, session: str) -> dict[str, Any] | None:
        st = self.states.get(session)
        return None if st is None else st.features()


__all__ = [
    "Event",
    "EventSource",
    "JsonlSource",
    "KafkaSource",
    "ListSource",
    "SessionFeatureUpdater",
    "SessionState",
]
