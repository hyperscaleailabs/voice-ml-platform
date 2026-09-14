"""Stage tracer. One JSON object per line, the same schema as alpha-core.

Row keys: `ts, session, turn, event, span, seq, ms, payload`. A span writes two
rows, `<stage>.start` and `<stage>.end`; the end row carries `ms`. Nested spans
record their parent span id in the payload under `parent`.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

TRACE_KEYS = ("ts", "session", "turn", "event", "span", "seq", "ms", "payload")


class Sink(Protocol):
    """Where trace rows go. `write` receives one complete row dict."""

    def write(self, row: dict[str, Any]) -> None: ...


class ListSink:
    """Keeps rows in memory. The reference sink for tests and demos."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def clear(self) -> None:
        self.rows.clear()


class JsonlSink:
    """Appends one JSON object per line. `fsync=True` flushes to disk on every row."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            if self.fsync:
                fh.flush()
                os.fsync(fh.fileno())


class OtlpSink:
    """Forwards rows as OpenTelemetry span events. Imports `opentelemetry` lazily.

    Each `<stage>.end` row becomes a span named `<stage>` with attributes for the
    session, turn, seq and ms; start rows are recorded as events on the current span.
    """

    def __init__(self, endpoint: str | None = None, service_name: str = "vmp") -> None:
        self.endpoint = endpoint
        self.service_name = service_name
        self._tracer: Any = None

    def _ensure(self) -> Any:
        if self._tracer is None:
            from opentelemetry import trace as otel_trace  # lazy
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create({"service.name": self.service_name}))
            if self.endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(self.endpoint)))
            otel_trace.set_tracer_provider(provider)
            self._tracer = otel_trace.get_tracer(self.service_name)
        return self._tracer

    def write(self, row: dict[str, Any]) -> None:
        tracer = self._ensure()
        event = str(row.get("event", ""))
        if not event.endswith(".end"):
            return
        stage = event[: -len(".end")]
        attrs = {
            "vmp.session": str(row.get("session")),
            "vmp.turn": int(row.get("turn") or 0),
            "vmp.seq": int(row.get("seq") or 0),
            "vmp.ms": float(row.get("ms") or 0.0),
            "vmp.span": str(row.get("span")),
        }
        with tracer.start_as_current_span(stage, attributes=attrs):
            pass


_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vmp_current_span", default=None
)


class Tracer:
    """Writes `<stage>.start` / `<stage>.end` rows to a sink.

    Span ids are unique for the life of the process: a random per-tracer prefix
    plus a monotonically increasing counter.
    """

    def __init__(self, sink: Sink | None = None, *, clock: Any = None) -> None:
        self.sink: Sink = sink if sink is not None else ListSink()
        self._prefix = os.urandom(4).hex()
        self._counter = itertools.count(1)
        self._clock = clock or time.time
        self._perf = time.perf_counter

    def new_span_id(self) -> str:
        return f"{self._prefix}-{next(self._counter):06d}"

    def _row(
        self,
        event: str,
        session: str,
        turn: int,
        span: str,
        seq: int | None,
        ms: float | None,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        row = {
            "ts": self._clock(),
            "session": session,
            "turn": turn,
            "event": event,
            "span": span,
            "seq": seq,
            "ms": ms,
            "payload": dict(payload) if payload else {},
        }
        self.sink.write(row)
        return row

    def event(
        self,
        name: str,
        session: str,
        turn: int,
        *,
        seq: int | None = None,
        ms: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a single row (no start/end pair)."""
        parent = _current_span.get()
        p = dict(payload or {})
        if parent is not None:
            p.setdefault("parent", parent)
        return self._row(name, session, turn, self.new_span_id(), seq, ms, p)

    @contextmanager
    def span(
        self,
        stage: str,
        session: str,
        turn: int,
        seq: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Write `<stage>.start`, run the body, write `<stage>.end` with `ms`.

        The yielded dict is the end-row payload; the body can add keys to it
        (for example `ttft_ms` on `llm`, `response_ms` on `playback`).
        """
        span_id = self.new_span_id()
        parent = _current_span.get()
        start_payload = dict(payload or {})
        end_payload = dict(payload or {})
        if parent is not None:
            start_payload.setdefault("parent", parent)
            end_payload.setdefault("parent", parent)
        self._row(f"{stage}.start", session, turn, span_id, seq, None, start_payload)
        token = _current_span.set(span_id)
        t0 = self._perf()
        try:
            yield end_payload
        except BaseException as e:
            end_payload["error"] = type(e).__name__
            raise
        finally:
            ms = (self._perf() - t0) * 1000.0
            _current_span.reset(token)
            self._row(f"{stage}.end", session, turn, span_id, seq, round(ms, 3), end_payload)


def read_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL trace file, skipping blank lines."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trace not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                yield json.loads(s)


def _stage_of(event: str) -> str | None:
    return event[: -len(".end")] if event.endswith(".end") else None


def summarise_turn(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-stage `count`, `total_ms`, `first_ms`, `max_ms` from `<stage>.end` rows.

    Stages appear in the order their first end row was seen.
    """
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        stage = _stage_of(str(row.get("event", "")))
        if stage is None:
            continue
        ms = float(row.get("ms") or 0.0)
        s = out.setdefault(stage, {"count": 0, "total_ms": 0.0, "first_ms": ms, "max_ms": ms})
        s["count"] += 1
        s["total_ms"] += ms
        s["max_ms"] = max(s["max_ms"], ms)
    for s in out.values():
        s["total_ms"] = round(s["total_ms"], 3)
    return out


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    """One line per turn, like alpha-core: `stt 1x 233ms | llm 1x 192ms (first 192) | ...`."""
    parts = []
    for stage, s in summary.items():
        n = int(s["count"])
        piece = f"{stage} {n}x {s['total_ms']:.0f}ms"
        if n > 1:
            piece += f" (first {s['first_ms']:.0f}, max {s['max_ms']:.0f})"
        parts.append(piece)
    return " | ".join(parts)


def time_to_first_audio(rows: list[dict[str, Any]]) -> float | None:
    """`response_ms` from the first `playback.end` row, or None when absent."""
    for row in rows:
        if row.get("event") == "playback.end":
            payload = row.get("payload") or {}
            value = payload.get("response_ms")
            if value is not None:
                return float(value)
    return None


def group_by_turn(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Split rows into (session, turn) groups, preserving order."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("session")), int(row.get("turn") or 0))
        groups.setdefault(key, []).append(row)
    return groups


__all__ = [
    "TRACE_KEYS",
    "JsonlSink",
    "ListSink",
    "OtlpSink",
    "Sink",
    "Tracer",
    "format_summary",
    "group_by_turn",
    "read_trace",
    "summarise_turn",
    "time_to_first_audio",
]
