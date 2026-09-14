"""HTTP and WebSocket API for the voice agent runtime.

`create_app` imports FastAPI lazily so `vmp.serving` stays importable with the
standard library alone. Endpoints:

- `POST /v1/sessions` (`{"system_prompt": ...}` optional) -> `{"id": ...}`
- `GET /v1/sessions/{id}` -> the session as JSON
- `POST /v1/sessions/{id}/turns` (`{"text": ...}` or multipart `audio`) -> turn JSON
- `WS /v1/sessions/{id}/stream`: client sends `{"text": ...}`; server streams
  `{"type": "sentence", "seq", "text"}`, `{"type": "audio", "seq", "bytes_b64"}`,
  then `{"type": "done", "turn", "stages"}`.
- `GET /healthz`, `GET /readyz` (calls `ready()` on backends that have it),
  `GET /metrics` (Prometheus text format from in-process counters).

Cross-cutting: request id (`X-Request-ID`), one JSON access-log line per request,
request size limit, per-turn timeout, optional bearer token (`VMP_API_TOKEN`).
"""

# No `from __future__ import annotations` here on purpose: FastAPI resolves endpoint
# annotations at runtime against module globals, and `Request` / `WebSocket` are
# imported lazily inside `create_app`. Every annotation in this file is valid on 3.11.

import asyncio
import base64
import email.parser
import email.policy
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from vmp.serving.backends import wav_duration_ms
from vmp.serving.runtime import SessionStore, TraceSink, VoiceRuntime
from vmp.types import Turn

log = logging.getLogger("vmp.serving.api")

DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024
DEFAULT_TURN_TIMEOUT_S = 60.0
UNAUTHENTICATED_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})
LATENCY_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


# ---------------------------------------------------------------- metrics


@runtime_checkable
class MetricsLike(Protocol):
    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None: ...

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None: ...

    def render(self) -> str: ...


class Metrics:
    """Counters and histograms rendered in Prometheus text format, no client library.

    Histograms use `LATENCY_BUCKETS_S` (seconds). Rendering follows the
    exposition format: `# HELP`, `# TYPE`, then samples with sorted labels.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}
        self._hist_sum: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._hist_count: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self.help: dict[str, str] = {}

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None):
        return (name, tuple(sorted((labels or {}).items())))

    def describe(self, name: str, help_text: str) -> None:
        self.help[name] = help_text

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        k = self._key(name, labels)
        with self._lock:
            self._counters[k] = self._counters.get(k, 0.0) + value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        k = self._key(name, labels)
        with self._lock:
            buckets = self._hist.setdefault(k, [0.0] * len(LATENCY_BUCKETS_S))
            for i, le in enumerate(LATENCY_BUCKETS_S):
                if value <= le:
                    buckets[i] += 1
            self._hist_sum[k] = self._hist_sum.get(k, 0.0) + value
            self._hist_count[k] = self._hist_count.get(k, 0) + 1

    @staticmethod
    def _fmt_labels(labels: tuple[tuple[str, str], ...], extra: dict[str, str] | None = None):
        items = list(labels) + list((extra or {}).items())
        if not items:
            return ""
        body = ",".join(f'{k}="{str(v).replace(chr(34), chr(92) + chr(34))}"' for k, v in items)
        return "{" + body + "}"

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            names = sorted({k[0] for k in self._counters})
            for name in names:
                lines.append(f"# HELP {name} {self.help.get(name, name)}")
                lines.append(f"# TYPE {name} counter")
                for (n, labels), v in sorted(self._counters.items()):
                    if n == name:
                        lines.append(f"{name}{self._fmt_labels(labels)} {v:g}")
            names = sorted({k[0] for k in self._hist})
            for name in names:
                lines.append(f"# HELP {name} {self.help.get(name, name)}")
                lines.append(f"# TYPE {name} histogram")
                for (n, labels), buckets in sorted(self._hist.items()):
                    if n != name:
                        continue
                    for le, count in zip(LATENCY_BUCKETS_S, buckets, strict=True):
                        lines.append(
                            f"{name}_bucket{self._fmt_labels(labels, {'le': f'{le:g}'})} {count:g}"
                        )
                    total = self._hist_count[(n, labels)]
                    lines.append(f"{name}_bucket{self._fmt_labels(labels, {'le': '+Inf'})} {total}")
                    lines.append(
                        f"{name}_sum{self._fmt_labels(labels)} {self._hist_sum[(n, labels)]:g}"
                    )
                    lines.append(f"{name}_count{self._fmt_labels(labels)} {total}")
        return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------- helpers


def parse_multipart(body: bytes, content_type: str) -> dict[str, bytes]:
    """Stdlib multipart/form-data parser: field name -> raw bytes."""
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(header + body)
    out: dict[str, bytes] = {}
    if not msg.is_multipart():
        return out
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        payload = part.get_payload(decode=True)
        out[str(name)] = payload if isinstance(payload, bytes) else b""
    return out


def check_bearer(header_value: str | None, token: str | None) -> bool:
    """True when auth is off, or the header carries exactly `token` (constant time)."""
    if not token:
        return True
    if not header_value or not header_value.startswith("Bearer "):
        return False
    return hmac.compare_digest(header_value[len("Bearer ") :].encode(), token.encode())


def turn_to_json(turn: Turn, sentences: list[str], audio_ms: float) -> dict[str, Any]:
    d = turn.to_dict()
    d["stages"] = {k: round(v, 3) for k, v in turn.stages.items()}
    d["sentences"] = sentences
    d["audio_ms"] = round(audio_ms, 3)
    return d


class _TurnCollector:
    """Collects sentence/audio callbacks from the runtime for a non-streaming response."""

    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.audio_ms = 0.0

    def on_sentence(self, seq: int, text: str) -> None:
        self.sentences.append(text)

    def on_audio(self, seq: int, wav: bytes) -> None:
        self.audio_ms += wav_duration_ms(wav)


def _record_turn(metrics: MetricsLike, turn: Turn) -> None:
    metrics.inc("vmp_turns_total")
    for stage, ms in turn.stages.items():
        metrics.observe("vmp_turn_stage_seconds", ms / 1000.0, {"stage": stage})
    if "first_audio" in turn.stages:
        metrics.observe("vmp_first_audio_seconds", turn.stages["first_audio"] / 1000.0)


# ------------------------------------------------------------------- app


def create_app(
    runtime: VoiceRuntime,
    session_store: SessionStore,
    tracer: TraceSink | None = None,
    metrics: MetricsLike | None = None,
    *,
    api_token: str | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    turn_timeout_s: float = DEFAULT_TURN_TIMEOUT_S,
    title: str = "voice-ml-platform",
) -> Any:
    """Build the FastAPI application. `fastapi` is imported here, not at module import.

    `api_token=None` reads `VMP_API_TOKEN` from the environment; an empty value
    disables auth. `tracer` overrides the runtime's tracer when given.
    """
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import JSONResponse, PlainTextResponse

    if tracer is not None:
        runtime.tracer = tracer
    m: MetricsLike = metrics if metrics is not None else Metrics()
    if isinstance(m, Metrics):
        m.describe("vmp_http_requests_total", "HTTP requests by method, route and status")
        m.describe("vmp_http_request_seconds", "HTTP request latency in seconds")
        m.describe("vmp_turns_total", "Completed voice turns")
        m.describe("vmp_turn_stage_seconds", "Per-stage latency of a voice turn in seconds")
        m.describe("vmp_first_audio_seconds", "Time to first audio in seconds")
    token = os.environ.get("VMP_API_TOKEN", "") if api_token is None else api_token
    session_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def lock_for(session_id: str) -> threading.Lock:
        with locks_guard:
            return session_locks.setdefault(session_id, threading.Lock())

    app = FastAPI(title=title, docs_url=None, redoc_url=None)
    app.state.runtime = runtime
    app.state.session_store = session_store
    app.state.metrics = m

    # ------------------------------------------------------------ middleware

    @app.middleware("http")
    async def request_middleware(request: Request, call_next: Callable) -> Any:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        path = request.url.path
        response: Any
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > max_request_bytes:
            response = JSONResponse({"error": "request too large"}, status_code=413)
        elif path not in UNAUTHENTICATED_PATHS and not check_bearer(
            request.headers.get("authorization"), token
        ):
            response = JSONResponse({"error": "unauthorized"}, status_code=401)
        else:
            response = await call_next(request)
        response.headers["x-request-id"] = rid
        elapsed = time.perf_counter() - t0
        route = request.scope.get("route")
        route_path = getattr(route, "path", path)
        m.inc(
            "vmp_http_requests_total",
            {"method": request.method, "path": route_path, "status": str(response.status_code)},
        )
        m.observe(
            "vmp_http_request_seconds", elapsed, {"method": request.method, "path": route_path}
        )
        log.info(
            json.dumps(
                {
                    "ts": time.time(),
                    "request_id": rid,
                    "method": request.method,
                    "path": path,
                    "status": response.status_code,
                    "ms": round(elapsed * 1000.0, 3),
                    "client": request.client.host if request.client else None,
                }
            )
        )
        return response

    # -------------------------------------------------------------- health

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Any:
        checks: dict[str, bool] = {}
        for name in ("stt", "llm", "tts"):
            backend = getattr(runtime, name)
            fn = getattr(backend, "ready", None)
            if callable(fn):
                try:
                    checks[name] = bool(await run_in_threadpool(fn))
                except Exception:  # a failing readiness probe is "not ready", not 500
                    checks[name] = False
            else:
                checks[name] = True
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ok" if ok else "not ready", "checks": checks},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics")
    async def metrics_endpoint() -> Any:
        return PlainTextResponse(m.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    # ------------------------------------------------------------ sessions

    @app.post("/v1/sessions", status_code=201)
    async def create_session(request: Request) -> dict[str, Any]:
        body = await request.body()
        data: dict[str, Any] = {}
        if body:
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                return JSONResponse({"error": "invalid json"}, status_code=400)  # type: ignore[return-value]
        system_prompt = str(data.get("system_prompt") or "")
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        session = session_store.create(system_prompt=system_prompt, meta=meta)
        return {"id": session.id, "created_at": session.created_at, "system_prompt": system_prompt}

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> Any:
        session = session_store.get(session_id)
        if session is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        return session.to_dict()

    @app.post("/v1/sessions/{session_id}/turns")
    async def post_turn(session_id: str, request: Request) -> Any:
        session = session_store.get(session_id)
        if session is None:
            return JSONResponse({"error": "session not found"}, status_code=404)
        ctype = request.headers.get("content-type", "")
        body = await request.body()
        if len(body) > max_request_bytes:
            return JSONResponse({"error": "request too large"}, status_code=413)
        text: str | None = None
        audio: bytes | None = None
        if ctype.startswith("multipart/form-data"):
            fields = parse_multipart(body, ctype)
            audio = fields.get("audio")
            if audio is None:
                return JSONResponse({"error": "multipart field 'audio' required"}, status_code=422)
        else:
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return JSONResponse({"error": "invalid json"}, status_code=400)
            text = data.get("text") if isinstance(data, dict) else None
            if not isinstance(text, str) or not text.strip():
                return JSONResponse({"error": "field 'text' required"}, status_code=422)
        collector = _TurnCollector()

        def work() -> Turn:
            with lock_for(session_id):
                if audio is not None:
                    turn = runtime.run_turn(
                        session, audio, collector.on_sentence, collector.on_audio
                    )
                else:
                    turn = runtime.run_text_turn(
                        session, text or "", collector.on_sentence, collector.on_audio
                    )
                session_store.put(session)
                return turn

        try:
            turn = await asyncio.wait_for(run_in_threadpool(work), timeout=turn_timeout_s)
        except TimeoutError:
            return JSONResponse({"error": "turn timed out"}, status_code=504)
        _record_turn(m, turn)
        return turn_to_json(turn, collector.sentences, collector.audio_ms)

    # ------------------------------------------------------------ streaming

    @app.websocket("/v1/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        if not check_bearer(websocket.headers.get("authorization"), token):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = session_store.get(session_id)
        if session is None:
            await websocket.send_json({"type": "error", "error": "session not found"})
            await websocket.close(code=4404)
            return
        loop = asyncio.get_running_loop()
        try:
            while True:
                msg = await websocket.receive_json()
                text = msg.get("text") if isinstance(msg, dict) else None
                if not isinstance(text, str) or not text.strip():
                    await websocket.send_json({"type": "error", "error": "field 'text' required"})
                    continue
                q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

                def on_sentence(seq: int, sentence: str, q=q) -> None:
                    loop.call_soon_threadsafe(
                        q.put_nowait, {"type": "sentence", "seq": seq, "text": sentence}
                    )

                def on_audio(seq: int, wav: bytes, q=q) -> None:
                    loop.call_soon_threadsafe(
                        q.put_nowait,
                        {
                            "type": "audio",
                            "seq": seq,
                            "bytes_b64": base64.b64encode(wav).decode("ascii"),
                            "sample_rate": getattr(runtime.tts, "sample_rate", None),
                            "ms": round(wav_duration_ms(wav), 3),
                        },
                    )

                def work(text=text) -> Turn:
                    with lock_for(session_id):
                        turn = runtime.run_text_turn(session, text, on_sentence, on_audio)
                        session_store.put(session)
                        return turn

                task = loop.create_task(run_in_threadpool(work))
                task.add_done_callback(lambda _t, q=q: q.put_nowait(None))
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    await websocket.send_json(item)
                try:
                    turn = task.result()
                except Exception as e:  # report to the client, keep the socket open
                    await websocket.send_json({"type": "error", "error": str(e)})
                    continue
                _record_turn(m, turn)
                await websocket.send_json(
                    {
                        "type": "done",
                        "turn": turn.turn,
                        "text": turn.assistant_text,
                        "stages": {k: round(v, 3) for k, v in turn.stages.items()},
                    }
                )
        except WebSocketDisconnect:
            return

    return app


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_TURN_TIMEOUT_S",
    "Metrics",
    "MetricsLike",
    "check_bearer",
    "create_app",
    "parse_multipart",
    "turn_to_json",
]
