"""The voice turn loop: STT -> retrieve -> LLM -> segment -> TTS -> playback.

`VoiceRuntime` is synchronous and backend-agnostic. TTS runs in one worker
thread per turn so sentence `n` is synthesised while the LLM is still streaming
sentence `n+1`; time to first audio (`playback` payload `response_ms`) is the
headline metric. Traces follow the schema in DESIGN.md through a `TraceSink`
protocol; `ListTraceSink` is the in-memory default so the module is
self-contained. Retrieval goes through a `Retriever` protocol; `NullRetriever`
returns nothing.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vmp.serving.backends import CONTEXT_MARKER, LLM, STT, TTS, Message, wav_duration_ms
from vmp.serving.segment import Segmenter
from vmp.types import Session, Turn, Utterance

DEFAULT_SYSTEM_PROMPT = (
    "You are a voice assistant. Answer in short spoken sentences. "
    "Do not use lists, markdown, or symbols."
)

# ------------------------------------------------------------------ protocols


@runtime_checkable
class Retriever(Protocol):
    """Minimal view of `vmp.rag.HybridRetriever`: `retrieve(query)` -> result with
    `.chunks` (objects with `.text`) and `.context_pack(max_chars) -> str`."""

    def retrieve(self, query: str) -> Any: ...


@runtime_checkable
class TraceSink(Protocol):
    """Minimal view of `vmp.observability.Tracer`: a `span(...)` context manager."""

    def span(
        self, stage: str, session: str, turn: int, seq: int | None = None
    ) -> contextlib.AbstractContextManager[Any]: ...


@runtime_checkable
class SessionStore(Protocol):
    def create(self, system_prompt: str = "", meta: dict[str, Any] | None = None) -> Session: ...

    def get(self, session_id: str) -> Session | None: ...

    def put(self, session: Session) -> None: ...

    def delete(self, session_id: str) -> bool: ...


# ------------------------------------------------------------- defaults: rag


@dataclass(frozen=True)
class NullRetrieval:
    chunks: tuple[Any, ...] = ()

    def context_pack(self, max_chars: int = 0) -> str:
        return ""


class NullRetriever:
    """Retrieves nothing. The runtime then behaves as a plain chat loop."""

    def retrieve(self, query: str) -> NullRetrieval:
        return NullRetrieval()


# ----------------------------------------------------------- defaults: trace


@dataclass
class SpanHandle:
    """What `ListTraceSink.span` yields. Put values in `payload` before the span ends."""

    stage: str
    session: str
    turn: int
    seq: int | None
    span: str
    started: float
    payload: dict[str, Any] = field(default_factory=dict)

    def set(self, **kv: Any) -> None:
        self.payload.update(kv)


class ListTraceSink:
    """In-memory tracer writing rows `ts, session, turn, event, span, seq, ms, payload`.

    `event` is `<stage>.start` / `<stage>.end`. Thread-safe: TTS spans come from
    the worker thread.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows.append(row)

    def _row(self, h: SpanHandle, event: str, ms: float, payload: dict[str, Any]) -> dict:
        return {
            "ts": time.time(),
            "session": h.session,
            "turn": h.turn,
            "event": event,
            "span": h.span,
            "seq": h.seq,
            "ms": round(ms, 3),
            "payload": dict(payload),
        }

    @contextlib.contextmanager
    def span(
        self, stage: str, session: str, turn: int, seq: int | None = None
    ) -> Iterator[SpanHandle]:
        h = SpanHandle(stage, session, turn, seq, uuid.uuid4().hex[:12], time.perf_counter())
        self.emit(self._row(h, f"{stage}.start", 0.0, {}))
        try:
            yield h
        except BaseException as e:
            h.payload.setdefault("error", f"{type(e).__name__}: {e}")
            raise
        finally:
            ms = (time.perf_counter() - h.started) * 1000.0
            self.emit(self._row(h, f"{stage}.end", ms, h.payload))

    def events(self, turn: int | None = None) -> list[str]:
        """Event names in emission order, optionally for one turn."""
        return [r["event"] for r in self.rows if turn is None or r["turn"] == turn]

    def clear(self) -> None:
        with self._lock:
            self.rows.clear()


class JsonlTraceSink(ListTraceSink):
    """`ListTraceSink` that also appends each row to a JSONL file."""

    def __init__(self, path: str | Path, keep_in_memory: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.keep_in_memory = keep_in_memory
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, row: dict[str, Any]) -> None:
        with self._lock:
            if self.keep_in_memory:
                self.rows.append(row)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class NullTraceSink:
    """Discards everything. Spans still yield a handle so payload writes are harmless."""

    @contextlib.contextmanager
    def span(
        self, stage: str, session: str, turn: int, seq: int | None = None
    ) -> Iterator[SpanHandle]:
        yield SpanHandle(stage, session, turn, seq, "", time.perf_counter())


def _set_payload(handle: Any, **kv: Any) -> None:
    """Attach payload to whatever a tracer's `span` yielded (duck typed)."""
    if handle is None:
        return
    payload = getattr(handle, "payload", None)
    if isinstance(payload, dict):
        payload.update(kv)
        return
    for name in ("set", "set_payload", "update"):
        fn = getattr(handle, name, None)
        if callable(fn):
            fn(**kv)
            return


# ------------------------------------------------------------ session stores


class InMemorySessionStore:
    """Dict-backed store with an idle TTL. `clock` is injectable for tests."""

    def __init__(self, ttl_s: float = 3600.0, clock: Callable[[], float] = time.time) -> None:
        self.ttl_s = ttl_s
        self.clock = clock
        self._items: dict[str, tuple[Session, float]] = {}
        self._lock = threading.Lock()

    def _sweep(self, now: float) -> None:
        dead = [k for k, (_, seen) in self._items.items() if now - seen > self.ttl_s]
        for k in dead:
            del self._items[k]

    def create(self, system_prompt: str = "", meta: dict[str, Any] | None = None) -> Session:
        now = self.clock()
        s = Session(id=uuid.uuid4().hex, created_at=now, system_prompt=system_prompt)
        s.meta = dict(meta or {})
        with self._lock:
            self._sweep(now)
            self._items[s.id] = (s, now)
        return s

    def get(self, session_id: str) -> Session | None:
        now = self.clock()
        with self._lock:
            self._sweep(now)
            item = self._items.get(session_id)
            if item is None:
                return None
            self._items[session_id] = (item[0], now)
            return item[0]

    def put(self, session: Session) -> None:
        with self._lock:
            self._items[session.id] = (session, self.clock())

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._items.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._items)


class RedisSessionStore:
    """Sessions as JSON in Redis with an expiry. Imports `redis` on first use."""

    def __init__(self, url: str = "redis://localhost:6379/0", ttl_s: int = 3600,
                 prefix: str = "vmp:session:") -> None:  # fmt: skip
        self.url = url
        self.ttl_s = int(ttl_s)
        self.prefix = prefix
        self._client: Any = None

    def _r(self) -> Any:
        if self._client is None:
            import redis

            self._client = redis.Redis.from_url(self.url)
        return self._client

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def create(self, system_prompt: str = "", meta: dict[str, Any] | None = None) -> Session:
        s = Session(id=uuid.uuid4().hex, created_at=time.time(), system_prompt=system_prompt)
        s.meta = dict(meta or {})
        self.put(s)
        return s

    def get(self, session_id: str) -> Session | None:
        raw = self._r().get(self._key(session_id))
        if raw is None:
            return None
        self._r().expire(self._key(session_id), self.ttl_s)
        return Session.from_dict(json.loads(raw))

    def put(self, session: Session) -> None:
        self._r().set(self._key(session.id), json.dumps(session.to_dict()), ex=self.ttl_s)

    def delete(self, session_id: str) -> bool:
        return bool(self._r().delete(self._key(session_id)))


# ------------------------------------------------------------------- runtime

OnSentence = Callable[[int, str], None]
OnAudio = Callable[[int, bytes], None]
_DONE = object()


class VoiceRuntime:
    """Runs one voice turn end to end and records per-stage milliseconds.

    Stages traced per turn, in order of their `.start` rows:
    `listen` (only when `listener` is set and no audio is given), `stt`,
    `retrieve`, `llm` (payload `ttft_ms`), `segment.emit` (`seq`), `tts` (`seq`),
    `playback` (`seq`; payload `response_ms` on the first), all inside `turn`.

    Conversation memory: the LLM sees the last `max_history_turns` turns of the
    session. Retrieved context is added to the system message for the current
    turn only; it is not written into history.
    """

    def __init__(
        self,
        stt: STT,
        llm: LLM,
        tts: TTS,
        retriever: Retriever | None = None,
        tracer: TraceSink | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history_turns: int = 12,
        max_context_chars: int = 1200,
        listener: Callable[[], bytes] | None = None,
        player: Callable[[bytes], None] | None = None,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.retriever: Retriever = retriever or NullRetriever()
        self.tracer: TraceSink = tracer or NullTraceSink()
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.max_context_chars = max_context_chars
        self.listener = listener
        self.player = player

    # -------------------------------------------------------------- public

    def run_turn(
        self,
        session: Session,
        audio: bytes | str | Path | None = None,
        on_sentence: OnSentence | None = None,
        on_audio: OnAudio | None = None,
    ) -> Turn:
        """Audio in, spoken answer out. `audio=None` uses `listener` (stage `listen`)."""
        turn_no = len(session.turns) + 1
        t0 = time.perf_counter()
        with self.tracer.span("turn", session.id, turn_no) as turn_span:
            stages: dict[str, float] = {}
            if audio is None:
                if self.listener is None:
                    raise ValueError("run_turn needs audio or a listener")
                with self.tracer.span("listen", session.id, turn_no):
                    t = time.perf_counter()
                    audio = self.listener()
                    stages["listen"] = _ms(t)
            with self.tracer.span("stt", session.id, turn_no) as sp:
                t = time.perf_counter()
                utt = self.stt.transcribe(audio)
                stages["stt"] = _ms(t)
                _set_payload(sp, chars=len(utt.text))
            turn = self._respond(session, utt, turn_no, t0, stages, on_sentence, on_audio)
            stages["turn"] = _ms(t0)
            _set_payload(turn_span, **{k: round(v, 3) for k, v in stages.items()})
        return turn

    def run_text_turn(
        self,
        session: Session,
        text: str,
        on_sentence: OnSentence | None = None,
        on_audio: OnAudio | None = None,
    ) -> Turn:
        """Text in (no STT), spoken answer out. Used by the HTTP and WebSocket API."""
        turn_no = len(session.turns) + 1
        t0 = time.perf_counter()
        utt = Utterance(id=uuid.uuid4().hex, text=text.strip(), speaker="user")
        with self.tracer.span("turn", session.id, turn_no) as turn_span:
            stages: dict[str, float] = {}
            turn = self._respond(session, utt, turn_no, t0, stages, on_sentence, on_audio)
            stages["turn"] = _ms(t0)
            _set_payload(turn_span, **{k: round(v, 3) for k, v in stages.items()})
        return turn

    def build_messages(self, session: Session, user_text: str, context: str = "") -> list[Message]:
        """System (+ this turn's context), last N turns, then the user message."""
        system = session.system_prompt or self.system_prompt
        if context:
            system = f"{system}\n\n{CONTEXT_MARKER}\n{context}"
        messages: list[Message] = [{"role": "system", "content": system}]
        history = session.turns[-self.max_history_turns :] if self.max_history_turns > 0 else []
        for t in history:
            messages.append({"role": "user", "content": t.user.text})
            if t.assistant_text:
                messages.append({"role": "assistant", "content": t.assistant_text})
        messages.append({"role": "user", "content": user_text})
        return messages

    # ------------------------------------------------------------ internals

    def _respond(
        self,
        session: Session,
        utt: Utterance,
        turn_no: int,
        t0: float,
        stages: dict[str, float],
        on_sentence: OnSentence | None,
        on_audio: OnAudio | None,
    ) -> Turn:
        sid = session.id
        # retrieve
        retrieved: list[str] = []
        context = ""
        with self.tracer.span("retrieve", sid, turn_no) as sp:
            t = time.perf_counter()
            result = self.retriever.retrieve(utt.text)
            chunks = list(getattr(result, "chunks", []) or [])
            retrieved = [str(getattr(c, "text", c)) for c in chunks]
            pack = getattr(result, "context_pack", None)
            context = pack(self.max_context_chars) if callable(pack) else "\n".join(retrieved)
            stages["retrieve"] = _ms(t)
            _set_payload(sp, n_chunks=len(chunks), context_chars=len(context))
        messages = self.build_messages(session, utt.text, context)

        # tts + playback worker
        q: queue.Queue[Any] = queue.Queue()
        audio_ms: dict[str, float] = {"tts": 0.0, "playback": 0.0}
        first_audio: dict[str, float] = {}
        worker_error: list[BaseException] = []

        def worker() -> None:
            try:
                while True:
                    item = q.get()
                    if item is _DONE:
                        return
                    seq, sentence = item
                    with self.tracer.span("tts", sid, turn_no, seq=seq) as tsp:
                        t = time.perf_counter()
                        wav = self.tts.synthesize(sentence)
                        audio_ms["tts"] += _ms(t)
                        _set_payload(tsp, chars=len(sentence), audio_ms=wav_duration_ms(wav))
                    with self.tracer.span("playback", sid, turn_no, seq=seq) as psp:
                        t = time.perf_counter()
                        if "first_audio" not in first_audio:
                            first_audio["first_audio"] = _ms(t0)
                            _set_payload(psp, response_ms=round(first_audio["first_audio"], 3))
                        if on_audio is not None:
                            on_audio(seq, wav)
                        if self.player is not None:
                            self.player(wav)
                        audio_ms["playback"] += _ms(t)
            except BaseException as e:  # surfaced to the caller after join
                worker_error.append(e)
                while True:  # drain so the producer never blocks
                    if q.get() is _DONE:
                        return

        th = threading.Thread(target=worker, name=f"vmp-tts-{sid[:8]}-{turn_no}", daemon=True)
        th.start()

        # llm + segment
        seg = Segmenter()
        pieces: list[str] = []
        seq = 0
        ttft: float | None = None
        first_emit: float | None = None
        try:
            with self.tracer.span("llm", sid, turn_no) as lsp:
                t_llm = time.perf_counter()

                def emit(sentence: str) -> None:
                    nonlocal seq, first_emit
                    seq += 1
                    with self.tracer.span("segment.emit", sid, turn_no, seq=seq) as esp:
                        since = _ms(t_llm)
                        if first_emit is None:
                            first_emit = since
                        _set_payload(esp, chars=len(sentence), since_llm_ms=round(since, 3))
                        if on_sentence is not None:
                            on_sentence(seq, sentence)
                        q.put((seq, sentence))

                for tok in self.llm.stream(messages):
                    if ttft is None:
                        ttft = _ms(t_llm)
                    pieces.append(tok)
                    for s in seg.feed(tok):
                        emit(s)
                for s in seg.flush():
                    emit(s)
                stages["llm"] = _ms(t_llm)
                _set_payload(
                    lsp,
                    ttft_ms=round(ttft if ttft is not None else stages["llm"], 3),
                    tokens=len(pieces),
                    sentences=seq,
                )
        finally:
            q.put(_DONE)
            th.join()
        if worker_error:
            raise worker_error[0]

        stages["ttft"] = ttft if ttft is not None else stages["llm"]
        stages["segment.emit"] = first_emit if first_emit is not None else stages["llm"]
        stages["tts"] = audio_ms["tts"]
        stages["playback"] = audio_ms["playback"]
        stages["first_audio"] = first_audio.get("first_audio", _ms(t0))

        turn = Turn(
            session_id=sid,
            turn=turn_no,
            user=utt,
            assistant_text="".join(pieces).strip(),
            retrieved=retrieved,
            stages=stages,
        )
        session.turns.append(turn)
        return turn


def _ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "InMemorySessionStore",
    "JsonlTraceSink",
    "ListTraceSink",
    "NullRetrieval",
    "NullRetriever",
    "NullTraceSink",
    "RedisSessionStore",
    "Retriever",
    "SessionStore",
    "SpanHandle",
    "TraceSink",
    "VoiceRuntime",
]
