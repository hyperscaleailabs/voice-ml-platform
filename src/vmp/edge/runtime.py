"""Edge runtime: the voice turn loop on a verified bundle under an `EdgePolicy`.

Backends are duck-typed so the runtime never imports `vmp.serving` at module
import time: an STT has `transcribe(audio) -> str`, an LLM has
`stream(messages) -> Iterator[str]` (or `complete(messages) -> str`), a TTS has
`synthesize(text) -> bytes`. If `vmp.serving.segment` is importable at call
time its segmenter is used, otherwise the local sentence splitter.
"""

from __future__ import annotations

import importlib
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vmp import __version__ as RUNTIME_VERSION
from vmp.edge.bundle import EdgeBundle, verify_bundle
from vmp.edge.policy import ACTION_CLOUD_FALLBACK, Action, EdgePolicy, PolicyViolation, check
from vmp.observability.trace import Tracer, summarise_turn, time_to_first_audio


@runtime_checkable
class CloudFallback(Protocol):
    """A remote completion used only when the policy allows `cloud` fallback."""

    host: str

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class _TracerLike(Protocol):
    def span(
        self, stage: str, session: str, turn: int, seq: int | None = None, payload: Any = None
    ): ...


_SENTENCE = re.compile(r"(.+?[.!?]+)(?:\s+|$)")


def local_segmenter(chunks: Iterable[str]) -> Iterator[str]:
    """Yield complete sentences as text arrives; flush the remainder at the end."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        while True:
            m = _SENTENCE.match(buf)
            if not m:
                break
            yield m.group(1).strip()
            buf = buf[m.end() :]
    tail = buf.strip()
    if tail:
        yield tail


def resolve_segmenter() -> Callable[[Iterable[str]], Iterator[str]]:
    """Prefer `vmp.serving.segment.segment` when present (looked up at call time)."""
    try:
        mod = importlib.import_module("vmp.serving.segment")
    except Exception:
        return local_segmenter
    for name in ("segment", "segment_stream", "iter_sentences"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return local_segmenter


@dataclass
class EdgeTurn:
    session: str
    turn: int
    transcript: str
    assistant_text: str
    segments: list[str] = field(default_factory=list)
    audio: list[bytes] = field(default_factory=list)
    ttfa_ms: float | None = None
    stages: dict[str, float] = field(default_factory=dict)
    refused: bool = False
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session,
            "turn": self.turn,
            "transcript": self.transcript,
            "assistant_text": self.assistant_text,
            "segments": list(self.segments),
            "audio_bytes": sum(len(a) for a in self.audio),
            "ttfa_ms": self.ttfa_ms,
            "stages": dict(self.stages),
            "refused": self.refused,
            "fallback_used": self.fallback_used,
        }


class EdgeRuntime:
    """Runs turns against local backends; verifies the bundle on start."""

    def __init__(
        self,
        bundle_dir: str | Path,
        stt: Any,
        llm: Any,
        tts: Any,
        policy: EdgePolicy | None = None,
        *,
        tracer: Any = None,
        cloud: CloudFallback | None = None,
        system_prompt: str = "You are a voice assistant. Answer in short spoken sentences.",
        runtime_version: str = RUNTIME_VERSION,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.cloud = cloud
        self.system_prompt = system_prompt
        self.runtime_version = runtime_version
        self.tracer = tracer if tracer is not None else Tracer()
        self.bundle: EdgeBundle | None = None
        self.policy = policy
        self.verified = False
        self.problems: list[str] = []
        self.last_ttfa_ms: float | None = None
        self.turns = 0
        self.refusals = 0
        self.errors = 0
        self._started_at: float | None = None
        self._history: list[dict[str, str]] = []

    # -- lifecycle --------------------------------------------------------

    def start(self, *, require_verified: bool = True) -> bool:
        self.bundle = EdgeBundle.load(self.bundle_dir)
        self.verified, self.problems = verify_bundle(
            self.bundle_dir, runtime_version=self.runtime_version
        )
        if self.policy is None:
            self.policy = self.bundle.policy()
        pol_problems = self.policy.validate()
        if pol_problems:
            self.problems.extend(f"policy: {p}" for p in pol_problems)
            self.verified = False
        if require_verified and not self.verified:
            raise RuntimeError("bundle verification failed: " + "; ".join(self.problems))
        self._started_at = time.time()
        return self.verified

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.verified else "degraded",
            "bundle": self.bundle.name if self.bundle else None,
            "bundle_version": self.bundle.version if self.bundle else None,
            "runtime_version": self.runtime_version,
            "verified": self.verified,
            "problems": list(self.problems),
            "last_ttfa_ms": self.last_ttfa_ms,
            "turns": self.turns,
            "refusals": self.refusals,
            "errors": self.errors,
            "offline_only": self.policy.offline_only if self.policy else None,
            "uptime_s": (time.time() - self._started_at) if self._started_at else None,
        }

    # -- turn loop --------------------------------------------------------

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self._history,
            {"role": "user", "content": user_text},
        ]

    def _stream_llm(self, messages: list[dict[str, str]]) -> Iterator[str]:
        stream = getattr(self.llm, "stream", None)
        if callable(stream):
            yield from stream(messages)
            return
        yield str(self.llm.complete(messages))

    def _fallback(self, messages: list[dict[str, str]], reason: str) -> tuple[str, bool, bool]:
        """Return `(text, refused, fallback_used)` after a local LLM failure."""
        assert self.policy is not None
        if self.policy.fallback == "refuse":
            return "", True, False
        if self.policy.fallback == "degrade":
            return "I cannot answer that right now.", False, False
        host = getattr(self.cloud, "host", None) if self.cloud is not None else None
        allowed, why = check(self.policy, Action(ACTION_CLOUD_FALLBACK, host, reason))
        if not allowed or self.cloud is None:
            self.refusals += 1
            raise PolicyViolation(why if not allowed else "no cloud fallback configured")
        allowed, why = check(self.policy, Action("network", host, reason))
        if not allowed:
            self.refusals += 1
            raise PolicyViolation(why)
        return str(self.cloud.complete(messages)), False, True

    def turn(self, audio: bytes | str, session: str, turn_no: int | None = None) -> EdgeTurn:
        """One voice turn: stt -> llm stream -> segment -> tts, traced per stage."""
        if self.bundle is None:
            self.start()
        assert self.policy is not None
        n = self.turns + 1 if turn_no is None else turn_no
        tr = self.tracer
        sink_rows = getattr(getattr(tr, "sink", None), "rows", None)
        row_start = len(sink_rows) if sink_rows is not None else 0
        segmenter = resolve_segmenter()  # resolved before t0: an import must not count as latency
        t0 = time.perf_counter()
        result = EdgeTurn(session=session, turn=n, transcript="", assistant_text="")
        with tr.span("turn", session, n) as turn_payload:
            with tr.span("stt", session, n):
                result.transcript = str(self.stt.transcribe(audio))
            messages = self._messages(result.transcript)
            pieces: list[str] = []
            seq = 0
            first_audio_at: float | None = None

            def llm_chunks() -> Iterator[str]:
                first = True
                with tr.span("llm", session, n) as p:
                    for chunk in self._stream_llm(messages):
                        if first:
                            p["ttft_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
                            first = False
                        pieces.append(chunk)
                        yield chunk

            try:
                for sentence in segmenter(llm_chunks()):
                    seq += 1
                    with tr.span("segment.emit", session, n, seq=seq):
                        result.segments.append(sentence)
                    with tr.span("tts", session, n, seq=seq):
                        clip = self.tts.synthesize(sentence)
                    with tr.span("playback", session, n, seq=seq) as pb:
                        result.audio.append(clip)
                        if first_audio_at is None:
                            first_audio_at = time.perf_counter()
                            pb["response_ms"] = round((first_audio_at - t0) * 1000.0, 3)
                result.assistant_text = "".join(pieces)
            except PolicyViolation:
                raise
            except Exception as e:
                self.errors += 1
                turn_payload["error"] = type(e).__name__
                text, refused, used = self._fallback(messages, type(e).__name__)
                result.assistant_text = text
                result.refused = refused
                result.fallback_used = used
                if refused:
                    self.refusals += 1
            if result.refused:
                turn_payload["refused"] = True
            result.ttfa_ms = (
                round((first_audio_at - t0) * 1000.0, 3) if first_audio_at is not None else None
            )
            # Written inside the span so it reaches the `turn.end` payload.
            if result.ttfa_ms is not None and result.ttfa_ms > self.policy.max_ttfa_ms:
                turn_payload["over_max_ttfa"] = True
        self.last_ttfa_ms = result.ttfa_ms
        self.turns += 1
        if result.assistant_text and not result.refused:
            self._history.append({"role": "user", "content": result.transcript})
            self._history.append({"role": "assistant", "content": result.assistant_text})
        if sink_rows is not None:
            rows = sink_rows[row_start:]
            result.stages = {k: v["total_ms"] for k, v in summarise_turn(rows).items()}
            traced = time_to_first_audio(rows)
            if traced is not None:
                result.ttfa_ms = traced
        return result


__all__ = [
    "CloudFallback",
    "EdgeRuntime",
    "EdgeTurn",
    "local_segmenter",
    "resolve_segmenter",
]
