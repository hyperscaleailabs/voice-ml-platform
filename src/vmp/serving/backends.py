"""Backend protocols for a voice turn (STT, LLM, TTS) and their implementations.

Reference implementations (`EchoSTT`, `TemplateLLM`, `SilentTTS`) are standard
library only and run in tests and demos. Adapters (`FasterWhisperSTT`,
`OllamaLLM`, `OpenAICompatLLM`, `VLLMEngineLLM`, `KokoroTTS`) import their
dependency lazily inside the class, never at module import.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vmp.types import Utterance

Message = dict[str, str]

# Marker the runtime puts in front of retrieved context inside the system message.
# `TemplateLLM` uses it to decide whether an answer can be grounded.
CONTEXT_MARKER = "Retrieved context:"


@runtime_checkable
class STT(Protocol):
    """Speech to text. `audio` is raw bytes or a path to an audio file."""

    def transcribe(self, audio: bytes | str | Path) -> Utterance: ...


@runtime_checkable
class LLM(Protocol):
    """Chat model. `messages` is a list of `{"role": ..., "content": ...}`."""

    def stream(self, messages: list[Message]) -> Iterator[str]: ...

    def complete(self, messages: list[Message]) -> str: ...


@runtime_checkable
class TTS(Protocol):
    """Text to speech. `synthesize` returns a WAV byte string at `sample_rate`."""

    sample_rate: int

    def synthesize(self, text: str) -> bytes: ...


# --------------------------------------------------------------------------- helpers


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    """Wrap raw little-endian PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_duration_ms(data: bytes) -> float:
    """Duration of a WAV byte string in milliseconds. Returns 0.0 for non-WAV bytes."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
    except (wave.Error, EOFError):
        return 0.0
    return 0.0 if rate == 0 else 1000.0 * frames / rate


def _read_audio(audio: bytes | str | Path) -> bytes:
    if isinstance(audio, bytes):
        return audio
    return Path(audio).read_bytes()


def _last_user_text(messages: list[Message]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _context_of(messages: list[Message]) -> str:
    """Retrieved context inside the system message, or an empty string."""
    for m in messages:
        if m.get("role") == "system" and CONTEXT_MARKER in m.get("content", ""):
            return m["content"].split(CONTEXT_MARKER, 1)[1].strip()
    return ""


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=timeout)


def _get_ok(url: str, headers: dict[str, str], timeout: float) -> bool:
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# --------------------------------------------------------------- reference backends


class EchoSTT:
    """The audio bytes carry UTF-8 text. For tests and demos; no model involved."""

    def __init__(self, speaker: str | None = "user") -> None:
        self.speaker = speaker

    def transcribe(self, audio: bytes | str | Path) -> Utterance:
        text = _read_audio(audio).decode("utf-8", errors="replace").strip()
        return Utterance(
            id=uuid.uuid4().hex,
            text=text,
            speaker=self.speaker,
            duration_s=None,
            meta={"backend": "echo"},
        )

    def ready(self) -> bool:
        return True


_CANNED = (
    "Sure. I can help with that.",
    "Understood. Let me look into it.",
    "Okay. Here is what I know so far.",
    "Got it. Tell me more if you need details.",
)


class TemplateLLM:
    """Rule-based streaming model.

    If the system message carries retrieved context (see `CONTEXT_MARKER`), the
    reply quotes its first line. Otherwise a short canned spoken reply is chosen
    deterministically from the user text. Tokens are words followed by a space;
    `delay_s` between tokens is 0 by default so tests run instantly.
    """

    def __init__(self, delay_s: float = 0.0, max_context_sentence_chars: int = 240) -> None:
        self.delay_s = delay_s
        self.max_context_sentence_chars = max_context_sentence_chars

    def complete(self, messages: list[Message]) -> str:
        return "".join(self.stream(messages))

    def stream(self, messages: list[Message]) -> Iterator[str]:
        text = self._reply(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield word if i == len(words) - 1 else word + " "

    def _reply(self, messages: list[Message]) -> str:
        context = _context_of(messages)
        user = _last_user_text(messages)
        if context:
            first = next((ln.strip() for ln in context.splitlines() if ln.strip()), "")
            first = first[: self.max_context_sentence_chars].rstrip(" .")
            return f"Based on what I found: {first}. Ask me if you want more detail."
        idx = sum(ord(c) for c in user) % len(_CANNED)
        return _CANNED[idx]

    def ready(self) -> bool:
        return True


class SilentTTS:
    """Returns a WAV of silence whose length is proportional to the text length.

    `ms_per_char` approximates spoken pace so latency traces have a plausible
    shape without any model. `delay_s` optionally sleeps to simulate work.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        ms_per_char: float = 60.0,
        min_ms: float = 100.0,
        delay_s: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.min_ms = min_ms
        self.delay_s = delay_s

    def synthesize(self, text: str) -> bytes:
        if self.delay_s:
            time.sleep(self.delay_s)
        ms = max(self.min_ms, self.ms_per_char * len(text))
        frames = int(self.sample_rate * ms / 1000.0)
        return pcm_to_wav(b"\x00\x00" * frames, self.sample_rate)

    def ready(self) -> bool:
        return True


# ------------------------------------------------------------------------- adapters


class FasterWhisperSTT:
    """`faster-whisper` adapter. Imports the package on first use."""

    def __init__(
        self,
        model: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
        beam_size: int = 1,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio: bytes | str | Path) -> Utterance:
        model = self._load()
        source: Any = io.BytesIO(audio) if isinstance(audio, bytes) else str(audio)
        segments, info = model.transcribe(
            source, language=self.language, beam_size=self.beam_size, vad_filter=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return Utterance(
            id=uuid.uuid4().hex,
            text=text,
            audio_path=None if isinstance(audio, bytes) else str(audio),
            duration_s=float(getattr(info, "duration", 0.0) or 0.0),
            meta={"backend": "faster-whisper", "model": self.model_name},
        )

    def ready(self) -> bool:
        try:
            self._load()
        except Exception:
            return False
        return True


class OllamaLLM:
    """Ollama `/api/chat` over `urllib` (streaming NDJSON). No client dependency."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        options: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = options or {}
        self.timeout_s = timeout_s

    def _body(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if self.options:
            body["options"] = self.options
        return body

    def stream(self, messages: list[Message]) -> Iterator[str]:
        with _post_json(
            f"{self.host}/api/chat", self._body(messages, True), {}, self.timeout_s
        ) as r:
            for raw in r:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                piece = row.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if row.get("done"):
                    break

    def complete(self, messages: list[Message]) -> str:
        with _post_json(
            f"{self.host}/api/chat", self._body(messages, False), {}, self.timeout_s
        ) as r:
            row = json.loads(r.read().decode("utf-8"))
        return row.get("message", {}).get("content", "")

    def ready(self) -> bool:
        return _get_ok(f"{self.host}/api/tags", {}, 2.0)


class OpenAICompatLLM:
    """OpenAI-compatible `/v1/chat/completions` over `urllib` with SSE streaming.

    Works against vLLM's OpenAI server and any compatible endpoint.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra = extra or {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _body(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        body.update(self.extra)
        return body

    def stream(self, messages: list[Message]) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        with _post_json(url, self._body(messages, True), self._headers(), self.timeout_s) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                row = json.loads(data)
                choices = row.get("choices") or []
                if not choices:
                    continue
                piece = choices[0].get("delta", {}).get("content")
                if piece:
                    yield piece

    def complete(self, messages: list[Message]) -> str:
        url = f"{self.base_url}/chat/completions"
        with _post_json(url, self._body(messages, False), self._headers(), self.timeout_s) as r:
            row = json.loads(r.read().decode("utf-8"))
        choices = row.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "") or ""

    def ready(self) -> bool:
        return _get_ok(f"{self.base_url}/models", self._headers(), 2.0)


class VLLMEngineLLM:
    """In-process vLLM engine (`vllm.LLM`). Imports `vllm` on first use.

    The offline engine returns the whole completion at once; `stream` yields it
    in word-sized pieces so the segmenter and TTS pipeline behave the same way.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.2,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.engine_kwargs = engine_kwargs or {}
        self._engine: Any = None
        self._params: Any = None

    def _load(self) -> Any:
        if self._engine is None:
            from vllm import LLM as _VLLM
            from vllm import SamplingParams

            self._engine = _VLLM(model=self.model, **self.engine_kwargs)
            self._params = SamplingParams(max_tokens=self.max_tokens, temperature=self.temperature)
        return self._engine

    def complete(self, messages: list[Message]) -> str:
        engine = self._load()
        outputs = engine.chat(messages, self._params)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def stream(self, messages: list[Message]) -> Iterator[str]:
        text = self.complete(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "

    def ready(self) -> bool:
        return self._engine is not None


class KokoroTTS:
    """Kokoro TTS adapter (`kokoro.KPipeline`). Imports `kokoro` and `numpy` on first use."""

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
        sample_rate: int = 24000,
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.sample_rate = sample_rate
        self._pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def synthesize(self, text: str) -> bytes:
        import numpy as np

        pipeline = self._load()
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed)
        ]
        if not chunks:
            return pcm_to_wav(b"", self.sample_rate)
        samples = np.concatenate(chunks)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        return pcm_to_wav(pcm, self.sample_rate)

    def ready(self) -> bool:
        try:
            self._load()
        except Exception:
            return False
        return True


# ---------------------------------------------------------------------- factory


def backends_from_config(config: dict[str, Any]) -> tuple[STT, LLM, TTS]:
    """Build (stt, llm, tts) from the `[serving.backends]` table of `configs/serving.toml`.

    Names: stt `echo | faster-whisper`; llm `template | ollama | openai-compat | vllm`;
    tts `silent | kokoro`. Unknown names raise `ValueError`. Adapters are only
    instantiated (and their dependency only imported on first call), never at
    config time.
    """
    b = config.get("backends", config)
    stt_name = str(b.get("stt", "echo"))
    llm_name = str(b.get("llm", "template"))
    tts_name = str(b.get("tts", "silent"))

    stt: STT
    if stt_name == "echo":
        stt = EchoSTT()
    elif stt_name == "faster-whisper":
        stt = FasterWhisperSTT(**b.get("faster_whisper", {}))
    else:
        raise ValueError(f"unknown stt backend: {stt_name}")

    llm: LLM
    if llm_name == "template":
        llm = TemplateLLM(**b.get("template", {}))
    elif llm_name == "ollama":
        llm = OllamaLLM(**b.get("ollama", {}))
    elif llm_name == "openai-compat":
        llm = OpenAICompatLLM(**b.get("openai_compat", {"model": "default"}))
    elif llm_name == "vllm":
        llm = VLLMEngineLLM(**b.get("vllm", {"model": "default"}))
    else:
        raise ValueError(f"unknown llm backend: {llm_name}")

    tts: TTS
    if tts_name == "silent":
        tts = SilentTTS(**b.get("silent", {}))
    elif tts_name == "kokoro":
        tts = KokoroTTS(**b.get("kokoro", {}))
    else:
        raise ValueError(f"unknown tts backend: {tts_name}")
    return stt, llm, tts


__all__ = [
    "CONTEXT_MARKER",
    "LLM",
    "STT",
    "TTS",
    "EchoSTT",
    "FasterWhisperSTT",
    "KokoroTTS",
    "Message",
    "OllamaLLM",
    "OpenAICompatLLM",
    "SilentTTS",
    "TemplateLLM",
    "VLLMEngineLLM",
    "backends_from_config",
    "pcm_to_wav",
    "wav_duration_ms",
]
