"""Voice agent serving: backends, sentence segmenter, turn runtime, HTTP/WebSocket API.

Standard library only at import time. `fastapi`, `uvicorn`, `ray`, `vllm`,
`faster-whisper` and `kokoro` are imported lazily where they are used.
"""

from __future__ import annotations

from vmp.serving.backends import (
    LLM,
    STT,
    TTS,
    EchoSTT,
    SilentTTS,
    TemplateLLM,
    backends_from_config,
    pcm_to_wav,
    wav_duration_ms,
)
from vmp.serving.runtime import (
    InMemorySessionStore,
    ListTraceSink,
    NullRetriever,
    Retriever,
    SessionStore,
    TraceSink,
    VoiceRuntime,
)
from vmp.serving.segment import Segmenter, segment

__all__ = [
    "LLM",
    "STT",
    "TTS",
    "EchoSTT",
    "InMemorySessionStore",
    "ListTraceSink",
    "NullRetriever",
    "Retriever",
    "Segmenter",
    "SessionStore",
    "SilentTTS",
    "TemplateLLM",
    "TraceSink",
    "VoiceRuntime",
    "backends_from_config",
    "pcm_to_wav",
    "segment",
    "wav_duration_ms",
]
