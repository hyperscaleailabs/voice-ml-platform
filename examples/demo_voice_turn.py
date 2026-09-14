"""A voice agent session end to end, with nothing but the standard library installed.

Runs three turns through `VoiceRuntime` with the reference backends (`EchoSTT`,
`TemplateLLM`, `SilentTTS`) and an in-memory trace sink, then prints per-stage
milliseconds, the time to first audio, and a sample of the raw trace rows.
The third turn is served with retrieved context, so its answer is grounded.

The stub backends are instant, so they sleep for the fixed amounts below to give
the trace a plausible shape. Every millisecond printed here is that simulated
pacing measured by the tracer, not a benchmark of any model.

    python examples/demo_voice_turn.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmp.observability.trace import format_summary, group_by_turn, summarise_turn
from vmp.serving.backends import EchoSTT, SilentTTS, TemplateLLM, wav_duration_ms
from vmp.serving.runtime import InMemorySessionStore, ListTraceSink, VoiceRuntime

# Simulated pacing: per LLM token, and per sentence sent to TTS.
TOKEN_DELAY_S = 0.02
TTS_DELAY_S = 0.05

QUESTIONS = [
    "hello there can you help me",
    "what should I ask you about",
    "how does streaming reduce time to first audio",
]

CONTEXT = [
    "The segmenter splits the token stream into sentences.",
    "Each sentence is synthesised as soon as it is complete, so playback starts early.",
]


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _Retrieval:
    def __init__(self, texts: list[str]) -> None:
        self.chunks = [_Chunk(t) for t in texts]

    def context_pack(self, max_chars: int = 0) -> str:
        return "\n".join(c.text for c in self.chunks)[:max_chars]


class LastTurnRetriever:
    """Returns context for the third question only, to show the one-turn scope."""

    def retrieve(self, query: str) -> _Retrieval:
        return _Retrieval(CONTEXT if "streaming" in query else [])


def main() -> int:
    t0 = time.perf_counter()
    tracer = ListTraceSink()
    store = InMemorySessionStore(ttl_s=300.0)
    runtime = VoiceRuntime(
        EchoSTT(),
        TemplateLLM(delay_s=TOKEN_DELAY_S),
        SilentTTS(delay_s=TTS_DELAY_S),
        retriever=LastTurnRetriever(),
        tracer=tracer,
        max_history_turns=4,
    )
    session = store.create(system_prompt="You are a voice assistant. Answer in short sentences.")
    print(f"session {session.id[:12]}  backends: EchoSTT / TemplateLLM / SilentTTS")
    print(
        f"simulated pacing: {TOKEN_DELAY_S * 1000:.0f} ms per LLM token, "
        f"{TTS_DELAY_S * 1000:.0f} ms per TTS sentence\n"
    )

    for question in QUESTIONS:
        sentences: list[str] = []
        audio_ms = 0.0

        def on_sentence(seq: int, text: str, out: list[str] = sentences) -> None:
            out.append(text)

        def on_audio(seq: int, wav: bytes) -> None:
            nonlocal audio_ms
            audio_ms += wav_duration_ms(wav)

        turn = runtime.run_text_turn(session, question, on_sentence, on_audio)
        store.put(session)

        print(f"turn {turn.turn}  user: {turn.user.text}")
        print(f"          assistant: {turn.assistant_text}")
        if turn.retrieved:
            print(f"          retrieved {len(turn.retrieved)} chunk(s) for this turn only")
        for i, sentence in enumerate(sentences, 1):
            print(f"          sentence {i}: {sentence}")
        stages = "  ".join(
            f"{name}={turn.stages[name]:.2f}ms"
            for name in ("retrieve", "llm", "ttft", "segment.emit", "tts", "playback", "turn")
            if name in turn.stages
        )
        print(f"          stages: {stages}")
        print(
            f"          time to first audio: {turn.stages['first_audio']:.2f} ms"
            f"   synthesised audio: {audio_ms:.0f} ms\n"
        )

    print("per-turn trace summary")
    for (_, number), rows in group_by_turn(tracer.rows).items():
        print(f"  turn {number}: {format_summary(summarise_turn(rows))}")

    print("\nfirst trace rows of turn 3")
    third = group_by_turn(tracer.rows)[(session.id, 3)]
    for row in third[:6]:
        print("  " + json.dumps({k: row[k] for k in ("event", "seq", "ms", "payload")}))
    print(f"\n{len(tracer.rows)} trace rows over {len(session.turns)} turns")
    print(f"done in {time.perf_counter() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
