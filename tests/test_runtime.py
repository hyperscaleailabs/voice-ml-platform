"""Voice turn loop tests: stage order, time to first audio, history, sessions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vmp.serving.backends import CONTEXT_MARKER, EchoSTT, SilentTTS, TemplateLLM, wav_duration_ms
from vmp.serving.runtime import (
    DEFAULT_SYSTEM_PROMPT,
    InMemorySessionStore,
    JsonlTraceSink,
    ListTraceSink,
    NullRetriever,
    NullTraceSink,
    Retriever,
    SessionStore,
    VoiceRuntime,
)
from vmp.types import Session, Turn


class RecordingLLM:
    """Streams a fixed answer and records the messages it was given."""

    def __init__(self, answer: str = "First sentence. Second sentence. Third one.") -> None:
        self.answer = answer
        self.seen: list[list[dict[str, str]]] = []

    def stream(self, messages):
        self.seen.append([dict(m) for m in messages])
        words = self.answer.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "

    def complete(self, messages) -> str:
        return "".join(self.stream(messages))


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _Result:
    def __init__(self, texts: list[str]) -> None:
        self.chunks = [_Chunk(t) for t in texts]

    def context_pack(self, max_chars: int = 0) -> str:
        return "\n".join(c.text for c in self.chunks)[:max_chars]


class FirstTurnRetriever:
    """Returns context on the first call only, nothing afterwards."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls = 0

    def retrieve(self, query: str) -> _Result:
        self.calls += 1
        return _Result(self.texts if self.calls == 1 else [])


def _runtime(**kw) -> tuple[VoiceRuntime, RecordingLLM, ListTraceSink]:
    llm = RecordingLLM()
    sink = ListTraceSink()
    kw.setdefault("tracer", sink)
    return VoiceRuntime(EchoSTT(), llm, SilentTTS(), **kw), llm, sink


def _session(sid: str = "s1") -> Session:
    return Session(id=sid, created_at=time.time())


def _first(sink: ListTraceSink, event: str) -> dict:
    return next(r for r in sink.rows if r["event"] == event)


# -- protocols --------------------------------------------------------------


def test_reference_backends_satisfy_the_protocols():
    assert isinstance(NullRetriever(), Retriever)
    assert isinstance(InMemorySessionStore(), SessionStore)


# -- stage order and presence ----------------------------------------------


def test_turn_traces_every_stage_in_pipeline_order():
    rt, _, sink = _runtime()
    turn = rt.run_turn(_session(), b"tell me about streaming")

    starts = [e for e in sink.events(1) if e.endswith(".start")]
    assert starts[0] == "turn.start"
    assert sink.events(1)[-1] == "turn.end"
    for stage in ("stt", "retrieve", "llm", "segment.emit", "tts", "playback"):
        assert f"{stage}.start" in starts, stage
        assert f"{stage}.end" in sink.events(1), stage
    # STT before retrieval before the LLM.
    assert starts.index("stt.start") < starts.index("retrieve.start") < starts.index("llm.start")
    # Every stage the turn measured is in the trace and in the Turn.
    for stage in ("stt", "retrieve", "llm", "segment.emit", "tts", "playback", "turn"):
        assert stage in turn.stages
    assert all(v >= 0.0 for v in turn.stages.values())


def test_turn_span_payload_carries_the_stage_milliseconds():
    rt, _, sink = _runtime()
    turn = rt.run_turn(_session(), b"hello")
    end = _first(sink, "turn.end")
    assert set(turn.stages) <= set(end["payload"])
    assert end["ms"] > 0.0
    assert end["seq"] is None


def test_llm_span_payload_has_ttft_and_sentence_count():
    rt, _, sink = _runtime()
    rt.run_turn(_session(), b"hello")
    llm_end = _first(sink, "llm.end")
    assert "ttft_ms" in llm_end["payload"]
    assert llm_end["payload"]["sentences"] == 3
    assert llm_end["payload"]["tokens"] > 0


def test_segment_and_tts_spans_are_numbered_per_sentence():
    rt, _, sink = _runtime()
    rt.run_turn(_session(), b"hello")
    for stage in ("segment.emit", "tts", "playback"):
        seqs = [r["seq"] for r in sink.rows if r["event"] == f"{stage}.end"]
        assert seqs == [1, 2, 3], stage


def test_listen_stage_only_when_the_listener_is_used():
    rt, _, sink = _runtime(listener=lambda: b"spoken words")
    turn = rt.run_turn(_session())
    assert "listen.start" in sink.events(1)
    assert turn.stages["listen"] >= 0.0
    assert turn.user.text == "spoken words"

    sink.clear()
    rt.run_turn(_session("s2"), b"typed words")
    assert "listen.start" not in sink.events()


def test_run_turn_without_audio_or_listener_raises():
    rt, _, _ = _runtime()
    with pytest.raises(ValueError):
        rt.run_turn(_session())


def test_text_turn_skips_stt():
    rt, _, sink = _runtime()
    turn = rt.run_text_turn(_session(), "  written question  ")
    assert "stt.start" not in sink.events(1)
    assert "stt" not in turn.stages
    assert turn.user.text == "written question"
    assert turn.user.speaker == "user"


# -- time to first audio ----------------------------------------------------


def test_time_to_first_audio_is_measured_and_recorded_once():
    rt, _, sink = _runtime()
    turn = rt.run_turn(_session(), b"hello")
    playback_ends = [r for r in sink.rows if r["event"] == "playback.end"]
    with_response = [r for r in playback_ends if "response_ms" in r["payload"]]
    assert len(with_response) == 1
    assert with_response[0]["seq"] == 1
    response_ms = with_response[0]["payload"]["response_ms"]
    assert response_ms > 0.0
    assert turn.stages["first_audio"] == pytest.approx(response_ms, rel=0.05)
    assert turn.stages["first_audio"] <= turn.stages["turn"]


def test_first_audio_beats_the_whole_turn_when_the_llm_is_slow():
    """Streaming means audio starts before the last token arrives."""

    class SlowTailLLM:
        def stream(self, messages):
            yield "Answer ready. "
            time.sleep(0.05)
            yield "A slow tail follows."

        def complete(self, messages):
            return "".join(self.stream(messages))

    sink = ListTraceSink()
    rt = VoiceRuntime(EchoSTT(), SlowTailLLM(), SilentTTS(), tracer=sink)
    turn = rt.run_turn(_session(), b"hello")
    assert turn.stages["first_audio"] < turn.stages["turn"]
    assert turn.stages["turn"] >= 50.0


# -- streaming callbacks ----------------------------------------------------


def test_streaming_emits_sentence_by_sentence_before_the_turn_ends():
    sentences: list[tuple[int, str]] = []
    audio: list[tuple[int, bytes]] = []
    rt, _, _ = _runtime()
    turn = rt.run_turn(
        _session(),
        b"hello",
        on_sentence=lambda seq, text: sentences.append((seq, text)),
        on_audio=lambda seq, wav: audio.append((seq, wav)),
    )
    assert [s for _, s in sentences] == [
        "First sentence.",
        "Second sentence.",
        "Third one.",
    ]
    assert [seq for seq, _ in sentences] == [1, 2, 3]
    assert [seq for seq, _ in audio] == [1, 2, 3]
    assert all(wav.startswith(b"RIFF") for _, wav in audio)
    # Longer sentences produce longer clips, so the TTS actually saw each sentence.
    assert wav_duration_ms(audio[0][1]) > wav_duration_ms(audio[2][1])
    assert turn.assistant_text == " ".join(s for _, s in sentences)


def test_player_receives_every_clip():
    played: list[bytes] = []
    rt, _, _ = _runtime(player=played.append)
    rt.run_turn(_session(), b"hello")
    assert len(played) == 3


def test_worker_error_is_raised_to_the_caller():
    class BoomTTS:
        sample_rate = 24000

        def synthesize(self, text: str) -> bytes:
            raise RuntimeError("tts exploded")

    rt = VoiceRuntime(EchoSTT(), RecordingLLM(), BoomTTS(), tracer=ListTraceSink())
    with pytest.raises(RuntimeError, match="tts exploded"):
        rt.run_turn(_session(), b"hello")


def test_span_records_the_error_and_reraises():
    class BoomLLM:
        def stream(self, messages):
            raise RuntimeError("llm exploded")
            yield ""  # pragma: no cover - unreachable, keeps this a generator

        def complete(self, messages):
            return ""

    sink = ListTraceSink()
    rt = VoiceRuntime(EchoSTT(), BoomLLM(), SilentTTS(), tracer=sink)
    with pytest.raises(RuntimeError, match="llm exploded"):
        rt.run_turn(_session(), b"hello")
    llm_end = _first(sink, "llm.end")
    assert "RuntimeError" in llm_end["payload"]["error"]
    assert sink.rows[-1]["event"] == "turn.end"


# -- history ----------------------------------------------------------------


def test_history_window_is_capped():
    rt, llm, _ = _runtime(max_history_turns=2)
    session = _session()
    for i in range(5):
        rt.run_text_turn(session, f"question {i}")
    assert len(session.turns) == 5
    last = llm.seen[-1]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in last if m["role"] == "user"] == [
        "question 2",
        "question 3",
        "question 4",
    ]


def test_zero_history_sends_only_the_current_turn():
    rt, llm, _ = _runtime(max_history_turns=0)
    session = _session()
    rt.run_text_turn(session, "one")
    rt.run_text_turn(session, "two")
    assert [m["role"] for m in llm.seen[-1]] == ["system", "user"]
    assert llm.seen[-1][-1]["content"] == "two"


def test_turn_numbering_and_session_accumulation():
    rt, _, _ = _runtime()
    session = _session()
    numbers = [rt.run_text_turn(session, f"q{i}").turn for i in range(3)]
    assert numbers == [1, 2, 3]
    assert [t.turn for t in session.turns] == [1, 2, 3]
    assert all(isinstance(t, Turn) for t in session.turns)


def test_session_system_prompt_overrides_the_runtime_default():
    rt, llm, _ = _runtime()
    session = Session(id="s", created_at=0.0, system_prompt="Speak like a pirate.")
    rt.run_text_turn(session, "hello")
    assert llm.seen[0][0]["content"] == "Speak like a pirate."
    rt.run_text_turn(_session("other"), "hello")
    assert llm.seen[1][0]["content"] == DEFAULT_SYSTEM_PROMPT


# -- retrieval --------------------------------------------------------------


def test_retrieved_context_applies_to_one_turn_only():
    retriever = FirstTurnRetriever(["Streaming reduces time to first audio."])
    rt, llm, _ = _runtime(retriever=retriever)
    session = _session()

    first = rt.run_text_turn(session, "how does streaming work")
    second = rt.run_text_turn(session, "and then")

    assert first.retrieved == ["Streaming reduces time to first audio."]
    assert second.retrieved == []
    assert CONTEXT_MARKER in llm.seen[0][0]["content"]
    assert "Streaming reduces" in llm.seen[0][0]["content"]
    # The second turn's system message is clean, and history never carries context.
    assert CONTEXT_MARKER not in llm.seen[1][0]["content"]
    assert all(CONTEXT_MARKER not in m["content"] for m in llm.seen[1][1:])
    assert retriever.calls == 2


def test_context_is_truncated_to_max_context_chars():
    retriever = FirstTurnRetriever(["x" * 500])
    rt, llm, sink = _runtime(retriever=retriever, max_context_chars=50)
    rt.run_text_turn(_session(), "question")
    system = llm.seen[0][0]["content"]
    assert len(system.split(CONTEXT_MARKER)[1].strip()) == 50
    retrieve_end = _first(sink, "retrieve.end")
    assert retrieve_end["payload"] == {"n_chunks": 1, "context_chars": 50}


def test_null_retriever_leaves_the_prompt_alone():
    rt, llm, _ = _runtime()
    turn = rt.run_text_turn(_session(), "question")
    assert turn.retrieved == []
    assert CONTEXT_MARKER not in llm.seen[0][0]["content"]


def test_build_messages_shape():
    rt, _, _ = _runtime(max_history_turns=12)
    session = _session()
    rt.run_text_turn(session, "first")
    messages = rt.build_messages(session, "second", context="some context")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].endswith("some context")
    assert messages[-1] == {"role": "user", "content": "second"}


# -- session store ----------------------------------------------------------


def test_session_store_crud():
    store = InMemorySessionStore()
    session = store.create(system_prompt="p", meta={"client": "cli"})
    assert store.get(session.id) is session
    assert session.system_prompt == "p"
    assert session.meta == {"client": "cli"}
    assert store.get("missing") is None
    assert store.delete(session.id) is True
    assert store.delete(session.id) is False
    assert len(store) == 0


def test_session_store_ttl_expires_idle_sessions():
    now = [1_000.0]
    store = InMemorySessionStore(ttl_s=10.0, clock=lambda: now[0])
    session = store.create()

    now[0] += 5.0
    assert store.get(session.id) is not None  # a read refreshes the deadline
    now[0] += 9.0
    assert store.get(session.id) is not None
    now[0] += 11.0
    assert store.get(session.id) is None
    assert len(store) == 0


def test_session_store_put_refreshes_the_deadline():
    now = [0.0]
    store = InMemorySessionStore(ttl_s=10.0, clock=lambda: now[0])
    session = store.create()
    now[0] += 9.0
    store.put(session)
    now[0] += 9.0
    assert store.get(session.id) is not None


def test_session_round_trips_through_json():
    rt, _, _ = _runtime()
    store = InMemorySessionStore()
    session = store.create(system_prompt="p")
    rt.run_text_turn(session, "hello")
    store.put(session)
    restored = Session.from_dict(json.loads(json.dumps(session.to_dict())))
    assert restored.id == session.id
    assert restored.turns[0].assistant_text == session.turns[0].assistant_text
    assert restored.turns[0].user.text == "hello"


# -- trace sinks ------------------------------------------------------------


def test_jsonl_trace_sink_writes_one_row_per_line(tmp_path: Path):
    path = tmp_path / "traces" / "run.jsonl"
    sink = JsonlTraceSink(path)
    rt = VoiceRuntime(EchoSTT(), TemplateLLM(), SilentTTS(), tracer=sink)
    rt.run_turn(_session(), b"hello there")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == sink.rows
    assert {"ts", "session", "turn", "event", "span", "seq", "ms", "payload"} <= set(rows[0])
    assert rows[0]["event"] == "turn.start"
    assert rows[-1]["event"] == "turn.end"


def test_null_trace_sink_runs_the_turn_without_recording():
    rt = VoiceRuntime(EchoSTT(), TemplateLLM(), SilentTTS(), tracer=NullTraceSink())
    turn = rt.run_turn(_session(), b"hello there")
    assert turn.assistant_text
    assert turn.stages["turn"] > 0.0


def test_default_tracer_is_a_null_sink():
    rt = VoiceRuntime(EchoSTT(), TemplateLLM(), SilentTTS())
    assert isinstance(rt.tracer, NullTraceSink)
    assert rt.run_turn(_session(), b"hi").assistant_text
