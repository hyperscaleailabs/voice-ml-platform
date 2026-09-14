"""Feature store demo. Runs with the standard library only.

1. Generate synthetic stage-trace events for a few voice sessions.
2. Stream them into `session_features` (in-memory online store).
3. Persist the per-turn rows offline (JSONL), then materialise offline -> online.
4. Do a point-in-time lookup: what did we know about each session mid-conversation?
5. Print a short table.

Run: `python examples/demo_features.py`
"""

from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vmp.features import (
    SESSION_FEATURES,
    InMemoryOnlineStore,
    JsonlOfflineStore,
    SessionFeatureUpdater,
    materialize,
)

INTENTS = ("weather", "timer", "music", "question", "smalltalk")
ACCENTS = ("en-IE", "en-IN", "en-US")


def synthetic_trace(n_sessions: int = 4, turns: int = 5, seed: int = 7) -> list[dict]:
    """Emit `<stage>.start/.end` events with the DESIGN.md keys, one turn at a time."""
    rng = random.Random(seed)
    t = 1_700_000_000.0
    events: list[dict] = []
    seq = 0

    def emit(session: str, turn: int, event: str, ms: float | None, payload: dict) -> None:
        nonlocal t, seq
        seq += 1
        events.append(
            {
                "ts": t,
                "session": session,
                "turn": turn,
                "event": event,
                "span": event.split(".")[0],
                "seq": seq,
                "ms": ms,
                "payload": payload,
            }
        )

    for s in range(n_sessions):
        sid = f"sess-{s:02d}"
        accent = ACCENTS[s % len(ACCENTS)]
        for turn in range(1, turns + 1):
            listen_ms = rng.uniform(800, 3200)
            stt_ms = rng.uniform(150, 400)
            llm_ms = rng.uniform(200, 900)
            tts_ms = rng.uniform(100, 300)
            emit(sid, turn, "listen.start", None, {})
            t += listen_ms / 1000
            emit(sid, turn, "listen.end", listen_ms, {})
            t += stt_ms / 1000
            emit(sid, turn, "stt.end", stt_ms, {"accent": accent})
            t += llm_ms / 1000
            intent = rng.choice(INTENTS)
            emit(sid, turn, "llm.end", llm_ms, {"ttft_ms": llm_ms * 0.4, "intent": intent})
            t += tts_ms / 1000
            emit(sid, turn, "tts.end", tts_ms, {})
            response_ms = stt_ms + llm_ms * 0.4 + tts_ms
            emit(sid, turn, "playback.end", response_ms, {"response_ms": response_ms})
            emit(sid, turn, "turn.end", listen_ms + stt_ms + llm_ms + tts_ms, {})
            t += rng.uniform(0.5, 2.0)
    return events


def fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main() -> int:
    events = synthetic_trace()
    online = InMemoryOnlineStore()
    updater = SessionFeatureUpdater(online=online)
    summary = updater.run(events)
    print(f"streamed {summary['events_seen']} events -> {summary['turns']} turns "
          f"in {summary['sessions']} sessions")

    with tempfile.TemporaryDirectory() as tmp:
        offline = JsonlOfflineStore(tmp)
        offline.write(SESSION_FEATURES, updater.rows)
        # "Now" for TTL checks is the end of the trace, not wall-clock time; the
        # synthetic events carry fixed timestamps.
        trace_end = max(e["ts"] for e in events)
        online2 = InMemoryOnlineStore(clock=lambda: trace_end)
        manifest = materialize(SESSION_FEATURES, offline, online2)
        print(f"materialized {manifest['entities_written']} entities "
              f"from {manifest['rows_read']} offline rows")

        # Point-in-time: ask for each session's state at the timestamp of its third turn.
        # The join must return turn_count == 3, never the final count.
        third = {r.entity_id: r.event_ts for r in updater.rows if r.values["turn_count"] == 3}
        requests = [(sid, ts) for sid, ts in sorted(third.items())]
        pit = offline.point_in_time(SESSION_FEATURES, requests)

    cols = ("session", "turns@t3", "turns@now", "avg_utt_s", "avg_resp_ms", "intent", "accent")
    widths = (10, 9, 9, 10, 12, 10, 7)
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True)))
    for row in pit:
        now = online2.get(SESSION_FEATURES.name, row.entity_id) or {}
        vals = (
            row.entity_id,
            row.values["turn_count"],
            now.get("turn_count"),
            row.values["avg_user_utterance_s"],
            row.values["avg_response_ms"],
            row.values["last_intent"],
            row.values["accent_profile"],
        )
        print("  ".join(fmt(v).ljust(w) for v, w in zip(vals, widths, strict=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
