# API

The voice agent service exposes a small HTTP surface for session management and
a WebSocket for streaming a turn. It is built by
`vmp.serving.api.create_app(runtime, session_store, ...)`, which imports FastAPI
lazily, and it is run by [`vmp serve api`](guides/serving-api.md). The Ray Serve
graph serves the same contract.

```bash
vmp serve api --config configs/serving.toml           # 127.0.0.1:8080 by default
```

Base URL below is `http://localhost:8080`.

## Authentication

Bearer token in the `Authorization` header:

```text
Authorization: Bearer <token>
```

The token comes from the `VMP_API_TOKEN` environment variable. **Unset or empty
disables authentication entirely** — correct for local development, wrong
anywhere else. Comparison is constant-time. Never put a token in a config file.

`/healthz`, `/readyz` and `/metrics` are exempt so an orchestrator and a scraper
need no credentials; expose them on a cluster-internal port only.

A rejected request returns `401` with `{"error": "unauthorized"}`. On the
WebSocket, a bad token closes the socket with code `1008` before the handshake
completes.

## Conventions

| | |
|---|---|
| Content type | `application/json`, except `POST /turns` with audio, which is `multipart/form-data` |
| Request id | `X-Request-ID` is honoured if supplied, generated otherwise, and echoed on every response |
| Size limit | `max_request_bytes` (10 MiB default); exceeded returns `413` |
| Turn timeout | `turn_timeout_s` (60 s default); exceeded returns `504` |
| Concurrency | one lock per session — turns in a session serialise, different sessions run in parallel |
| Errors | `{"error": "<message>"}` with the status code |

Status codes in use: `200`, `201`, `400` (invalid JSON), `401`, `404`
(unknown session), `413`, `422` (missing or empty required field), `504`.

## `POST /v1/sessions`

Create a session. Body is optional.

```json
{"system_prompt": "You are a voice assistant.", "meta": {"client": "demo"}}
```

**`201 Created`**

```json
{
  "id": "1ea0c666295a48ccac400e40ae44769f",
  "created_at": 1789401268.796044,
  "system_prompt": "You are a voice assistant."
}
```

An empty or absent `system_prompt` means the runtime's configured default is
used. Sessions live in the session store and expire after `session_ttl_s`
(3600 s default). With the in-memory store they are **per replica**: see the
pitfalls in the [serving guide](guides/serving-api.md).

## `GET /v1/sessions/{id}`

The whole session, including every turn.

**`200 OK`**

```json
{
  "id": "1ea0c666295a48ccac400e40ae44769f",
  "created_at": 1789401268.796044,
  "system_prompt": "You are a voice assistant.",
  "turns": [
    {
      "session_id": "1ea0c666295a48ccac400e40ae44769f",
      "turn": 1,
      "user": {
        "id": "b480f5a295e5451ba2781d2b88513f38",
        "text": "what is the time to first audio",
        "audio_path": null,
        "speaker": "user",
        "duration_s": null,
        "meta": {}
      },
      "assistant_text": "Understood. Let me look into it.",
      "retrieved": [],
      "stages": {
        "retrieve": 0.0035, "llm": 0.0339, "ttft": 0.0079,
        "segment.emit": 0.0167, "tts": 0.0785, "playback": 0.0133,
        "first_audio": 0.2681, "turn": 0.3230
      }
    }
  ],
  "meta": {}
}
```

**`404 Not Found`** — `{"error": "session not found"}`.

The `stages` map is milliseconds per stage, unrounded here and rounded in the
turn response. `first_audio` is time to first audio for the turn.

## `POST /v1/sessions/{id}/turns`

Run one complete turn and return it. Two request shapes.

**Text** (`application/json`):

```json
{"text": "what is the time to first audio"}
```

**Audio** (`multipart/form-data`) with a field named `audio` carrying a WAV
body. A missing `audio` field returns `422`.

**`200 OK`**

```json
{
  "session_id": "1ea0c666295a48ccac400e40ae44769f",
  "turn": 1,
  "user": {
    "id": "b480f5a295e5451ba2781d2b88513f38",
    "text": "what is the time to first audio",
    "audio_path": null,
    "speaker": "user",
    "duration_s": null,
    "meta": {}
  },
  "assistant_text": "Understood. Let me look into it.",
  "retrieved": [],
  "stages": {
    "retrieve": 0.004, "llm": 0.034, "ttft": 0.008, "segment.emit": 0.017,
    "tts": 0.079, "playback": 0.013, "first_audio": 0.268, "turn": 0.323
  },
  "sentences": ["Understood.", "Let me look into it."],
  "audio_ms": 1860.0
}
```

`sentences` is what the segmenter emitted, in order; `audio_ms` is the total
duration of synthesised audio. This endpoint **waits for the whole turn**, so
the caller experiences the serial latency even though the runtime streamed
internally. For the streaming benefit, use the WebSocket.

Errors: `404` unknown session, `422` missing or empty `text`, `400` invalid
JSON, `413` body too large, `504` turn exceeded `turn_timeout_s`.

## `WS /v1/sessions/{id}/stream`

The streaming path. The client sends one JSON message per turn and receives a
stream of messages, then sends the next one on the same socket.

**Client → server**

```json
{"text": "tell me about ray serve"}
```

**Server → client**, in the order the runtime produces them:

| `type` | Fields | Meaning |
|---|---|---|
| `sentence` | `seq`, `text` | The segmenter completed sentence `seq` |
| `audio` | `seq`, `bytes_b64`, `sample_rate`, `ms` | Synthesised audio for sentence `seq`, base64 WAV |
| `done` | `turn`, `text`, `stages` | The turn finished; `stages` is the timing map |
| `error` | `error` | Something failed; **the socket stays open** |

A real exchange (audio bodies truncated):

```json
{"type": "sentence", "seq": 1, "text": "Got it."}
{"type": "sentence", "seq": 2, "text": "Tell me more if you need details."}
{"type": "audio", "seq": 1, "bytes_b64": "UklGRuROAABXQVZF…", "sample_rate": 24000, "ms": 780.0}
{"type": "audio", "seq": 2, "bytes_b64": "UklGRmRzAQBXQVZF…", "sample_rate": 24000, "ms": 1080.0}
{"type": "done", "turn": 2, "text": "Got it. Tell me more if you need details.",
 "stages": {"retrieve": 0.007, "llm": 0.056, "ttft": 0.013, "segment.emit": 0.026,
            "tts": 0.031, "playback": 0.102, "first_audio": 0.191, "turn": 0.238}}
```

Notes that matter for a client implementation:

- **Join on `seq`, not on arrival order.** `sentence` and `audio` for the same
  sentence share a `seq`. With fast stub backends several sentences can be
  segmented before the first audio is queued; with real backends they interleave.
  A client that assumes strict alternation will break on one of the two.
- **Play `audio` frames as they arrive.** That is the entire streaming benefit.
  A client that buffers until `done` gets the serial latency.
- **`error` does not close the socket.** Send the next turn or close it
  yourself.
- **Audio is base64 inside JSON, not a binary frame.** Convenient, and roughly
  33% overhead; a binary protocol is the change to make before high concurrency.
- **One turn at a time per socket.** Sending a second message before `done`
  queues behind the session lock.

Closure codes: `1008` unauthorized, `4404` session not found (after an `error`
message).

## Operational endpoints

### `GET /healthz`

Liveness. Always `200` while the process is running.

```json
{"status": "ok"}
```

### `GET /readyz`

Readiness. Calls `ready()` on any backend that defines one.

**`200 OK`**

```json
{"status": "ok", "checks": {"stt": true, "llm": true, "tts": true}}
```

**`503 Service Unavailable`** with `"status": "not ready"` and the failing
backend `false`. A probe that raises counts as not ready, never as a 500 — a
model still loading keeps the pod out of the load balancer instead of failing
turns.

### `GET /metrics`

Prometheus text exposition, rendered from in-process counters and histograms
with no client library. Media type `text/plain; version=0.0.4; charset=utf-8`.

```text
# HELP vmp_http_requests_total HTTP requests by method, route and status
# TYPE vmp_http_requests_total counter
vmp_http_requests_total{method="POST",path="/v1/sessions",status="201"} 1
vmp_http_requests_total{method="POST",path="/v1/sessions/{session_id}/turns",status="200"} 1
# HELP vmp_turns_total Completed voice turns
# TYPE vmp_turns_total counter
vmp_turns_total 1
# HELP vmp_first_audio_seconds Time to first audio in seconds
# TYPE vmp_first_audio_seconds histogram
vmp_first_audio_seconds_bucket{le="0.5"} 1
vmp_first_audio_seconds_bucket{le="+Inf"} 1
vmp_first_audio_seconds_sum 0.268
vmp_first_audio_seconds_count 1
```

Metrics exposed: `vmp_http_requests_total`, `vmp_http_request_seconds`,
`vmp_turns_total`, `vmp_turn_stage_seconds{stage}`, `vmp_first_audio_seconds`.
Paths are templated (`/v1/sessions/{session_id}`), so session ids never become
label cardinality.

## Example: a curl session

```bash
export VMP_API_TOKEN=dev-token
export BASE=http://localhost:8080

# 1. create a session
SESSION=$(curl -sS -X POST "$BASE/v1/sessions" \
  -H "Authorization: Bearer $VMP_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"system_prompt": "You are a voice assistant. Answer in short spoken sentences."}' \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

# 2. run a turn from text
curl -sS -X POST "$BASE/v1/sessions/$SESSION/turns" \
  -H "Authorization: Bearer $VMP_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -H "X-Request-ID: demo-1" \
  -d '{"text": "what is the time to first audio"}' | python -m json.tool

# 3. run a turn from a WAV file
curl -sS -X POST "$BASE/v1/sessions/$SESSION/turns" \
  -H "Authorization: Bearer $VMP_API_TOKEN" \
  -F "audio=@utterance.wav;type=audio/wav" | python -m json.tool

# 4. read the session back
curl -sS "$BASE/v1/sessions/$SESSION" \
  -H "Authorization: Bearer $VMP_API_TOKEN" | python -m json.tool

# 5. operational endpoints need no token
curl -sS "$BASE/healthz"
curl -sS "$BASE/readyz"
curl -sS "$BASE/metrics" | head -20
```

## Example: a Python WebSocket client

Needs `websockets`. It writes each sentence's audio to a WAV file as it
arrives, which is the shape a real player has: hand the bytes to the output
device the moment they show up, never wait for `done`.

```python
"""Stream one turn over the voice agent WebSocket."""

import asyncio
import base64
import json
import os
import pathlib

import websockets

BASE_HTTP = os.environ.get("VMP_BASE", "http://localhost:8080")
BASE_WS = BASE_HTTP.replace("http://", "ws://").replace("https://", "wss://")
TOKEN = os.environ.get("VMP_API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}


async def stream_turn(session_id: str, text: str, out_dir: str = ".") -> dict:
    url = f"{BASE_WS}/v1/sessions/{session_id}/stream"
    async with websockets.connect(url, additional_headers=HEADERS) as ws:
        await ws.send(json.dumps({"text": text}))
        sentences: dict[int, str] = {}
        while True:
            msg = json.loads(await ws.recv())
            kind = msg.get("type")

            if kind == "sentence":
                sentences[msg["seq"]] = msg["text"]
                print(f"[{msg['seq']}] {msg['text']}")

            elif kind == "audio":
                wav = base64.b64decode(msg["bytes_b64"])
                path = pathlib.Path(out_dir) / f"reply-{msg['seq']:02d}.wav"
                path.write_bytes(wav)          # a real client plays it here
                print(f"[{msg['seq']}] {len(wav)} bytes, {msg['ms']:.0f} ms "
                      f"@ {msg['sample_rate']} Hz -> {path}")

            elif kind == "error":
                print("error:", msg["error"])   # the socket stays open

            elif kind == "done":
                print("stages:", json.dumps(msg["stages"], indent=2))
                return msg


async def main() -> None:
    import urllib.request

    req = urllib.request.Request(
        f"{BASE_HTTP}/v1/sessions",
        data=json.dumps({"system_prompt": "You are a voice assistant."}).encode(),
        headers={"Content-Type": "application/json", **HEADERS},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        session_id = json.load(resp)["id"]

    await stream_turn(session_id, "tell me about ray serve")


if __name__ == "__main__":
    asyncio.run(main())
```

Time to first audio, as the client experiences it, is the wall-clock gap between
sending the request and the first `audio` message. It should match the
`first_audio` entry in the `done` message's `stages`; when it does not, the gap
is network or client-side buffering, and the trace tells you which.

## See also

- [Serving API guide](guides/serving-api.md) — configuration, backends, the
  turn loop and the segmenter.
- [Ray](guides/ray.md) — the same contract served by a Ray Serve graph.
- [Observability](guides/observability.md) — the trace rows behind `stages`.
- [Security and reliability](architecture/security-reliability.md) — identity,
  authorization and failure behaviour.
