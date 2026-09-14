# Guide: Serving API

## Purpose

`vmp.serving` is the runtime that conducts a voice turn and the HTTP/WebSocket
service that exposes it. The runtime owns three protocols — `STT`, `LLM`, `TTS`
— and one algorithm, the sentence segmenter. It does not transcribe, infer,
retrieve or synthesise; it sequences those, owns the session, and writes the
trace.

The consequence is that swapping a model changes one TOML table and nothing
else. The reference backends (echo STT, template LLM, silent TTS) have no
dependencies at all, so the whole service runs, is tested, and is demonstrable
with nothing installed beyond FastAPI.

Two deployment shapes run the same runtime: a single process
(`vmp serve api`) and a Ray Serve graph (`vmp serve ray`, see [Ray](ray.md)).
The full endpoint contract is on the [API](../api.md) page; this guide is about
running and configuring the service.

## Config

`configs/serving.toml`:

```toml
# Defaults run with no optional dependency: echo STT, template LLM, silent TTS.

[serving]
system_prompt = "You are a voice assistant. Answer in short spoken sentences. Do not use lists, markdown, or symbols."
max_history_turns = 12
max_context_chars = 1200

[serving.api]
host = "127.0.0.1"
port = 8080
max_request_bytes = 10485760   # 10 MiB
turn_timeout_s = 60.0
session_ttl_s = 3600.0
# Bearer auth is enabled by setting VMP_API_TOKEN in the environment; never put a token here.

[serving.backends]
stt = "echo"             # echo | faster-whisper
llm = "template"         # template | ollama | openai-compat | vllm
tts = "silent"           # silent | kokoro

[serving.backends.faster_whisper]
model = "small.en"
device = "cpu"
compute_type = "int8"

[serving.backends.ollama]
host = "http://localhost:11434"
model = "gemma3:4b"

[serving.backends.openai_compat]
base_url = "http://localhost:8000/v1"
model = "default"

[serving.backends.vllm]
model = "Qwen/Qwen2.5-0.5B-Instruct"
max_tokens = 256

[serving.backends.kokoro]
voice = "af_heart"
lang_code = "a"
```

Three of these settings are latency settings in disguise:

- **`system_prompt`** is prepended to every turn. It is the cheapest lever on
  spoken style, and the one that costs prompt tokens on every single turn.
- **`max_history_turns` (12)** bounds how far the conversation can grow the
  prompt. Unbounded history eventually overruns the context window, and long
  before that it inflates time-to-first-token.
- **`max_context_chars` (1200)** bounds the retrieved context. Context is paid
  for twice in a voice turn: once in prompt processing, once again in the
  time-to-first-token that gates time-to-first-audio.

There is no token in this file, ever. Auth is enabled by setting
`VMP_API_TOKEN` in the environment; an empty or unset value disables it, which
is correct for local development and wrong everywhere else.

## The turn loop

```text
listen -> stt -> retrieve -> llm -> segment.emit -> tts -> playback -> turn
```

Each stage writes `<stage>.start` and `<stage>.end` trace rows. `tts` and
`playback` run once per sentence and share the sentence's `seq`, so a sentence
can be followed across `segment.emit seq=3 -> tts seq=3 -> playback seq=3`.
`playback.end` on `seq=1` carries `response_ms` — time to first audio, the
headline metric.

The segmenter is what makes this feel fast: the first sentence is synthesised
and played while the model is still writing the rest. On an eight-sentence
answer with playback stubbed to real time, streaming reached first audio in
2,688 ms against 11,049 ms serial (alpha-core, cycle 5, 2026-09-12, `README.md`
"Streaming"). It is also the component that must never be wrong: a sentence
emitted at a decimal point or an abbreviation is *spoken* truncated and cannot
be unspoken. The segmenter therefore emits only when a sentence is provably
complete — terminal punctuation, optional closing quote or bracket, then
whitespace — and holds a boundary at the end of the buffer until `flush()`,
because the next token could turn `3.` into `3.5`. It knows abbreviations
(`Dr.`, `e.g.`, `a.m.`) and treats an ellipsis as a boundary only when the next
word starts a new sentence.

## CLI

```bash
vmp serve api --config configs/serving.toml --dry-run          # resolved plan, exits
vmp serve api --config configs/serving.toml                    # uvicorn, needs [serve]
vmp serve api --backend echo --port 8080                       # no dependencies at all
vmp serve api --backend ollama --model gemma3:4b --base-url http://localhost:11434
vmp serve api --backend openai-compat --base-url http://localhost:8000/v1 --model my-model
vmp serve api --config configs/serving.toml --trace .vmp/trace.jsonl
```

`--backend echo` forces all three backends to their reference implementations at
once, which is the fastest way to check that a deployment problem is not a model
problem. `--trace` appends JSONL rows that [`vmp obs`](observability.md) reads.

## What a dry run returns

The fully resolved plan — config file merged with flags, every default filled
in — printed as JSON, with nothing imported and no port bound:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backends": {
    "stt": "echo",
    "llm": "template",
    "tts": "silent",
    "faster_whisper": {"model": "small.en", "device": "cpu", "compute_type": "int8"},
    "ollama": {"host": "http://localhost:11434", "model": "gemma3:4b"},
    "openai_compat": {"base_url": "http://localhost:8000/v1", "model": "default"},
    "vllm": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "max_tokens": 256},
    "kokoro": {"voice": "af_heart", "lang_code": "a"}
  },
  "system_prompt": "You are a voice assistant. Answer in short spoken sentences. Do not use lists, markdown, or symbols.",
  "max_history_turns": 12,
  "max_context_chars": 1200,
  "session_ttl_s": 3600.0,
  "max_request_bytes": 10485760,
  "turn_timeout_s": 60.0,
  "trace": null
}
```

Note that the per-backend option tables are all present regardless of which
backend is selected. That is deliberate: the dry run shows what *would* be used
if you switched, which is the question you are usually asking.

## What the real run needs

```bash
pip install -e ".[serve]"     # fastapi, uvicorn
```

That alone runs the whole service with the reference backends. Real models are
separate extras, each one imported lazily inside its adapter:

| Backend | Extra / package | Notes |
|---|---|---|
| `faster-whisper` STT | `faster-whisper` (CTranslate2) | CPU or CUDA. **No Metal device exists** in CTranslate2 on macOS — `device = "cpu"` is not a default there, it is the only option. |
| `ollama` LLM | HTTP, no package | Streams tokens; easiest local LLM. |
| `openai-compat` LLM | HTTP, no package | Points at a vLLM server, or anything speaking the same streaming API. |
| `vllm` LLM | `".[serve]"` + `vllm` | In-process engine; continuous batching and paged KV cache. |
| `kokoro` TTS | `kokoro` | Measured real-time factor 0.105–0.157 (alpha-core, cycle 5, 2026-09-12, `README.md` "Streaming"). |

`/readyz` calls `ready()` on any backend that defines it and returns 503 with a
per-backend `checks` map when one is not ready, so a model still loading keeps
the pod out of the load balancer instead of failing turns.

## Operational behaviour worth knowing

- **One lock per session.** Turns within a session are serialised; different
  sessions run concurrently. A second concurrent turn on the same session waits
  rather than interleaving into the history.
- **Turns run in a thread pool.** The runtime is synchronous; the endpoint
  wraps it with `run_in_threadpool` and an `asyncio.wait_for` bounded by
  `turn_timeout_s`, returning 504 on expiry.
- **Request id on everything.** `X-Request-ID` is honoured if supplied and
  generated otherwise, echoed on the response, and included in the single JSON
  access-log line per request.
- **Size limit before auth.** A request over `max_request_bytes` is rejected
  413 from the `Content-Length` header, before the body is read.
- **`/healthz`, `/readyz` and `/metrics` skip auth** so the orchestrator and the
  scraper need no credentials. Expose them on the cluster-internal port only.
- **Metrics have no client library.** `Metrics` renders Prometheus text format
  from in-process counters and histograms:
  `vmp_http_requests_total`, `vmp_http_request_seconds`, `vmp_turns_total`,
  `vmp_turn_stage_seconds{stage}` and `vmp_first_audio_seconds`.

## Pitfalls

- **Sessions are in memory, per replica.** `InMemorySessionStore` expires after
  `session_ttl_s`. With more than one replica, either pin a session to a replica
  or move the store behind a shared backend; a round-robined session loses its
  history. The trace file, not the session store, is the durable record.
- **The WebSocket sends base64 audio in JSON, not binary frames.** Convenient
  for a demo client, wasteful at scale — a binary frame protocol is the change
  to make before high concurrency, and it changes the client contract.
- **`turn_timeout_s = 60` is far above any acceptable voice latency.** It is a
  backstop against a hung backend, not an SLO. The SLO is
  [`ttfa_p95`](evaluation-and-gates.md), 3,000 ms.
- **The template LLM is not a model.** It returns a fixed reply token by token.
  It is there so that latency plumbing, segmentation and playback can be tested
  without a GPU — never quote a latency measured against it as a system result.
- **Markdown in a reply is a defect, not a formatting choice.** The system
  prompt forbids it, DPO trains it out ([DPO](training-dpo.md)), and the
  retriever strips it ([RAG](rag-vector-graph.md)). Anything that reaches the
  TTS with `**` in it is read aloud as noise.
