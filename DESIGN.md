# DESIGN — contract for every module in this repository

This file is the source of truth for structure, naming, and rules. Every
subsystem is built against it. Read it fully before writing code or docs.

## What this repository is

`voice-ml-platform` is a public, production-oriented ML platform for **voice
agents**, shown as a progression: **research -> development -> productization**.
It generalises the private `alpha-core` project (a fully local voice agent on one
Apple M4 laptop: energy VAD -> Whisper STT -> Ollama LLM -> sentence segmenter ->
Kokoro TTS, with JSONL stage tracing, a golden evaluation set, pgvector RAG, and a
hypothesis-driven SDLC) into a platform that can train, evaluate, serve, and
ship models to both a **cloud API** and an **edge device**.

Package name: `vmp`. Repo: `hyperscaleailabs/voice-ml-platform`. License: Apache-2.0.

## The three stages (top-level layout)

```
research/        Stage 1. Hypotheses, spikes, notebooks. Claim / falsifier / outcome.
src/vmp/         Stage 2. The platform package. Typed, tested, protocol-driven.
deploy/          Stage 3. Docker, Kubernetes (kustomize), KubeRay, Feast, OTel,
                 Prometheus SLO rules, Grafana dashboards.
runbooks/        Stage 3. Operational procedures: incident, rollback, drift, on-call.
docs/            MkDocs Material site (GitHub Pages): blog, whitepapers, references.
examples/        Runnable demos. `python examples/demo_*.py` with NO extras installed.
configs/         TOML configs (stdlib `tomllib`). No YAML in the core package.
tests/           pytest. No network, no models, no GPU in the default run.
```

## Package layout (`src/vmp/`)

| Module | Responsibility | Heavy backends (optional, lazy) |
|---|---|---|
| `types.py` | Shared dataclasses (below). Stdlib only. | — |
| `config.py` | `load_config(path) -> dict`, `Settings` dataclass, env overrides. | — |
| `cli.py` | `vmp` entry point. Subcommand registry; each module adds its own. | — |
| `data/` | Datasets: turn corpora, preference pairs, synthetic golden set builder, JSONL IO, splits, PII scrub. | `datasets` |
| `features/` | Feature store: `FeatureView` definitions, offline store (Parquet/JSONL), online store (in-memory / Redis), point-in-time join, materialize, Feast definitions export. | `feast`, `redis`, `pyarrow` |
| `training/` | `TrainingPlan`, LoRA/PEFT config, TRL `SFTTrainer` + `DPOTrainer` wrappers, Ray Train job launcher, Ray Data preprocessing, adapter merge, dry-run planner. | `torch`, `transformers`, `peft`, `trl`, `ray` |
| `registry/` | Model registry: `ModelArtifact` versions, stages (candidate/staging/production), lineage (data hash, config hash, git sha), local file backend, MLflow adapter. | `mlflow` |
| `rag/` | Chunking, `Embedder` protocol (default: deterministic hashing embedder), `VectorStore` (in-memory / pgvector / Qdrant), `GraphStore` (in-memory / Neo4j), entity+relation extraction, hybrid retriever with a **relevance floor**, context packing for spoken answers. | `psycopg`, `qdrant-client`, `neo4j`, `sentence-transformers` |
| `serving/` | Voice agent runtime (`Session`, `Turn` loop: STT -> retrieve -> LLM -> segment -> TTS), `STT`/`LLM`/`TTS` protocols with echo/stub implementations, sentence segmenter, FastAPI app (`/v1/sessions`, `/v1/sessions/{id}/turns`, WebSocket stream, `/healthz`, `/readyz`, `/metrics`), Ray Serve deployment graph, vLLM backend adapter. | `fastapi`, `uvicorn`, `ray[serve]`, `vllm`, `faster-whisper`, `kokoro` |
| `edge/` | Export pipeline (merge adapter -> ONNX / GGUF / MLX), quantization config, `EdgeBundle` (manifest + checksums + policy), edge runtime (same turn loop, offline-only backends), OTA-style bundle verification. | `onnx`, `onnxruntime`, `optimum`, `mlx`, `llama-cpp-python` |
| `eval/` | Golden set (text -> synthesized audio -> WER), WER/CER, exact-match, preference win-rate, latency percentiles from traces, LLM-judge protocol (stub default), **release gates** (`GateDecision`). | `jiwer` |
| `observability/` | Stage tracer (JSONL spans, same schema as alpha-core: `<stage>.start/.end`, `session`, `turn`, `span`, `seq`, `ms`), metrics registry, OTel exporter, drift detectors (PSI / KS on features and on WER), SLO definitions (time-to-first-audio p50/p95, WER, error rate). | `opentelemetry-*`, `prometheus-client` |

Rule: **every module has a `Protocol` for its backend, one stdlib reference
implementation that runs in tests and demos, and optional adapters that import
their dependency lazily inside the class (never at module import).** Importing
`vmp` or any submodule with nothing but the standard library installed must
succeed.

## Shared types (`vmp/types.py`) — use these, do not redefine

```python
@dataclass(frozen=True) class Utterance:  id, text, audio_path: str|None, speaker: str|None, duration_s: float|None, meta: dict
@dataclass class Turn:                    session_id, turn: int, user: Utterance, assistant_text: str|None, retrieved: list[str], stages: dict[str, float]  # stage -> ms
@dataclass class Session:                 id, created_at: float, turns: list[Turn], system_prompt: str, meta: dict
@dataclass(frozen=True) class PreferencePair: prompt, chosen, rejected, source: str, meta: dict
@dataclass(frozen=True) class FeatureRow:  entity_id, event_ts: float, values: dict[str, float|int|str|None]
@dataclass(frozen=True) class Document:    id, path, text, sha: str, meta: dict
@dataclass(frozen=True) class Chunk:       id, document_id, start_line, end_line, text, embedding: tuple[float, ...]|None
@dataclass(frozen=True) class ModelArtifact: name, version, stage: str, base_model, adapter_path: str|None, config_hash, data_hash, git_sha: str|None, metrics: dict, created_at: float
@dataclass(frozen=True) class EvalResult:  name, metrics: dict[str, float], n: int, details: dict
@dataclass(frozen=True) class GateDecision: passed: bool, reasons: list[str], results: list[EvalResult]
```

Enums as `str` constants, not `Enum`, so they serialize plainly: stages are
`"candidate" | "staging" | "production" | "retired"`.

## Conventions

- Python `>=3.11`. `src/` layout, hatchling, `pytest`, `ruff` (line length 100,
  rules `E,F,I,UP,B,SIM,RUF`). Type hints everywhere. `from __future__ import annotations`.
- Configs are TOML in `configs/`; loaded with `tomllib`. Env vars override with
  prefix `VMP_` (e.g. `VMP_REGISTRY_ROOT`).
- Every heavy operation accepts `dry_run: bool`. A dry run validates inputs,
  builds and returns the plan/manifest as a dict, and imports nothing heavy.
- Every CLI subcommand is registered by its module via
  `vmp.cli.register(name, help, add_args(parser), run(args) -> int)`.
- Traces: one JSON object per line, keys `ts, session, turn, event, span, seq, ms, payload`.
  `event` is `<stage>.start` / `<stage>.end`. Stages for a voice turn:
  `listen, stt, retrieve, llm, segment.emit, tts, playback, turn`. `llm` payload
  carries `ttft_ms`. `playback` payload carries `response_ms` (time to first audio) —
  the headline metric.
- Numbers in docs and READMEs must have a source. Measurements that came from the
  private `alpha-core` project are cited as
  "alpha-core, cycle 5, 2026-09-12, `<file>`" — never presented as this repo's
  benchmark. The date is the date the measurement was taken, so a fact from an
  earlier run carries that run's date instead (the golden set below is
  2026-09-05, not cycle 5's 2026-09-12). Demo output may be quoted only if the
  demo actually prints it. Do not invent benchmarks, throughput figures, or
  cost numbers.
- Documentation tone: neutral, technical, concise. Explain what it is, how it
  works, why it matters. No marketing language. Lines <= 100 chars in Markdown.
- No secrets, no absolute local paths, no references to private machines.
- Tests: `tests/test_<module>.py`. Default run needs only `pytest` (+ `fastapi`,
  `httpx` for the API tests, which `pytest.importorskip`). Markers: `slow`, `gpu`,
  `integration` (needs a service). Keep the default run under ~30 s.

## Measured facts allowed to be cited (source: alpha-core, private)

| Fact | Value | Source file (alpha-core, cycle 5, 2026-09-12) |
|---|---|---|
| STT median latency, mlx-whisper on M4 GPU vs faster-whisper CPU, same audio, small.en | 233 ms vs 1,761 ms (7.55x) | `spikes/streaming-whisper-m4/compare_stt.py` |
| Whisper accent LoRA, exact-match on a 35-sentence operator-read corpus | 11/35 -> 32/35 (train set; no held-out set yet) | `notebook_whisper_accent_lora.ipynb` |
| Wake-word correction (Jaro-Winkler) on 4,855 historical turns | 18 altered, 0 false positives | `test_no_new_wake_words.py` |
| Streaming vs serial time-to-first-audio, 8-sentence answer, stubbed playback | 2,688 ms vs 11,049 ms | `README.md` "Streaming" |
| Golden set WER, faster-whisper small.en, 108 synthetic clips | 2.69% overall; `asr_names` 28.9% | `evals/golden/README.md` (2026-09-05) |
| gemma3:270m vs gemma3 4B on 6 factual questions | llm median 192 ms vs 4,427 ms; 2/6 wrong vs 0/6 | `notebook_optimized.ipynb` |
| RAG cosine score spread, rank 1 vs rank 20 | 0.494 vs 0.442 | `README.md` "Talking about this codebase" |
| Kokoro real-time factor | 0.105–0.157 | `README.md` "Streaming" |

## The five lenses (used to organise architecture docs)

Full stack -> Data -> Architecture -> Secure & reliable -> Production operations.
Full-stack spine of a voice turn:
User -> Client -> API/Session -> Application Runtime -> AI Services -> Infrastructure -> User.

## Hypothesis-driven development (research stage format)

Every `research/<nn>-<topic>/README.md` has exactly these sections:
`## Claim`, `## Falsifier`, `## Method`, `## Outcome`, `## What moved into src/`.
Outcome is one of: `not run`, `run — falsifier passes`, `run — falsified`,
`run — inconclusive`. Anything "not run" says so.
