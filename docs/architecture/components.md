# Components: responsibilities, boundaries, contracts

The third lens divides the system into components with a responsibility each, a
boundary around it, and a contract across the boundary. In `vmp` the contract is
a `Protocol`, the boundary is a module, and the responsibility is one line in
the table below.

Every module follows the same rule: **one `Protocol` for its backend, one
standard-library reference implementation that runs in tests and demos, and
optional adapters that import their dependency lazily inside the class.**
`import vmp` succeeds with nothing but the standard library installed. Every
heavy operation accepts `dry_run=True` and returns its plan as a dict.

## Module map

```mermaid
flowchart TB
    subgraph core["stdlib core"]
        T[types] --> C[config] --> CLI[cli]
    end
    subgraph data["data plane"]
        D[data] --> F[features]
        D --> R[rag]
    end
    subgraph model["model plane"]
        TR[training] --> REG[registry]
        REG --> E[edge]
    end
    subgraph run["runtime"]
        S[serving]
    end
    subgraph quality["quality plane"]
        EV[eval] --> REG
        O[observability]
    end
    core --> data & model & run & quality
    F --> S
    R --> S
    REG --> S
    S --> O
    O --> F
    O --> EV
```

## Module table

| Module | Responsibility | Protocol(s) | Reference implementation | Adapters (lazy import) |
|---|---|---|---|---|
| `types` | Shared dataclasses; stages as `str` constants | — | — | — |
| `config` | `load_config(path)`, `Settings`, `VMP_*` env overrides | — | `tomllib` | — |
| `cli` | `vmp` entry point; each module registers `vmp <module> <cmd>` | — | `argparse` | — |
| `data` | Turn corpora, preference pairs, synthetic golden set, JSONL IO, splits, PII scrub | `PairRule` | rule-based pair builder; regex scrub | `datasets` |
| `features` | `FeatureView`, offline and online stores, point-in-time join, materialize, Feast export | `OfflineStore`, `OnlineStore` | JSONL/Parquet offline; dict online | `feast`, `redis`, `pyarrow` |
| `training` | `TrainingPlan`, LoRA config, SFT and DPO wrappers, Ray Train launcher, Ray Data preprocessing, adapter merge, dry-run planner | `Runner` | dry-run planner (scans data, estimates steps) | `torch`, `transformers`, `peft`, `trl`, `ray` |
| `registry` | `ModelArtifact` versions, stages, lineage | `Registry` | local file backend | `mlflow` |
| `rag` | Chunking, embedding, vector store, graph store, extraction, hybrid retriever with relevance floor, context packing | `Embedder`, `VectorStore`, `GraphStore`, `Retriever` | hashing embedder; in-memory stores | `sentence-transformers`, `psycopg` (pgvector), `qdrant-client`, `neo4j` |
| `serving` | `Session`/`Turn` loop, segmenter, FastAPI app, WebSocket stream, Ray Serve graph, vLLM adapter | `STT`, `LLM`, `TTS` | echo STT, stub LLM, stub TTS | `fastapi`, `uvicorn`, `ray[serve]`, `vllm`, `faster-whisper`, `kokoro` |
| `edge` | Export pipeline, quantisation config, `EdgeBundle`, edge runtime, bundle verification | `Exporter`, `EdgeBackend` | manifest builder; checksum verifier | `onnx`, `onnxruntime`, `optimum`, `mlx`, `llama-cpp-python` |
| `eval` | Golden set, WER/CER, exact match, win-rate, latency percentiles, LLM judge, release gates | `Judge`, `Scorer` | pure-Python WER; stub judge | `jiwer` |
| `observability` | Stage tracer, metrics registry, OTel exporter, drift detectors, SLO definitions | `Sink`, `MetricsBackend` | `ListSink`, file sink; in-process metrics | `opentelemetry-*`, `prometheus-client` |

## The serving runtime

The runtime owns three protocols and one algorithm.

- `STT`: audio -> transcript. Reference: echo (returns the text the demo put in
  `Utterance.text`). Adapters: faster-whisper (CTranslate2, CPU or CUDA),
  mlx-whisper (Apple GPU).
- `LLM`: messages -> token stream. Reference: stub that emits a fixed reply
  token by token. Adapters: vLLM (OpenAI-compatible endpoint), Ollama.
- `TTS`: sentence -> audio. Reference: stub that returns a silent buffer of the
  expected duration. Adapter: Kokoro.
- The **sentence segmenter**: token stream -> sentences, emitting only when a
  sentence is provably complete. It is the component the streaming design
  depends on, it has its own tests, and it is shared unchanged by the edge
  runtime.

The `Turn` loop is `listen -> stt -> retrieve -> llm -> segment -> tts -> playback`,
with each stage traced. The runtime asks for a capability and the config
decides the adapter; a model swap changes one TOML table.

## Where vLLM sits

vLLM is an `LLM` adapter, nothing more. The runtime speaks to it over the
OpenAI-compatible streaming endpoint, so time-to-first-token (`llm.ttft_ms`) is
measured the same way as with the stub. What vLLM brings is continuous batching
and a paged KV cache, which is what keeps `ttft_ms` flat as concurrent sessions
rise. In the Ray Serve deployment graph, vLLM is one deployment behind the
runtime deployment, scaled on the number of in-flight requests. A LoRA adapter
produced by training is served either merged into the base weights (the
`merge` step in `vmp.training`) or, with vLLM's multi-LoRA support, loaded by
name so that several adapters share one base model.

## Ray's three roles

Ray appears in three places, and each is a separate extra so that none is
required for the others.

| Role | Module | What it does | Runs as |
|---|---|---|---|
| **Ray Data** | `vmp.training` preprocessing | Reads JSONL, tokenises and packs in parallel, streams into Train | Part of a `RayJob` |
| **Ray Train** | `vmp.training` launcher | Wraps the same `Runner` (SFT or DPO) in a `TorchTrainer` with N workers and G GPUs each | `RayJob` on KubeRay |
| **Ray Serve** | `vmp.serving` deployment graph | Runtime, STT, TTS and vLLM as deployments with independent autoscaling | `RayService` on KubeRay |

The key design point is **one `TrainingPlan`, two backends**. `compute.backend`
is `local` or `ray`; the plan is identical, and the Ray launcher only wraps the
local runner in a `TorchTrainer`. A dry run with `backend = "ray"` returns the
`RayJob` spec it would submit. See [Ray](../guides/ray.md).

## TRL: SFT vs DPO

Both are `Runner`s over the same `TrainingPlan`, differing in `kind` and in what
the dataset rows contain.

- **SFT** (`SFTTrainer`): rows are `{prompt, completion}` or chat messages. The
  model learns to produce the completion. Used to teach a chat model the spoken
  register in the first place, and to teach Whisper (as `whisper-lora`, a
  `SEQ_2_SEQ_LM` task) an accent.
- **DPO** (`DPOTrainer`): rows are `PreferencePair` (`prompt, chosen, rejected`).
  The model learns to prefer `chosen` over `rejected` relative to a frozen
  reference model, with `beta` controlling how far it may move. When the policy
  is a PEFT model, TRL can use the adapter-disabled base as the reference,
  which halves the memory of holding two models. Used after SFT to push the
  model toward replies that sound right when spoken.

## PEFT and LoRA

Every fine-tune in the platform is a LoRA adapter, never a full-weight update.
`LoraConfig` in `vmp.training` mirrors `peft.LoraConfig` without importing it:
rank `r`, `alpha`, `dropout`, `target_modules`, `task_type`. The adapter is a
small file with a `config_hash` and `data_hash`; the base model is referenced by
name. This is what makes the registry cheap, the edge export tractable (merge,
then convert), and the rollback trivial (point at the previous adapter).

The adapter must be **merged** before export to GGUF, ONNX or MLX, because those
runtimes load a single weight set. `vmp training merge` does this and writes a
new `ModelArtifact` whose lineage points at both the base and the adapter.

## Contracts summarised

| Boundary | Input | Output | Invariant |
|---|---|---|---|
| Runtime -> STT | audio (16 kHz mono PCM) | transcript string | STT never sees session state |
| Runtime -> Retriever | query string | list of packed chunks | nothing below the relevance floor |
| Runtime -> LLM | messages | token stream | first token stamps `ttft_ms` |
| LLM -> Segmenter | token stream | sentence stream | never a truncated sentence |
| Runtime -> TTS | one sentence | audio (24 kHz PCM) | one worker owns one pipeline |
| Training -> Registry | plan, adapter, metrics | `ModelArtifact` | hashes present |
| Eval -> Registry | `GateDecision` | promotion or not | no promotion without a decision |
| Registry -> Edge | merged artifact | `EdgeBundle` | checksums verified before load |
