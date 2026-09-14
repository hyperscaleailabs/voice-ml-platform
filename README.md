# voice-ml-platform

An ML platform for voice agents, shown as a progression from **research** to
**development** to **productization**. It trains, evaluates, serves and ships
speech-to-speech models to a cloud API and to edge devices, and it keeps every
claim tied to a measurement.

[![CI](https://github.com/hyperscaleailabs/voice-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/hyperscaleailabs/voice-ml-platform/actions/workflows/ci.yml)
[![Docs](https://github.com/hyperscaleailabs/voice-ml-platform/actions/workflows/docs.yml/badge.svg)](https://hyperscaleailabs.github.io/voice-ml-platform/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Documentation, blog, whitepapers and references:**
https://hyperscaleailabs.github.io/voice-ml-platform/

```text
                 ┌────────────────────────── research/ ──────────────────────────┐
                 │ hypotheses, spikes, notebooks: claim → falsifier → outcome      │
                 └──────────────────────────────┬────────────────────────────────┘
                                                ▼
 ┌──────────────────────────────────────── src/vmp/ ────────────────────────────────────────┐
 │ data ─▶ features ─▶ training (Ray · TRL SFT/DPO · PEFT LoRA) ─▶ registry ─▶ eval gates  │
 │                                                                                          │
 │ rag (vector + graph) ─▶ serving (voice runtime · FastAPI · Ray Serve · vLLM) ─▶ API      │
 │                                          └────────────▶ edge (export · bundle · runtime)│
 │ observability: stage traces · metrics · drift · SLOs                                     │
 └──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                            ▼
                 ┌────────────────────────── deploy/ · runbooks/ ────────────────────────┐
                 │ Docker · Kubernetes (kustomize) · KubeRay · Feast · OTel · Prometheus  │
                 │ SLO alerts · Grafana · edge OTA rollout · incident and rollback runbooks│
                 └────────────────────────────────────────────────────────────────────────┘
```

## What a voice turn looks like

```text
microphone / client audio
    ↓
STT              Whisper (LoRA accent adapter) — protocol, swappable backend
    ↓
retrieve         hybrid RAG: keyword prefilter + vector store + graph expansion + relevance floor
    ↓
LLM              streams tokens (Ollama, OpenAI-compatible / vLLM, or in-process)
    ↓
segmenter        cuts sentences the moment each is provably complete
    ↓
TTS              one sentence at a time, one worker thread
    ↓
playback         time to first audio is the headline metric
```

Every stage writes a `<stage>.start` / `<stage>.end` trace row keyed by session
and turn. The trace is what evaluation, SLOs and drift detection read.

## Quickstart

The core package is standard library only. The demos run with nothing installed
but Python 3.11+.

```bash
git clone https://github.com/hyperscaleailabs/voice-ml-platform
cd voice-ml-platform
python examples/demo_voice_turn.py       # a session of three turns, stage timings, trace rows
python examples/demo_rag.py              # vector + graph retrieval with a relevance floor
python examples/demo_features.py         # streaming session features, point-in-time lookup
python examples/demo_training_plan.py    # SFT / DPO / Whisper-LoRA plans, local and Ray
python examples/demo_release_gate.py     # golden-set WER, SLIs, gate decision
python examples/demo_edge_bundle.py      # export plan, bundle, verify, tamper, edge turn
```

Development install, lint and tests:

```bash
pip install -e ".[dev]"
ruff check src tests examples
pytest -q
vmp --help
```

Run the voice agent API locally with stub backends, then with real ones:

```bash
pip install -e ".[serve]"
vmp serve api --backend echo --port 8080
vmp serve api --backend ollama --port 8080          # local Ollama
vmp serve api --backend openai-compat --port 8080   # vLLM or any OpenAI-compatible server
```

## The three stages

| Stage | Directory | Unit of work | Exit criterion |
|---|---|---|---|
| Research | `research/` | a hypothesis with a falsifier and a disposable spike | the falsifier was run, or the spike says `not run` |
| Development | `src/vmp/`, `tests/` | a module with a protocol, a reference implementation and adapters | tests pass with no extras installed; adapters import lazily |
| Productization | `deploy/`, `runbooks/`, `configs/` | manifests, SLOs, gates, runbooks | `kustomize build` succeeds; SLO rules and gates are declared; every failure has a runbook |

The stages are cumulative. A number moves from a research README into the
package only after it has a source file; it moves into an SLO only after the
trace that measures it exists in production.

## Subsystems

| Module | What it does | Reference implementation | Optional backends |
|---|---|---|---|
| `vmp.data` | turn corpora, DPO preference pairs, synthetic golden set, PII scrub, leak-free splits | JSONL | `datasets` |
| `vmp.features` | feature store: views, offline point-in-time join, online store with TTL, streaming updates from trace events, Feast export | JSONL + in-memory | Feast, Redis, Parquet, Kafka |
| `vmp.training` | `TrainingPlan` → TRL `SFTTrainer` / `DPOTrainer` with PEFT LoRA; Whisper LoRA; Ray Train launcher; Ray Data preprocessing; adapter merge | dry-run planner | torch, transformers, peft, trl, ray |
| `vmp.registry` | model versions, lifecycle stages, lineage (data hash, config hash, git sha) | file registry | MLflow |
| `vmp.rag` | chunking, embeddings, vector store, graph store, entity extraction, hybrid retriever with a relevance floor | hashing embedder, in-memory stores | pgvector, Qdrant, Neo4j, sentence-transformers, Ollama |
| `vmp.serving` | voice runtime, sentence segmenter, session store, FastAPI + WebSocket API, Ray Serve graph | echo / template / silent backends | faster-whisper, Kokoro, Ollama, vLLM, Ray Serve |
| `vmp.edge` | export (ONNX / GGUF / MLX), quantization plan, signed bundle, offline policy, edge runtime | dry-run + bundle verification | optimum, llama.cpp, mlx |
| `vmp.eval` | WER / CER, golden set, rubric and LLM judges, preference win-rate, latency percentiles, release gates | pure Python | jiwer |
| `vmp.observability` | stage tracer, metrics, PSI / KS drift, SLOs, error budgets, Prometheus rules | JSONL sink | OpenTelemetry, prometheus-client |

Every module follows one rule: a `Protocol` for the backend, one standard-library
implementation that runs in tests and demos, and adapters that import their
dependency inside the class, never at module import.

## Productization

```text
deploy/
├── docker/        api, train, edge images
├── compose/       full local stack behind profiles: core, rag, obs, train
├── k8s/           kustomize base + local and cloud overlays: API, pgvector, Redis,
│                  Qdrant, Neo4j, MinIO, OTel collector, Prometheus, Grafana
├── ray/           KubeRay RayCluster, RayJob (SFT, DPO), RayService (voice agent)
├── feast/         feature repo generated from the in-code views, materialize CronJob
├── otel/ prometheus/ grafana/   collector, scrape config, SLO alert rules, dashboard
└── edge/          systemd unit, k3s DaemonSet, OTA canary rollout
runbooks/          incident response, model rollback, drift, TTFA SLO breach,
                   RAG relevance, edge bundle verification, on-call, capacity, privacy
```

## Where the numbers come from

This repository does not publish benchmarks of its own yet. The measurements it
cites come from the private predecessor project (`alpha-core`, cycle 5,
2026-09-12) and are labelled as such wherever they appear. Each is tied to the
file that produced it. See [`DESIGN.md`](DESIGN.md) for the full table and the
citation rule.

## Layout

```text
DESIGN.md        the contract every module is built against
research/        stage 1: hypotheses and spikes
src/vmp/         stage 2: the package
deploy/          stage 3: manifests
runbooks/        stage 3: operations
configs/         TOML configuration
examples/        demos that need no extras
tests/           pytest; no network, no models, no GPU by default
docs/            MkDocs site: architecture, guides, blog, whitepapers, references
```

## License

Apache-2.0. See [LICENSE](LICENSE).
