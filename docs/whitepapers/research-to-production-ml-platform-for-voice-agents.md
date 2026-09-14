# Research to production: an ML platform for voice agents

## Abstract

Applied machine-learning work fails between stages more often than it fails
within them. A notebook produces a number that cannot be reproduced; a training
script produces an artifact whose provenance is a memory; a deployment produces
behaviour nobody can attribute to a change. Each stage is individually competent
and the handoffs lose the evidence.

This paper describes a platform for voice agents organised as an explicit
maturity path — research, development, productization — where the interesting
engineering is in the contracts between stages rather than in any one component.
It states what each stage owes the next, what artifacts each must produce, and
the evidence discipline that makes the artifacts trustworthy: **a number with no
source file does not go in a table.**

Voice makes the argument concrete. The metric a user perceives, time to first
audio, is produced by a trace row in the serving runtime, consumed by an
evaluation gate, fed into a feature view, and enforced as a service level
objective. That one number crosses every stage boundary in the system, and it
only survives the crossing because the schema that carries it is identical
everywhere.

The paper closes with a capability matrix across seven subsystems — training,
feature store, RAG, serving, edge, evaluation, observability — describing what
each looks like at research, development and production maturity.

---

## 1. The problem with stages

### 1.1 Where work is actually lost

A working prototype answers one question: can these components be assembled into
something that behaves? For a voice agent, that means a microphone, a voice
activity detector, a speech-to-text model, a language model, a sentence
segmenter, a text-to-speech engine and a speaker, wired so that a person can talk
to it.

Every question that follows is about the layer around the prototype. Is the new
model better than the old one? Better for whom — did it regress for one speaker
while improving the average? When a reply arrives late, which of eight stages was
slow? How does the same behaviour reach both a cloud API and a device with no
network? When production regresses, how do you recover the exact data and
configuration that produced the model now running?

None of those are answered by the pipeline. They are answered by the platform,
and a platform is mostly contracts.

### 1.2 Why voice sharpens it

Three properties of voice make platform weaknesses visible faster than text does.

**Latency is perceived, not measured.** The user experiences one number: the gap
between finishing their sentence and hearing the first syllable. Everything after
that is invisible while audio keeps flowing. So the platform's headline metric is
not throughput or tokens per second but *time to first audio*, and it has to be
attributable to a stage.

**Errors are unrecoverable in real time.** A wrong retrieval in a chat interface
is skimmed past in two seconds. Spoken aloud it is ninety seconds the listener
cannot skip or scan. A sentence cut at the wrong boundary is spoken before
anything downstream can reconsider.

**Deployment targets diverge structurally.** The same model must serve a cloud
API with autoscaling and a device with no network, no observability pipeline and
no rollback lever. An export pipeline and a bundle format stop being nice-to-have
and become the only way the second target exists.

---

## 2. Three stages and what each owes the next

### 2.1 Stage 1 — Research

**Directory:** `research/`, one subdirectory per spike.

A spike is the smallest runnable thing that can decide whether an idea is worth
building properly. Each `research/<nn>-<topic>/README.md` has exactly five
sections: `## Claim`, `## Falsifier`, `## Method`, `## Outcome`, `## What moved
into src/`.

Three rules keep it disposable. A spike is **never imported by `src/`** — the
dependency runs one way, so nothing downstream can come to rely on throwaway
code. A spike is **frozen once its outcome is recorded**; if the idea survives,
the implementation restarts from the codebase rather than from the spike's diff.
A spike **ships something runnable**, so a second person can re-run the falsifier
without the author.

The outcome is one of four values: `not run`, `run — falsifier passes`, `run —
falsified`, `run — inconclusive`. Anything not run says so.

**Exit criterion (Gate 1).** The falsifier was executed and the result recorded,
or the spike is explicitly `not run` and **nothing downstream claims its
result**. "What moved into src/" names the protocol or function development will
build, or says `nothing`.

**Contract owed to development:** a decision and a named interface. Not code.

### 2.2 Stage 2 — Development

**Directory:** `src/vmp/`, with `tests/`, `configs/`, `examples/`.

One package per subsystem, each with the same five properties:

- a `Protocol` for its backend;
- one standard-library reference implementation that runs in tests and demos;
- optional adapters that import their dependency **lazily inside the class**,
  never at module import;
- a `dry_run` path on every heavy operation that validates inputs and returns the
  plan or manifest as a dict;
- a CLI subcommand registered through a single registry.

`import vmp` succeeds with nothing but the standard library installed. That is
not minimalism for its own sake — it is what makes every dry run, every test and
every example executable in a CI job with no GPU, no model download and no
service.

**Exit criterion (Gate 2).** Tests pass in the default run — no network, no
models, no GPU, under about thirty seconds. The protocol is stable: a second
adapter can be written against it without changing the runtime that consumes it.
Every heavy operation has a dry run a CI job can execute. `examples/demo_*.py`
runs with no extras. Any number in the module's docs has a source file.

**Contract owed to productization:** a protocol that will not move, a manifest
that describes what a real run would do, and a test suite that runs anywhere.

### 2.3 Stage 3 — Productization

**Directory:** `deploy/` and `runbooks/`.

Dockerfiles, Kubernetes manifests with a base and per-environment overlays,
KubeRay `RayJob` and `RayService` specifications, a Feast feature repository, an
OpenTelemetry collector configuration, Prometheus recording and alerting rules,
Grafana dashboards, and nine operational runbooks.

**Exit criterion (Gate 3, into a bounded rollout).** SLOs are defined as
Prometheus rules with an error budget, not as prose. The release gate passes on
the candidate artifact and the `GateDecision` is stored in the registry beside
it. Every alert routes to a runbook and every runbook ends in a rollback step.
The rollout is bounded: canary first, with an automatic rollback trigger tied to
the SLO.

**Contract owed back to research:** production evidence. Traces, drift reports
and incidents are the context that starts the next cycle. The loop closes;
production is not the end of the path.

---

## 3. Three things that cross every boundary unchanged

The stages are useful only because something survives them intact.

### 3.1 The trace schema

One JSON object per line:

```json
{"ts": 1789399628.622, "session": "84acc7bd…", "turn": 1, "event": "playback.end",
 "span": "e63be86e2caa", "seq": 1, "ms": 0.063, "payload": {"response_ms": 1.021}}
```

`event` is `<stage>.start` or `<stage>.end`. Stages for a voice turn are
`listen, stt, retrieve, llm, segment.emit, tts, playback, turn`. A research
notebook, a unit test, a laptop run and a production pod write identical rows,
so a latency percentile means the same thing at every stage of maturity.

Two design choices carry most of the weight. **Stage names describe the role, not
the engine** — swapping the STT backend leaves `stt` meaning "audio became text",
so before and after are comparable. And **`seq` joins a sentence across stages**:
`segment.emit seq=3 → tts seq=3 → playback seq=3` is one sentence at three
points in its life, which is what makes the streaming design legible at all.

`playback.end` at `seq=1` carries `response_ms`. That is time to first audio, and
it exists because a row stamps it.

### 3.2 The shared types

`Utterance`, `Turn`, `Session`, `PreferencePair`, `FeatureRow`, `Document`,
`Chunk`, `ModelArtifact`, `EvalResult`, `GateDecision` are defined once and never
redefined. They serialise with `to_dict` / `from_dict`, so a JSONL file, a
registry entry and a trace stay readable without the package installed.

Lifecycle stages are plain string constants (`candidate`, `staging`,
`production`, `retired`) rather than enums, for the same reason: an artifact
record should be readable by a tool that has never imported this codebase.

### 3.3 The evidence rule

**A number with no source file does not go in a table.**

Measurements from the private predecessor project are cited with their cycle,
date and file — "alpha-core, cycle 5, 2026-09-12, `<file>`" — and are never
presented as this repository's benchmark. Demo output may be quoted only if the
demo actually prints it. Benchmarks, throughput figures and cost numbers are
never invented.

This is a discipline with teeth, and it produces uncomfortable results. All four
research spikes in this repository are `not run`. Each has a real measurement
behind it from the predecessor, and each of those measurements has a caveat that
makes it insufficient:

| Claim | Predecessor measurement | Why it is not a result here |
|---|---|---|
| Whisper LoRA adapts to an accent | 11/35 → 32/35 exact match | Measured on the **training set**, no held-out set |
| A smaller model trades correctness for latency | 192 ms vs 4,427 ms median; 2/6 vs 0/6 wrong | Six questions; the correctness figure is a two-item difference |
| One plan scales to Ray | — | Never met a cluster |
| Rule-derived pairs shift style via DPO | — | Never trained |

(All from alpha-core, cycle 5, 2026-09-12: `notebook_whisper_accent_lora.ipynb`
and `notebook_optimized.ipynb`.)

Marking any of those `run — falsifier passes` because a related number exists
elsewhere would be exactly the failure the format prevents. `not run` is a
status, and an honest one: it says the code path executes, the plan validates,
the harness works, and the experiment awaits hardware and a corpus.

---

## 4. Artifacts each stage must produce

A stage is complete when its artifacts exist, not when its work feels finished.

### 4.1 Research artifacts

| Artifact | Purpose |
|---|---|
| `README.md` with the five sections | The claim, the falsifier written **before** the run, the outcome |
| A runnable script or notebook | Someone else can re-run the falsifier without the author |
| A results file, if the falsifier was executed | The evidence, not a summary of it |
| The "what moved into src/" line | The interface development will build |

### 4.2 Development artifacts

| Artifact | Purpose |
|---|---|
| `Protocol` definition | The contract a second adapter can be written against |
| Reference implementation | Runs in tests and demos with no dependencies |
| Dry-run manifest | A reviewable plan, produced without importing anything heavy |
| `tests/test_<module>.py` | Passes with no network, no models, no GPU |
| TOML config in `configs/` | The recipe as data, hashable for lineage |
| CLI subcommand | The same command a developer and a cluster job both run |

The dry run deserves emphasis. Every heavy operation returns its plan as a dict:
a training run returns hashes, row counts and estimated steps; an export returns
each conversion step with the dependency it needs; a Ray submission returns the
entire `TorchTrainer(...)` call as serialisable data. This makes CI able to lint
configurations that reference models and datasets it will never download, and it
surfaces effects that would otherwise appear only in a training log — such as
optimizer step count falling when worker count rises, because the global batch is
`per_device_batch_size × gradient_accumulation × num_workers`.

### 4.3 Productization artifacts

| Artifact | Purpose |
|---|---|
| `ModelArtifact` with lineage | `config_hash`, `data_hash`, `git_sha`, metrics, stage |
| `GateDecision` stored beside it | Why this was promoted, as an object |
| SLO definitions and Prometheus rules | Objectives with an error budget, generated from one source |
| Runbook per alert | Every alert answerable; every runbook ends in rollback |
| `EdgeBundle` manifest | Checksums, lineage, minimum runtime version, policy |

---

## 5. The contract in practice: one metric across five stages

Time to first audio demonstrates the whole argument, because it touches every
boundary.

**Research** established the shape. Serial synthesis reaches first audio in
11,049 ms on an eight-sentence answer; streaming reaches it in 2,688 ms
(alpha-core, cycle 5, 2026-09-12, `README.md` "Streaming"). Both paths write the
same trace, so `playback.response_ms` is directly comparable — the comparison is
meaningful *because* the schema was shared, not despite it.

**Development** encoded the conditions that make streaming safe. Synthesis must
outrun playback, so `tts.queue_wait_ms` is recorded. Playback must be gapless, so
`underruns` and `gap_ms_max` are recorded. And the segmenter must never emit a
truncated sentence — a sentence cut at a decimal point or an abbreviation is
*spoken* wrong and cannot be unspoken. The segmenter emits only on provable
completion and holds a boundary at the end of the buffer until flush, because the
next token may turn `3.` into `3.5`. It is the one component shared verbatim by
research, cloud and edge runtimes, and that is not a coincidence: its correctness
is the precondition for the design rather than a quality improvement within it.

**Serving** stamps `response_ms` on the first `playback.end`.

**Evaluation** reads it: `vmp obs slo` computes `ttfa_p50_ms` and `ttfa_p95_ms`
from those rows and reports the source of each indicator alongside its value. The
release gate consumes `ttfa_p95_ms` as one of its rules.

**Operations** enforces it: `ttfa_p95` at or below 3,000 ms over thirty days,
rendered as a Prometheus alerting rule generated from the same SLO object, routed
to the `ttfa-slo-breach` runbook.

Five consumers, one number, no translation layer. That is what the contract
buys, and it is invisible until you try to do it without one.

---

## 6. Capability matrix

What each capability looks like at each stage of maturity.

| Capability | Research | Development | Productization |
|---|---|---|---|
| **Training** | Notebook LoRA on one GPU; train-set metric only | `TrainingPlan` with canonical hashing, SFT/DPO/Whisper runners, dry-run planner, local backend | Ray Train on KubeRay as a `RayJob`, registry lineage, gate before promotion |
| **Feature store** | Ad-hoc dicts per session | `FeatureView`, JSONL offline and in-memory online stores, point-in-time join, streaming updater fed from traces | Feast repository generated from the same views, Redis online store, scheduled materialisation |
| **RAG** | Top-k cosine over a repository, no floor | Line-range chunking, keyword prefilter over rare literals, one-hop graph expansion, RRF fusion, relevance floor, context packing for speech | pgvector/Qdrant and Postgres/Neo4j adapters, relevance dashboards, degradation runbook |
| **Serving** | One process, one microphone | `Session`/`Turn` loop, sentence segmenter, FastAPI app with WebSocket streaming, echo/stub backends | Ray Serve graph with four independently autoscaled deployments, vLLM adapter, TTFA SLO |
| **Edge** | Manual model conversion | Export pipeline (merge → ONNX/GGUF/MLX), quantisation validation, `EdgeBundle` manifest with checksums and policy | OTA canary with bundle diff, verification as a start precondition, offline policy enforced in code |
| **Evaluation** | Notebook exact match | Synthetic golden set with nine strata, WER/CER, win-rate, latency percentiles from traces, `GateDecision` | Gate in CI and in the promotion path; regression check against production; drift on WER |
| **Observability** | JSONL trace file | Stage tracer, metrics registry, PSI/KS drift detectors, SLO definitions and error budgets | OTel export, Prometheus rules generated from SLO objects, Grafana boards, alerts routed to runbooks |

Two patterns run across every row.

**The reference implementation is never thrown away.** The in-memory vector
store, the JSONL offline store, the stub TTS and the hashing embedder remain the
default in tests and demos after the production adapters exist. They are what
keeps the default test run free of network, models and GPUs, and they are what
makes a protocol violation obvious: if the reference and the adapter disagree,
the protocol was underspecified.

**Production maturity is mostly about evidence, not capability.** The difference
between development and productization in almost every row is not a better
algorithm. It is lineage, a gate, a dashboard, a runbook and a bounded rollout.

---

## 7. What this platform has not demonstrated

Stating this plainly is part of the method.

No model has been trained here. No speech has been transcribed, no adapter fitted,
no cluster started, no device shipped. The four research spikes are `not run`.
The SLO thresholds are targets to design against, not measurements. No
quantisation level has been benchmarked. No throughput, latency or cost figure
originates in this repository.

What exists: protocols with reference implementations and adapters; dry-run paths
that validate a plan and return it as data; an evaluation harness whose scorers
run on synthetic input; a trace schema exercised end to end; deployment manifests
for KubeRay, Prometheus and Feast; and nine runbooks.

That is a platform ready to receive experiments, described honestly. The
alternative — a repository full of plausible numbers with no source files — would
be more impressive and worth less, and it is precisely what the evidence rule
exists to prevent.

---

## 8. Conclusion

Stages are where applied ML work is lost, and contracts are what stop the loss. A
platform organised as a maturity path makes those contracts explicit: a spike
owes development a decision and a named interface; development owes production a
stable protocol, a reviewable plan and a test suite that runs anywhere; production
owes research its evidence, so the loop closes.

Three things must cross every boundary unchanged — the trace schema, the shared
types, and the evidence rule — and the third is the one most often skipped. A
number with no source file is not a small documentation problem. It is the point
at which the platform stops being able to tell you whether it is getting better.

---

## References

Full annotations in [References](../references.md).

- LoRA — Hu et al., 2021, arXiv:2106.09685. See
  [Parameter-efficient fine-tuning](../references.md#parameter-efficient-fine-tuning).
- QLoRA — Dettmers et al., 2023, arXiv:2305.14314. See
  [Parameter-efficient fine-tuning](../references.md#parameter-efficient-fine-tuning).
- Direct Preference Optimization — Rafailov et al., 2023, arXiv:2305.18290. See
  [Preference optimisation](../references.md#preference-optimisation).
- Whisper — Radford et al., 2022, arXiv:2212.04356. See
  [Speech](../references.md#speech).
- Retrieval-Augmented Generation — Lewis et al., 2020, arXiv:2005.11401. See
  [Retrieval](../references.md#retrieval).
- GraphRAG — Edge et al., 2024, arXiv:2404.16130. See
  [Retrieval](../references.md#retrieval).
- Ray — Moritz et al., 2018, arXiv:1712.05889. See
  [Distributed training and serving](../references.md#distributed-training-and-serving).
- vLLM / PagedAttention — Kwon et al., 2023, arXiv:2309.06180. See
  [Distributed training and serving](../references.md#distributed-training-and-serving).
- Ray Train, Ray Serve and KubeRay documentation. See
  [Distributed training and serving](../references.md#distributed-training-and-serving).
- Feast documentation. See [Feature store](../references.md#feature-store).
- ONNX Runtime, Optimum, llama.cpp and GGUF. See
  [Edge and export](../references.md#edge-and-export).
- OpenTelemetry and Prometheus. See
  [Observability and operations](../references.md#observability-and-operations).
- Google SRE Workbook, "Implementing SLOs". See
  [Observability and operations](../references.md#observability-and-operations).
- pgvector, Qdrant and Neo4j. See [Retrieval](../references.md#retrieval).
- faster-whisper, CTranslate2, MLX and Kokoro. See
  [Speech](../references.md#speech).
- LiveKit Agents. See [Voice agent runtime](../references.md#voice-agent-runtime).

Measurements attributed to alpha-core are from a private repository, cited by
cycle, date and file, and are not this repository's benchmarks.
