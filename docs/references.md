# References

The papers, specifications and documentation this platform leans on, grouped by
the part of the system they inform. Each entry says in one line why it matters
*here*, not what it is in general.

Entries are things this design actually uses. Where a technique is implemented
behind a `Protocol` with a standard-library reference implementation, the paper
describes the adapter, not the default.

## Parameter-efficient fine-tuning

- **LoRA: Low-Rank Adaptation of Large Language Models.** Hu et al., 2021.
  [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) —
  Every fine-tune in this platform is a low-rank adapter over a named base
  model, never a full-weight update. It is what makes a per-speaker Whisper
  adapter and a spoken-style LLM adapter affordable, and what lets one base
  model serve several adapters.
- **QLoRA: Efficient Finetuning of Quantized LLMs.** Dettmers et al., 2023.
  [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) —
  The path to training an adapter on a single consumer GPU, and the source of
  the quantised-base intuition that also applies on the
  [edge export](guides/edge-export.md) side.
- **PEFT documentation.** <https://huggingface.co/docs/peft> —
  `LoraConfig` is mirrored field-for-field by `vmp.training.plan.LoraConfig`, so
  the plan can be validated and hashed without importing PEFT.

## Preference optimisation

- **Direct Preference Optimization: Your Language Model is Secretly a Reward
  Model.** Rafailov et al., 2023.
  [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) —
  The method behind [DPO training](guides/training-dpo.md): preference pairs and
  a `beta` against a frozen reference, with no reward model and no RL loop. The
  reference-free-of-a-second-model property is what makes the PEFT trick
  (adapter-disabled base as its own reference) possible.
- **TRL documentation.** <https://huggingface.co/docs/trl> —
  `SFTTrainer` and `DPOTrainer` are the trainers both runners wrap. The `beta`
  default of 0.1 and the left-truncation behaviour of `max_prompt_length` come
  from here.

## Speech

- **Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).**
  Radford et al., 2022.
  [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) —
  The STT model family the platform adapts. The paper's inverse text
  normalisation behaviour is why golden-set references must be written the way
  ASR writes them (`555-0199`, not "five five five…").
- **faster-whisper.** <https://github.com/SYSTRAN/faster-whisper> —
  The CPU and CUDA STT adapter.
- **CTranslate2.** <https://github.com/OpenNMT/CTranslate2> —
  faster-whisper's inference engine. Worth reading for one operational fact:
  it has no Metal backend, so `device = "cpu"` on macOS is the only option, not
  a default.
- **MLX.** <https://github.com/ml-explore/mlx> —
  The Apple-silicon inference path. It converts and runs Whisper; it has no
  training API, which is the framework mismatch that forces the
  merge-then-convert step in [Whisper accent LoRA](guides/whisper-accent-lora.md).
- **Kokoro.** <https://github.com/hexgrad/kokoro> —
  The TTS adapter. Its real-time factor is what makes sentence-at-a-time
  streaming safe: synthesis must outrun playback or the queue starves.

## Retrieval

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.**
  Lewis et al., 2020.
  [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) —
  The pattern behind `vmp.rag`: retrieve, then condition generation on what was
  retrieved. The platform's departure from the canonical form is the relevance
  floor — retrieving nothing is a valid outcome.
- **From Local to Global: A Graph RAG Approach to Query-Focused
  Summarization.** Edge et al., 2024.
  [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) —
  The graph-expansion path: follow entities one hop to assemble an answer split
  across sections instead of truncating at the first chunk.
- **pgvector.** <https://github.com/pgvector/pgvector> —
  The production vector store adapter, and the cheapest production shape: one
  Postgres instance holds both the vectors and the graph edges.
- **Qdrant.** <https://qdrant.tech/documentation/> —
  The dedicated vector-database adapter, for corpora that outgrow Postgres.
- **Neo4j.** <https://neo4j.com/docs/> —
  The graph-store adapter for entity and relation expansion at scale.

## Distributed training and serving

- **Ray: A Distributed Framework for Emerging AI Applications.**
  Moritz et al., 2018.
  [arXiv:1712.05889](https://arxiv.org/abs/1712.05889) —
  The actor-and-task model underneath Ray Data, Ray Train and Ray Serve, and
  the reason one plan can target a laptop or a cluster without a rewrite.
- **Ray Train documentation.**
  <https://docs.ray.io/en/latest/train/train.html> —
  `TorchTrainer`, `ScalingConfig` and `RunConfig`, which
  `RayTrainLauncher.build_config` emits as plain serialisable dicts.
- **Ray Serve documentation.**
  <https://docs.ray.io/en/latest/serve/index.html> —
  Deployments, the `serve` config schema v2 document that `vmp serve ray
  --yaml` prints, and the `target_ongoing_requests` autoscaling signal.
- **KubeRay.** <https://ray-project.github.io/kuberay/> —
  `RayJob` and `RayService`. The update semantics matter operationally: a
  `serveConfigV2` change applies in place, a `rayClusterConfig` change starts a
  second cluster.
- **Efficient Memory Management for Large Language Model Serving with
  PagedAttention (vLLM).** Kwon et al., 2023.
  [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) —
  Continuous batching and a paged KV cache are what keep `llm.ttft_ms` flat as
  concurrent sessions rise — and `ttft_ms` is the floor on time-to-first-audio.
- **vLLM documentation.** <https://docs.vllm.ai/> —
  The OpenAI-compatible streaming endpoint the runtime speaks, and multi-LoRA
  serving, which lets several adapters share one base model.

## Feature store

- **Feast documentation.** <https://docs.feast.dev/> —
  The `Entity` / `FeatureView` / `FileSource` vocabulary that
  `vmp features feast-export` generates, and the online/offline split with
  point-in-time correctness that `vmp.features.offline` implements directly.

## Edge and export

- **ONNX Runtime.** <https://onnxruntime.ai/docs/> —
  The most portable inference target, and the dynamic int8 quantisation used by
  the `onnx` export path.
- **Optimum.** <https://huggingface.co/docs/optimum> —
  `main_export` is the converter behind `vmp edge export --target onnx`.
- **llama.cpp and the GGUF format.**
  <https://github.com/ggml-org/llama.cpp> —
  The CPU-first edge runtime and its quantisation ladder (`Q4_K_M`, `Q8_0`),
  reached through `convert_hf_to_gguf` and `llama-cpp-python`.

## Observability and operations

- **OpenTelemetry specification.** <https://opentelemetry.io/docs/specs/otel/> —
  The trace and span model `OtlpSink` maps stage rows onto. The platform keeps
  JSONL as the durable evidence and uses OTel for the live system.
- **Prometheus documentation.** <https://prometheus.io/docs/> —
  The exposition format `vmp.observability.metrics` renders by hand, and the
  recording and alerting rules `to_prometheus_rules_yaml` generates from the SLO
  objects.
- **The Site Reliability Workbook, Chapter 2: Implementing SLOs.** Google.
  <https://sre.google/workbook/implementing-slos/> —
  SLI, SLO, error budget and burn rate as this platform uses them. The insight
  that shapes `vmp obs slo`: a metric inside its objective but burning budget
  fast is the interesting state, and burn rate sees it weeks before a threshold
  alert does.

## Voice agent runtime

- **LiveKit Agents.** <https://docs.livekit.io/agents/> —
  The real-time transport and agent-orchestration layer a production voice
  deployment sits on: room-based audio, turn detection and barge-in. This
  repository's runtime covers the turn loop and leaves transport to it.

## A note on citation

Measurements from the private predecessor project are cited in the form
"alpha-core, cycle 5, 2026-09-12, `<file>`" and are never presented as this
repository's benchmark. That project is not linked here because it is not
public. Anything this repository has not measured says so.
