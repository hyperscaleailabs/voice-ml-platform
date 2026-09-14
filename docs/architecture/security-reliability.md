# Secure and reliable: identity, protection, reliability

The fourth lens asks who can do what, what must be protected, and what happens
when something fails. The scaffold is **Identity -> Protection -> Reliability**.

## Identity

### Authentication

The API authenticates every request with a bearer token in the `Authorization`
header. Tokens are issued outside the platform (an identity provider or a
service-account secret); the service validates them and maps them to a
principal. `/healthz`, `/readyz` and `/metrics` are exempt so that the
orchestrator and the scraper need no credentials, and they are exposed only on
the cluster-internal port.

### Authorization

A session belongs to the principal that created it. `GET /v1/sessions/{id}`,
`POST /v1/sessions/{id}/turns` and the WebSocket stream reject any other
principal. There is one privileged scope, `vmp:admin`, used by the registry
promotion path and the drift-acknowledge endpoint; nothing in the voice path
needs it.

### Service identity

Inside the cluster, services identify each other with Kubernetes service
accounts and mutual TLS at the mesh or ingress layer. The runtime talks to
vLLM, Redis, pgvector and Neo4j with per-service credentials mounted as secrets,
never with a shared password.

## Protection

### Network policy

`deploy/` ships a default-deny `NetworkPolicy` for the namespace. Allowed
edges:

| From | To | Port |
|---|---|---|
| ingress | serving | HTTPS / WSS |
| serving | vLLM (Ray Serve) | internal |
| serving | Redis (online store) | internal |
| serving | pgvector / Qdrant, Neo4j | internal |
| serving, training | OTel collector | OTLP |
| Prometheus | serving `/metrics` | internal |
| training (RayJob) | object storage, registry | egress allow-list |

Nothing else reaches the model servers. Training jobs have no route to the
serving namespace.

### Secrets

Secrets are mounted, never baked into images or configs. Configs are TOML and
must not contain credentials; `config.py` reads `VMP_*` environment variables
for anything secret, and the manifests source those from Kubernetes secrets or
an external secrets operator. A test asserts that no file under `configs/`
contains a key, a token, or an absolute local path.

### PII scrub

Transcripts are scrubbed before they leave the runtime for any store other
than the short-lived session: `vmp.data.pii` replaces names, phone numbers,
addresses, card-like numbers and emails with typed placeholders. The trace
holds timings and counts, never transcript text or audio. Training corpora go
through the same scrub before `data_hash` is computed, so a hash also attests
to a scrubbed dataset.

### Audio

Audio is the most sensitive artifact. It is held in memory for the duration of
a turn, written to disk only where the retention policy allows (golden-set
audio is synthetic and may be kept; user audio defaults to not stored), and is
never sent to a third party by the reference configuration.

### Edge offline policy

An `EdgeBundle` carries a policy document with three fields:
`network: "offline" | "metrics-only" | "full"`, `audio_egress: false`, and
`telemetry_egress: "aggregates"`. The edge runtime enforces the policy at the
adapter layer: with `network = "offline"` no adapter may open a socket, and the
bundle verifier refuses a bundle whose policy is weaker than the device's
configured minimum. The default is offline.

## Reliability

### Failure modes and fallbacks

| Failure | Detected by | Fallback | Runbook |
|---|---|---|---|
| STT adapter slow or down | `stt.ms` above budget; adapter timeout | Retry once; then the CPU adapter if the GPU one failed; then end the turn with a spoken error | `incident-response` |
| LLM time-to-first-token high | `llm.ttft_ms` p95 alert | Shorter context (drop retrieval), then smaller model tier | `ttfa-slo-breach` |
| LLM unavailable | health check fails | Serve fails over to the standby deployment; sessions see one slow turn | `incident-response` |
| Retrieval returns nothing above the floor | `retrieve` payload `kept = 0` | Answer without context and say so; never pack low-score chunks | `rag-relevance-degraded` |
| Retrieval store down | connection error | Skip retrieval for the turn; alert | `rag-relevance-degraded` |
| TTS queue starves | `tts.queue_wait_ms`, `playback.underruns > 0` | Increase sentence buffer to two before first playback | `ttfa-slo-breach` |
| Online feature store down | Redis timeout | Use defaults from the `FeatureView`; flag the turn | `incident-response` |
| Model regression after promotion | WER drift, win-rate drop | Roll back the registry stage to the previous artifact | `rollback-model` |
| Edge bundle checksum mismatch | verifier at load | Keep the previous bundle; report | `edge-bundle-verify-failed` |
| Segmenter emits a truncated sentence | test suite; `segment.emit` audit | Cannot be fixed at runtime; blocked by the release gate | — |

Timeouts, retries and circuit breakers live in the adapters, not in the
runtime. The runtime sees a protocol call that either returns or raises one
typed error, and decides the fallback.

### Graceful degradation order

When capacity is short the platform degrades in a fixed order, cheapest loss
first: drop graph expansion, drop vector retrieval, shorten history, switch to
the smaller model tier, then queue new sessions. Each step is a config value
the on-call can set, and each is visible in the trace payload so that a
degraded turn is never mistaken for a healthy one.

The last step has a measured cost. In the predecessor, a 270M-parameter model
answered in a median 192 ms where the 4B model took 4,427 ms, but got 2 of 6
factual questions wrong where the 4B model got 0 wrong (alpha-core, cycle 5,
2026-09-12, `notebook_optimized.ipynb`). The platform reports the speed-up and
the error rate together and never switches tiers silently.

### Release gates

Nothing reaches `production` in the registry without a `GateDecision`. The
gate (`vmp eval gate`) runs the golden set (WER overall and per stratum), exact
match, preference win-rate against the current production artifact, and
latency percentiles from a trace, then compares each to a threshold in the
gate config. `passed` is false if any threshold fails, and `reasons` lists
which. The decision is stored beside the artifact.

Two properties make the gate trustworthy:

- **It is independent.** The gate runs in CI on the artifact, not in the
  training job that produced it. A training run's own account of its metrics
  is evidence about the run, not about the model.
- **It is the same code at every stage.** The gate a notebook can call and the
  gate the promotion path calls are one function, so a number that passed in
  development is the same number in production.

### Rollback

Rollback is a registry operation: set the previous artifact back to
`production`, and the serving deployment reloads the adapter on its next health
check. Because adapters are small and the base model is unchanged, rollback
completes in the time it takes to load one adapter. For the edge, rollback is
the verifier keeping the last good bundle. `runbooks/rollback-model.md` is the
procedure.
