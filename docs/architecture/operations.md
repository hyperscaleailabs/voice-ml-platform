# Production operations: deploy, observe, scale, recover, improve

The fifth lens asks how the system is run. The cycle is **Deploy -> Observe ->
Scale -> Recover -> Improve**, and each step has a concrete artifact in
`deploy/` or `runbooks/`.

## Deploy paths

Three targets, one package.

| Target | What runs | How it is deployed | Extras |
|---|---|---|---|
| **Local** | FastAPI app with echo/stub or real adapters, one process | `vmp serve api --config configs/serving.toml` | `serve`, optionally `audio` |
| **Cloud API** | Ray Serve graph: runtime, STT, TTS, vLLM deployments | `RayService` on KubeRay (`deploy/ray/rayservice-voice.yaml`), kustomize overlay per environment | `ray`, `serve`, `vllm` |
| **Edge** | Edge runtime with offline-only backends | `EdgeBundle` pushed over an OTA channel, verified on device | `edge` or `mlx` |

`vmp serve api --dry-run` prints the resolved plan — host, port, the three
backend choices and their settings, the history and context limits, and the
trace path — without importing FastAPI or binding a port. It is the fastest way
to confirm which config a container actually picked up.

Training runs as a `RayJob` on the same KubeRay operator
(`deploy/ray/rayjob-sft.yaml`, `deploy/ray/rayjob-dpo.yaml`), whose `entrypoint`
is the same `vmp train …` command a developer runs. A dry run does **not** print
a `RayJob` manifest: it prints the training manifest, which for
`compute.backend = "ray"` carries a `ray` block holding the entire
`ray.train.torch.TorchTrainer(...)` call as data. Feature materialisation is a
scheduled job (`deploy/k8s/base/feast-materialize-cronjob.yaml`). All manifests
live in `deploy/k8s/` with a `base/` and two overlays, `local` and `cloud`.

There are three images, not one, because their bases differ:
`deploy/docker/Dockerfile.api` (slim Python, `serve,obs`, a `/healthz`
healthcheck), `deploy/docker/Dockerfile.train` (a CUDA PyTorch base,
`train,ray,eval`) and `deploy/docker/Dockerfile.edge` (slim Python,
`edge,obs`). Each installs the same package from the same source tree and
differs only in its extras and entrypoint, never in the code.

## SLOs

Defined in `vmp.observability.slo` and rendered as Prometheus recording and
alerting rules in `deploy/prometheus/`. The SLOs are stated as objectives with
a window and an error budget; the thresholds in the repository are placeholders
to be set from measured traffic, not measurements.

Three objectives are defined, and no more, because an objective with no
indicator behind it is decoration:

| SLO (`name`) | Indicator | Objective | Where the number comes from |
|---|---|---|---|
| `ttfa_p95` | `ttfa_p95_ms` | `<= 3000` over 30 d | `playback.end` payload `response_ms` |
| `error_rate` | `error_rate` | `<= 0.01` over 30 d | turns with `payload.error`, or with no `turn.end` |
| `wer` | `wer` | `<= 0.05` over 30 d | `vmp eval golden`, passed in with `--wer` |

`vmp obs slo --trace <trace.jsonl> --config configs/slo.toml` computes them.
It also derives `ttfa_p50_ms`, which is reported but is not an objective. Two
signals that appear on the dashboards — `llm.ttft_ms` and per-stage p95 — are
indicators without objectives: they explain a TTFA breach rather than defining
one.

`deploy/prometheus/rules/voice-slo.yml` holds the recording rules
(`vmp:ttfa_ms:p50_5m`, `vmp:ttfa_ms:p95_5m`, `vmp:stage_ms:p95_5m`,
`vmp:turns:rate_5m`, `vmp:error_ratio:` at 5 m, 30 m, 1 h and 6 h) and seven
alerts: `VoiceTTFAP95High`, `VoiceErrorBudgetBurnFast`,
`VoiceErrorBudgetBurnSlow`, `VoiceWERHigh`, `ApiDown`,
`EdgeBundleVerifyFailed` and `FeatureDriftPSIHigh`. The two burn-rate alerts
are the fast/slow pair: the fast one pages, the slow one tickets. Every alert
carries a `runbook` annotation naming the file in `runbooks/`.

## Drift

`vmp.observability.drift` implements two statistics, both exposed through
`DriftDetector.check` and through `vmp obs drift --reference <a> --current <b>`,
which reads two numeric series and reports a `DriftReport`:

- **PSI** (population stability index) — for feature distributions, the online
  store's current window against the training snapshot.
- **KS** (Kolmogorov-Smirnov) — for continuous signals pulled from traces, such
  as stage `ms` values and `llm.ttft_ms`.

Drift is computed as a scheduled job over a window and written as a metric;
the alert threshold is in the rules file. Drift does not roll back by itself; it
opens the `drift-detected` runbook, which decides between re-materialising
features, re-running the gate on the production artifact, and scheduling a
retrain.

## Dashboards

`deploy/grafana/` holds one board, `dashboards/voice-agent.json` ("Voice
agent"), provisioned with its Prometheus datasource from `provisioning/`. One
board, not one per concern, because every panel on it is backed by a metric
something in this repository actually exports. Five rows:

1. **Latency**: time-to-first-audio p50 and p95, stage latency p95 by stage,
   LLM time-to-first-token p50 and p95, stage share of turn.
2. **Traffic and errors**: turns per second, error rate, API up, active
   sessions.
3. **Quality and drift**: golden-set WER, feature drift (PSI).
4. **Ray cluster**: GPU utilisation, Serve replicas, queued requests. The row
   title in the JSON says "placeholders until KubeRay metrics are scraped", and
   that caveat is part of the board.
5. **Edge fleet**: bundle version by device count, bundle verify failures over
   15 minutes.

There are no panels for retrieval depth, training loss or judge scores,
because nothing exports those metrics yet. A panel with no series behind it
reads as an outage.

## Runbooks

Every alert's `runbook` annotation names a file in `runbooks/`, and the
incident runbooks share one shape: **Symptom, Severity, Diagnose, Mitigate,
Verify, Follow-up**. `ttfa-slo-breach.md`, `drift-detected.md`,
`rag-relevance-degraded.md`, `edge-bundle-verify-failed.md` and
`rollback-model.md` all follow it. The three standing procedures —
`on-call.md`, `incident-response.md` and `data-retention-and-privacy.md` — are
organised around their own content instead, because a rotation handover is not
a symptom. The set is listed on the [Runbooks](../runbooks.md) page.

## On-call

One rotation covers the voice agent API, the Ray training and serving clusters,
the feature store, the RAG stores and the edge fleet, with a primary and a
secondary (`runbooks/on-call.md`). The secondary is not a spare pair of hands:
they take over after the primary has been heads-down for an hour, and they run
comms during a SEV-1.

Severity is defined in `runbooks/incident-response.md`, on user impact rather
than on which subsystem broke:

| Severity | Meaning | Response |
|---|---|---|
| **SEV-1** | The agent does not answer, for everyone | Page, incident commander within 5 min |
| **SEV-2** | Badly degraded, or a subset broken | Page, respond within 15 min |
| **SEV-3** | Noticeable but working | Next business day |
| **SEV-4** | Cosmetic or internal | Backlog |

Audio or transcript data exposed is SEV-1 regardless of how few users are
affected. A model that shipped while failing a release gate is SEV-2.

What pages and what becomes a ticket is a table in `runbooks/on-call.md`; the
rule behind it is to page on user impact and on what will become user impact
within the shift. The on-call engineer may reverse a change without approval —
`rollback-model.md` covers demoting a version in the registry, rolling the Ray
Serve graph back and rolling an edge bundle back. Promoting a new artifact to
`production` is not an on-call action.

## Capacity levers

Ordered from cheapest to most expensive to pull.

| Lever | What it changes | Cost |
|---|---|---|
| Retrieval depth (`rag.retrieval.top_k`, `graph_hops`) | prompt length -> TTFT | none; small quality risk |
| History window (`serving.max_history_turns`, `max_context_chars`) | prompt length -> TTFT | none; context loss on long sessions |
| Model tier (`serving.backends.*.model`) | TTFT, GPU memory | correctness (see the small-model fact on the reliability page) |
| Concurrency per replica (`target_ongoing_requests`, `max_ongoing_requests`) | throughput per replica | queueing, so TTFA |
| Ray Serve replicas per deployment (`min_replicas`, `max_replicas`) | concurrency | GPU nodes |
| Ray worker group autoscaling (`minReplicas`, `idleTimeoutSeconds`) | headroom | money and start-up time |
| Quantisation of the exported model (`vmp edge export --quant`) | memory and speed | accuracy, must re-run the gate |

Every knob in the middle column is a real key: the first two in
`configs/rag.toml` and `configs/serving.toml`, the next three in the
`serveConfigV2` block of `deploy/ray/rayservice-voice.yaml`, and the worker
group in `deploy/ray/raycluster.yaml`.

The predecessor's guidance on GPU cost applies: avoid paying for idle GPU
capacity while preserving acceptable start-up. Ray Serve's autoscaler scales on
in-flight requests per replica; every serving deployment keeps `min_replicas`
at one or more, so nothing on the serving path scales to zero. The GPU worker
group of the training cluster does — `minReplicas: 0` with
`idleTimeoutSeconds: 300` — and the cold-start cost of that scale-from-zero is
a measured number to obtain per environment, not a number this documentation
supplies. `runbooks/capacity-and-cost.md` is the procedure.

## Improve: closing the loop

Production evidence is the input to the next research cycle. Three feeds are
built in:

- **Traces -> golden set.** Turns that failed a quality check (a
  `payload.error` on a span, a judge score, a user correction) are candidates
  for the golden set. A real recording lands as `source: "human"` and is never
  mixed with synthetic rows.
- **Traces -> features.** `SessionFeatureUpdater` in `vmp.features.stream`
  folds trace events into the `session_features` view and pushes a snapshot
  online each turn: `turn_count` from `turn.end`, `avg_response_ms` from
  `playback.end` `response_ms`, `avg_user_utterance_s` from `listen.end`,
  plus `last_intent` and `accent_profile` when the backends supply them. The
  runtime can read those back to choose a tier per session.
- **Drift -> hypothesis.** A drift alert that is not explained by
  infrastructure becomes a research spike with a claim and a falsifier, not a
  ticket to "retrain".
