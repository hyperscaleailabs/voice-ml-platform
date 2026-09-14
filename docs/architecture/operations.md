# Production operations: deploy, observe, scale, recover, improve

The fifth lens asks how the system is run. The cycle is **Deploy -> Observe ->
Scale -> Recover -> Improve**, and each step has a concrete artifact in
`deploy/` or `runbooks/`.

## Deploy paths

Three targets, one package.

| Target | What runs | How it is deployed | Extras |
|---|---|---|---|
| **Local** | FastAPI app with echo/stub or real adapters, one process | `vmp serve run --config configs/serve.toml` | `serve`, optionally `audio` |
| **Cloud API** | Ray Serve graph: runtime, STT, TTS, vLLM deployments | `RayService` on KubeRay, kustomize overlay per environment | `ray`, `serve`, `vllm` |
| **Edge** | Edge runtime with offline-only backends | `EdgeBundle` pushed over an OTA channel, verified on device | `edge` or `mlx` |

Training runs as a `RayJob` on the same KubeRay operator; a dry run prints the
job spec. Feature materialisation is a scheduled job. All manifests live in
`deploy/k8s/` with a `base/` and overlays for `dev`, `staging`, `prod`.

The image is built once (`deploy/docker/Dockerfile`) with a build argument
selecting the extras, so the serving image and the training image differ only
in what is installed, never in the code.

## SLOs

Defined in `vmp.observability.slo` and rendered as Prometheus recording and
alerting rules in `deploy/prometheus/`. The SLOs are stated as objectives with
a window and an error budget; the thresholds in the repository are placeholders
to be set from measured traffic, not measurements.

| SLO | Indicator | Source |
|---|---|---|
| Time-to-first-audio p50 and p95 | `playback.response_ms` on `seq=1` | trace -> metrics |
| Time-to-first-token p95 | `llm.ttft_ms` | trace -> metrics |
| Turn error rate | `turn.end` with `outcome != ok` over all turns | trace -> metrics |
| STT quality | WER on the golden set (batch) and on labelled samples | `vmp eval golden` |
| Availability | `/readyz` success ratio | probe |
| Playback integrity | `playback.summary.underruns == 0` | trace |

An error budget burn alert at two rates (fast burn, slow burn) pages for the
first and tickets for the second. Every alert names its runbook.

## Drift

`vmp.observability.drift` implements two detectors:

- **PSI** (population stability index) on feature distributions from the online
  store versus the training snapshot.
- **KS** (Kolmogorov-Smirnov) on continuous signals such as `stt.ms`,
  `llm.ttft_ms` and the WER estimate on labelled samples.

Drift is computed as a scheduled job over a window and written as a metric;
the alert threshold is in the rules file. Drift does not roll back by itself; it
opens the `drift-detected` runbook, which decides between re-materialising
features, re-running the gate on the production artifact, and scheduling a
retrain.

## Dashboards

`deploy/grafana/` holds board definitions:

1. **Voice turn**: stage latencies as stacked percentiles, TTFA against SLO,
   TTFT, sentences per turn, underruns.
2. **Quality**: golden-set WER by stratum over time, exact match, win-rate of
   the current production artifact versus the previous one, judge scores.
3. **Retrieval**: hit rate above the floor, chunks kept per turn, score
   distribution, keyword-prefilter hit rate, graph expansion depth.
4. **Serving**: replicas per deployment, queue depth, GPU utilisation, vLLM
   batch size and KV-cache usage.
5. **Training**: RayJob status, steps per second, loss, eval metrics at the end.
6. **Edge**: bundle versions in the fleet, verification failures, offline
   policy compliance, canary cohort health.

## Runbooks

Every alert routes to one runbook in `runbooks/`, and every runbook has the
same shape: symptoms, first checks, mitigation, rollback, verification, and
what to record. The set is listed on the [Runbooks](../runbooks.md) page.

## On-call

One rotation covers the serving path; training and edge pages route to the
same rotation during business hours and to a ticket otherwise. The on-call
runbook defines severity: SEV1 is TTFA SLO fast burn or availability below
target; SEV2 is quality regression or drift; SEV3 is anything with a working
fallback. The on-call engineer may roll back a model, switch a model tier, or
disable retrieval without approval; promoting a new artifact is not an on-call
action.

## Capacity levers

Ordered from cheapest to most expensive to pull.

| Lever | What it changes | Cost |
|---|---|---|
| Retrieval depth (`top_k`, graph expansion) | prompt length -> TTFT | none; small quality risk |
| History window | prompt length -> TTFT | none; context loss on long sessions |
| Sentence buffer before first playback | trades TTFA for underrun safety | latency |
| Model tier | TTFT, GPU memory | correctness (see the small-model fact on the reliability page) |
| vLLM max batch and KV-cache budget | throughput per replica | memory |
| Ray Serve replicas per deployment | concurrency | GPU nodes |
| Node pool autoscaling | headroom | money and start-up time |
| Quantisation of the served model | memory and speed | accuracy, must re-run the gate |

The predecessor's guidance on GPU cost applies: avoid paying for idle GPU
capacity while preserving acceptable start-up. Ray Serve's autoscaler scales on
in-flight requests per replica with a minimum of one replica for the runtime
and zero for the vLLM deployment in non-production overlays; the cold-start
cost of a scale-from-zero is a measured number to obtain per environment, not a
number this documentation supplies.

## Improve: closing the loop

Production evidence is the input to the next research cycle. Three feeds are
built in:

- **Traces -> golden set.** Turns that failed a quality check (`turn.outcome`,
  judge score, user correction) are candidates for the golden set. A real
  recording lands as `source: "human"` and is never mixed with synthetic rows.
- **Traces -> features.** `turn.end` rows materialise into online features
  such as recent TTFA and turn count, which the runtime can use to choose a
  tier per session.
- **Drift -> hypothesis.** A drift alert that is not explained by
  infrastructure becomes a research spike with a claim and a falsifier, not a
  ticket to "retrain".
