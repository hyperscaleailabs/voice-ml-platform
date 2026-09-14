# Guide: Observability

## Purpose

For a voice agent, observability is not an add-on that reports on the system —
it *is* the measurement system. The headline metric, time to first audio, only
exists because a trace row stamps it. The evaluation gate consumes a latency
percentile computed from traces. The feature store is fed from trace events.
Drift is detected on series extracted from traces.

So the trace schema is the platform's most load-bearing interface, and it is
deliberately boring: **one JSON object per line**, the same in a research
notebook, a unit test and a production pod.

```json
{"ts": 1789399628.622, "session": "84acc7bd…", "turn": 1, "event": "playback.end",
 "span": "e63be86e2caa", "seq": 1, "ms": 0.063, "payload": {"response_ms": 1.021}}
```

| Key | Meaning |
|---|---|
| `ts` | wall-clock time of the row |
| `session`, `turn` | which conversation, which exchange |
| `event` | `<stage>.start` or `<stage>.end` |
| `span` | id pairing a start with its end; nested spans record `parent` in the payload |
| `seq` | sentence number for per-sentence stages (`segment.emit`, `tts`, `playback`) |
| `ms` | duration, on the `.end` row only |
| `payload` | stage-specific fields |

Stages, in pipeline order: `listen, stt, retrieve, llm, segment.emit, tts,
playback, turn`. `llm` carries `ttft_ms`; `playback` on `seq=1` carries
`response_ms`, which is time to first audio. Because stage names describe the
*role* and not the engine filling it, a percentile means the same thing before
and after a backend swap.

An exception inside a span stamps `payload.error` with the exception type and
re-raises, so a failed turn is visible in the trace rather than being a gap.

## Config

`configs/slo.toml`:

```toml
# Service level objectives for the voice agent.
# These are targets to design against, not measurements. Measured values come
# from `vmp obs slo --trace <trace.jsonl>`.

[[slo]]
name = "ttfa_p95"
indicator = "ttfa_p95_ms"
objective = 3000.0
window = "30d"
comparator = "<="
description = "target: p95 time to first audio at or below 3000 ms"

[[slo]]
name = "error_rate"
indicator = "error_rate"
objective = 0.01
window = "30d"
comparator = "<="
description = "target: at most 1% of turns error"

[[slo]]
name = "wer"
indicator = "wer"
objective = 0.05
window = "30d"
comparator = "<="
description = "target: golden-set WER at or below 5%"
```

The comment is part of the file on purpose. These are objectives to design
against; nothing in this repository has measured them.

## CLI

```bash
vmp obs summarise --trace .vmp/trace.jsonl                 # one line per turn
vmp obs summarise --trace .vmp/trace.jsonl --json          # same, structured
vmp obs slo --trace .vmp/trace.jsonl --config configs/slo.toml --wer 0.027
vmp obs drift --reference baseline_ttfa.txt --current today_ttfa.txt \
    --psi-threshold 0.2 --ks-threshold 0.1
```

Produce a trace by running the service with `--trace`:

```bash
vmp serve api --config configs/serving.toml --trace .vmp/trace.jsonl
```

## Reading a turn

`vmp obs summarise` prints one line per turn, in the same shape the predecessor
project used, so a trace from either is read the same way:

```text
84acc7bd… turn 1 ttfa 1ms: retrieve 1x 0ms | segment.emit 2x 0ms (first 0, max 0) |
  tts 2x 0ms (first 0, max 0) | llm 1x 1ms | playback 2x 0ms (first 0, max 0) | turn 1x 1ms
```

(Those milliseconds are the stub backends: a template LLM and a silent TTS. They
measure the plumbing, not a model.)

Two details make this readable at a glance. **Repeated stages accumulate** into
count, total, first and max rather than overwriting — `tts 2x` with a `first`
and a `max` shows immediately whether the first sentence was the slow one, which
is the one that matters for time-to-first-audio. And `ttfa` is pulled from the
`playback.end` payload, not recomputed, so it is the same number the SLO and the
gate see.

`--json` emits `{"session", "turn", "ttfa_ms", "stages"}` per turn for
downstream processing.

## SLOs and error budgets

`vmp obs slo` computes the indicators from trace rows and compares them to the
objectives. Real output:

```json
{
  "sli": {
    "values": {"ttfa_p50_ms": 0.557, "ttfa_p95_ms": 0.975, "error_rate": 0.0, "wer": 0.027},
    "n_turns": 3,
    "sources": {
      "ttfa_p50_ms": "trace: playback.end response_ms",
      "ttfa_p95_ms": "trace: playback.end response_ms",
      "error_rate": "trace: turns with payload.error or missing turn.end",
      "wer": "eval: golden set"
    }
  },
  "budgets": [
    {"slo": "ttfa_p95", "indicator": "ttfa_p95_ms", "objective": 3000.0,
     "measured": 0.975, "burn_rate": 0.000325, "consumed": 0.000325,
     "remaining": 0.999675, "met": true},
    {"slo": "error_rate", "objective": 0.01, "measured": 0.0, "burn_rate": 0.0,
     "remaining": 1.0, "met": true},
    {"slo": "wer", "objective": 0.05, "measured": 0.027, "burn_rate": 0.54,
     "consumed": 0.54, "remaining": 0.46, "met": true}
  ]
}
```

`sources` is the evidence rule made mechanical: every indicator says where it
came from. A turn counts as an error when any `.end` row carries
`payload.error`, **or** when it has a `turn.start` with no `turn.end` — a
crashed turn is an error even though it never reported one.

`burn_rate` is measured / objective: 1.0 is exactly on target, 2.0 is consuming
the budget twice as fast as allowed. `remaining` goes negative when the budget
is exhausted. That third row is the one to read carefully: `wer` is *inside* its
objective and has already consumed 54% of the budget. A metric that is passing
and burning fast is the interesting state, and an alert on burn rate sees it
weeks before an alert on the threshold does.

The command exits non-zero when any budget is not met, so CI can enforce it.

### Prometheus rules

`to_prometheus_rules_yaml` renders the same SLO objects as alerting rules,
generated from the TOML rather than maintained separately:

```yaml
groups:
  - name: vmp-slo
    rules:
      - alert: VmpSloTtfaP95
        expr: "(histogram_quantile(0.95, sum(rate(vmp_ttfa_ms_bucket[5m])) by (le))) > 3000.0"
        for: 10m
        labels:
          severity: page
          slo: ttfa_p95
        annotations:
          summary: "ttfa_p95_ms outside objective (<= 3000.0, window 30d)"
```

They live in `deploy/prometheus/rules/`, with the Grafana board in
`deploy/grafana/dashboards/voice-agent.json`. Every alert should route to a
[runbook](../runbooks.md), and every runbook should end in a rollback step.

## Metrics

Two metric surfaces, both Prometheus text format, neither requiring a client
library:

- **`vmp.observability.metrics`** — `MetricsRegistry` with `Counter`, `Gauge`
  and fixed-bucket `Histogram`. Standard names: `vmp_turn_total`,
  `vmp_stage_ms`, `vmp_ttfa_ms`, `vmp_errors_total`, `vmp_wer`,
  `vmp_llm_ttft_ms`, `vmp_drift_psi`, `vmp_edge_bundle_verify_failed_total`,
  `vmp_edge_bundle_info`. `PrometheusClientAdapter` mirrors writes into
  `prometheus_client` when it is installed.
- **The API's own `/metrics`** — `vmp_http_requests_total`,
  `vmp_http_request_seconds`, `vmp_turns_total`,
  `vmp_turn_stage_seconds{stage}`, `vmp_first_audio_seconds`.

```text
# HELP vmp_first_audio_seconds Time to first audio in seconds
# TYPE vmp_first_audio_seconds histogram
vmp_first_audio_seconds_bucket{le="0.5"} 1
vmp_first_audio_seconds_bucket{le="1"} 1
...
```

`vmp_edge_bundle_info` is the pattern worth copying: a gauge whose *labels* are
the interesting part, so a fleet's bundle-version distribution is one query.

## Drift

`DriftDetector` computes two statistics on any numeric series — feature values,
per-item WER, per-turn latency — in pure Python:

- **PSI** (population stability index) over equal-width bins spanning both
  samples. Conventional reading: below 0.1 stable, 0.1–0.2 moderate, above 0.2
  shifted.
- **Two-sample KS**, the maximum gap between the empirical CDFs. Sensitive to
  shape changes that leave the mean alone.

```json
{
  "name": "feature", "n_reference": 100, "n_current": 100,
  "psi": 15.5749, "ks": 1.0,
  "mean_reference": 2168.3, "mean_current": 2835.62, "mean_shift": 667.32,
  "psi_threshold": 0.2, "ks_threshold": 0.1,
  "drifted": true,
  "reasons": ["feature: psi 15.5749 > 0.2", "feature: ks 1.0000 > 0.1"]
}
```

Both statistics are reported whether or not they fire, so the trend is visible
before the threshold is crossed. `check_many` runs the same check across a dict
of series, which is the shape of a feature-view drift sweep. `vmp obs drift`
exits 1 when drift is detected, and the response is
[drift-detected](../runbooks.md) — not an automatic retrain. Drift is a reason
to look, not a diagnosis.

## OpenTelemetry

`OtlpSink` forwards trace rows as OTel spans and imports `opentelemetry` lazily.
Each `<stage>.end` row becomes a span named for the stage, with
`vmp.session`, `vmp.turn`, `vmp.seq`, `vmp.ms` and `vmp.span` attributes; start
rows are recorded as events on the current span. Collector config is in
`deploy/otel/collector.yaml`.

The JSONL sink is not replaced by OTel, and that is deliberate. The file is the
one artifact that a notebook, a test, a laptop run and a pod all produce
identically and that can be replayed offline months later. OTel is for the live
system; JSONL is for the evidence.

## What the real run needs

```bash
pip install -e ".[obs]"     # opentelemetry-*, prometheus-client
```

Nothing above requires it. The tracer, the summariser, the SLO computation, the
drift detectors, the metrics registry and the Prometheus rendering are all
standard library. The extra buys the OTLP exporter and the `prometheus_client`
mirror — the two places where the platform talks to someone else's collector.

## Pitfalls

- **`JsonlSink` is not `fsync` by default.** A process killed mid-turn can lose
  the tail. Pass `fsync=True` when the trace is the evidence for an incident;
  accept the write cost.
- **Traces contain transcripts.** Payloads can carry user speech. Apply the
  retention policy from
  [data-retention-and-privacy](../runbooks.md), scrub with
  `vmp.data.pii.scrub_records` before anything leaves the boundary, and never
  ship a raw trace into a third-party tool without checking.
- **`ms` is wall clock, not CPU.** A stage that waited on a queue looks slow.
  That is correct for a latency SLO and wrong for capacity planning — for that,
  read `tts.queue_wait_ms` and the stage's own payload.
- **`ttfa` only exists if the first `playback.end` carries `response_ms`.** A
  backend that skips the payload yields a `None` ttfa, which silently shrinks
  the sample the percentile is computed from. Check `n_turns` in the SLI against
  the turns you know happened.
- **PSI needs a stable reference window.** Comparing today with yesterday makes
  a slow drift invisible, because the reference drifts with it. Pin the
  reference to a known-good period and move it deliberately.
- **`error_rate` counts missing `turn.end`, so truncated traces look like
  errors.** A trace copied while the process was still writing will report a
  false breach. Rotate, then read.
