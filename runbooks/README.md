# Runbooks

Operational procedures for the voice agent platform. Each runbook is written for
someone who is paged at 03:00 and has not read the codebase today: symptoms
first, then diagnosis in a fixed order, then the exact commands.

Every runbook has the same shape:

`## Symptom` — `## Severity` — `## Diagnose` — `## Mitigate` — `## Verify` —
`## Follow-up`

## Index

| Runbook | Use it when |
|---|---|
| [incident-response.md](incident-response.md) | Anything is on fire. Severity matrix, the first fifteen minutes, comms, postmortem. |
| [on-call.md](on-call.md) | You are starting a shift, or handing one over. |
| [ttfa-slo-breach.md](ttfa-slo-breach.md) | Time to first audio is over budget. The headline latency metric. |
| [rollback-model.md](rollback-model.md) | A model or bundle needs to go back: registry, Ray Serve, edge. |
| [drift-detected.md](drift-detected.md) | A PSI or KS drift alert fired on features or on WER. |
| [rag-relevance-degraded.md](rag-relevance-degraded.md) | Answers are wrong or vague; retrieval is suspect. |
| [edge-bundle-verify-failed.md](edge-bundle-verify-failed.md) | A device refuses to start a bundle. |
| [capacity-and-cost.md](capacity-and-cost.md) | Saturation, GPU utilisation, batching and autoscaling levers. |
| [data-retention-and-privacy.md](data-retention-and-privacy.md) | Audio retention, PII, consent, a deletion request, the edge policy. |

## What the platform gives you to diagnose with

Three sources, in the order you usually reach for them.

**Traces** — one JSON object per line: `ts, session, turn, event, span, seq, ms,
payload`. `event` is `<stage>.start` / `<stage>.end`. The stages of a voice turn
are `listen, stt, retrieve, llm, segment.emit, tts, playback, turn`. Two payload
fields carry the latency story: `llm` has `ttft_ms` (time to first token) and
`playback` has `response_ms` (time to first audio — the headline metric).

```bash
vmp obs summarise --trace /var/lib/vmp/trace.jsonl
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml
```

**Metrics** — Prometheus scrapes `/metrics` on the API pods; SLO recording and
alerting rules are in `deploy/prometheus/rules/voice-slo.yml`, and the dashboard
is `deploy/grafana/dashboards/voice-agent.json`.

**Evaluation** — the golden set and the release gates.

```bash
vmp eval golden --set data/golden.jsonl
vmp eval gate --rules configs/gates.toml --metrics /tmp/metrics.json
```

## Rules that apply to every runbook here

1. **Mitigate before you diagnose.** Restoring service and finding the cause are
   different activities. Roll back first; the trace is still there afterwards.
2. **One person changes things.** The incident commander says who. Two people
   fixing the same thing is how a small incident becomes a long one.
3. **Write it down as you go.** The timeline in the incident channel is the
   postmortem's first draft. Reconstructing it later loses the detail that
   mattered.
4. **No number without a source.** Quote the command whose output you read.
   Targets in `configs/slo.toml` and `configs/gates.toml` are targets, not
   measurements.
