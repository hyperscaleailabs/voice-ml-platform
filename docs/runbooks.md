# Runbooks

Operational procedures live in
[`runbooks/`](https://github.com/hyperscaleailabs/voice-ml-platform/tree/main/runbooks)
in the repository. Each one is written for someone who is paged at 03:00 and has
not read the codebase today: symptoms first, then diagnosis in a fixed order,
then the exact commands.

Every runbook has the same shape:

`## Symptom` — `## Severity` — `## Diagnose` — `## Mitigate` — `## Verify` —
`## Follow-up`

Two rules tie them to the rest of the platform: **every alert routes to a
runbook**, and **every runbook ends in a rollback step**. An alert with no
runbook is an alert nobody knows how to answer.

## Index

| Runbook | Use it when |
|---|---|
| [incident-response](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/incident-response.md) | Anything is on fire. The severity matrix, the first fifteen minutes, comms cadence, and the blameless postmortem. |
| [on-call](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/on-call.md) | Starting a shift or handing one over: what to check in ten minutes, and what the secondary is actually for. |
| [ttfa-slo-breach](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/ttfa-slo-breach.md) | `ttfa_p95_ms` is over its 3,000 ms objective, or the Prometheus p95 alert is firing. The headline latency metric. |
| [rollback-model](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/rollback-model.md) | A promoted model or a shipped edge bundle is worse than what it replaced. Registry, Ray Serve and edge rollback paths. |
| [drift-detected](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/drift-detected.md) | A PSI or KS alert fired on an input feature or on WER. Drift is a reason to look, not a diagnosis. |
| [rag-relevance-degraded](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/rag-relevance-degraded.md) | Answers are vague, off-topic, or confidently wrong about the corpus. Retrieval is suspect. |
| [edge-bundle-verify-failed](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/edge-bundle-verify-failed.md) | `vmp edge bundle verify` fails and a device refuses to start the new bundle. |
| [capacity-and-cost](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/capacity-and-cost.md) | Telling a capacity shortage from waste: saturation, GPU utilisation, batching and the autoscaling levers. |
| [data-retention-and-privacy](https://github.com/hyperscaleailabs/voice-ml-platform/blob/main/runbooks/data-retention-and-privacy.md) | Audio retention, PII, a deletion request, or a suspected privacy incident. Raw audio is biometric. |

## What you diagnose with

Three sources, in the order an on-call engineer usually reaches for them.

1. **Traces.** One JSON object per line: `ts, session, turn, event, span, seq,
   ms, payload`. `vmp obs summarise --trace <file>` gives one line per turn with
   per-stage timings, which answers "which stage got slow" before any dashboard
   does. See [Observability](guides/observability.md).
2. **Metrics and SLOs.** `vmp obs slo --trace <file> --config configs/slo.toml`
   computes the indicators and the error-budget burn. Prometheus rules in
   `deploy/prometheus/rules/` and the Grafana board in
   `deploy/grafana/dashboards/` render the same objects.
3. **Registry and gate decisions.** `vmp registry list` and `vmp registry show`
   say what is deployed and which `GateDecision` let it through. "Why was this
   promoted?" is an object, not a memory. See
   [Evaluation and gates](guides/evaluation-and-gates.md).

## Related

- [Production operations](architecture/operations.md) — the deploy, observe,
  scale, recover, improve cycle these runbooks sit inside.
- [Security and reliability](architecture/security-reliability.md) — identity,
  protection and failure behaviour.
- [Progression](progression.md) — Gate 3 and what a bounded rollout requires.
