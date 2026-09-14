# On-call

## Scope

One rotation covers the voice agent API, the Ray training and serving clusters,
the feature store, the RAG stores and the edge fleet. One primary, one secondary.
The secondary is not a spare pair of hands: they take over when the primary has
been heads-down for an hour, and they run comms during a SEV-1 so the primary
can debug.

## Starting a shift

Ten minutes, before you need them.

```bash
# everything is where it should be
kubectl get pods -n platform
kubectl get rayservice,raycluster,rayjob -n platform
kubectl get pods -n vmp-edge -o wide

# the SLOs as of now
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml

# what shipped recently
kubectl rollout history deployment/api -n platform
vmp registry list --name voice-agent
```

Then read: open incidents, anything the previous shift flagged, any rollout
stage currently in soak (`deploy/edge/ota/rollout.toml`), and any silenced
alert — a silence that outlives the shift that created it is a bug.

Confirm you can actually act: cluster credentials work, you can reach the
incident channel, and you can page the secondary. Finding out during a SEV-1
that your kubeconfig expired is a preventable half-hour.

## What pages, and what does not

| Signal | Pages? | Runbook |
|---|---|---|
| API 5xx rate over threshold | yes | `incident-response.md` |
| p95 time to first audio over budget, sustained | yes | `ttfa-slo-breach.md` |
| Ray Serve application not `RUNNING` | yes | `rollback-model.md` |
| Edge rollout gate failed | yes | `edge-bundle-verify-failed.md` |
| Bundle verification failed on a device | no — ticket | `edge-bundle-verify-failed.md` |
| Drift alert (PSI / KS) | no — ticket | `drift-detected.md` |
| Golden-set WER regression | no — ticket | `drift-detected.md` |
| Materialize CronJob failed once | no — ticket | `drift-detected.md` |
| Materialize CronJob failed three times running | yes | `drift-detected.md` |
| Grafana or an exporter down | no — ticket | — |

The rule behind the table: page on user impact and on things that will become
user impact within the shift. Everything else is a ticket. An alert that pages
and is always acknowledged without action is worse than no alert, because it
trains the responder to ignore the next one.

## Triage

1. Is it real? Check a second source before acting — a trace against a
   dashboard, or a metric against `vmp obs slo`.
2. Is it user-visible? That sets severity (`incident-response.md`).
3. What changed? Deploy, model promotion, config, rollout stage, dependency.
4. Can it be reversed? Reverse it. Diagnose afterwards.
5. Is there a runbook? Follow it, and fix it afterwards if it was wrong.

## Escalate when

- Fifteen minutes into a SEV-1 or SEV-2 with no mitigation.
- The mitigation is not in a runbook and you are not sure of it.
- Personal data may be exposed — escalate immediately and read
  `data-retention-and-privacy.md` before touching the data.
- You need an action you do not have access to.
- You have been on it for two hours. Fatigue produces the second incident.

Escalating is information, not failure. The cost of a needless escalation is one
person's ten minutes.

## Handover

Written, in the channel, not verbal:

```
Shift handover <date> <time> UTC — <from> -> <to>

Open incidents:   <id, severity, state, what the next step is>
Ongoing:          <rollouts in soak, jobs running, anything mid-flight>
Watch:            <things that looked odd but did not page>
Silences:         <what is silenced, why, and when it expires>
Changes:          <deploys, promotions, config changes during the shift>
Nothing else.
```

"Quiet shift" is a valid handover. A handover that omits a silenced alert is not.

## Access you need before the shift starts

Cluster credentials for `platform` and `vmp-edge`; the container registry;
Grafana and Prometheus; the incident channel and the status page; the paging
tool, with your contact details current.

## Practice

Run one drill per rotation, in a non-production cluster, from a cold start with
only the runbook:

- Roll a model back through the registry and Ray Serve (`rollback-model.md`).
- Roll an edge bundle back on one device (`edge-bundle-verify-failed.md`).
- Diagnose a TTFA breach from a trace alone (`ttfa-slo-breach.md`).

The drill's output is corrections to the runbook. A drill that changes nothing
was either perfect or not done seriously.
