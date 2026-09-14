# Incident response

How to run an incident on this platform: classify it, stabilise it, tell people,
then learn from it.

## Severity matrix

Severity is set by user impact, not by how alarming the graph looks. When two
rows apply, take the higher one. Anything involving personal data is at least
SEV-2 regardless of how few users are affected.

| Severity | Impact | Examples | Response | Comms |
|---|---|---|---|---|
| **SEV-1** | The agent does not answer, for everyone | API returning 5xx fleet-wide; Ray Serve ingress down; every turn times out; audio or transcript data exposed | Page immediately, incident commander named within 5 min, all hands | Status update every 30 min until mitigated |
| **SEV-2** | Badly degraded, or a subset broken | p95 time to first audio above budget for over 30 min; one region or one device cohort failing; RAG returning nothing; a model shipped that fails a release gate | Page, respond within 15 min | Update at start, at mitigation, at resolution |
| **SEV-3** | Noticeable but working | Elevated error rate under 1%; drift alert; one edge cohort on a stale bundle; a dashboard or exporter down | Next business day | A ticket and a note in the channel |
| **SEV-4** | Cosmetic or internal | Flaky test; noisy alert; docs wrong | Backlog | None |

Declaring too high is cheap and gets downgraded in ten minutes. Declaring too
low costs the hour it takes someone to notice it was worse than it looked.

## The first fifteen minutes

Do these in order. Do not skip step 2 because the cause seems obvious.

**0–2 min — declare and name a commander.**
Open the incident channel, state the severity and the symptom in one sentence.
The commander coordinates and communicates; they do not debug. Name them
explicitly ("I am IC") so the role is never ambiguous.

**2–5 min — establish blast radius.**
Who is affected, since when, where.

```bash
kubectl get pods -n platform -l app.kubernetes.io/name=vmp-api
kubectl get rayservice -n platform vmp-voice -o wide
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml
```

Record the start time. Almost every incident answers to "what changed just
before that time".

**5–10 min — what changed.**
A deploy, a model promotion, a config change, an edge rollout stage, or a
dependency. In that order, because that is the order of likelihood.

```bash
kubectl rollout history deployment/api -n platform
vmp registry list --name voice-agent
kubectl get rayservice -n platform vmp-voice \
  -o jsonpath='{.status.activeServiceStatus.applicationStatuses}{"\n"}'
```

**10–15 min — mitigate.**
Reverse the change. Do not fix forward during a SEV-1 or SEV-2.

```bash
kubectl rollout undo deployment/api -n platform          # bad deploy
# model or bundle: follow runbooks/rollback-model.md
kubectl scale deployment/api -n platform --replicas=6    # saturation, buys time
```

If fifteen minutes pass with no mitigation, escalate — add a second responder
and raise the severity by one. An incident that is not shrinking is growing.

## Comms template

Post to the incident channel and to the status page. Keep it factual, keep the
next-update time, and never speculate about cause in a customer-facing update.

```
[SEV-<n>] <one-line symptom> — <INVESTIGATING | IDENTIFIED | MITIGATED | RESOLVED>

Started:    <UTC timestamp>
Impact:     <who is affected and how; "voice answers are delayed by ~8 s for
             all users" beats "elevated latency">
Status:     <what is known, in one or two sentences. No speculation.>
Actions:    <what is being done right now, and by whom>
Workaround: <what a user can do, or "none">
Next update: <UTC timestamp — always give one>
IC: <name>
```

Resolution message adds one line: what will stop it recurring, and the ticket
number tracking that work.

## Blameless postmortem template

Within five business days for SEV-1 and SEV-2. The subject is the system, never
a person: write "the deploy was not gated on the golden set", not "X deployed
without running the golden set".

```markdown
# Postmortem: <short title>

- **Date:** <date>   **Severity:** SEV-<n>   **Duration:** <detection to resolution>
- **Authors:** <names>   **Status:** draft | reviewed

## Summary
Two or three sentences: what broke, who it affected, how it was resolved.

## Impact
Users affected, turns affected, duration, SLO budget consumed. Numbers with the
query or command that produced them.

## Timeline (UTC)
| Time | Event |
|---|---|
| 02:14 | Change X deployed |
| 03:02 | Alert `VoiceTTFAP95High` fired |
| 03:05 | Paged, IC named |
| 03:19 | Rolled back to <version> |
| 03:26 | p95 back within budget |

## Root cause
The chain of conditions, not the last one. "The adapter was promoted without a
golden-set run **because** the gate was advisory **because** the eval job was
flaky **because** ...". Stop when you reach something the team controls.

## Detection
How it was found, and how long that took. Did an alert fire, or did a user tell
us? If a user told us, that is a finding in its own right.

## What went well
Genuinely — the parts of the response worth keeping.

## What went badly
Confusion, missing tooling, misleading dashboards, a runbook that was wrong.
A runbook that was wrong is an action item, not an aside.

## Action items
| # | Action | Type | Owner | Due | Ticket |
|---|---|---|---|---|---|
| 1 | Make the WER gate blocking in CI | prevent | | | |
| 2 | Alert on p95 TTFA burn rate, not the raw value | detect | | | |
| 3 | Add the Ray Serve rollback to the on-call drill | respond | | | |

Each action has an owner and a due date, or it is not an action item. Type is
one of prevent / detect / mitigate / respond.

## Lessons
What someone who did not attend the incident should learn from it.
```

## After the incident

- Close the status page entry and the channel with a resolution message.
- File the action items as issues before the channel is archived.
- If a runbook was used, fix what was wrong with it while it is fresh. If no
  runbook existed, write one — that is the cheapest action item on the list.
