---
date: 2026-09-08
authors: [cg]
categories:
  - Features
  - Data
slug: feature-store-for-real-time-sessions
---

# A feature store for real-time sessions

There is a moment in every ML system where somebody computes an average inline.
The turn loop needs to know how long this speaker's utterances usually run, so a
helper is written where it is needed. Three months later the same quantity is
computed in the training pipeline, in a dashboard query, and in a routing rule —
four implementations, four subtly different definitions of "usually", and no way
to tell which one the model was trained against.

A feature store is the discipline that prevents that. For a voice agent it has to
do it while a turn is in flight.

<!-- more -->

## What a voice session wants to know

Three entities, three kinds of state.

**The session.** Turn count, mean user utterance length, mean time to first
audio, the last intent, the accent tag STT reported. Read during the turn, so
read latency is in the critical path of a number the user hears. TTL one hour — a
session is over long before then.

**The speaker.** Word error rate over the trailing seven days, exact-match rate,
how many wake-word corrections were applied. These come from evaluation runs, not
from the live path, and they answer "is this person being understood well?" — the
question behind any per-speaker adaptation decision. TTL seven days.

**The device.** Installed edge bundle version, p95 time-to-first-audio over 24
hours. This is the fleet view, and it is what makes an OTA canary decidable:
which cohort is on which bundle, and did their latency move. TTL one day.

Each is defined once, in a `FeatureView`, with a name, a dtype and a description
per feature. The same definitions drive the offline store, the online store,
materialisation and the Feast export, so a feature's type or TTL changes in
exactly one place.

## TTL is not a cache policy

The most useful thing about that definition is the least exciting field.

`ttl_s` is not "how long to keep this in memory". It is the statement *a value
older than this is not a usable answer*, and it is enforced identically in both
directions. An online read past the TTL returns nothing. An offline
point-in-time join past the TTL returns nulls.

One number, one meaning, both paths. Which matters because the two paths are
where training/serving skew lives, and skew usually enters through a definition
that was only ever enforced on one side.

## Point-in-time correctness, concretely

This is the reason feature stores exist, and it is worth stating without
abstraction.

Training rows are pairs of `(entity_id, label_ts)`. The join must attach the
freshest feature row whose `event_ts <= label_ts`, and only if it falls within
the view's TTL.

```python
store.point_in_time(view, [("sess-1", 1_789_000_000.0), ("sess-2", 1_789_000_500.0)])
```

Three properties, each preventing a distinct class of bug:

**`event_ts <= label_ts`.** No feature computed after the label leaks into the
training row. This is *the* leak — the one that produces an offline model with
excellent metrics and a production model that does not work, because at inference
time the future is not available. It is easy to introduce with a naive join on
entity id and almost impossible to detect from metrics alone.

**TTL enforcement.** A stale row yields all-null values rather than a confidently
wrong one. The model learns that the feature can be missing, which is true, and
which it will need to handle in production.

**The output row's `event_ts` is the request time.** Not the feature's time. So
the result joins straight back to the label table with no second alignment step —
one fewer place for an off-by-one to live.

A missing entity, an entity whose first row postdates the request, and an entity
whose freshest row is past the TTL all produce the same thing: a row of nulls at
the requested timestamp. There is no silent drop, which means the training set
row count matches the label count and a discrepancy is visible immediately.

## Fed from the trace, because the trace already exists

The online store is updated from the same trace rows the rest of the platform
writes. No separate instrumentation, no second schema.

`SessionFeatureUpdater` folds events into per-session running sums:

| Trace event | Field read | Feature |
|---|---|---|
| `listen.end` | `ms` | `avg_user_utterance_s` |
| `playback.end` | `payload.response_ms` | `avg_response_ms` |
| `llm.end` | `payload.intent` | `last_intent` |
| `stt.end` | `payload.accent` | `accent_profile` |
| `turn.end` | — | `turn_count`, and the write online |

Means are computed on read from the sums, so each event costs O(1) and the
updater holds one small dataclass per live session. `KafkaSource` is the same
loop fed from a topic, importing `confluent_kafka` in its constructor rather than
at module import — the same lazy-adapter rule as everywhere else in the platform.

The report it returns is more useful than it looks:

```json
{"view": "session_features", "events_seen": 96, "events_used": 21,
 "sessions": 3, "turns": 9, "rows": 9}
```

`events_seen` against `events_used` is the diagnostic. A high ratio means the
trace is not carrying the payload fields the view expects — a backend that never
stamps `intent` on `llm.end` leaves that feature permanently null, and nothing
errors, anywhere. The feature is simply always missing, the model learns to
ignore it, and six months later somebody asks why intent routing does nothing.

## Two stores, two jobs

The online store answers "what is true about this entity right now" in
single-digit milliseconds. In-memory is the reference implementation — correct
for a demo, and **per process**, which means two API replicas have two different
views of the same session. Redis is the adapter for anything with a replica count
above one, and the moment the answer to "how many replicas" stops being "one" is
the moment that stops being optional.

The offline store answers "what was true about this entity at this past moment".
JSONL, append-only, one file per view, is the reference; Parquet via `pyarrow` is
the adapter for when scanning stops being free.

`materialize` connects them: for a window, take the freshest row per entity and
write it online with the view's TTL. The manifest reports what happened, which is
what a scheduled job needs to log:

```json
{"view": "session_features", "rows_read": 0, "entities": 0, "entities_written": 0,
 "ttl_s": 3600.0, "online_backend": "none", "dry_run": true, "elapsed_ms": 0.02}
```

`rows_read` against `entities` is the compression the materialisation achieves.
`online_backend` records where the write actually went — `none`, `memory` or
`redis` — so a cron job that silently wrote to an in-memory store it then
discarded shows up in its own log rather than as a mystery.

## Generating the Feast repository

Feature views defined in code, Feast definitions generated from them:

```bash
vmp features feast-export --out . --config configs/features.toml
```

The output is real Feast code with a header that says not to edit it — `Entity`,
`FileSource`, `FeatureView` with the TTL as a `timedelta` and the schema as typed
`Field`s. `feast apply` and `feast materialize-incremental` take it from there,
and `deploy/k8s/base/feast-materialize-cronjob.yaml` runs the schedule.

The generation itself is standard-library string building, which is why
`--dry-run` works with nothing installed. `feast` is only needed to *run* the
repository. That is a small thing that compounds: a CI job can lint every feature
definition on every commit without installing a feature-store framework.

## The failure that hides in the gap

The generic pitfalls are real — dtype mismatches that pass JSONL and fail the
Feast schema, in-memory stores behind multiple replicas, JSONL files that
eventually want to be Parquet. The two that specifically bite a real-time system
are subtler.

**Materialisation lag is a freshness bound.** A feature materialised hourly with
a one-hour TTL is null for part of every hour, by construction. Either shorten
the schedule or lengthen the TTL, deliberately, at design time — the alternative
is inferring it from a puzzling null rate in production.

**Streaming and batch must compute the same number.** `avg_response_ms` from the
stream updater and `avg_response_ms` recomputed from the same trace file offline
must agree. When they diverge, training data and serving data have already
diverged, and every downstream metric is measuring a model against inputs it was
not trained on. This is testable: run both paths over one trace and assert
equality. It is worth a test, because the divergence is silent, gradual, and
attributed to everything except its cause.

## Why this belongs in a voice platform at all

It would be reasonable to ask whether a voice agent needs a feature store. The
answer is that the three views above are already the evidence base for three
decisions the platform has to make: whether to degrade to a smaller model for
this session, whether this speaker's transcription quality justifies an adapter,
and whether an edge cohort's latency moved after a bundle rollout.

Those decisions get made regardless. The only question is whether they are made
against a defined, versioned, point-in-time-correct number, or against whatever a
helper function computed the day someone needed it.
