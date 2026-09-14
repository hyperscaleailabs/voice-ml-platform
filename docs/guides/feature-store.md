# Guide: Feature store

## Purpose

A voice session has state that the model, the retriever and the routing logic
all want to read: how many turns have happened, how long this speaker's
utterances usually are, what the last intent was, what their word error rate
looked like over the past week, which bundle version their device is running.
Computing that inline in the turn loop is how a platform ends up with four
slightly different definitions of "average response time".

`vmp.features` defines each feature once, in a `FeatureView`, and serves it two
ways:

- **Online**, keyed by entity, read in single-digit milliseconds during a turn.
- **Offline**, as history, read with a **point-in-time join** so that training
  data contains only what was actually knowable at the moment of the label.

The same view definitions drive both stores, materialisation, and the Feast
export, so a feature's type or TTL changes in one place.

## The views

Three entities, three views, defined in `src/vmp/features/views.py` and mirrored
in `configs/features.toml` for the CLI:

```toml
# Feature views, mirrored from src/vmp/features/views.py for the CLI.
# Sources: traces (stage trace JSONL), eval (evaluation runs), edge (device reports).

[store]
offline = "jsonl"
online = "memory"

[[views]]
name = "session_features"
entity = "session"
ttl_s = 3600
source = "traces"
description = "Rolling per-session features updated from stage traces."
features = [
  { name = "turn_count", dtype = "int", description = "completed turns in the session" },
  { name = "avg_user_utterance_s", dtype = "float", description = "mean listen-stage duration in seconds" },
  { name = "avg_response_ms", dtype = "float", description = "mean time to first audio (playback response_ms)" },
  { name = "last_intent", dtype = "str", description = "intent of the most recent turn" },
  { name = "accent_profile", dtype = "str", description = "accent tag reported by STT" },
]

[[views]]
name = "speaker_features"
entity = "speaker"
ttl_s = 604800
source = "eval"
description = "Seven-day quality aggregates per speaker, from evaluation runs."
features = [
  { name = "wer_7d", dtype = "float", description = "word error rate over the trailing seven days" },
  { name = "exact_match_rate_7d", dtype = "float", description = "exact transcript match rate, seven days" },
  { name = "wake_word_corrections_7d", dtype = "int", description = "wake-word corrections applied, seven days" },
]

[[views]]
name = "device_features"
entity = "device"
ttl_s = 86400
source = "edge"
description = "Edge device state and one-day latency percentile."
features = [
  { name = "edge_bundle_version", dtype = "str", description = "installed EdgeBundle version" },
  { name = "p95_ttfa_ms_24h", dtype = "float", description = "p95 time to first audio over 24 hours" },
]
```

`ttl_s` is not a cache eviction policy. It is the statement "a value older than
this is not a usable answer", and it is enforced identically online (a read past
the TTL returns nothing) and offline (a point-in-time join past the TTL returns
nulls). One number, one meaning, both paths.

## Point-in-time correctness

The reason a feature store exists at all. Training rows are `(entity_id,
label_ts)` pairs; the join must attach the freshest feature row whose
`event_ts <= label_ts`, and only if it is within the view's TTL:

```python
from vmp.features.offline import JsonlOfflineStore
from vmp.features.views import get_view

store = JsonlOfflineStore(".vmp/features")
view = get_view("session_features")
rows = store.point_in_time(view, [("sess-1", 1_789_000_000.0), ("sess-2", 1_789_000_500.0)])
```

Three properties matter and each one is a class of bug it prevents:

- **`event_ts <= label_ts`** — no feature computed after the label leaks into
  the row. This is the leak that makes an offline model look excellent and a
  production model look broken.
- **TTL enforcement** — a stale row yields all-null values rather than a
  confidently wrong one, so the model learns that the feature can be missing.
- **The output `event_ts` is the request time**, not the feature's time, so the
  result joins straight back to the label table without a second alignment step.

A missing entity, an entity whose first row is later than the request, and an
entity whose freshest row is past the TTL all produce the same thing: a row of
nulls with the requested timestamp. There is no silent drop.

## Streaming updates from trace events

The online store is fed from the same trace rows the rest of the platform
writes. `SessionFeatureUpdater` folds events into per-session running sums and
pushes a snapshot online when a turn completes:

| Trace event | Field read | Feature |
|---|---|---|
| `listen.end` | `ms` | `avg_user_utterance_s` |
| `playback.end` | `payload.response_ms` | `avg_response_ms` |
| `llm.end` | `payload.intent` | `last_intent` |
| `stt.end` | `payload.accent` | `accent_profile` |
| `turn.end` | — | `turn_count`, and the write online |

```python
from vmp.features.online import InMemoryOnlineStore
from vmp.features.stream import JsonlSource, SessionFeatureUpdater

updater = SessionFeatureUpdater(online=InMemoryOnlineStore())
report = updater.run(JsonlSource(".vmp/trace.jsonl"))
# {'view': 'session_features', 'events_seen': 96, 'events_used': 21,
#  'sessions': 3, 'turns': 9, 'rows': 9}
```

Means are computed on read from running sums, so each event costs O(1) and the
updater holds one small dataclass per live session. `KafkaSource` is the same
loop fed from a topic and imports `confluent_kafka` in its constructor, never at
module import. The reported `events_seen` against `events_used` is the useful
diagnostic: a high ratio of seen-to-used means the trace is not carrying the
payload fields the view expects.

## CLI

```bash
# Copy the freshest offline row per entity into the online store
vmp features materialize --view session_features --config configs/features.toml --dry-run
vmp features materialize --view session_features --offline-root .vmp/features
vmp features materialize --view session_features --redis-url redis://localhost:6379/0 \
    --start 1789000000 --end 1789003600

# Generate a Feast feature repository from the same view definitions
vmp features feast-export --out . --config configs/features.toml --dry-run
vmp features feast-export --out . --project vmp --data-root data
```

## What a dry run returns

`materialize --dry-run` reads and plans but writes nothing. Real output:

```json
{"view": "session_features", "entity": "session",
 "features": ["turn_count", "avg_user_utterance_s", "avg_response_ms", "last_intent", "accent_profile"],
 "ttl_s": 3600.0, "window": {"start_ts": null, "end_ts": null},
 "rows_read": 0, "entities": 0, "entities_written": 0,
 "dry_run": true, "elapsed_ms": 0.02, "online_backend": "none"}
```

`rows_read` against `entities` is the compression the materialisation achieves;
`entities_written` is zero in a dry run and equals `entities` in a real one.
`online_backend` is `none`, `memory` or `redis`, which is how a scheduled job
logs where the write actually went.

`feast-export --dry-run` renders the repository file in memory and reports what
it would write:

```json
{"path": "feature_repo/features.py", "project": "vmp",
 "views": ["device_features", "session_features", "speaker_features"],
 "entities": ["device", "session", "speaker"], "bytes": 2162, "dry_run": true}
```

## What the real run needs

```bash
pip install -e ".[features]"     # feast, redis, pyarrow
```

Each dependency serves exactly one adapter and none is needed for the others:

- **`redis`** — `RedisOnlineStore`, for an online store shared across replicas.
  The in-memory store is per-process, which is correct for a demo and wrong for
  anything with more than one pod.
- **`pyarrow`** — `ParquetOfflineStore`, for offline history large enough that
  scanning JSONL stops being reasonable.
- **`feast`** — only to *run* the exported repository. The export itself is
  stdlib string generation, which is why `--dry-run` works with nothing
  installed.

The generated `feature_repo/features.py` is real Feast code, and it is
generated, not maintained:

```python
# Generated by vmp.features.feast_export for project 'vmp'. Do not edit.
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

session = Entity(name="session", join_keys=["session_id"])

session_features_source = FileSource(
    path="data/session_features.parquet",
    timestamp_field="event_ts",
)

session_features = FeatureView(
    name="session_features",
    entities=[session],
    ttl=timedelta(seconds=3600),
    schema=[
        Field(name="turn_count", dtype=Int64),
        Field(name="avg_user_utterance_s", dtype=Float32),
        ...
    ],
    online=True,
    source=session_features_source,
    tags={"source": "traces"},
)
```

Then `feast apply` and `feast materialize-incremental` from that directory.
`deploy/k8s/base/feast-materialize-cronjob.yaml` runs the same thing on a
schedule.

## Pitfalls

- **`dtype` is a contract with the serving path, not documentation.** A feature
  declared `float` and written as a string passes the JSONL store and fails at
  the Feast schema. Write the value the type says.
- **The in-memory online store is per-process.** Two API replicas have two
  different views of the same session. Use Redis for anything with a replica
  count above one.
- **JSONL offline stores are append-only.** `materialize` reads the whole file
  for the window; at some volume that stops being free, which is when the
  Parquet store earns its dependency.
- **Materialisation lag is a real freshness bound.** A feature materialised
  hourly and given a one-hour TTL is null for part of every hour. Either
  shorten the schedule or lengthen the TTL deliberately; do not discover it
  from nulls in production.
- **Streaming and batch must compute the same number.** `avg_response_ms` from
  the stream updater and `avg_response_ms` recomputed from the trace file must
  agree. When they diverge, the training data and the serving data have already
  diverged. Recompute both paths over the same trace in a test.
- **The trace is the source, so the trace must carry the payload.** `intent`
  and `accent` are read from `llm.end` and `stt.end` payloads. A backend that
  does not stamp them leaves those features permanently null, and nothing
  errors.
