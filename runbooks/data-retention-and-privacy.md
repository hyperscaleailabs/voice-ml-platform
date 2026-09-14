# Data retention and privacy

## Scope

What this platform stores, for how long, and what to do when someone asks for
their data to be removed or when a privacy incident is suspected. A voice agent
handles two especially sensitive classes: **raw audio**, which is biometric and
identifies a speaker even with the words removed, and **transcripts**, which
contain whatever the user happened to say.

This is an operational runbook, not legal advice. Jurisdictional obligations
(lawful basis, retention maxima, breach notification deadlines) come from
counsel; what follows is how the platform is wired and which levers exist.

## What is stored, where

| Data | Where | Default | Notes |
|---|---|---|---|
| Raw audio, cloud | not persisted by the turn loop | — | Persist it only behind an explicit config change and a consent record |
| Raw audio, edge | not persisted | `retention.audio = "none"` in `configs/edge.toml` | Enforced by the policy check before any audio write |
| Transcripts | session state, and trace payloads | in-memory for the session TTL | `session_ttl_s` in `configs/serving.toml` |
| Stage traces | JSONL at `VMP_TRACE_PATH` | grows without bound | The main retention risk; set a window |
| Session features | Redis db 0, offline Parquet | 1 h TTL online | `session_features` in `views.py` |
| Speaker features | Redis db 1, offline Parquet | 7 d TTL | Aggregates, but keyed by speaker |
| Device features | Redis db 1, offline Parquet | 1 d TTL | `device_id`, bundle version, latency |
| Training corpora | MinIO / object storage | until deleted | Scrub before they land |
| Golden set | repository | permanent | Must be synthetic or consented |
| Model artefacts | registry | permanent | May memorise training data |

Two facts follow from this table and are worth stating plainly. Traces are the
largest privacy surface, because they are the one store with no TTL. And a
feature TTL is not a deletion guarantee: the online copy expires, the offline
Parquet that fed it does not.

## Audio retention

The default is not to keep it. Keeping raw audio requires a decision, a consent
record, and a retention window with something that actually enforces it — a TTL
nobody deletes against is a statement of intent.

If audio must be retained (for an accent LoRA, say, or a WER investigation):

- Store it separately from transcripts and traces, so it can be deleted
  independently.
- Keep the shortest window that serves the purpose, and write the purpose down
  next to the window.
- Never place audio, transcripts or identifiers in log lines or metric labels —
  those propagate into systems with their own, longer retention.

On the edge, `retention.audio = "none"` is checked by `enforce(policy, action)`
in `vmp/edge/policy.py` before any audio write, so the runtime cannot write audio
even if a code path tries. Changing that value is a privacy change and needs the
review below, not a routine bundle ship.

## PII scrub

`vmp.data.pii.scrub` redacts emails, card-like digit runs and phone numbers,
replacing them with `[EMAIL]`, `[CARD]` and `[PHONE]`. It runs before text is
written to a corpus.

```python
from vmp.data.pii import scrub, scrub_records
```

What it does not catch: names, addresses, dates of birth, account numbers in
unusual formats, and anything identifying by context ("my wife's operation last
Tuesday"). Pattern scrubbing is a floor, not a guarantee. Treat a scrubbed
corpus as pseudonymised, not anonymised, and keep the access controls on it.

Scrub at the point data enters storage, not at the point it leaves. A corpus
that is scrubbed on export has already been written unscrubbed.

Traces need the same discipline: `payload` is a free-form dict, and it is easy to
put a transcript in one. Keep payloads to measurements — `ttft_ms`,
`response_ms`, counts — and keep content out.

## Consent

- Record consent per speaker, with what was consented to (use of audio, use of
  transcripts, retention period) and when. Consent for a live turn is not consent
  to retain it for training.
- Withdrawal must be actionable: a `speaker_id` that can be found in every store
  that holds their data. A withdrawal you cannot execute is not a withdrawal.
- The golden set is synthetic by construction. Keep it that way — a real
  utterance in a permanent, repository-checked-in evaluation set is a retention
  decision nobody made.

## Deletion request

```bash
# 1. establish scope: which speaker, which sessions, which devices
vmp registry list --name voice-agent      # was their data in a training run?

# 2. online stores — expire now rather than waiting for the TTL
redis-cli -u redis://redis:6379/0 --scan --pattern '*<session_id>*' \
  | xargs -r redis-cli -u redis://redis:6379/0 del
redis-cli -u redis://redis:6379/1 --scan --pattern '*<speaker_id>*' \
  | xargs -r redis-cli -u redis://redis:6379/1 del

# 3. offline Parquet under VMP_FEATURE_STORE_ROOT: rewrite without those rows

# 4. traces: filter the JSONL by session and speaker

# 5. RAG stores, if any utterance was indexed
```

Then answer the two questions that are easy to skip:

- **Was their data in a training corpus?** If so, it may be in a model's weights.
  Record which artefact versions, and say so honestly — deleting the corpus does
  not delete the model. Whether retraining is required is a decision for counsel,
  not for on-call.
- **Are there backups?** A deletion that leaves the rows in a snapshot is not
  complete. Record the backup retention window and when the rows age out of it.

Record what was deleted, from where, and when.

## Edge offline-only policy

The edge default is `offline_only = true`: audio never leaves the device, because
the device never egresses.

```toml
[policy]
offline_only = true
allowed_egress = []
fallback = "degrade"

[policy.retention]
audio = "none"
```

Three mechanisms enforce the same statement, deliberately:

1. **The policy check** in the runtime, before any network call, cloud fallback
   or audio write. `EdgePolicy.validate()` rejects contradictions at build time —
   `offline_only` with a non-empty `allowed_egress`, or with
   `fallback = "cloud"`.
2. **The systemd sandbox**: `IPAddressDeny=any` in
   `deploy/edge/systemd/vmp-edge.service`.
3. **The NetworkPolicy** in `deploy/edge/k3s/daemonset.yaml`, which permits DNS
   and nothing else.

`fallback = "degrade"` is the privacy-preserving choice: when a local backend is
unavailable the device gives a worse answer rather than sending audio to a cloud
the user did not agree to.

Changing any of these is a privacy change, not a model change. `vmp edge bundle
diff` surfaces policy differences between bundles precisely so this is visible in
review. A bundle that flips `offline_only` or widens `allowed_egress` needs a
consent review before it ships, at any rollout percentage.

## Suspected privacy incident

SEV-2 at minimum, SEV-1 if data left the system. Then:

1. **Stop the bleeding.** Roll back the change that caused it
   (`rollback-model.md`); halt any rollout in progress.
2. **Do not delete evidence.** Preserve the traces and logs that show the scope.
   Deleting them makes the scope unknowable, which is worse.
3. **Scope it**: what data, whose, how much, how long, who could have seen it.
4. **Escalate immediately.** Notification obligations are time-bounded and start
   at discovery, not at resolution.
5. **Postmortem** as in `incident-response.md`, with the scope assessment and its
   evidence attached.

## Review checklist

Quarterly, and before any change that touches storage:

- [ ] Trace retention has a window, and something enforces it.
- [ ] No transcripts, audio paths or identifiers in trace payloads, log lines or
      metric labels.
- [ ] PII scrub runs on every corpus write path, not only on export.
- [ ] Feature TTLs match the documented retention, online **and** offline.
- [ ] Consent records exist for every retained audio set, with a purpose.
- [ ] The edge policy still reads `offline_only = true`, `audio = "none"`, and the
      sandbox and NetworkPolicy still agree with it.
- [ ] A deletion request can be executed end to end — try it on a test speaker.
- [ ] Backups have a stated retention window.
