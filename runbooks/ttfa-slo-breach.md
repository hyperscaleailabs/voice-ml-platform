# Time to first audio over budget

## Symptom

`vmp obs slo` reports `ttfa_p95_ms` above the objective in `configs/slo.toml`
(target: 3000 ms p95 over 30 days), or the Prometheus alert on the p95 recording
rule is firing. Users describe it as "it takes ages before it starts talking".

## Severity

SEV-2 while p95 is over budget for more than 30 minutes. SEV-1 if turns are
timing out rather than merely being slow. SEV-3 if only one cohort (one edge
bundle, one region) is affected and the fleet p95 is fine.

## What the metric is

`playback` payload `response_ms` — the wall time from the end of the user's
utterance to the first audio sample leaving the speaker. It is the sum of every
stage in between, so a breach is always "which stage grew", never "the system is
slow". Confirm the breach before diagnosing it:

```bash
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml
vmp obs summarise --trace /var/lib/vmp/trace.jsonl
```

`summarise` gives per-stage totals for the turns in the trace. The stage that
grew is nearly always visible there in one line.

## Diagnose — in this order

Work down the pipeline. Each step names the trace field that shows it, so you
can confirm or eliminate the step from the trace alone. Do not skip ahead: an
endpointing problem looks exactly like a slow model if you start at the model.

### 1. Endpoint silence — is the agent waiting too long to decide the user stopped?

**Trace field:** the `listen` stage — `listen.start` to `listen.end`, `ms` on the
`listen.end` row.

A voice-activity detector that waits too long for silence adds its whole timeout
to every turn before any work begins. This is the most common cause of a
platform-wide TTFA regression and the easiest to misattribute, because every
downstream stage still looks normal.

Look for `listen` durations clustered at a constant value — that constant is the
silence timeout, and it means the detector is timing out rather than detecting.
Compare against the same percentile a day earlier. A change to the VAD threshold
or a noisier room both show up here.

Eliminate it before going further: if `listen` is unchanged, the problem is
downstream.

### 2. STT — is transcription slow?

**Trace field:** the `stt` stage — `ms` on `stt.end`.

Check the backend and the device it is on. A model that fell back from GPU to CPU
is the usual cause, and the ratio is large: on one laptop, mlx-whisper on the M4
GPU against faster-whisper on CPU over the same audio measured 233 ms versus
1,761 ms, a 7.55x difference (alpha-core, cycle 5, 2026-09-12,
`spikes/streaming-whisper-m4/compare_stt.py`) — that is a private-project
measurement, not a benchmark of this repo, but it is the right order of magnitude
to expect when an accelerator disappears.

```bash
kubectl describe pod -n platform -l app.kubernetes.io/component=serve-worker-gpu \
  | grep -A3 "nvidia.com/gpu"
kubectl logs -n platform -l app.kubernetes.io/name=vmp-voice --tail=200 \
  | grep -i "cuda\|device\|fallback"
```

Also check `stt` duration against the audio duration. If STT time is growing in
proportion to utterance length, it is compute-bound; if it is constant and large,
it is queueing or loading.

### 3. LLM time-to-first-token — is the model slow to start speaking?

**Trace field:** `llm` payload `ttft_ms` (and the `llm` stage `ms` for the full
generation).

`ttft_ms` is what TTFA cares about; total generation time mostly does not, because
the response is streamed sentence by sentence. A rising `ttft_ms` with a flat
total means queueing — requests are waiting for a replica. A rising total with a
flat `ttft_ms` means slower decoding, which affects later sentences, not the
first audio.

```bash
kubectl get rayservice -n platform vmp-voice \
  -o jsonpath='{.status.activeServiceStatus.applicationStatuses}{"\n"}'
# replica counts and queue depth per deployment, from the Serve dashboard metrics
kubectl port-forward -n platform svc/vmp-voice-head-svc 8265:8265
```

Model size dominates here: on six factual questions, a 270M model and a 4B model
measured 192 ms and 4,427 ms median `llm` time (alpha-core, cycle 5, 2026-09-12,
`notebook_optimized.ipynb`) — again a private measurement, cited for the shape of
the trade-off, not as this platform's number. If a larger model was promoted
recently, that is the change to reverse.

### 4. TTS queue wait — is audio synthesis waiting to start?

**Trace fields:** the gap between `segment.emit.end` (a sentence is ready) and
`tts.start` (synthesis begins), plus the `tts` stage `ms`.

That gap is queue wait: the sentence existed and nothing was synthesising it.
A non-zero gap on the *first* sentence goes straight into TTFA. Distinguish the
two failure modes:

- **Gap large, `tts` duration normal** — not enough TTS replicas. The deployment
  is at `max_replicas`, or autoscaling has not caught up
  (`upscale_delay_s` in `deploy/ray/rayservice-voice.yaml`).
- **Gap small, `tts` duration large** — synthesis itself is slow. Check the real
  time factor: `tts` ms against the duration of the audio produced. Kokoro
  measured 0.105–0.157 RTF on one laptop (alpha-core, cycle 5, 2026-09-12,
  `README.md` "Streaming").

Also confirm the segmenter is emitting early. If `segment.emit` only fires once,
at the end, the response is not being streamed and TTFA collapses into total
generation time — serial versus streaming measured 11,049 ms against 2,688 ms for
an eight-sentence answer (alpha-core, cycle 5, 2026-09-12, `README.md`
"Streaming").

### 5. Playback underruns — is audio produced but not played?

**Trace fields:** the `playback` stage `ms`, and `response_ms` in its payload
compared against the sum of the stages above it.

If `response_ms` is much larger than the sum of `listen + stt + retrieve + llm +
segment.emit + tts`, the time went into playback: the audio device was not ready,
the buffer underran and restarted, or the client was not draining the stream.
Look for repeated `playback` spans within a single turn — a healthy turn has one.

On the device side this is a hardware or driver problem, not a model problem; on
the cloud side it is the client or the network between the ingress and the
client, and the WebSocket timeouts in `deploy/k8s/overlays/cloud/api-ingress.yaml`
are worth checking.

## Mitigate

Pick by the stage you identified.

```bash
# stage 1: revert a VAD/endpointing config change
kubectl rollout undo deployment/api -n platform

# stage 2 or 3: a model was promoted — roll it back
#   see runbooks/rollback-model.md

# stage 3 or 4: capacity — raise the floor so autoscaling stops being the
# critical path. Edit min_replicas in deploy/ray/rayservice-voice.yaml and:
kubectl apply -f deploy/ray/rayservice-voice.yaml

# API-side saturation
kubectl scale deployment/api -n platform --replicas=6
```

If the stage is not yet identified after fifteen minutes and the breach is
SEV-2, roll back the most recent change anyway and diagnose from the trace
afterwards.

## Verify

```bash
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml
vmp obs summarise --trace /var/lib/vmp/trace.jsonl
```

p95 back within the objective, and the stage that grew back to its previous
share of the turn. Watch for 30 minutes before closing: autoscaling can make a
breach look resolved for one scrape interval.

## Follow-up

- If the cause was a promotion, the gate that let it through is the action item.
  `ttfa_p95_ms` is `required = false` in `configs/gates.toml`; consider whether
  it should block.
- If the cause was autoscaling latency, raise `min_replicas` rather than tuning
  `upscale_delay_s` down — a cold replica cannot be made warm by asking sooner.
- If diagnosis required a stage the trace does not record, add the span. The
  trace schema is in `vmp/observability/trace.py`.
