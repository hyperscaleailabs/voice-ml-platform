# Capacity and cost

## Scope

How to tell whether this platform is short of capacity or wasting it, and which
levers to pull. No cost figures appear here: prices depend on the accelerator,
the region and the contract, and an invented number would be worse than none.
What is here is the set of things that drive cost, and how to measure each one
in this deployment.

## Where the money goes

In descending order for a voice agent of this shape:

1. **GPU hours for inference.** Replicas of `LLMDeployment` held warm, whether
   or not they are busy. Dominant, because latency requires them warm.
2. **GPU hours for training.** Bursty and schedulable: the `gpu` worker group
   in `deploy/ray/raycluster.yaml` starts at `minReplicas: 0`, so an idle
   cluster holds no accelerators.
3. **CPU for STT and TTS.** Scales with concurrent turns rather than with total
   traffic.
4. **Storage and egress.** Model artefacts, offline feature Parquet, edge
   bundles, traces. Steady, and usually small next to the above — until trace
   retention is left unbounded.

## Measure before tuning

### GPU utilisation

The number that matters is utilisation *while a replica is alive*, not average
utilisation across the day. A replica at 15% for eight hours is the same spend as
one at 100% for eight hours.

```bash
# per-pod GPU allocation
kubectl describe nodes | grep -A5 "Allocated resources" | grep nvidia
kubectl get pods -n platform -o json \
  | jq -r '.items[]
         | select(.spec.containers[].resources.requests["nvidia.com/gpu"])
         | .metadata.name'

# Ray's own view: replica counts, queue depth, per-deployment latency
kubectl port-forward -n platform svc/vmp-voice-head-svc 8265:8265
```

Three readings, three different problems:

| Reading | Meaning | Lever |
|---|---|---|
| Low utilisation, low queue | Over-provisioned | Lower `min_replicas`, shorten `downscale_delay_s` |
| Low utilisation, high queue | Not compute-bound — blocked elsewhere | Find the real stage in the trace; adding GPUs will not help |
| High utilisation, high queue | Genuinely short of capacity | Raise `max_replicas`, or batch |
| High utilisation, low queue | Right-sized | Leave it alone |

The second row is the expensive mistake: buying accelerators to fix a queue that
is actually STT contention or retrieval latency. Confirm against the trace first
(`vmp obs summarise --trace ...`) before scaling anything.

### Saturation, per stage

```bash
vmp obs summarise --trace /var/lib/vmp/trace.jsonl
vmp obs slo --trace /var/lib/vmp/trace.jsonl --config configs/slo.toml
```

Capacity problems show as a growing gap between when a stage's input was ready
and when the stage started — for TTS, the gap between `segment.emit.end` and
`tts.start`. Growing durations at constant load mean contention; constant
durations with a growing gap mean queueing. They have different fixes.

## The levers

### 1. Batching

Batching raises throughput per accelerator and raises latency per request, and
for a voice agent the second half of that sentence has a hard limit: the
`ttfa_p95_ms` objective in `configs/slo.toml`. Batch the LLM only up to the point
where `ttft_ms` still leaves room within the TTFA budget for STT, TTS and
playback.

`max_ongoing_requests` per deployment in `deploy/ray/rayservice-voice.yaml` is
the practical knob — it bounds how much work one replica accepts at once.
Raise it while watching `llm` `ttft_ms` in the trace, not while watching
throughput.

TTS batches poorly here by design: sentences are synthesised as the segmenter
emits them, precisely so audio starts before generation finishes. Batching TTS
gives back the streaming win — serial versus streaming measured 11,049 ms against
2,688 ms for an eight-sentence answer (alpha-core, cycle 5, 2026-09-12,
`README.md` "Streaming"), a private-project measurement cited for the shape of
the trade-off.

### 2. Autoscaling

Per deployment in `serveConfigV2`:

| Field | Effect | For this workload |
|---|---|---|
| `min_replicas` | Warm floor | The real latency lever. A cold LLM replica cannot be warmed by scaling sooner. |
| `max_replicas` | Ceiling | Bounds spend, and bounds the blast radius of a traffic spike. |
| `target_ongoing_requests` | Scale-out trigger | Lower reacts sooner and costs more. |
| `upscale_delay_s` | Reaction lag | Short for STT/TTS, longer for the LLM. |
| `downscale_delay_s` | Idle hold | Long for the LLM (600 s) — releasing a replica means reloading weights. |

For the API Deployment the same shape lives in `deploy/k8s/base/api-hpa.yaml`
(and the cloud overlay's `hpa-large.yaml`): 70% CPU target, scale-up stabilisation
30 s, scale-down 300 s. The asymmetry is deliberate — scale out fast, scale in
slowly.

For training, the Ray cluster's `gpu` group with `minReplicas: 0` and
`idleTimeoutSeconds: 300` means an idle cluster costs nothing in accelerators.
Check that it actually returns to zero:

```bash
kubectl get pods -n platform -l app.kubernetes.io/component=ray-worker-gpu
```

A GPU worker alive with no RayJob running is pure waste, and it is the single
most common source of it.

### 3. Model size

The largest lever, and the one with a quality cost. A 270M model and a 4B model
measured 192 ms and 4,427 ms median `llm` time on six factual questions, with 2/6
and 0/6 answers wrong (alpha-core, cycle 5, 2026-09-12, `notebook_optimized.ipynb`)
— a private measurement, quoted for the shape of the trade-off, not as this
repo's benchmark. The decision belongs to the release gates
(`configs/gates.toml`), not to a capacity review.

Routing is the middle path: a small model for most turns, a larger one for the
turns that need it, gated on retrieval confidence or intent.

### 4. Scheduling training

Training is the only workload here that can wait. Submit RayJobs against an
ephemeral cluster (`rayClusterSpec` plus `shutdownAfterJobFinishes: true`) rather
than the shared one when accelerators are rented by the minute, and let the
autoscaler return the `gpu` group to zero between runs.

### 5. Storage and retention

Traces are the one that grows without anyone deciding to. Set a retention window
and enforce it — see `data-retention-and-privacy.md`, where the same decision is
made for privacy reasons and should not be made twice with two different answers.

## Reviewing capacity

Monthly, in this order:

1. GPU utilisation while alive, per deployment. Under-utilised warm replicas are
   the first thing to cut.
2. `min_replicas` against the actual traffic floor — read the daily minimum, not
   the average.
3. Autoscaler behaviour: did `max_replicas` bind, and when?
4. GPU workers idle with no job running.
5. Storage growth: traces, artefacts, bundles.
6. Whether the TTFA SLO still has headroom. Capacity taken out of a system with
   no headroom is a latency incident next month.

Record the readings with the commands that produced them. A capacity review
without its queries cannot be compared to the next one.
