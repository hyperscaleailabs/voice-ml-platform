---
date: 2026-08-31
authors: [cg]
categories:
  - Infrastructure
  - Training
slug: ray-as-the-platform-spine
---

# Ray as the platform spine

Most ML platforms end up with two training code paths. There is the one that runs
on a laptop, and there is the distributed one, and they diverge within a month.
A flag appears that only makes sense on a cluster. A path is hard-coded to shared
storage. The local script grows a branch for `if world_size > 1`. Eventually the
two paths train subtly different models and nobody notices until an evaluation
disagrees with a production result.

The design goal here is one plan, two backends, and no second code path.

<!-- more -->

## Three roles, three extras

Ray shows up in three places in this platform, and each is a separate optional
dependency so that none of them drags in the others.

| Role | Module | What it does | Runs as |
|---|---|---|---|
| **Ray Data** | `vmp.training.ray_jobs` | Maps one row transform over a JSONL corpus in parallel | part of a `RayJob` |
| **Ray Train** | `vmp.training.ray_jobs` | Wraps the same SFT / DPO / Whisper runner in a `TorchTrainer` | `RayJob` on KubeRay |
| **Ray Serve** | `vmp.serving.ray_serve` | Runtime, STT, LLM and TTS as deployments with independent autoscaling | `RayService` on KubeRay |

Grouping them under one name is convenient and slightly misleading. They solve
different problems and could be replaced independently. What they share is a
scheduler and a mental model — actors, tasks, placement — which is worth
something when the alternative is three unrelated systems.

## One TrainingPlan, two backends

A `TrainingPlan` describes a run completely: kind, base model, dataset paths,
LoRA configuration, hyperparameters, seed, and a `[compute]` table. It is
validated eagerly and hashed canonically, and that hash is the lineage key in the
registry.

Moving a run to a cluster is this:

```toml
[compute]
backend = "ray"        # was "local"
num_workers = 4
gpu_per_worker = 1.0
```

Nothing else changes. Not the dataset format, not the entry point, not the
runner. `run_sft` does not know whether it is on a laptop or a cluster, and the
spike's falsifier says so explicitly: *if `run_sft` has to branch on
`compute.backend`, the plan is not the interface, the runner is.*

The mechanism is that Ray Train supplies the process group and the
`RANK`/`WORLD_SIZE` environment that the Hugging Face trainers already know how
to read. `train_loop_per_worker` reconstructs the plan from `train_loop_config`
and calls the same `run_sft` / `run_dpo` / `run_whisper_lora` the local path
calls, then reports the manifest's result block through `ray.train.report`. The
distribution lives in the launcher. The training lives in the runner. They do not
know about each other.

`RayTrainLauncher.build_config` returns the entire `TorchTrainer(...)` call as a
plain serialisable dict — trainer class, entry point, config, scaling, run
config — **without importing Ray**. That is what makes `--dry-run` work with the
standard library alone, and it is also the design being tested: if the Ray call
can be built as data, then the plan really is the interface.

```json
{"trainer": "ray.train.torch.TorchTrainer",
 "train_loop_per_worker": "vmp.training.ray_jobs.train_loop_per_worker",
 "train_loop_config": {"…the whole plan…"},
 "scaling_config": {"num_workers": 4, "use_gpu": true, "resources_per_worker": {"GPU": 1.0}},
 "run_config": {"name": "sft-1924cf7bd788", "storage_path": "…"}}
```

## The trap the dry run catches

Two things change when you raise `num_workers`, and only one of them is obvious.

`config_hash` changes, because `compute` is part of the plan and therefore part
of the lineage. That is deliberate: a 4-worker run is not recorded as the same
run as a local one, even though the data and the hyperparameters are identical.
The claim is that scaling needs no code change, not that it produces the same
run.

And `estimated_steps` **falls**. The global batch is
`per_device_batch_size × gradient_accumulation × num_workers`, so four workers
over the same data means a quarter of the optimizer steps at four times the
effective batch. Same corpus, different optimisation. Anyone who scales workers
without touching the learning rate or the epoch count has silently changed the
experiment.

The manifest reports it before the job starts, which is the entire point of
having a planning path. A number that only appears in a training log is a number
you read after paying for the run.

## Ray Data, and why the transform is testable

`RayDataPreprocessor` maps one row transform over a JSONL corpus. The same
function runs under `ray.data` or in a pure-Python loop:

```python
pre = RayDataPreprocessor(transform_for_kind("dpo"))
pre.run("data/preference_pairs.jsonl", "data/prepared", use_ray=False)   # python
pre.run("data/preference_pairs.jsonl", "data/prepared", use_ray=True)    # ray.data
```

The manifest reports `engine`, `rows_in` and `rows_out`. A mismatch between the
two paths shows up as a count rather than as a silent difference in training
data, which is the failure you want to catch cheaply: a preprocessing step that
behaves differently in the distributed path is a class of bug that produces a
working model with wrong behaviour.

## Ray Serve: four deployments, four autoscalers

The serving graph is an ingress and three model deployments, each with its own
autoscaling config. That separation is the useful part, because the four
saturate at completely different points.

```toml
[serving.ray_serve.stt]
min_replicas = 1
max_replicas = 4
target_ongoing_requests = 2

[serving.ray_serve.llm]
min_replicas = 1
max_replicas = 2
target_ongoing_requests = 8
```

STT and TTS are short calls that queue badly, so their target is low. The LLM
deployment fronts a server doing its own continuous batching, so it tolerates a
much higher number — that is what a paged KV cache buys. The ingress does almost
no work per request and mostly holds the session.

`vmp serve ray --yaml` prints the Ray Serve config schema v2 document, which is
what goes into a `RayService`'s `serveConfigV2`. Generate it from the TOML rather
than maintaining both; the manifest in `deploy/ray/rayservice-voice.yaml` says as
much in a comment, because the failure mode of editing both is a serving graph
that no longer matches the config the team reads.

## KubeRay: two resources, two update semantics

**`RayJob`** for training. Its `entrypoint` is literally the command a developer
runs locally:

```yaml
spec:
  entrypoint: vmp train sft --config /app/configs/train_sft.toml --dry-run
  clusterSelector:
    ray.io/cluster: vmp-ray
```

Drop `--dry-run` to train. `clusterSelector` attaches to a long-lived cluster;
replacing it with `rayClusterSpec` gets an ephemeral one torn down when the job
finishes, which is usually right for a scheduled retrain.

**`RayService`** for serving, and its update semantics are the operational fact
worth internalising: a change to `serveConfigV2` alone is applied **in place**,
while a change to `rayClusterConfig` starts a **second cluster** and cuts traffic
over once the new Serve applications report healthy. The first is cheap. The
second doubles your cluster footprint for the duration of the rollout — fine if
you planned for it, a capacity incident if you did not.

## Autoscaling, and the thing scale-to-zero costs

Two layers, independent of each other. Serve replica autoscaling moves replicas
of one deployment to keep in-flight requests per replica near the target. Cluster
autoscaling adds nodes when replicas cannot be placed — it reacts to pending
actors, not to latency, so it is always the slower of the two. `upscale_delay_s`
and `downscale_delay_s` damp the replica layer; scaling up fast and down slow is
right for a voice workload.

`min_replicas = 0` is where the money is and where the user experience goes. For
a voice agent, the first request after an idle period pays the model load, and
that cost lands entirely on time-to-first-audio — the one number the user
actually perceives. Keep one warm replica, or accept the SLO breach and say so in
the SLO document rather than discovering it in a complaint.

The other trap is fractional GPUs. Two deployments at `num_gpus = 0.5` share one
device's memory with no enforcement between them. Give the LLM deployment a whole
device when it holds a KV cache.

## What has not been demonstrated

There is no Ray cluster in this repository and none was started. The research
spike (`research/03-ray-scaling/`) is `not run`, and its outcome section is
explicit: the dry-run path executes on both sides, the plan serialises, the
launcher builds a complete `TorchTrainer` call from it, and the only differences
between the two manifests are the compute block and what derives from it.

That is evidence for the **shape** of the claim. It is not evidence for the
claim. The two falsifier clauses that matter — the job runs on a real cluster,
and four workers converge comparably to one — need hardware. No throughput
figure, no speed-up and no scaling result appears anywhere in this repository,
because none has been measured.

What did come out of the dry run is the step-count finding, and that one is real
and free: scaling workers rescales the effective batch. Any future run has to
hold it fixed, or report that it did not.
