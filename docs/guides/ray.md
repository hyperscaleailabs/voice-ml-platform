# Guide: Ray

## Purpose

Ray appears in three roles, and each is a separate optional extra so that none
of them is required for the others.

| Role | Module | What it does | Runs as |
|---|---|---|---|
| **Ray Data** | `vmp.training.ray_jobs` | Maps one row transform over a JSONL corpus in parallel | part of a `RayJob` |
| **Ray Train** | `vmp.training.ray_jobs` | Wraps the same SFT / DPO / Whisper runner in a `TorchTrainer` | `RayJob` on KubeRay |
| **Ray Serve** | `vmp.serving.ray_serve` | Runtime, STT, LLM and TTS as deployments with independent autoscaling | `RayService` on KubeRay |

The design point is **one `TrainingPlan`, two backends**. `compute.backend` is
`local` or `ray`; the plan is byte-for-byte identical, its `config_hash` is the
same, and the Ray launcher only wraps the local runner. Nothing about the
training recipe knows whether it is running on a laptop or on a cluster.

## Config

Training moves to Ray by changing one table in the training TOML:

```toml
[compute]
backend = "ray"        # local | ray
num_workers = 4
gpu_per_worker = 1.0
```

`num_workers > 1` requires `backend = "ray"`; the plan validator rejects the
combination with `compute.num_workers > 1 requires compute.backend = 'ray'`
rather than silently training on one process.

Serving is configured by `[serving.ray_serve]` in `configs/serving.toml`:

```toml
[serving.ray_serve]
app_name = "voice-agent"
route_prefix = "/"
http_host = "0.0.0.0"
http_port = 8000
llm_mode = "openai-compat"   # openai-compat (proxy to a vLLM server) | vllm (in-process engine)

[serving.ray_serve.stt]
min_replicas = 1
max_replicas = 4
target_ongoing_requests = 2
num_cpus = 1.0
num_gpus = 0.0

[serving.ray_serve.llm]
min_replicas = 1
max_replicas = 2
target_ongoing_requests = 8
num_cpus = 1.0
num_gpus = 0.0

[serving.ray_serve.tts]
min_replicas = 1
max_replicas = 4
target_ongoing_requests = 2
num_cpus = 1.0
num_gpus = 0.0

[serving.ray_serve.ingress]
min_replicas = 1
max_replicas = 2
target_ongoing_requests = 16
num_cpus = 0.5
```

The four deployments have separate autoscaling because they saturate at
different points. STT and TTS are short, CPU-or-GPU bound calls that queue
badly, so `target_ongoing_requests` is low. The LLM deployment fronts a server
that does its own continuous batching, so it tolerates a much higher number.
The ingress does almost no work per request and mostly holds the session.

## CLI

```bash
# Training: same commands, backend chosen in the TOML
vmp train plan --config configs/train_sft.toml                 # validate and hash
vmp train sft  --config configs/train_sft.toml --dry-run       # returns the Ray config it would submit
vmp train sft  --config configs/train_sft.toml --ray-address ray://head:10001

# Serving
vmp serve ray --config configs/serving.toml --dry-run          # the deployment graph as JSON
vmp serve ray --config configs/serving.toml --yaml             # a `serve run` config file
vmp serve ray --config configs/serving.toml                    # serve.run(), needs [ray]
```

## What a dry run returns

**Training with `backend = "ray"`** returns the normal plan manifest plus a
`ray` block that is the entire `TorchTrainer(...)` call as data — serialisable,
reviewable, and produced without importing Ray:

```json
{
  "kind": "sft",
  "config_hash": "…",
  "ray": {
    "trainer": "ray.train.torch.TorchTrainer",
    "train_loop_per_worker": "vmp.training.ray_jobs.train_loop_per_worker",
    "train_loop_config": { "…the whole plan…" },
    "scaling_config": {
      "num_workers": 4,
      "use_gpu": true,
      "resources_per_worker": {"GPU": 1.0}
    },
    "run_config": {"name": "sft-1924cf7bd788", "storage_path": "…"},
    "address": null
  },
  "dry_run": true
}
```

The run name is `<kind>-<first 12 of config_hash>`, so two runs of the same
plan collide by design and a changed hyperparameter produces a new run
directory.

**Serving** returns the deployment graph. This is the real output of
`vmp serve ray --config configs/serving.toml --dry-run`, abridged:

```json
{
  "app_name": "voice-agent",
  "route_prefix": "/",
  "http": {"host": "0.0.0.0", "port": 8000},
  "import_path": "vmp.serving.ray_serve:app",
  "deployments": {
    "stt": {
      "class": "STTDeployment",
      "autoscaling_config": {"min_replicas": 1, "max_replicas": 4, "target_ongoing_requests": 2.0},
      "ray_actor_options": {"num_cpus": 1.0},
      "backend": "echo"
    },
    "llm": {
      "class": "LLMDeployment",
      "autoscaling_config": {"min_replicas": 1, "max_replicas": 2, "target_ongoing_requests": 8.0},
      "ray_actor_options": {"num_cpus": 1.0},
      "backend": "openai-compat",
      "model": "default"
    },
    "tts": {"class": "TTSDeployment", "…": "…", "backend": "silent"},
    "ingress": {"class": "VoiceAgentIngress", "…": "…"}
  },
  "edges": [["ingress", "stt"], ["ingress", "llm"], ["ingress", "tts"]],
  "dry_run": true
}
```

`--yaml` prints the same graph as a Ray Serve config schema v2 document, which
is exactly what goes into a `RayService`'s `serveConfigV2`.

## What the real run needs

```bash
pip install -e ".[ray]"          # ray[train] for jobs, ray[serve] for the graph
```

For training, `RayTrainLauncher.submit` calls `ray.init(address=...)` when Ray
is not already initialised, builds `ScalingConfig` and `RunConfig` from the
dicts above, and runs `TorchTrainer.fit()`. Every worker executes
`train_loop_per_worker`, which rebuilds the plan from `train_loop_config` and
calls the same `run_sft` / `run_dpo` / `run_whisper_lora` function the local
backend calls, then reports the manifest's `result` block through
`ray.train.report`. `TorchTrainer` has already set up the process group and the
`RANK` / `WORLD_SIZE` environment, which the Hugging Face trainers pick up for
data-parallel training on their own.

For serving, `serve.run(app, name=..., route_prefix=...)` starts the graph and
the process blocks until interrupted.

### Ray Data preprocessing

`RayDataPreprocessor` maps one row transform over a JSONL corpus. The same
function runs under `ray.data` or in a pure-Python loop, which is what lets the
transform be unit-tested with no cluster:

```python
from vmp.training.ray_jobs import RayDataPreprocessor, transform_for_kind

pre = RayDataPreprocessor(transform_for_kind("dpo"))
pre.run("data/preference_pairs.jsonl", "data/prepared", use_ray=False)   # python
pre.run("data/preference_pairs.jsonl", "data/prepared", use_ray=True)    # ray.data
```

The manifest reports `engine` (`python` or `ray.data`), `rows_in` and
`rows_out`, so a mismatch between the two paths shows up as a count rather than
as a silent difference in training data.

## On KubeRay

Two custom resources, both in `deploy/ray/`.

**`RayJob`** for training (`deploy/ray/rayjob-sft.yaml`,
`deploy/ray/rayjob-dpo.yaml`). Its `entrypoint` is the same command a developer
runs locally:

```yaml
spec:
  entrypoint: vmp train sft --config /app/configs/train_sft.toml --dry-run
  clusterSelector:
    ray.io/cluster: vmp-ray
  shutdownAfterJobFinishes: false
  ttlSecondsAfterFinished: 86400
```

Drop `--dry-run` to train. `clusterSelector` attaches to the long-lived cluster
in `deploy/ray/raycluster.yaml`; removing it and supplying `rayClusterSpec`
instead gets an ephemeral cluster that is torn down when the job finishes,
which is usually what a scheduled retrain wants.

**`RayService`** for serving (`deploy/ray/rayservice-voice.yaml`). Its
`serveConfigV2` is the document `vmp serve ray --yaml` prints; regenerate it
from the TOML rather than editing both. KubeRay's update semantics are the part
worth internalising: a change to `serveConfigV2` alone is applied in place,
while a change to `rayClusterConfig` starts a **second** cluster and cuts
traffic over once the new Serve applications report healthy. The first is
cheap, the second doubles your cluster footprint for the duration of the
rollout.

## Autoscaling

Two layers, and they are independent.

- **Serve replica autoscaling** moves replicas of one deployment between
  `min_replicas` and `max_replicas` to keep the average number of in-flight
  requests per replica near `target_ongoing_requests`. `upscale_delay_s` and
  `downscale_delay_s` in the KubeRay manifest damp it: scaling up fast and down
  slow is right for a voice workload, where a cold replica costs a user a turn.
- **Ray cluster autoscaling** adds and removes nodes when replicas cannot be
  placed. It reacts to pending actors, not to latency, so it is always slower
  than the replica layer.

Because the four deployments scale independently, the meaningful question is
which stage is actually saturating. `vmp obs summarise` answers it per stage,
and the stage that dominates `ttfa` is the one whose `max_replicas` or GPU
fraction should move. Raising every deployment together buys cost and not
latency.

## Pitfalls

- **`min_replicas = 0` costs a turn.** Scale-to-zero is attractive for the TTS
  and STT deployments and terrible for time-to-first-audio: the first request
  after an idle period pays the model load. Keep one warm replica, or accept
  the SLO breach and say so.
- **`num_gpus` is a fraction, and fractions are not isolation.** Two
  deployments at `num_gpus = 0.5` share one device's memory with no enforcement.
  Give the LLM deployment a whole GPU when it holds a KV cache.
- **The plan is hashed, the cluster is not.** `config_hash` covers the training
  recipe, not the Ray version, the image, or the node type. Record those in the
  registry artifact's metrics or lineage if you intend to compare runs across
  clusters.
- **`storage_path` defaults to the parent of `output_dir`.** On a multi-node
  cluster that path must be shared storage, or checkpoints land on whichever
  node happened to run the worker.
- **A dry run never imports Ray, so it never validates the cluster.** It checks
  the plan, not that `ray://head:10001` is reachable or that four GPUs exist.
  The first real submission is where those fail.
- **Ray Serve and the FastAPI app are two deployment paths, not two products.**
  The single-process [`vmp serve api`](serving-api.md) and the Serve graph run
  the same runtime with the same protocols. Develop against the first; deploy
  the second.
