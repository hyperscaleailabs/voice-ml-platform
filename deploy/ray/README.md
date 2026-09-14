# deploy/ray/ — KubeRay manifests

Ray is used three times in this platform, for three different reasons. The
manifests here are one file per role.

| File | Ray library | Role |
|---|---|---|
| `raycluster.yaml` | Ray Core + Ray Data | The long-lived cluster. CPU workers run preprocessing; GPU workers are added on demand. |
| `rayjob-sft.yaml`, `rayjob-dpo.yaml` | Ray Train | One training run, submitted, watched, finished. |
| `rayservice-voice.yaml` | Ray Serve | The inference graph: STT, LLM, TTS behind a FastAPI ingress. |

Prerequisite: the KubeRay operator.

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator -n platform --create-namespace
```

## 1. Ray Data — preprocessing

`vmp.training.ray_jobs.RayDataPreprocessor` applies one row transform to a JSONL
corpus: PII scrub, chat formatting, split assignment. The same transform runs in
a plain Python loop when Ray is not installed, which is how the tests and
`examples/demo_training_plan.py` exercise it. At corpus scale it runs as a
lazy `ray.data` pipeline on the `cpu` worker group, reading from and writing back
to MinIO (`VMP_S3_ENDPOINT`).

The CPU group has `minReplicas: 1` so a submitted job never waits for a cold
node just to read data.

## 2. Ray Train — the training job

`RayTrainLauncher.build_config(plan)` turns a `TrainingPlan` into the arguments
of a `ray.train.torch.TorchTrainer`. Nothing about the plan changes between a
laptop run and a cluster run.

### How a TrainingPlan maps onto a RayJob

A `TrainingPlan` is a TOML file in `configs/` loaded into the dataclass in
`vmp/training/plan.py`. Each part of it lands somewhere concrete:

| TrainingPlan field | Where it goes |
|---|---|
| `kind` (`sft` / `dpo` / `whisper-lora`) | The `vmp train <kind>` subcommand in `spec.entrypoint`, and the runner selected inside `train_loop_per_worker`. |
| the TOML file itself | `--config /app/configs/<file>.toml`, baked into the training image. |
| `compute.num_workers` | `ScalingConfig.num_workers` — how many GPU worker pods the autoscaler is asked for. |
| `compute.gpu_per_worker` | `ScalingConfig.resources_per_worker` and the `nvidia.com/gpu` request on the `gpu` group. |
| `compute.backend = "ray"` | Selects `RayTrainLauncher` over the in-process runner. Everything else is identical. |
| `output_dir` | `RunConfig.storage_path` — checkpoints and the merged adapter. |
| `config_hash()` | `RunConfig.name`, so a rerun of the same plan is recognisable, and the `config_hash` field of the resulting `ModelArtifact`. |

So a RayJob is a `TrainingPlan` plus a submission: the plan decides the shape of
the cluster, the job decides when it runs.

```bash
kubectl apply -f deploy/ray/rayjob-sft.yaml
kubectl get rayjob -n platform vmp-sft -o jsonpath='{.status.jobStatus}{"\n"}'
kubectl logs -n platform -l job-name=vmp-sft -f
```

Both job manifests ship with `--dry-run` in the entrypoint. A dry run validates
the plan, prints the manifest and imports nothing heavy, so applying them to a
new cluster proves the wiring without booking a GPU. Remove the flag to train.

By default the jobs attach to the existing `vmp-ray` cluster
(`clusterSelector`). Replace that with a `rayClusterSpec` block and
`shutdownAfterJobFinishes: true` for an ephemeral per-job cluster, which is the
better choice when GPUs are rented by the minute.

## 3. Ray Serve — inference

`rayservice-voice.yaml` carries four deployments, matching
`vmp.serving.ray_serve.DEPLOYMENTS`:

- **STTDeployment** — audio to text. Short requests, CPU by default.
- **LLMDeployment** — the model, one GPU per replica. Emits `ttft_ms`.
- **TTSDeployment** — text to audio, one call per emitted sentence.
- **VoiceAgentIngress** — binds the FastAPI app from `vmp.serving.api`, holds
  sessions, runs the turn loop, writes the trace.

`serveConfigV2` is the same document `vmp serve ray --yaml` prints:

```bash
vmp serve ray --config configs/serving.toml --yaml
```

Regenerate it from the TOML rather than editing the manifest and the config
separately. The autoscaling settings differ from the defaults in `serving.toml`
in one deliberate way: the LLM has a long `downscale_delay_s` (600 s) because
dropping a replica means reloading weights, while STT and TTS scale in and out
quickly. `min_replicas` for the ingress is 2 so an ingress restart never leaves a
session with nowhere to land.

KubeRay updates a RayService in one of two ways. A change confined to
`serveConfigV2` is applied in place, deployment by deployment. A change to
`rayClusterConfig` triggers a zero-downtime upgrade: a second cluster starts,
Serve applications become healthy there, traffic moves, the old cluster is torn
down. That second path is what `runbooks/rollback-model.md` uses to roll a model
back.

## Observability

Head and worker pods export spans to `otel-collector:4317` and expose Ray's own
metrics on port 8080, which Prometheus scrapes through the pod annotations. Ray's
dashboard is told where Grafana and Prometheus live (`RAY_GRAFANA_HOST`,
`RAY_PROMETHEUS_HOST`) so its built-in panels resolve inside the cluster.
