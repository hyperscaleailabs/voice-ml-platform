# 03 — Ray scaling

## Claim

The same `TrainingPlan` that runs on one machine submits unchanged to Ray Train
with more than one worker. Scaling out is a change to the `[compute]` table of a
TOML file — `backend`, `num_workers`, `gpu_per_worker` — and not a rewrite of the
training code, a second entry point, or a separate distributed script.

The underlying assertion is about where the distribution lives: the runner
(`run_sft`, `run_dpo`, `run_whisper_lora`) stays single-process code, and Ray
Train supplies the process group and the `RANK`/`WORLD_SIZE` environment that the
Hugging Face trainers already know how to read.

## Falsifier

- The plan needs any edit beyond `[compute]` to submit — a different dataset
  path shape, a different entry point, a distributed-only flag.
- The runner needs a branch on `compute.backend`. If `run_sft` has to know
  whether it is on a cluster, the claim is false: the plan is not the interface,
  the runner is.
- A 4-worker run does not converge to a comparable loss on the same data as the
  1-worker run within the same number of epochs, after accounting for the
  effective batch size. Submitting is not the same as training.
- The submitted job fails on a real cluster for a reason that is not a resource
  shortage — a serialisation failure of the plan, a missing import on the worker,
  a path that only exists on the submitting machine.

The last two cannot be evaluated without a cluster, which is why the outcome
below is what it is.

## Method

`spike_ray_plan.py` loads `configs/train_sft.toml` twice into two
`TrainingPlan` objects over the same 64-row synthetic corpus. One keeps the
file's local compute block; the other replaces it with
`ComputeConfig(backend="ray", num_workers=4, gpu_per_worker=1.0)`. Nothing else
is touched. The first goes through `run_sft(plan, dry_run=True)`, the second
through `RayTrainLauncher(address="auto", storage_path="runs/ray").submit(plan,
dry_run=True)`. The script prints both manifests side by side and then diffs
them key by key.

`ray` is never imported. The launcher builds the `TorchTrainer(...)` call as a
plain serialisable dict — `train_loop_per_worker`, `train_loop_config`,
`scaling_config`, `run_config` — and imports Ray only when `dry_run=False`. That
is what lets this spike run with the standard library, and it is also the design
point being tested: if the Ray call can be built as data, the plan is the
interface.

**What the diff shows.**

- The plan differs in exactly three keys, all of them in `compute`:
  `backend`, `num_workers`, `gpu_per_worker`.
- `data_hash` and the dataset row count are identical. Nothing about the data
  changed.
- `config_hash` differs, because `compute` is part of the plan and therefore part
  of the lineage. This is deliberate: a 4-worker run is not recorded as the same
  run as a local one. The claim is that it needs no code change, not that it is
  the same run.
- `estimated_steps` falls from 4 to 1, because the global batch is
  `per_device_batch_size x gradient_accumulation x num_workers`. Same data,
  larger effective batch, fewer optimizer steps. Anyone scaling workers without
  touching the learning rate or the epoch count is changing the optimisation,
  and the manifest says so before the job starts.
- The runner-specific keys (`trainer`, `peft`, `system_prompt`, `formatting`)
  appear only on the local side. They are added by `run_sft`; under Ray the same
  `run_sft` adds them on the worker, inside `train_loop_per_worker`.
- 29 keys exist only under `ray`, 21 of which are the plan itself, serialised
  into `train_loop_config` and handed to each worker verbatim.

## Outcome

`not run`

There is no Ray cluster in this repository and none was started. What has been
executed is the dry-run path on both sides: the plan serialises, the launcher
builds a complete `TorchTrainer` call from it, and the only differences between
the two manifests are the compute block and the values derived from it.

That is evidence for the *shape* of the claim and not for the claim. The two
falsifier clauses that matter — that the job runs on a cluster, and that 4
workers converge comparably to 1 — need hardware. Nothing here may be cited as
a scaling result, a throughput figure, or a speed-up.

The one substantive finding available without a cluster is the step count:
scaling workers silently rescales the effective batch. Any future run must hold
the effective batch fixed, or report that it did not.

## What moved into src/

- `vmp.training.plan.ComputeConfig` — `backend`, `num_workers`,
  `gpu_per_worker`, with the validation that `num_workers > 1` requires
  `backend = "ray"`, and inclusion in `config_hash` so the lineage records where
  a run executed.
- `vmp.training.ray_jobs.RayTrainLauncher` — `scaling_config`, `run_config` and
  `build_config` as plain dicts; `submit(plan, dry_run=True)` returns the
  manifest without importing Ray.
- `vmp.training.ray_jobs.train_loop_per_worker` — the single worker entry point
  that reconstructs the plan from `train_loop_config` and calls the same runner
  the local path calls.
- `vmp.training.ray_jobs.RayDataPreprocessor` — one row transform, applied
  either through `ray.data` or through a pure-Python loop, so the transform is
  testable without a cluster.
- `estimate_steps` in `vmp.training.plan` — the global-batch calculation that
  makes the step-count effect visible in the dry run.
