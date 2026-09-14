"""Ray Train launcher and Ray Data preprocessing for a `TrainingPlan`.

`RayTrainLauncher` turns a plan into a `ray.train.torch.TorchTrainer` with a
`ScalingConfig` derived from `plan.compute` and a `RunConfig` derived from
`plan.output_dir`. The same plan that runs locally submits to Ray with
`compute.backend = "ray"`; nothing else changes. `ray` is imported only when
`dry_run=False`.

`RayDataPreprocessor` applies one row transform to a JSONL corpus, either with
`ray.data` (lazy) or with a pure-Python loop over the same function, so tests
and demos exercise the transform without Ray.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from vmp.training.plan import TrainingPlan, base_manifest, read_jsonl, write_jsonl
from vmp.training.sft import format_chat_example

RowTransform = Callable[[dict[str, Any]], dict[str, Any]]

TRAIN_LOOP = "vmp.training.ray_jobs.train_loop_per_worker"


def train_loop_per_worker(config: dict[str, Any]) -> None:
    """Entry point executed on every Ray Train worker.

    The plan arrives as `train_loop_config`; `TorchTrainer` has already set up
    the `torch.distributed` process group and the `RANK`/`WORLD_SIZE`
    environment, which the HF trainers pick up for data-parallel training.
    """
    from ray import train

    plan = TrainingPlan.from_dict(config)
    if plan.kind == "sft":
        from vmp.training.sft import run_sft as runner
    elif plan.kind == "dpo":
        from vmp.training.dpo import run_dpo as runner
    else:
        from vmp.training.whisper_lora import run_whisper_lora as runner
    manifest = runner(plan, dry_run=False)
    train.report(manifest.get("result", {}))


class RayTrainLauncher:
    """Build and submit a `TorchTrainer` for a plan. Configs are plain dicts first."""

    def __init__(self, address: str | None = None, storage_path: str | None = None) -> None:
        self.address = address
        self.storage_path = storage_path

    def scaling_config(self, plan: TrainingPlan) -> dict[str, Any]:
        gpu = float(plan.compute.gpu_per_worker)
        return {
            "num_workers": plan.compute.num_workers,
            "use_gpu": gpu > 0,
            "resources_per_worker": {"GPU": gpu} if gpu > 0 else {"CPU": 1},
        }

    def run_config(self, plan: TrainingPlan) -> dict[str, Any]:
        out = Path(plan.output_dir)
        return {
            "name": f"{plan.kind}-{plan.config_hash()[:12]}",
            "storage_path": self.storage_path or str(out.parent.resolve()),
        }

    def build_config(self, plan: TrainingPlan) -> dict[str, Any]:
        """The full `TorchTrainer(...)` call as data. Serialisable, no Ray import."""
        return {
            "trainer": "ray.train.torch.TorchTrainer",
            "train_loop_per_worker": TRAIN_LOOP,
            "train_loop_config": plan.to_dict(),
            "scaling_config": self.scaling_config(plan),
            "run_config": self.run_config(plan),
            "address": self.address,
        }

    def submit(self, plan: TrainingPlan, dry_run: bool = True) -> dict[str, Any]:
        """Submit the plan to Ray Train, or return the manifest with the Ray config."""
        if plan.compute.backend != "ray":
            raise ValueError(
                f"plan.compute.backend is {plan.compute.backend!r}; set it to 'ray' to submit"
            )
        manifest = base_manifest(plan, dry_run, required_fields=())
        manifest["ray"] = self.build_config(plan)
        if dry_run:
            return manifest

        import ray
        from ray.train import RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer

        if not ray.is_initialized():
            ray.init(address=self.address)
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop_per_worker,
            train_loop_config=plan.to_dict(),
            scaling_config=ScalingConfig(**self.scaling_config(plan)),
            run_config=RunConfig(**self.run_config(plan)),
        )
        result = trainer.fit()
        manifest["result"] = {
            "metrics": dict(result.metrics or {}),
            "checkpoint": str(result.checkpoint.path) if result.checkpoint else None,
            "path": str(result.path),
        }
        return manifest


# -- preprocessing -------------------------------------------------------------


def tokenize_whitespace(row: dict[str, Any], text_field: str = "text") -> dict[str, Any]:
    """Reference transform: whitespace tokens and their count. Stdlib only."""
    tokens = str(row.get(text_field, "")).split()
    return {**row, "tokens": tokens, "n_tokens": len(tokens)}


def transform_for_kind(kind: str) -> RowTransform:
    """The row transform each training kind expects."""
    if kind == "sft":
        return format_chat_example
    if kind == "dpo":

        def dpo_rows(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
                "chosen_words": len(str(row["chosen"]).split()),
                "rejected_words": len(str(row["rejected"]).split()),
            }

        return dpo_rows
    if kind == "whisper-lora":

        def audio_rows(row: dict[str, Any]) -> dict[str, Any]:
            return {"audio_path": row["audio_path"], "text": row["text"].strip()}

        return audio_rows
    raise ValueError(f"unknown kind {kind!r}")


class RayDataPreprocessor:
    """Map a JSONL corpus row by row. Same function on Ray Data or in pure Python."""

    def __init__(self, transform: RowTransform = tokenize_whitespace) -> None:
        self.transform = transform

    def map_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pure-Python path. Used by tests and by `run(..., use_ray=False)`."""
        return [self.transform(row) for row in rows]

    def run(
        self,
        in_path: str | Path,
        out_path: str | Path,
        dry_run: bool = False,
        use_ray: bool = False,
    ) -> dict[str, Any]:
        """Transform `in_path` into `out_path`. Returns counts and the engine used."""
        src = Path(in_path)
        if not src.is_file():
            raise FileNotFoundError(f"corpus not found: {src}")
        manifest: dict[str, Any] = {
            "in_path": str(src),
            "out_path": str(out_path),
            "transform": getattr(self.transform, "__name__", repr(self.transform)),
            "engine": "ray.data" if use_ray else "python",
            "dry_run": dry_run,
            "rows_in": None,
            "rows_out": None,
        }
        if dry_run:
            manifest["rows_in"] = len(read_jsonl(src))
            return manifest
        if use_ray:
            import ray

            ds = ray.data.read_json(str(src))
            out = ds.map(self.transform)
            Path(out_path).mkdir(parents=True, exist_ok=True)
            out.write_json(str(out_path))
            manifest["rows_in"] = ds.count()
            manifest["rows_out"] = out.count()
            return manifest
        rows = read_jsonl(src)
        manifest["rows_in"] = len(rows)
        manifest["rows_out"] = write_jsonl(out_path, self.map_rows(rows))
        return manifest


__all__ = [
    "TRAIN_LOOP",
    "RayDataPreprocessor",
    "RayTrainLauncher",
    "tokenize_whitespace",
    "train_loop_per_worker",
    "transform_for_kind",
]
