"""`TrainingPlan`: the single description of a training run, and the dry-run planner.

A plan is built from a TOML file in `configs/`, validated eagerly, hashed
canonically (`config_hash`) so the registry can record lineage, and handed to
one of the runners (`sft`, `dpo`, `whisper_lora`) or to the Ray launcher. Every
runner accepts `dry_run=True`, in which case only this module's stdlib helpers
are used: the dataset is scanned, steps are estimated and a manifest is returned.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vmp.config import deep_merge, load_config

KINDS = ("sft", "dpo", "whisper-lora")
BACKENDS = ("local", "ray")
TASK_TYPES = ("CAUSAL_LM", "SEQ_2_SEQ_LM")

# Hyperparameters every runner understands. A TOML `[hyperparams]` table is
# merged on top of the entry for its kind, so a config only lists what it changes.
DEFAULT_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "sft": {
        "epochs": 1,
        "learning_rate": 2e-4,
        "per_device_batch_size": 4,
        "gradient_accumulation": 4,
        "max_seq_length": 1024,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "max_steps": 0,
    },
    "dpo": {
        "epochs": 1,
        "learning_rate": 5e-6,
        "per_device_batch_size": 2,
        "gradient_accumulation": 8,
        "max_seq_length": 1024,
        "max_prompt_length": 512,
        "beta": 0.1,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "max_steps": 0,
    },
    "whisper-lora": {
        "epochs": 3,
        "learning_rate": 1e-3,
        "per_device_batch_size": 8,
        "gradient_accumulation": 1,
        "warmup_ratio": 0.05,
        "logging_steps": 5,
        "max_steps": 0,
        "language": "en",
        "task": "transcribe",
    },
}


class PlanError(ValueError):
    """A `TrainingPlan` failed validation. The message names the offending field."""


@dataclass(frozen=True)
class LoraConfig:
    """PEFT LoRA hyperparameters. Mirrors `peft.LoraConfig` without importing it."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    task_type: str = "CAUSAL_LM"

    def validate(self) -> None:
        if not isinstance(self.r, int) or self.r <= 0:
            raise PlanError(f"lora.r must be a positive int, got {self.r!r}")
        if not isinstance(self.alpha, int) or self.alpha <= 0:
            raise PlanError(f"lora.alpha must be a positive int, got {self.alpha!r}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise PlanError(f"lora.dropout must be in [0, 1), got {self.dropout!r}")
        if not self.target_modules or not all(
            isinstance(m, str) and m for m in self.target_modules
        ):
            raise PlanError("lora.target_modules must be a non-empty list of module names")
        if self.task_type not in TASK_TYPES:
            raise PlanError(f"lora.task_type must be one of {TASK_TYPES}, got {self.task_type!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoraConfig:
        d = dict(data)
        if "target_modules" in d:
            d["target_modules"] = tuple(d["target_modules"])
        return cls(**d)

    def to_peft_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `peft.LoraConfig(...)`."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "task_type": self.task_type,
            "bias": "none",
        }


@dataclass(frozen=True)
class ComputeConfig:
    """Where the run executes. `ray` hands the same plan to `RayTrainLauncher`."""

    backend: str = "local"
    num_workers: int = 1
    gpu_per_worker: float = 0.0

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise PlanError(f"compute.backend must be one of {BACKENDS}, got {self.backend!r}")
        if not isinstance(self.num_workers, int) or self.num_workers < 1:
            raise PlanError(f"compute.num_workers must be >= 1, got {self.num_workers!r}")
        if self.backend == "local" and self.num_workers != 1:
            raise PlanError("compute.num_workers > 1 requires compute.backend = 'ray'")
        if float(self.gpu_per_worker) < 0:
            raise PlanError(f"compute.gpu_per_worker must be >= 0, got {self.gpu_per_worker!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComputeConfig:
        return cls(**data)


@dataclass
class TrainingPlan:
    """Everything a training run needs, in one serialisable object.

    `datasets` maps split name -> JSONL path; `train` is required, `eval` optional.
    `hyperparams` is kept as a plain dict because each kind has its own set; the
    keys every runner reads are listed in `DEFAULT_HYPERPARAMS`.
    """

    kind: str
    base_model: str
    datasets: dict[str, str]
    output_dir: str
    lora: LoraConfig = field(default_factory=LoraConfig)
    hyperparams: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    compute: ComputeConfig = field(default_factory=ComputeConfig)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation -----------------------------------------------------------

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise PlanError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise PlanError("base_model must be a non-empty string")
        if not isinstance(self.datasets, dict) or "train" not in self.datasets:
            raise PlanError("datasets must be a table with at least a 'train' path")
        for split, path in self.datasets.items():
            if not isinstance(path, str) or not path:
                raise PlanError(f"datasets.{split} must be a non-empty path string")
        if not isinstance(self.output_dir, str) or not self.output_dir:
            raise PlanError("output_dir must be a non-empty string")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise PlanError(f"seed must be a non-negative int, got {self.seed!r}")
        if not isinstance(self.lora, LoraConfig):
            raise PlanError("lora must be a LoraConfig")
        self.lora.validate()
        if not isinstance(self.compute, ComputeConfig):
            raise PlanError("compute must be a ComputeConfig")
        self.compute.validate()
        expected_task = "SEQ_2_SEQ_LM" if self.kind == "whisper-lora" else "CAUSAL_LM"
        if self.lora.task_type != expected_task:
            raise PlanError(
                f"kind {self.kind!r} needs lora.task_type = {expected_task!r}, "
                f"got {self.lora.task_type!r}"
            )
        hp = self.hyperparams
        if not isinstance(hp, dict):
            raise PlanError("hyperparams must be a table")
        for key in ("epochs", "learning_rate"):
            if key in hp and (not isinstance(hp[key], int | float) or hp[key] <= 0):
                raise PlanError(f"hyperparams.{key} must be a positive number, got {hp[key]!r}")
        for key in ("per_device_batch_size", "gradient_accumulation"):
            if key in hp and (not isinstance(hp[key], int) or hp[key] < 1):
                raise PlanError(f"hyperparams.{key} must be an int >= 1, got {hp[key]!r}")
        if "max_steps" in hp and (not isinstance(hp["max_steps"], int) or hp["max_steps"] < 0):
            raise PlanError(f"hyperparams.max_steps must be an int >= 0, got {hp['max_steps']!r}")
        if self.kind == "dpo":
            beta = hp.get("beta", DEFAULT_HYPERPARAMS["dpo"]["beta"])
            if not isinstance(beta, int | float) or beta <= 0:
                raise PlanError(f"hyperparams.beta must be a positive number, got {beta!r}")

    # -- serialisation --------------------------------------------------------

    def hyperparam(self, key: str) -> Any:
        """A hyperparameter with the kind's default applied."""
        if key in self.hyperparams:
            return self.hyperparams[key]
        return DEFAULT_HYPERPARAMS[self.kind][key]

    def effective_hyperparams(self) -> dict[str, Any]:
        return deep_merge(DEFAULT_HYPERPARAMS[self.kind], self.hyperparams)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "base_model": self.base_model,
            "datasets": dict(self.datasets),
            "output_dir": self.output_dir,
            "lora": self.lora.to_dict(),
            "hyperparams": self.effective_hyperparams(),
            "seed": self.seed,
            "compute": self.compute.to_dict(),
        }

    def config_hash(self) -> str:
        """sha256 of the canonical JSON of the plan, excluding `output_dir`.

        The output directory does not change what is trained, so two runs of
        the same recipe into different directories share a hash and the
        registry can link them to the same configuration.
        """
        d = self.to_dict()
        d.pop("output_dir")
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPlan:
        if not isinstance(data, dict):
            raise PlanError("plan must be a table")
        missing = [k for k in ("kind", "base_model", "datasets", "output_dir") if k not in data]
        if missing:
            raise PlanError(f"plan is missing required keys: {', '.join(missing)}")
        kind = data["kind"]
        if kind not in KINDS:
            raise PlanError(f"kind must be one of {KINDS}, got {kind!r}")
        lora_data = dict(data.get("lora", {}))
        if kind == "whisper-lora":
            lora_data.setdefault("task_type", "SEQ_2_SEQ_LM")
        try:
            lora = LoraConfig.from_dict(lora_data)
            compute = ComputeConfig.from_dict(dict(data.get("compute", {})))
        except TypeError as e:  # unknown key
            raise PlanError(f"unknown field in plan: {e}") from e
        return cls(
            kind=kind,
            base_model=data["base_model"],
            datasets=dict(data["datasets"]),
            output_dir=data["output_dir"],
            lora=lora,
            hyperparams=dict(data.get("hyperparams", {})),
            seed=int(data.get("seed", 0)),
            compute=compute,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingPlan:
        """Load a plan. Top-level keys, or a `[train]` table when the file is shared."""
        data = load_config(path)
        if "kind" not in data and isinstance(data.get("train"), dict):
            data = data["train"]
        try:
            return cls.from_dict(data)
        except PlanError as e:
            raise PlanError(f"{path}: {e}") from e


# -- dry-run helpers (standard library only) -----------------------------------


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file into dicts. Blank lines are skipped; bad lines raise."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: expected an object, got {type(obj).__name__}")
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def dataset_stats(path: str | Path, required: tuple[str, ...] = ()) -> dict[str, Any]:
    """Row count, field presence and a content hash of one JSONL split.

    Missing files are reported, not raised, so a dry run can be printed before
    the data exists. `data_hash` is the sha256 of the file bytes, which is what
    the registry records as lineage.
    """
    p = Path(path)
    stats: dict[str, Any] = {
        "path": str(path),
        "exists": p.is_file(),
        "rows": 0,
        "bytes": 0,
        "data_hash": None,
        "fields": [],
        "missing_required": [],
    }
    if not stats["exists"]:
        return stats
    rows = read_jsonl(p)
    fields: dict[str, int] = {}
    for row in rows:
        for k in row:
            fields[k] = fields.get(k, 0) + 1
    stats["rows"] = len(rows)
    stats["bytes"] = p.stat().st_size
    stats["data_hash"] = hashlib.sha256(p.read_bytes()).hexdigest()
    stats["fields"] = sorted(fields)
    stats["missing_required"] = [f for f in required if fields.get(f, 0) < len(rows)]
    return stats


def estimate_steps(rows: int, plan: TrainingPlan) -> int:
    """Optimizer steps for `rows` examples under the plan's batch settings.

    Global batch = per_device_batch_size * gradient_accumulation * num_workers.
    `max_steps > 0` overrides the epoch-derived count, as in the HF Trainer.
    """
    max_steps = int(plan.hyperparam("max_steps"))
    if max_steps > 0:
        return max_steps
    if rows <= 0:
        return 0
    global_batch = (
        int(plan.hyperparam("per_device_batch_size"))
        * int(plan.hyperparam("gradient_accumulation"))
        * plan.compute.num_workers
    )
    per_epoch = math.ceil(rows / global_batch)
    return math.ceil(per_epoch * float(plan.hyperparam("epochs")))


def base_manifest(
    plan: TrainingPlan, dry_run: bool, required_fields: tuple[str, ...]
) -> dict[str, Any]:
    """The manifest skeleton shared by every runner."""
    splits = {name: dataset_stats(path, required_fields) for name, path in plan.datasets.items()}
    warnings: list[str] = []
    for name, st in splits.items():
        if not st["exists"]:
            warnings.append(f"dataset '{name}' not found: {st['path']}")
        elif st["missing_required"]:
            warnings.append(
                f"dataset '{name}' rows lack required fields: {', '.join(st['missing_required'])}"
            )
    train_rows = int(splits["train"]["rows"])
    return {
        "kind": plan.kind,
        "dry_run": dry_run,
        "config_hash": plan.config_hash(),
        "data_hash": splits["train"]["data_hash"],
        "plan": plan.to_dict(),
        "datasets": splits,
        "estimated_steps": estimate_steps(train_rows, plan),
        "warnings": warnings,
    }


__all__ = [
    "BACKENDS",
    "DEFAULT_HYPERPARAMS",
    "KINDS",
    "TASK_TYPES",
    "ComputeConfig",
    "LoraConfig",
    "PlanError",
    "TrainingPlan",
    "base_manifest",
    "dataset_stats",
    "estimate_steps",
    "read_jsonl",
    "write_jsonl",
]
