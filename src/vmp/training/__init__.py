"""Training: plans, LoRA/PEFT configs, TRL SFT/DPO wrappers, Whisper LoRA, Ray launcher.

Everything importable here is standard library only. `torch`, `transformers`,
`peft`, `trl`, `datasets` and `ray` are imported lazily inside the functions that
need them, and only when `dry_run=False`.
"""

from __future__ import annotations

from vmp.training.plan import (
    BACKENDS,
    KINDS,
    ComputeConfig,
    LoraConfig,
    PlanError,
    TrainingPlan,
    dataset_stats,
    estimate_steps,
    read_jsonl,
)

__all__ = [
    "BACKENDS",
    "KINDS",
    "ComputeConfig",
    "LoraConfig",
    "PlanError",
    "TrainingPlan",
    "dataset_stats",
    "estimate_steps",
    "read_jsonl",
]
