"""`vmp train sft|dpo|whisper-lora|plan --config <toml> [--dry-run]`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from vmp.cli import register
from vmp.training.plan import PlanError, TrainingPlan

INSTALL_HINT = "pip install 'voice-ml-platform[train]'  (add [ray] for compute.backend = 'ray')"


def _add_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="train_command", required=True)
    for name, help_text in (
        ("sft", "supervised fine-tuning with TRL SFTTrainer and a LoRA adapter"),
        ("dpo", "direct preference optimisation with TRL DPOTrainer"),
        ("whisper-lora", "LoRA on Whisper attention projections"),
        ("plan", "validate a plan, print it with its config hash"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, help="TOML training plan")
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="validate, print the manifest, import nothing heavy",
        )
        p.add_argument("--ray-address", default=None, help="Ray cluster address (ray backend)")


def _dispatch(plan: TrainingPlan, args: argparse.Namespace) -> dict[str, Any]:
    if args.train_command == "plan":
        return {"plan": plan.to_dict(), "config_hash": plan.config_hash()}
    expected = args.train_command
    if plan.kind != expected:
        raise PlanError(f"config kind is {plan.kind!r}, command expects {expected!r}")
    if plan.compute.backend == "ray":
        from vmp.training.ray_jobs import RayTrainLauncher

        return RayTrainLauncher(address=args.ray_address).submit(plan, dry_run=args.dry_run)
    if plan.kind == "sft":
        from vmp.training.sft import run_sft

        return run_sft(plan, dry_run=args.dry_run)
    if plan.kind == "dpo":
        from vmp.training.dpo import run_dpo

        return run_dpo(plan, dry_run=args.dry_run)
    from vmp.training.whisper_lora import run_whisper_lora

    return run_whisper_lora(plan, dry_run=args.dry_run)


def _run(args: argparse.Namespace) -> int:
    try:
        plan = TrainingPlan.from_toml(args.config)
        out = _dispatch(plan, args)
    except (PlanError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as e:
        print(f"error: missing dependency {e.name!r}; {INSTALL_HINT}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def register_cli() -> None:
    register("train", "train adapters (sft, dpo, whisper-lora) or inspect a plan", _add_args, _run)


__all__ = ["register_cli"]
