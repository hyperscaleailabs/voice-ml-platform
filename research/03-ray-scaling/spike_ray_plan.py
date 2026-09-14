"""Spike 03 — the same TrainingPlan, local and on Ray, printed side by side.

Runs with nothing but the standard library installed. It loads
`configs/train_sft.toml`, produces the local dry-run manifest with `run_sft`,
switches `compute` to Ray, produces the Ray manifest with
`RayTrainLauncher.submit`, prints the two side by side and diffs them key by
key. `ray` is never imported: the launcher builds the `TorchTrainer` call as a
plain dict and only imports Ray when `dry_run=False`.

The diff is the evidence. Everything it lists should be attributable to the
compute block; anything else in the diff would falsify the claim.

    python research/03-ray-scaling/spike_ray_plan.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vmp.training.plan import ComputeConfig, TrainingPlan, write_jsonl
from vmp.training.ray_jobs import RayTrainLauncher
from vmp.training.sft import run_sft

CONFIG = ROOT / "configs" / "train_sft.toml"
NUM_WORKERS = 4
ROWS = 64


def synthetic_rows(n: int) -> list[dict[str, str]]:
    """A corpus that exists only so the planner has real bytes to hash."""
    return [
        {
            "prompt": f"Question {i}: what does stage {i % 7} of the turn loop do?",
            "response": f"**Stage {i % 7}** handles step {i}.\n- see the docs\n",
        }
        for i in range(n)
    ]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """`{"a.b": value}` for nested dicts, so two manifests can be compared key by key."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def diff(left: dict[str, Any], right: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(key, local value, ray value) for every key that differs or is one-sided."""
    a, b = flatten(left), flatten(right)
    rows: list[tuple[str, Any, Any]] = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va != vb:
            rows.append((key, va, vb))
    return rows


def short(value: Any, width: int = 34) -> str:
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= width else text[: width - 3] + "..."


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "sft_train.jsonl"
        write_jsonl(train_path, synthetic_rows(ROWS))

        local_plan = TrainingPlan.from_toml(CONFIG)
        local_plan.datasets = {"train": str(train_path)}
        local = run_sft(local_plan, dry_run=True)

        ray_plan = TrainingPlan.from_toml(CONFIG)
        ray_plan.datasets = {"train": str(train_path)}
        ray_plan.compute = ComputeConfig(
            backend="ray", num_workers=NUM_WORKERS, gpu_per_worker=1.0
        )
        launcher = RayTrainLauncher(address="auto", storage_path="runs/ray")
        remote = launcher.submit(ray_plan, dry_run=True)

        print("=" * 78)
        print("1. one config, two compute blocks")
        print("=" * 78)
        print(f"  config     {CONFIG.relative_to(ROOT)}")
        print(f"  rows       {ROWS}")
        print(f"  local      {json.dumps(local_plan.compute.to_dict())}")
        print(f"  ray        {json.dumps(ray_plan.compute.to_dict())}")
        print("  the TOML file is read twice and edited nowhere; only compute is replaced")

        print()
        print("=" * 78)
        print("2. manifests side by side")
        print("=" * 78)
        fields = (
            ("kind", lambda m: m["kind"]),
            ("base_model", lambda m: m["plan"]["base_model"]),
            ("train rows", lambda m: m["datasets"]["train"]["rows"]),
            ("data_hash", lambda m: str(m["data_hash"])[:12]),
            ("config_hash", lambda m: m["config_hash"][:12]),
            ("estimated_steps", lambda m: m["estimated_steps"]),
            ("num_workers", lambda m: m["plan"]["compute"]["num_workers"]),
            ("submits as", lambda m: m.get("trainer") or m["ray"]["trainer"]),
            ("entrypoint", lambda m: m.get("ray", {}).get("train_loop_per_worker", "in-process")),
            ("warnings", lambda m: len(m["warnings"])),
        )
        print(f"  {'field':<18} {'local':<36} ray")
        print(f"  {'-' * 18} {'-' * 36} {'-' * 20}")
        for name, get in fields:
            print(f"  {name:<18} {short(get(local)):<36} {short(get(remote))}")

        print()
        print("=" * 78)
        print("3. diff")
        print("=" * 78)
        plan_rows = diff(local["plan"], remote["plan"])
        print("  a. the plan itself")
        for key, left, right in plan_rows:
            print(f"     {key:<28} {short(left, 16):<16} -> {short(right, 20)}")
        if not plan_rows:
            print("     (identical)")

        skip = ("plan", "ray", "datasets")
        top_rows = [
            (k, a, b)
            for k, a, b in diff(
                {k: v for k, v in local.items() if k not in skip},
                {k: v for k, v in remote.items() if k not in skip},
            )
        ]
        print("\n  b. the rest of the manifest")
        for key, left, right in top_rows:
            print(f"     {key:<28} {short(left, 16):<16} -> {short(right, 20)}")

        ray_only = sorted(flatten(remote["ray"]))
        print(f"\n  c. keys only the Ray manifest has: {len(ray_only)}, all under 'ray'")
        print("     of which " + str(sum(k.startswith("train_loop_config") for k in ray_only))
              + " are the plan itself, passed to the workers verbatim")

        print("\n  reading the diff:")
        print("   - the dataset and its data_hash are identical; nothing about the data moved")
        print("   - config_hash differs because compute is part of the plan and therefore part")
        print("     of the lineage: a 4-worker run is not claimed to be the same run as a local")
        print("     one, it is claimed to need no code change")
        print("   - estimated_steps falls because the global batch is per_device x accumulation")
        print("     x num_workers; same data, fewer optimizer steps, a different effective batch")
        print("   - trainer / peft / formatting appear on the local side only because those are")
        print("     added by run_sft; on Ray the same runner adds them on the worker, inside")
        print("     train_loop_per_worker")

        print()
        print("=" * 78)
        print("4. the Ray call, as data")
        print("=" * 78)
        print(json.dumps(remote["ray"], indent=2, sort_keys=True))
        print()
        print("  Nothing above imported ray. The launcher builds the TorchTrainer")
        print("  arguments as a dict and imports ray only when dry_run=False, so this")
        print("  spike cannot and does not demonstrate that a cluster accepts the job.")
        print("  See the Outcome section of README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
