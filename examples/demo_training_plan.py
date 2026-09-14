"""Training plans end to end, with nothing but the standard library installed.

Loads the three TOML plans in `configs/`, prints their dry-run manifests for
local and Ray compute, then registers a candidate artifact in a temporary
`FileRegistry` and promotes it through staging to production.

    python examples/demo_training_plan.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vmp.registry.store import FileRegistry
from vmp.training.dpo import run_dpo
from vmp.training.plan import ComputeConfig, TrainingPlan, write_jsonl
from vmp.training.ray_jobs import RayTrainLauncher
from vmp.training.sft import run_sft
from vmp.training.whisper_lora import run_whisper_lora
from vmp.types import ModelArtifact

RUNNERS = {"sft": run_sft, "dpo": run_dpo, "whisper-lora": run_whisper_lora}


def synthetic_rows(kind: str) -> list[dict]:
    if kind == "sft":
        return [{"prompt": f"Question {i}?", "response": f"**Answer** {i}."} for i in range(12)]
    if kind == "dpo":
        return [
            {"prompt": f"Q{i}", "chosen": f"Short reply {i}.", "rejected": f"## Reply {i}\n- long"}
            for i in range(12)
        ]
    return [
        {"audio_path": f"clip_{i}.wav", "text": f"sentence {i}", "duration_s": 2.0}
        for i in range(12)
    ]


def summarise(manifest: dict) -> str:
    keys = ("kind", "config_hash", "estimated_steps", "trainer")
    parts = [f"{k}={manifest[k]}" for k in keys if k in manifest]
    parts.append(f"rows={manifest['datasets']['train']['rows']}")
    if manifest["warnings"]:
        parts.append(f"warnings={len(manifest['warnings'])}")
    return "  ".join(parts)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifests: dict[str, dict] = {}
        for name in ("train_sft", "train_dpo", "train_whisper_lora"):
            plan = TrainingPlan.from_toml(ROOT / "configs" / f"{name}.toml")
            train_path = tmp_path / f"{plan.kind}.jsonl"
            write_jsonl(train_path, synthetic_rows(plan.kind))
            plan.datasets = {"train": str(train_path)}

            local = RUNNERS[plan.kind](plan, dry_run=True)
            print(f"[local] {summarise(local)}")

            plan.compute = ComputeConfig(backend="ray", num_workers=4, gpu_per_worker=1)
            ray = RayTrainLauncher(address="auto").submit(plan, dry_run=True)
            print(f"[ray]   {summarise(ray)}  scaling={json.dumps(ray['ray']['scaling_config'])}")
            manifests[plan.kind] = local

        print("\nfull SFT manifest:")
        print(json.dumps(manifests["sft"], indent=2, sort_keys=True))

        registry = FileRegistry(tmp_path / "registry")
        artifact = ModelArtifact(
            name="spoken-sft",
            version="1",
            stage="candidate",
            base_model=manifests["sft"]["plan"]["base_model"],
            adapter_path=manifests["sft"]["plan"]["output_dir"],
            config_hash=manifests["sft"]["config_hash"],
            data_hash=manifests["sft"]["data_hash"],
            git_sha=None,
            metrics={},
        )
        stored = registry.register(artifact)
        print(f"\nregistered {stored.name}:{stored.version} as {stored.stage}")
        for stage in ("staging", "production"):
            promoted = registry.promote(stored.name, stored.version, stage)
            print(f"promoted   {promoted.name}:{promoted.version} -> {promoted.stage}")
        final = registry.get(stored.name, stored.version)
        print("lineage:", json.dumps(dataclasses.asdict(final), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
