"""Merge a LoRA adapter into its base model, and read an adapter's manifest.

`merge_adapter` is the first step of the edge export pipeline (merged weights
are what ONNX / GGUF / MLX converters consume). `peft` and `transformers` are
imported only for a real merge; `adapter_manifest` is standard library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ADAPTER_CONFIG = "adapter_config.json"


def adapter_manifest(adapter_path: str | Path) -> dict[str, Any]:
    """Read `adapter_config.json` if present. Never raises for a missing adapter."""
    root = Path(adapter_path)
    cfg_path = root / ADAPTER_CONFIG
    manifest: dict[str, Any] = {
        "adapter_path": str(adapter_path),
        "exists": root.is_dir(),
        "has_config": cfg_path.is_file(),
        "peft_type": None,
        "base_model_name_or_path": None,
        "r": None,
        "lora_alpha": None,
        "target_modules": None,
        "weight_files": [],
    }
    if root.is_dir():
        manifest["weight_files"] = sorted(
            p.name for p in root.iterdir() if p.suffix in (".safetensors", ".bin")
        )
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("peft_type", "base_model_name_or_path", "r", "lora_alpha", "target_modules"):
            manifest[key] = cfg.get(key)
        manifest["config"] = cfg
    return manifest


def merge_adapter(
    base: str, adapter_path: str | Path, out: str | Path, dry_run: bool = True
) -> dict[str, Any]:
    """`peft` `merge_and_unload()` into `out`, or the plan for doing so."""
    info = adapter_manifest(adapter_path)
    manifest: dict[str, Any] = {
        "base_model": base,
        "adapter": info,
        "out": str(out),
        "dry_run": dry_run,
        "method": "peft.PeftModel.merge_and_unload",
        "warnings": [],
    }
    if info["base_model_name_or_path"] and info["base_model_name_or_path"] != base:
        manifest["warnings"].append(
            f"adapter was trained on {info['base_model_name_or_path']!r}, merging into {base!r}"
        )
    if not info["has_config"]:
        manifest["warnings"].append(f"no {ADAPTER_CONFIG} under {adapter_path}")
    if dry_run:
        return manifest
    if not info["has_config"]:
        raise FileNotFoundError(f"no {ADAPTER_CONFIG} under {adapter_path}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperForConditionalGeneration

    peft_type = info["peft_type"]
    task_type = (info.get("config") or {}).get("task_type")
    model_cls = (
        WhisperForConditionalGeneration if task_type == "SEQ_2_SEQ_LM" else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(base)
    merged = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload()
    Path(out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    try:
        AutoTokenizer.from_pretrained(str(adapter_path)).save_pretrained(str(out))
    except (OSError, ValueError):
        AutoTokenizer.from_pretrained(base).save_pretrained(str(out))
    manifest["result"] = {"merged_path": str(out), "peft_type": peft_type}
    return manifest


__all__ = ["ADAPTER_CONFIG", "adapter_manifest", "merge_adapter"]
