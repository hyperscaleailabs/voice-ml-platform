"""Export a base model plus adapter to an edge format: ONNX, GGUF or MLX.

`export(plan, dry_run=True)` validates the plan and returns a manifest whose
`steps` list describes each conversion step. A real export imports the
converter lazily (`optimum`/`onnx`, `llama_cpp`/`gguf`, `mlx_lm`) and records
what each step produced.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TARGETS = ("onnx", "gguf", "mlx")
QUANT_METHODS = ("int8", "int4", "q4_k_m", "none")

# Which quantization methods each target understands.
_TARGET_QUANT = {
    "onnx": ("int8", "none"),
    "gguf": ("q4_k_m", "int8", "none"),
    "mlx": ("int4", "int8", "none"),
}

# Optional dependency each target needs for a real export.
_TARGET_DEPS = {
    "onnx": ("optimum", "onnx"),
    "gguf": ("llama_cpp", "gguf"),
    "mlx": ("mlx_lm",),
}


@dataclass(frozen=True)
class ExportPlan:
    base_model: str
    target: str
    out_dir: str
    adapter_path: str | None = None
    quantization: dict[str, Any] = field(default_factory=lambda: {"method": "none"})
    merge_adapter: bool = True

    @property
    def method(self) -> str:
        return str(self.quantization.get("method", "none"))

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.base_model:
            problems.append("base_model is empty")
        if self.target not in TARGETS:
            problems.append(f"target {self.target!r} not in {list(TARGETS)}")
        if self.method not in QUANT_METHODS:
            problems.append(f"quantization.method {self.method!r} not in {list(QUANT_METHODS)}")
        elif self.target in _TARGET_QUANT and self.method not in _TARGET_QUANT[self.target]:
            problems.append(
                f"quantization {self.method!r} is not supported for target {self.target!r}"
            )
        if self.adapter_path and not Path(self.adapter_path).exists():
            problems.append(f"adapter_path does not exist: {self.adapter_path}")
        if not self.out_dir:
            problems.append("out_dir is empty")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportPlan:
        return cls(
            base_model=str(data["base_model"]),
            target=str(data["target"]),
            out_dir=str(data["out_dir"]),
            adapter_path=data.get("adapter_path"),
            quantization=dict(data.get("quantization", {"method": "none"})),
            merge_adapter=bool(data.get("merge_adapter", True)),
        )


def _steps_for(plan: ExportPlan) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"name": "validate", "detail": "check plan fields and adapter path"},
    ]
    if plan.adapter_path and plan.merge_adapter:
        steps.append(
            {
                "name": "merge_adapter",
                "detail": f"merge LoRA adapter {plan.adapter_path} into {plan.base_model}",
                "requires": ["peft", "transformers"],
            }
        )
    if plan.target == "onnx":
        steps.append(
            {
                "name": "convert_onnx",
                "detail": "optimum.exporters.onnx.main_export -> model.onnx",
                "requires": list(_TARGET_DEPS["onnx"]),
            }
        )
        if plan.method == "int8":
            steps.append(
                {
                    "name": "quantize_int8",
                    "detail": "onnxruntime.quantization.quantize_dynamic (QInt8 weights)",
                    "requires": ["onnxruntime"],
                }
            )
    elif plan.target == "gguf":
        steps.append(
            {
                "name": "convert_gguf",
                "detail": "llama.cpp convert_hf_to_gguf -> model-f16.gguf",
                "requires": list(_TARGET_DEPS["gguf"]),
            }
        )
        if plan.method != "none":
            steps.append(
                {
                    "name": f"quantize_{plan.method}",
                    "detail": f"llama.cpp quantize model-f16.gguf -> model-{plan.method}.gguf",
                    "requires": ["llama_cpp"],
                }
            )
    elif plan.target == "mlx":
        detail = "mlx_lm.convert(hf_path, mlx_path"
        if plan.method != "none":
            bits = 4 if plan.method == "int4" else 8
            detail += f", quantize=True, q_bits={bits}"
        steps.append(
            {"name": "convert_mlx", "detail": detail + ")", "requires": list(_TARGET_DEPS["mlx"])}
        )
    steps.append({"name": "write_manifest", "detail": "export_manifest.json in out_dir"})
    return steps


def _expected_files(plan: ExportPlan) -> list[str]:
    if plan.target == "onnx":
        return ["model.onnx", "config.json", "tokenizer.json"] + (
            ["model-int8.onnx"] if plan.method == "int8" else []
        )
    if plan.target == "gguf":
        return ["model-f16.gguf"] + ([f"model-{plan.method}.gguf"] if plan.method != "none" else [])
    return ["weights.safetensors", "config.json", "tokenizer.json"]


def export(plan: ExportPlan, *, dry_run: bool = True) -> dict[str, Any]:
    """Run or plan an export. Returns the manifest dict.

    Manifest keys: `plan`, `dry_run`, `steps` (each with `status`), `files`
    (expected or produced, relative to `out_dir`), `problems`, `created_at`.
    """
    problems = plan.validate()
    manifest: dict[str, Any] = {
        "plan": plan.to_dict(),
        "dry_run": dry_run,
        "target": plan.target,
        "quantization": dict(plan.quantization),
        "steps": [],
        "files": _expected_files(plan),
        "problems": problems,
        "created_at": time.time(),
    }
    steps = _steps_for(plan)
    if problems:
        for s in steps:
            s["status"] = "skipped"
        manifest["steps"] = steps
        return manifest
    if dry_run:
        for s in steps:
            s["status"] = "planned"
        manifest["steps"] = steps
        return manifest
    out = Path(plan.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in steps:
        try:
            _run_step(s["name"], plan, out)
            s["status"] = "done"
        except ImportError as e:
            s["status"] = "failed"
            s["error"] = f"missing dependency: {e}"
            manifest["problems"].append(f"{s['name']}: {s['error']}")
            break
    manifest["steps"] = steps
    manifest["files"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _run_step(name: str, plan: ExportPlan, out: Path) -> None:
    if name in ("validate", "write_manifest"):
        return
    if name == "merge_adapter":
        _merge_adapter(plan, out)
    elif name == "convert_onnx":
        _convert_onnx(plan, out)
    elif name == "quantize_int8":
        _quantize_onnx_int8(out)
    elif name == "convert_gguf":
        _convert_gguf(plan, out)
    elif name.startswith("quantize_") and plan.target == "gguf":
        _quantize_gguf(plan, out)
    elif name == "convert_mlx":
        _convert_mlx(plan, out)


def _merge_adapter(plan: ExportPlan, out: Path) -> None:
    from peft import PeftModel  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    base = AutoModelForCausalLM.from_pretrained(plan.base_model)
    merged = PeftModel.from_pretrained(base, plan.adapter_path).merge_and_unload()
    merged_dir = out / "merged"
    merged.save_pretrained(merged_dir)
    AutoTokenizer.from_pretrained(plan.base_model).save_pretrained(merged_dir)


def _source_dir(plan: ExportPlan, out: Path) -> str:
    merged = out / "merged"
    return str(merged) if merged.exists() else plan.base_model


def _convert_onnx(plan: ExportPlan, out: Path) -> None:
    from optimum.exporters.onnx import main_export  # lazy

    main_export(_source_dir(plan, out), output=str(out), task="text-generation")


def _quantize_onnx_int8(out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic  # lazy

    quantize_dynamic(
        str(out / "model.onnx"), str(out / "model-int8.onnx"), weight_type=QuantType.QInt8
    )


def _convert_gguf(plan: ExportPlan, out: Path) -> None:
    import subprocess
    import sys

    import gguf  # noqa: F401  # lazy; the converter script ships with llama.cpp

    script = "convert_hf_to_gguf.py"
    subprocess.run(
        [sys.executable, script, _source_dir(plan, out), "--outfile", str(out / "model-f16.gguf")],
        check=True,
    )


def _quantize_gguf(plan: ExportPlan, out: Path) -> None:
    import llama_cpp  # lazy

    src = str(out / "model-f16.gguf")
    dst = str(out / f"model-{plan.method}.gguf")
    ftype = {"q4_k_m": "Q4_K_M", "int8": "Q8_0"}[plan.method]
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, f"LLAMA_FTYPE_MOSTLY_{ftype}")
    llama_cpp.llama_model_quantize(src.encode(), dst.encode(), params)


def _convert_mlx(plan: ExportPlan, out: Path) -> None:
    from mlx_lm import convert  # lazy

    kwargs: dict[str, Any] = {}
    if plan.method != "none":
        kwargs = {"quantize": True, "q_bits": 4 if plan.method == "int4" else 8}
    convert(_source_dir(plan, out), mlx_path=str(out), **kwargs)


def plan_from_config(data: dict[str, Any]) -> ExportPlan:
    """Build an `ExportPlan` from the `[export]` table of `configs/edge.toml`."""
    section = data.get("export", data)
    return ExportPlan.from_dict(section)


__all__ = ["QUANT_METHODS", "TARGETS", "ExportPlan", "export", "plan_from_config"]
