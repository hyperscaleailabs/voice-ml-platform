"""Supervised fine-tuning with TRL `SFTTrainer` and a PEFT LoRA adapter.

Input rows are JSONL. Three shapes are accepted and normalised to chat messages:

    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", ...}]}
    {"prompt": ..., "response": ...}
    {"user": ..., "assistant": ...}

Assistant turns are rewritten for speech: markdown is stripped and the system
prompt asks for short spoken sentences, because the target of this platform is a
voice agent whose replies are read aloud by a TTS engine.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from vmp.training.plan import TrainingPlan, base_manifest

SPOKEN_SYSTEM_PROMPT = (
    "You are a voice assistant. Reply in one to three short spoken sentences. "
    "Do not use markdown, lists, headings, code blocks or URLs."
)

_MD_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_WS = re.compile(r"[ \t]+")


def strip_markdown(text: str) -> str:
    """Remove markdown syntax so the text reads naturally when spoken."""
    out = _MD_CODE_FENCE.sub(" ", text)
    out = _MD_INLINE_CODE.sub(r"\1", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_BULLET.sub("", out)
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_EMPHASIS.sub(r"\2", out)
    lines = [_WS.sub(" ", ln).strip() for ln in out.splitlines()]
    return " ".join(ln for ln in lines if ln).strip()


def is_spoken_style(text: str, max_words: int = 60) -> bool:
    """Heuristic used by the data builders: short and free of markdown markers."""
    if len(text.split()) > max_words:
        return False
    return strip_markdown(text) == _WS.sub(" ", text.replace("\n", " ")).strip()


def format_chat_example(
    row: dict[str, Any], system_prompt: str = SPOKEN_SYSTEM_PROMPT
) -> dict[str, Any]:
    """Normalise one JSONL row to `{"messages": [...]}` with a spoken-style target."""
    if "messages" in row:
        messages = [dict(m) for m in row["messages"]]
    elif "prompt" in row and "response" in row:
        messages = [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["response"])},
        ]
    elif "user" in row and "assistant" in row:
        messages = [
            {"role": "user", "content": str(row["user"])},
            {"role": "assistant", "content": str(row["assistant"])},
        ]
    else:
        raise ValueError(
            "row needs 'messages', or 'prompt'+'response', or 'user'+'assistant'; "
            f"got keys {sorted(row)}"
        )
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant"):
            raise ValueError(f"unknown chat role {m.get('role')!r}")
        if m["role"] == "assistant":
            m["content"] = strip_markdown(str(m["content"]))
    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    return {"messages": messages}


def _filter_kwargs(config_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the installed TRL/transformers version does not accept."""
    names = {f.name for f in dataclasses.fields(config_cls)}
    return {k: v for k, v in kwargs.items() if k in names}


def run_sft(plan: TrainingPlan, dry_run: bool = True) -> dict[str, Any]:
    """Train a LoRA adapter with TRL `SFTTrainer`, or return the plan as a manifest."""
    if plan.kind != "sft":
        raise ValueError(f"run_sft expects kind 'sft', got {plan.kind!r}")
    manifest = base_manifest(plan, dry_run, required_fields=())
    manifest.update(
        {
            "trainer": "trl.SFTTrainer",
            "peft": plan.lora.to_peft_kwargs(),
            "system_prompt": SPOKEN_SYSTEM_PROMPT,
            "formatting": "chat messages, assistant turns stripped of markdown",
        }
    )
    if dry_run:
        return manifest

    # Heavy imports only on a real run.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hp = plan.effective_hyperparams()
    tokenizer = AutoTokenizer.from_pretrained(plan.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(plan.base_model, torch_dtype=dtype)

    data_files = {split: path for split, path in plan.datasets.items()}
    raw = load_dataset("json", data_files=data_files)
    formatted = raw.map(format_chat_example, remove_columns=raw["train"].column_names)

    args = SFTConfig(
        **_filter_kwargs(
            SFTConfig,
            {
                "output_dir": plan.output_dir,
                "num_train_epochs": float(hp["epochs"]),
                "max_steps": int(hp["max_steps"]) or -1,
                "learning_rate": float(hp["learning_rate"]),
                "per_device_train_batch_size": int(hp["per_device_batch_size"]),
                "gradient_accumulation_steps": int(hp["gradient_accumulation"]),
                "max_length": int(hp["max_seq_length"]),
                "max_seq_length": int(hp["max_seq_length"]),
                "warmup_ratio": float(hp["warmup_ratio"]),
                "logging_steps": int(hp["logging_steps"]),
                "seed": plan.seed,
                "bf16": dtype == torch.bfloat16,
                "report_to": "none",
                "save_strategy": "epoch",
            },
        )
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=formatted["train"],
        eval_dataset=formatted.get("eval"),
        processing_class=tokenizer,
        peft_config=PeftLoraConfig(**plan.lora.to_peft_kwargs()),
    )
    result = trainer.train()
    trainer.save_model(plan.output_dir)
    tokenizer.save_pretrained(plan.output_dir)
    manifest["result"] = {
        "global_step": int(result.global_step),
        "train_loss": float(result.training_loss),
        "adapter_path": plan.output_dir,
    }
    return manifest


__all__ = [
    "SPOKEN_SYSTEM_PROMPT",
    "format_chat_example",
    "is_spoken_style",
    "run_sft",
    "strip_markdown",
]
