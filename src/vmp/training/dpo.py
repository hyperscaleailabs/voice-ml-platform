"""Direct Preference Optimization with TRL `DPOTrainer` on `PreferencePair` JSONL.

Rows are `{"prompt": ..., "chosen": ..., "rejected": ...}` (plus optional
`source`, `meta`), the serialised form of `vmp.types.PreferencePair`.

Reference model with PEFT: the policy is the base model plus a LoRA adapter.
`DPOTrainer` is given `ref_model=None`; it then computes reference log-probs by
running the same model with the adapter disabled (`peft` `disable_adapter()`),
so no second copy of the base weights is loaded. This is the documented TRL
behaviour when `peft_config` is passed, and the reason DPO on an adapter fits
where two full models would not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vmp.training.plan import TrainingPlan, base_manifest, read_jsonl
from vmp.training.sft import SPOKEN_SYSTEM_PROMPT, _filter_kwargs, is_spoken_style
from vmp.types import PreferencePair

PAIR_FIELDS = ("prompt", "chosen", "rejected")
REFERENCE_MODEL_NOTE = (
    "ref_model=None: DPOTrainer evaluates the reference log-probs with the LoRA "
    "adapter disabled on the policy model, so only one copy of the base weights is loaded."
)


def load_preference_pairs(path: str | Path) -> list[PreferencePair]:
    """Read pairs, rejecting rows with missing or empty fields."""
    pairs: list[PreferencePair] = []
    for i, row in enumerate(read_jsonl(path), start=1):
        missing = [f for f in PAIR_FIELDS if not isinstance(row.get(f), str) or not row[f]]
        if missing:
            raise ValueError(f"{path}: row {i} missing non-empty {', '.join(missing)}")
        if row["chosen"] == row["rejected"]:
            raise ValueError(f"{path}: row {i} has identical chosen and rejected")
        pairs.append(PreferencePair.from_dict(row))
    return pairs


def preference_stats(pairs: list[PreferencePair]) -> dict[str, Any]:
    """Length statistics that show what the pairs teach before any training."""
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    chosen_words = [len(p.chosen.split()) for p in pairs]
    rejected_words = [len(p.rejected.split()) for p in pairs]
    chosen_shorter = sum(c < r for c, r in zip(chosen_words, rejected_words, strict=True))
    chosen_spoken = sum(is_spoken_style(p.chosen) for p in pairs)
    rejected_spoken = sum(is_spoken_style(p.rejected) for p in pairs)
    sources: dict[str, int] = {}
    for p in pairs:
        sources[p.source] = sources.get(p.source, 0) + 1
    return {
        "n": n,
        "mean_chosen_words": sum(chosen_words) / n,
        "mean_rejected_words": sum(rejected_words) / n,
        "chosen_shorter_fraction": chosen_shorter / n,
        "chosen_spoken_fraction": chosen_spoken / n,
        "rejected_spoken_fraction": rejected_spoken / n,
        "sources": sources,
    }


def rule_preference_pairs(
    rows: list[dict[str, Any]], source: str = "rule:spoken-style"
) -> list[PreferencePair]:
    """Derive pairs from `{"prompt": ..., "candidates": [...]}` rows with a fixed rule.

    For each prompt, every spoken-style candidate (short, no markdown) is
    preferred over every candidate that is not. Prompts with no contrast yield
    nothing, so the output size is reported rather than assumed.
    """
    pairs: list[PreferencePair] = []
    for row in rows:
        prompt = str(row["prompt"])
        candidates = [str(c) for c in row.get("candidates", [])]
        good = [c for c in candidates if is_spoken_style(c)]
        bad = [c for c in candidates if not is_spoken_style(c)]
        for g in good:
            for b in bad:
                pairs.append(
                    PreferencePair(
                        prompt=prompt,
                        chosen=g,
                        rejected=b,
                        source=source,
                        meta={"rule": "spoken-style over markdown/long"},
                    )
                )
    return pairs


def run_dpo(plan: TrainingPlan, dry_run: bool = True) -> dict[str, Any]:
    """Train a LoRA adapter with TRL `DPOTrainer`, or return the plan as a manifest."""
    if plan.kind != "dpo":
        raise ValueError(f"run_dpo expects kind 'dpo', got {plan.kind!r}")
    manifest = base_manifest(plan, dry_run, required_fields=PAIR_FIELDS)
    train_path = Path(plan.datasets["train"])
    pair_stats: dict[str, Any] = {"n": 0}
    if manifest["datasets"]["train"]["exists"] and not manifest["datasets"]["train"][
        "missing_required"
    ]:
        pair_stats = preference_stats(load_preference_pairs(train_path))
    manifest.update(
        {
            "trainer": "trl.DPOTrainer",
            "peft": plan.lora.to_peft_kwargs(),
            "beta": float(plan.hyperparam("beta")),
            "reference_model": REFERENCE_MODEL_NOTE,
            "pairs": pair_stats,
        }
    )
    if dry_run:
        return manifest

    import torch
    from datasets import load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hp = plan.effective_hyperparams()
    tokenizer = AutoTokenizer.from_pretrained(plan.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(plan.base_model, torch_dtype=dtype)

    def to_conversational(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": [
                {"role": "system", "content": SPOKEN_SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }

    raw = load_dataset("json", data_files=dict(plan.datasets))
    data = raw.map(to_conversational, remove_columns=raw["train"].column_names)

    args = DPOConfig(
        **_filter_kwargs(
            DPOConfig,
            {
                "output_dir": plan.output_dir,
                "num_train_epochs": float(hp["epochs"]),
                "max_steps": int(hp["max_steps"]) or -1,
                "learning_rate": float(hp["learning_rate"]),
                "per_device_train_batch_size": int(hp["per_device_batch_size"]),
                "gradient_accumulation_steps": int(hp["gradient_accumulation"]),
                "max_length": int(hp["max_seq_length"]),
                "max_prompt_length": int(hp["max_prompt_length"]),
                "beta": float(hp["beta"]),
                "warmup_ratio": float(hp["warmup_ratio"]),
                "logging_steps": int(hp["logging_steps"]),
                "seed": plan.seed,
                "bf16": dtype == torch.bfloat16,
                "report_to": "none",
                "save_strategy": "epoch",
            },
        )
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # adapter-disabled policy acts as the reference
        args=args,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
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
    "PAIR_FIELDS",
    "REFERENCE_MODEL_NOTE",
    "load_preference_pairs",
    "preference_stats",
    "rule_preference_pairs",
    "run_dpo",
]
