"""LoRA fine-tuning of Whisper attention projections for one speaker's accent.

Dataset rows are `{"audio_path": ..., "text": ...}` (optional `duration_s`,
`speaker`). The adapter targets `q_proj` and `v_proj` in every attention block
and is trained with the HF `Seq2SeqTrainer`, the same recipe as the alpha-core
accent-adaptation notebook (alpha-core, cycle 5, 2026-09-12,
`notebook_whisper_accent_lora.ipynb`). The evaluation helpers here run on lists
of strings without `torch`, so results from any STT backend can be scored.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

from vmp.training.plan import TrainingPlan, base_manifest, read_jsonl
from vmp.types import EvalResult

AUDIO_FIELDS = ("audio_path", "text")
WHISPER_TARGET_MODULES = ("q_proj", "v_proj")

_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lower-case, drop punctuation, collapse whitespace. Applied to both sides."""
    return _WS.sub(" ", text.lower().translate(_PUNCT)).strip()


def exact_match(references: list[str], hypotheses: list[str]) -> EvalResult:
    """Fraction of utterances transcribed exactly (after normalisation)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    pairs = zip(references, hypotheses, strict=True)
    correct = sum(normalize_text(r) == normalize_text(h) for r, h in pairs)
    n = len(references)
    return EvalResult(
        name="exact_match",
        metrics={"exact_match": correct / n if n else 0.0, "correct": float(correct)},
        n=n,
        details={"correct": correct, "total": n},
    )


def word_edits(reference: str, hypothesis: str) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) between two word sequences.

    Standard Levenshtein alignment on normalised words, pure Python.
    """
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    # dp[i][j] = (cost, subs, dels, ins) for ref[:i] vs hyp[:j]
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub = prev[j - 1]
            dele = prev[j]
            ins = cur[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
            cur.append(best)
        prev = cur
    _, subs, dels, ins = prev[len(hyp)]
    return subs, dels, ins


def word_error_rate(references: list[str], hypotheses: list[str]) -> EvalResult:
    """Corpus WER = (S + D + I) / reference words, plus per-word accuracy."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    subs = dels = ins = ref_words = 0
    for r, h in zip(references, hypotheses, strict=True):
        s, d, i = word_edits(r, h)
        subs, dels, ins = subs + s, dels + d, ins + i
        ref_words += len(normalize_text(r).split())
    errors = subs + dels + ins
    wer = errors / ref_words if ref_words else 0.0
    return EvalResult(
        name="wer",
        metrics={
            "wer": wer,
            "word_accuracy": max(0.0, 1.0 - wer),
            "substitutions": float(subs),
            "deletions": float(dels),
            "insertions": float(ins),
            "reference_words": float(ref_words),
        },
        n=len(references),
    )


def load_audio_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for i, row in enumerate(rows, start=1):
        missing = [f for f in AUDIO_FIELDS if not isinstance(row.get(f), str) or not row[f]]
        if missing:
            raise ValueError(f"{path}: row {i} missing non-empty {', '.join(missing)}")
    return rows


def audio_manifest_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(r["duration_s"]) for r in rows if r.get("duration_s") is not None]
    missing_audio = sum(not Path(r["audio_path"]).is_file() for r in rows)
    speakers = sorted({str(r["speaker"]) for r in rows if r.get("speaker")})
    return {
        "n": len(rows),
        "total_duration_s": sum(durations) if durations else None,
        "rows_with_duration": len(durations),
        "missing_audio_files": missing_audio,
        "speakers": speakers,
        "mean_words": (sum(len(r["text"].split()) for r in rows) / len(rows)) if rows else 0.0,
    }


def run_whisper_lora(plan: TrainingPlan, dry_run: bool = True) -> dict[str, Any]:
    """LoRA on Whisper `q_proj`/`v_proj` with `Seq2SeqTrainer`, or the manifest."""
    if plan.kind != "whisper-lora":
        raise ValueError(f"run_whisper_lora expects kind 'whisper-lora', got {plan.kind!r}")
    manifest = base_manifest(plan, dry_run, required_fields=AUDIO_FIELDS)
    train = manifest["datasets"]["train"]
    audio_stats: dict[str, Any] = {"n": 0}
    if train["exists"] and not train["missing_required"]:
        audio_stats = audio_manifest_stats(load_audio_manifest(plan.datasets["train"]))
        if audio_stats["missing_audio_files"]:
            manifest["warnings"].append(
                f"{audio_stats['missing_audio_files']} audio files referenced by the "
                "train manifest do not exist"
            )
    extra = set(plan.lora.target_modules) - set(WHISPER_TARGET_MODULES)
    if extra:
        manifest["warnings"].append(
            f"target_modules {sorted(extra)} are outside the attention projections "
            f"used by the reference recipe {list(WHISPER_TARGET_MODULES)}"
        )
    manifest.update(
        {
            "trainer": "transformers.Seq2SeqTrainer",
            "model_class": "transformers.WhisperForConditionalGeneration",
            "peft": plan.lora.to_peft_kwargs(),
            "audio": audio_stats,
            "language": plan.hyperparam("language"),
            "task": plan.hyperparam("task"),
            "eval": ["exact_match", "wer"],
        }
    )
    if dry_run:
        return manifest

    import torch
    from datasets import Audio, load_dataset
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    hp = plan.effective_hyperparams()
    processor = WhisperProcessor.from_pretrained(
        plan.base_model, language=hp["language"], task=hp["task"]
    )
    model = WhisperForConditionalGeneration.from_pretrained(plan.base_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, PeftLoraConfig(**plan.lora.to_peft_kwargs()))

    sampling_rate = processor.feature_extractor.sampling_rate
    raw = load_dataset("json", data_files=dict(plan.datasets))
    raw = raw.rename_column("audio_path", "audio").cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        audio = row["audio"]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        labels = processor.tokenizer(row["text"]).input_ids
        return {"input_features": features, "labels": labels}

    data = raw.map(prepare, remove_columns=raw["train"].column_names)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        feats = processor.feature_extractor.pad(
            [{"input_features": b["input_features"]} for b in batch], return_tensors="pt"
        )
        labels = processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch], return_tensors="pt"
        )
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (label_ids[:, 0] == model.config.decoder_start_token_id).all().cpu().item():
            label_ids = label_ids[:, 1:]
        feats["labels"] = label_ids
        return feats

    args = Seq2SeqTrainingArguments(
        output_dir=plan.output_dir,
        num_train_epochs=float(hp["epochs"]),
        max_steps=int(hp["max_steps"]) or -1,
        learning_rate=float(hp["learning_rate"]),
        per_device_train_batch_size=int(hp["per_device_batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation"]),
        warmup_ratio=float(hp["warmup_ratio"]),
        logging_steps=int(hp["logging_steps"]),
        seed=plan.seed,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        save_strategy="epoch",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
        data_collator=collate,
        processing_class=processor.feature_extractor,
    )
    result = trainer.train()
    model.save_pretrained(plan.output_dir)
    processor.save_pretrained(plan.output_dir)
    manifest["result"] = {
        "global_step": int(result.global_step),
        "train_loss": float(result.training_loss),
        "adapter_path": plan.output_dir,
    }
    return manifest


__all__ = [
    "AUDIO_FIELDS",
    "WHISPER_TARGET_MODULES",
    "audio_manifest_stats",
    "exact_match",
    "load_audio_manifest",
    "normalize_text",
    "run_whisper_lora",
    "word_edits",
    "word_error_rate",
]
