"""Spike 01 — plan a Whisper accent LoRA and exercise the exact-match scorer.

Runs with nothing but the standard library installed. It does three things:

1. builds the real `TrainingPlan` from `configs/train_whisper_lora.toml`;
2. writes a small clean-label corpus manifest (the label is the sentence the
   speaker was asked to read) and prints the planner's dry-run manifest, which
   validates the fields, hashes the data and estimates the optimizer steps;
3. scores a synthetic before/after transcript set with `exact_match` and
   `word_error_rate` from `vmp.training.whisper_lora` — the same helpers a real
   run would use.

The transcripts in `READ_SET` are written by hand to exercise the scorer. They
are not a measurement of any model, and the script says so in its output. The
measured accent-LoRA result this spike is derived from is cited in README.md.

    python research/01-whisper-accent-lora/spike_whisper_lora_plan.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vmp.training.plan import TrainingPlan, write_jsonl
from vmp.training.whisper_lora import (
    WHISPER_TARGET_MODULES,
    audio_manifest_stats,
    exact_match,
    load_audio_manifest,
    run_whisper_lora,
    word_error_rate,
)

CONFIG = ROOT / "configs" / "train_whisper_lora.toml"
SPEAKER = "operator-01"

# (sentence the speaker read, baseline transcript, post-adaptation transcript).
# Synthetic. The errors are the kind an accent produces — a vowel or a proper
# noun mis-heard, a digit run split — but no model produced them.
READ_SET: tuple[tuple[str, str, str], ...] = (
    ("Set a timer for fifteen minutes.", "Set a timer for fifty minutes.", "Set a timer for fifteen minutes."),
    ("Call Aoife at half past two.", "Call Ava at half past two.", "Call Aoife at half past two."),
    ("The order number is four seven two one.", "The order number is four seven to one.", "The order number is four seven two one."),
    ("Turn the kitchen lights off.", "Turn the kitchen lights off.", "Turn the kitchen lights off."),
    ("Add oat milk to my shopping list.", "Add oat milk to my shopping list.", "Add oat milk to my shopping list."),
    ("Read that back to me slowly.", "Read that back to me slowly.", "Read that back to me slowly."),
    ("Confirm the delivery address in Cork.", "Confirm the delivery address in Cark.", "Confirm the delivery address in Cork."),
    ("The meeting moved to Thursday.", "The meeting moved to Tuesday.", "The meeting moved to Thursday."),
    ("Play the next track please.", "Play the next trick please.", "Play the next track please."),
    ("What is the weather in Gdansk tomorrow?", "What is the weather in Gadansk tomorrow?", "What is the weather in Gadansk tomorrow?"),
    ("Remind me to ring Ciaran at nine.", "Remind me to ring Kieran at nine.", "Remind me to ring Ciaran at nine."),
    ("Cancel that and start again.", "Cancel that and start again.", "Cancel that and start again."),
    ("The reference code is eight three nine zero.", "The reference code is eight three nine oh.", "The reference code is eight three nine zero."),
    ("Louder, I cannot hear you.", "Louder, I can not hear you.", "Louder, I cannot hear you."),
    ("Say the second part once more.", "Say the second part once more.", "Say the second part once more."),
    ("My name is Fionnuala Ni Bhriain.", "My name is Finola Nee Vreen.", "My name is Fionnuala Nee Bhriain."),
)


def corpus_rows(tmp: Path) -> list[dict[str, object]]:
    """The clean-label manifest: one row per prompt card the speaker read aloud.

    `text` is the prompt, not a transcript. Nothing the model produced is ever
    written into this field, which is the whole point of a read corpus.
    """
    rows: list[dict[str, object]] = []
    for i, (sentence, _, _) in enumerate(READ_SET):
        rows.append(
            {
                "audio_path": str(tmp / "audio" / f"{SPEAKER}-{i:03d}.wav"),
                "text": sentence,
                "speaker": SPEAKER,
                "duration_s": round(1.2 + 0.32 * len(sentence.split()), 2),
            }
        )
    return rows


def print_scores(title: str, references: list[str], hypotheses: list[str]) -> None:
    em = exact_match(references, hypotheses)
    wer = word_error_rate(references, hypotheses)
    print(
        f"  {title:<24} exact_match {em.details['correct']:>2}/{em.details['total']}"
        f"  ({em.metrics['exact_match'] * 100:5.1f}%)"
        f"   wer {wer.metrics['wer'] * 100:5.2f}%"
        f"   S/D/I {int(wer.metrics['substitutions'])}/"
        f"{int(wer.metrics['deletions'])}/{int(wer.metrics['insertions'])}"
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        plan = TrainingPlan.from_toml(CONFIG)

        print("=" * 78)
        print("1. plan")
        print("=" * 78)
        print(f"  config        {CONFIG.relative_to(ROOT)}")
        print(f"  kind          {plan.kind}")
        print(f"  base_model    {plan.base_model}")
        print(f"  lora          r={plan.lora.r} alpha={plan.lora.alpha} "
              f"dropout={plan.lora.dropout} task_type={plan.lora.task_type}")
        print(f"  targets       {list(plan.lora.target_modules)} "
              f"(recipe: {list(WHISPER_TARGET_MODULES)})")
        print(f"  config_hash   {plan.config_hash()}")
        print(f"  hyperparams   {json.dumps(plan.effective_hyperparams(), sort_keys=True)}")

        rows = corpus_rows(tmp)
        manifest_path = tmp / "accent_train.jsonl"
        write_jsonl(manifest_path, rows)
        plan.datasets = {"train": str(manifest_path)}

        print()
        print("=" * 78)
        print("2. clean-label corpus")
        print("=" * 78)
        stats = audio_manifest_stats(load_audio_manifest(manifest_path))
        print(f"  rows            {stats['n']}")
        print(f"  speakers        {stats['speakers']}")
        print(f"  total audio     {stats['total_duration_s']:.1f} s "
              f"({stats['rows_with_duration']} rows carry a duration)")
        print(f"  mean words      {stats['mean_words']:.1f}")
        print(f"  missing audio   {stats['missing_audio_files']}  "
              "(this spike ships no .wav files; the planner is expected to say so)")
        print("  label source    the sentence on the prompt card, never a model transcript")

        print()
        print("=" * 78)
        print("3. dry-run manifest")
        print("=" * 78)
        dry = run_whisper_lora(plan, dry_run=True)
        print(json.dumps(dry, indent=2, sort_keys=True))

        print()
        print("=" * 78)
        print("4. exact match on a SYNTHETIC before/after transcript set")
        print("=" * 78)
        references = [r for r, _, _ in READ_SET]
        baseline = [b for _, b, _ in READ_SET]
        adapted = [a for _, _, a in READ_SET]
        print_scores("baseline (synthetic)", references, baseline)
        print_scores("adapted (synthetic)", references, adapted)
        print()
        print("  These two lines are the scorer working on hand-written strings.")
        print("  They are not a result. This repository has not run the experiment;")
        print("  see the Outcome section of README.md.")

        changed = [
            (ref, base)
            for ref, base, adap in zip(references, baseline, adapted, strict=True)
            if base != adap
        ]
        print(f"\n  sentences the synthetic adapter changed: {len(changed)}/{len(READ_SET)}")
        for ref, base in changed[:3]:
            print(f"    read     {ref}")
            print(f"    baseline {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
