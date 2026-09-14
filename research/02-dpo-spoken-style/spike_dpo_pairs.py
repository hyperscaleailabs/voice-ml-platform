"""Spike 02 — derive DPO preference pairs by rule and plan the run.

Runs with nothing but the standard library installed:

1. takes prompts from `vmp.data.synthetic.generate_golden_set`;
2. attaches three candidate replies to each — one spoken, one markdown, one
   multi-paragraph — and derives pairs with `vmp.data.corpus.pairs_from_style`,
   the same rule the package ships;
3. prints pair statistics with `vmp.training.dpo.preference_stats` and a couple
   of pairs in full;
4. writes the pairs as JSONL, loads them back with `load_preference_pairs`, and
   prints `run_dpo(plan, dry_run=True)` for `configs/train_dpo.toml`.

No model is loaded and no training happens. The point of the spike is that the
labelling function is a rule, so the data exists before any model does.

    python research/02-dpo-spoken-style/spike_dpo_pairs.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vmp.data.corpus import (
    MAX_SPOKEN_WORDS,
    PreferencePairBuilder,
    spoken_style_violations,
)
from vmp.data.synthetic import generate_golden_set
from vmp.training.dpo import load_preference_pairs, preference_stats, run_dpo
from vmp.training.plan import TrainingPlan, write_jsonl

CONFIG = ROOT / "configs" / "train_dpo.toml"
CATEGORIES = ("factual_short", "tool_intent", "instruction_format")
PER_CATEGORY = 6


SPOKEN_BY_CATEGORY = {
    "tool_intent": "Done. I've set that up for you, and I'll say when it's finished.",
    "instruction_format": "Yes. That's the short answer, and I've kept it to one line.",
}


def spoken_reply(category: str, answer: str | None) -> str:
    """One or two short sentences, nothing a TTS engine cannot read aloud."""
    if answer:
        return f"It's {answer}. Let me know if you want the longer explanation."
    return SPOKEN_BY_CATEGORY.get(category, "Done. Anything else?")


def markdown_reply(category: str, answer: str | None) -> str:
    """The same content, formatted for a screen. A TTS engine reads the syntax."""
    body = answer or "the requested action"
    return (
        "## Answer\n"
        f"- **Result:** {body}\n"
        "- **Confidence:** high\n"
        "- See `docs/index.md` for details, or https://example.invalid/reference\n"
    )


def paragraph_reply(category: str, answer: str | None) -> str:
    """Correct, and far too long to hear. Two paragraphs, well over the word limit."""
    body = answer or "the action you asked for"
    return (
        f"That is a good question, and the short version of the answer is {body}. "
        "Before getting to that, it is worth setting out a little of the background, "
        "because the answer depends on a few assumptions that are easy to miss and "
        "that people often disagree about in practice when they first encounter it.\n\n"
        "With that context in place, the reasoning runs as follows. There are several "
        "conventions in common use, they mostly agree, and where they differ the "
        "difference rarely matters for everyday purposes. If you need the precise "
        "figure for a specific standard, tell me which standard you have in mind and "
        "I will work through it with you step by step in as much detail as you like."
    )


def candidate_records() -> list[dict[str, object]]:
    """`{"prompt", "replies": [...]}` records — the input `pairs_from_style` expects."""
    records: list[dict[str, object]] = []
    for u in generate_golden_set(per_category=PER_CATEGORY, seed=7, categories=CATEGORIES):
        answer = u.meta.get("answer")
        records.append(
            {
                "prompt": u.text,
                "category": u.meta["category"],
                "replies": [
                    spoken_reply(u.meta["category"], answer),
                    markdown_reply(u.meta["category"], answer),
                    paragraph_reply(u.meta["category"], answer),
                ],
            }
        )
    return records


def rule_report(records: list[dict[str, object]]) -> dict[str, int]:
    """How often each spoken-style rule fires across every candidate reply."""
    counts: dict[str, int] = {}
    for rec in records:
        for reply in rec["replies"]:  # type: ignore[union-attr]
            for reason in spoken_style_violations(str(reply)) or ["clean"]:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        records = candidate_records()

        print("=" * 78)
        print("1. synthetic corpus")
        print("=" * 78)
        print(f"  prompts          {len(records)} "
              f"({PER_CATEGORY} per category from {list(CATEGORIES)})")
        print(f"  replies each     {len(records[0]['replies'])}  "
              "(spoken / markdown / multi-paragraph)")
        print(f"  spoken limit     {MAX_SPOKEN_WORDS} words, no markdown, no bullets, no URLs")
        print(f"  rule hits        {json.dumps(rule_report(records))}")

        pairs = PreferencePairBuilder(source="rule:spoken-style").build(records)

        print()
        print("=" * 78)
        print("2. pairs derived by rule")
        print("=" * 78)
        stats = preference_stats(pairs)
        for key in (
            "n",
            "mean_chosen_words",
            "mean_rejected_words",
            "chosen_shorter_fraction",
            "chosen_spoken_fraction",
            "rejected_spoken_fraction",
        ):
            value = stats[key]
            print(f"  {key:<26} {value:.3f}" if isinstance(value, float) else
                  f"  {key:<26} {value}")
        print(f"  {'sources':<26} {json.dumps(stats['sources'])}")
        reasons: dict[str, int] = {}
        for p in pairs:
            for r in p.meta.get("rejected_reasons", []):
                reasons[r] = reasons.get(r, 0) + 1
        print(f"  {'rejected_reasons':<26} {json.dumps(dict(sorted(reasons.items())))}")

        print()
        print("=" * 78)
        print("3. two pairs in full")
        print("=" * 78)
        for p in (pairs[0], pairs[1]):
            print(f"  prompt    {p.prompt}")
            print(f"  chosen    {p.chosen}")
            head = p.rejected.splitlines()[0]
            print(f"  rejected  {head[:88]}{' ...' if len(p.rejected) > 88 else ''}")
            print(f"  source    {p.source}   reasons {p.meta.get('rejected_reasons')}")
            print("  " + "-" * 74)

        pair_path = tmp / "preference_pairs.jsonl"
        write_jsonl(pair_path, [p.to_dict() for p in pairs])
        reloaded = load_preference_pairs(pair_path)
        print(f"  wrote and re-read {len(reloaded)} pairs from JSONL")

        print()
        print("=" * 78)
        print("4. dry-run manifest")
        print("=" * 78)
        plan = TrainingPlan.from_toml(CONFIG)
        plan.datasets = {"train": str(pair_path)}
        print(json.dumps(run_dpo(plan, dry_run=True), indent=2, sort_keys=True))

        print()
        print("  No SFT stage precedes this plan: the adapter is trained on the")
        print("  base model directly. Whether that is enough is the claim, and this")
        print("  repository has not run it — see the Outcome section of README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
