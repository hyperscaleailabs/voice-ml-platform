"""Datasets for voice agents: turn corpora, preference pairs, synthetic golden sets.

Everything here is standard library. `to_hf_dataset` imports `datasets` lazily.
"""

from __future__ import annotations

from vmp.data.corpus import (
    PreferencePairBuilder,
    TurnCorpus,
    pairs_from_scores,
    pairs_from_style,
    spoken_style_violations,
    to_hf_dataset,
)
from vmp.data.io import (
    atomic_write_text,
    read_jsonl,
    read_utterances,
    sha256_file,
    sha256_record,
    sha256_records,
    write_jsonl,
    write_utterances,
)
from vmp.data.pii import ScrubResult, scrub, scrub_utterance
from vmp.data.splits import SPLITS, assign_split, split_by_key, split_utterances
from vmp.data.synthetic import CATEGORIES, generate_golden_set

__all__ = [
    "CATEGORIES",
    "SPLITS",
    "PreferencePairBuilder",
    "ScrubResult",
    "TurnCorpus",
    "assign_split",
    "atomic_write_text",
    "generate_golden_set",
    "pairs_from_scores",
    "pairs_from_style",
    "read_jsonl",
    "read_utterances",
    "scrub",
    "scrub_utterance",
    "sha256_file",
    "sha256_record",
    "sha256_records",
    "split_by_key",
    "split_utterances",
    "spoken_style_violations",
    "to_hf_dataset",
    "write_jsonl",
    "write_utterances",
]
