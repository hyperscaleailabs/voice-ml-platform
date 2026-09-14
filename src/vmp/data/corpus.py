"""Turn corpora and preference pairs.

A `TurnCorpus` is a list of `Utterance`s built from JSONL transcripts or from
`Session` objects. Preference pairs for DPO come from two sources:

1. scored candidates: for one prompt, several replies with a numeric score; the
   best becomes `chosen`, the worst `rejected` (optionally every ordered pair).
2. spoken-style rules: a voice agent must not read out markdown, bullet lists or
   long paragraphs. A reply that violates these rules is `rejected`, a short
   plain reply to the same prompt is `chosen`. No model is needed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vmp.data.io import read_jsonl, read_utterances, sha256_records, write_utterances
from vmp.types import PreferencePair, Session, Utterance

# Spoken-style limits. A TTS engine reads every character, so anything a listener
# cannot hear as language is a defect.
MAX_SPOKEN_WORDS = 60
MAX_SPOKEN_SENTENCES = 4

_MARKDOWN = re.compile(r"(\*\*|__|`|^#{1,6}\s|\[[^\]]+\]\([^)]+\))", re.MULTILINE)
_BULLET = re.compile(r"^\s*([-*+•]|\d+[.)])\s+", re.MULTILINE)
_URL = re.compile(r"https?://\S+")
_SENTENCE_END = re.compile(r"[.!?]+(\s|$)")


def spoken_style_violations(text: str) -> list[str]:
    """Return the rule names a reply breaks. Empty list means it is fine to speak."""
    reasons: list[str] = []
    if _MARKDOWN.search(text):
        reasons.append("markdown")
    if _BULLET.search(text):
        reasons.append("bullets")
    if _URL.search(text):
        reasons.append("url")
    words = len(text.split())
    if words > MAX_SPOKEN_WORDS:
        reasons.append("too_long")
    sentences = len(_SENTENCE_END.findall(text.strip()))
    if sentences > MAX_SPOKEN_SENTENCES:
        reasons.append("too_many_sentences")
    if "\n\n" in text.strip():
        reasons.append("paragraphs")
    return reasons


def is_spoken_style(text: str) -> bool:
    return not spoken_style_violations(text)


@dataclass
class TurnCorpus:
    """An ordered list of utterances with a stable content hash."""

    utterances: list[Utterance] = field(default_factory=list)
    name: str = "corpus"

    @classmethod
    def from_jsonl(cls, path: str | Path, name: str | None = None) -> TurnCorpus:
        p = Path(path)
        return cls(read_utterances(p), name=name or p.stem)

    @classmethod
    def from_sessions(cls, sessions: Iterable[Session], name: str = "sessions") -> TurnCorpus:
        """Flatten sessions into user utterances. `meta` carries session and turn."""
        out: list[Utterance] = []
        for s in sessions:
            for t in s.turns:
                meta = dict(t.user.meta)
                meta.setdefault("session", s.id)
                meta.setdefault("turn", t.turn)
                if t.assistant_text is not None:
                    meta.setdefault("assistant_text", t.assistant_text)
                out.append(
                    Utterance(
                        id=t.user.id,
                        text=t.user.text,
                        audio_path=t.user.audio_path,
                        speaker=t.user.speaker,
                        duration_s=t.user.duration_s,
                        meta=meta,
                    )
                )
        return cls(out, name=name)

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self):
        return iter(self.utterances)

    def filter(self, category: str) -> TurnCorpus:
        return TurnCorpus(
            [u for u in self.utterances if u.meta.get("category") == category],
            name=f"{self.name}/{category}",
        )

    def data_hash(self) -> str:
        return sha256_records(u.to_dict() for u in self.utterances)

    def to_jsonl(self, path: str | Path) -> int:
        return write_utterances(path, self.utterances)

    def to_hf_dataset(self):
        return to_hf_dataset([u.to_dict() for u in self.utterances])


def to_hf_dataset(records: Iterable[dict[str, Any]]):
    """Build a `datasets.Dataset` from dicts. Imports `datasets` here, not at module load."""
    try:
        import datasets  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "to_hf_dataset needs the 'datasets' package: pip install 'voice-ml-platform[train]'"
        ) from e
    return datasets.Dataset.from_list(list(records))


def pairs_from_scores(
    records: Iterable[dict[str, Any]],
    *,
    min_margin: float = 0.0,
    all_pairs: bool = False,
    source: str = "scores",
) -> list[PreferencePair]:
    """Derive pairs from `{"prompt": str, "candidates": [{"text": str, "score": float}]}`.

    Default: one pair per prompt, best vs worst. `all_pairs=True` emits every
    (higher, lower) combination whose score gap is at least `min_margin`.
    Candidates with equal scores never form a pair.
    """
    out: list[PreferencePair] = []
    for rec in records:
        prompt = rec.get("prompt")
        cands = rec.get("candidates") or []
        if not prompt or len(cands) < 2:
            continue
        ranked = sorted(cands, key=lambda c: float(c["score"]), reverse=True)
        combos = (
            [(a, b) for i, a in enumerate(ranked) for b in ranked[i + 1 :]]
            if all_pairs
            else [(ranked[0], ranked[-1])]
        )
        for hi, lo in combos:
            gap = float(hi["score"]) - float(lo["score"])
            if gap <= 0 or gap < min_margin:
                continue
            out.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=hi["text"],
                    rejected=lo["text"],
                    source=source,
                    meta={
                        "chosen_score": float(hi["score"]),
                        "rejected_score": float(lo["score"]),
                        "margin": gap,
                        **{k: v for k, v in rec.items() if k not in ("prompt", "candidates")},
                    },
                )
            )
    return out


def pairs_from_style(
    records: Iterable[dict[str, Any]],
    *,
    source: str = "spoken_style",
) -> list[PreferencePair]:
    """Derive pairs from `{"prompt": str, "replies": [str, ...]}` using spoken-style rules.

    For each prompt, every reply that passes the rules is paired with every reply
    that fails them. A prompt with no passing reply or no failing reply yields nothing.
    """
    out: list[PreferencePair] = []
    for rec in records:
        prompt = rec.get("prompt")
        replies = rec.get("replies") or []
        if not prompt or len(replies) < 2:
            continue
        good: list[str] = []
        bad: list[tuple[str, list[str]]] = []
        for r in replies:
            v = spoken_style_violations(r)
            if v:
                bad.append((r, v))
            else:
                good.append(r)
        for g in good:
            for b, reasons in bad:
                out.append(
                    PreferencePair(
                        prompt=prompt,
                        chosen=g,
                        rejected=b,
                        source=source,
                        meta={"rejected_reasons": reasons},
                    )
                )
    return out


@dataclass
class PreferencePairBuilder:
    """Combine both derivations over a mixed JSONL file.

    A record with `candidates` (scored) goes through `pairs_from_scores`; a record
    with `replies` goes through `pairs_from_style`. Records with both yield both.
    """

    min_margin: float = 0.0
    all_pairs: bool = False
    source: str | None = None

    def build(self, records: Iterable[dict[str, Any]]) -> list[PreferencePair]:
        scored: list[dict[str, Any]] = []
        styled: list[dict[str, Any]] = []
        for rec in records:
            if rec.get("candidates"):
                scored.append(rec)
            if rec.get("replies"):
                styled.append(rec)
        pairs = pairs_from_scores(
            scored,
            min_margin=self.min_margin,
            all_pairs=self.all_pairs,
            source=self.source or "scores",
        )
        pairs += pairs_from_style(styled, source=self.source or "spoken_style")
        return pairs

    def build_from_jsonl(self, path: str | Path) -> list[PreferencePair]:
        return self.build(read_jsonl(path))


__all__ = [
    "MAX_SPOKEN_SENTENCES",
    "MAX_SPOKEN_WORDS",
    "PreferencePairBuilder",
    "TurnCorpus",
    "is_spoken_style",
    "pairs_from_scores",
    "pairs_from_style",
    "spoken_style_violations",
    "to_hf_dataset",
]
