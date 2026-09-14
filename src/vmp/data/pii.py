"""Regex scrub for personally identifiable strings in transcripts.

Covers what shows up in voice transcripts: e-mail addresses, phone numbers, and
card-like digit runs. It is a floor, not a guarantee; a review step is still
needed before a corpus leaves the machine that recorded it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from vmp.types import Utterance

# Order matters: cards before phones, because a 16-digit run also matches the
# looser phone pattern.
PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"), "[CARD]"),
    (
        "phone",
        re.compile(
            r"(?<![\w])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]?\d{3,4}[ .-]?\d{3,4}\b"
        ),
        "[PHONE]",
    ),
)


@dataclass(frozen=True)
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def scrub(text: str) -> ScrubResult:
    """Replace matches with placeholder tokens. Returns the new text and per-kind counts."""
    counts: dict[str, int] = {}
    out = text
    for name, pattern, token in PATTERNS:
        out, n = pattern.subn(token, out)
        if n:
            counts[name] = n
    return ScrubResult(out, counts)


def scrub_utterance(u: Utterance) -> Utterance:
    """Return a copy with scrubbed text; `meta["pii"]` records what was removed."""
    r = scrub(u.text)
    if not r.counts:
        return u
    meta = dict(u.meta)
    meta["pii"] = dict(r.counts)
    return Utterance(
        id=u.id,
        text=r.text,
        audio_path=u.audio_path,
        speaker=u.speaker,
        duration_s=u.duration_s,
        meta=meta,
    )


def scrub_records(
    records: Iterable[dict[str, Any]], fields: tuple[str, ...] = ("text",)
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Scrub the named string fields of every record. Returns records and total counts."""
    totals: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for rec in records:
        rec = dict(rec)
        for f in fields:
            if isinstance(rec.get(f), str):
                r = scrub(rec[f])
                rec[f] = r.text
                for k, v in r.counts.items():
                    totals[k] = totals.get(k, 0) + v
        out.append(rec)
    return out, totals


__all__ = ["PATTERNS", "ScrubResult", "scrub", "scrub_records", "scrub_utterance"]
