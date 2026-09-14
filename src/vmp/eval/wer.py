"""Word and character error rate with the same semantics as `jiwer`.

`wer = (S + D + I) / N` where N is the number of reference words after
normalisation. Alignment is Levenshtein on word sequences with unit costs.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _number_words(tokens: list[str]) -> list[str]:
    """Collapse runs of number words ("twenty one", "one hundred") into digits."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok not in _UNITS and tok not in _TENS:
            out.append(tok)
            i += 1
            continue
        total = 0
        current = 0
        consumed = 0
        while i < len(tokens):
            t = tokens[i]
            if t in _UNITS:
                current += _UNITS[t]
            elif t in _TENS:
                current += _TENS[t]
            elif t == "hundred" and consumed:
                current *= 100
            elif t == "thousand" and consumed:
                total += current * 1000
                current = 0
            elif t == "and" and consumed and i + 1 < len(tokens) and tokens[i + 1] in _UNITS:
                pass
            else:
                break
            consumed += 1
            i += 1
        out.append(str(total + current))
    return out


def normalise(text: str, *, numbers: bool = False) -> list[str]:
    """Lower-case, strip punctuation, split on whitespace. `numbers` maps words to digits."""
    lowered = text.lower().replace("-", " ")
    stripped = _PUNCT.sub("", lowered)
    tokens = [t for t in _WS.split(stripped) if t]
    return _number_words(tokens) if numbers else tokens


@dataclass(frozen=True)
class Alignment:
    """Edit counts between a reference and a hypothesis sequence."""

    hits: int
    substitutions: int
    deletions: int
    insertions: int
    n_ref: int
    n_hyp: int
    ops: tuple[tuple[str, str | None, str | None], ...] = ()

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.n_ref == 0:
            return 0.0 if self.n_hyp == 0 else float(self.n_hyp)
        return self.errors / self.n_ref

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "n_ref": self.n_ref,
            "n_hyp": self.n_hyp,
            "errors": self.errors,
            "rate": self.rate,
        }


def align(ref: Sequence[str], hyp: Sequence[str], *, keep_ops: bool = False) -> Alignment:
    """Levenshtein alignment with unit costs and a deterministic backtrace.

    Ties prefer substitution, then deletion, then insertion, which matches the
    counts `jiwer` reports for its default configuration.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row = d[i]
        prev = d[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            row[j] = min(prev[j - 1] + cost, prev[j] + 1, row[j - 1] + 1)
    hits = subs = dels = ins = 0
    ops: list[tuple[str, str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            hits += 1
            if keep_ops:
                ops.append(("equal", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            subs += 1
            if keep_ops:
                ops.append(("substitute", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dels += 1
            if keep_ops:
                ops.append(("delete", ref[i - 1], None))
            i -= 1
        else:
            ins += 1
            if keep_ops:
                ops.append(("insert", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return Alignment(hits, subs, dels, ins, n, m, tuple(ops) if keep_ops else ())


def align_words(reference: str, hypothesis: str, *, numbers: bool = False) -> Alignment:
    return align(
        normalise(reference, numbers=numbers), normalise(hypothesis, numbers=numbers), keep_ops=True
    )


def wer(reference: str, hypothesis: str, *, numbers: bool = False) -> float:
    """Word error rate for one pair. Empty reference and non-empty hypothesis -> n_hyp."""
    return align(
        normalise(reference, numbers=numbers), normalise(hypothesis, numbers=numbers)
    ).rate


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate on normalised text with spaces removed."""
    r = "".join(normalise(reference))
    h = "".join(normalise(hypothesis))
    return align(list(r), list(h)).rate


def exact_match(reference: str, hypothesis: str, *, numbers: bool = False) -> bool:
    return normalise(reference, numbers=numbers) == normalise(hypothesis, numbers=numbers)


def corpus_wer(
    references: Sequence[str], hypotheses: Sequence[str], *, numbers: bool = False
) -> dict[str, float]:
    """Corpus-level WER: total errors over total reference words (not a mean of rates)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses differ in length")
    hits = subs = dels = ins = n_ref = 0
    matches = 0
    cer_err = 0
    cer_n = 0
    for r, h in zip(references, hypotheses, strict=True):
        a = align(normalise(r, numbers=numbers), normalise(h, numbers=numbers))
        hits += a.hits
        subs += a.substitutions
        dels += a.deletions
        ins += a.insertions
        n_ref += a.n_ref
        matches += int(a.errors == 0 and a.n_ref == a.n_hyp)
        c = align(list("".join(normalise(r))), list("".join(normalise(h))))
        cer_err += c.errors
        cer_n += c.n_ref
    n = len(references)
    return {
        "wer": (subs + dels + ins) / n_ref if n_ref else 0.0,
        "cer": cer_err / cer_n if cer_n else 0.0,
        "exact_match": matches / n if n else 0.0,
        "hits": float(hits),
        "substitutions": float(subs),
        "deletions": float(dels),
        "insertions": float(ins),
        "n_ref_words": float(n_ref),
        "n": float(n),
    }


def jiwer_wer(reference: str, hypothesis: str) -> float:
    """Cross-check with `jiwer` (lazy import). Same normalisation as `wer`."""
    import jiwer  # lazy

    return float(jiwer.wer(" ".join(normalise(reference)), " ".join(normalise(hypothesis))))


__all__ = [
    "Alignment",
    "align",
    "align_words",
    "cer",
    "corpus_wer",
    "exact_match",
    "jiwer_wer",
    "normalise",
    "wer",
]
