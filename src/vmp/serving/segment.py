"""Sentence segmenter over a token stream.

A sentence is emitted only when it is provably complete: terminal punctuation
(`.`, `!`, `?`, `…`), optionally followed by closing quotes or brackets, and
then whitespace. A boundary at the very end of the buffer is not emitted until
`flush()`, because the next token could continue it (`3.` -> `3.5`).
Abbreviations (`Dr.`, `e.g.`, single-letter initials) and decimals are not
boundaries. An ellipsis is a boundary only when the next word starts a new
sentence (uppercase, digit, or opening quote). Nothing is ever truncated: text
that is not a complete sentence stays buffered until `flush()`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

TERMINALS = ".!?\u2026"
CLOSERS = "\"'\u201d\u2019)]\u00bb"
OPENERS = "\"'\u201c\u2018([\u00ab"

ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e",
        "inc", "ltd", "co", "corp", "no", "fig", "approx", "dept", "est", "gen", "gov",
        "sgt", "capt", "col", "lt", "hon", "rev", "ph.d", "u.s", "u.k", "a.m", "p.m",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun", "ave", "blvd", "rd", "min", "max",
    }
)  # fmt: skip


class Segmenter:
    """Stateful segmenter. `feed(token)` returns any sentences completed by the token."""

    def __init__(self, abbreviations: Iterable[str] = ABBREVIATIONS) -> None:
        self.abbreviations = frozenset(a.lower().rstrip(".") for a in abbreviations)
        self._buf = ""
        self.emitted = 0

    @property
    def pending(self) -> str:
        """Text buffered but not yet emitted."""
        return self._buf

    def feed(self, token: str) -> list[str]:
        if not token:
            return []
        self._buf += token
        out: list[str] = []
        while True:
            end = self._boundary(self._buf, final=False)
            if end is None:
                break
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:].lstrip()
            if sentence:
                out.append(sentence)
                self.emitted += 1
        return out

    def flush(self) -> list[str]:
        """Emit everything left, splitting at boundaries that end the buffer."""
        out: list[str] = []
        while self._buf.strip():
            end = self._boundary(self._buf, final=True)
            if end is None:
                end = len(self._buf)
            sentence = self._buf[:end].strip()
            self._buf = self._buf[end:].lstrip()
            if sentence:
                out.append(sentence)
                self.emitted += 1
        self._buf = ""
        return out

    # ----------------------------------------------------------------- internals

    def _is_abbreviation(self, buf: str, i: int) -> bool:
        """`buf[i]` is a period. True when the word before it should not end a sentence."""
        k = i
        while k > 0 and not buf[k - 1].isspace():
            k -= 1
        word = buf[k:i].lstrip(OPENERS)
        if not word:
            return False
        low = word.lower()
        if low in self.abbreviations:
            return True
        # Single-letter initial: "J. K. Rowling", "E. coli".
        return len(word) == 1 and word.isalpha()

    @staticmethod
    def _is_list_number(buf: str, i: int, j: int) -> bool:
        """`1. apples`: a bare number followed by a lowercase word is an enumerator."""
        k = i
        while k > 0 and buf[k - 1].isdigit():
            k -= 1
        if k == i or (k > 0 and not buf[k - 1].isspace()):
            return False
        m = j
        while m < len(buf) and buf[m].isspace():
            m += 1
        return m < len(buf) and buf[m].islower()

    def _boundary(self, buf: str, final: bool) -> int | None:
        """Index just past the first provable sentence end, or None."""
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if ch not in TERMINALS:
                i += 1
                continue
            # Extend over a run of terminals (?!, ..., ...) then closing marks.
            j = i
            while j < n and buf[j] in TERMINALS:
                j += 1
            run = buf[i:j]
            while j < n and buf[j] in CLOSERS:
                j += 1
            if j >= n:
                # Ends the buffer: only provable at flush time.
                return j if final else None
            if not buf[j].isspace():
                # "3.5", "a.b.c", "U.S.A": no whitespace after the run.
                i = j
                continue
            if run == "." and self._is_abbreviation(buf, i):
                i = j
                continue
            if run == "." and self._is_list_number(buf, i, j):
                i = j
                continue
            if "..." in run or "\u2026" in run:
                # Ellipsis: needs the next word to start a sentence.
                k = j
                while k < n and buf[k].isspace():
                    k += 1
                if k >= n:
                    return j if final else None
                nxt = buf[k]
                if not (nxt.isupper() or nxt.isdigit() or nxt in OPENERS):
                    i = j
                    continue
            return j
        return None


def segment(tokens: Iterable[str]) -> Iterator[str]:
    """Pure function form: yields complete sentences from a token stream, then flushes."""
    seg = Segmenter()
    for tok in tokens:
        yield from seg.feed(tok)
    yield from seg.flush()


__all__ = ["ABBREVIATIONS", "Segmenter", "segment"]
