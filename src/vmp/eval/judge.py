"""Judges score an answer to a prompt in 0..1.

`RubricJudge` checks rules for spoken style (no markdown, few sentences, no
URLs). `LLMJudge` asks any object with `complete(messages) -> str` for a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from vmp.eval.wer import normalise


class Judge(Protocol):
    def score(self, prompt: str, answer: str, reference: str | None = None) -> float: ...


class ExactMatchJudge:
    """1.0 when the normalised answer equals the normalised reference, else 0.0."""

    def score(self, prompt: str, answer: str, reference: str | None = None) -> float:
        if reference is None:
            return 0.0
        return 1.0 if normalise(answer) == normalise(reference) else 0.0


_MARKDOWN = re.compile(r"(^|\n)\s*(#{1,6}\s|[-*+]\s|\d+\.\s|>\s|```)|\*\*|__|`")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?]+(\s|$)")


def count_sentences(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    n = len(_SENTENCE_END.findall(stripped))
    return max(n, 1)


@dataclass
class RubricJudge:
    """Rule checks for spoken answers. Score is the fraction of rules that pass."""

    max_sentences: int = 3
    max_words: int | None = None
    forbid_markdown: bool = True
    forbid_urls: bool = True
    require_nonempty: bool = True
    forbidden_phrases: list[str] = field(default_factory=list)

    def check(self, answer: str) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        if self.require_nonempty:
            checks["nonempty"] = bool(answer.strip())
        checks["max_sentences"] = count_sentences(answer) <= self.max_sentences
        if self.max_words is not None:
            checks["max_words"] = len(answer.split()) <= self.max_words
        if self.forbid_markdown:
            checks["no_markdown"] = _MARKDOWN.search(answer) is None
        if self.forbid_urls:
            checks["no_urls"] = _URL.search(answer) is None
        for phrase in self.forbidden_phrases:
            checks[f"no_phrase:{phrase}"] = phrase.lower() not in answer.lower()
        return checks

    def score(self, prompt: str, answer: str, reference: str | None = None) -> float:
        checks = self.check(answer)
        return sum(checks.values()) / len(checks) if checks else 1.0


class LLMLike(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

DEFAULT_JUDGE_PROMPT = (
    "You grade answers for a voice assistant. Reply with one number between 0 and 1, "
    "where 1 means the answer is correct, spoken in plain sentences, and concise."
)


@dataclass
class LLMJudge:
    """Sends prompt/answer/reference to an LLM and parses the first number in 0..1.

    Any object with `complete(messages: list[dict]) -> str` works, including a
    stub. A reply without a parseable number scores 0.0.
    """

    llm: Any
    system_prompt: str = DEFAULT_JUDGE_PROMPT

    def build_messages(
        self, prompt: str, answer: str, reference: str | None
    ) -> list[dict[str, str]]:
        user = f"Question: {prompt}\nAnswer: {answer}"
        if reference:
            user += f"\nReference answer: {reference}"
        user += "\nScore (0 to 1):"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def parse_score(text: str) -> float:
        m = _NUMBER.search(text)
        if not m:
            return 0.0
        value = float(m.group(0))
        if value > 1.0 and value <= 10.0:
            value = value / 10.0
        elif value > 10.0:
            value = value / 100.0
        return min(max(value, 0.0), 1.0)

    def score(self, prompt: str, answer: str, reference: str | None = None) -> float:
        reply = self.llm.complete(self.build_messages(prompt, answer, reference))
        return self.parse_score(str(reply))


__all__ = [
    "DEFAULT_JUDGE_PROMPT",
    "ExactMatchJudge",
    "Judge",
    "LLMJudge",
    "LLMLike",
    "RubricJudge",
    "count_sentences",
]
