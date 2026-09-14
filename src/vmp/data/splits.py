"""Deterministic train/val/test splits keyed on an entity or session id.

The split is a function of `sha256(salt + key)`, not of position or of a random
generator, so adding rows never moves an existing session between splits and
every row of one session lands in the same split.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from typing import TypeVar

from vmp.types import Utterance

SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)

T = TypeVar("T")


def split_fraction(key: str, salt: str = "") -> float:
    """Map a key to a stable float in [0, 1)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_split(
    key: str,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    salt: str = "",
) -> str:
    """Return "train" | "val" | "test" for a key. Ratios must sum to 1 (within 1e-6)."""
    if abs(sum(ratios) - 1.0) > 1e-6 or any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative and sum to 1: {ratios}")
    x = split_fraction(key, salt)
    edge = 0.0
    for name, r in zip(SPLITS, ratios, strict=True):
        edge += r
        if x < edge:
            return name
    return SPLITS[-1]


def split_by_key(
    items: Iterable[T],
    key: Callable[[T], str],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    salt: str = "",
) -> dict[str, list[T]]:
    """Group items by `assign_split(key(item))`. Every split name is present, maybe empty."""
    out: dict[str, list[T]] = {name: [] for name in SPLITS}
    for item in items:
        out[assign_split(key(item), ratios, salt)].append(item)
    return out


def utterance_group_key(u: Utterance) -> str:
    """Session id if present, then speaker, then the utterance id."""
    for k in ("session", "session_id"):
        v = u.meta.get(k)
        if v:
            return str(v)
    if u.speaker:
        return u.speaker
    return u.id


def split_utterances(
    utterances: Iterable[Utterance],
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    salt: str = "",
    key: Callable[[Utterance], str] = utterance_group_key,
) -> dict[str, list[Utterance]]:
    return split_by_key(utterances, key, ratios, salt)


__all__ = [
    "DEFAULT_RATIOS",
    "SPLITS",
    "assign_split",
    "split_by_key",
    "split_fraction",
    "split_utterances",
    "utterance_group_key",
]
