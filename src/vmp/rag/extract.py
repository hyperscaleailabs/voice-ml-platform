"""Rule-based entity and relation extraction. No model, deterministic.

Entities are capitalised terms (`Gate A`, `Kokoro`) and code identifiers
(`snake_case`, `CamelCase`, `dotted.names`, anything in backticks). Relations are
co-occurrence within one chunk, plus a `mentions` edge from the chunk node to
each entity. The output feeds one-hop graph expansion in the hybrid retriever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from vmp.types import Chunk

REL_MENTIONS = "mentions"
REL_COOCCURS = "cooccurs"
LABEL_CHUNK = "Chunk"
LABEL_ENTITY = "Entity"

_BACKTICK_RE = re.compile(r"`([^`\n]{1,80})`")
_CAP_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ \t]+[A-Z][A-Za-z0-9]*)*\b")
_SNAKE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")
_CAMEL_RE = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")

# Sentence-initial words that are capitalised for grammar, not because they name
# something. Kept short on purpose; a longer list would start to hide entities.
_CAP_STOP_TEXT = """
    A An The This That These Those It Its If In On At To For From By With As And Or But
    Not No Yes We You They He She I Is Are Was Were Be Do Does Did Has Have Had Can
    Could Should Would Will May Might Must When Where What Which Who Why How Then So
    Each Every All Some Any Here There Note See Also Use Run Set Get New Only Such
"""
_CAP_STOP = frozenset(_CAP_STOP_TEXT.split())


def entity_key(name: str) -> str:
    return "ent:" + " ".join(name.lower().split())


def chunk_key(chunk_id: str) -> str:
    return "chunk:" + chunk_id


def extract_entities(text: str) -> list[str]:
    """Sorted, de-duplicated entity surface forms found in `text`."""
    found: set[str] = set()
    for m in _BACKTICK_RE.finditer(text):
        token = m.group(1).strip()
        if token and " " not in token and len(token) >= 2:
            found.add(token)
    for m in _CAP_RE.finditer(text):
        phrase = " ".join(m.group(0).split())
        words = phrase.split(" ")
        # Drop leading grammar words: "The Gate A" -> "Gate A".
        while words and words[0] in _CAP_STOP:
            words = words[1:]
        if not words:
            continue
        phrase = " ".join(words)
        if len(words) == 1 and (len(phrase) < 2 or phrase in _CAP_STOP):
            continue
        found.add(phrase)
    for rx in (_SNAKE_RE, _CAMEL_RE, _DOTTED_RE):
        for m in rx.finditer(text):
            token = m.group(0)
            if token.lower() in {"e.g", "i.e", "etc"} or (token.isupper() and len(token) < 2):
                continue
            found.add(token)
    # Merge case variants: keep the first sorted surface form per key.
    by_key: dict[str, str] = {}
    for name in sorted(found):
        by_key.setdefault(entity_key(name), name)
    return [by_key[k] for k in sorted(by_key)]


@dataclass
class Extraction:
    """Nodes and edges produced from one chunk."""

    chunk_id: str
    entities: list[str] = field(default_factory=list)
    nodes: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    edges: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)


def extract_chunk(chunk: Chunk, max_entities: int = 40) -> Extraction:
    """Entities in the chunk, `mentions` edges, and pairwise `cooccurs` edges."""
    entities = extract_entities(chunk.text)[:max_entities]
    ck = chunk_key(chunk.id)
    ex = Extraction(chunk_id=chunk.id, entities=entities)
    ex.nodes.append(
        (
            LABEL_CHUNK,
            ck,
            {
                "document_id": chunk.document_id,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            },
        )
    )
    keys = [entity_key(e) for e in entities]
    for name, key in zip(entities, keys, strict=True):
        ex.nodes.append((LABEL_ENTITY, key, {"name": name}))
        ex.edges.append((ck, key, REL_MENTIONS, {}))
    for a, b in combinations(sorted(keys), 2):
        ex.edges.append((a, b, REL_COOCCURS, {"chunk_id": chunk.id}))
    return ex


__all__ = [
    "LABEL_CHUNK",
    "LABEL_ENTITY",
    "REL_COOCCURS",
    "REL_MENTIONS",
    "Extraction",
    "chunk_key",
    "entity_key",
    "extract_chunk",
    "extract_entities",
]
