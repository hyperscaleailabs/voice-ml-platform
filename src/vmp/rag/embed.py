"""Embedders. The default is a deterministic feature-hashing embedder.

`HashingEmbedder` needs no model and no dependency: it tokenises, hashes unigrams
and bigrams into `dim` signed buckets and L2-normalises. It is the embedder used
by tests and demos. Model-backed embedders import their dependency lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Small closed-class list. Dropping these keeps off-topic cosine near zero, which
# is what makes a relevance floor usable with a bag-of-words embedder.
_STOPWORD_TEXT = """
    a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why will with you your do does did
    can could should would i we they he she them their our not no yes if then than
    so into over under about after before also may might must
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def tokenise(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def _hash64(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


def l2_normalise(vec: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return tuple(vec)
    return tuple(v / norm for v in vec)


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into fixed-size vectors. `dim` is the vector length."""

    dim: int

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]: ...


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams, L2-normalised. Pure Python.

    Deterministic across processes (blake2b, not the salted built-in `hash`).
    Empty text embeds to the zero vector.
    """

    def __init__(self, dim: int = 512, bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.bigrams = bigrams

    def features(self, text: str) -> list[str]:
        toks = tokenise(text)
        feats = list(toks)
        if self.bigrams:
            feats.extend(f"{a} {b}" for a, b in pairwise(toks))
        return feats

    def embed_one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dim
        for feat in self.features(text):
            h = _hash64(feat)
            sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
            vec[h % self.dim] += sign
        return l2_normalise(vec)

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """`sentence-transformers` model. Imported lazily on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int | None = None) -> None:
        self.model_name = model_name
        self._model: Any = None
        self.dim = dim or 0

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        model = self._load()
        arr = model.encode(texts, normalize_embeddings=True)
        return [tuple(float(x) for x in row) for row in arr]


class OllamaEmbedder:
    """POSTs to a local Ollama `/api/embeddings`. Uses `urllib`; no dependency.

    Not exercised by tests. `dim` is learned from the first response.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.dim = 0

    def embed_one(self, text: str) -> tuple[float, ...]:
        body = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vec = l2_normalise([float(x) for x in data["embedding"]])
        self.dim = len(vec)
        return vec

    def embed(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self.embed_one(t) for t in texts]


__all__ = [
    "STOPWORDS",
    "Embedder",
    "HashingEmbedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "l2_normalise",
    "tokenise",
]
