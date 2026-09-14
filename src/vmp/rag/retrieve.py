"""Hybrid retrieval: keyword prefilter, vector top-k, graph expansion, RRF, floor.

Why each step exists, with the observation that motivated it:

- Keyword prefilter. In alpha-core (cycle 5, 2026-09-12, `README.md` "Talking
  about this codebase") a question about "Gate A" was answered from the wrong
  chunk because cosine ranked the chunk containing the literal at position 5-10,
  outside the top-k. A rare literal term is strong evidence on its own, so it is
  matched exactly and its chunk enters the fusion regardless of cosine rank.
- Graph expansion. Entities in the top chunks are followed one hop to the other
  chunks that mention them (GraphRAG-style), so an answer split across two
  sections of a document is assembled rather than truncated.
- Reciprocal-rank fusion. Combines the three rankings without needing the score
  scales to be comparable.
- Relevance floor. alpha-core's cosine spread between rank 1 and rank 20 was
  0.494 vs 0.442 (same source), so a plain top-k always returns something and an
  off-topic question received "code echo": the nearest junk, read aloud. The floor
  (min cosine, or a keyword hit, and min fused score) returns nothing instead.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from vmp.rag.embed import STOPWORDS, Embedder, tokenise
from vmp.rag.extract import REL_MENTIONS, chunk_key, entity_key, extract_entities
from vmp.rag.graph import GraphStore
from vmp.rag.vector import VectorStore
from vmp.types import Chunk

PATH_KEYWORD = "keyword"
PATH_VECTOR = "vector"
PATH_GRAPH = "graph"
PATH_FUSED = "fused"


class KeywordIndex:
    """Inverted index over unigrams and bigrams with document frequencies.

    Stopwords are kept so that literals such as "gate a" survive as bigrams.
    """

    def __init__(self) -> None:
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._grams_by_chunk: dict[str, set[str]] = {}
        self._doc_of_chunk: dict[str, str] = {}

    @staticmethod
    def grams(text: str) -> set[str]:
        toks = tokenise(text, drop_stopwords=False)
        out = set(toks)
        out.update(f"{a} {b}" for a, b in pairwise(toks))
        return out

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.remove_chunk(c.id)
            grams = self.grams(c.text)
            self._grams_by_chunk[c.id] = grams
            self._doc_of_chunk[c.id] = c.document_id
            for g in grams:
                self._postings[g].add(c.id)

    def remove_chunk(self, chunk_id: str) -> None:
        for g in self._grams_by_chunk.pop(chunk_id, set()):
            posting = self._postings.get(g)
            if posting is not None:
                posting.discard(chunk_id)
                if not posting:
                    del self._postings[g]
        self._doc_of_chunk.pop(chunk_id, None)

    def delete_document(self, document_id: str) -> int:
        doomed = [cid for cid, d in self._doc_of_chunk.items() if d == document_id]
        for cid in doomed:
            self.remove_chunk(cid)
        return len(doomed)

    def count(self) -> int:
        return len(self._grams_by_chunk)

    def df(self, gram: str) -> int:
        return len(self._postings.get(gram, ()))

    @staticmethod
    def _informative(gram: str) -> bool:
        """A gram made only of stopwords is never a literal worth matching."""
        return any(t not in STOPWORDS for t in gram.split(" "))

    def rare_terms(self, query: str, max_df: int, max_df_fraction: float = 0.2) -> list[str]:
        """Query grams present in at most `max_df` chunks and `max_df_fraction` of them.

        The fraction keeps "rare" meaningful on a small corpus, where an absolute
        cut-off would admit most of the vocabulary.
        """
        limit = max(1, min(max_df, int(self.count() * max_df_fraction)))
        return sorted(
            g for g in self.grams(query) if self._informative(g) and 0 < self.df(g) <= limit
        )

    def search(
        self, query: str, max_df: int, k: int, max_df_fraction: float = 0.2
    ) -> list[tuple[str, float]]:
        """Chunks matching rare query grams, scored by summed idf. Bigrams count double."""
        n = max(self.count(), 1)
        scores: dict[str, float] = defaultdict(float)
        for g in self.rare_terms(query, max_df, max_df_fraction):
            idf = math.log(1.0 + n / self.df(g))
            weight = 2.0 if " " in g else 1.0
            for cid in self._postings[g]:
                scores[cid] += weight * idf
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """RRF: score(id) = sum over lists of weight / (k + rank), rank starting at 1.

    Weights default to 1.0. The keyword path is weighted above the vector path by
    default in `HybridRetriever`: an exact hit on a term rare enough to be a
    literal is a stronger signal than a cosine rank, which is the failure mode
    this retriever exists to cover (alpha-core, cycle 5, 2026-09-12, `README.md`
    "Talking about this codebase": rank 1 scored 0.494 against rank 20 at 0.442,
    and a rare literal drifted from rank 5 to rank 10 as unrelated files landed).
    """
    weights = weights or {}
    fused: dict[str, float] = defaultdict(float)
    for name, ids in rankings.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ids, start=1):
            fused[cid] += w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPH = re.compile(r"(\*{1,3}|`{1,3})")
_MD_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_WS = re.compile(r"[ \t]+")


def strip_markdown(text: str) -> str:
    """Plain text suitable for a spoken answer: no headings, emphasis, links."""
    t = _MD_HEADING.sub("", text)
    t = _MD_LINK.sub(r"\1", t)
    t = _MD_EMPH.sub("", t)
    t = _MD_BULLET.sub("", t)
    t = _WS.sub(" ", t)
    return "\n".join(line.strip() for line in t.splitlines() if line.strip())


@dataclass
class RetrievalResult:
    """Ranked chunks with fused scores, per-chunk path, and raw cosine per chunk."""

    query: str
    chunks: list[Chunk] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    cosines: dict[str, float] = field(default_factory=dict)
    rare_terms: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    floored: int = 0

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.chunks]

    def context_pack(self, max_chars: int = 2000) -> str:
        """Render the chunks as plain prose for a spoken answer. No markdown."""
        parts: list[str] = []
        used = 0
        for c in self.chunks:
            body = strip_markdown(c.text)
            head = f"From {c.document_id}, lines {c.start_line} to {c.end_line}."
            block = f"{head}\n{body}"
            if used + len(block) > max_chars:
                room = max_chars - used - len(head) - 1
                if room <= 0:
                    break
                cut = body[:room]
                if " " in cut:
                    cut = cut[: cut.rfind(" ")]
                block = f"{head}\n{cut.rstrip()}"
                parts.append(block)
                break
            parts.append(block)
            used += len(block) + 2
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "score": s,
                    "path": p,
                    "cosine": self.cosines.get(c.id),
                }
                for c, s, p in zip(self.chunks, self.scores, self.paths, strict=True)
            ],
            "rare_terms": list(self.rare_terms),
            "entities": list(self.entities),
            "floored": self.floored,
        }


class HybridRetriever:
    """Keyword + vector + graph, fused with RRF, gated by a relevance floor."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        graph_store: GraphStore,
        keyword_index: KeywordIndex,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
        min_cosine: float = 0.15,
        min_fused: float = 0.0,
        graph_hops: int = 1,
        keyword_max_df: int = 3,
        keyword_max_df_fraction: float = 0.2,
        rrf_k: int = 60,
        rrf_weights: dict[str, float] | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.keyword_index = keyword_index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.min_cosine = min_cosine
        self.min_fused = min_fused
        self.graph_hops = graph_hops
        self.keyword_max_df = keyword_max_df
        self.keyword_max_df_fraction = keyword_max_df_fraction
        # The keyword path outweighs the vector path: see reciprocal_rank_fusion.
        self.rrf_weights = dict(rrf_weights or {PATH_KEYWORD: 1.5})
        self.rrf_k = rrf_k
        self._chunks: dict[str, Chunk] = {}

    def register(self, chunks: list[Chunk]) -> None:
        """Keep a chunk lookup so keyword and graph hits can be materialised."""
        for c in chunks:
            self._chunks[c.id] = c

    def forget_document(self, document_id: str) -> None:
        for cid in [k for k, c in self._chunks.items() if c.document_id == document_id]:
            del self._chunks[cid]

    def _graph_expand(self, query: str, seed_ids: list[str]) -> tuple[list[str], list[str]]:
        """Chunks that mention entities from the query or from the seed chunks."""
        names = list(extract_entities(query))
        keys = {entity_key(n) for n in names}
        for cid in seed_ids:
            for nb in self.graph_store.neighbors(chunk_key(cid), rel=REL_MENTIONS, depth=1):
                if nb.startswith("ent:"):
                    keys.add(nb)
        hits: dict[str, int] = defaultdict(int)
        for key in sorted(keys):
            for nb in self.graph_store.neighbors(key, rel=REL_MENTIONS, depth=self.graph_hops):
                if nb.startswith("chunk:"):
                    cid = nb[len("chunk:") :]
                    if cid not in seed_ids:
                        hits[cid] += 1
        ranked = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
        return [cid for cid, _ in ranked[: self.candidate_k]], names

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = self.top_k if k is None else k
        result = RetrievalResult(query=query)
        if not query.strip():
            return result

        max_df, frac = self.keyword_max_df, self.keyword_max_df_fraction
        keyword = self.keyword_index.search(query, max_df, self.candidate_k, frac)
        result.rare_terms = self.keyword_index.rare_terms(query, max_df, frac)
        kw_ids = [cid for cid, _ in keyword]

        qvec = self.embedder.embed([query])[0]
        # Zero cosine means no shared feature at all; such hits carry no rank signal.
        vec_hits = [(c, s) for c, s in self.vector_store.search(qvec, self.candidate_k) if s > 0]
        vec_ids = [c.id for c, _ in vec_hits]
        cosines = {c.id: s for c, s in vec_hits}
        for c, _ in vec_hits:
            self._chunks.setdefault(c.id, c)

        seeds = list(dict.fromkeys(kw_ids[:k] + vec_ids[:k]))
        graph_ids, entities = self._graph_expand(query, seeds)
        result.entities = entities

        rankings = {PATH_KEYWORD: kw_ids, PATH_VECTOR: vec_ids, PATH_GRAPH: graph_ids}
        fused = reciprocal_rank_fusion(rankings, self.rrf_k, self.rrf_weights)
        kw_set = set(kw_ids)

        for cid, score in fused:
            chunk = self._chunks.get(cid)
            if chunk is None:
                continue
            cos = cosines.get(cid)
            if cos is None:
                cos = self._cosine_of(qvec, chunk)
                cosines[cid] = cos
            relevant = cid in kw_set or cos >= self.min_cosine
            if not relevant or score < self.min_fused:
                result.floored += 1
                continue
            sources = [name for name, ids in rankings.items() if cid in ids]
            result.chunks.append(chunk)
            result.scores.append(score)
            result.paths.append(PATH_FUSED if len(sources) > 1 else sources[0])
            if len(result.chunks) >= k:
                break
        result.cosines = {c.id: cosines[c.id] for c in result.chunks}
        return result

    def _cosine_of(self, qvec: tuple[float, ...], chunk: Chunk) -> float:
        from vmp.rag.vector import cosine

        emb = chunk.embedding
        if emb is None:
            emb = self.embedder.embed([chunk.text])[0]
        return cosine(qvec, emb)


__all__ = [
    "PATH_FUSED",
    "PATH_GRAPH",
    "PATH_KEYWORD",
    "PATH_VECTOR",
    "HybridRetriever",
    "KeywordIndex",
    "RetrievalResult",
    "reciprocal_rank_fusion",
    "strip_markdown",
]
