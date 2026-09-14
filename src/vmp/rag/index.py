"""`Index`: build, persist, load and check a retrieval index.

Orchestrates the chunker, embedder, vector store, graph store, keyword index and
retriever. With the default in-memory stores the index persists as JSON files
under a root directory, which is enough for demos, tests and small corpora.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vmp.rag.chunk import DEFAULT_INCLUDE, chunk_document, load_documents, plan_incremental
from vmp.rag.embed import Embedder, HashingEmbedder
from vmp.rag.extract import chunk_key, extract_chunk
from vmp.rag.graph import InMemoryGraphStore
from vmp.rag.retrieve import HybridRetriever, KeywordIndex, RetrievalResult
from vmp.rag.vector import InMemoryVectorStore
from vmp.types import Chunk, Document

MANIFEST = "manifest.json"
CHUNKS = "chunks.json"
GRAPH = "graph.json"


class Index:
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        embedder: Embedder | None = None,
        vector_store: InMemoryVectorStore | None = None,
        graph_store: InMemoryGraphStore | None = None,
        chunk_lines: int = 20,
        overlap: int = 4,
        top_k: int = 5,
        candidate_k: int = 20,
        relevance_floor: float = 0.15,
        min_fused: float = 0.0,
        graph_hops: int = 1,
        keyword_max_df: int = 3,
        keyword_max_df_fraction: float = 0.2,
        include: tuple[str, ...] = DEFAULT_INCLUDE,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.embedder = embedder or HashingEmbedder()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.graph_store = graph_store or InMemoryGraphStore()
        self.keyword_index = KeywordIndex()
        self.chunk_lines = chunk_lines
        self.overlap = overlap
        self.include = include
        self.documents: dict[str, dict[str, Any]] = {}
        self.retriever = HybridRetriever(
            self.embedder,
            self.vector_store,
            self.graph_store,
            self.keyword_index,
            top_k=top_k,
            candidate_k=candidate_k,
            min_cosine=relevance_floor,
            min_fused=min_fused,
            graph_hops=graph_hops,
            keyword_max_df=keyword_max_df,
            keyword_max_df_fraction=keyword_max_df_fraction,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], root: str | Path | None = None) -> Index:
        """Build from the `[rag]` tables of `configs/rag.toml` (already loaded)."""
        rag = cfg.get("rag", cfg)
        chunk = rag.get("chunk", {})
        emb = rag.get("embedder", {})
        ret = rag.get("retrieval", {})
        embedder: Embedder
        if emb.get("kind", "hashing") == "hashing":
            embedder = HashingEmbedder(dim=int(emb.get("dim", 512)))
        else:
            raise ValueError(f"unsupported embedder kind for from_config: {emb.get('kind')!r}")
        return cls(
            root,
            embedder=embedder,
            chunk_lines=int(chunk.get("lines", 20)),
            overlap=int(chunk.get("overlap", 4)),
            top_k=int(ret.get("top_k", 5)),
            candidate_k=int(ret.get("candidate_k", 20)),
            relevance_floor=float(ret.get("relevance_floor", 0.15)),
            min_fused=float(ret.get("min_fused", 0.0)),
            graph_hops=int(ret.get("graph_hops", 1)),
            keyword_max_df=int(ret.get("keyword_max_df", 3)),
            keyword_max_df_fraction=float(ret.get("keyword_max_df_fraction", 0.2)),
            include=tuple(rag.get("corpus", {}).get("include", DEFAULT_INCLUDE)),
        )

    # -- building -----------------------------------------------------------

    def add_documents(self, docs: list[Document]) -> int:
        """Chunk, embed, extract and store. Returns the number of chunks added."""
        total = 0
        for doc in docs:
            self.remove_document(doc.id)
            chunks = chunk_document(doc, self.chunk_lines, self.overlap)
            vecs = self.embedder.embed([c.text for c in chunks])
            embedded = [
                Chunk(c.id, c.document_id, c.start_line, c.end_line, c.text, v)
                for c, v in zip(chunks, vecs, strict=True)
            ]
            self.vector_store.upsert(embedded)
            self.keyword_index.add(embedded)
            self.retriever.register(embedded)
            for c in embedded:
                ex = extract_chunk(c)
                for label, key, props in ex.nodes:
                    self.graph_store.add_node(label, key, props)
                for src, dst, rel, props in ex.edges:
                    self.graph_store.add_edge(src, dst, rel, props)
            self.documents[doc.id] = {
                "path": doc.path,
                "sha": doc.sha,
                "chunks": len(embedded),
                "indexed_at": time.time(),
            }
            total += len(embedded)
        return total

    def remove_document(self, document_id: str) -> int:
        removed = 0
        if isinstance(self.vector_store, InMemoryVectorStore):
            for c in self.vector_store.all():
                if c.document_id == document_id:
                    self.graph_store.remove_node(chunk_key(c.id))
        removed += self.vector_store.delete_document(document_id)
        self.keyword_index.delete_document(document_id)
        self.retriever.forget_document(document_id)
        self.documents.pop(document_id, None)
        return removed

    def build(
        self,
        corpus_dir: str | Path,
        include: tuple[str, ...] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Index a directory incrementally. `dry_run` only reports the plan."""
        docs = load_documents(corpus_dir, include or self.include)
        previous = {k: v["sha"] for k, v in self.documents.items()}
        plan = plan_incremental(previous, docs)
        report: dict[str, Any] = {
            "corpus": str(corpus_dir),
            "documents": len(docs),
            "plan": plan.counts(),
            "chunk_lines": self.chunk_lines,
            "overlap": self.overlap,
            "embedder": type(self.embedder).__name__,
            "dim": self.embedder.dim,
            "dry_run": dry_run,
        }
        if dry_run:
            report["chunks_planned"] = sum(
                len(chunk_document(d, self.chunk_lines, self.overlap))
                for d in plan.added + plan.changed
            )
            return report
        for doc_id in plan.removed:
            self.remove_document(doc_id)
        added = self.add_documents(plan.added + plan.changed)
        report["chunks_added"] = added
        report["chunks_total"] = self.vector_store.count()
        report["graph_nodes"] = self.graph_store.node_count()
        report["graph_edges"] = self.graph_store.edge_count()
        return report

    # -- querying -----------------------------------------------------------

    def query(self, text: str, k: int | None = None) -> RetrievalResult:
        return self.retriever.retrieve(text, k)

    def check(self, queries: list[str], k: int = 20) -> dict[str, Any]:
        """Score spread rank 1 vs rank N per query, as alpha-core's `check_rag` did.

        A small spread means cosine alone cannot separate relevant from
        irrelevant chunks, which is the case for a relevance floor.
        """
        out: dict[str, Any] = {"k": k, "chunks": self.vector_store.count(), "queries": []}
        for q in queries:
            qvec = self.embedder.embed([q])[0]
            hits = self.vector_store.search(qvec, k)
            res = self.retriever.retrieve(q, k)
            top = hits[0][1] if hits else None
            bottom = hits[-1][1] if hits else None
            out["queries"].append(
                {
                    "query": q,
                    "n": len(hits),
                    "rank1_cosine": top,
                    f"rank{len(hits)}_cosine": bottom,
                    "spread": (top - bottom) if hits else None,
                    "returned_after_floor": len(res),
                    "rare_terms": res.rare_terms,
                }
            )
        return out

    # -- persistence ----------------------------------------------------------

    def save(self, root: str | Path | None = None) -> Path:
        base = Path(root) if root is not None else self.root
        if base is None:
            raise ValueError("no root given for save()")
        if not isinstance(self.vector_store, InMemoryVectorStore):
            raise TypeError("save() persists in-memory stores only; adapters persist themselves")
        base.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 1,
            "embedder": type(self.embedder).__name__,
            "dim": self.embedder.dim,
            "chunk_lines": self.chunk_lines,
            "overlap": self.overlap,
            "documents": self.documents,
        }
        (base / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
        (base / CHUNKS).write_text(json.dumps(self.vector_store.to_dict()))
        (base / GRAPH).write_text(json.dumps(self.graph_store.to_dict()))
        return base

    def load(self, root: str | Path | None = None) -> Index:
        base = Path(root) if root is not None else self.root
        if base is None:
            raise ValueError("no root given for load()")
        manifest = json.loads((base / MANIFEST).read_text())
        if manifest.get("dim") != self.embedder.dim:
            raise ValueError(f"index dim {manifest.get('dim')} != embedder dim {self.embedder.dim}")
        self.documents = dict(manifest.get("documents", {}))
        self.chunk_lines = int(manifest.get("chunk_lines", self.chunk_lines))
        self.overlap = int(manifest.get("overlap", self.overlap))
        self.vector_store = InMemoryVectorStore.from_dict(json.loads((base / CHUNKS).read_text()))
        self.graph_store = InMemoryGraphStore.from_dict(json.loads((base / GRAPH).read_text()))
        self.keyword_index = KeywordIndex()
        chunks = self.vector_store.all()
        self.keyword_index.add(chunks)
        self.retriever.vector_store = self.vector_store
        self.retriever.graph_store = self.graph_store
        self.retriever.keyword_index = self.keyword_index
        self.retriever._chunks = {}
        self.retriever.register(chunks)
        return self

    @staticmethod
    def exists(root: str | Path) -> bool:
        return (Path(root) / MANIFEST).exists()


__all__ = ["CHUNKS", "GRAPH", "MANIFEST", "Index"]
