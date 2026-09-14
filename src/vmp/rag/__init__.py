"""Retrieval: chunking, embeddings, vector and graph stores, hybrid retrieval.

The default path is standard library only: a deterministic hashing embedder, an
in-memory cosine store, an in-memory graph, and a rule-based entity extractor.
pgvector, Qdrant, Neo4j and sentence-transformers are optional adapters that
import their dependency lazily inside the class.
"""

from __future__ import annotations

from vmp.rag.chunk import IncrementalPlan, chunk_document, load_documents, plan_incremental
from vmp.rag.embed import Embedder, HashingEmbedder
from vmp.rag.extract import Extraction, extract_chunk, extract_entities
from vmp.rag.graph import GraphStore, InMemoryGraphStore
from vmp.rag.index import Index
from vmp.rag.retrieve import HybridRetriever, KeywordIndex, RetrievalResult
from vmp.rag.vector import InMemoryVectorStore, VectorStore

__all__ = [
    "Embedder",
    "Extraction",
    "GraphStore",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryGraphStore",
    "InMemoryVectorStore",
    "IncrementalPlan",
    "Index",
    "KeywordIndex",
    "RetrievalResult",
    "VectorStore",
    "chunk_document",
    "extract_chunk",
    "extract_entities",
    "load_documents",
    "plan_incremental",
]
