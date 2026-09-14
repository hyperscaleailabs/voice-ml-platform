"""Vector stores: in-memory cosine (reference), pgvector and Qdrant adapters.

The in-memory store is what tests and demos use. The adapters import their
client lazily and mirror the same two-table shape: `documents` and `chunks`.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

from vmp.types import Chunk


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@runtime_checkable
class VectorStore(Protocol):
    """Stores embedded chunks and answers nearest-neighbour queries by cosine."""

    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, query_vec: tuple[float, ...], k: int) -> list[tuple[Chunk, float]]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self) -> int: ...


class InMemoryVectorStore:
    """Exhaustive cosine search over a dict. Fine for thousands of chunks."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.id} has no embedding")
            self._chunks[c.id] = c

    def search(self, query_vec: tuple[float, ...], k: int) -> list[tuple[Chunk, float]]:
        if k <= 0:
            return []
        scored = [(c, cosine(query_vec, c.embedding or ())) for c in self._chunks.values()]
        # Ties break on id so results are stable across runs.
        scored.sort(key=lambda cs: (-cs[1], cs[0].id))
        return scored[:k]

    def delete_document(self, document_id: str) -> int:
        doomed = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
        for cid in doomed:
            del self._chunks[cid]
        return len(doomed)

    def count(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def all(self) -> list[Chunk]:
        return [self._chunks[k] for k in sorted(self._chunks)]

    def to_dict(self) -> dict[str, Any]:
        return {"chunks": [c.to_dict() for c in self.all()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InMemoryVectorStore:
        store = cls()
        store.upsert([Chunk.from_dict(c) for c in data.get("chunks", [])])
        return store


def schema_sql(dim: int, prefix: str = "rag_") -> str:
    """DDL for the pgvector backend: documents, chunks, HNSW cosine index."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    d, c = f"{prefix}documents", f"{prefix}chunks"
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {d} (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    sha         TEXT NOT NULL,
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {c} (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES {d}(id) ON DELETE CASCADE,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL
);

CREATE INDEX IF NOT EXISTS {c}_document_id_idx ON {c} (document_id);
CREATE INDEX IF NOT EXISTS {c}_embedding_hnsw_idx
    ON {c} USING hnsw (embedding vector_cosine_ops);
""".strip()


def _vector_literal(vec: tuple[float, ...]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PgVectorStore:
    """PostgreSQL + pgvector. `psycopg` is imported on first connection."""

    def __init__(self, dsn: str, dim: int, prefix: str = "rag_") -> None:
        self.dsn = dsn
        self.dim = dim
        self.prefix = prefix
        self._conn: Any = None

    @staticmethod
    def schema_sql(dim: int, prefix: str = "rag_") -> str:
        return schema_sql(dim, prefix)

    def connect(self) -> Any:
        if self._conn is None:
            import psycopg  # lazy

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def ensure_schema(self) -> None:
        self.connect().execute(schema_sql(self.dim, self.prefix))

    def upsert_document(self, document_id: str, path: str, sha: str) -> None:
        self.connect().execute(
            f"INSERT INTO {self.prefix}documents (id, path, sha) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET path = EXCLUDED.path, sha = EXCLUDED.sha, "
            "indexed_at = now()",
            (document_id, path, sha),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            for c in chunks:
                if c.embedding is None:
                    raise ValueError(f"chunk {c.id} has no embedding")
                cur.execute(
                    f"INSERT INTO {self.prefix}chunks "
                    "(id, document_id, start_line, end_line, text, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s::vector) "
                    "ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, "
                    "embedding = EXCLUDED.embedding, start_line = EXCLUDED.start_line, "
                    "end_line = EXCLUDED.end_line",
                    (
                        c.id,
                        c.document_id,
                        c.start_line,
                        c.end_line,
                        c.text,
                        _vector_literal(c.embedding),
                    ),
                )

    def search(self, query_vec: tuple[float, ...], k: int) -> list[tuple[Chunk, float]]:
        rows = (
            self.connect()
            .execute(
                f"SELECT id, document_id, start_line, end_line, text, "
                f"1 - (embedding <=> %s::vector) AS score FROM {self.prefix}chunks "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vector_literal(query_vec), _vector_literal(query_vec), k),
            )
            .fetchall()
        )
        return [
            (Chunk(id=r[0], document_id=r[1], start_line=r[2], end_line=r[3], text=r[4]), r[5])
            for r in rows
        ]

    def delete_document(self, document_id: str) -> int:
        cur = self.connect().execute(
            f"DELETE FROM {self.prefix}chunks WHERE document_id = %s", (document_id,)
        )
        self.connect().execute(f"DELETE FROM {self.prefix}documents WHERE id = %s", (document_id,))
        return int(cur.rowcount or 0)

    def count(self) -> int:
        row = self.connect().execute(f"SELECT count(*) FROM {self.prefix}chunks").fetchone()
        return int(row[0]) if row else 0


class QdrantVectorStore:
    """Qdrant collection with cosine distance. `qdrant-client` imported lazily."""

    def __init__(
        self,
        dim: int,
        collection: str = "vmp_chunks",
        url: str | None = None,
        path: str | None = None,
    ) -> None:
        self.dim = dim
        self.collection = collection
        self.url = url
        self.path = path
        self._client: Any = None

    def connect(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy
            from qdrant_client.http import models

            self._client = QdrantClient(url=self.url, path=self.path)
            if not self._client.collection_exists(self.collection):
                self._client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=self.dim, distance=models.Distance.COSINE
                    ),
                )
        return self._client

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        import uuid

        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def upsert(self, chunks: list[Chunk]) -> None:
        from qdrant_client.http import models  # lazy

        points = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"chunk {c.id} has no embedding")
            payload = c.to_dict()
            payload.pop("embedding", None)
            points.append(
                models.PointStruct(
                    id=self._point_id(c.id), vector=list(c.embedding), payload=payload
                )
            )
        if points:
            self.connect().upsert(self.collection, points=points)

    def search(self, query_vec: tuple[float, ...], k: int) -> list[tuple[Chunk, float]]:
        hits = self.connect().search(self.collection, query_vector=list(query_vec), limit=k)
        return [(Chunk.from_dict(dict(h.payload)), float(h.score)) for h in hits]

    def delete_document(self, document_id: str) -> int:
        from qdrant_client.http import models  # lazy

        flt = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )
        before = self.count()
        self.connect().delete(self.collection, points_selector=models.FilterSelector(filter=flt))
        return before - self.count()

    def count(self) -> int:
        return int(self.connect().count(self.collection, exact=True).count)


__all__ = [
    "InMemoryVectorStore",
    "PgVectorStore",
    "QdrantVectorStore",
    "VectorStore",
    "cosine",
    "schema_sql",
]
