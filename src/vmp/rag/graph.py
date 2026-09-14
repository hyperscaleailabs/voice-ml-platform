"""Graph stores: a labelled property graph for GraphRAG-style expansion.

Nodes are addressed by a global `key` (for example `ent:gate a`, `chunk:<id>`).
Edges are directed but `neighbors` walks both directions, because "which chunks
mention this entity" and "which entities does this chunk mention" are both needed.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStore(Protocol):
    def add_node(self, label: str, key: str, props: dict[str, Any] | None = None) -> None: ...

    def add_edge(
        self, src: str, dst: str, rel: str, props: dict[str, Any] | None = None
    ) -> None: ...

    def neighbors(self, key: str, rel: str | None = None, depth: int = 1) -> list[str]: ...

    def subgraph_for(self, keys: list[str]) -> dict[str, Any]: ...


class InMemoryGraphStore:
    """Dict-backed graph. Deterministic ordering everywhere (sorted keys)."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._out: dict[str, set[tuple[str, str]]] = {}
        self._in: dict[str, set[tuple[str, str]]] = {}

    def add_node(self, label: str, key: str, props: dict[str, Any] | None = None) -> None:
        node = self._nodes.get(key)
        if node is None:
            self._nodes[key] = {"label": label, "key": key, "props": dict(props or {})}
        else:
            node["label"] = label
            node["props"].update(props or {})

    def add_edge(self, src: str, dst: str, rel: str, props: dict[str, Any] | None = None) -> None:
        for k in (src, dst):
            if k not in self._nodes:
                self.add_node("unknown", k)
        edge = self._edges.get((src, dst, rel))
        if edge is None:
            self._edges[(src, dst, rel)] = dict(props or {})
        else:
            edge.update(props or {})
        self._out.setdefault(src, set()).add((dst, rel))
        self._in.setdefault(dst, set()).add((src, rel))

    def node(self, key: str) -> dict[str, Any] | None:
        return self._nodes.get(key)

    def has_node(self, key: str) -> bool:
        return key in self._nodes

    def neighbors(self, key: str, rel: str | None = None, depth: int = 1) -> list[str]:
        """Keys reachable within `depth` hops (either direction), excluding `key`."""
        if key not in self._nodes or depth <= 0:
            return []
        seen = {key}
        frontier: deque[tuple[str, int]] = deque([(key, 0)])
        found: list[str] = []
        while frontier:
            cur, d = frontier.popleft()
            if d >= depth:
                continue
            adjacent = {n for n, r in self._out.get(cur, ()) if rel is None or r == rel}
            adjacent |= {n for n, r in self._in.get(cur, ()) if rel is None or r == rel}
            for nxt in sorted(adjacent):
                if nxt not in seen:
                    seen.add(nxt)
                    found.append(nxt)
                    frontier.append((nxt, d + 1))
        return found

    def subgraph_for(self, keys: list[str]) -> dict[str, Any]:
        wanted = {k for k in keys if k in self._nodes}
        nodes = [self._nodes[k] for k in sorted(wanted)]
        edges = [
            {"src": s, "dst": d, "rel": r, "props": p}
            for (s, d, r), p in sorted(self._edges.items())
            if s in wanted and d in wanted
        ]
        return {"nodes": nodes, "edges": edges}

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def remove_node(self, key: str) -> None:
        if key not in self._nodes:
            return
        del self._nodes[key]
        for dst, rel in self._out.pop(key, set()):
            self._edges.pop((key, dst, rel), None)
            self._in.get(dst, set()).discard((key, rel))
        for src, rel in self._in.pop(key, set()):
            self._edges.pop((src, key, rel), None)
            self._out.get(src, set()).discard((key, rel))

    def nodes_with_label(self, label: str) -> list[str]:
        return sorted(k for k, n in self._nodes.items() if n["label"] == label)

    def to_dict(self) -> dict[str, Any]:
        return self.subgraph_for(sorted(self._nodes))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InMemoryGraphStore:
        g = cls()
        for n in data.get("nodes", []):
            g.add_node(n["label"], n["key"], n.get("props"))
        for e in data.get("edges", []):
            g.add_edge(e["src"], e["dst"], e["rel"], e.get("props"))
        return g


def postgres_graph_schema_sql(prefix: str = "graph_") -> str:
    """Generic nodes/edges tables (JSONB props) for a PostgreSQL-backed graph."""
    n, e = f"{prefix}nodes", f"{prefix}edges"
    return f"""
CREATE TABLE IF NOT EXISTS {n} (
    id      BIGSERIAL PRIMARY KEY,
    label   TEXT NOT NULL,
    key     TEXT NOT NULL,
    props   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    UNIQUE (label, key)
);

CREATE TABLE IF NOT EXISTS {e} (
    id      BIGSERIAL PRIMARY KEY,
    src     TEXT NOT NULL,
    dst     TEXT NOT NULL,
    rel     TEXT NOT NULL,
    props   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    UNIQUE (src, dst, rel)
);

CREATE INDEX IF NOT EXISTS {e}_src_idx ON {e} (src, rel);
CREATE INDEX IF NOT EXISTS {e}_dst_idx ON {e} (dst, rel);
CREATE INDEX IF NOT EXISTS {n}_key_idx ON {n} (key);
""".strip()


class PostgresGraphStore:
    """Nodes/edges tables in PostgreSQL. `psycopg` imported on first connection.

    `neighbors` walks one hop per query; depth > 1 iterates.
    """

    def __init__(self, dsn: str, prefix: str = "graph_") -> None:
        self.dsn = dsn
        self.prefix = prefix
        self._conn: Any = None

    @staticmethod
    def schema_sql(prefix: str = "graph_") -> str:
        return postgres_graph_schema_sql(prefix)

    def connect(self) -> Any:
        if self._conn is None:
            import psycopg  # lazy

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def ensure_schema(self) -> None:
        self.connect().execute(postgres_graph_schema_sql(self.prefix))

    def add_node(self, label: str, key: str, props: dict[str, Any] | None = None) -> None:
        import json

        self.connect().execute(
            f"INSERT INTO {self.prefix}nodes (label, key, props) VALUES (%s, %s, %s::jsonb) "
            "ON CONFLICT (label, key) DO UPDATE SET props = "
            f"{self.prefix}nodes.props || EXCLUDED.props",
            (label, key, json.dumps(props or {})),
        )

    def add_edge(self, src: str, dst: str, rel: str, props: dict[str, Any] | None = None) -> None:
        import json

        self.connect().execute(
            f"INSERT INTO {self.prefix}edges (src, dst, rel, props) "
            "VALUES (%s, %s, %s, %s::jsonb) ON CONFLICT (src, dst, rel) DO UPDATE SET "
            f"props = {self.prefix}edges.props || EXCLUDED.props",
            (src, dst, rel, json.dumps(props or {})),
        )

    def neighbors(self, key: str, rel: str | None = None, depth: int = 1) -> list[str]:
        conn = self.connect()
        seen = {key}
        frontier = [key]
        found: list[str] = []
        for _ in range(max(depth, 0)):
            if not frontier:
                break
            rel_sql = "" if rel is None else " AND rel = %s"
            params: list[Any] = [frontier, frontier]
            if rel is not None:
                params = [frontier, rel, frontier, rel]
            rows = conn.execute(
                f"SELECT dst FROM {self.prefix}edges WHERE src = ANY(%s){rel_sql} "
                f"UNION SELECT src FROM {self.prefix}edges WHERE dst = ANY(%s){rel_sql}",
                params,
            ).fetchall()
            nxt = sorted({r[0] for r in rows} - seen)
            seen.update(nxt)
            found.extend(nxt)
            frontier = nxt
        return found

    def subgraph_for(self, keys: list[str]) -> dict[str, Any]:
        conn = self.connect()
        nodes = conn.execute(
            f"SELECT label, key, props FROM {self.prefix}nodes WHERE key = ANY(%s) ORDER BY key",
            (keys,),
        ).fetchall()
        edges = conn.execute(
            f"SELECT src, dst, rel, props FROM {self.prefix}edges "
            "WHERE src = ANY(%s) AND dst = ANY(%s) ORDER BY src, dst, rel",
            (keys, keys),
        ).fetchall()
        return {
            "nodes": [{"label": r[0], "key": r[1], "props": r[2]} for r in nodes],
            "edges": [{"src": r[0], "dst": r[1], "rel": r[2], "props": r[3]} for r in edges],
        }


class Neo4jGraphStore:
    """Neo4j over the Bolt driver. `neo4j` is imported on first connection.

    Labels and relationship types are interpolated into Cypher, so they are
    restricted to identifier characters. Keys and props are passed as parameters.
    """

    def __init__(self, uri: str, user: str, password: str, database: str | None = None) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver: Any = None

    def connect(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase  # lazy

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    @staticmethod
    def _ident(name: str) -> str:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"unsafe identifier: {name!r}")
        return name

    def _run(self, query: str, **params: Any) -> list[Any]:
        with self.connect().session(database=self.database) as session:
            return [r for r in session.run(query, **params)]

    def add_node(self, label: str, key: str, props: dict[str, Any] | None = None) -> None:
        lbl = self._ident(label)
        self._run(f"MERGE (n:{lbl} {{key: $key}}) SET n += $props", key=key, props=props or {})

    def add_edge(self, src: str, dst: str, rel: str, props: dict[str, Any] | None = None) -> None:
        r = self._ident(rel)
        self._run(
            "MERGE (a {key: $src}) MERGE (b {key: $dst}) "
            f"MERGE (a)-[e:{r}]->(b) SET e += $props",
            src=src,
            dst=dst,
            props=props or {},
        )

    def neighbors(self, key: str, rel: str | None = None, depth: int = 1) -> list[str]:
        r = f":{self._ident(rel)}" if rel else ""
        rows = self._run(
            f"MATCH (a {{key: $key}})-[{r}*1..{int(depth)}]-(b) "
            "WHERE b.key <> $key RETURN DISTINCT b.key AS key ORDER BY key",
            key=key,
        )
        return [row["key"] for row in rows]

    def subgraph_for(self, keys: list[str]) -> dict[str, Any]:
        nodes = self._run(
            "MATCH (n) WHERE n.key IN $keys "
            "RETURN labels(n)[0] AS label, n.key AS key, properties(n) AS props ORDER BY key",
            keys=keys,
        )
        edges = self._run(
            "MATCH (a)-[e]->(b) WHERE a.key IN $keys AND b.key IN $keys "
            "RETURN a.key AS src, b.key AS dst, type(e) AS rel, properties(e) AS props "
            "ORDER BY src, dst, rel",
            keys=keys,
        )
        return {
            "nodes": [
                {"label": r["label"], "key": r["key"], "props": dict(r["props"])} for r in nodes
            ],
            "edges": [
                {"src": r["src"], "dst": r["dst"], "rel": r["rel"], "props": dict(r["props"])}
                for r in edges
            ],
        }


__all__ = [
    "GraphStore",
    "InMemoryGraphStore",
    "Neo4jGraphStore",
    "PostgresGraphStore",
    "postgres_graph_schema_sql",
]
