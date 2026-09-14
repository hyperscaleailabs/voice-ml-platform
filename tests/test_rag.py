"""Retrieval subsystem tests. Standard library only; adapters are importorskip'ed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vmp.rag.chunk import chunk_document, load_documents, plan_incremental
from vmp.rag.embed import HashingEmbedder
from vmp.rag.extract import chunk_key, entity_key, extract_chunk, extract_entities
from vmp.rag.graph import InMemoryGraphStore, postgres_graph_schema_sql
from vmp.rag.index import Index
from vmp.rag.retrieve import (
    PATH_KEYWORD,
    HybridRetriever,
    reciprocal_rank_fusion,
    strip_markdown,
)
from vmp.rag.vector import InMemoryVectorStore, cosine, schema_sql
from vmp.types import Chunk, Document

STREAMING = """# Streaming

The sentence segmenter splits the LLM token stream into sentences.
Each sentence is sent to TTS as soon as it is complete.
The playback stage records response_ms, the time to first audio.
Streaming reduces time to first audio because TTS starts on the first sentence.
"""

GATES = """# Release gates

Every candidate model passes a release gate before promotion.
The WER gate compares golden set WER against the production model.
The latency gate checks the time-to-first-audio p95.
The error gate checks the error rate over the last evaluation run.

Gate A verifies the adapter checksum before export.
"""

STT = """# Speech to text

The STT stage transcribes audio with Whisper.
faster-whisper runs on CPU and mlx-whisper runs on the M4 GPU.
Wake-word correction uses Jaro-Winkler on the transcript.
"""


def _doc(doc_id: str, text: str) -> Document:
    return Document(id=doc_id, path=doc_id, text=text, sha="x")


def _write_corpus(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return root


# -- chunker ----------------------------------------------------------------


def test_chunker_boundaries_and_overlap():
    text = "\n".join(f"line {i}" for i in range(1, 11))  # 10 lines
    chunks = chunk_document(_doc("d", text), chunk_lines=4, overlap=1)
    spans = [(c.start_line, c.end_line) for c in chunks]
    assert spans == [(1, 4), (4, 7), (7, 10)]
    assert chunks[0].text.splitlines()[-1] == chunks[1].text.splitlines()[0] == "line 4"
    assert chunks[-1].end_line == 10
    assert all(c.id == f"d:{s}-{e}" for c, (s, e) in zip(chunks, spans, strict=True))


def test_chunker_short_document_and_blank_windows():
    assert len(chunk_document(_doc("d", "one\ntwo"), chunk_lines=10, overlap=2)) == 1
    assert chunk_document(_doc("d", "\n\n\n"), chunk_lines=2, overlap=0) == []
    with pytest.raises(ValueError):
        chunk_document(_doc("d", "x"), chunk_lines=3, overlap=3)


def test_loader_and_incremental_sha(tmp_path):
    corpus = _write_corpus(tmp_path / "c", {"a.md": "alpha\n", "b.md": "beta\n", "c.py": "x=1\n"})
    docs = load_documents(corpus, include=("**/*.md",))
    assert [d.id for d in docs] == ["a.md", "b.md"]
    previous = {d.id: d.sha for d in docs}

    (corpus / "b.md").write_text("beta changed\n")
    (corpus / "d.md").write_text("delta\n")
    (corpus / "a.md").unlink()
    plan = plan_incremental(previous, load_documents(corpus, include=("**/*.md",)))
    assert plan.counts() == {"added": 1, "changed": 1, "unchanged": 0, "removed": 1}
    assert plan.removed == ["a.md"]


def test_index_incremental_keeps_unchanged_chunks(tmp_path):
    corpus = _write_corpus(tmp_path / "c", {"s.md": STREAMING, "g.md": GATES})
    ix = Index(chunk_lines=4, overlap=1)
    first = ix.build(corpus)
    assert first["plan"]["added"] == 2
    before = {c.id: c.embedding for c in ix.vector_store.all()}

    (corpus / "g.md").write_text(GATES + "\nAnother line about the smoke gate.\n")
    second = ix.build(corpus)
    assert second["plan"] == {"added": 0, "changed": 1, "unchanged": 1, "removed": 0}
    after = {c.id: c.embedding for c in ix.vector_store.all()}
    for cid, emb in before.items():
        if cid.startswith("s.md:"):
            assert after[cid] == emb
    dry = ix.build(corpus, dry_run=True)
    assert dry["dry_run"] is True and dry["plan"]["unchanged"] == 2


# -- embedder ---------------------------------------------------------------


def test_hashing_embedder_deterministic_and_normalised():
    e1, e2 = HashingEmbedder(dim=64), HashingEmbedder(dim=64)
    a = e1.embed(["The quick brown fox", "jumps"])
    b = e2.embed(["The quick brown fox", "jumps"])
    assert a == b
    assert len(a[0]) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in a[0])), 1.0, rel_tol=1e-9)
    assert e1.embed([""])[0] == tuple([0.0] * 64)
    assert cosine(a[0], a[0]) == pytest.approx(1.0)
    assert cosine(a[0], e1.embed(["quick fox"])[0]) > cosine(a[0], e1.embed(["kokoro tts"])[0])


# -- vector store -----------------------------------------------------------


def test_in_memory_cosine_ordering_and_delete():
    store = InMemoryVectorStore()
    store.upsert(
        [
            Chunk("a", "d1", 1, 1, "x", (1.0, 0.0)),
            Chunk("b", "d1", 2, 2, "y", (0.7, 0.7)),
            Chunk("c", "d2", 1, 1, "z", (0.0, 1.0)),
        ]
    )
    hits = store.search((1.0, 0.0), k=3)
    assert [c.id for c, _ in hits] == ["a", "b", "c"]
    assert hits[0][1] == pytest.approx(1.0) and hits[2][1] == pytest.approx(0.0)
    assert store.delete_document("d1") == 2 and store.count() == 1
    with pytest.raises(ValueError):
        store.upsert([Chunk("n", "d", 1, 1, "no embedding")])


# -- graph ------------------------------------------------------------------


def test_graph_neighbors_depth_and_rel():
    g = InMemoryGraphStore()
    g.add_node("Chunk", "chunk:1")
    g.add_node("Entity", "ent:a")
    g.add_node("Entity", "ent:b")
    g.add_node("Chunk", "chunk:2")
    g.add_edge("chunk:1", "ent:a", "mentions")
    g.add_edge("ent:a", "ent:b", "cooccurs")
    g.add_edge("chunk:2", "ent:b", "mentions")
    assert g.neighbors("ent:a") == ["chunk:1", "ent:b"]
    assert g.neighbors("ent:a", rel="mentions") == ["chunk:1"]
    assert g.neighbors("chunk:1", depth=2) == ["ent:a", "ent:b"]
    assert g.neighbors("chunk:1", depth=3) == ["ent:a", "ent:b", "chunk:2"]
    assert g.neighbors("missing") == []
    sub = g.subgraph_for(["chunk:1", "ent:a"])
    assert [n["key"] for n in sub["nodes"]] == ["chunk:1", "ent:a"]
    assert len(sub["edges"]) == 1
    g.remove_node("ent:a")
    assert g.neighbors("chunk:1") == [] and g.edge_count() == 1
    assert InMemoryGraphStore.from_dict(g.to_dict()).to_dict() == g.to_dict()


# -- extraction -------------------------------------------------------------


def test_extraction_deterministic():
    text = "Gate A verifies the adapter checksum. The WER gate calls `check_wer` in vmp.eval."
    ents = extract_entities(text)
    assert ents == extract_entities(text)
    assert "Gate A" in ents and "WER" in ents and "check_wer" in ents and "vmp.eval" in ents
    assert "The" not in ents
    ex = extract_chunk(Chunk("c1", "d", 1, 1, text))
    assert ex.nodes[0][1] == chunk_key("c1")
    mentions = {(s, d) for s, d, r, _ in ex.edges if r == "mentions"}
    assert (chunk_key("c1"), entity_key("Gate A")) in mentions
    co = [(s, d) for s, d, r, _ in ex.edges if r == "cooccurs"]
    assert len(co) == math.comb(len(ents), 2)
    assert ex == extract_chunk(Chunk("c1", "d", 1, 1, text))


# -- retrieval --------------------------------------------------------------


def _rare_literal_index() -> tuple[Index, str]:
    """Distractors are short and dense in the query's vocabulary; the target chunk
    contains the literal "Gate A" once, diluted by unrelated lines, so cosine
    ranks it low. Mirrors the alpha-core "Gate A" miss (cycle 5, 2026-09-12)."""
    filler = "\n".join(
        f"Operational note {i}: rotate logs, archive traces, prune old bundles." for i in range(8)
    )
    target = "Gate A verifies the adapter checksum before export.\n" + filler
    distractors = {
        f"gate{i}.md": (
            f"The {name} gate checks the {name} metric.\nThe gate check fails when "
            f"the {name} check regresses.\nThe gate check runs on every candidate.\n"
        )
        for i, name in enumerate(["wer", "latency", "error", "smoke", "regression"])
    }
    ix = Index(chunk_lines=12, overlap=0, top_k=3, candidate_k=20)
    docs = [_doc("target.md", target)] + [_doc(k, v) for k, v in distractors.items()]
    ix.add_documents(docs)
    return ix, "target.md:1-9"


def test_rare_literal_found_by_keyword_when_cosine_misranks():
    ix, target_id = _rare_literal_index()
    query = "what does Gate A check"
    qvec = ix.embedder.embed([query])[0]
    top_cosine = [c.id for c, _ in ix.vector_store.search(qvec, ix.retriever.top_k)]
    assert target_id not in top_cosine, top_cosine

    res = ix.query(query)
    assert "gate a" in res.rare_terms
    assert res.ids[0] == target_id
    assert res.paths[0] == "fused"  # keyword + graph (entity "Gate A") + low-rank vector

    # Keyword path alone, with no graph and a tiny vector candidate pool.
    r = HybridRetriever(
        ix.embedder, ix.vector_store, InMemoryGraphStore(), ix.keyword_index, top_k=3, candidate_k=2
    )
    r.register(ix.vector_store.all())
    res = r.retrieve(query)
    assert res.ids[0] == target_id and res.paths[0] == PATH_KEYWORD


def test_relevance_floor_returns_empty_for_off_topic(tmp_path):
    corpus = _write_corpus(tmp_path / "c", {"s.md": STREAMING, "g.md": GATES, "t.md": STT})
    ix = Index(chunk_lines=4, overlap=1, relevance_floor=0.15)
    ix.build(corpus)
    off = ix.query("what is the capital of France")
    assert len(off) == 0 and off.floored > 0
    assert off.context_pack() == ""
    on = ix.query("how does streaming reduce time to first audio")
    assert on.ids[0].startswith("s.md:")
    assert all(on.cosines[c.id] >= 0.15 or c.id in on.ids for c in on.chunks)
    assert on.scores == sorted(on.scores, reverse=True)

    loose = Index(chunk_lines=4, overlap=1, relevance_floor=0.0)
    loose.build(corpus)
    # Without the floor the same question returns the nearest junk: the code-echo case.
    assert len(loose.query("what is the capital of France")) > 0


def test_graph_expansion_adds_chunks_that_share_entities():
    ix = Index(chunk_lines=3, overlap=0, top_k=5, candidate_k=2, relevance_floor=0.0)
    ix.add_documents(
        [
            _doc("a.md", "Kokoro synthesises each sentence.\nThe playback queue receives audio.\n"),
            _doc("b.md", "Latency notes.\nKokoro real-time factor is measured per sentence.\n"),
            _doc("c.md", "Unrelated: rotate logs, archive traces.\n"),
        ]
    )
    res = ix.query("How does Kokoro synthesise audio")
    assert "Kokoro" in res.entities
    assert {i.split(":")[0] for i in res.ids} >= {"a.md", "b.md"}


def test_rrf_ordering():
    fused = reciprocal_rank_fusion({"x": ["a", "b", "c"], "y": ["b", "c"], "z": ["c"]}, k=60)
    assert [cid for cid, _ in fused] == ["c", "b", "a"]
    assert fused[0][1] == pytest.approx(1 / 63 + 1 / 62 + 1 / 61)
    assert fused[2][1] == pytest.approx(1 / 61)


def test_context_pack_is_plain_text_and_bounded():
    md = "# Title\n- **bold** [link](http://x)\n`code`"
    assert strip_markdown(md) == "Title\nbold link\ncode"
    ix = Index(chunk_lines=4, overlap=1)
    ix.add_documents([_doc("s.md", STREAMING)])
    res = ix.query("sentence segmenter splits the token stream")
    pack = res.context_pack(max_chars=120)
    assert pack.startswith("From s.md, lines ")
    assert len(pack) <= 120 and "#" not in pack and "**" not in pack


# -- persistence and check ----------------------------------------------------


def test_persistence_round_trip(tmp_path):
    corpus = _write_corpus(tmp_path / "c", {"s.md": STREAMING, "g.md": GATES})
    root = tmp_path / "index"
    ix = Index(root, chunk_lines=4, overlap=1)
    ix.build(corpus)
    ix.save()
    assert Index.exists(root)
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest["documents"]) == {"s.md", "g.md"}

    back = Index(root, chunk_lines=4, overlap=1).load()
    assert back.vector_store.count() == ix.vector_store.count()
    assert back.graph_store.to_dict() == ix.graph_store.to_dict()
    assert back.documents == ix.documents
    q = "what does Gate A verify"
    assert back.query(q).to_dict()["results"] == ix.query(q).to_dict()["results"]
    assert back.build(corpus, dry_run=True)["plan"]["unchanged"] == 2
    with pytest.raises(ValueError):
        Index(root, embedder=HashingEmbedder(dim=8)).load()


def test_check_reports_spread(tmp_path):
    corpus = _write_corpus(tmp_path / "c", {"s.md": STREAMING, "g.md": GATES, "t.md": STT})
    ix = Index(chunk_lines=4, overlap=1)
    ix.build(corpus)
    rep = ix.check(["what does Gate A verify"], k=5)
    q = rep["queries"][0]
    assert q["n"] == 5 and q["rank1_cosine"] >= q["rank5_cosine"]
    assert q["spread"] == pytest.approx(q["rank1_cosine"] - q["rank5_cosine"])


def test_from_config_and_cli(tmp_path, capsys):
    from vmp.cli import main
    from vmp.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "rag.toml")
    ix = Index.from_config(cfg)
    assert ix.embedder.dim == cfg["rag"]["embedder"]["dim"]
    assert ix.retriever.min_cosine == cfg["rag"]["retrieval"]["relevance_floor"]

    corpus = _write_corpus(tmp_path / "c", {"s.md": STREAMING, "g.md": GATES})
    root = str(tmp_path / "idx")
    assert main(["rag", "--root", root, "index", str(corpus), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert main(["rag", "--root", root, "index", str(corpus)]) == 0
    assert json.loads(capsys.readouterr().out)["chunks_total"] > 0
    assert main(["rag", "--root", root, "query", "what does Gate A verify", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["results"] and out["results"][0]["document_id"] == "g.md"
    assert main(["rag", "--root", root, "check", "--query", "streaming", "--k", "3"]) == 0
    assert 1 <= json.loads(capsys.readouterr().out)["queries"][0]["n"] <= 3


# -- schema helpers ------------------------------------------------------------


def test_schema_sql_strings():
    sql = schema_sql(384)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "vector(384)" in sql
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "indexed_at" in sql and "REFERENCES rag_documents(id)" in sql
    g = postgres_graph_schema_sql()
    assert "UNIQUE (label, key)" in g and "props   JSONB" in g
    assert "graph_edges_src_idx" in g and "graph_edges_dst_idx" in g


# -- adapters (need a service) --------------------------------------------------


@pytest.mark.integration
def test_pgvector_adapter_constructs():
    pytest.importorskip("psycopg")
    from vmp.rag.vector import PgVectorStore

    store = PgVectorStore("postgresql://localhost/vmp", dim=16)
    assert "vector(16)" in store.schema_sql(16)


@pytest.mark.integration
def test_qdrant_adapter_constructs():
    pytest.importorskip("qdrant_client")
    from vmp.rag.vector import QdrantVectorStore

    assert QdrantVectorStore(dim=16, path=":memory:").collection == "vmp_chunks"


@pytest.mark.integration
def test_neo4j_adapter_constructs():
    pytest.importorskip("neo4j")
    from vmp.rag.graph import Neo4jGraphStore

    assert Neo4jGraphStore("bolt://localhost:7687", "neo4j", "x").uri.startswith("bolt://")


@pytest.mark.integration
def test_sentence_transformer_adapter_constructs():
    pytest.importorskip("sentence_transformers")
    from vmp.rag.embed import SentenceTransformerEmbedder

    assert SentenceTransformerEmbedder().model_name == "all-MiniLM-L6-v2"
