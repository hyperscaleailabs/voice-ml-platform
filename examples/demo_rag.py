"""Hybrid retrieval demo. Standard library only; runs in well under five seconds.

Writes a small corpus to a temporary directory, indexes it with the hashing
embedder and the in-memory graph, then runs three queries: on-topic, a rare
literal ("Gate A") and off-topic. Prints what each path returned and the cosine
spread between rank 1 and rank N, the number alpha-core's `check_rag` reported.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vmp.rag.index import Index

CORPUS = {
    "streaming.md": """# Streaming

The sentence segmenter splits the LLM token stream into sentences.
Each sentence is sent to TTS as soon as it is complete.
The playback stage records response_ms, the time to first audio.
Streaming reduces time to first audio because TTS starts on the first sentence.
""",
    "gates.md": """# Release gates

Every candidate model passes a release gate before promotion.
The WER gate compares golden set WER against the production model.
The latency gate checks the time-to-first-audio p95.
The error gate checks the error rate over the last evaluation run.
The smoke gate checks that the model loads and answers one turn.

Gate A verifies the adapter checksum before export.
Operational note: rotate logs, archive traces, prune old bundles.
Operational note: the export job writes a manifest next to the bundle.
""",
    "stt.md": """# Speech to text

The STT stage transcribes audio with Whisper.
faster-whisper runs on CPU and mlx-whisper runs on the M4 GPU.
Wake-word correction uses Jaro-Winkler on the transcript.
""",
    "tts.md": """# Text to speech

Kokoro synthesises each sentence. The real-time factor is below one.
Audio is written to the playback queue.
""",
}

QUERIES = [
    ("on-topic", "how does streaming reduce time to first audio"),
    ("rare literal", "what does Gate A check"),
    ("off-topic", "what is the capital of France"),
]


def main() -> int:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus"
        corpus.mkdir()
        for name, text in CORPUS.items():
            (corpus / name).write_text(text)

        index = Index(chunk_lines=4, overlap=1, top_k=3, relevance_floor=0.15)
        report = index.build(corpus)
        print(
            f"indexed {report['documents']} documents, {report['chunks_total']} chunks, "
            f"{report['graph_nodes']} graph nodes, {report['graph_edges']} edges "
            f"(embedder {report['embedder']}, dim {report['dim']})"
        )

        for label, query in QUERIES:
            res = index.query(query)
            print(f"\n[{label}] {query}")
            print(f"  rare terms: {res.rare_terms}  entities: {res.entities}")
            if not res.chunks:
                print(f"  no relevant context ({res.floored} candidates below the floor)")
                continue
            for c, s, p in zip(res.chunks, res.scores, res.paths, strict=True):
                print(f"  {s:.4f}  cos={res.cosines[c.id]:.3f}  {p:8s}  {c.id}")
            print("  context pack:")
            for line in res.context_pack(300).splitlines():
                print(f"    {line}")

        print("\ncosine spread, rank 1 vs rank N (alpha-core check_rag style):")
        for q in index.check([q for _, q in QUERIES], k=8)["queries"]:
            n = q["n"]
            print(
                f"  {q['query']!r}: rank1={q['rank1_cosine']:.3f} "
                f"rank{n}={q[f'rank{n}_cosine']:.3f} spread={q['spread']:.3f} "
                f"returned_after_floor={q['returned_after_floor']}"
            )
    print(f"\ndone in {time.perf_counter() - t0:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
