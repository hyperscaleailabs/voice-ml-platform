"""Line-based chunking and directory loading.

Chunks are windows of `chunk_lines` lines that advance by `chunk_lines - overlap`,
so neighbouring chunks share `overlap` lines. Line numbers are 1-based and
inclusive, which matches how editors and stack traces refer to source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from vmp.types import Chunk, Document

DEFAULT_INCLUDE = ("**/*.md", "**/*.txt", "**/*.py", "**/*.toml")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id(document_id: str, start_line: int, end_line: int) -> str:
    return f"{document_id}:{start_line}-{end_line}"


def chunk_document(doc: Document, chunk_lines: int = 20, overlap: int = 4) -> list[Chunk]:
    """Split one document into overlapping line windows.

    Windows that would be entirely blank are skipped. The last window is shorter
    when the document does not divide evenly; no window is emitted twice.
    """
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if overlap < 0 or overlap >= chunk_lines:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_lines")
    lines = doc.text.splitlines()
    n = len(lines)
    step = chunk_lines - overlap
    out: list[Chunk] = []
    i = 0
    while i < n:
        end = min(i + chunk_lines, n)
        text = "\n".join(lines[i:end])
        if text.strip():
            out.append(
                Chunk(
                    id=chunk_id(doc.id, i + 1, end),
                    document_id=doc.id,
                    start_line=i + 1,
                    end_line=end,
                    text=text,
                )
            )
        if end >= n:
            break
        i += step
    return out


def load_documents(
    root: str | Path,
    include: tuple[str, ...] = DEFAULT_INCLUDE,
    exclude_dirs: tuple[str, ...] = (".git", ".venv", "__pycache__", "node_modules"),
) -> list[Document]:
    """Read every file under `root` matching one of the `include` globs.

    The document id is the POSIX relative path; `sha` is sha256 of the bytes.
    Files that are not valid UTF-8 are skipped. Output is sorted by id.
    """
    base = Path(root)
    seen: dict[str, Document] = {}
    for pattern in include:
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(base)
            if any(part in exclude_dirs for part in rel.parts):
                continue
            rid = rel.as_posix()
            if rid in seen:
                continue
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            seen[rid] = Document(
                id=rid,
                path=str(path),
                text=text,
                sha=hashlib.sha256(raw).hexdigest(),
                meta={"bytes": len(raw), "lines": text.count("\n") + 1},
            )
    return [seen[k] for k in sorted(seen)]


@dataclass
class IncrementalPlan:
    """Which documents need (re)indexing given the previously indexed shas."""

    added: list[Document] = field(default_factory=list)
    changed: list[Document] = field(default_factory=list)
    unchanged: list[Document] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            "removed": len(self.removed),
        }


def plan_incremental(previous: dict[str, str], docs: list[Document]) -> IncrementalPlan:
    """Compare `previous` (document id -> sha) with freshly loaded `docs`.

    Unchanged documents keep their chunks; changed ones are deleted and re-added;
    ids missing from `docs` are removed.
    """
    plan = IncrementalPlan()
    current = {d.id for d in docs}
    for doc in docs:
        old = previous.get(doc.id)
        if old is None:
            plan.added.append(doc)
        elif old == doc.sha:
            plan.unchanged.append(doc)
        else:
            plan.changed.append(doc)
    plan.removed = sorted(k for k in previous if k not in current)
    return plan


__all__ = [
    "DEFAULT_INCLUDE",
    "IncrementalPlan",
    "chunk_document",
    "chunk_id",
    "load_documents",
    "plan_incremental",
    "sha256_text",
]
