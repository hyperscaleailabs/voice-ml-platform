"""JSONL input/output, atomic writes, and content hashes.

One JSON object per line, UTF-8, no trailing whitespace games. Hashes are
SHA-256 over canonical JSON (sorted keys, no spaces) so the same record hashes
the same way regardless of insertion order.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from vmp.types import Utterance


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-blank line. Raises FileNotFoundError with the path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"jsonl not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no}: invalid JSON: {e.msg}") from e


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write `text` to a temp file in the same directory, then rename over `path`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, p)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return p


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    atomic: bool = True,
    append: bool = False,
) -> int:
    """Write records as JSONL. Returns the number written.

    `atomic=True` (default) writes to a temp file and renames, so a reader never
    sees a half-written file. `append=True` disables the atomic path and appends.
    """
    p = Path(path)
    if append:
        p.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with p.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        return n
    lines = [json.dumps(rec, ensure_ascii=False) for rec in records]
    text = "\n".join(lines) + ("\n" if lines else "")
    if atomic:
        atomic_write_text(p, text)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return len(lines)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_record(record: dict[str, Any]) -> str:
    """Hex SHA-256 of one record's canonical JSON."""
    return hashlib.sha256(_canonical(record)).hexdigest()


def sha256_records(records: Iterable[dict[str, Any]]) -> str:
    """Hex SHA-256 over a sequence of records (order matters). Used as a data hash."""
    h = hashlib.sha256()
    for rec in records:
        h.update(_canonical(rec))
        h.update(b"\n")
    return h.hexdigest()


def read_utterances(path: str | Path) -> list[Utterance]:
    """Read a JSONL transcript into `Utterance` objects.

    Each line needs at least `text`. A missing `id` becomes `<stem>-<line index>`.
    """
    p = Path(path)
    out: list[Utterance] = []
    for i, rec in enumerate(read_jsonl(p)):
        if "text" not in rec:
            raise ValueError(f"{p}: record {i} has no 'text'")
        rec = dict(rec)
        rec.setdefault("id", f"{p.stem}-{i:06d}")
        out.append(Utterance.from_dict(rec))
    return out


def write_utterances(path: str | Path, utterances: Iterable[Utterance]) -> int:
    return write_jsonl(path, (u.to_dict() for u in utterances))


__all__ = [
    "atomic_write_text",
    "read_jsonl",
    "read_utterances",
    "sha256_file",
    "sha256_record",
    "sha256_records",
    "write_jsonl",
    "write_utterances",
]
