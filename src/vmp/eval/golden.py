"""Golden set: reference utterances -> synthesised clips -> transcripts -> WER.

The file format is one `Utterance.to_dict()` per line with `meta.category`
(compatible with `vmp.data.synthetic`). Backends are duck-typed: a TTS has
`synthesize(text) -> bytes`, an STT has `transcribe(audio: bytes | str) -> str`.
The stubs here encode text as the "audio" so the pipeline runs without models.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from vmp.eval.wer import align, corpus_wer, normalise
from vmp.types import EvalResult, Utterance


class TTSLike(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class STTLike(Protocol):
    def transcribe(self, audio: bytes | str) -> str: ...


class StubTTS:
    """Writes the UTF-8 text as the clip so a stub STT can read it back."""

    def synthesize(self, text: str) -> bytes:
        return text.encode("utf-8")


def _load_audio(audio: bytes | str) -> bytes:
    if isinstance(audio, bytes):
        return audio
    return Path(audio).read_bytes()


class IdentitySTT:
    """Returns the text a `StubTTS` encoded. WER is 0 by construction."""

    def transcribe(self, audio: bytes | str) -> str:
        return _load_audio(audio).decode("utf-8", errors="replace")


class NoisySTT:
    """Deterministic word-drop and substitution noise, seeded per transcription."""

    def __init__(self, drop_rate: float = 0.1, sub_rate: float = 0.0, seed: int = 0) -> None:
        self.drop_rate = drop_rate
        self.sub_rate = sub_rate
        self.seed = seed

    def transcribe(self, audio: bytes | str) -> str:
        text = _load_audio(audio).decode("utf-8", errors="replace")
        rng = random.Random(f"{self.seed}:{text}")
        out: list[str] = []
        for word in text.split():
            r = rng.random()
            if r < self.drop_rate:
                continue
            if r < self.drop_rate + self.sub_rate:
                out.append(word[::-1])
            else:
                out.append(word)
        return " ".join(out)


def _category(u: Utterance) -> str:
    return str(u.meta.get("category", "uncategorised"))


@dataclass
class GoldenSet:
    """Reference utterances grouped by `meta.category`."""

    items: list[Utterance] = field(default_factory=list)
    name: str = "golden"

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> GoldenSet:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"golden set not found: {p}")
        items: list[Utterance] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    items.append(Utterance.from_dict(json.loads(s)))
        return cls(items=items, name=name or p.stem)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for u in self.items:
                fh.write(json.dumps(u.to_dict(), ensure_ascii=False) + "\n")
        return p

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self.items)

    def categories(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for u in self.items:
            out[_category(u)] = out.get(_category(u), 0) + 1
        return dict(sorted(out.items()))

    def build_audio(
        self,
        tts: TTSLike,
        out_dir: str | Path,
        *,
        dry_run: bool = False,
        ext: str = "wav",
    ) -> dict[str, Any]:
        """Synthesise one clip per utterance. Returns a manifest dict.

        Manifest: `{"name", "out_dir", "dry_run", "items": [{id, text, category,
        audio_path}]}`. In dry-run mode no clip is written and `audio_path` is
        the path that would be used.
        """
        out = Path(out_dir)
        items: list[dict[str, Any]] = []
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
        for u in self.items:
            path = out / f"{u.id}.{ext}"
            if not dry_run:
                path.write_bytes(tts.synthesize(u.text))
            items.append(
                {
                    "id": u.id,
                    "text": u.text,
                    "category": _category(u),
                    "audio_path": str(path),
                }
            )
        return {"name": self.name, "out_dir": str(out), "dry_run": dry_run, "items": items}


def score_asr(
    stt: STTLike,
    manifest: dict[str, Any],
    *,
    dry_run: bool = False,
    numbers: bool = False,
) -> EvalResult:
    """Transcribe every manifest item and score WER overall and per category.

    In dry-run mode nothing is transcribed; the result has `n` items and a
    `details.plan` describing what would run.
    """
    items = list(manifest.get("items", []))
    name = str(manifest.get("name", "golden"))
    if dry_run or manifest.get("dry_run"):
        return EvalResult(
            name=f"{name}_asr",
            metrics={},
            n=len(items),
            details={
                "dry_run": True,
                "plan": {
                    "stt": type(stt).__name__,
                    "categories": sorted({i["category"] for i in items}),
                },
                "per_category": {},
            },
        )
    per_item: list[dict[str, Any]] = []
    refs: list[str] = []
    hyps: list[str] = []
    by_cat: dict[str, tuple[list[str], list[str]]] = {}
    for item in items:
        hyp = stt.transcribe(item["audio_path"])
        ref = item["text"]
        a = align(normalise(ref, numbers=numbers), normalise(hyp, numbers=numbers))
        per_item.append(
            {"id": item["id"], "category": item["category"], "hyp": hyp, **a.to_dict()}
        )
        refs.append(ref)
        hyps.append(hyp)
        r, h = by_cat.setdefault(item["category"], ([], []))
        r.append(ref)
        h.append(hyp)
    overall = corpus_wer(refs, hyps, numbers=numbers)
    per_category = {
        cat: corpus_wer(r, h, numbers=numbers) for cat, (r, h) in sorted(by_cat.items())
    }
    return EvalResult(
        name=f"{name}_asr",
        metrics={
            "wer": overall["wer"],
            "cer": overall["cer"],
            "exact_match": overall["exact_match"],
        },
        n=len(items),
        details={
            "dry_run": False,
            "stt": type(stt).__name__,
            "per_category": per_category,
            "items": per_item,
        },
    )


def utterances_from_texts(
    texts: Iterable[tuple[str, str]], prefix: str = "g"
) -> list[Utterance]:
    """Helper for demos and tests: `[(category, text), ...]` -> utterances."""
    return [
        Utterance(id=f"{prefix}{i:04d}", text=text, meta={"category": cat})
        for i, (cat, text) in enumerate(texts)
    ]


__all__ = [
    "GoldenSet",
    "IdentitySTT",
    "NoisySTT",
    "STTLike",
    "StubTTS",
    "TTSLike",
    "score_asr",
    "utterances_from_texts",
]
