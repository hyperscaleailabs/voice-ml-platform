"""Shared types used across every subsystem. Standard library only.

These are the nouns of the platform. Modules exchange these objects rather than
ad-hoc dicts, and serialise them with `to_dict` / `from_dict` so JSONL files,
registries and traces stay readable without the package installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

# Model lifecycle stages. Plain strings so they serialise without an Enum.
STAGE_CANDIDATE = "candidate"
STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_RETIRED = "retired"
STAGES = (STAGE_CANDIDATE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_RETIRED)

# Stages of one voice turn, in pipeline order. Names describe the role, not the
# engine filling it, so traces stay comparable when a backend is swapped.
TURN_STAGES = ("listen", "stt", "retrieve", "llm", "segment.emit", "tts", "playback", "turn")


class _Serialisable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in names})  # type: ignore[call-arg]


@dataclass(frozen=True)
class Utterance(_Serialisable):
    """One thing a speaker said. `text` is the transcript or the reference text."""

    id: str
    text: str
    audio_path: str | None = None
    speaker: str | None = None
    duration_s: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn(_Serialisable):
    """One exchange inside a session. `stages` holds measured milliseconds per stage."""

    session_id: str
    turn: int
    user: Utterance
    assistant_text: str | None = None
    retrieved: list[str] = field(default_factory=list)
    stages: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["user"] = self.user.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        d = dict(data)
        d["user"] = Utterance.from_dict(d["user"])
        return cls(**d)


@dataclass
class Session(_Serialisable):
    """A conversation. Holds the system prompt and the ordered turns."""

    id: str
    created_at: float
    system_prompt: str = ""
    turns: list[Turn] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "system_prompt": self.system_prompt,
            "turns": [t.to_dict() for t in self.turns],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            system_prompt=data.get("system_prompt", ""),
            turns=[Turn.from_dict(t) for t in data.get("turns", [])],
            meta=dict(data.get("meta", {})),
        )


@dataclass(frozen=True)
class PreferencePair(_Serialisable):
    """A DPO training example: for `prompt`, `chosen` is preferred over `rejected`."""

    prompt: str
    chosen: str
    rejected: str
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureRow(_Serialisable):
    """One feature vector for an entity at an event time (point-in-time semantics)."""

    entity_id: str
    event_ts: float
    values: dict[str, float | int | str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Document(_Serialisable):
    """A source document for retrieval. `sha` makes re-indexing incremental."""

    id: str
    path: str
    text: str
    sha: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk(_Serialisable):
    """A retrievable slice of a document, with an optional embedding."""

    id: str
    document_id: str
    start_line: int
    end_line: int
    text: str
    embedding: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["embedding"] = list(self.embedding) if self.embedding is not None else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        d = dict(data)
        if d.get("embedding") is not None:
            d["embedding"] = tuple(d["embedding"])
        return cls(**d)


@dataclass(frozen=True)
class ModelArtifact(_Serialisable):
    """A registered model version with its lineage."""

    name: str
    version: str
    stage: str
    base_model: str
    adapter_path: str | None
    config_hash: str
    data_hash: str
    git_sha: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass(frozen=True)
class EvalResult(_Serialisable):
    """The outcome of one evaluation over `n` items."""

    name: str
    metrics: dict[str, float]
    n: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision(_Serialisable):
    """A release-gate verdict, with every reason it passed or failed."""

    passed: bool
    reasons: list[str]
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "results": [r.to_dict() for r in self.results],
        }


__all__ = [
    "STAGES",
    "STAGE_CANDIDATE",
    "STAGE_PRODUCTION",
    "STAGE_RETIRED",
    "STAGE_STAGING",
    "TURN_STAGES",
    "Chunk",
    "Document",
    "EvalResult",
    "FeatureRow",
    "GateDecision",
    "ModelArtifact",
    "PreferencePair",
    "Session",
    "Turn",
    "Utterance",
]
