"""Feature views defined in code, plus a registry.

A `FeatureView` names an entity, a TTL and the features it carries. The same
definitions drive the offline store, the online store, materialisation and the
Feast export, so there is one place to change a feature's type or TTL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DTYPES = ("float", "int", "str")

# Trace stages the stream updater reads. Kept here so `views` stays stdlib-only.
SOURCE_TRACES = "traces"
SOURCE_EVAL = "eval"
SOURCE_EDGE = "edge"


@dataclass(frozen=True)
class Feature:
    name: str
    dtype: str = "float"
    description: str = ""

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"feature {self.name!r}: dtype must be one of {DTYPES}")


@dataclass(frozen=True)
class FeatureView:
    """A named set of features for one entity type, valid for `ttl_s` after a row's timestamp."""

    name: str
    entity: str
    ttl_s: float
    features: tuple[Feature, ...]
    source: str = SOURCE_TRACES
    description: str = ""

    def __post_init__(self) -> None:
        if self.ttl_s <= 0:
            raise ValueError(f"view {self.name!r}: ttl_s must be positive")
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"view {self.name!r}: duplicate feature names")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    def empty_values(self) -> dict[str, Any]:
        return dict.fromkeys(self.feature_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity": self.entity,
            "ttl_s": self.ttl_s,
            "source": self.source,
            "description": self.description,
            "features": [
                {"name": f.name, "dtype": f.dtype, "description": f.description}
                for f in self.features
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureView:
        return cls(
            name=data["name"],
            entity=data["entity"],
            ttl_s=float(data["ttl_s"]),
            features=tuple(
                Feature(f["name"], f.get("dtype", "float"), f.get("description", ""))
                for f in data.get("features", [])
            ),
            source=data.get("source", SOURCE_TRACES),
            description=data.get("description", ""),
        )


SESSION_FEATURES = FeatureView(
    name="session_features",
    entity="session",
    ttl_s=3600,
    source=SOURCE_TRACES,
    description="Rolling per-session features updated from stage traces.",
    features=(
        Feature("turn_count", "int", "completed turns in the session"),
        Feature("avg_user_utterance_s", "float", "mean listen-stage duration in seconds"),
        Feature("avg_response_ms", "float", "mean time to first audio (playback response_ms)"),
        Feature("last_intent", "str", "intent of the most recent turn"),
        Feature("accent_profile", "str", "accent tag reported by STT"),
    ),
)

SPEAKER_FEATURES = FeatureView(
    name="speaker_features",
    entity="speaker",
    ttl_s=7 * 24 * 3600,
    source=SOURCE_EVAL,
    description="Seven-day quality aggregates per speaker, from evaluation runs.",
    features=(
        Feature("wer_7d", "float", "word error rate over the trailing seven days"),
        Feature("exact_match_rate_7d", "float", "exact transcript match rate, seven days"),
        Feature("wake_word_corrections_7d", "int", "wake-word corrections applied, seven days"),
    ),
)

DEVICE_FEATURES = FeatureView(
    name="device_features",
    entity="device",
    ttl_s=24 * 3600,
    source=SOURCE_EDGE,
    description="Edge device state and one-day latency percentile.",
    features=(
        Feature("edge_bundle_version", "str", "installed EdgeBundle version"),
        Feature("p95_ttfa_ms_24h", "float", "p95 time to first audio over 24 hours"),
    ),
)

_REGISTRY: dict[str, FeatureView] = {}


def register_view(view: FeatureView, replace: bool = False) -> FeatureView:
    if view.name in _REGISTRY and not replace and _REGISTRY[view.name] != view:
        raise ValueError(f"view {view.name!r} already registered with a different definition")
    _REGISTRY[view.name] = view
    return view


def unregister_view(name: str) -> bool:
    """Remove a view from the registry. Returns whether it was present."""
    return _REGISTRY.pop(name, None) is not None


def get_view(name: str) -> FeatureView:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown feature view {name!r}; known: {sorted(_REGISTRY)}") from None


def list_views() -> list[FeatureView]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def views_from_config(config: dict[str, Any], register: bool = True) -> list[FeatureView]:
    """Build views from a `configs/features.toml` dict (`[[views]]` array of tables)."""
    out: list[FeatureView] = []
    for item in config.get("views", []):
        v = FeatureView.from_dict(item)
        if register:
            register_view(v, replace=True)
        out.append(v)
    return out


@dataclass
class ViewSet:
    """A small helper for tests and CLIs that want an isolated registry."""

    views: dict[str, FeatureView] = field(default_factory=dict)

    def add(self, view: FeatureView) -> FeatureView:
        self.views[view.name] = view
        return view

    def get(self, name: str) -> FeatureView:
        return self.views[name]


for _v in (SESSION_FEATURES, SPEAKER_FEATURES, DEVICE_FEATURES):
    register_view(_v)

__all__ = [
    "DEVICE_FEATURES",
    "DTYPES",
    "SESSION_FEATURES",
    "SOURCE_EDGE",
    "SOURCE_EVAL",
    "SOURCE_TRACES",
    "SPEAKER_FEATURES",
    "Feature",
    "FeatureView",
    "ViewSet",
    "get_view",
    "list_views",
    "register_view",
    "unregister_view",
    "views_from_config",
]
