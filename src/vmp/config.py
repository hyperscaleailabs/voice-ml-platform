"""Configuration: TOML files under `configs/`, overridden by `VMP_*` environment variables.

Standard library only (`tomllib`). No YAML in the core package.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_PREFIX = "VMP_"


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one TOML file. Raises FileNotFoundError with the path in the message."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("rb") as fh:
        return tomllib.load(fh)


def env_overrides(prefix: str = ENV_PREFIX) -> dict[str, str]:
    """`VMP_REGISTRY_ROOT=/x` -> {"registry_root": "/x"}. Keys are lower-cased."""
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            out[key[len(prefix) :].lower()] = value
    return out


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge. `override` wins on conflicts; nested dicts are merged."""
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


@dataclass
class Settings:
    """Process-wide settings with sensible local defaults.

    Everything is a path or a plain string so the object can be printed and
    serialised without surprises. Environment variables win over the config
    file, which wins over these defaults.
    """

    root: Path = field(default_factory=lambda: Path(os.getcwd()))
    registry_root: Path = field(default_factory=lambda: Path(".vmp/registry"))
    feature_store_root: Path = field(default_factory=lambda: Path(".vmp/features"))
    trace_path: Path = field(default_factory=lambda: Path(".vmp/trace.jsonl"))
    rag_index_root: Path = field(default_factory=lambda: Path(".vmp/rag"))
    edge_bundle_root: Path = field(default_factory=lambda: Path(".vmp/edge"))
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sources(cls, config_path: str | Path | None = None) -> Settings:
        data: dict[str, Any] = {}
        if config_path is not None:
            data = load_config(config_path)
        env = env_overrides()
        s = cls()
        names = (
            "registry_root",
            "feature_store_root",
            "trace_path",
            "rag_index_root",
            "edge_bundle_root",
        )
        for name in names:
            value = env.get(name, data.get("paths", {}).get(name))
            if value is not None:
                setattr(s, name, Path(value))
        s.extra = {k: v for k, v in data.items() if k != "paths"}
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "registry_root": str(self.registry_root),
            "feature_store_root": str(self.feature_store_root),
            "trace_path": str(self.trace_path),
            "rag_index_root": str(self.rag_index_root),
            "edge_bundle_root": str(self.edge_bundle_root),
            "extra": self.extra,
        }


__all__ = ["ENV_PREFIX", "Settings", "deep_merge", "env_overrides", "load_config"]
