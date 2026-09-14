"""`Registry` protocol and the JSON-on-disk `FileRegistry`.

Lifecycle rules, enforced by every backend through `transition_error`:

- new artifacts enter as `candidate`;
- `candidate -> staging -> production`, one step at a time;
- at most one `production` version per name: promoting a new version retires
  the current one and records the swap in the event log;
- `retired` is terminal, and any non-retired version may be retired.

Lineage comes from `ModelArtifact`: `config_hash` (the training plan),
`data_hash` (the training split) and `git_sha`.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any, Protocol

from vmp.types import (
    STAGE_CANDIDATE,
    STAGE_PRODUCTION,
    STAGE_RETIRED,
    STAGE_STAGING,
    STAGES,
    ModelArtifact,
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    STAGE_CANDIDATE: frozenset({STAGE_STAGING, STAGE_RETIRED}),
    STAGE_STAGING: frozenset({STAGE_PRODUCTION, STAGE_RETIRED}),
    STAGE_PRODUCTION: frozenset({STAGE_RETIRED}),
    STAGE_RETIRED: frozenset(),
}

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RegistryError(ValueError):
    """A registry rule was violated. The message says which."""


def transition_error(current: str, target: str) -> str | None:
    """Why `current -> target` is not allowed, or None when it is."""
    if target not in STAGES:
        return f"unknown stage {target!r}; expected one of {STAGES}"
    if current == STAGE_RETIRED:
        return "retired is terminal"
    if target == current:
        return f"already in stage {current!r}"
    if target not in ALLOWED_TRANSITIONS[current]:
        allowed = sorted(ALLOWED_TRANSITIONS[current])
        return f"cannot move {current!r} -> {target!r}; allowed: {allowed}"
    return None


class Registry(Protocol):
    """What every registry backend provides."""

    def register(self, artifact: ModelArtifact) -> ModelArtifact: ...

    def get(self, name: str, version: str) -> ModelArtifact: ...

    def list(self, name: str | None = None) -> list[ModelArtifact]: ...

    def promote(self, name: str, version: str, stage: str) -> ModelArtifact: ...


def _check_id(kind: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE.match(value):
        raise RegistryError(f"{kind} {value!r} must match {_SAFE.pattern}")


class FileRegistry:
    """JSON files under a root directory.

    Layout: `<root>/index.json` holds `{name: {version: stage}}` plus an event
    log; `<root>/<name>/<version>.json` holds the artifact. The index is
    rewritten atomically (write to a temp file, then replace) on every change.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write_index({"artifacts": {}, "events": []})

    # -- storage --------------------------------------------------------------

    def _read_index(self) -> dict[str, Any]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, index: dict[str, Any]) -> None:
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.index_path)

    def _artifact_path(self, name: str, version: str) -> Path:
        return self.root / name / f"{version}.json"

    def _write_artifact(self, artifact: ModelArtifact) -> None:
        p = self._artifact_path(artifact.name, artifact.version)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def _log(self, index: dict[str, Any], **event: Any) -> None:
        index["events"].append({"ts": time.time(), **event})

    # -- Registry protocol ----------------------------------------------------

    def register(self, artifact: ModelArtifact) -> ModelArtifact:
        """Store a new `candidate`. Names and versions are unique per registry."""
        _check_id("name", artifact.name)
        _check_id("version", artifact.version)
        if artifact.stage != STAGE_CANDIDATE:
            raise RegistryError(
                f"new artifacts must be registered as {STAGE_CANDIDATE!r}, got {artifact.stage!r}"
            )
        if not artifact.config_hash or not artifact.data_hash:
            raise RegistryError("config_hash and data_hash are required lineage fields")
        index = self._read_index()
        versions = index["artifacts"].setdefault(artifact.name, {})
        if artifact.version in versions:
            raise RegistryError(f"{artifact.name}:{artifact.version} already registered")
        stored = artifact
        if not artifact.created_at:
            stored = dataclasses.replace(artifact, created_at=time.time())
        self._write_artifact(stored)
        versions[artifact.version] = STAGE_CANDIDATE
        self._log(index, name=artifact.name, version=artifact.version, to=STAGE_CANDIDATE)
        self._write_index(index)
        return stored

    def get(self, name: str, version: str) -> ModelArtifact:
        p = self._artifact_path(name, version)
        if not p.is_file():
            raise KeyError(f"{name}:{version} not in registry {self.root}")
        return ModelArtifact.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list(self, name: str | None = None) -> list[ModelArtifact]:
        """All versions, newest first, optionally for one name."""
        index = self._read_index()
        names = [name] if name is not None else sorted(index["artifacts"])
        out: list[ModelArtifact] = []
        for n in names:
            for v in index["artifacts"].get(n, {}):
                out.append(self.get(n, v))
        return sorted(out, key=lambda a: (-a.created_at, a.name, a.version))

    def latest(self, name: str, stage: str | None = None) -> ModelArtifact | None:
        """Newest version of `name`, optionally restricted to one stage."""
        for a in self.list(name):
            if stage is None or a.stage == stage:
                return a
        return None

    def promote(self, name: str, version: str, stage: str) -> ModelArtifact:
        """Move one version to `stage` under the lifecycle rules."""
        current = self.get(name, version)
        problem = transition_error(current.stage, stage)
        if problem:
            raise RegistryError(f"{name}:{version}: {problem}")
        index = self._read_index()
        versions = index["artifacts"][name]
        if stage == STAGE_PRODUCTION:
            for other_version, other_stage in list(versions.items()):
                if other_stage == STAGE_PRODUCTION and other_version != version:
                    other = self.get(name, other_version)
                    demoted = dataclasses.replace(other, stage=STAGE_RETIRED)
                    self._write_artifact(demoted)
                    versions[other_version] = STAGE_RETIRED
                    self._log(
                        index,
                        name=name,
                        version=other_version,
                        **{"from": STAGE_PRODUCTION},
                        to=STAGE_RETIRED,
                        reason=f"replaced by {version}",
                    )
        promoted = dataclasses.replace(current, stage=stage)
        self._write_artifact(promoted)
        versions[version] = stage
        self._log(index, name=name, version=version, **{"from": current.stage}, to=stage)
        self._write_index(index)
        return promoted

    def events(self) -> list[dict[str, Any]]:
        return list(self._read_index()["events"])


__all__ = ["ALLOWED_TRANSITIONS", "FileRegistry", "Registry", "RegistryError", "transition_error"]
