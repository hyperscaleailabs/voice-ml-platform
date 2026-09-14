"""`MlflowRegistry`: the `Registry` protocol on the MLflow Model Registry.

MLflow's own stages are deprecated in favour of aliases, so each of this
platform's stages is stored as a registered-model alias (`candidate`,
`staging`, `production`, `retired`) and the lineage fields become model-version
tags. `mlflow` is imported inside the methods; importing this module needs
nothing but the standard library.
"""

from __future__ import annotations

from typing import Any

from vmp.registry.store import RegistryError, transition_error
from vmp.types import STAGE_CANDIDATE, STAGE_PRODUCTION, STAGE_RETIRED, ModelArtifact

LINEAGE_TAGS = ("base_model", "adapter_path", "config_hash", "data_hash", "git_sha")


def to_mlflow_tags(artifact: ModelArtifact) -> dict[str, str]:
    """Lineage as flat string tags. Metrics are prefixed `metric.`."""
    tags = {f"vmp.{k}": str(getattr(artifact, k) or "") for k in LINEAGE_TAGS}
    tags["vmp.stage"] = artifact.stage
    for k, v in artifact.metrics.items():
        tags[f"metric.{k}"] = str(v)
    return tags


def from_mlflow_version(name: str, mv: Any) -> ModelArtifact:
    """Build a `ModelArtifact` from an `mlflow.entities.ModelVersion`."""
    tags = dict(getattr(mv, "tags", {}) or {})
    metrics = {k[len("metric.") :]: float(v) for k, v in tags.items() if k.startswith("metric.")}
    return ModelArtifact(
        name=name,
        version=str(mv.version),
        stage=tags.get("vmp.stage", STAGE_CANDIDATE),
        base_model=tags.get("vmp.base_model", ""),
        adapter_path=tags.get("vmp.adapter_path") or None,
        config_hash=tags.get("vmp.config_hash", ""),
        data_hash=tags.get("vmp.data_hash", ""),
        git_sha=tags.get("vmp.git_sha") or None,
        metrics=metrics,
        created_at=float(getattr(mv, "creation_timestamp", 0) or 0) / 1000.0,
    )


class MlflowRegistry:
    """Registry backed by an MLflow tracking server. Requires the `registry` extra."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri
        self._client: Any = None

    def _mlflow(self) -> Any:
        import mlflow

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self._client is None:
            self._client = mlflow.MlflowClient()
        return self._client

    def register(self, artifact: ModelArtifact) -> ModelArtifact:
        if artifact.stage != STAGE_CANDIDATE:
            raise RegistryError(
                f"new artifacts must be {STAGE_CANDIDATE!r}, got {artifact.stage!r}"
            )
        client = self._mlflow()
        try:
            client.get_registered_model(artifact.name)
        except Exception:  # mlflow raises its own RestException here
            client.create_registered_model(artifact.name)
        source = artifact.adapter_path or artifact.base_model
        mv = client.create_model_version(
            name=artifact.name, source=source, tags=to_mlflow_tags(artifact)
        )
        client.set_registered_model_alias(artifact.name, STAGE_CANDIDATE, mv.version)
        stored = client.get_model_version(artifact.name, mv.version)
        return from_mlflow_version(artifact.name, stored)

    def get(self, name: str, version: str) -> ModelArtifact:
        client = self._mlflow()
        return from_mlflow_version(name, client.get_model_version(name, version))

    def list(self, name: str | None = None) -> list[ModelArtifact]:
        client = self._mlflow()
        query = f"name = '{name}'" if name else ""
        versions = client.search_model_versions(query)
        return [from_mlflow_version(mv.name, mv) for mv in versions]

    def promote(self, name: str, version: str, stage: str) -> ModelArtifact:
        client = self._mlflow()
        current = self.get(name, version)
        problem = transition_error(current.stage, stage)
        if problem:
            raise RegistryError(f"{name}:{version}: {problem}")
        if stage == STAGE_PRODUCTION:
            for other in self.list(name):
                if other.stage == STAGE_PRODUCTION and other.version != version:
                    client.set_model_version_tag(name, other.version, "vmp.stage", STAGE_RETIRED)
        client.set_model_version_tag(name, version, "vmp.stage", stage)
        client.set_registered_model_alias(name, stage, version)
        return self.get(name, version)


__all__ = ["LINEAGE_TAGS", "MlflowRegistry", "from_mlflow_version", "to_mlflow_tags"]
