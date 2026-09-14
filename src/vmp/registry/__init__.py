"""Model registry: versions, lifecycle stages, lineage. File backend by default.

`Registry` is the Protocol; `FileRegistry` is the standard-library implementation
used in tests and demos; `MlflowRegistry` (in `mlflow_adapter`) imports `mlflow`
lazily.
"""

from __future__ import annotations

from vmp.registry.store import (
    ALLOWED_TRANSITIONS,
    FileRegistry,
    Registry,
    RegistryError,
    transition_error,
)

__all__ = ["ALLOWED_TRANSITIONS", "FileRegistry", "Registry", "RegistryError", "transition_error"]
