"""Edge: export pipeline, signed bundles, runtime policy, offline turn loop.

Standard library only at import time. ONNX, llama.cpp and MLX converters are
imported lazily inside `export` when a real (non dry-run) export is requested.
"""

from __future__ import annotations

from vmp.edge.bundle import EdgeBundle, build_bundle, diff_bundles, verify_bundle
from vmp.edge.export import ExportPlan, export
from vmp.edge.policy import EdgePolicy, PolicyViolation, enforce
from vmp.edge.runtime import CloudFallback, EdgeRuntime, EdgeTurn

__all__ = [
    "CloudFallback",
    "EdgeBundle",
    "EdgePolicy",
    "EdgeRuntime",
    "EdgeTurn",
    "ExportPlan",
    "PolicyViolation",
    "build_bundle",
    "diff_bundles",
    "enforce",
    "export",
    "verify_bundle",
]
