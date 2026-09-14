"""voice-ml-platform: an ML platform for voice agents, from research to production.

The core package is standard library only. Heavy backends (Ray, TRL, PEFT, vLLM,
pgvector, Neo4j, Feast, ONNX) are optional adapters behind Protocols and are
imported lazily where they are used.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
