"""Feature store for real-time voice sessions.

Views are defined in code (`views.py`), rows live offline (JSONL or Parquet)
and online (in-memory or Redis), a point-in-time join keeps training honest,
and a stream updater folds stage traces into per-session features as they
happen. Standard library only; adapters import their backend lazily.
"""

from __future__ import annotations

from vmp.features.feast_export import feast_source, write_feast_repo
from vmp.features.materialize import materialize
from vmp.features.offline import (
    JsonlOfflineStore,
    OfflineStore,
    ParquetOfflineStore,
    latest_per_entity,
    point_in_time_join,
)
from vmp.features.online import InMemoryOnlineStore, OnlineStore, RedisOnlineStore
from vmp.features.stream import (
    JsonlSource,
    KafkaSource,
    ListSource,
    SessionFeatureUpdater,
    SessionState,
)
from vmp.features.views import (
    DEVICE_FEATURES,
    SESSION_FEATURES,
    SPEAKER_FEATURES,
    Feature,
    FeatureView,
    get_view,
    list_views,
    register_view,
    unregister_view,
    views_from_config,
)

__all__ = [
    "DEVICE_FEATURES",
    "SESSION_FEATURES",
    "SPEAKER_FEATURES",
    "Feature",
    "FeatureView",
    "InMemoryOnlineStore",
    "JsonlOfflineStore",
    "JsonlSource",
    "KafkaSource",
    "ListSource",
    "OfflineStore",
    "OnlineStore",
    "ParquetOfflineStore",
    "RedisOnlineStore",
    "SessionFeatureUpdater",
    "SessionState",
    "feast_source",
    "get_view",
    "latest_per_entity",
    "list_views",
    "materialize",
    "point_in_time_join",
    "register_view",
    "unregister_view",
    "views_from_config",
    "write_feast_repo",
]
