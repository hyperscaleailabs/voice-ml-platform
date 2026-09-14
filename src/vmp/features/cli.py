"""`vmp features ...` subcommands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vmp.cli import register

DEFAULT_CONFIG = "configs/features.toml"


def _add_args(p: argparse.ArgumentParser) -> None:
    sub = p.add_subparsers(dest="features_cmd", required=True)

    m = sub.add_parser("materialize", help="copy latest offline rows per entity to online")
    m.add_argument("--view", required=True)
    m.add_argument("--offline-root", default=".vmp/features")
    m.add_argument("--config", default=DEFAULT_CONFIG, help="TOML with [[views]]")
    m.add_argument("--start", type=float, default=None, help="window start (unix seconds)")
    m.add_argument("--end", type=float, default=None, help="window end (unix seconds)")
    m.add_argument("--redis-url", default=None, help="write to Redis instead of memory")
    m.add_argument("--dry-run", action="store_true")

    f = sub.add_parser("feast-export", help="write feature_repo/features.py for Feast")
    f.add_argument("--out", default=".", help="repo root; writes <out>/feature_repo/features.py")
    f.add_argument("--config", default=DEFAULT_CONFIG)
    f.add_argument("--project", default="vmp")
    f.add_argument("--data-root", default="data")
    f.add_argument("--dry-run", action="store_true")


def _load_views(config: str):
    from vmp.config import load_config
    from vmp.features.views import list_views, views_from_config

    if config and Path(config).exists():
        views_from_config(load_config(config))
    return list_views()


def _run_materialize(args: argparse.Namespace) -> int:
    from vmp.features.materialize import materialize
    from vmp.features.offline import JsonlOfflineStore
    from vmp.features.views import get_view

    _load_views(args.config)
    view = get_view(args.view)
    offline = JsonlOfflineStore(args.offline_root)
    online = None
    backend = "none"
    if not args.dry_run:
        if args.redis_url:
            from vmp.features.online import RedisOnlineStore

            online = RedisOnlineStore(args.redis_url)
            backend = "redis"
        else:
            from vmp.features.online import InMemoryOnlineStore

            online = InMemoryOnlineStore()
            backend = "memory"
    manifest = materialize(view, offline, online, args.start, args.end, dry_run=args.dry_run)
    manifest["online_backend"] = backend
    print(json.dumps(manifest))
    return 0


def _run_feast_export(args: argparse.Namespace) -> int:
    from vmp.features.feast_export import write_feast_repo

    views = _load_views(args.config)
    manifest = write_feast_repo(
        args.out, views, data_root=args.data_root, project=args.project, dry_run=args.dry_run
    )
    print(json.dumps(manifest))
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.features_cmd == "materialize":
        return _run_materialize(args)
    if args.features_cmd == "feast-export":
        return _run_feast_export(args)
    return 2


def register_cli() -> None:
    register("features", "feature store: materialize, feast-export", _add_args, _run)


__all__ = ["register_cli"]
