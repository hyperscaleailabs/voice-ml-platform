"""`vmp serve api` and `vmp serve ray`. Registered through `vmp.cli.register`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from vmp.cli import register

DEFAULT_CONFIG = Path("configs") / "serving.toml"


def _load(path: str | None) -> dict[str, Any]:
    from vmp.config import load_config

    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return {"serving": {}}
    return load_config(p)


def _add_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="serve_command")

    api = sub.add_parser("api", help="run the FastAPI voice agent (uvicorn)")
    api.add_argument("--config", default=None, help=f"TOML config (default {DEFAULT_CONFIG})")
    api.add_argument("--host", default=None)
    api.add_argument("--port", type=int, default=None)
    api.add_argument(
        "--backend",
        choices=("echo", "ollama", "openai-compat"),
        default=None,
        help="LLM backend: echo (template LLM, no dependency), ollama, openai-compat",
    )
    api.add_argument("--model", default=None, help="model name for ollama / openai-compat")
    api.add_argument("--base-url", default=None, help="server URL for ollama / openai-compat")
    api.add_argument("--trace", default=None, help="append JSONL trace rows to this file")
    api.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")

    ray = sub.add_parser("ray", help="Ray Serve deployment graph")
    ray.add_argument("--config", default=None, help=f"TOML config (default {DEFAULT_CONFIG})")
    ray.add_argument("--dry-run", action="store_true", help="print the graph description")
    ray.add_argument("--yaml", action="store_true", help="print a `serve run` config file")


def build_api_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve config + flags into the plan `vmp serve api` runs. No heavy imports."""
    cfg = _load(args.config).get("serving", {})
    api = dict(cfg.get("api", {}))
    backends = dict(cfg.get("backends", {}))
    backend = args.backend
    if backend == "echo":
        backends.update({"stt": "echo", "llm": "template", "tts": "silent"})
    elif backend == "ollama":
        backends["llm"] = "ollama"
        opts = dict(backends.get("ollama", {}))
        if args.model:
            opts["model"] = args.model
        if args.base_url:
            opts["host"] = args.base_url
        backends["ollama"] = opts
    elif backend == "openai-compat":
        backends["llm"] = "openai-compat"
        opts = dict(backends.get("openai_compat", {"model": "default"}))
        if args.model:
            opts["model"] = args.model
        if args.base_url:
            opts["base_url"] = args.base_url
        backends["openai_compat"] = opts
    backends.setdefault("stt", "echo")
    backends.setdefault("llm", "template")
    backends.setdefault("tts", "silent")
    return {
        "host": args.host or api.get("host", "127.0.0.1"),
        "port": int(args.port or api.get("port", 8080)),
        "backends": backends,
        "system_prompt": cfg.get("system_prompt"),
        "max_history_turns": int(cfg.get("max_history_turns", 12)),
        "max_context_chars": int(cfg.get("max_context_chars", 1200)),
        "session_ttl_s": float(api.get("session_ttl_s", 3600.0)),
        "max_request_bytes": int(api.get("max_request_bytes", 10 * 1024 * 1024)),
        "turn_timeout_s": float(api.get("turn_timeout_s", 60.0)),
        "trace": args.trace,
    }


def _run_api(args: argparse.Namespace) -> int:
    plan = build_api_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from vmp.serving.api import create_app
    from vmp.serving.backends import backends_from_config
    from vmp.serving.runtime import (
        DEFAULT_SYSTEM_PROMPT,
        InMemorySessionStore,
        JsonlTraceSink,
        NullTraceSink,
        VoiceRuntime,
    )

    stt, llm, tts = backends_from_config({"backends": plan["backends"]})
    tracer = JsonlTraceSink(plan["trace"], keep_in_memory=False) if plan["trace"] else None
    runtime = VoiceRuntime(
        stt,
        llm,
        tts,
        tracer=tracer or NullTraceSink(),
        system_prompt=plan["system_prompt"] or DEFAULT_SYSTEM_PROMPT,
        max_history_turns=plan["max_history_turns"],
        max_context_chars=plan["max_context_chars"],
    )
    app = create_app(
        runtime,
        InMemorySessionStore(ttl_s=plan["session_ttl_s"]),
        max_request_bytes=plan["max_request_bytes"],
        turn_timeout_s=plan["turn_timeout_s"],
    )
    try:
        import uvicorn
    except ModuleNotFoundError:
        print("uvicorn is not installed: pip install 'voice-ml-platform[serve]'", file=sys.stderr)
        return 1
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(app, host=plan["host"], port=plan["port"], log_level="info")
    return 0


def _run_ray(args: argparse.Namespace) -> int:
    from vmp.serving.ray_serve import build_graph, serve_config_yaml

    config = _load(args.config)
    if args.yaml:
        print(serve_config_yaml(config), end="")
        return 0
    if args.dry_run:
        print(json.dumps(build_graph(config, dry_run=True), indent=2))
        return 0
    try:
        from ray import serve
    except ModuleNotFoundError:
        print("ray[serve] is not installed: pip install 'voice-ml-platform[ray]'", file=sys.stderr)
        return 1
    app = build_graph(config)
    rs = config.get("serving", config).get("ray_serve", {})
    serve.run(
        app,
        name=str(rs.get("app_name", "voice-agent")),
        route_prefix=str(rs.get("route_prefix", "/")),
    )
    print("serving; press Ctrl-C to stop", file=sys.stderr)
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        serve.shutdown()
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.serve_command == "api":
        return _run_api(args)
    if args.serve_command == "ray":
        return _run_ray(args)
    print("usage: vmp serve {api,ray} ...", file=sys.stderr)
    return 2


def register_cli() -> None:
    register("serve", "voice agent API (FastAPI) and Ray Serve graph", _add_args, _run)


__all__ = ["build_api_plan", "register_cli"]
