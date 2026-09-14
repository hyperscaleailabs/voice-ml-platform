"""`vmp` command line. Each subsystem registers its own subcommands.

Registration is explicit so importing the CLI never imports a heavy backend:
a module's `register_cli()` adds an `argparse` subparser and a `run(args)` callable.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass

AddArgs = Callable[[argparse.ArgumentParser], None]
Run = Callable[[argparse.Namespace], int]


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    add_args: AddArgs
    run: Run


_COMMANDS: dict[str, Command] = {}

# Modules that expose `register_cli()`. Imported on demand in `build_parser`.
_CLI_MODULES = (
    "vmp.data.cli",
    "vmp.features.cli",
    "vmp.training.cli",
    "vmp.registry.cli",
    "vmp.rag.cli",
    "vmp.serving.cli",
    "vmp.edge.cli",
    "vmp.eval.cli",
    "vmp.observability.cli",
)


def register(name: str, help: str, add_args: AddArgs, run: Run) -> None:
    """Add a top-level subcommand. Re-registering the same name replaces it."""
    _COMMANDS[name] = Command(name, help, add_args, run)


def _load_modules() -> None:
    for mod in _CLI_MODULES:
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:  # a subsystem may not exist yet
            if e.name and e.name.startswith("vmp."):
                continue
            raise
        fn = getattr(m, "register_cli", None)
        if callable(fn):
            fn()


def build_parser() -> argparse.ArgumentParser:
    _load_modules()
    parser = argparse.ArgumentParser(prog="vmp", description="voice-ml-platform command line")
    parser.add_argument("--version", action="store_true", help="print the package version")
    sub = parser.add_subparsers(dest="command")
    for cmd in sorted(_COMMANDS.values(), key=lambda c: c.name):
        p = sub.add_parser(cmd.name, help=cmd.help)
        cmd.add_args(p)
        p.set_defaults(_run=cmd.run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from vmp import __version__

        print(__version__)
        return 0
    run = getattr(args, "_run", None)
    if run is None:
        parser.print_help()
        return 2
    return int(run(args))

