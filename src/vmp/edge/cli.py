"""`vmp edge` subcommands: export planning, bundle build / verify / diff."""

from __future__ import annotations

import argparse
import json

from vmp.cli import register
from vmp.config import load_config
from vmp.edge.bundle import build_bundle, diff_bundles, verify_bundle
from vmp.edge.export import ExportPlan, export, plan_from_config
from vmp.edge.policy import EdgePolicy, policy_from_config


def _add_args(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="edge_command", required=True)

    p = sub.add_parser("export", help="plan or run a model export")
    p.add_argument("--config", default=None, help="TOML with an [export] table")
    p.add_argument("--base-model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--target", choices=("onnx", "gguf", "mlx"), default=None)
    p.add_argument("--quant", default=None, help="int8 | int4 | q4_k_m | none")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")

    b = sub.add_parser("bundle", help="build, verify or diff edge bundles")
    bsub = b.add_subparsers(dest="bundle_command", required=True)

    p = bsub.add_parser("build")
    p.add_argument("--src", required=True, help="directory with exported model files")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter-version", default=None)
    p.add_argument("--config", default=None, help="TOML with a [policy] table")
    p.add_argument("--dry-run", action="store_true")

    p = bsub.add_parser("verify")
    p.add_argument("dir")

    p = bsub.add_parser("diff")
    p.add_argument("a")
    p.add_argument("b")


def _run(args: argparse.Namespace) -> int:
    if args.edge_command == "export":
        if args.config:
            plan = plan_from_config(load_config(args.config))
            data = plan.to_dict()
        else:
            data = {"quantization": {"method": "none"}}
        if args.base_model:
            data["base_model"] = args.base_model
        if args.adapter:
            data["adapter_path"] = args.adapter
        if args.target:
            data["target"] = args.target
        if args.quant:
            data["quantization"] = {"method": args.quant}
        if args.out:
            data["out_dir"] = args.out
        for key in ("base_model", "target", "out_dir"):
            if not data.get(key):
                print(f"missing --{key.replace('_', '-')} (or set it in --config)")
                return 2
        manifest = export(ExportPlan.from_dict(data), dry_run=args.dry_run)
        print(json.dumps(manifest, indent=2))
        return 0 if not manifest["problems"] else 1
    if args.edge_command == "bundle":
        if args.bundle_command == "build":
            policy = policy_from_config(load_config(args.config)) if args.config else EdgePolicy()
            bundle = build_bundle(
                args.src,
                args.out,
                name=args.name,
                version=args.version,
                target=args.target,
                base_model=args.base_model,
                adapter_version=args.adapter_version,
                policy=policy,
                dry_run=args.dry_run,
            )
            print(json.dumps(bundle.manifest, indent=2, sort_keys=True))
            return 0
        if args.bundle_command == "verify":
            ok, problems = verify_bundle(args.dir)
            print(json.dumps({"ok": ok, "problems": problems}, indent=2))
            return 0 if ok else 1
        if args.bundle_command == "diff":
            print(json.dumps(diff_bundles(args.a, args.b), indent=2))
            return 0
    return 2


def register_cli() -> None:
    register("edge", "edge: export, bundle build/verify/diff", _add_args, _run)


__all__ = ["register_cli"]
