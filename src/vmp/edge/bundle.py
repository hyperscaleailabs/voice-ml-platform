"""Edge bundles: a directory with `manifest.json`, checksummed files and a policy.

`verify_bundle` checks required files, SHA-256 of every listed file, unlisted
files, and runtime version compatibility. `diff_bundles` lists what an OTA
update would have to ship.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vmp import __version__ as RUNTIME_VERSION
from vmp.edge.policy import EdgePolicy

MANIFEST_NAME = "manifest.json"
REQUIRED_FILES: tuple[str, ...] = (MANIFEST_NAME,)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _version_tuple(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in v.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


@dataclass
class EdgeBundle:
    """A bundle on disk. `manifest` mirrors `manifest.json`."""

    path: Path
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> EdgeBundle:
        p = Path(path)
        mf = p / MANIFEST_NAME
        if not mf.exists():
            raise FileNotFoundError(f"bundle manifest not found: {mf}")
        return cls(path=p, manifest=json.loads(mf.read_text(encoding="utf-8")))

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", ""))

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", ""))

    @property
    def files(self) -> dict[str, dict[str, Any]]:
        return dict(self.manifest.get("files", {}))

    def policy(self) -> EdgePolicy:
        return EdgePolicy.from_dict(self.manifest.get("policy", {}))

    def verify(self, runtime_version: str = RUNTIME_VERSION) -> tuple[bool, list[str]]:
        return verify_bundle(self.path, runtime_version=runtime_version)


def build_bundle(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    name: str,
    version: str,
    target: str,
    base_model: str,
    adapter_version: str | None = None,
    lineage: dict[str, Any] | None = None,
    min_runtime_version: str = RUNTIME_VERSION,
    policy: EdgePolicy | None = None,
    required: list[str] | None = None,
    dry_run: bool = False,
) -> EdgeBundle:
    """Copy every file under `src_dir` into `out_dir` and write `manifest.json`.

    `lineage` is free-form registry provenance (artifact name/version, data
    hash, config hash, git sha). `required` names files the runtime needs; they
    default to every copied file. In dry-run mode nothing is written and the
    returned bundle holds the manifest that would be produced.
    """
    src = Path(src_dir)
    out = Path(out_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"bundle source not found: {src}")
    files: dict[str, dict[str, Any]] = {}
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            rel = p.relative_to(src).as_posix()
            files[rel] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    pol = policy or EdgePolicy()
    problems = pol.validate()
    if problems:
        raise ValueError("invalid policy: " + "; ".join(problems))
    manifest: dict[str, Any] = {
        "schema": 1,
        "name": name,
        "version": version,
        "target": target,
        "base_model": base_model,
        "adapter_version": adapter_version,
        "lineage": dict(lineage or {}),
        "min_runtime_version": min_runtime_version,
        "built_with_runtime": RUNTIME_VERSION,
        "created_at": time.time(),
        "required": sorted(required) if required is not None else sorted(files),
        "files": files,
        "policy": pol.to_dict(),
    }
    if dry_run:
        return EdgeBundle(path=out, manifest=manifest)
    out.mkdir(parents=True, exist_ok=True)
    for rel in files:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dst)
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return EdgeBundle(path=out, manifest=manifest)


def verify_bundle(
    bundle_dir: str | Path,
    *,
    runtime_version: str = RUNTIME_VERSION,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """Return `(ok, problems)`. `strict` also flags files present but not listed."""
    d = Path(bundle_dir)
    problems: list[str] = []
    mf = d / MANIFEST_NAME
    if not d.is_dir():
        return False, [f"bundle directory not found: {d}"]
    if not mf.exists():
        return False, [f"missing {MANIFEST_NAME}"]
    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, [f"{MANIFEST_NAME} is not valid JSON: {e.msg}"]
    for key in ("name", "version", "target", "files", "min_runtime_version", "policy"):
        if key not in manifest:
            problems.append(f"manifest missing key {key!r}")
    files = manifest.get("files", {}) or {}
    for rel in manifest.get("required", []):
        if rel not in files:
            problems.append(f"required file not in manifest: {rel}")
        elif not (d / rel).exists():
            problems.append(f"required file missing: {rel}")
    for rel, info in sorted(files.items()):
        p = d / rel
        if not p.exists():
            if rel not in manifest.get("required", []):
                problems.append(f"listed file missing: {rel}")
            continue
        actual = sha256_file(p)
        if actual != info.get("sha256"):
            problems.append(f"checksum mismatch: {rel}")
        if info.get("size") is not None and p.stat().st_size != info["size"]:
            problems.append(f"size mismatch: {rel}")
    if strict:
        listed = set(files)
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != MANIFEST_NAME:
                rel = p.relative_to(d).as_posix()
                if rel not in listed:
                    problems.append(f"unlisted file: {rel}")
    min_rt = str(manifest.get("min_runtime_version", "0"))
    if _version_tuple(runtime_version) < _version_tuple(min_rt):
        problems.append(f"runtime {runtime_version} < min_runtime_version {min_rt}")
    pol_problems = EdgePolicy.from_dict(manifest.get("policy", {})).validate()
    problems.extend(f"policy: {p}" for p in pol_problems)
    return (not problems), problems


def diff_bundles(a: str | Path | EdgeBundle, b: str | Path | EdgeBundle) -> dict[str, Any]:
    """Files an OTA update from `a` to `b` must ship: added, changed, removed."""
    ba = a if isinstance(a, EdgeBundle) else EdgeBundle.load(a)
    bb = b if isinstance(b, EdgeBundle) else EdgeBundle.load(b)
    fa, fb = ba.files, bb.files
    added = sorted(k for k in fb if k not in fa)
    removed = sorted(k for k in fa if k not in fb)
    changed = sorted(k for k in fb if k in fa and fa[k].get("sha256") != fb[k].get("sha256"))
    unchanged = sorted(k for k in fb if k in fa and k not in changed)
    return {
        "from": {"name": ba.name, "version": ba.version},
        "to": {"name": bb.name, "version": bb.version},
        "added": added,
        "changed": changed,
        "removed": removed,
        "unchanged": unchanged,
        "download_bytes": sum(int(fb[k].get("size", 0)) for k in added + changed),
        "policy_changed": ba.manifest.get("policy") != bb.manifest.get("policy"),
    }


__all__ = [
    "MANIFEST_NAME",
    "REQUIRED_FILES",
    "EdgeBundle",
    "build_bundle",
    "diff_bundles",
    "sha256_file",
    "verify_bundle",
]
