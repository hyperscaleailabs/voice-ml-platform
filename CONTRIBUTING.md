# Contributing

Thank you for looking at this. `voice-ml-platform` is a public, production-oriented
ML platform for voice agents, organised as a progression from research to
development to productization. `DESIGN.md` is the contract for structure,
naming and rules — read it before writing code, and treat a conflict between it
and this file as a bug in this file.

## Getting set up

```bash
git clone https://github.com/hyperscaleailabs/voice-ml-platform
cd voice-ml-platform
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 or newer. The core package is standard library only; every heavy
backend is an optional extra.

## Before you open a pull request

```bash
ruff check src tests examples
pytest -q
for demo in examples/demo_*.py; do python "$demo"; done
kubectl kustomize deploy/k8s/overlays/local > /dev/null
kubectl kustomize deploy/k8s/overlays/cloud > /dev/null
```

CI runs the same on Python 3.11, 3.12 and 3.13, plus a job that installs only
`pytest` and runs the suite and the demos.

## The rules that are actually load-bearing

These are the ones a review will send a change back for.

**The core imports with only the standard library.** `import vmp` and every
submodule must succeed with nothing installed. A heavy dependency is imported
lazily *inside* the adapter class that needs it, never at module import.

```python
class QdrantVectorStore:
    def _client(self):
        from qdrant_client import QdrantClient   # here, not at the top
        ...
```

**Every backend has a Protocol and a stdlib reference implementation.** The
reference implementation is what the tests and demos run against. An adapter
without a working stdlib sibling is not finished.

**Every heavy operation takes `dry_run`.** A dry run validates its inputs,
returns the plan or manifest as a dict, and imports nothing heavy. This is what
makes the manifests in `deploy/` testable without a GPU.

**Use the shared types.** `vmp/types.py` holds `Utterance`, `Turn`, `Session`,
`PreferencePair`, `FeatureRow`, `Document`, `Chunk`, `ModelArtifact`,
`EvalResult`, `GateDecision`. Do not redefine them locally.

**CLI subcommands register themselves** through
`vmp.cli.register(name, help, add_args, run)` from the owning module.

**Configs are TOML in `configs/`,** loaded with `tomllib`. No YAML in the core
package. Environment variables override with the `VMP_` prefix.

**Traces keep one schema:** one JSON object per line with
`ts, session, turn, event, span, seq, ms, payload`, where `event` is
`<stage>.start` / `<stage>.end`. Payloads carry measurements, never transcripts
or identifiers.

## Style

`ruff` with line length 100 and rules `E,F,I,UP,B,SIM,RUF`. Type hints
everywhere, `from __future__ import annotations` at the top of every module.
Markdown lines at or under 100 characters.

Documentation tone is neutral, technical and concise: what it is, how it works,
why it matters. No marketing language.

## Numbers

**Every number needs a source.** Measurements from the private `alpha-core`
project are cited as "alpha-core, cycle 5, 2026-09-12, `<file>`" and are never
presented as this repository's benchmark. Demo output may be quoted only if the
demo actually prints it. Do not invent benchmarks, throughput figures or costs.
`DESIGN.md` lists the measured facts that may be cited.

## Tests

`tests/test_<module>.py`, pytest. The default run needs only `pytest` (plus
`fastapi` and `httpx` for the API tests, which `pytest.importorskip`). No
network, no model downloads, no GPU. Markers: `slow`, `gpu`, `integration`.
Keep the default run under about 30 seconds.

## Hypothesis-driven changes

A change to modelling, retrieval or latency behaviour starts as a hypothesis,
not as a patch. Open a [hypothesis issue](.github/ISSUE_TEMPLATE/hypothesis.md);
accepted ones land as `research/<nn>-<topic>/README.md` with exactly these
sections: `## Claim`, `## Falsifier`, `## Method`, `## Outcome`,
`## What moved into src/`. Outcome is one of `not run`, `run — falsifier
passes`, `run — falsified`, `run — inconclusive`. Anything not run says so.

A falsified hypothesis is a result worth keeping, and keeping it is what stops
the next person trying the same thing.

## Deployment and operations changes

Manifests live in `deploy/`; see `deploy/README.md` for how the pieces fit and
for the DNS contract shared between compose and Kubernetes. Operational
procedures live in `runbooks/`. If a change alters how something is operated —
a new failure mode, a new rollback path — the runbook is part of the change,
not a follow-up.

## Commits and pull requests

Small, focused commits with a present-tense subject line under about 72
characters. Explain *why* in the body when it is not obvious. Fill in the pull
request template: it asks for the verification you ran and the sources for any
numbers, and both are checked in review.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## Conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are licensed under Apache-2.0, matching the repository.
