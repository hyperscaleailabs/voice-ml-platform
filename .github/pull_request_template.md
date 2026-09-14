## What and why

<!-- One paragraph. What changes, and what problem it solves. -->

## Hypothesis

<!-- For a research or modelling change, link the research/<nn>-<topic>/ entry
     or the hypothesis issue. Delete this section for a plain code change. -->

- Claim:
- Falsifier:
- Outcome: <!-- not run | run — falsifier passes | run — falsified | run — inconclusive -->

## How it was verified

<!-- Commands and their result. "Tests pass" is not a verification; the command
     you ran and what it printed is. -->

```
ruff check src tests examples
pytest -q
```

- [ ] `pytest -q` passes
- [ ] `ruff check src tests examples` passes
- [ ] Demos still run: `python examples/demo_*.py`
- [ ] The core still imports with only the standard library
- [ ] Manifests still render: `kubectl kustomize deploy/k8s/overlays/{local,cloud}`

## Numbers

<!-- Every number in the diff needs a source. Measurements from the private
     alpha-core project are cited as "alpha-core, cycle N, YYYY-MM-DD, <file>"
     and never presented as this repository's benchmark. Demo output may be
     quoted only if the demo actually prints it. Delete if the diff has none. -->

## Risk and rollback

- Blast radius:
- How to roll it back: <!-- link the runbook if one applies -->

## Checklist

- [ ] Type hints, `from __future__ import annotations`, lines within 100 chars
- [ ] New backends are behind a `Protocol`, imported lazily inside the adapter
- [ ] Heavy operations accept `dry_run`
- [ ] No secrets, no absolute local paths, no references to private machines
- [ ] Docs and runbooks updated if behaviour or operations changed
