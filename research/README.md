# Stage 1 — research

This directory holds the hypotheses that came before the package. Each one is a
**spike**: the smallest runnable thing that can decide whether an idea is worth
building properly. A spike answers one question and is then finished. It is not a
prototype of the code in `src/vmp/`, and no part of it is carried over by copying.

## What a spike is here

One directory, `research/<nn>-<topic>/`, containing a `README.md` in the format
below and whatever the question needed: a script, a notebook, a small corpus, a
results file. The script is written against the real package API so that the
question is asked of the system that exists, not of an imagined one.

## Rules

- **A spike is never imported by `src/`.** The dependency runs one way only:
  a spike may import `vmp`, `vmp` may never import a spike. This keeps a spike
  disposable by construction.
- **A spike is frozen once its gate decision is made.** After the outcome is
  written down, the directory stops changing. If the idea survives, the
  implementation starts again in `src/vmp/` from the codebase, not from the
  spike's diff.
- **A spike ships something runnable.** Someone else must be able to re-run the
  falsifier, or the plan that stands in for it, without the author.
- **Nothing here runs in CI.** These scripts are not tests, they are not
  imported by tests, and a broken spike never fails a build. A repository-wide
  lint may still read these files; executing them is what CI does not do. The
  tests that guard behaviour live in `tests/` against `src/vmp/`.

The scripts in this directory run with **no heavy dependencies installed**. They
use the package's dry-run and planning paths, and they compute real numbers on
synthetic data. Synthetic numbers are labelled as synthetic everywhere they are
printed: they exercise the scorer, they do not measure a model.

## Required README sections

Every `research/<nn>-<topic>/README.md` has exactly these five sections, in this
order:

| Section | Contains |
|---|---|
| `## Claim` | One falsifiable sentence. What is asserted, about what, under what conditions. |
| `## Falsifier` | The observation that would make the claim false. Written before the run. |
| `## Method` | How the question is asked: data, procedure, what the spike script does. |
| `## Outcome` | One of the four values below, plus what was actually observed. |
| `## What moved into src/` | The protocol, module or function development built, or `nothing`. |

Allowed `## Outcome` values:

- `not run` — the falsifier has not been executed. Nothing downstream may cite a
  result from this spike.
- `run — falsifier passes` — the falsifier was executed and did not fire; the
  claim survives under the stated conditions.
- `run — falsified` — the falsifier fired; the claim is false as stated.
- `run — inconclusive` — the falsifier was executed but the observation does not
  decide either way; the reason is recorded.

Measurements quoted in these READMEs follow the citation rule in
[`DESIGN.md`](../DESIGN.md): a number from the private predecessor project is
cited as "alpha-core, cycle 5, 2026-09-12, `<file>`" and is never presented as a
benchmark of this repository.

## Index

| Spike | Claim (short form; the full claim is in each README) | Outcome |
|---|---|---|
| [01](01-whisper-accent-lora/) | Whisper LoRA adapts to one speaker's accent | `not run` |
| [02](02-dpo-spoken-style/) | Rule-derived pairs shift reply style through DPO | `not run` |
| [03](03-ray-scaling/) | One plan submits unchanged to Ray Train with N workers | `not run` |
| [04](04-small-model-tradeoff/) | A smaller model trades correctness for latency | `not run` |

## Running the spikes

From the repository root, with nothing but the standard library installed:

```bash
python research/01-whisper-accent-lora/spike_whisper_lora_plan.py
python research/02-dpo-spoken-style/spike_dpo_pairs.py
python research/03-ray-scaling/spike_ray_plan.py
python research/04-small-model-tradeoff/spike_correctness_harness.py
```

The two notebooks are checked in **unexecuted**. They record the procedure a real
run would follow, including the cells that need `transformers`, `peft` and audio
hardware; they are not evidence of a run.
