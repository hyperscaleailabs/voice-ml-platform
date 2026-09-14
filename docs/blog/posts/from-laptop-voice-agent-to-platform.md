---
date: 2026-08-15
authors: [cg]
categories:
  - Platform
  - Architecture
slug: from-laptop-voice-agent-to-platform
---

# From a laptop voice agent to a platform

A voice agent that runs entirely on one laptop is a satisfying thing to build.
Microphone in, speaker out, three local models in between: an energy VAD that
waits for speech and ends the turn on silence, Whisper for transcription, a local
LLM streaming tokens, a segmenter cutting sentences as they complete, and a TTS
engine speaking them one at a time. No wake word, no key to press, nothing
leaving the machine.

It works. And the moment it works, the interesting problem changes.

<!-- more -->

## The thing that works is not the thing that ships

The prototype answers one question: can these pieces be assembled into something
that feels like a conversation? Once the answer is yes, every subsequent question
is about the layer around it.

How do you know the new model is better than the old one? How do you know it did
not get worse for one particular speaker? When the reply arrives late, which of
the six stages was slow? How do you deploy the same behaviour to a cloud API and
to a device with no network? When something regresses in production, how do you
get back to the exact data and configuration that produced the model that is
running?

None of those are answered by the pipeline. They are answered by a platform: the
training path, the evaluation harness, the registry, the feature store, the
retrieval layer, the serving runtime, the export pipeline, the traces. The
pipeline is the part that demos. The platform is the part that lets you change
the pipeline without breaking it.

## What generalises and what does not

`voice-ml-platform` is the second version of that laptop agent, written as a
platform rather than as an application. Three things carried over unchanged, and
they turned out to be the load-bearing ones.

**The trace schema.** One JSON object per line: `ts, session, turn, event, span,
seq, ms, payload`, where `event` is `<stage>.start` or `<stage>.end`. A research
notebook, a unit test, a laptop run and a production pod write identical rows.
That is what makes a latency percentile mean the same thing at every stage of
maturity — and it is why the evaluation gate can consume a number the serving
runtime produced without a translation layer in between.

**The stage names.** `listen, stt, retrieve, llm, segment.emit, tts, playback,
turn`. They name the *role*, not the engine filling it. Swap faster-whisper for
mlx-whisper and the `stt` stage still means "audio became text", so the before
and after are comparable. A schema that named the vendor would have made every
backend change a migration.

**The segmenter.** The one component shared verbatim by the research spike, the
cloud runtime and the edge runtime. It is also the one component that must never
be wrong, which is a nice illustration of how those two properties travel
together.

What did not generalise: everything that assumed one process, one microphone,
one user and one machine. Session state, model loading, configuration, the
retrieval index, the audio device. All of that had to be rebuilt against a
protocol rather than against an implementation.

## Why the value sits in the platform layer

The models are not the moat. Whisper is public, the instruct models are public,
the TTS is public, and the person competing with you can download all three this
afternoon. What they cannot download is the answer to "is this version better
than the one we shipped last week, for our users, on our data, measured the same
way".

Concretely, here is what the platform layer buys that the pipeline cannot.

**A number you can defend.** The evaluation harness synthesises audio from text
you wrote, so the reference transcript is exact by construction. Word error rate
stops being an estimate and becomes a measurement — and the strata matter more
than the headline. The predecessor's golden set measured 2.69% WER overall and
**28.9% on the proper-noun stratum** (alpha-core, 2026-09-05,
`evals/golden/README.md`). An assistant that is asked about its own stack
mishears every technical noun, and the overall number hides that completely.

**Lineage.** Every registered artifact carries a `config_hash`, a `data_hash`,
a git sha and the `GateDecision` that let it through. "Why is this model in
production?" has an answer that is an object, not a recollection.

**A trade made visible.** Under load you want a smaller model. The predecessor
measured that swap: llm median **192 ms** against **4,427 ms**, and **2 of 6**
factual questions wrong against **0 of 6** (alpha-core, cycle 5, 2026-09-12,
`notebook_optimized.ipynb`). A 23× latency win that invents an author's name is
not a latency win, it is a different product. The platform's job is to make that
trade impossible to report half-way — so the release gate evaluates latency and
correctness in the same decision, and a run that improves one while breaching
the other does not pass.

**One artifact, two destinations.** The same adapter serves a cloud API through
a Ray Serve graph and a device through a checksummed bundle with an offline
policy. Without an export pipeline and a bundle format, "edge support" is a
person copying files.

## The shape: research, development, productization

The repository is laid out as a maturity path rather than as a product, because
the path is the transferable part.

**Research** is `research/`: one directory per spike, each with a claim, a
falsifier written *before* the run, a method, an outcome and a line saying what
moved into `src/`. A spike is never imported by the package, and it is frozen
once its outcome is recorded. The exit criterion is that the falsifier was
executed and the result written down — or that the spike says `not run` and
nothing downstream cites it.

**Development** is `src/vmp/`: one package per subsystem, each with a `Protocol`
for its backend, one standard-library reference implementation that runs in tests
and demos, optional adapters that import their dependency lazily inside the
class, and a `dry_run` on every heavy operation. `import vmp` succeeds with
nothing installed. The exit criterion is that the tests pass with no network, no
models and no GPU; that a second adapter can be written against the protocol
without touching the runtime; and that every heavy operation has a dry run CI can
execute.

**Productization** is `deploy/` and `runbooks/`: Kubernetes, KubeRay, Prometheus
SLO rules, Grafana boards, and nine operational procedures. The exit criterion is
that SLOs are Prometheus rules with an error budget rather than prose; that the
release gate passes and its decision is stored beside the artifact; that every
alert routes to a runbook and every runbook ends in a rollback; and that the
rollout is bounded — canary first, with an automatic trigger tied to the SLO.

The arrows between those stages are gates, and the gates are the product. Anyone
can write a training script. The question is what has to be true before its
output is allowed to speak to a user.

## The rule that makes it worth reading

One discipline runs through all of it: **a number with no source file does not go
in a table.** Measurements from the predecessor are cited with their cycle, date
and file, and never presented as this repository's own benchmark. Spikes that
have not been run say `not run`, and nothing downstream is allowed to quote them.

All four spikes in this repository currently say `not run`. That is not an
oversight. Each has a real measurement behind it from the laptop project, and
each of those measurements has a caveat that makes it insufficient — the accent
LoRA had no held-out set, the small-model comparison had six questions, the Ray
claim has never met a cluster. Marking them as passed because a related number
exists somewhere would be exactly the failure the format is designed to prevent.

A platform that reports honest gaps is more useful than one that reports
confident numbers you cannot trace. The gaps tell you what to build next.

## Where to start reading

The [progression](../../progression.md) page lays out the three stages and their
exit criteria. [Full stack](../../architecture/full-stack.md) walks one voice
turn from the user's mouth back to their ear. The
[guides](../../guides/training-lora-sft.md) cover one subsystem each, with the
real config, the real CLI, what a dry run returns and what a real run needs.

The next post is about the one number the user actually feels.
