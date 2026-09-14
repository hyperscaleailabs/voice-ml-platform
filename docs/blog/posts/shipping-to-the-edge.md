---
date: 2026-09-12
authors: [cg]
categories:
  - Edge
  - Operations
slug: shipping-to-the-edge
---

# Shipping to the edge

An edge deployment is not a smaller cloud deployment. It runs on hardware you do
not control, frequently with no network, and you cannot roll it back by changing
a load balancer weight. Whatever is on that device is what the user has until
they accept an update.

Which changes what "deploying a model" has to mean. In the cloud it means
pointing traffic at a new replica. On a device it means: prove which files are
there, prove they are intact, prove the runtime can load them, prove the thing is
allowed to do what it is about to do, and be able to get back.

<!-- more -->

## Three targets, and the choice is portability first

| Target | Runtime | Quantisation accepted |
|---|---|---|
| `onnx` | ONNX Runtime | `int8`, `none` |
| `gguf` | llama.cpp | `q4_k_m`, `int8` (Q8_0), `none` |
| `mlx` | MLX, Apple silicon | `int4`, `int8`, `none` |

The plan validator rejects an invalid pairing rather than discovering it during
conversion, which matters because conversions are slow and the error arrives an
hour in.

Choosing between them is a portability question before it is a performance one.
ONNX Runtime runs nearly everywhere and is the safe answer for a heterogeneous
fleet. GGUF with llama.cpp is the best-supported path for CPU-only devices and
has the richest quantisation ladder. MLX is Apple silicon only and is the fast
path where the device is a Mac or an iPhone-class SoC — and, as covered in the
Whisper post, it has no training API, which is exactly why every export begins
with merging the adapter into the base weights.

That merge step is not a flag. `vmp edge export --dry-run` lists it as a step
with its own dependencies, because a LoRA adapter and a GGUF file are different
kinds of object and the conversion between them has its own failure modes:

```json
{"name": "merge_adapter",
 "detail": "merge LoRA adapter .vmp/adapters/voice-sft into google/gemma-3-270m",
 "requires": ["peft", "transformers"], "status": "planned"}
```

Forget it and you export the base model. Nothing tells you.

## Quantisation, qualitatively

Lower precision buys memory and memory bandwidth, and on a memory-bound decoder
that converts directly into latency. It costs accuracy.

The cost is not uniform, and that is the part worth internalising. It lands
hardest on the rare and the specific: proper nouns, numbers, long-tail facts —
precisely the material a voice agent gets asked about. Common phrasings survive
aggressive quantisation nearly intact, which is what makes casual evaluation so
misleading. The model still *sounds* right.

So sweeping the quantisation level without an evaluation set is guessing. Run the
golden set at each level, read the per-category strata rather than the headline,
and choose on measured numbers. Then record which level shipped, in the bundle
manifest, where it can be correlated with a field report.

This repository has not benchmarked any quantisation level, and so no figures
appear here.

## The bundle is the unit

Not a model file. A directory with a `manifest.json` describing everything in it:

```json
{
  "schema": 1,
  "name": "voice-agent-edge",
  "version": "0.1.0",
  "target": "gguf",
  "base_model": "google/gemma-3-270m",
  "adapter_version": "voice-sft@3",
  "lineage": {},
  "min_runtime_version": "0.1.0",
  "built_with_runtime": "0.1.0",
  "required": ["model-q4_k_m.gguf", "tokenizer.json"],
  "files": {
    "model-q4_k_m.gguf": {"sha256": "a7acdcee40…", "size": 10},
    "tokenizer.json": {"sha256": "e346432021…", "size": 8}
  },
  "policy": {
    "offline_only": true, "allowed_egress": [], "max_ttfa_ms": 3000,
    "fallback": "degrade", "retention": {"audio": "none"}
  }
}
```

`lineage` is free-form registry provenance — artifact name and version, data
hash, config hash, git sha — filled from the `ModelArtifact` that was promoted. A
bundle in the field with an empty `lineage` cannot be traced back to a training
run, and "which run produced the model on this device" is the first question
asked in every edge incident.

`min_runtime_version` is a real compatibility contract: bump it when the bundle
needs something an older runtime lacks, and the old device refuses the bundle
rather than loading it and misbehaving. Refusing is the good outcome.

## Verification as a precondition, not a health check

`verify_bundle` checks, in order: the directory and manifest exist and parse;
every mandatory key is present; every `required` file is listed and present;
**every listed file's SHA-256 and size match**; no unlisted file is present; the
runtime version is at least `min_runtime_version`; and the policy is internally
consistent.

```bash
vmp edge bundle verify .vmp/edge/bundles/voice-agent-edge-0.1.0
# {"ok": true, "problems": []}
```

`EdgeRuntime.start(require_verified=True)` refuses to start on a failure. The
temptation to serve from an unverified bundle because verification was
inconvenient is exactly how an unknown model ends up in a fleet, and it always
looks locally reasonable at the moment it happens.

The strict unlisted-file check earns its keep more than it looks. A file in the
bundle directory that the manifest does not describe is either a partial download
or something that should not be there. Both are reasons not to load.

One honest limitation: checksums catch corruption, not tampering. The manifest is
not signed. For a hostile-network threat model, sign it and verify the signature
before the checksums — and note that in the threat model rather than assuming
SHA-256 covers it.

## The offline policy, enforced in code

The policy is data in the manifest, and `enforce` is the single check point the
runtime calls before any network call, cloud fallback or audio write:

| Action | Refused when |
|---|---|
| `network` | `offline_only = true`, or the host is not in `allowed_egress` |
| `cloud_fallback` | `offline_only = true`, or `fallback != "cloud"` |
| `store_audio` | `retention.audio = "none"` |

`validate()` catches contradictions at build time: `offline_only` with a
non-empty `allowed_egress`, `offline_only` with `fallback = "cloud"`, an egress
entry written as a URL instead of a host. A policy that cannot be satisfied is
caught before it ships, not on a device in someone's kitchen.

`fallback` is where the product decision lives. `degrade` answers locally with
something simpler. `refuse` says it cannot answer. `cloud` sends the turn
upstream and is only legal when the bundle is not offline-only. For a device sold
on "nothing leaves the room", `offline_only = true` with `fallback = "degrade"`
is the only defensible pair — and the manifest is the artifact that proves it,
which is a better position than a sentence in a privacy policy.

## OTA: diff, canary, and the flag nobody should miss

```json
{"from": {"version": "0.1.0"}, "to": {"version": "0.2.0"},
 "added": [], "changed": ["model-q4_k_m.gguf"], "removed": [],
 "unchanged": ["tokenizer.json"],
 "download_bytes": 184320000, "policy_changed": false}
```

`download_bytes` is the update's real cost on a metered connection — worth
knowing before you push to a fleet.

`policy_changed` is the one that must never be missed. A bundle that quietly
loosens `offline_only` or changes `retention.audio` is a privacy change wearing a
version bump. It will pass every functional test, improve no metric, and change
what the product is. Treat `true` there as requiring the same review as a legal
change, because it is one.

The rollout is a canary, not a push. Ship to a small cohort; watch
`p95_ttfa_ms_24h` and `edge_bundle_version` from the `device_features` view;
promote only when the cohort holds. The failure path has a runbook —
`edge-bundle-verify-failed` — and its severity model is instructive: SEV-3 on one
device, because that device keeps serving the previous bundle, which is the
design working. SEV-2 across a rollout wave, or if a device ends up with no
working bundle at all.

## The cost the edge tempts you into

Edge pressure pushes toward the smallest model that fits, and the cost is
measurable. On six factual questions, `gemma3:270m` answered with a median LLM
latency of **192 ms** against **4,427 ms** for the 4B model, and got **2 of 6
wrong** where the larger model got **0 of 6** wrong (alpha-core, cycle 5,
2026-09-12, `notebook_optimized.ipynb`).

The wrong answers were not hedges or refusals. One said there are eight
continents. The other attributed *Romeo and Juliet* to a fabricated name. That is
not degraded quality; it is confident invention, delivered in the same calm
synthesised voice as every correct answer, to a listener with no way to see a
source.

Six questions is a small set, and the correctness figure is a difference of two
items — its precision should not be overstated. The direction is not in doubt.

So: report the speedup and the wrong-answer rate together, always. A 23× latency
improvement that invents facts is not a latency improvement, it is a different
product. The release gate is where that becomes structural rather than a matter
of discipline — `GateDecision` evaluates latency thresholds and correctness
thresholds in one decision, and a candidate that improves one while breaching the
other does not pass. Put a correctness gate on the edge candidate, gate the
**quantised** artifact rather than the one you trained, and let it fail.

Letting it fail is the feature. Everything else is a way of finding out later.
