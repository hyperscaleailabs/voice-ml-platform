# Guide: Edge export

## Purpose

An edge deployment is not a smaller cloud deployment. It runs on a device you do
not control, often with no network, and it must be possible to say exactly what
model is on that device and prove the files were not corrupted in transit.

`vmp.edge` covers three things:

1. **Export** — merge the LoRA adapter into the base weights and convert to a
   runtime format: ONNX, GGUF or MLX, optionally quantised.
2. **Bundle** — wrap the exported files in a directory with a `manifest.json`
   carrying SHA-256 per file, lineage, a minimum runtime version, and a
   **policy**.
3. **Verify** — check every checksum, every required file, the runtime version
   and the policy before the runtime is allowed to load it.

The edge runtime runs the same turn loop and the **same segmenter** as the
cloud runtime. That is the one component shared unchanged from research to
edge, and it is the reason a sentence is cut identically in both places.

## Config

`configs/edge.toml`:

```toml
# Edge export, bundle and runtime policy.

[export]
base_model = "google/gemma-3-270m"
adapter_path = ".vmp/adapters/voice-sft"
target = "gguf"          # onnx | gguf | mlx
out_dir = ".vmp/edge/export"

[export.quantization]
method = "q4_k_m"        # int8 | int4 | q4_k_m | none

[bundle]
name = "voice-agent-edge"
version = "0.1.0"
min_runtime_version = "0.1.0"
out_dir = ".vmp/edge/bundles"

[policy]
offline_only = true
allowed_egress = []
max_ttfa_ms = 3000
fallback = "degrade"     # cloud | degrade | refuse

[policy.retention]
audio = "none"           # none | local
```

## Targets and quantisation

Not every method is valid for every target, and the plan validator says so
rather than failing at conversion time:

| Target | Runtime | Quantisation accepted | Dependencies |
|---|---|---|---|
| `onnx` | ONNX Runtime | `int8`, `none` | `optimum`, `onnx`, `onnxruntime` |
| `gguf` | llama.cpp | `q4_k_m`, `int8` (Q8_0), `none` | `llama_cpp`, `gguf` |
| `mlx` | MLX, Apple silicon | `int4`, `int8`, `none` | `mlx_lm` |

Choosing among them is a portability question before it is a performance one.
ONNX Runtime runs nearly everywhere and is the safe answer for a heterogeneous
fleet. GGUF with llama.cpp is the best-supported path for CPU-only devices and
has the richest quantisation ladder. MLX is Apple silicon only and is the fast
path where the device is a Mac or an iPhone-class SoC — and, as the
[Whisper accent LoRA](whisper-accent-lora.md) guide notes, it has no training
API, which is precisely why the merge-then-convert step exists.

**On quantisation, qualitatively.** Lower precision buys memory and bandwidth,
which on a memory-bound decoder buys latency directly, and it costs accuracy.
The cost is not uniform: it lands hardest on the rare and the specific — proper
nouns, numbers, long-tail facts — which is exactly the material a voice agent
gets asked about. Sweeping the quantisation level without an evaluation set is
guessing. Run [`vmp eval golden`](evaluation-and-gates.md) at each level and
choose on measured numbers, then record which level shipped in the bundle
manifest.

This repository has not benchmarked any quantisation level. Nothing here
reports one.

## CLI

```bash
# Export
vmp edge export --config configs/edge.toml --dry-run
vmp edge export --config configs/edge.toml
vmp edge export --base-model google/gemma-3-270m --adapter .vmp/adapters/voice-sft \
    --target mlx --quant int4 --out .vmp/edge/export-mlx --dry-run

# Bundle
vmp edge bundle build --src .vmp/edge/export --out .vmp/edge/bundles/voice-agent-edge-0.1.0 \
    --name voice-agent-edge --version 0.1.0 --target gguf \
    --base-model google/gemma-3-270m --config configs/edge.toml --dry-run
vmp edge bundle verify .vmp/edge/bundles/voice-agent-edge-0.1.0
vmp edge bundle diff  .vmp/edge/bundles/voice-agent-edge-0.1.0 \
                      .vmp/edge/bundles/voice-agent-edge-0.2.0
```

`--out` for `bundle build` is the bundle directory itself, not a parent: the
files are copied directly into it alongside `manifest.json`.

## What a dry run returns

`vmp edge export --dry-run` validates the plan and returns every step it would
run, with the dependency each one needs. Real output from the config above, with
the adapter directory absent:

```json
{
  "plan": {
    "base_model": "google/gemma-3-270m",
    "target": "gguf",
    "out_dir": ".vmp/edge/export",
    "adapter_path": ".vmp/adapters/voice-sft",
    "quantization": {"method": "q4_k_m"},
    "merge_adapter": true
  },
  "dry_run": true,
  "steps": [
    {"name": "validate", "detail": "check plan fields and adapter path", "status": "skipped"},
    {"name": "merge_adapter",
     "detail": "merge LoRA adapter .vmp/adapters/voice-sft into google/gemma-3-270m",
     "requires": ["peft", "transformers"], "status": "skipped"},
    {"name": "convert_gguf", "detail": "llama.cpp convert_hf_to_gguf -> model-f16.gguf",
     "requires": ["llama_cpp", "gguf"], "status": "skipped"},
    {"name": "quantize_q4_k_m",
     "detail": "llama.cpp quantize model-f16.gguf -> model-q4_k_m.gguf",
     "requires": ["llama_cpp"], "status": "skipped"},
    {"name": "write_manifest", "detail": "export_manifest.json in out_dir", "status": "skipped"}
  ],
  "files": ["model-f16.gguf", "model-q4_k_m.gguf"],
  "problems": ["adapter_path does not exist: .vmp/adapters/voice-sft"]
}
```

Step status tells you which mode you are in: `skipped` means the plan had
problems and nothing ran, `planned` means a clean dry run, `done` and `failed`
come from a real run. A non-empty `problems` list makes the command exit 1, so
CI can dry-run every export config as a lint.

`bundle build --dry-run` computes SHA-256 of every source file and returns the
manifest that would be written, without writing anything:

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
  "created_at": 1789399653.553517,
  "required": ["model-q4_k_m.gguf", "tokenizer.json"],
  "files": {
    "model-q4_k_m.gguf": {"sha256": "a7acdcee4083198948ed726fa4c68d4dec15f931b03698aa58bb2986ac6a7022", "size": 10},
    "tokenizer.json": {"sha256": "e346432021b04179518d9614f3560ccd71354a4ee101ddcb893d6959a9d6301c", "size": 8}
  },
  "policy": {
    "offline_only": true,
    "allowed_egress": [],
    "max_ttfa_ms": 3000,
    "fallback": "degrade",
    "retention": {"audio": "none"}
  }
}
```

`lineage` is free-form registry provenance — artifact name and version, data
hash, config hash, git sha. Fill it from the `ModelArtifact` you promoted. A
bundle on a device with an empty `lineage` cannot be traced back to a training
run, which is the question asked first in every edge incident.

## Verification

`verify_bundle` is the gate the runtime passes before loading. It checks, in
order: the directory and manifest exist and parse; every mandatory manifest key
is present; every `required` file is listed and present; **every listed file's
SHA-256 and size match**; in strict mode, no unlisted file is present; the
runtime version is at least `min_runtime_version`; and the policy is internally
consistent.

```bash
vmp edge bundle verify .vmp/edge/bundles/voice-agent-edge-0.1.0
# {"ok": true, "problems": []}
```

Exit code 1 on failure, with one line per problem. The strict unlisted-file
check matters more than it looks: a file in the bundle directory that the
manifest does not describe is either a partial download or something that should
not be there, and both are reasons not to load.

`EdgeRuntime.start(require_verified=True)` refuses to start on a failed
verification. Serving from an unverified bundle because verification was
inconvenient is how an unknown model ends up in a fleet.

## The offline policy

The policy is data in the manifest, and `vmp.edge.policy.enforce` is the single
check point the runtime calls before any network call, cloud fallback or audio
write. Three action kinds:

| Action | Refused when |
|---|---|
| `network` | `offline_only = true`, or the host is not in `allowed_egress` |
| `cloud_fallback` | `offline_only = true`, or `fallback != "cloud"` |
| `store_audio` | `retention.audio = "none"` |

`validate()` rejects contradictions before they ship: `offline_only` with a
non-empty `allowed_egress`, `offline_only` with `fallback = "cloud"`, an egress
entry written as a URL rather than a host. A policy that cannot be satisfied is
caught at build time, not on a device in someone's kitchen.

`fallback` is the interesting one. `degrade` answers locally with a smaller or
simpler response; `refuse` says it cannot answer; `cloud` sends the turn
upstream and is only legal when the bundle is not offline-only. For a device
sold on "nothing leaves the room", `offline_only = true` and `fallback =
"degrade"` are the only defensible pair, and the manifest is the artifact that
proves it.

## OTA and the canary

`vmp edge bundle diff` computes exactly what an update has to ship:

```json
{
  "from": {"name": "voice-agent-edge", "version": "0.1.0"},
  "to": {"name": "voice-agent-edge", "version": "0.2.0"},
  "added": [], "changed": ["model-q4_k_m.gguf"], "removed": [],
  "unchanged": ["tokenizer.json"],
  "download_bytes": 184320000,
  "policy_changed": false
}
```

`download_bytes` is the update's real cost on a metered connection, and
`policy_changed` is the flag that must never be missed: a bundle that quietly
loosens `offline_only` or `retention.audio` is a privacy change wearing a
version bump. Treat a `true` there as requiring the same review as a legal
change, because it is one.

The rollout is a canary, not a push. Ship to a small cohort; watch
`p95_ttfa_ms_24h` and `edge_bundle_version` from the `device_features` view
(see [Feature store](feature-store.md)); promote only when the cohort's metrics
hold. The runbook for a failed verification is
[edge-bundle-verify-failed](../runbooks.md).

## What the real run needs

```bash
pip install -e ".[edge]"     # onnx, onnxruntime, optimum, plus llama-cpp-python or mlx-lm
```

Only the converters for the target you use. A real export merges the adapter
with `PeftModel.merge_and_unload()` into `<out_dir>/merged`, then converts from
that directory — every later step reads `merged` if it exists and the base model
otherwise, so an export with no adapter still works. `export_manifest.json` is
written into `out_dir` with the actual produced file list, which is what
`bundle build` then checksums.

## The small-model correctness cost

Edge pressure pushes toward the smallest model that fits, and the cost is
measurable. On six factual questions, `gemma3:270m` answered with a median LLM
latency of **192 ms** against **4,427 ms** for the 4B model, and got **2 of 6
wrong** where the larger model got **0 of 6** wrong (alpha-core, cycle 5,
2026-09-12, `notebook_optimized.ipynb`). The wrong answers were not hedges; one
was a confidently fabricated author name.

Report the speedup and the wrong-answer rate together, always. A 23× latency
improvement that invents facts is not a latency improvement, it is a different
product. The release gate exists so that this trade is made explicitly: put a
correctness gate on the edge candidate, and let it fail.

## Pitfalls

- **Merge before export.** GGUF, ONNX and MLX each load one weight set. An
  unmerged adapter exports the base model and nothing tells you.
- **A quantised model is a different model.** Re-run the evaluation after
  quantising, not before. Gate on the quantised artifact, because that is what
  ships.
- **`min_runtime_version` is a real compatibility contract.** Bump it when the
  bundle needs something the old runtime lacks; an old device then refuses the
  bundle instead of loading it and misbehaving.
- **Checksums catch corruption, not tampering.** The manifest is not signed.
  For a hostile-network threat model, sign it and verify the signature before
  the checksums.
- **`allowed_egress` takes hosts, not URLs.** `https://api.example.com` is
  rejected at validation; `api.example.com` is what it wants.
- **Audio retention is a policy decision with a legal shadow.** `none` means
  the runtime refuses to write audio at all — enforced in code, not by
  convention. Changing it is a `policy_changed: true` in the next diff, and
  should be reviewed as such.
