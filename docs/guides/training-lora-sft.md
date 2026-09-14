# Guide: LoRA SFT training

## Purpose

Supervised fine-tuning teaches a chat model the spoken register: short replies,
no Markdown, numbers in sayable form, one idea per sentence. It is the first
training step; [DPO](training-dpo.md) refines it afterwards. Every fine-tune in
`vmp` is a LoRA adapter over a named base model, never a full-weight update.

`vmp.training` has one description of a run — the `TrainingPlan` — and one
runner function per kind: `run_sft`, `run_dpo` and `run_whisper_lora`, each
taking a plan and a `dry_run` flag. The plan is validated eagerly in
`__post_init__` and hashed canonically (`config_hash`) so the registry can
record lineage. `output_dir` is excluded from the hash, so two runs of the same
recipe into different directories share one configuration identity.

## Config

`configs/train_sft.toml`, in full:

```toml
# Supervised fine-tuning of a small instruct model for short spoken replies.
# Load with `vmp train sft --config configs/train_sft.toml --dry-run`.

kind = "sft"
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = "runs/sft-spoken"
seed = 42

[datasets]
train = "data/sft_train.jsonl"   # rows: {"prompt": ..., "response": ...} or {"messages": [...]}
eval = "data/sft_eval.jsonl"

[lora]
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
task_type = "CAUSAL_LM"

[hyperparams]
epochs = 1
learning_rate = 2e-4
per_device_batch_size = 4
gradient_accumulation = 4
max_seq_length = 1024
warmup_ratio = 0.03
logging_steps = 10

[compute]
backend = "local"
num_workers = 1
gpu_per_worker = 0
```

A `[hyperparams]` table is merged on top of `DEFAULT_HYPERPARAMS["sft"]`, so a
config lists only what it changes. `datasets` needs a `train` entry; `eval` is
optional, and any other split name is carried through and scanned the same way.

`vmp.training.sft.format_chat_example` accepts three JSONL row shapes and
normalises all of them to chat messages:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "response": "..."}
{"user": "...", "assistant": "..."}
```

Anything else raises with the keys it actually found. Assistant turns are run
through `strip_markdown` and a spoken-style system prompt is prepended unless
the row already starts with one — the manifest reports both, so what the model
was trained on is visible without reading the code.

## CLI

```bash
vmp train plan --config configs/train_sft.toml             # validate, print the plan and its hash
vmp train sft  --config configs/train_sft.toml --dry-run   # manifest, no heavy imports
vmp train sft  --config configs/train_sft.toml             # real run (needs [train])
```

The command name must match `kind` in the config. `vmp train dpo --config
configs/train_sft.toml --dry-run` prints `error: config kind is 'sft', command
expects 'dpo'` and exits 2, rather than training the wrong thing.

There is no `vmp train merge` and no `vmp registry register`. Merging an
adapter and registering an artifact are Python APIs, covered below.

## What a dry run returns

A dict, printed as JSON on stdout. It scans every split, hashes the training
file, estimates the optimizer steps and reports the PEFT keyword arguments,
importing nothing outside the standard library.

The run below used a 12-row training split and a 4-row eval split written into
a scratch directory, with `configs/train_sft.toml` copied beside them
unmodified — which is why `config_hash` is the same value
`vmp train plan --config configs/train_sft.toml` prints from the repository.
The `plan` block, which echoes the config above, is elided:

```json
{
  "config_hash": "fa8ee03e1b2f5fd4f0c914fc8bcccc1b8ded93c7ab233d91ed9985368f0034e3",
  "data_hash": "7d83ec617a2f73165c438a76797b4044344f2525edbcd229be30f707acd4c7df",
  "datasets": {
    "eval": {
      "bytes": 325,
      "data_hash": "ec8dacd762f7bb3ecd5fcb470e3f06571c2f1faa9dc380de42d30fb0f8684a41",
      "exists": true,
      "fields": ["prompt", "response"],
      "missing_required": [],
      "path": "data/sft_eval.jsonl",
      "rows": 4
    },
    "train": {
      "bytes": 1149,
      "data_hash": "7d83ec617a2f73165c438a76797b4044344f2525edbcd229be30f707acd4c7df",
      "exists": true,
      "fields": ["prompt", "response"],
      "missing_required": [],
      "path": "data/sft_train.jsonl",
      "rows": 12
    }
  },
  "dry_run": true,
  "estimated_steps": 1,
  "formatting": "chat messages, assistant turns stripped of markdown",
  "kind": "sft",
  "peft": {
    "bias": "none",
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": "CAUSAL_LM"
  },
  "system_prompt": "You are a voice assistant. Reply in one to three short spoken sentences. Do not use markdown, lists, headings, code blocks or URLs.",
  "trainer": "trl.SFTTrainer",
  "warnings": []
}
```

Read `estimated_steps` before starting anything. The global batch is
`per_device_batch_size * gradient_accumulation * num_workers`, here 16, so 12
rows for one epoch is a single optimizer step — enough to prove the plumbing
and nothing else. A missing file is a `warning` and a row of zeroes, not an
exception, so the manifest can be printed before the data exists.

## What the real run needs

```bash
pip install -e ".[train]"      # torch, transformers, peft, trl, datasets, accelerate
```

The runner builds `peft.LoraConfig` from `lora.to_peft_kwargs()`, loads the base
model with `AutoModelForCausalLM`, and hands both to TRL's `SFTTrainer` with the
hyperparameters from the plan. Keys the installed TRL version does not accept
are dropped by `_filter_kwargs` rather than raising. At the end the adapter and
the tokenizer are written to `output_dir` and the manifest gains a `result`
block with `global_step`, `train_loss` and `adapter_path`. With
`compute.backend = "ray"` the same plan goes to `RayTrainLauncher` instead; see
[Ray](ray.md).

## Merging the adapter

Merging is `vmp.training.merge.merge_adapter(base, adapter_path, out)`, not a
subcommand. It defaults to `dry_run=True`, which reads the adapter's
`adapter_config.json` and reports what a merge would do without importing
`peft`. Against an adapter directory that does not exist yet:

```python
import json

from vmp.training.merge import merge_adapter

print(json.dumps(
    merge_adapter("Qwen/Qwen2.5-0.5B-Instruct", "runs/sft-spoken", "runs/sft-spoken-merged"),
    indent=2, sort_keys=True,
))
```

```json
{
  "adapter": {
    "adapter_path": "runs/sft-spoken",
    "base_model_name_or_path": null,
    "exists": false,
    "has_config": false,
    "lora_alpha": null,
    "peft_type": null,
    "r": null,
    "target_modules": null,
    "weight_files": []
  },
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "dry_run": true,
  "method": "peft.PeftModel.merge_and_unload",
  "out": "runs/sft-spoken-merged",
  "warnings": ["no adapter_config.json under runs/sft-spoken"]
}
```

With a real adapter present the `adapter` block carries the values from
`adapter_config.json` and `weight_files` lists the `.safetensors` files.
`merge_adapter(..., dry_run=False)` imports `peft` and `transformers`, calls
`merge_and_unload()` and writes the merged weights plus a tokenizer to `out`. A
`base` that disagrees with the adapter's own `base_model_name_or_path` is a
warning, not an error — the merge is still attempted, and the mismatch is
recorded.

The edge exporter drives the same function; see [Edge export](edge-export.md).

## Recording the artifact

The registry CLI is `vmp registry list|show|promote`. Writing a new version is
`Registry.register`, a Python call, because the lineage fields come from the
training run rather than from a shell:

```python
from vmp.registry.store import FileRegistry
from vmp.training.plan import TrainingPlan
from vmp.training.sft import run_sft
from vmp.types import ModelArtifact

manifest = run_sft(TrainingPlan.from_toml("configs/train_sft.toml"), dry_run=True)

reg = FileRegistry(".vmp/registry")
reg.register(
    ModelArtifact(
        name="spoken-llm",
        version="1",
        stage="candidate",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path="runs/sft-spoken",
        config_hash=manifest["config_hash"],
        data_hash=manifest["data_hash"],
        git_sha=None,
        metrics={},
        created_at=0.0,
    )
)
```

`register` stamps `created_at` itself and returns the stored artifact. After
that the CLI can read and move it:

```bash
vmp registry --root .vmp/registry list
vmp registry --root .vmp/registry promote --name spoken-llm --version 1 --stage staging
vmp registry --root .vmp/registry show --name spoken-llm --version 1
```

```text
spoken-llm	1	candidate	Qwen/Qwen2.5-0.5B-Instruct	fa8ee03e1b2f
spoken-llm:1 -> staging
```

```json
{
  "adapter_path": "runs/sft-spoken",
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "config_hash": "fa8ee03e1b2f5fd4f0c914fc8bcccc1b8ded93c7ab233d91ed9985368f0034e3",
  "created_at": 1789402752.8368251,
  "data_hash": "7d83ec617a2f73165c438a76797b4044344f2525edbcd229be30f707acd4c7df",
  "git_sha": null,
  "metrics": {},
  "name": "spoken-llm",
  "stage": "staging",
  "version": "1"
}
```

The `config_hash` in the artifact is the one the dry run printed, which is what
makes the link from a promoted model back to the recipe that produced it an
identity rather than a note.

Without `--root` the registry directory comes from `Settings`, overridable with
`VMP_REGISTRY_ROOT`.

## Pitfalls

- **Target modules differ per architecture.** `q_proj`/`v_proj` is right for
  Llama-style models; other families name their projections differently. The
  dry run cannot check this because it does not load the model; the real run
  fails at PEFT wrap time with a clear message.
- **Sequence length is the memory knob.** Spoken-style data is short; a
  `max_seq_length` of 512 is usually enough and halves activation memory.
- **A train-set metric is not a result.** Keep an `eval` split. The predecessor's
  accent LoRA reported 11/35 -> 32/35 on the training sentences with no
  held-out set (alpha-core, cycle 5, 2026-09-12,
  `notebook_whisper_accent_lora.ipynb`) and the caveat is part of the number.
- **Merge before export.** GGUF, ONNX and MLX load one weight set. Serving with
  vLLM can load the adapter by name instead.
- **Hash the scrubbed data.** PII scrubbing is a Python API, not a command:
  `vmp.data.pii.scrub_records` rewrites the fields you name and returns the
  counts it removed. Run it before writing the JSONL the plan points at, so
  `data_hash` attests to a scrubbed corpus.

  ```python
  >>> from vmp.data.pii import scrub_records
  >>> scrub_records(
  ...     [{"prompt": "remind me to mail ops@example.com", "response": "I will remind you."}],
  ...     fields=("prompt", "response"),
  ... )
  ([{'prompt': 'remind me to mail [EMAIL]', 'response': 'I will remind you.'}], {'email': 1})
  ```

  It covers e-mail addresses, card-like digit runs and phone numbers, in that
  order. It is a floor, not a guarantee; a human review step still belongs
  before a corpus leaves the machine that recorded it.
