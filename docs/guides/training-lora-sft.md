# Guide: LoRA SFT training

## Purpose

Supervised fine-tuning teaches a chat model the spoken register: short replies,
no Markdown, numbers in sayable form, one idea per sentence. It is the first
training step; [DPO](training-dpo.md) refines it afterwards. Every fine-tune in
`vmp` is a LoRA adapter over a named base model, never a full-weight update.

`vmp.training` exposes one description of a run, the `TrainingPlan`, and a
`Runner` per kind (`sft`, `dpo`, `whisper-lora`). The plan is validated eagerly
and hashed canonically so the registry can record lineage.

## Config

`configs/train_sft.toml`:

```toml
kind = "sft"
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = ".vmp/runs/sft-spoken-v1"
seed = 0

[datasets]
train = "data/spoken_sft_train.jsonl"
eval = "data/spoken_sft_eval.jsonl"

[lora]
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
task_type = "CAUSAL_LM"

[hyperparams]
epochs = 2
learning_rate = 2e-4
per_device_batch_size = 4
gradient_accumulation = 4
max_seq_length = 1024
warmup_ratio = 0.03
logging_steps = 10

[compute]
backend = "local"      # or "ray"
num_workers = 1
gpu_per_worker = 1.0
```

A `[hyperparams]` table is merged on top of the defaults for its kind, so a
config lists only what it changes. Dataset rows are JSONL with either
`{"prompt": ..., "completion": ...}` or `{"messages": [...]}`.

## CLI

```bash
vmp training plan  --config configs/train_sft.toml            # validate, print the plan
vmp training run   --config configs/train_sft.toml --dry-run  # manifest, no heavy imports
vmp training run   --config configs/train_sft.toml            # real run (needs [train])
vmp training merge --adapter .vmp/runs/sft-spoken-v1/adapter --out .vmp/runs/sft-spoken-v1/merged
vmp registry register --name spoken-llm --adapter .vmp/runs/sft-spoken-v1/adapter \
    --config configs/train_sft.toml --data data/spoken_sft_train.jsonl
```

## What a dry run returns

A dict, printed as JSON. It scans the dataset, estimates steps, and reports
the hashes, importing nothing outside the standard library:

```json
{
  "kind": "sft",
  "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
  "config_hash": "…",
  "data_hash": "…",
  "n_train": 1200,
  "n_eval": 120,
  "estimated_steps": 150,
  "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "task_type": "CAUSAL_LM"},
  "compute": {"backend": "local", "num_workers": 1, "gpu_per_worker": 1.0},
  "dry_run": true
}
```

The numbers above illustrate the shape; the real values come from your files.

## What the real run needs

```bash
pip install -e ".[train]"      # torch, transformers, peft, trl, datasets, accelerate
```

The runner builds `peft.LoraConfig` from `lora.to_peft_kwargs()`, wraps the base
model, and hands it to TRL's `SFTTrainer` with the hyperparameters from the
plan. At the end it writes the adapter, an `EvalResult` with training and eval
loss, and a `ModelArtifact` record with `config_hash`, `data_hash` and the
current `git_sha`. With `compute.backend = "ray"` the same runner is wrapped in
Ray Train; see [Ray](ray.md).

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
- **Hash the scrubbed data.** Run `vmp data scrub` before training so
  `data_hash` attests to a PII-free corpus.
