# Guide: DPO training

## Purpose

Direct Preference Optimization moves a model toward replies a listener prefers,
using pairs of `(prompt, chosen, rejected)` and no reward model. In this platform
the pairs encode what "sounds right when spoken": a shorter reply over a longer
one with the same content, prose over bullets, a sayable number over a
formatted one. DPO runs after [SFT](training-lora-sft.md) on the same base and
produces a new LoRA adapter.

## Config

`configs/train_dpo.toml`:

```toml
kind = "dpo"
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
output_dir = ".vmp/runs/dpo-spoken-v1"
seed = 0

[datasets]
train = "data/spoken_pairs_train.jsonl"
eval = "data/spoken_pairs_eval.jsonl"

[lora]
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
task_type = "CAUSAL_LM"

[hyperparams]
epochs = 1
learning_rate = 5e-6
per_device_batch_size = 2
gradient_accumulation = 8
max_seq_length = 1024
max_prompt_length = 512
beta = 0.1

[compute]
backend = "local"
```

Rows are `PreferencePair` records: `{"prompt", "chosen", "rejected", "source",
"meta"}`. `source` names the rule that produced the pair.

## Building pairs

```bash
vmp data pairs --in data/spoken_candidates.jsonl --out data/spoken_pairs_train.jsonl \
    --min-margin 0.2
```

The input holds several candidate replies per prompt with a rubric score each.
The builder emits a pair for every (higher, lower) candidate whose score gap
exceeds `--min-margin`, or every pair with `--all-pairs`. The manifest reports
counts by `source`, so you can see how many pairs came from the "no Markdown"
rule versus the "shorter" rule before training on them.

## CLI

```bash
vmp training run --config configs/train_dpo.toml --dry-run
vmp training run --config configs/train_dpo.toml
vmp eval winrate --candidate .vmp/runs/dpo-spoken-v1/adapter --baseline registry:spoken-llm@production \
    --prompts data/spoken_eval_prompts.jsonl --judge stub
```

## What a dry run returns

The same manifest shape as SFT, with `kind = "dpo"`, plus pair statistics:

```json
{
  "kind": "dpo",
  "n_train": 800,
  "pairs_by_source": {"no_markdown": 310, "shorter": 290, "sayable_numbers": 200},
  "mean_chosen_chars": 142,
  "mean_rejected_chars": 231,
  "beta": 0.1,
  "estimated_steps": 50,
  "dry_run": true
}
```

## What the real run needs

```bash
pip install -e ".[train]"
```

The runner uses TRL's `DPOTrainer`. Two details matter:

- **Reference model with PEFT.** DPO computes log-probabilities under a frozen
  reference. When the policy is a PEFT model, the runner passes no separate
  reference and lets TRL use the adapter-disabled base as the reference, which
  avoids holding two full models in memory. If you start DPO from an SFT
  adapter, merge it first so that the reference is the SFT model, not the raw
  base.
- **`beta`.** Controls how far the policy may move from the reference. The
  default 0.1 is TRL's default; lower is more conservative.

## Evaluating

Win-rate is the metric: for each held-out prompt, generate with the candidate
and the baseline and ask a judge which reply is better for speech. `vmp eval
winrate` implements this with the `Judge` protocol. The stub judge applies the
same rubric the pair builder used (length, Markdown, sayability), which is
useful for a smoke test and useless as evidence, because the model was trained
on the rubric. A real judge is an LLM prompted with the rubric and a pairwise
comparison, with position swapped and both orders scored.

The release gate takes the win-rate against the current production artifact as
one of its inputs. See [Evaluation and gates](evaluation-and-gates.md).

## Pitfalls

- **Rule-derived pairs teach the rule.** A model trained on "shorter wins" will
  become terse in cases where terseness loses information. Keep a rule that
  penalises missing content, and check exact-match and factual strata on the
  golden set after DPO, not only win-rate.
- **Prompt length.** `max_prompt_length` truncates from the left by default in
  TRL; a system prompt at the start of a long prompt can be cut. Keep prompts
  short or check the truncation side.
- **Memory.** DPO holds chosen and rejected sequences per example, so effective
  batch memory is roughly double SFT at the same batch size. The defaults halve
  the per-device batch for this reason.
- **Judge bias.** Judges prefer longer, more formatted replies unless told not
  to. The rubric must say "for speech" explicitly, and win-rate should be
  reported with the judge named.
