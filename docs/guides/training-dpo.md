# Guide: DPO training

## Purpose

Direct Preference Optimization moves a model toward replies a listener prefers,
using pairs of `(prompt, chosen, rejected)` and no reward model. In this platform
the pairs encode what "sounds right when spoken": a shorter reply over a longer
one with the same content, prose over bullets, plain text over Markdown. DPO
runs after [SFT](training-lora-sft.md) on the same base and produces a new LoRA
adapter.

## Config

`configs/train_dpo.toml`, in full:

```toml
# Direct preference optimisation: spoken short replies preferred over markdown / long.
# Load with `vmp train dpo --config configs/train_dpo.toml --dry-run`.

kind = "dpo"
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = "runs/dpo-spoken"
seed = 42

[datasets]
train = "data/preference_pairs.jsonl"   # rows: {"prompt", "chosen", "rejected", "source"}

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
beta = 0.1              # KL penalty toward the adapter-disabled reference
warmup_ratio = 0.03
logging_steps = 10

[compute]
backend = "local"
num_workers = 1
gpu_per_worker = 0
```

Rows are serialised `PreferencePair` records: `{"prompt", "chosen", "rejected",
"source", "meta"}`. `source` names the rule that produced the pair.
`load_preference_pairs` rejects a row whose `prompt`, `chosen` or `rejected` is
missing, empty or not a string, and rejects a row whose `chosen` and `rejected`
are identical, naming the line number in both cases.

## Building pairs

```bash
vmp data pairs --in data/reply_candidates.jsonl --out data/preference_pairs.jsonl
vmp data pairs --in data/reply_candidates.jsonl --out data/preference_pairs.jsonl --dry-run
vmp data pairs --in data/reply_candidates.jsonl --out data/all_pairs.jsonl \
    --all-pairs --min-margin 0.4
```

`PreferencePairBuilder` reads one JSONL file and derives pairs two ways. A
record with `candidates` goes through `pairs_from_scores`; a record with
`replies` goes through `pairs_from_style`; a record with both yields both.

```json
{"prompt": "how long is the meeting",
 "candidates": [{"text": "It runs for forty-five minutes.", "score": 0.9},
                {"text": "The meeting is scheduled to run for …", "score": 0.4}]}
{"prompt": "how do I restart the service",
 "replies": ["Run the restart command, then check that the health endpoint answers.",
             "Here are the steps:\n- stop the unit\n- start the unit\n- check `/healthz`"]}
```

- **Scored candidates** (`source: "scores"`). By default one pair per prompt,
  best against worst. `--all-pairs` emits every (higher, lower) combination
  instead, and `--min-margin` drops any combination whose score gap is smaller.
  Equal scores never form a pair. `meta` records `chosen_score`,
  `rejected_score` and `margin`.
- **Spoken-style rules** (`source: "spoken_style"`). Every reply that passes
  `spoken_style_violations` is paired against every reply that fails it, and
  `meta.rejected_reasons` lists which rules failed. The rules are `markdown`,
  `bullets`, `url`, `too_long` (over 60 words), `too_many_sentences` (over
  four) and `paragraphs` (a blank line). A prompt with no passing reply, or no
  failing reply, yields nothing.

The command prints a one-line manifest. Against an eight-record input of the
shape above:

```json
{"in": "data/reply_candidates.jsonl", "out": "data/preference_pairs.jsonl", "n": 8,
 "by_source": {"scores": 2, "spoken_style": 6}, "dry_run": false}
```

`by_source` is the number worth reading before training: it says how many pairs
came from a human-ish score and how many from a mechanical style rule, and the
second kind teaches exactly that rule and nothing more.

## CLI

```bash
vmp train plan --config configs/train_dpo.toml             # validate, print the plan and its hash
vmp train dpo  --config configs/train_dpo.toml --dry-run   # manifest, no heavy imports
vmp train dpo  --config configs/train_dpo.toml             # real run (needs [train])
```

There is no `vmp eval winrate` subcommand — `vmp eval` has `wer`, `golden` and
`gate`. Win-rate is a Python API, covered under [Evaluating](#evaluating).

## What a dry run returns

The same manifest shape as SFT, with `kind = "dpo"`, plus a `pairs` block from
`preference_stats` and the reference-model note. The run below used the
eight-pair file built above, with `configs/train_dpo.toml` copied beside it
unmodified — which is why `config_hash` matches what `vmp train plan --config
configs/train_dpo.toml` prints from the repository. The `plan` block, which
echoes the config above, is elided:

```json
{
  "beta": 0.1,
  "config_hash": "b1c40cba98cf3c6f3d06ac40b0971e6d9fa93c3e8f0f39de76b235efca3cdac5",
  "data_hash": "712f3f05632e699e74de957b4d8738c6b5bb21c4a66324ea6ec4fbd968a1c7be",
  "datasets": {
    "train": {
      "bytes": 2525,
      "data_hash": "712f3f05632e699e74de957b4d8738c6b5bb21c4a66324ea6ec4fbd968a1c7be",
      "exists": true,
      "fields": ["chosen", "meta", "prompt", "rejected", "source"],
      "missing_required": [],
      "path": "data/preference_pairs.jsonl",
      "rows": 8
    }
  },
  "dry_run": true,
  "estimated_steps": 1,
  "kind": "dpo",
  "pairs": {
    "chosen_shorter_fraction": 0.875,
    "chosen_spoken_fraction": 1.0,
    "mean_chosen_words": 8.25,
    "mean_rejected_words": 18.375,
    "n": 8,
    "rejected_spoken_fraction": 0.5,
    "sources": {"scores": 2, "spoken_style": 6}
  },
  "peft": {
    "bias": "none",
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "r": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "task_type": "CAUSAL_LM"
  },
  "reference_model": "ref_model=None: DPOTrainer evaluates the reference log-probs with the LoRA adapter disabled on the policy model, so only one copy of the base weights is loaded.",
  "trainer": "trl.DPOTrainer",
  "warnings": []
}
```

The `pairs` block is the point of the dry run. `chosen_shorter_fraction` at
0.875 and a mean length of 8.25 words against 18.375 say plainly what this set
teaches: be shorter. `rejected_spoken_fraction` at 0.5 says half the rejected
replies would have passed the spoken-style rules anyway, so half the signal is
about something else. Those are the numbers to argue with before spending a GPU
hour, not after.

`pairs` is only computed when the training file exists and no row is missing a
required field; otherwise it is `{"n": 0}` and the missing file appears in
`warnings`.

## What the real run needs

```bash
pip install -e ".[train]"
```

The runner uses TRL's `DPOTrainer`. Two details matter:

- **Reference model with PEFT.** DPO computes log-probabilities under a frozen
  reference. `run_dpo` passes `ref_model=None` together with a `peft_config`,
  so TRL evaluates the reference with the adapter disabled on the policy model
  and no second copy of the base weights is loaded. The manifest states this
  verbatim in `reference_model`. If you start DPO from an SFT adapter, merge it
  first so that the reference is the SFT model, not the raw base.
- **`beta`.** Controls how far the policy may move from the reference. The
  default 0.1 is TRL's default; lower is more conservative. The plan validates
  it as a positive number before anything heavy is imported.

Rows are converted to TRL's conversational format on the way in: the prompt
becomes a system message carrying `SPOKEN_SYSTEM_PROMPT` plus the user turn, and
`chosen` and `rejected` become single assistant turns.

## Evaluating

Win-rate is the metric: for each held-out prompt, generate with the candidate
and compare against a reference answer under a judge.
`vmp.eval.preference.win_rate(pairs, policy_answers, judge)` implements it and
returns an `EvalResult`. Ties count as half a win, and `reference="chosen"`
(the default) or `"rejected"` picks which side of each pair the policy is
measured against.

```python
import json

from vmp.eval.judge import RubricJudge
from vmp.eval.preference import win_rate
from vmp.training.plan import read_jsonl
from vmp.types import PreferencePair

pairs = [PreferencePair.from_dict(r) for r in read_jsonl("data/preference_pairs.jsonl")]
result = win_rate(pairs, [p.chosen for p in pairs], RubricJudge(), reference="rejected")
print(json.dumps({"name": result.name, "metrics": result.metrics, "n": result.n,
                  "details": {k: v for k, v in result.details.items() if k != "items"}},
                 indent=2, sort_keys=True))
```

Against the same eight pairs, with the `chosen` strings standing in for
generated answers:

```json
{
  "details": {"judge": "RubricJudge", "reference": "rejected"},
  "metrics": {"losses": 0.0, "ties": 2.0, "win_rate": 0.875, "wins": 6.0},
  "n": 8,
  "name": "preference"
}
```

That 0.875 is a worked example of the circularity trap, not a result. Six of
these eight pairs were produced by the spoken-style rule and then scored by
`RubricJudge`, which applies the same rule; the number measures the rule
agreeing with itself. The full `details["items"]` list, elided above, gives the
per-prompt policy and reference scores, which is where that becomes obvious.

A judge worth quoting is an `LLMJudge` prompted with the rubric and a pairwise
comparison, run with the two answers in both orders and both scored, and
reported with the judge named. See
[Evaluation and gates](evaluation-and-gates.md).

The release gate consumes `win_rate` as one metric among several; in
`configs/gates.toml` it is declared `required = false`, so a gate set stays
usable while the measurement is still being built rather than pretending it
exists.

## Pitfalls

- **Rule-derived pairs teach the rule.** A model trained on "shorter wins" will
  become terse in cases where terseness loses information. Watch
  `by_source` and `chosen_shorter_fraction`, keep scored pairs in the mix, and
  check exact-match and the factual strata on the golden set after DPO, not
  only win-rate.
- **Prompt length.** `max_prompt_length` truncates from the left by default in
  TRL; a system prompt at the start of a long prompt can be cut. Keep prompts
  short or check the truncation side.
- **Memory.** DPO holds chosen and rejected sequences per example, so effective
  batch memory is roughly double SFT at the same batch size. The defaults halve
  the per-device batch and double the accumulation for this reason.
- **Judge bias.** Judges prefer longer, more formatted replies unless told not
  to. The rubric must say "for speech" explicitly, and win-rate should be
  reported with the judge named.
