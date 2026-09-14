# 02 — DPO for spoken style

## Claim

Preference pairs derived by a **rule** — for one prompt, a short spoken reply is
preferred over a markdown or multi-paragraph one — shift a small instruct model's
reply style through DPO **without a supervised fine-tune first**.

Two things are asserted. First, that the preference label can come from a
deterministic function of the text, so no human rater and no reward model is
needed to build the dataset. Second, that DPO applied straight to the base model
plus a LoRA adapter is enough; the usual SFT-then-DPO sequence is not required
for a change this narrow, because the behaviour wanted is a style the base model
can already produce and merely does not prefer.

## Falsifier

Any of the following, on a held-out set of prompts the pairs were not built from:

- the rate of spoken-style violations in the model's replies does not fall
  relative to the base model — the rule is the metric, so this is measurable
  without a judge;
- replies become short at the cost of answering: correctness on the factual
  subset of the golden set drops beyond a stated tolerance, meaning the adapter
  learned brevity rather than spoken style;
- the same shift is obtained by the system prompt alone, with no training. If a
  one-line instruction gets there, DPO was not needed and the claim is not worth
  the complexity even if it is technically true.

The third clause is the one most likely to fire and it is written first on
purpose: the honest control for this experiment is a prompt, not a null model.

## Method

**Rule-derived pairs.** `vmp.data.corpus.spoken_style_violations` scores a reply
against what a TTS engine can say out loud: markdown syntax, bullets, URLs, more
than 60 words, more than 4 sentences, or a blank-line paragraph break. A reply
with no violations is a `chosen`; a reply to the same prompt with at least one
violation is a `rejected`. `pairs_from_style` emits every (clean, violating)
combination per prompt, so a prompt with no contrast contributes nothing and the
dataset size is reported rather than assumed.

Nothing in that path needs a model. The labelling function is the same code the
serving-side style check uses, which means the training signal and the
production check cannot drift apart.

**Why no SFT stage.** DPO needs a reference policy. With PEFT, `DPOTrainer` is
given `ref_model=None` and computes reference log-probs by running the same
weights with the adapter disabled, so the base model is its own reference and
only one copy of the weights is loaded. That is what makes the no-SFT shape
practical rather than merely cheap: there is no separately fine-tuned checkpoint
to carry around as the reference.

**What the spike script does.** `spike_dpo_pairs.py` runs with no heavy
dependencies installed:

1. takes 18 prompts from `vmp.data.synthetic.generate_golden_set`
   (`factual_short`, `tool_intent`, `instruction_format`, seed 7);
2. attaches three candidate replies to each — spoken, markdown, multi-paragraph
   — and derives pairs with `PreferencePairBuilder`, giving 36 pairs;
3. prints `preference_stats`: mean chosen length 12.1 words against 70.1 for
   rejected, `chosen_shorter_fraction` 1.000, and the histogram of which rules
   fired on the rejected side;
4. prints two pairs in full, writes the pairs as JSONL, reads them back with
   `load_preference_pairs`, and prints `run_dpo(plan, dry_run=True)` for
   `configs/train_dpo.toml`.

Those counts describe the synthetic corpus the script builds. They say the
derivation works and the plan validates; they say nothing about a model.

## Outcome

`not run`

No DPO training has been run in this repository: no base model was downloaded,
no adapter was trained, and no reply was generated or scored. The dataset
derivation and the training plan are the only things that have been executed,
and both run in the dry-run path.

The pair counts the spike prints are properties of a synthetic corpus written
for this spike, not a measurement of anything. No number from this directory may
be cited as a result.

`notebook_dpo_spoken_style.ipynb` is the unexecuted procedure: corpus, pair
derivation, baseline, DPO run, evaluation against the rule and against
correctness, caveats.

## What moved into src/

- `vmp.data.corpus` — `spoken_style_violations` and `is_spoken_style` as the
  rule, `pairs_from_style` and `pairs_from_scores` as the two derivations, and
  `PreferencePairBuilder` over a mixed corpus.
- `vmp.types.PreferencePair` — the serialised pair, with `source` recording
  which derivation produced it so a mixed dataset stays auditable.
- `vmp.training.dpo` — `run_dpo` (dry-run manifest and the TRL `DPOTrainer`
  path), `load_preference_pairs` with its identical-sides check, and
  `preference_stats` as the pre-training description of what the pairs teach.
- `configs/train_dpo.toml` — the recipe, including `beta` and the LoRA targets.
- `vmp.eval.preference` — `win_rate` and `dpo_margins`, the evaluation side the
  falsifier needs.
