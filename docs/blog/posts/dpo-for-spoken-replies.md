---
date: 2026-08-27
authors: [cg]
categories:
  - Training
  - Evaluation
slug: dpo-for-spoken-replies
---

# DPO for spoken replies

Ask a chat model a question and it writes you an answer. Headings where the topic
shifts. A bulleted list for the three options. A bold phrase for the important
part. A code fence if anything technical came up. Two paragraphs of context
before the thing you asked about.

It is good writing. Read it aloud and it falls apart.

<!-- more -->

## What a TTS engine does with prose written for the eye

Markdown is a visual encoding. A speech synthesiser has no visual channel, so it
does one of two things with `**important**`: it reads the asterisks, or it drops
them and the emphasis with them. Neither is what the author meant.

A bulleted list read aloud is a sequence of sentence fragments with no
connectives and identical falling intonation. The structure that made it scannable
on a screen is precisely the structure that is inaudible. Headings become
disconnected noun phrases. A URL becomes forty seconds of spelled-out
punctuation. A table is unspeakable.

Length is the other axis. A four-paragraph answer is a fifteen-second read and a
ninety-second monologue that the listener cannot skim, cannot skip, and cannot
back up through without starting the turn again. In a voice interface, brevity is
not a style preference; it is the difference between an answer and a hostage
situation.

And then numbers. `$107` and `555-0199` read correctly by a good engine and
badly by a mediocre one. `3.5%` might come out as "three point five percent" or
"three dot five percent". The model does not know which engine is downstream.

So the target behaviour is: short, prose, one idea per sentence, no Markdown, no
URLs, numbers in a form that survives synthesis. Every instruct model can produce
that. Almost none of them *prefers* to.

## Preference, not supervision

That last sentence is the whole design argument. The behaviour is already in the
model. What is missing is a preference between two outputs it is equally capable
of producing.

That is what DPO is for. Direct Preference Optimization takes triples of
`(prompt, chosen, rejected)` and moves the policy toward `chosen` relative to a
frozen reference, with no reward model and no reinforcement learning loop. It is
a supervised objective over pairs. For "prefer this phrasing over that one", it
is a much better fit than supervised fine-tuning, which would need a corpus of
ideal answers that somebody has to write.

## The pairs come from a rule

Here is the part that makes the dataset affordable: the preference label is a
**deterministic function of the text**. No human rater, no reward model, no LLM
in the labelling loop.

`vmp.data.corpus.spoken_style_violations` scores a reply against what a TTS
engine can say out loud: Markdown syntax, bullets, URLs, more than 60 words,
more than 4 sentences, a blank-line paragraph break. A reply with zero violations
is a `chosen`. A reply to the same prompt with at least one violation is a
`rejected`. `pairs_from_style` emits every (clean, violating) combination per
prompt, so a prompt with no contrast contributes nothing and the dataset size is
reported rather than assumed.

Each pair records which rule produced it in its `source` field, so the manifest
can say what the dataset actually teaches before you train on it:

```json
{"n_train": 800,
 "pairs_by_source": {"no_markdown": 310, "shorter": 290, "sayable_numbers": 200},
 "mean_chosen_chars": 142, "mean_rejected_chars": 231}
```

Read that manifest before every run. If 80% of your pairs come from the
"shorter" rule, you are not training spoken style, you are training terseness,
and you will find out in evaluation rather than in the config.

There is a second, quieter benefit. The labelling function is the same code the
serving-side style check uses. The training signal and the production check
cannot drift apart, because they are one function.

## DPO with a PEFT adapter as its own reference

DPO needs a reference policy to compute log-probability ratios against. The naive
implementation holds two models in memory: the policy being trained and a frozen
copy.

With PEFT there is a better shape. `DPOTrainer` is given `ref_model=None`, and
the reference log-probabilities are computed by running the *same weights* with
the adapter disabled. One copy of the base model. The policy is base+adapter, the
reference is base, and the difference between them is exactly the thing being
trained.

This is what makes the no-SFT-first shape practical rather than merely cheap.
There is no separately fine-tuned checkpoint to carry around, version, and keep
in sync. The research spike (`research/02-dpo-spoken-style/`) claims precisely
that: rule-derived pairs shift style through DPO **without** a supervised
fine-tune first, because the behaviour wanted is one the base model can already
produce.

One consequence to keep straight: if you *do* start from an SFT adapter, merge it
into the base first, so that the reference is the SFT model rather than the raw
base. Otherwise the KL term is pulling against the wrong thing.

`beta` controls how far the policy may move from that reference. The default of
0.1 is TRL's; lower is more conservative. Memory is roughly double SFT at the
same batch size, because every example carries both a chosen and a rejected
sequence, which is why the default config halves `per_device_batch_size` and
doubles `gradient_accumulation`.

## Evaluating: the circularity trap

Win-rate is the natural metric. For each held-out prompt, generate with the
candidate and with the baseline, ask a judge which reply is better for speech,
report the fraction the candidate wins.

The trap is right there in "ask a judge".

If the preference pairs were generated by a rule, and then the win-rate is scored
by the same rule, the number measures whether training successfully fit the rule.
It does not measure whether the replies got better. It is a very expensive way of
confirming that gradient descent works.

`vmp.eval.judge` offers three judges and is explicit about which is which.
`RubricJudge` applies the rule checks — at most N sentences, no Markdown, no
URLs, non-empty, no forbidden phrases — and scores the fraction that pass. It is
deterministic, free, and useful as a smoke test. It is **useless as evidence**
when the model was trained on the same rubric, and the guide says so.

A real judge is an LLM, prompted with the rubric and a pairwise comparison, run
with the two answers in **both orders** and both orders scored. Judges have a
well-known preference for longer and more formatted answers unless explicitly
told the answer will be spoken aloud; the rubric has to say "for speech"
outright. And the win-rate is reported **with the judge named**, because a
win-rate without a judge identity is not comparable to anything.

## The failure mode the falsifier is aimed at

The spike's falsifier has three clauses, and the ordering is deliberate. The
first two are what you would expect: violations must fall on held-out prompts,
and correctness on the factual stratum of the golden set must not drop beyond a
stated tolerance — because a model that learned "shorter wins" will be terse in
cases where terseness loses the answer.

The third clause is the one written first on purpose:

> The same shift is obtained by the system prompt alone, with no training.

If a one-line instruction — "Answer in short spoken sentences. Do not use lists,
markdown, or symbols." — gets you there, then DPO was not needed, and the claim
is not worth the complexity even if it is technically true. The honest control
for this experiment is a prompt, not a null model.

That control is cheap to run and almost never run, because it is the one that can
make the interesting work unnecessary. Which is exactly why it belongs in the
falsifier, written down before the training starts, rather than in the discussion
section afterwards.

## Outcome: `not run`

No DPO training has happened in this repository. No base model was downloaded, no
adapter trained, no reply generated or scored. What has executed is the dataset
derivation and the training plan, both in the dry-run path. The pair counts the
spike prints describe a synthetic corpus written for the spike, and no number
from that directory may be cited as a result.

What exists is the shape: a rule that produces pairs and also checks production
output; a plan that validates and hashes; a trainer path that uses the adapter as
its own reference; a preference-statistics report you read before training; and a
judge protocol with the circularity trap documented rather than stepped in.

The experiment is ready to be performed. It has not been performed, and this post
does not claim otherwise.
