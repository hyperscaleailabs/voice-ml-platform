# Research

Stage 1 of the [progression](progression.md). `research/` holds the hypotheses
that came before the package. Each one is a **spike**: the smallest runnable
thing that can decide whether an idea is worth building properly. A spike
answers one question and is then finished.

Every `research/<nn>-<topic>/README.md` has exactly five sections — `## Claim`,
`## Falsifier`, `## Method`, `## Outcome`, `## What moved into src/` — and the
outcome is one of four values:

| Outcome | Meaning |
|---|---|
| `not run` | The falsifier has not been executed. **Nothing downstream may cite a result.** |
| `run — falsifier passes` | Executed, the falsifier did not fire; the claim survives under the stated conditions. |
| `run — falsified` | Executed, the falsifier fired; the claim is false as stated. |
| `run — inconclusive` | Executed, but the observation does not decide either way; the reason is recorded. |

Three rules keep a spike honest. A spike is **never imported by `src/`** — the
dependency runs one way, so the spike stays disposable. A spike is **frozen once
its outcome is written**; if the idea survives, the implementation starts again
from the codebase, not from the spike's diff. And a spike **ships something
runnable**, so someone else can re-run the falsifier without the author.

## Index

| Spike | Claim (short form) | Outcome |
|---|---|---|
| 01 | Whisper LoRA adapts to one speaker's accent, from clean labels | `not run` |
| 02 | Rule-derived pairs shift reply style through DPO, without SFT first | `not run` |
| 03 | One `TrainingPlan` submits unchanged to Ray Train with N workers | `not run` |
| 04 | A smaller model trades correctness for latency, inseparably | `not run` |

**All four are `not run`.** The spike scripts execute — they run with nothing but
the standard library, using the package's dry-run and planning paths, and they
compute real numbers over synthetic data. What they have not done is train a
model, transcribe real speech, or start a cluster. Numbers those scripts print
are properties of the corpora they generate, labelled as synthetic everywhere
they appear, and none of them is a result.

Where a claim came from a measurement, that measurement belongs to the private
predecessor project and is cited as such. It is never presented as this
repository's benchmark.

---

## 01 — Whisper accent LoRA

**Claim.** A LoRA adapter on the attention projections (`q_proj`, `v_proj`) of
Whisper `small.en` adapts transcription to one speaker's accent from a small
corpus of read speech — tens of sentences, minutes of audio — and the labels for
that corpus must come from the text the speaker was asked to read, not from the
model's own transcripts.

The second half is the load-bearing half. Fine-tuning on self-transcribed audio
teaches the model the errors it already makes; the corpus is only useful if the
label is independent of the model.

**Falsifier.** Any of: exact match on the read sentences does not improve beyond
the baseline decoder's run-to-run variance; word error rate **rises on a
held-out set** of sentences by the same speaker that were not in the training
corpus; or the adapter improves the target speaker only by degrading a
general-speech control set beyond a stated tolerance. A result on the training
sentences alone cannot pass this falsifier — it can only fail it.

**Outcome: `not run`.** No audio has been recorded, no adapter trained, no
transcription measured in this repository.

The claim is carried over from the predecessor, where LoRA on Whisper attention
projections took exact match on a 35-sentence operator-read corpus from
**11/35 to 32/35** (alpha-core, cycle 5, 2026-09-12,
`notebook_whisper_accent_lora.ipynb`). That number was measured **on the
training set, with no held-out set**, so it does not pass the falsifier above —
which is exactly why the falsifier is written the way it is. The caveat travels
with the number wherever it is quoted.

The spike also records a consequence that has to be planned for rather than
discovered: the fast inference path (MLX on Apple silicon) has no training API,
so training happens in Hugging Face `transformers` with `peft`, and **the
resulting adapter does not load into the MLX runtime**. Serving it needs an
explicit merge-then-convert step with its own failure modes.

**What moved into `src/`.** `vmp.training.whisper_lora` (`run_whisper_lora`,
`load_audio_manifest`, `audio_manifest_stats`, `exact_match`,
`word_error_rate`, `normalize_text`); the `whisper-lora` kind in
`vmp.training.plan`, which forces `lora.task_type = "SEQ_2_SEQ_LM"` and warns
when `target_modules` strays outside the measured recipe;
`configs/train_whisper_lora.toml`; and the conversion step in
`vmp.edge.export`.

Guide: [Whisper accent LoRA](guides/whisper-accent-lora.md).

---

## 02 — DPO for spoken style

**Claim.** Preference pairs derived by a **rule** — for one prompt, a short
spoken reply is preferred over a Markdown or multi-paragraph one — shift a small
instruct model's reply style through DPO **without a supervised fine-tune
first**. Two assertions: the preference label can come from a deterministic
function of the text, so no human rater and no reward model is needed; and DPO
straight onto the base model plus a LoRA adapter suffices, because the wanted
behaviour is a style the base model can already produce and merely does not
prefer.

**Falsifier.** On a held-out set of prompts the pairs were not built from: the
rate of spoken-style violations does not fall relative to the base model; or
replies become short at the cost of answering, with correctness on the factual
subset of the golden set dropping beyond a stated tolerance; or **the same shift
is obtained by the system prompt alone, with no training**.

The third clause is written first on purpose. It is the one most likely to fire,
and the honest control for this experiment is a prompt, not a null model. If a
one-line instruction gets there, DPO was not needed even if the claim is
technically true.

**Outcome: `not run`.** No base model was downloaded, no adapter trained, no
reply generated or scored. Only the dataset derivation and the training plan
have been executed, both in the dry-run path. The pair counts the spike prints
describe a synthetic corpus written for the spike.

**What moved into `src/`.** `vmp.data.corpus` (`spoken_style_violations`,
`is_spoken_style`, `pairs_from_style`, `pairs_from_scores`,
`PreferencePairBuilder`); `vmp.types.PreferencePair`, whose `source` field
records which derivation produced each pair so a mixed dataset stays auditable;
`vmp.training.dpo` (`run_dpo`, `load_preference_pairs`, `preference_stats`);
`configs/train_dpo.toml`; and `vmp.eval.preference` (`win_rate`,
`dpo_margins`).

One design detail from this spike is worth repeating because it is what makes
the no-SFT shape practical: with PEFT, `DPOTrainer` is given `ref_model=None`
and computes reference log-probabilities by running the same weights with the
adapter disabled. The base model is its own reference, and only one copy of the
weights is loaded.

Guide: [DPO training](guides/training-dpo.md).

---

## 03 — Ray scaling

**Claim.** The same `TrainingPlan` that runs on one machine submits unchanged to
Ray Train with more than one worker. Scaling out is a change to the `[compute]`
table of a TOML file — `backend`, `num_workers`, `gpu_per_worker` — and not a
rewrite, a second entry point, or a separate distributed script. The runners stay
single-process code; Ray Train supplies the process group and the
`RANK`/`WORLD_SIZE` environment the Hugging Face trainers already read.

**Falsifier.** The plan needs any edit beyond `[compute]` to submit; or the
runner needs a branch on `compute.backend` (if `run_sft` has to know it is on a
cluster, the plan is not the interface, the runner is); or a 4-worker run does
not converge to a comparable loss on the same data within the same epochs after
accounting for effective batch size; or the job fails on a real cluster for a
reason that is not a resource shortage — a serialisation failure, a missing
import on the worker, a path that only exists on the submitting machine.

**Outcome: `not run`.** There is no Ray cluster here and none was started. The
dry-run path executes on both sides: the plan serialises, the launcher builds a
complete `TorchTrainer` call from it as plain data, and the only differences
between the two manifests are the compute block and what derives from it. That
is evidence for the *shape* of the claim, not for the claim. Nothing here may be
cited as a scaling result, a throughput figure, or a speed-up.

One substantive finding is available without a cluster, and it is a trap worth
knowing: **`estimated_steps` falls when workers rise**, because the global batch
is `per_device_batch_size × gradient_accumulation × num_workers`. Same data,
larger effective batch, fewer optimizer steps. Anyone scaling workers without
touching the learning rate or the epoch count is changing the optimisation, and
the manifest says so before the job starts. `config_hash` also changes, because
`compute` is part of the plan and therefore part of the lineage — a 4-worker run
is deliberately not recorded as the same run as a local one.

**What moved into `src/`.** `vmp.training.plan.ComputeConfig` with its
validation and its inclusion in `config_hash`;
`vmp.training.ray_jobs.RayTrainLauncher` (`scaling_config`, `run_config`,
`build_config`, `submit`); `train_loop_per_worker`; `RayDataPreprocessor`; and
`estimate_steps`, which is what makes the step-count effect visible in a dry run.

Guide: [Ray](guides/ray.md).

---

## 04 — Small model trade-off

**Claim.** Swapping the language model for a smaller one cuts turn latency
substantially and costs answer correctness, and the two effects are coupled
tightly enough that **neither may be reported without the other**. A latency
figure for a model tier that does not carry its correctness figure from the same
run and the same question set is not a result; it is a selection.

Operationally: the smaller tier is a legitimate degradation step for a voice
agent under load, and it is only legitimate while the trade is visible at the
moment it is made.

**Falsifier.** A smaller model is found that is faster **and** not measurably
worse on the same question set beyond its sampling error; or correctness does
not move at all between tiers on a set large enough to detect a difference,
meaning the coupling was an artefact of a six-item set; or latency and
correctness turn out to be separable in practice — prompt changes, retrieval, or
a verification pass recovering the large model's correctness at the small
model's latency, in which case the trade is against effort, not against size.

**Outcome: `not run`.** No model of any size has been run here. Only the scoring
harness has been executed, over a file it generated itself.

The claim comes from the predecessor, comparing two tiers of one model family on
a small factual set: **gemma3:270m against gemma3 4B on 6 factual questions —
llm median 192 ms against 4,427 ms; 2/6 wrong against 0/6** (alpha-core, cycle 5,
2026-09-12, `notebook_optimized.ipynb`). Six questions is a small set. The
latency gap is large enough that its direction is not in doubt; the correctness
figure is a difference of two items and its precision should not be overstated.
That caveat travels with the number, and it is why the first falsifier clause is
written in terms of sampling error.

The harness enforces the coupling structurally: the output is one table with one
row per model tier, in which `p50`, `p95` and accuracy are columns of the same
row. The table cannot be quoted half-way.

**What moved into `src/`.** `vmp.eval.wer` (`exact_match`, `normalise` with
number-word mapping, `corpus_wer`); `vmp.eval.latency` and
`vmp.observability.slo.percentile`, so the latency side comes from the same
traces production emits; `vmp.eval.gates`, where the coupling is enforced — a
`GateDecision` evaluates latency and correctness together and a run that improves
one while breaching the other does not pass; the `ttft_ms` and `response_ms`
trace payloads; and `configs/gates.toml` with `configs/slo.toml`.

Guides: [Evaluation and gates](guides/evaluation-and-gates.md),
[Edge export](guides/edge-export.md).

---

## Why the outcomes are what they are

A reader could reasonably ask why a repository built around hypothesis-driven
development ships four spikes that have not been run.

Because the alternative is worse. Each of these four claims has a real
measurement behind it, made in the predecessor project on one laptop, and each
of those measurements has a caveat that makes it insufficient: the accent LoRA
had no held-out set; the small-model comparison had six questions; the Ray claim
has never met a cluster; the DPO claim has never been trained. Writing `run —
falsifier passes` against any of them because a related number exists elsewhere
would be the exact failure the spike format is designed to prevent.

`not run` is a status, and it is an honest one. It says the code path executes,
the plan validates, the evaluation harness works, and the experiment is ready to
be performed by whoever has the hardware and the corpus. What it does not say is
that anything has been proven here.

See [Hypothesis-driven development for ML platforms](whitepapers/hypothesis-driven-development-for-ml-platforms.md)
for the loop this stage belongs to.
