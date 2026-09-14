# 01 — Whisper accent LoRA

## Claim

A LoRA adapter on the attention projections (`q_proj`, `v_proj`) of Whisper
`small.en` adapts transcription to one speaker's accent from a small corpus of
read speech — on the order of tens of sentences, minutes of audio — and the
labels for that corpus must come from the text the speaker was asked to read,
not from the model's own transcripts.

The second half of the claim is the load-bearing half. Fine-tuning on
self-transcribed audio teaches the model the errors it already makes; the
corpus is only useful if the label is independent of the model.

## Falsifier

Any of the following, on the same speaker and the same recording setup:

- exact match on the read sentences does not improve after adaptation, or
  improves by less than the run-to-run variance of the baseline decoder;
- word error rate rises on a **held-out** set of sentences by the same speaker
  that were not in the training corpus (adaptation to a speaker collapsed into
  memorisation of a script);
- the adapter improves the target speaker only by degrading a general-speech
  control set beyond a stated tolerance.

A result on the training sentences alone cannot pass this falsifier. It can
only fail it.

## Method

**Clean-label corpus.** The speaker reads from prompt cards. Each card's text is
recorded verbatim as the label, so the pair `(audio, text)` is correct by
construction and no transcription step sits between the speaker and the target.
One JSONL row per card: `{"audio_path", "text", "speaker", "duration_s"}`, the
shape `vmp.training.whisper_lora.load_audio_manifest` validates. The corpus is
one speaker only: this is speaker adaptation, not a general accent model.

**Framework mismatch.** The inference path this work came from runs Whisper
through MLX on Apple silicon, which is fast but has no training API — MLX
exposes conversion and inference for Whisper, not a trainer. Training therefore
happens in a second framework: Hugging Face `transformers`
(`WhisperForConditionalGeneration` + `Seq2SeqTrainer`) with `peft` supplying the
LoRA modules. The consequence is the part that must be planned for rather than
discovered: **the adapter that comes out of PEFT does not load into the MLX
runtime.** Serving it on the fast path needs an explicit conversion step —
merge the adapter into the base weights, then convert the merged model — and
that step is a separate piece of work with its own failure modes, not a flag.

**What the spike script does.** `spike_whisper_lora_plan.py` runs with no heavy
dependencies installed:

1. builds the real `TrainingPlan` from `configs/train_whisper_lora.toml` and
   prints the LoRA targets, the effective hyperparameters and the `config_hash`;
2. writes a clean-label manifest for 16 read sentences and prints
   `run_whisper_lora(plan, dry_run=True)` — field validation, data hash,
   estimated optimizer steps, and the warning that the audio files are absent;
3. scores a **synthetic** before/after transcript set with the real
   `exact_match` and `word_error_rate` helpers.

The transcripts in step 3 are hand-written to exercise the scorer. The script
prints 5/16 before and 14/16 after, and prints, next to those numbers, that they
measure nothing. Step 3 exists so that the evaluation code is known to run
before any audio is recorded.

## Outcome

`not run`

This repository has not recorded audio, trained an adapter or measured a
transcription. Nothing downstream may cite a transcription result from here.

The claim is carried over from a measurement in the predecessor project: LoRA on
Whisper attention projections took exact match on a 35-sentence operator-read
corpus from **11/35 to 32/35** (alpha-core, cycle 5, 2026-09-12,
`notebook_whisper_accent_lora.ipynb`). That number was measured **on the
training set, with no held-out set**, so it does not pass the falsifier above —
it is the reason the falsifier is written the way it is. The caveat travels with
the number wherever it is quoted.

`notebook_whisper_accent_lora.ipynb` in this directory is the unexecuted
procedure: corpus, recording, baseline, fine-tune, evaluation, caveats. It is a
plan, not a record.

## What moved into src/

- `vmp.training.whisper_lora` — `run_whisper_lora` (dry-run manifest and the
  `Seq2SeqTrainer` path), `load_audio_manifest` and `audio_manifest_stats` for
  the clean-label corpus shape, and `exact_match` / `word_error_rate` /
  `normalize_text` as the scorers.
- `vmp.training.plan` — the `whisper-lora` kind, which forces
  `lora.task_type = "SEQ_2_SEQ_LM"` and warns when `target_modules` strays
  outside the attention projections this recipe was measured on.
- `configs/train_whisper_lora.toml` — the recipe as configuration.
- `vmp.edge.export` — the conversion step the framework mismatch forces: merge
  the adapter, then convert to the target runtime (`mlx`, `onnx` or `gguf`).
