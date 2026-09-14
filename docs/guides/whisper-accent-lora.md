# Guide: Whisper accent LoRA

## Purpose

A general speech model transcribes a general speaker. One speaker with a strong
accent is a distribution the model has seen less of, and the errors it makes are
systematic rather than random: the same word is wrong the same way every time.
A LoRA adapter on Whisper's attention projections adapts the decoder to that one
speaker from a small corpus of read speech.

This is **speaker adaptation**, not a general accent model. It is also the one
training kind in `vmp` that is a sequence-to-sequence task: the plan forces
`lora.task_type = "SEQ_2_SEQ_LM"` and rejects a config that says otherwise.

The recipe follows the predecessor project's accent-adaptation notebook, where
exact match on a 35-sentence operator-read corpus went from 11/35 to 32/35
(alpha-core, cycle 5, 2026-09-12, `notebook_whisper_accent_lora.ipynb`). That
number was measured **on the training sentences, with no held-out set**, and the
caveat travels with it everywhere it is quoted. See
[Research](../research.md) for the spike that states the falsifier this result
does not pass.

## The corpus must have clean labels

The label for each clip comes from the text the speaker was asked to read, not
from the model's own transcript of it. Fine-tuning on self-transcribed audio
teaches the model the errors it already makes. Prompt cards, read verbatim,
give a pair `(audio, text)` that is correct by construction.

One JSONL row per clip, which `vmp.training.whisper_lora.load_audio_manifest`
validates:

```json
{"audio_path": "data/accent/0001.wav", "text": "Set a timer for ten minutes.", "speaker": "op1", "duration_s": 2.4}
```

`audio_path` and `text` are required and must be non-empty; `speaker` and
`duration_s` are optional and feed the manifest statistics.

## Config

`configs/train_whisper_lora.toml`:

```toml
# LoRA on Whisper attention projections for one speaker's accent.
kind = "whisper-lora"
base_model = "openai/whisper-small.en"
output_dir = "runs/whisper-accent"
seed = 42

[datasets]
train = "data/accent_train.jsonl"   # rows: {"audio_path": ..., "text": ..., "duration_s": ...}

[lora]
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "v_proj"]
task_type = "SEQ_2_SEQ_LM"

[hyperparams]
epochs = 3
learning_rate = 1e-3
per_device_batch_size = 8
gradient_accumulation = 1
warmup_ratio = 0.05
logging_steps = 5
language = "en"
task = "transcribe"

[compute]
backend = "local"
num_workers = 1
gpu_per_worker = 0
```

`language` and `task` are passed to `WhisperProcessor.from_pretrained` and pin
the decoder prompt; they are hyperparameters here because changing either
changes what the adapter learns. The learning rate is two orders of magnitude
above the SFT default: a Whisper LoRA on a few minutes of audio sees very few
optimizer steps, and a conservative rate never leaves the initialisation.

## CLI

```bash
vmp train plan         --config configs/train_whisper_lora.toml   # validate, print, hash
vmp train whisper-lora --config configs/train_whisper_lora.toml --dry-run
vmp train whisper-lora --config configs/train_whisper_lora.toml   # real run, needs [train]
vmp eval wer --ref data/accent_ref.txt --hyp data/accent_hyp.txt  # score transcripts
```

The command name must match `kind` in the config; `vmp train sft --config
configs/train_whisper_lora.toml` fails with
`config kind is 'whisper-lora', command expects 'sft'` rather than training the
wrong thing.

## What a dry run returns

A JSON manifest on stdout. It scans the manifest, hashes the data, checks that
every referenced audio file exists, and imports nothing outside the standard
library. Abridged from an actual run against the config above, with the audio
corpus absent:

```json
{
  "kind": "whisper-lora",
  "config_hash": "1924cf7bd788a10715cf5e435d7b9276d9d4c0c7163316a051d610e0621aaaa1",
  "data_hash": null,
  "trainer": "transformers.Seq2SeqTrainer",
  "model_class": "transformers.WhisperForConditionalGeneration",
  "peft": {
    "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "task_type": "SEQ_2_SEQ_LM", "bias": "none"
  },
  "audio": {"n": 0},
  "language": "en",
  "task": "transcribe",
  "eval": ["exact_match", "wer"],
  "estimated_steps": 0,
  "datasets": {"train": {"path": "data/accent_train.jsonl", "exists": false, "rows": 0}},
  "warnings": ["dataset 'train' not found: data/accent_train.jsonl"],
  "dry_run": true
}
```

With a real manifest present, `audio` carries `n`, `total_duration_s`,
`rows_with_duration`, `missing_audio_files`, `speakers` and `mean_words`, and
`estimated_steps` is computed from the row count, batch size, accumulation and
epochs. Two warnings are worth reading before you start a run:

- `N audio files referenced by the train manifest do not exist` — the JSONL and
  the audio directory have drifted apart.
- `target_modules [...] are outside the attention projections used by the
  reference recipe ['q_proj', 'v_proj']` — you have changed the recipe, so the
  11/35 → 32/35 result no longer describes what you are about to run.

## What the real run needs

```bash
pip install -e ".[train]"      # torch, transformers, peft, trl, datasets, accelerate
```

The runner loads `WhisperForConditionalGeneration`, clears
`forced_decoder_ids` and `suppress_tokens` (leaving them set fights the
adapter), wraps the model with `peft.get_peft_model`, casts the `audio_path`
column to `datasets.Audio` at the processor's sampling rate, and trains with
`Seq2SeqTrainer`. Labels are padded and the leading decoder-start token is
stripped when every row carries it. `fp16` is enabled only when CUDA is
available. At the end the adapter and the processor are written to
`output_dir` and the manifest gains a `result` block with `global_step`,
`train_loss` and `adapter_path`.

Scoring is separate and needs no GPU: `exact_match` and `word_error_rate` in
`vmp.training.whisper_lora` operate on lists of strings after the same
normalisation (lower-case, punctuation stripped, whitespace collapsed), so
transcripts from any STT backend can be compared.

## The framework mismatch, and the conversion step it implies

The fast inference path on Apple silicon is MLX, which has no Whisper training
API — it converts and runs models, it does not fit them. Training therefore
happens in Hugging Face `transformers` with `peft`. The consequence is
structural rather than incidental: **the adapter PEFT writes does not load into
the MLX runtime.** Serving it on the fast path needs an explicit step — merge
the adapter into the base weights, then convert the merged model — which is a
piece of work with its own failure modes, not a flag. That step is
[`vmp edge export`](edge-export.md) with `target = "mlx"`.

Plan for it. An adapter that trains beautifully and cannot be loaded by the
runtime that has to serve it has not shipped anything.

## Pitfalls

- **A train-set metric is not a result.** Split the read corpus into a training
  set and a held-out set of sentences the adapter never saw, by the same
  speaker in the same session. Without that split, an improvement is
  indistinguishable from memorising the script.
- **Keep a control set.** Adaptation that helps one speaker can hurt everyone
  else. Score a general-speech set before and after and state the tolerance you
  accept.
- **Audio must be resampled, not assumed.** The dataset cast uses the
  processor's own `sampling_rate`; supplying 44.1 kHz WAVs and skipping the
  cast silently trains on the wrong spectrogram.
- **Whisper normalises text on the way out.** It returns `555-0199`, not "five
  five five oh one nine nine". Write the reference the way the model writes it,
  or the WER you measure is mostly an artifact of formatting. The predecessor's
  first golden set scored 27% WER for exactly this reason (alpha-core,
   2026-09-05, `evals/golden/README.md`).
- **Technical vocabulary is the hard stratum.** The same golden set measured
  2.69% WER overall and 28.9% on the `asr_names` stratum — gRPC, SQLite,
  OpenTelemetry (alpha-core, 2026-09-05, `evals/golden/README.md`).
  Accent adaptation does not fix vocabulary; biasing the decoder prompt might.
- **Three epochs on minutes of audio overfits quickly.** Watch the training
  loss, keep the checkpoint from the epoch that still generalises, and record
  which one you kept in the registry alongside the `config_hash`.
