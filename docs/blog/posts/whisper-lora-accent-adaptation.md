---
date: 2026-08-23
authors: [cg]
categories:
  - Training
  - Speech
slug: whisper-lora-accent-adaptation
---

# Whisper LoRA and the honest version of an accent result

A general speech model transcribes a general speaker. Give it one speaker with a
strong accent and the errors stop being random: the same word comes back wrong
the same way, every time. That is a distribution problem, and a distribution
problem responds to adaptation.

A LoRA adapter on Whisper's attention projections, trained on a few minutes of
that speaker reading sentences, took exact match on a 35-sentence corpus from
**11/35 to 32/35** (alpha-core, cycle 5, 2026-09-12,
`notebook_whisper_accent_lora.ipynb`).

That number has a caveat that is more interesting than the number.

<!-- more -->

## The caveat

It was measured **on the training sentences, with no held-out set**.

Which means the honest reading is: after training on 35 sentences, the model can
transcribe those 35 sentences. It might have learned the speaker's accent. It
might have memorised the script. From that measurement alone, the two are
indistinguishable.

This is not a footnote to be dropped when the number gets quoted. It is the
finding. And it is why the research spike for this work in
`research/01-whisper-accent-lora/` writes its falsifier the way it does:

> Word error rate rises on a **held-out** set of sentences by the same speaker
> that were not in the training corpus (adaptation to a speaker collapsed into
> memorisation of a script).

Followed by the line that makes it enforceable:

> A result on the training sentences alone cannot pass this falsifier. It can
> only fail it.

The spike's outcome is `not run`. No audio has been recorded here, no adapter
trained, no transcription measured. The 11/35 → 32/35 figure travels with its
caveat wherever it appears, and nothing downstream is allowed to cite it as a
validated result.

There is a temptation, when a repository is partly a demonstration of capability,
to let a real measurement from a related project stand in as if it were this
project's. Resisting that is most of what "evidence discipline" means in
practice.

## What a clean-label corpus is, and why it is the whole method

The adapter is only as good as the pairs it trains on, and the pairs are
`(audio, text)`. Where does the text come from?

The wrong answer, which is also the convenient answer: run the model over the
audio and use its transcripts. This fails in a specific and self-defeating way.
The model's transcripts contain exactly the errors you are trying to remove.
Training on them teaches the model to reproduce its own mistakes with more
confidence. You have built a very efficient system for entrenching a defect.

The right answer: the speaker reads from prompt cards, and each card's text *is*
the label. The pair is correct by construction, and no transcription step sits
between the speaker and the target. One JSONL row per card:

```json
{"audio_path": "data/accent/0001.wav", "text": "Set a timer for ten minutes.",
 "speaker": "op1", "duration_s": 2.4}
```

`vmp.training.whisper_lora.load_audio_manifest` validates that shape and refuses
a row with an empty `audio_path` or `text`. The dry-run manifest reports how many
referenced audio files are actually missing, because a JSONL file and an audio
directory drift apart faster than anyone expects.

Two properties of a corpus like this are worth stating explicitly. It is **one
speaker** — this is speaker adaptation, not a general accent model, and the
distinction matters when someone asks whether it "works for Irish English". And
it is **small on purpose**: tens of sentences, minutes of audio. That is what
makes the technique practical for a personal assistant, and it is also what makes
overfitting the default outcome rather than the risk.

Three epochs at a learning rate of 1e-3 — two orders of magnitude above the SFT
default — on minutes of audio will fit the training set very well. That
configuration is right, because a Whisper LoRA on this much data sees very few
optimizer steps and a conservative rate never leaves its initialisation. It also
means the held-out split is not optional.

## What you have to hold onto besides the target speaker

Adaptation that helps one speaker can quietly hurt everyone else. The third
falsifier clause covers it: the adapter must not improve the target speaker only
by degrading a general-speech control set beyond a stated tolerance.

That tolerance has to be stated *before* the run. Afterwards, whatever number
appears becomes the tolerance.

And accent adaptation does not fix the failure mode that actually hurts a
developer-facing assistant. The predecessor's golden set measured 2.69% WER
overall and **28.9% on the proper-noun stratum** — gRPC as "JU-RPC", SQLite as
"SQ Light", OpenTelemetry as "open telemetry" (alpha-core, 2026-09-05,
`evals/golden/README.md`). Those are not accent errors; they are vocabulary
errors, ten times worse than the headline rate, in exactly the stratum an agent
gets asked about when someone asks it about its own stack. Different problem,
different fix — decoder prompt biasing is the thing to try — and a reason to
always read `per_category` rather than the overall number.

## The framework mismatch

Here is the part that is structural rather than incidental, and the part most
likely to be discovered at the worst moment.

The fast inference path on Apple silicon is MLX. It is genuinely fast: the
predecessor measured mlx-whisper on an M4 GPU at a **233 ms** median against
**1,761 ms** for CPU faster-whisper on identical audio, same model size class —
7.55× (alpha-core, cycle 5, 2026-09-12,
`spikes/streaming-whisper-m4/compare_stt.py`). The reason is not mysterious:
CTranslate2, faster-whisper's backend, has no Metal device on macOS at all. Only
`cpu` and `cuda`. There is no flag that puts that engine on the machine's GPU.

But MLX converts and runs Whisper. It does not train it. There is no trainer.

So training happens somewhere else: Hugging Face `transformers`, with
`WhisperForConditionalGeneration` and `Seq2SeqTrainer`, and `peft` supplying the
LoRA modules. Which produces a PEFT adapter. Which **does not load into the MLX
runtime**.

This is not a bug in either framework. It is two ecosystems with different
serialisation, and the consequence is a step in the pipeline that has to exist:
merge the adapter into the base weights, then convert the merged model to the
target runtime. In this platform that is `vmp edge export --target mlx`, and it
is the reason the export module lists `merge_adapter` as its own step with its
own dependencies (`peft`, `transformers`) rather than treating it as a flag.

The general lesson, which applies well beyond Whisper: **the training framework
and the inference framework are separate choices, and the conversion between them
is a piece of work with its own failure modes.** Plan for it at design time. An
adapter that trains beautifully and cannot be loaded by the runtime that has to
serve it has not shipped anything.

The dry run makes the plan visible before anything heavy is imported:

```json
{"name": "merge_adapter",
 "detail": "merge LoRA adapter .vmp/adapters/voice-sft into openai/whisper-small.en",
 "requires": ["peft", "transformers"], "status": "planned"}
```

## What the platform enforces

The `whisper-lora` training kind forces `lora.task_type = "SEQ_2_SEQ_LM"` and
rejects a config that says otherwise — Whisper is an encoder-decoder and a plan
that calls it causal is wrong in a way that is easy to write and annoying to
debug. The dry run warns when `target_modules` strays outside `q_proj` and
`v_proj`, because the 11/35 → 32/35 result describes that specific recipe and
stops describing whatever you changed it to.

And the scorers — `exact_match`, `word_error_rate`, `normalize_text` — operate on
lists of strings with no torch dependency, so transcripts from any STT backend
can be scored by the same code that scores the adapter. That matters more than it
sounds: the measurement path has to be usable before the training path exists,
or the first thing you do after a successful training run is write an evaluation
harness in a hurry.

## What would make this a result

Record the corpus with a held-out split — same speaker, same session, same
recording setup, sentences the adapter never sees. Score exact match and WER on
both halves. Score a general-speech control set before and after, against a
tolerance stated in advance. Report all four numbers together.

Then the claim is either true or falsified, and either outcome is worth more than
the number this post started with.
