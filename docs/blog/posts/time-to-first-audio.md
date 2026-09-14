---
date: 2026-08-19
authors: [cg]
categories:
  - Latency
  - Observability
slug: time-to-first-audio
---

# Time to first audio

Ask someone what makes a voice assistant feel fast and they will not describe
tokens per second. They will describe the pause. The gap between finishing their
sentence and hearing the first syllable of the reply is the entire perceptual
experience of latency, and everything after it is invisible as long as the audio
keeps flowing.

So that is the metric: **time to first audio**, stamped as `response_ms` in the
`playback` payload of the first spoken sentence.

<!-- more -->

## Serial against streaming

The obvious implementation synthesises the answer and then plays it. Generate all
eight sentences, hand them to the TTS, play the result. Time to first audio is
then the sum of generating every sentence plus synthesising every sentence — it
grows with the length of the answer. A thorough reply is punished.

The streaming implementation changes *when the first sound arrives*, not how long
the whole answer takes. The LLM streams tokens; a segmenter cuts a sentence the
moment it is provably complete; that sentence is synthesised and played while the
model is still writing the next one. Time to first audio becomes bounded by the
first sentence: time to first token, plus generating one sentence, plus
synthesising it.

Measured on an eight-sentence answer with playback stubbed to run in real time:

| Path | Time to first audio |
|---|---|
| serial | 11,049 ms |
| streaming | 2,688 ms |

(alpha-core, cycle 5, 2026-09-12, `README.md` "Streaming".)

Both paths write the same trace, so `playback.response_ms` is directly
comparable between them — which is the only reason the comparison is worth
anything. This repository keeps a serial mode for the same A/B purpose; it has
not been benchmarked here.

Notice what the streaming number is *not*. The whole answer does not arrive
faster. The total work is the same, possibly slightly more. What changed is that
the user stops waiting at 2.7 seconds instead of 11.

## Three conditions that make it safe rather than merely fast

**Synthesis must outrun playback.** Once the first sentence is playing, the queue
has exactly one sentence of runway. If synthesis is slower than real time, the
audio stutters and the fast start turns into a worse experience than the slow
one. Kokoro's measured real-time factor was 0.105–0.157 (same source), roughly
eight times faster than playback, so the queue does not starve. `tts.queue_wait_ms`
is the signal if it ever does.

**Playback must be gapless.** One long-lived output stream, sentences appended to
it. `underruns` and `gap_ms_max` in the playback summary are the regression
guard: non-zero means the listener heard a seam.

**The segmenter must never emit a truncated sentence.** This is the one that
deserves its own section.

## Why a wrong sentence boundary is unrecoverable

In a text interface, a bad line break is a cosmetic defect. You see the whole
paragraph anyway.

In a voice interface, the sentence is *spoken* before anything downstream can
reconsider. If the segmenter cuts "the value is 3." and hands it to the TTS, the
user hears "the value is three", full stop, spoken with falling intonation — and
then hears ".5 percent" as a separate utterance. The first fragment cannot be
unspoken. The prosody is wrong, the meaning is wrong, and the user has already
formed an impression before the correction arrives.

So the segmenter emits only when a sentence is **provably** complete: terminal
punctuation (`.`, `!`, `?`, `…`), optionally followed by closing quotes or
brackets, then whitespace. And critically, a boundary at the very end of the
buffer is *not* emitted until `flush()`, because the next token could continue
it. `3.` becomes `3.5`. `Dr.` becomes `Dr. Okonkwo`. The segmenter would rather
wait one token than guess.

It knows abbreviations — `Dr.`, `e.g.`, `i.e.`, `a.m.`, single-letter initials —
and treats an ellipsis as a boundary only when the next word starts a new
sentence (uppercase, digit, or an opening quote). Nothing is ever truncated: text
that is not a complete sentence stays buffered until the stream ends.

This is why the segmenter has its own test suite and why it is the one component
shared verbatim from research through the cloud runtime to the edge runtime. Its
correctness is not a nice-to-have that improves quality; it is the precondition
for streaming being safe at all. Every other component can be swapped behind a
protocol. This one is the same code everywhere, because the same sentence must
be cut the same way everywhere.

## What a trace row holds

The measurement only exists because a row stamps it. Every stage writes a start
and an end:

```json
{"ts": 1789399628.622, "session": "84acc7bd…", "turn": 1, "event": "playback.end",
 "span": "e63be86e2caa", "seq": 1, "ms": 0.063, "payload": {"response_ms": 1.021}}
```

| Key | What it carries |
|---|---|
| `ts` | wall-clock time of the row |
| `session`, `turn` | which conversation, which exchange |
| `event` | `<stage>.start` or `<stage>.end` |
| `span` | id pairing a start with its end; nested spans record `parent` in the payload |
| `seq` | sentence number, for the per-sentence stages |
| `ms` | duration, on the `.end` row only |
| `payload` | stage-specific fields |

`seq` is what makes the streaming design legible. `segment.emit seq=3`,
`tts seq=3` and `playback seq=3` are the same sentence at three points in its
life, so you can ask "how long did the third sentence wait for the synthesiser"
and get an answer. `response_ms` appears on `playback.end` at `seq=1` and nowhere
else, because there is only one first audio.

The payloads worth watching, per stage:

| Stage | Payload |
|---|---|
| `listen` | `vad_wait_ms`, `endpoint_ms`, `audio_s`, `end_reason` |
| `stt` | `audio_s` — against `ms`, that is the real-time factor |
| `retrieve` | which path fired, chunks kept, characters injected |
| `llm` | `ttft_ms` — the floor on everything downstream |
| `segment.emit` | `seq`, `chars`, `since_ttft_ms` |
| `tts` | `chars`, `audio_s`, `queue_wait_ms` |
| `playback` | `response_ms` on `seq=1`, `buffered_s`, `underruns` |
| `turn` | `outcome` |

An exception inside a span stamps `payload.error` with the exception type and
re-raises, so a failed turn is a row rather than a gap. That is what lets
`error_rate` count turns that crashed as well as turns that reported an error —
and turns with a `turn.start` and no `turn.end` count too.

## Reading it

`vmp obs summarise` prints one line per turn:

```text
84acc7bd… turn 1 ttfa 1ms: retrieve 1x 0ms | segment.emit 2x 0ms (first 0, max 0) |
  tts 2x 0ms (first 0, max 0) | llm 1x 1ms | playback 2x 0ms (first 0, max 0) | turn 1x 1ms
```

Those milliseconds are stub backends — a template LLM and a silent TTS. They
measure the plumbing, not a model. But the shape is the point: repeated stages
accumulate into count, total, **first** and **max** rather than overwriting. For
time-to-first-audio, `tts first` is the number that matters and `tts max` is a
different question entirely. A summary that averaged them would hide both.

## The floor nobody can get under

Nothing downstream can start before the LLM emits its first token, so
`llm.ttft_ms` is a hard floor on time to first audio. There are exactly three
levers on it:

- **Prompt length.** Retrieval context and conversation history are both paid for
  in prompt processing, on every single turn. The platform bounds them
  deliberately: `max_context_chars = 1200`, `max_history_turns = 12`.
- **Model size.** Smaller is faster and less correct, and the two must be
  reported together — the predecessor measured a tier swap at 192 ms against
  4,427 ms median with 2 of 6 factual answers wrong against 0 of 6 (alpha-core,
  cycle 5, 2026-09-12, `notebook_optimized.ipynb`).
- **Serving engine.** Continuous batching and a paged KV cache are what keep
  `ttft_ms` flat as concurrent sessions rise.

And one that is not the model at all: the VAD's end-of-speech silence window is
charged in full to the reply latency. On the laptop agent that was 1,500 ms —
more than half the streaming time-to-first-audio budget, spent waiting to be sure
the user had stopped talking. It is the first thing to cut when the agent feels
slow, and the most likely thing to be overlooked because it happens before any
model runs.

## The SLO

`ttfa_p95_ms` at or below 3,000 ms over 30 days. That is a target to design
against, not a measurement — nothing in this repository has measured it. What the
repository does provide is the path from a trace row to a gate decision:
`vmp obs slo` computes the percentile and the error-budget burn from the same
rows production writes, and `vmp eval gate` consumes the number.

The burn rate is the part worth watching. A metric comfortably inside its
objective while consuming budget at 0.54 is passing today and in trouble next
month. A threshold alert sees that weeks later than a burn-rate alert does.
