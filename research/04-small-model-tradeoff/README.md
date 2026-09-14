# 04 — Small model trade-off

## Claim

Swapping the language model for a smaller one cuts turn latency substantially and
costs answer correctness, and the two effects are large enough, and coupled
tightly enough, that neither may be reported without the other. A latency figure
for a model tier that does not carry its correctness figure from the same run and
the same question set is not a result; it is a selection.

The operational form of the claim: the smaller tier is a legitimate degradation
step for a voice agent under load, and it is only legitimate while the trade is
visible at the moment it is made.

## Falsifier

- A smaller model is found that is faster **and** not measurably worse on the
  same question set, beyond the sampling error of that set. The trade would then
  not exist at that size, and the claim as stated would be false.
- Correctness does not move at all between tiers on a question set large enough
  to detect the difference — meaning the coupling is an artefact of a six-item
  set rather than a property of the tiers.
- Latency and correctness turn out to be separable in practice: prompt changes,
  a retrieval step, or a verification pass recover the large model's correctness
  at the small model's latency. The honest reading would then be that the trade
  is against effort, not against size.

## Method

**The harness, not the models.** `spike_correctness_harness.py` scores a JSONL
where each row is one answered question: `{"id", "model", "question",
"expected", "answer", "llm_ms"}`. Correctness comes from
`vmp.eval.wer.exact_match` with `numbers=True`, so "sixty" and "60" are the same
answer, with a normalised token-containment fallback because the expected answer
is a phrase inside a spoken sentence. Latency percentiles come from
`vmp.eval.latency.percentile`. The output is one table with one row per model
tier, in which `p50`, `p95` and accuracy are columns of the same row. The table
cannot be quoted half-way.

**The data is synthetic and the script writes it itself.** Questions come from
`vmp.data.synthetic.generate_golden_set` (`factual_short`, seed 3); the large
tier is generated always correct, the small tier wrong on every third question;
latencies are drawn from two fixed distributions. Running it prints 18 rows per
tier drawn from 6 distinct questions, a 20.1x p50 gap and a 33.3-point accuracy
gap. Those numbers are the generator's parameters coming back out. They
demonstrate that the harness runs; they measure nothing.

**Why `factual_short`.** The trade-off is only visible on questions with a
checkable answer. Style, tone and refusal behaviour do not degrade the same way,
and a question set mixing them dilutes the correctness signal until the latency
figure looks free.

## Outcome

`not run`

No model of any size has been run in this repository, and no latency or
correctness has been measured here. Only the harness has been executed, over a
file it generated.

The claim comes from the predecessor project, which compared two tiers of the
same model family on a small factual set: **gemma3:270m vs gemma3 4B on 6
factual questions — llm median 192 ms vs 4,427 ms; 2/6 wrong vs 0/6**
(alpha-core, cycle 5, 2026-09-12, `notebook_optimized.ipynb`).

Six questions is a small set. The latency gap is large enough that its direction
is not in doubt; the correctness figure, 2 wrong against 0 wrong, is a difference
of two items and its precision should not be overstated. That caveat travels with
the number wherever it is quoted, and it is the reason the first falsifier clause
above is written in terms of sampling error.

## What moved into src/

- `vmp.eval.wer` — `exact_match`, `normalise` (including the number-word
  mapping) and `corpus_wer`, the correctness side of the table.
- `vmp.eval.latency` and `vmp.observability.slo.percentile` — `p50`/`p95` from
  trace rows, so the latency side comes from the same traces production emits
  rather than from a stopwatch in a notebook.
- `vmp.eval.gates` — the release gate, which is where the coupling is enforced:
  a `GateDecision` evaluates latency thresholds and correctness thresholds
  together, and a run that improves one while breaching the other does not pass.
- `vmp.observability` stage tracing — `llm` payloads carry `ttft_ms` and
  `playback` carries `response_ms`, so the latency of a tier switch is
  attributable to a stage rather than to the turn as a whole.
- `configs/gates.toml` and `configs/slo.toml` — the thresholds as configuration.
