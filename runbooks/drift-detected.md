# Drift detected

## Symptom

A drift check reports PSI above 0.2 or a KS statistic above 0.1 between a
reference window and the current window — on an input feature, or on WER itself.

## Severity

SEV-3 by default: drift is a warning about the future, not an outage. SEV-2 when
drift is accompanied by a quality metric already moving (WER rising, win-rate
falling) — at that point it is no longer a leading indicator.

## Diagnose

### 1. Confirm the drift, and on what

```bash
vmp obs drift --reference /tmp/reference.txt --current /tmp/current.txt \
  --psi-threshold 0.2 --ks-threshold 0.1
```

Both series are one number per line. PSI is bucketed and sensitive to a shifted
mass; KS is distribution-free and sensitive to a shifted shape. They disagree
usefully: PSI alone often means a new mode appeared in one bucket, KS alone often
means a gradual shift across the range.

### 2. Rule out the boring causes first

In order of how often they are the answer:

- **A window artefact.** Is the current window shorter, or does it span a weekend,
  a holiday, a deploy? Re-run with a window of the same length and shape.
- **A pipeline change.** Did the feature definition change? A new feature, a
  changed dtype or a changed TTL in `src/vmp/features/views.py` makes the two
  windows incomparable by construction. `git log -- src/vmp/features/views.py`.
- **Missing data.** A source that stopped writing shows as drift because nulls
  collapse the distribution. Check the materialize CronJob before believing the
  world changed:

```bash
kubectl get cronjob -n platform feast-materialize
kubectl get jobs -n platform -l app.kubernetes.io/name=feast-materialize
kubectl logs -n platform -l app.kubernetes.io/name=feast-materialize --tail=100
```

- **Real population change.** New users, new devices, a new accent cohort, a new
  room acoustic. This is the case worth acting on.

### 3. Locate it — which feature, which cohort

Check the three views one at a time; each has a different meaning when it drifts.

| View | Feature | Drift usually means |
|---|---|---|
| `session_features` | `avg_response_ms` | a latency change, not a data change — go to `ttfa-slo-breach.md` |
| `session_features` | `avg_user_utterance_s` | users are speaking differently, or endpointing changed |
| `session_features` | `accent_profile`, `last_intent` | a genuinely new population or new use case |
| `speaker_features` | `wer_7d`, `exact_match_rate_7d` | model quality is moving — the serious case |
| `device_features` | `p95_ttfa_ms_24h` | a device cohort is degrading; check `edge_bundle_version` alongside |

Split by cohort before concluding fleet-wide drift. Drift concentrated in one
`edge_bundle_version` is a bundle problem with a rollback; drift spread evenly is
a population problem with a retraining answer.

### 4. Check whether quality actually moved

Drift on inputs is a prediction. Test it against the golden set:

```bash
vmp eval golden --set data/golden.jsonl
vmp eval gate --rules configs/gates.toml --metrics /tmp/metrics.json
```

The golden set is fixed, so it does not drift with the population — which is the
point: it separates "the inputs changed" from "the model got worse". Known weak
spot: the `asr_names` category measured 28.9% WER against 2.69% overall on 108
synthetic clips (alpha-core, 2026-09-05, `evals/golden/README.md`,
2026-09-05). Names drift first and drift hardest.

## Mitigate

Drift alone rarely needs a mitigation — it needs a decision.

- **Gates still pass, inputs drifted.** No action beyond recording it. Re-baseline
  the reference window so the alert stops repeating a known fact, and note why.
- **Gates fail.** Treat it as a quality regression: `runbooks/rollback-model.md`.
- **Drift is one cohort on one bundle.** Halt the rollout for that cohort and roll
  it back (`deploy/edge/ota/rollout.toml`, `runbooks/edge-bundle-verify-failed.md`
  for the verification side).
- **Drift is real and broad.** Queue a retraining cycle: collect the new
  population into a corpus, extend the golden set to cover it, then SFT or DPO.
  Extending the golden set comes first — retraining against an evaluation that
  does not contain the new population cannot show improvement.

```bash
vmp data synth --help       # build a golden text set covering the new cohort
vmp train sft --config configs/train_sft.toml --dry-run
```

## Verify

```bash
vmp obs drift --reference /tmp/reference.txt --current /tmp/current.txt
vmp eval gate --rules configs/gates.toml --metrics /tmp/metrics.json
```

After a re-baseline, the drift report should be below threshold against the new
reference and the gates unchanged.

## Follow-up

- Record the reference window used, with dates. A drift alert without a stated
  baseline is unfalsifiable.
- If the same feature drifts repeatedly and never predicts anything, its
  threshold is wrong or it should not be alerting. Tune it or drop it.
- If drift was found by a person rather than by the check, the check's window or
  cadence is the action item.
