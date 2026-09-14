# Whitepapers

Two long-form papers on the reasoning behind this platform. The
[guides](../guides/training-lora-sft.md) say how each subsystem works; these say
why the structure around them is shaped the way it is.

## [Research to production: an ML platform for voice agents](research-to-production-ml-platform-for-voice-agents.md)

The maturity path from a research spike to a production deployment, the contract
each stage owes the next, the artifacts each stage must produce, and the evidence
discipline that holds the whole thing together — *a number with no source file
does not go in a table*. Ends with a capability matrix across training, feature
store, RAG, serving, edge, evaluation and observability, showing what each
capability looks like at each of the three stages.

**Read it for:** how to organise an ML platform as a progression rather than as a
product, and what has to be true before an artifact is allowed to speak to a
user.

## [Hypothesis-driven development for ML platforms](hypothesis-driven-development-for-ml-platforms.md)

The nine-station loop — context, hypothesis, expansion, contract, factory,
evaluation, ledger, decision, production — with the two stations a human must
own, the rules that keep a spike disposable, and how falsifiers written before a
run prevent the most common failure in applied ML: a result that was never at
risk of being wrong. Includes how an autonomous build loop uses an independent
second model to review changes while being structurally prevented from merging
its own work.

**Read it for:** how to run experiments in an ML platform so that the outcomes
mean something, and how to delegate most of the loop to agents without
delegating the judgement.

## The rule both papers share

Measurements from the private predecessor project are cited as "alpha-core, cycle
5, 2026-09-12, `<file>`" and are never presented as this repository's own
benchmark. Anything this repository has not run says `not run`, and nothing
downstream cites it.
