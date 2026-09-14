---
name: Hypothesis
about: A falsifiable claim to test before it becomes code
title: "[hypothesis] "
labels: ["hypothesis", "research"]
---

<!-- This repository develops hypothesis-first: a claim, a way to prove it
     wrong, a method, an outcome. An issue that cannot be falsified is a task,
     not a hypothesis — open a normal issue instead.
     The accepted entry lands as research/<nn>-<topic>/README.md with these
     same sections. -->

## Claim

<!-- One sentence, specific and measurable. Name the metric, the magnitude and
     the conditions.

     Good: "Quantising the LLM to q4_k_m keeps golden-set exact match within
     2 points while reducing median llm time by at least 30%."
     Bad:  "Quantisation will make it faster." -->

## Falsifier

<!-- What result would make the claim false. Decide it now, before running
     anything. If no result could falsify the claim, it is not a hypothesis.

     e.g. "Exact match drops by more than 2 points, or median llm time
     improves by less than 30%, on the 108-clip golden set." -->

## Method

<!-- How it will be tested, precisely enough for someone else to repeat it:
     dataset and size, configs, commands, what is held fixed, how many runs,
     what counts as noise. -->

```
vmp ... --dry-run
```

- Data:
- Config:
- Commands:
- Controls / what is held fixed:

## Outcome

<!-- Filled in after running. Exactly one of:
       not run
       run — falsifier passes
       run — falsified
       run — inconclusive
     "Not run" is an acceptable and expected state. Say so rather than leaving
     it blank. Numbers need the command that produced them. -->

**Outcome:** not run

## What moved into src/

<!-- What, if anything, this justifies changing in the platform. A hypothesis
     that is falsified still has an answer here: usually "nothing, and here is
     the note that stops us trying it again". -->
