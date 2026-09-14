"""Preference evaluation: win-rate against a reference and DPO implicit rewards.

`win_rate` compares a policy's answers with the `chosen` answer of each
`PreferencePair` under a judge. `dpo_margins` computes the implicit reward
margin `beta * ((pi_c - ref_c) - (pi_r - ref_r))` from sequence log-probs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from vmp.types import EvalResult, PreferencePair


def win_rate(
    pairs: Sequence[PreferencePair],
    policy_answers: Sequence[str],
    judge: Any,
    *,
    reference: str = "chosen",
    name: str = "preference",
) -> EvalResult:
    """Fraction of prompts where the policy answer scores higher than the reference.

    Ties count as half a win (`win_rate = (wins + 0.5 * ties) / n`). `reference`
    picks `chosen` (default) or `rejected` from each pair.
    """
    if len(pairs) != len(policy_answers):
        raise ValueError("pairs and policy_answers differ in length")
    wins = ties = losses = 0
    rows: list[dict[str, Any]] = []
    for pair, answer in zip(pairs, policy_answers, strict=True):
        ref_answer = pair.chosen if reference == "chosen" else pair.rejected
        s_policy = float(judge.score(pair.prompt, answer, pair.chosen))
        s_ref = float(judge.score(pair.prompt, ref_answer, pair.chosen))
        if s_policy > s_ref:
            wins += 1
            outcome = "win"
        elif s_policy < s_ref:
            losses += 1
            outcome = "loss"
        else:
            ties += 1
            outcome = "tie"
        rows.append(
            {"prompt": pair.prompt, "policy": s_policy, "reference": s_ref, "outcome": outcome}
        )
    n = len(pairs)
    rate = (wins + 0.5 * ties) / n if n else 0.0
    return EvalResult(
        name=name,
        metrics={
            "win_rate": rate,
            "wins": float(wins),
            "ties": float(ties),
            "losses": float(losses),
        },
        n=n,
        details={"reference": reference, "judge": type(judge).__name__, "items": rows},
    )


def dpo_margins(
    policy_chosen: Sequence[float],
    policy_rejected: Sequence[float],
    ref_chosen: Sequence[float],
    ref_rejected: Sequence[float],
    *,
    beta: float = 0.1,
) -> dict[str, Any]:
    """Implicit reward margins from sequence log-probs (one value per example).

    Returns `margins`, `mean_margin`, `accuracy` (fraction with margin > 0) and
    `mean_loss` (`-log sigmoid(margin)`, the DPO objective).
    """
    n = len(policy_chosen)
    if not (n == len(policy_rejected) == len(ref_chosen) == len(ref_rejected)):
        raise ValueError("all log-prob lists must have the same length")
    margins: list[float] = []
    losses: list[float] = []
    for pc, pr, rc, rr in zip(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, strict=True
    ):
        m = beta * ((pc - rc) - (pr - rr))
        margins.append(m)
        losses.append(math.log1p(math.exp(-m)) if m > -700 else -m)
    return {
        "margins": margins,
        "mean_margin": sum(margins) / n if n else 0.0,
        "accuracy": sum(1 for m in margins if m > 0) / n if n else 0.0,
        "mean_loss": sum(losses) / n if n else 0.0,
        "beta": beta,
        "n": n,
    }


__all__ = ["dpo_margins", "win_rate"]
