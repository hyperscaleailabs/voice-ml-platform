"""Evaluation tests: WER/CER, golden set, judges, preference, latency, release gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vmp.eval.gates import (
    Gate,
    collect_metrics,
    evaluate_gates,
    load_rules,
    no_regression,
)
from vmp.eval.golden import (
    GoldenSet,
    IdentitySTT,
    NoisySTT,
    StubTTS,
    score_asr,
    utterances_from_texts,
)
from vmp.eval.judge import (
    ExactMatchJudge,
    LLMJudge,
    RubricJudge,
    count_sentences,
)
from vmp.eval.latency import stage_percentiles, ttfa_percentiles
from vmp.eval.preference import dpo_margins, win_rate
from vmp.eval.wer import (
    align_words,
    cer,
    corpus_wer,
    exact_match,
    normalise,
    wer,
)
from vmp.types import EvalResult, GateDecision, PreferencePair, Utterance

ROOT = Path(__file__).resolve().parents[1]

TEXTS = [
    ("basic", "hello there how are you"),
    ("basic", "what is the time please"),
    ("names", "call Alice Bergstrom now"),
    ("numbers", "set a timer for twenty one minutes"),
]


def _golden() -> GoldenSet:
    return GoldenSet(utterances_from_texts(TEXTS), name="mini")


# ===========================================================================
# WER / CER
# ===========================================================================


def test_wer_of_identical_text_is_zero():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0
    assert cer("the quick brown fox", "the quick brown fox") == 0.0


@pytest.mark.parametrize(
    ("ref", "hyp", "expected"),
    [
        ("the quick brown fox", "the quick fox", 1 / 4),  # one deletion
        ("the quick brown fox", "the quick brown red fox", 1 / 4),  # one insertion
        ("the quick brown fox", "the slow brown fox", 1 / 4),  # one substitution
        ("a b c", "a x c d", 2 / 3),  # one substitution + one insertion
        ("one two three", "", 1.0),  # everything deleted
        ("", "", 0.0),  # both empty
    ],
)
def test_wer_hand_computed_cases(ref, hyp, expected):
    assert wer(ref, hyp) == pytest.approx(expected)


def test_wer_with_an_empty_reference_counts_hypothesis_words():
    assert wer("", "two extra words here") == 4.0


def test_alignment_counts_each_operation():
    alignment = align_words("the cat sat on the mat", "the dog sat the mat now")
    assert alignment.to_dict() == {
        "hits": 4,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
        "n_ref": 6,
        "n_hyp": 6,
        "errors": 3,
        "rate": pytest.approx(0.5),
    }
    kinds = [op[0] for op in alignment.ops]
    assert kinds.count("substitute") == 1
    assert kinds.count("delete") == 1
    assert kinds.count("insert") == 1
    assert ("substitute", "cat", "dog") in alignment.ops


def test_cer_counts_characters_not_words():
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    # "hello" -> "helo": one deleted character out of ten reference characters.
    assert cer("hello world", "helo world") == pytest.approx(1 / 10)


# -- normalisation ----------------------------------------------------------


def test_normalisation_lowercases_strips_punctuation_and_splits_hyphens():
    assert normalise("Hello, World-Wide! It's 3.") == ["hello", "world", "wide", "its", "3"]
    assert normalise("   spaced \n out  ") == ["spaced", "out"]
    assert normalise("") == []


def test_normalisation_makes_punctuation_and_case_free():
    assert wer("Hello, world!", "hello world") == 0.0
    assert wer("It is twenty-one", "it is twenty one") == 0.0


def test_number_words_map_to_digits_when_requested():
    assert normalise("twenty one dogs", numbers=True) == ["21", "dogs"]
    assert normalise("one hundred and five", numbers=True) == ["105"]
    assert normalise("two thousand", numbers=True) == ["2000"]
    assert wer("set a timer for 21 minutes", "set a timer for twenty one minutes") == pytest.approx(
        2 / 6
    )
    assert (
        wer(
            "set a timer for 21 minutes",
            "set a timer for twenty one minutes",
            numbers=True,
        )
        == 0.0
    )


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("Hello, world!", "hello world") is True
    assert exact_match("hello world", "hello  there") is False
    assert exact_match("21 dogs", "twenty one dogs", numbers=True) is True


# -- corpus -----------------------------------------------------------------


def test_corpus_wer_pools_errors_rather_than_averaging_rates():
    refs = ["a b c d", "e"]
    hyps = ["a b c d", "x"]  # 0/4 errors and 1/1 error
    metrics = corpus_wer(refs, hyps)
    assert metrics["wer"] == pytest.approx(1 / 5)  # not the mean of 0.0 and 1.0
    assert metrics["substitutions"] == 1.0
    assert metrics["hits"] == 4.0
    assert metrics["n_ref_words"] == 5.0
    assert metrics["n"] == 2.0
    assert metrics["exact_match"] == 0.5


def test_corpus_wer_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        corpus_wer(["a"], ["a", "b"])


# ===========================================================================
# golden set
# ===========================================================================


def test_golden_set_categories_and_round_trip(tmp_path: Path):
    golden = _golden()
    assert len(golden) == 4
    assert golden.categories() == {"basic": 2, "names": 1, "numbers": 1}

    path = golden.save(tmp_path / "nested" / "golden.jsonl")
    reloaded = GoldenSet.load(path)
    assert reloaded.name == "golden"
    assert [u.text for u in reloaded] == [u.text for u in golden]
    assert all(isinstance(u, Utterance) for u in reloaded)

    with pytest.raises(FileNotFoundError):
        GoldenSet.load(tmp_path / "missing.jsonl")


def test_build_audio_dry_run_writes_nothing(tmp_path: Path):
    out = tmp_path / "clips"
    manifest = _golden().build_audio(StubTTS(), out, dry_run=True)

    assert manifest["dry_run"] is True
    assert manifest["name"] == "mini"
    assert len(manifest["items"]) == 4
    assert manifest["items"][0]["audio_path"].endswith("g0000.wav")
    assert manifest["items"][2]["category"] == "names"
    assert not out.exists()


def test_score_asr_dry_run_plans_without_transcribing(tmp_path: Path):
    manifest = _golden().build_audio(StubTTS(), tmp_path / "clips", dry_run=True)
    result = score_asr(IdentitySTT(), manifest)

    assert isinstance(result, EvalResult)
    assert result.name == "mini_asr"
    assert result.n == 4
    assert result.metrics == {}
    assert result.details["dry_run"] is True
    assert result.details["plan"] == {
        "stt": "IdentitySTT",
        "categories": ["basic", "names", "numbers"],
    }


def test_identity_stt_scores_a_perfect_run(tmp_path: Path):
    out = tmp_path / "clips"
    manifest = _golden().build_audio(StubTTS(), out)
    assert sorted(p.name for p in out.iterdir()) == [f"g{i:04d}.wav" for i in range(4)]
    assert (out / "g0000.wav").read_bytes() == b"hello there how are you"

    result = score_asr(IdentitySTT(), manifest)
    assert result.metrics == {"wer": 0.0, "cer": 0.0, "exact_match": 1.0}
    assert result.details["dry_run"] is False
    assert result.details["stt"] == "IdentitySTT"
    assert sorted(result.details["per_category"]) == ["basic", "names", "numbers"]
    assert len(result.details["items"]) == 4


def test_noisy_stt_is_deterministic_and_drops_words(tmp_path: Path):
    stt = NoisySTT(drop_rate=0.34, seed=7)
    assert stt.transcribe(b"hello there friend") == stt.transcribe(b"hello there friend")
    assert len(stt.transcribe(b"hello there friend").split()) < 3
    # A different seed gives a different noise pattern for the same text.
    assert NoisySTT(drop_rate=0.34, seed=99).transcribe(b"a b c d e f g h") != stt.transcribe(
        b"a b c d e f g h"
    )
    # No noise at all is the identity.
    assert NoisySTT(drop_rate=0.0).transcribe(b"hello there friend") == "hello there friend"


def test_noisy_stt_substitutions_keep_the_word_count(tmp_path: Path):
    noisy = NoisySTT(drop_rate=0.0, sub_rate=1.0, seed=1)
    assert noisy.transcribe(b"hello there") == "olleh ereht"


def test_score_asr_with_the_noisy_stt_is_reproducible(tmp_path: Path):
    manifest = _golden().build_audio(StubTTS(), tmp_path / "clips")
    first = score_asr(NoisySTT(drop_rate=0.2, seed=3), manifest)
    second = score_asr(NoisySTT(drop_rate=0.2, seed=3), manifest)

    assert first.metrics == second.metrics
    assert 0.0 < first.metrics["wer"] < 1.0
    assert first.metrics["exact_match"] < 1.0
    # Per-category metrics have the same shape as the overall ones.
    for metrics in first.details["per_category"].values():
        assert {"wer", "cer", "exact_match", "n"} <= set(metrics)
    assert sum(m["n"] for m in first.details["per_category"].values()) == 4
    # Dropping words means deletions, never insertions.
    assert sum(item["insertions"] for item in first.details["items"]) == 0
    assert sum(item["deletions"] for item in first.details["items"]) > 0


# ===========================================================================
# judges
# ===========================================================================


def test_exact_match_judge():
    judge = ExactMatchJudge()
    assert judge.score("q", "Hello, world!", "hello world") == 1.0
    assert judge.score("q", "something else", "hello world") == 0.0
    assert judge.score("q", "no reference given") == 0.0


def test_count_sentences():
    assert count_sentences("") == 0
    assert count_sentences("One sentence.") == 1
    assert count_sentences("One. Two! Three?") == 3
    assert count_sentences("no terminal punctuation") == 1


def test_rubric_judge_rewards_spoken_style():
    judge = RubricJudge()
    assert judge.score("q", "The timer is set for five minutes.") == 1.0
    assert judge.check("The timer is set.") == {
        "nonempty": True,
        "max_sentences": True,
        "no_markdown": True,
        "no_urls": True,
    }


def test_rubric_judge_penalises_markdown_urls_and_length():
    judge = RubricJudge(max_sentences=2, max_words=10, forbidden_phrases=["as an ai"])
    checks = judge.check("- A bullet. Visit https://example.com now. As an AI, I cannot. Four.")
    assert checks["no_markdown"] is False
    assert checks["no_urls"] is False
    assert checks["max_sentences"] is False
    assert checks["max_words"] is False
    assert checks["no_phrase:as an ai"] is False
    # Four of the six rules still pass: it is non-empty, short, and phrase-clean.
    assert judge.score("q", "- A bullet. Visit https://example.com now.") == pytest.approx(4 / 6)
    assert judge.check("")["nonempty"] is False


class StubJudgeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return self.reply


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("0.8", 0.8),
        ("Score: 1", 1.0),
        ("0", 0.0),
        ("7/10", 0.7),  # a 0-10 scale is rescaled
        ("85", 0.85),  # so is a 0-100 scale
        ("no number at all", 0.0),
        ("-3", 0.0),  # clamped
    ],
)
def test_llm_judge_parses_and_clamps_the_score(reply, expected):
    assert LLMJudge.parse_score(reply) == pytest.approx(expected)


def test_llm_judge_prompt_and_scoring():
    llm = StubJudgeLLM("0.75")
    judge = LLMJudge(llm)
    assert judge.score("What time is it?", "It is five.", "It is five o'clock.") == 0.75

    messages = llm.messages[0]
    assert messages[0]["role"] == "system"
    assert "one number between 0 and 1" in messages[0]["content"]
    assert "What time is it?" in messages[1]["content"]
    assert "Reference answer: It is five o'clock." in messages[1]["content"]


# ===========================================================================
# preference
# ===========================================================================


class KeywordJudge:
    """Scores 1.0 when the answer mentions the keyword, else 0.0."""

    def __init__(self, keyword: str = "timer") -> None:
        self.keyword = keyword

    def score(self, prompt: str, answer: str, reference: str | None = None) -> float:
        return 1.0 if self.keyword in answer.lower() else 0.0


def test_win_rate_counts_wins_ties_and_losses():
    pairs = [
        PreferencePair("set a timer", "I set a reminder.", "No."),
        PreferencePair("set a timer", "I set a timer.", "No."),
        PreferencePair("set a timer", "I set a timer.", "No."),
    ]
    answers = ["Timer set.", "Nothing useful.", "Your timer is running."]
    result = win_rate(pairs, answers, KeywordJudge())

    assert result.metrics == {"win_rate": 0.5, "wins": 1.0, "ties": 1.0, "losses": 1.0}
    assert result.n == 3
    assert [row["outcome"] for row in result.details["items"]] == ["win", "loss", "tie"]
    assert result.details["judge"] == "KeywordJudge"
    assert result.details["reference"] == "chosen"


def test_win_rate_against_the_rejected_answer():
    pairs = [PreferencePair("set a timer", "I set a timer.", "No idea.")]
    result = win_rate(pairs, ["Timer set."], KeywordJudge(), reference="rejected")
    assert result.metrics["win_rate"] == 1.0
    assert result.details["reference"] == "rejected"


def test_win_rate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        win_rate([PreferencePair("p", "c", "r")], ["a", "b"], ExactMatchJudge())


def test_dpo_margin_math():
    # beta * ((pi_c - ref_c) - (pi_r - ref_r))
    out = dpo_margins([-1.0, -3.0], [-2.0, -1.0], [-1.5, -2.0], [-1.5, -2.0], beta=0.5)
    # 0.5 * ((-1.0 + 1.5) - (-2.0 + 1.5)) = 0.5; 0.5 * ((-3.0 + 2.0) - (-1.0 + 2.0)) = -1.0
    assert out["margins"] == pytest.approx([0.5, -1.0])
    assert out["mean_margin"] == pytest.approx(-0.25)
    assert out["accuracy"] == 0.5
    assert out["beta"] == 0.5
    assert out["n"] == 2


def test_dpo_loss_is_negative_log_sigmoid_of_the_margin():
    import math

    out = dpo_margins([0.0], [0.0], [0.0], [0.0])  # margin 0 -> -log(0.5)
    assert out["margins"] == [0.0]
    assert out["mean_loss"] == pytest.approx(math.log(2))

    strong = dpo_margins([10.0], [0.0], [0.0], [0.0], beta=1.0)
    assert strong["margins"] == [10.0]
    assert strong["mean_loss"] < 0.001
    assert strong["accuracy"] == 1.0


def test_dpo_margins_rejects_ragged_input():
    with pytest.raises(ValueError):
        dpo_margins([1.0], [1.0], [1.0], [1.0, 2.0])


# ===========================================================================
# latency
# ===========================================================================


def _trace_rows() -> list[dict]:
    rows = []
    for turn, (stt_ms, ttfa) in enumerate([(100.0, 900.0), (300.0, 1100.0), (200.0, 2000.0)], 1):
        rows += [
            {"event": "stt.end", "ms": stt_ms, "session": "s", "turn": turn, "payload": {}},
            {
                "event": "playback.end",
                "ms": 5.0,
                "session": "s",
                "turn": turn,
                "seq": 1,
                "payload": {"response_ms": ttfa},
            },
            {
                "event": "playback.end",
                "ms": 5.0,
                "session": "s",
                "turn": turn,
                "seq": 2,
                "payload": {},
            },
            {"event": "turn.end", "ms": ttfa + 500.0, "session": "s", "turn": turn, "payload": {}},
        ]
    return rows


def test_stage_percentiles():
    stages = stage_percentiles(_trace_rows())
    assert set(stages) == {"stt", "playback", "turn"}
    assert stages["stt"]["n"] == 3.0
    assert stages["stt"]["p50"] == 200.0
    assert stages["stt"]["mean"] == 200.0
    assert stages["stt"]["max"] == 300.0
    assert stages["stt"]["p95"] == pytest.approx(290.0)
    assert stages["playback"]["n"] == 6.0  # one row per sentence


def test_stage_percentiles_accepts_custom_percentiles():
    stages = stage_percentiles(_trace_rows(), ps=(50, 90, 99))
    assert set(stages["stt"]) == {"p50", "p90", "p99", "n", "mean", "max"}


def test_ttfa_percentiles_take_one_value_per_turn():
    ttfa = ttfa_percentiles(_trace_rows())
    assert ttfa["n"] == 3.0
    assert ttfa["p50"] == 1100.0
    assert ttfa["mean"] == pytest.approx((900 + 1100 + 2000) / 3)
    assert ttfa["p95"] == pytest.approx(1910.0)
    assert ttfa_percentiles([]) == {}
    assert ttfa_percentiles([{"event": "stt.end", "ms": 1, "session": "s", "turn": 1}]) == {}


# ===========================================================================
# gates
# ===========================================================================


def test_collect_metrics_exposes_bare_and_prefixed_names():
    results = [
        EvalResult(name="golden_asr", metrics={"wer": 0.03}, n=10),
        EvalResult(name="preference", metrics={"win_rate": 0.6}, n=5),
    ]
    metrics = collect_metrics(results)
    assert metrics == {
        "wer": 0.03,
        "golden_asr.wer": 0.03,
        "win_rate": 0.6,
        "preference.win_rate": 0.6,
    }
    assert collect_metrics({"wer": 1}) == {"wer": 1.0}


def test_gate_needs_a_bound():
    with pytest.raises(ValueError, match="needs min or max"):
        Gate(metric="wer")


def test_gates_pass_with_reasons():
    results = [EvalResult(name="golden_asr", metrics={"wer": 0.03, "exact_match": 0.95}, n=10)]
    decision = evaluate_gates(results, load_rules(ROOT / "configs" / "gates.toml"))

    assert isinstance(decision, GateDecision)
    assert decision.passed is True
    assert decision.reasons[0] == "PASS wer: 0.03 ok (<= 0.05)"
    assert decision.reasons[1] == "PASS exact_match: 0.95 ok (>= 0.9)"
    # Optional metrics that were never measured do not fail the release.
    assert "PASS ttfa_p95_ms: missing (optional, skipped)" in decision.reasons
    assert decision.results == results
    assert json.loads(json.dumps(decision.to_dict()))["passed"] is True


def test_gates_fail_and_say_why():
    decision = evaluate_gates(
        {"wer": 0.09, "exact_match": 0.5, "ttfa_p95_ms": 4200.0},
        load_rules(ROOT / "configs" / "gates.toml"),
    )
    assert decision.passed is False
    assert "FAIL wer: 0.09 > max 0.05" in decision.reasons
    assert "FAIL exact_match: 0.5 < min 0.9" in decision.reasons
    assert "FAIL ttfa_p95_ms: 4200 > max 3000" in decision.reasons
    assert decision.results == []


def test_a_required_missing_metric_fails():
    decision = evaluate_gates({}, [Gate(metric="wer", max=0.05)])
    assert decision.passed is False
    assert decision.reasons == ["FAIL wer: missing (required)"]


def test_no_gates_configured_is_reported():
    decision = evaluate_gates({"wer": 0.5}, [])
    assert decision.passed is True
    assert decision.reasons == ["no gates configured"]


def test_load_rules_from_a_path_and_from_text():
    from_file = load_rules(ROOT / "configs" / "gates.toml")
    assert [g.metric for g in from_file] == [
        "wer",
        "exact_match",
        "ttfa_p95_ms",
        "win_rate",
        "error_rate",
    ]
    assert from_file[0].max == 0.05
    assert from_file[2].required is False
    assert from_file[0].description == "golden-set word error rate"

    inline = load_rules('[[gate]]\nmetric = "wer"\nmax = 0.1\nrequired = false\n')
    assert inline == [Gate(metric="wer", max=0.1, required=False)]
    assert Gate.from_dict(inline[0].to_dict()) == inline[0]
    assert load_rules("") == []


def test_gates_accept_rule_dicts():
    decision = evaluate_gates({"wer": 0.01}, [{"metric": "wer", "max": 0.05}])
    assert decision.passed is True


# -- no regression ----------------------------------------------------------


def test_no_regression_allows_improvements():
    candidate = EvalResult(name="candidate", metrics={"wer": 0.03, "win_rate": 0.62}, n=10)
    baseline = EvalResult(name="production", metrics={"wer": 0.05, "win_rate": 0.55}, n=10)
    decision = no_regression(candidate, baseline)

    assert decision.passed is True
    assert decision.reasons[0].startswith("PASS wer: baseline 0.05 -> candidate 0.03")
    assert "delta -0.02" in decision.reasons[0]
    assert decision.results == [candidate, baseline]


def test_no_regression_fails_when_a_lower_is_better_metric_rises():
    decision = no_regression({"wer": 0.06}, {"wer": 0.05})
    assert decision.passed is False
    assert decision.reasons == [
        "FAIL wer: baseline 0.05 -> candidate 0.06 (delta +0.01)",
    ]


def test_no_regression_fails_when_a_higher_is_better_metric_drops():
    decision = no_regression({"exact_match": 0.80}, {"exact_match": 0.95})
    assert decision.passed is False
    assert decision.reasons[0].startswith("FAIL exact_match")


def test_no_regression_tolerance_absorbs_noise():
    assert no_regression({"wer": 0.051}, {"wer": 0.05}, tolerance=0.005).passed is True
    assert no_regression({"wer": 0.06}, {"wer": 0.05}, tolerance=0.005).passed is False


def test_no_regression_on_selected_metrics_only():
    decision = no_regression(
        {"wer": 0.06, "win_rate": 0.9}, {"wer": 0.05, "win_rate": 0.5}, metrics=["win_rate"]
    )
    assert decision.passed is True
    assert len(decision.reasons) == 1


def test_no_regression_skips_metrics_missing_on_one_side():
    decision = no_regression({"wer": 0.05}, {"wer": 0.05, "cer": 0.01}, metrics=["wer", "cer"])
    assert decision.passed is True
    assert "SKIP cer: not in both results" in decision.reasons


def test_no_regression_without_shared_metrics():
    decision = no_regression({"a": 1.0}, {"b": 1.0})
    assert decision.passed is True
    assert decision.reasons == ["no shared metrics to compare"]
