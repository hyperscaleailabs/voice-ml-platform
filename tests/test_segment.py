"""Sentence segmenter tests. The invariant: nothing emitted is ever truncated."""

from __future__ import annotations

import pytest

from vmp.serving.segment import ABBREVIATIONS, Segmenter, segment


def _tokens(text: str) -> list[str]:
    """Split into word-plus-space tokens, the shape `TemplateLLM.stream` produces."""
    words = text.split(" ")
    return [w if i == len(words) - 1 else w + " " for i, w in enumerate(words)]


def _by_char(text: str) -> list[str]:
    return list(text)


# -- the core invariant -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Hello there. How are you? Fine!",
        "Dr. Smith paid 3.5 dollars to Mr. Jones on Jan. 4th. Then he left.",
        'She said "Go home." He left, and never came back.',
        "Wait... Then it ended. Really?!",
        "The U.S.A. is large. It has fifty states.",
        "No terminal punctuation at all",
    ],
)
def test_never_emits_a_truncated_sentence(text):
    """However the stream is chopped, the emitted sentences rejoin to the input."""
    for tokens in (_tokens(text), _by_char(text), [text]):
        out = list(segment(tokens))
        assert " ".join(out) == text.strip()
        for sentence in out:
            assert sentence == sentence.strip()
            assert sentence


def test_word_stream_and_char_stream_agree():
    text = "One sentence here. A second one follows! And a third?"
    assert list(segment(_tokens(text))) == list(segment(_by_char(text)))


def test_sentences_are_split_on_every_terminal():
    assert list(segment(["One. Two! Three? Four"])) == ["One.", "Two!", "Three?", "Four"]


# -- abbreviations ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith went home.",
        "Mr. and Mrs. Jones arrived.",
        "See e.g. the appendix for details.",
        "That is i.e. the same thing.",
        "Acme Inc. filed the report.",
        "J. K. Rowling wrote it.",
        "The meeting is at 4 p.m. sharp.",
    ],
)
def test_abbreviation_is_not_a_boundary(text):
    assert list(segment(_tokens(text))) == [text]


def test_abbreviation_list_is_case_insensitive_and_dotless():
    seg = Segmenter()
    assert "dr" in seg.abbreviations
    assert "e.g" in seg.abbreviations
    assert all(not a.endswith(".") for a in ABBREVIATIONS)


def test_custom_abbreviation_set():
    seg = Segmenter(abbreviations={"Foo."})
    assert seg.feed("Foo. bar is here. ") == ["Foo. bar is here."]
    plain = Segmenter()
    assert plain.feed("Foo. bar is here. ") == ["Foo.", "bar is here."]


def test_dotted_acronym_mid_sentence_is_not_a_boundary():
    """`U.S.A.` followed by a lowercase word continues the sentence."""
    assert list(segment(["The U.S.A. is large. ", "It has states."])) == [
        "The U.S.A. is large.",
        "It has states.",
    ]
    # Followed by a new sentence, it is a boundary.
    assert list(segment(["He moved to the U.S.A. ", "Then he stayed."])) == [
        "He moved to the U.S.A.",
        "Then he stayed.",
    ]


def test_enumerator_is_not_a_boundary():
    assert list(segment(["1. apples 2. pears. ", "Done."])) == ["1. apples 2. pears.", "Done."]


# -- decimals ---------------------------------------------------------------


def test_decimal_split_across_tokens_is_not_a_boundary():
    """The classic `3.` -> `3.5` case: a boundary at the buffer end is never emitted."""
    seg = Segmenter()
    assert seg.feed("The value is 3") == []
    assert seg.feed(".") == []
    assert seg.pending == "The value is 3."
    assert seg.feed("5 units. ") == ["The value is 3.5 units."]


@pytest.mark.parametrize(
    "text", ["It costs 3.5 units.", "Version 1.2.3 shipped.", "Pi is 3.14159."]
)
def test_decimals_stay_in_one_sentence(text):
    assert list(segment(_tokens(text))) == [text]


# -- ellipsis ---------------------------------------------------------------


def test_ellipsis_is_a_boundary_only_before_a_new_sentence():
    assert list(segment(["Wait... ", "Then it ended."])) == ["Wait...", "Then it ended."]
    assert list(segment(["I said hi... ", "she left."])) == ["I said hi... she left."]
    assert list(segment(["Hold on… ", "Ready now."])) == ["Hold on…", "Ready now."]
    assert list(segment(["Maybe… ", "maybe not."])) == ["Maybe… maybe not."]


def test_ellipsis_before_a_digit_or_quote_is_a_boundary():
    assert list(segment(['Counting... ', '3 remain.'])) == ["Counting...", "3 remain."]
    assert list(segment(['Listen... ', '"Go away."'])) == ["Listen...", '"Go away."']


# -- quotes and brackets ----------------------------------------------------


def test_closing_quote_stays_with_the_sentence():
    assert list(segment(['She said "Go home." ', "He left."])) == [
        'She said "Go home."',
        "He left.",
    ]
    assert list(segment(["It ended (finally!) ", "We went home."])) == [
        "It ended (finally!)",
        "We went home.",
    ]
    assert list(segment(["He asked “why?” ", "Nobody knew."])) == [
        "He asked “why?”",
        "Nobody knew.",
    ]


def test_opening_quote_starts_a_new_sentence():
    assert list(segment(['He stopped. ', '"Enough," she said.'])) == [
        "He stopped.",
        '"Enough," she said.',
    ]


# -- flush ------------------------------------------------------------------


def test_flush_emits_the_tail_without_terminal_punctuation():
    seg = Segmenter()
    assert seg.feed("Complete one. Tail without end") == ["Complete one."]
    assert seg.pending == "Tail without end"
    assert seg.flush() == ["Tail without end"]
    assert seg.pending == ""
    assert seg.flush() == []


def test_flush_emits_a_sentence_that_ends_the_buffer():
    seg = Segmenter()
    assert seg.feed("Ends exactly here.") == []
    assert seg.flush() == ["Ends exactly here."]


def test_flush_splits_multiple_buffered_sentences():
    seg = Segmenter()
    assert seg.feed("A first one. A second one.") == ["A first one."]
    assert seg.flush() == ["A second one."]


def test_flush_on_whitespace_only_buffer_emits_nothing():
    seg = Segmenter()
    assert seg.feed("   \n  ") == []
    assert seg.flush() == []
    assert seg.emitted == 0


# -- the two APIs -----------------------------------------------------------


def test_functional_api_matches_stateful_api():
    tokens = _tokens("First sentence here. Second one now! A trailing fragment")
    seg = Segmenter()
    stateful: list[str] = []
    for tok in tokens:
        stateful.extend(seg.feed(tok))
    stateful.extend(seg.flush())
    assert list(segment(tokens)) == stateful
    assert seg.emitted == len(stateful) == 3


def test_functional_api_is_lazy():
    """`segment` yields a sentence before the generator is exhausted."""
    consumed: list[str] = []

    def tokens():
        for tok in _tokens("Ready now. And later on."):
            consumed.append(tok)
            yield tok

    gen = segment(tokens())
    assert next(gen) == "Ready now."
    assert len(consumed) < len(_tokens("Ready now. And later on."))


def test_feed_ignores_empty_tokens_and_counts_emissions():
    seg = Segmenter()
    assert seg.feed("") == []
    assert seg.feed("Hi there. ") == ["Hi there."]
    assert seg.feed("Bye now. ") == ["Bye now."]
    assert seg.emitted == 2
    assert seg.pending == ""
