from __future__ import annotations

import json

import pytest

from vmp.cli import main
from vmp.data import (
    CATEGORIES,
    PreferencePairBuilder,
    TurnCorpus,
    assign_split,
    atomic_write_text,
    generate_golden_set,
    pairs_from_scores,
    pairs_from_style,
    read_jsonl,
    read_utterances,
    scrub,
    scrub_utterance,
    sha256_file,
    sha256_record,
    sha256_records,
    split_by_key,
    split_utterances,
    spoken_style_violations,
    write_jsonl,
    write_utterances,
)
from vmp.data.pii import scrub_records
from vmp.types import PreferencePair, Session, Turn, Utterance

# --- io -------------------------------------------------------------------------


def test_jsonl_roundtrip_and_blank_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    n = write_jsonl(p, [{"a": 1}, {"b": "ü"}])
    assert n == 2
    p.write_text(p.read_text() + "\n\n")
    assert list(read_jsonl(p)) == [{"a": 1}, {"b": "ü"}]


def test_jsonl_append_and_invalid_line(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    write_jsonl(p, [{"a": 2}], append=True)
    assert [r["a"] for r in read_jsonl(p)] == [1, 2]
    p.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match=":2: invalid JSON"):
        list(read_jsonl(p))


def test_read_jsonl_missing_file_names_path(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"missing\.jsonl"):
        list(read_jsonl(tmp_path / "missing.jsonl"))


def test_atomic_write_leaves_no_temp_files(tmp_path):
    p = tmp_path / "sub" / "f.txt"
    atomic_write_text(p, "one")
    atomic_write_text(p, "two")
    assert p.read_text() == "two"
    assert [q.name for q in (tmp_path / "sub").iterdir()] == ["f.txt"]


def test_sha256_helpers(tmp_path):
    assert sha256_record({"a": 1, "b": 2}) == sha256_record({"b": 2, "a": 1})
    assert sha256_record({"a": 1}) != sha256_record({"a": 2})
    assert sha256_records([{"a": 1}, {"a": 2}]) != sha256_records([{"a": 2}, {"a": 1}])
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_read_utterances_assigns_ids(tmp_path):
    p = tmp_path / "turns.jsonl"
    write_jsonl(p, [{"text": "hi"}, {"id": "u2", "text": "there", "speaker": "op"}])
    us = read_utterances(p)
    assert [u.id for u in us] == ["turns-000000", "u2"]
    assert us[1].speaker == "op"
    write_jsonl(p, [{"nope": 1}])
    with pytest.raises(ValueError, match="no 'text'"):
        read_utterances(p)


# --- corpus ---------------------------------------------------------------------


def test_corpus_from_sessions_and_jsonl(tmp_path):
    u = Utterance(id="u1", text="hello")
    s = Session(id="s1", created_at=0.0, turns=[Turn("s1", 1, u, assistant_text="hi")])
    c = TurnCorpus.from_sessions([s])
    assert len(c) == 1
    assert c.utterances[0].meta == {"session": "s1", "turn": 1, "assistant_text": "hi"}
    p = tmp_path / "c.jsonl"
    c.to_jsonl(p)
    c2 = TurnCorpus.from_jsonl(p)
    assert c2.utterances == c.utterances
    assert c2.data_hash() == c.data_hash()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sure, it is twelve degrees and cloudy.", []),
        ("Here you go:\n- one\n- two", ["bullets"]),
        ("**Bold** answer", ["markdown"]),
        ("See https://example.com for details", ["url"]),
        ("First.\n\nSecond paragraph.", ["paragraphs"]),
    ],
)
def test_spoken_style_rules(text, expected):
    assert spoken_style_violations(text) == expected


def test_spoken_style_long_reply_rejected():
    long = " ".join(["word"] * 70) + "."
    assert "too_long" in spoken_style_violations(long)
    many = "One. Two. Three. Four. Five. Six."
    assert "too_many_sentences" in spoken_style_violations(many)


def test_pairs_from_scores_best_vs_worst_and_margin():
    recs = [
        {
            "prompt": "p",
            "candidates": [
                {"text": "mid", "score": 0.5},
                {"text": "best", "score": 0.9},
                {"text": "worst", "score": 0.1},
            ],
        },
        {"prompt": "tie", "candidates": [{"text": "a", "score": 1}, {"text": "b", "score": 1}]},
        {"prompt": "single", "candidates": [{"text": "a", "score": 1}]},
    ]
    pairs = pairs_from_scores(recs)
    assert len(pairs) == 1
    assert (pairs[0].chosen, pairs[0].rejected) == ("best", "worst")
    assert pairs[0].meta["margin"] == pytest.approx(0.8)
    assert len(pairs_from_scores(recs, all_pairs=True)) == 3
    assert len(pairs_from_scores(recs, all_pairs=True, min_margin=0.5)) == 1


def test_pairs_from_style():
    recs = [
        {
            "prompt": "what time is it",
            "replies": ["It is half past three.", "## Time\n- 15:30\n- local", "Just after three."],
        },
        {"prompt": "all bad", "replies": ["- a", "- b"]},
    ]
    pairs = pairs_from_style(recs)
    assert len(pairs) == 2
    for p in pairs:
        assert not spoken_style_violations(p.chosen)
        assert p.rejected.startswith("## Time")
        assert "bullets" in p.meta["rejected_reasons"]
        assert p.source == "spoken_style"


def test_builder_mixes_sources_and_roundtrips(tmp_path):
    p = tmp_path / "in.jsonl"
    write_jsonl(
        p,
        [
            {"prompt": "a", "candidates": [{"text": "x", "score": 2}, {"text": "y", "score": 1}]},
            {"prompt": "b", "replies": ["Short and spoken.", "**Markdown**"]},
        ],
    )
    pairs = PreferencePairBuilder().build_from_jsonl(p)
    assert sorted(x.source for x in pairs) == ["scores", "spoken_style"]
    back = PreferencePair.from_dict(json.loads(json.dumps(pairs[0].to_dict())))
    assert back == pairs[0]


def test_to_hf_dataset_lazy():
    pytest.importorskip("datasets")
    ds = TurnCorpus([Utterance(id="a", text="b")]).to_hf_dataset()
    assert len(ds) == 1


# --- synthetic ------------------------------------------------------------------


def test_golden_set_is_deterministic_and_categorised():
    a = generate_golden_set(per_category=5, seed=3)
    b = generate_golden_set(per_category=5, seed=3)
    assert a == b
    assert len(a) == 5 * len(CATEGORIES)
    assert {u.meta["category"] for u in a} == set(CATEGORIES)
    assert len({u.id for u in a}) == len(a)
    assert all(u.text.strip() for u in a)
    assert generate_golden_set(per_category=5, seed=4) != a


def test_golden_set_category_isolation_and_meta():
    full = generate_golden_set(per_category=4, seed=1)
    only = generate_golden_set(per_category=4, seed=1, categories=["refusal", "tool_intent"])
    subset = [u for u in full if u.meta["category"] in ("refusal", "tool_intent")]
    assert sorted(subset, key=lambda u: u.id) == sorted(only, key=lambda u: u.id)
    assert all(u.meta["expected"] == "refuse" for u in only if u.meta["category"] == "refusal")
    assert all("tool" in u.meta for u in only if u.meta["category"] == "tool_intent")
    sv = [u for u in full if u.meta["category"] == "speaker_variance"]
    assert len({u.speaker for u in sv}) > 1
    with pytest.raises(ValueError, match="unknown category"):
        generate_golden_set(categories=["nope"])


# --- splits ---------------------------------------------------------------------


def test_assign_split_is_deterministic_and_salted():
    keys = [f"sess-{i}" for i in range(2000)]
    first = [assign_split(k) for k in keys]
    assert first == [assign_split(k) for k in keys]
    assert first != [assign_split(k, salt="v2") for k in keys]
    counts = {s: first.count(s) for s in ("train", "val", "test")}
    assert 1500 < counts["train"] < 1700
    assert 150 < counts["val"] < 250
    assert 150 < counts["test"] < 250
    with pytest.raises(ValueError):
        assign_split("k", ratios=(0.5, 0.5, 0.5))


def test_session_never_leaks_across_splits():
    utts = [
        Utterance(id=f"u{i}", text="t", meta={"session": f"s{i % 40}"}) for i in range(400)
    ]
    parts = split_utterances(utts)
    seen: dict[str, str] = {}
    for name, rows in parts.items():
        for u in rows:
            sess = u.meta["session"]
            assert seen.setdefault(sess, name) == name
    assert sum(len(v) for v in parts.values()) == 400
    assert len(parts["train"]) > len(parts["test"])


def test_split_by_key_generic():
    parts = split_by_key(range(100), key=str, ratios=(1.0, 0.0, 0.0))
    assert len(parts["train"]) == 100 and not parts["val"] and not parts["test"]


# --- pii ------------------------------------------------------------------------


def test_scrub_email_phone_card():
    r = scrub("mail a.b+c@ex.io, ring +1 415-555-0199 or (020) 7946 0958, card 4111 1111 1111 1111")
    assert r.text == "mail [EMAIL], ring [PHONE] or [PHONE], card [CARD]"
    assert r.counts == {"email": 1, "phone": 2, "card": 1}
    assert r.total == 4


def test_scrub_leaves_short_numbers_alone():
    r = scrub("order 4721 was placed in 2024 at 10.30")
    assert r.counts == {}
    assert r.text == "order 4721 was placed in 2024 at 10.30"


def test_scrub_utterance_and_records():
    u = Utterance(id="u", text="call 415-555-0199", meta={"k": 1})
    s = scrub_utterance(u)
    assert s.text == "call [PHONE]"
    assert s.meta == {"k": 1, "pii": {"phone": 1}}
    assert scrub_utterance(Utterance(id="c", text="clean")) is not None
    recs, totals = scrub_records([{"text": "x@y.io", "other": "x@y.io"}], fields=("text",))
    assert recs == [{"text": "[EMAIL]", "other": "x@y.io"}]
    assert totals == {"email": 1}


# --- cli ------------------------------------------------------------------------


def test_cli_synth_and_pairs(tmp_path, capsys):
    out = tmp_path / "golden.jsonl"
    assert main(["data", "synth", "--out", str(out), "--per-category", "2", "--seed", "9"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["n"] == 2 * len(CATEGORIES)
    assert len(read_utterances(out)) == m["n"]

    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, [{"prompt": "b", "replies": ["Short and spoken.", "- bullet"]}])
    pairs_out = tmp_path / "pairs.jsonl"
    assert main(["data", "pairs", "--in", str(inp), "--out", str(pairs_out), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 1
    assert not pairs_out.exists()
    assert main(["data", "pairs", "--in", str(inp), "--out", str(pairs_out)]) == 0
    assert len(list(read_jsonl(pairs_out))) == 1


def test_write_utterances_roundtrip(tmp_path):
    us = generate_golden_set(per_category=1, seed=0)
    p = tmp_path / "g.jsonl"
    assert write_utterances(p, us) == len(us)
    assert read_utterances(p) == us
