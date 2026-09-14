from __future__ import annotations

import json

from vmp import __version__
from vmp.cli import main
from vmp.config import Settings, deep_merge, load_config
from vmp.types import Chunk, GateDecision, Session, Turn, Utterance


def test_roundtrip_session():
    u = Utterance(id="u1", text="hello", speaker="op")
    t = Turn(session_id="s", turn=1, user=u, assistant_text="hi", stages={"stt": 12.5})
    s = Session(id="s", created_at=1.0, system_prompt="be brief", turns=[t])
    back = Session.from_dict(json.loads(json.dumps(s.to_dict())))
    assert back == s


def test_chunk_embedding_roundtrip():
    c = Chunk(id="c", document_id="d", start_line=1, end_line=2, text="x", embedding=(0.1, 0.2))
    assert Chunk.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_gate_decision_serialises():
    g = GateDecision(passed=False, reasons=["wer too high"])
    assert json.loads(json.dumps(g.to_dict()))["passed"] is False


def test_settings_env_override(monkeypatch, tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[paths]\nregistry_root = "/from/file"\n')
    monkeypatch.setenv("VMP_TRACE_PATH", "/from/env.jsonl")
    s = Settings.from_sources(cfg)
    assert str(s.registry_root) == "/from/file"
    assert str(s.trace_path) == "/from/env.jsonl"
    assert load_config(cfg)["paths"]["registry_root"] == "/from/file"


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}


def test_cli_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_no_command_prints_help():
    assert main([]) == 2
