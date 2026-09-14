from __future__ import annotations

import json
from typing import ClassVar

import pytest

from vmp.cli import main
from vmp.registry.mlflow_adapter import from_mlflow_version, to_mlflow_tags
from vmp.registry.store import FileRegistry, RegistryError, transition_error
from vmp.types import STAGES, ModelArtifact


def _artifact(version: str = "1", stage: str = "candidate", **kw) -> ModelArtifact:
    data = {
        "name": "spoken-sft",
        "version": version,
        "stage": stage,
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": f"runs/sft/{version}",
        "config_hash": "c" * 64,
        "data_hash": "d" * 64,
        "git_sha": "abc123",
        "metrics": {"exact_match": 0.5},
        "created_at": float(version) if version.isdigit() else 1.0,
    }
    data.update(kw)
    return ModelArtifact(**data)


def test_transition_rules():
    assert transition_error("candidate", "staging") is None
    assert transition_error("staging", "production") is None
    assert transition_error("candidate", "retired") is None
    assert transition_error("production", "retired") is None
    assert "cannot move" in transition_error("candidate", "production")
    assert "cannot move" in transition_error("production", "staging")
    assert transition_error("retired", "candidate") == "retired is terminal"
    assert transition_error("retired", "retired") == "retired is terminal"
    assert "already" in transition_error("staging", "staging")
    assert "unknown stage" in transition_error("candidate", "shipped")
    for s in STAGES:
        assert transition_error(s, "shipped") is not None


def test_register_and_get_roundtrip(tmp_path):
    reg = FileRegistry(tmp_path / "reg")
    stored = reg.register(_artifact("1", created_at=0.0))
    assert stored.created_at > 0
    assert reg.get("spoken-sft", "1") == stored
    assert (tmp_path / "reg" / "spoken-sft" / "1.json").is_file()
    with pytest.raises(KeyError):
        reg.get("spoken-sft", "2")


def test_register_rules(tmp_path):
    reg = FileRegistry(tmp_path)
    reg.register(_artifact("1"))
    with pytest.raises(RegistryError, match="already registered"):
        reg.register(_artifact("1"))
    with pytest.raises(RegistryError, match="must be registered as 'candidate'"):
        reg.register(_artifact("2", stage="production"))
    with pytest.raises(RegistryError, match="lineage"):
        reg.register(_artifact("3", config_hash=""))
    with pytest.raises(RegistryError, match="must match"):
        reg.register(_artifact("../4"))
    with pytest.raises(RegistryError, match="must match"):
        reg.register(_artifact("5", name="a/b"))


def test_promotion_path_and_single_production(tmp_path):
    reg = FileRegistry(tmp_path)
    reg.register(_artifact("1"))
    reg.register(_artifact("2"))
    with pytest.raises(RegistryError, match="cannot move"):
        reg.promote("spoken-sft", "1", "production")
    assert reg.promote("spoken-sft", "1", "staging").stage == "staging"
    assert reg.promote("spoken-sft", "1", "production").stage == "production"
    reg.promote("spoken-sft", "2", "staging")
    reg.promote("spoken-sft", "2", "production")
    assert reg.get("spoken-sft", "1").stage == "retired"
    assert reg.get("spoken-sft", "2").stage == "production"
    assert [a.stage for a in reg.list("spoken-sft")].count("production") == 1
    assert reg.latest("spoken-sft", "production").version == "2"
    assert reg.latest("spoken-sft", "staging") is None
    assert reg.latest("other") is None
    with pytest.raises(RegistryError, match="terminal"):
        reg.promote("spoken-sft", "1", "staging")
    with pytest.raises(RegistryError, match="unknown stage"):
        reg.promote("spoken-sft", "2", "shipped")
    events = reg.events()
    assert [e["to"] for e in events] == [
        "candidate",
        "candidate",
        "staging",
        "production",
        "staging",
        "retired",
        "production",
    ]
    assert events[5]["reason"] == "replaced by 2"


def test_retire_from_any_active_stage(tmp_path):
    reg = FileRegistry(tmp_path)
    reg.register(_artifact("1"))
    assert reg.promote("spoken-sft", "1", "retired").stage == "retired"


def test_persistence_across_instances(tmp_path):
    reg = FileRegistry(tmp_path)
    reg.register(_artifact("1"))
    reg.promote("spoken-sft", "1", "staging")
    again = FileRegistry(tmp_path)
    assert again.get("spoken-sft", "1").stage == "staging"
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["artifacts"] == {"spoken-sft": {"1": "staging"}}
    assert not (tmp_path / "index.json.tmp").exists()


def test_list_is_newest_first_and_filters(tmp_path):
    reg = FileRegistry(tmp_path)
    reg.register(_artifact("1", created_at=1.0))
    reg.register(_artifact("2", created_at=2.0))
    reg.register(_artifact("1", name="whisper-accent", created_at=3.0))
    assert [a.version for a in reg.list("spoken-sft")] == ["2", "1"]
    assert [a.name for a in reg.list()] == ["whisper-accent", "spoken-sft", "spoken-sft"]
    assert reg.list("nothing") == []


def test_mlflow_tag_mapping_is_pure():
    a = _artifact("7")
    tags = to_mlflow_tags(a)
    assert tags["vmp.config_hash"] == "c" * 64 and tags["metric.exact_match"] == "0.5"

    class FakeVersion:
        version = "7"
        creation_timestamp = 7000
        tags: ClassVar[dict] = dict(to_mlflow_tags(a))

    back = from_mlflow_version("spoken-sft", FakeVersion())
    assert back == a


def test_registry_cli(tmp_path, capsys):
    root = str(tmp_path / "reg")
    FileRegistry(root).register(_artifact("1"))
    assert main(["registry", "--root", root, "list"]) == 0
    assert "spoken-sft\t1\tcandidate" in capsys.readouterr().out
    promote = ["registry", "--root", root, "promote", "--name", "spoken-sft", "--version", "1"]
    assert main([*promote, "--stage", "staging"]) == 0
    assert capsys.readouterr().out.strip() == "spoken-sft:1 -> staging"
    assert main(["registry", "--root", root, "show", "--name", "spoken-sft", "--version", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "staging"
    assert main([*promote, "--stage", "candidate"]) == 2
    assert "cannot move" in capsys.readouterr().err
    assert main(["registry", "--root", root, "show", "--name", "x", "--version", "1"]) == 2


def test_registry_cli_root_from_env(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("VMP_REGISTRY_ROOT", str(tmp_path / "envreg"))
    assert main(["registry", "list"]) == 0
    assert (tmp_path / "envreg" / "index.json").is_file()
