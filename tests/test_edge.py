"""Edge tests: export plans, bundles and their verification, policy, offline runtime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vmp import __version__ as RUNTIME_VERSION
from vmp.edge.bundle import (
    MANIFEST_NAME,
    EdgeBundle,
    build_bundle,
    diff_bundles,
    sha256_file,
    verify_bundle,
)
from vmp.edge.export import (
    QUANT_METHODS,
    TARGETS,
    ExportPlan,
    export,
    plan_from_config,
)
from vmp.edge.policy import (
    ACTION_CLOUD_FALLBACK,
    ACTION_NETWORK,
    ACTION_STORE_AUDIO,
    Action,
    EdgePolicy,
    PolicyViolation,
    check,
    enforce,
    policy_from_config,
)
from vmp.edge.runtime import EdgeRuntime, EdgeTurn, local_segmenter, resolve_segmenter
from vmp.observability.trace import ListSink, Tracer

ROOT = Path(__file__).resolve().parents[1]


# -- stub backends ----------------------------------------------------------


class StubSTT:
    """Decodes the "audio" bytes, the way the edge runtime expects a plain string."""

    def transcribe(self, audio: bytes | str) -> str:
        return audio.decode("utf-8") if isinstance(audio, bytes) else str(audio)


class StubLLM:
    def __init__(self, answer: str = "Local answer here. A second sentence follows.") -> None:
        self.answer = answer
        self.seen: list[list[dict[str, str]]] = []

    def stream(self, messages):
        self.seen.append([dict(m) for m in messages])
        for word in self.answer.split(" "):
            yield word + " "

    def complete(self, messages) -> str:
        return "".join(self.stream(messages))


class BrokenLLM:
    def __init__(self) -> None:
        self.seen: list[list[dict[str, str]]] = []

    def stream(self, messages):
        self.seen.append([dict(m) for m in messages])
        raise RuntimeError("local model failed to load")
        yield ""  # pragma: no cover - unreachable, keeps this a generator


class StubTTS:
    sample_rate = 24000

    def synthesize(self, text: str) -> bytes:
        return b"\x00" * max(len(text), 1)


class StubCloud:
    host = "api.example.com"

    def complete(self, messages) -> str:
        return "Cloud answer."


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def src_dir(tmp_path: Path) -> Path:
    src = tmp_path / "export"
    (src / "nested").mkdir(parents=True)
    (src / "model-q4_k_m.gguf").write_bytes(b"weights-v1")
    (src / "tokenizer.json").write_text('{"vocab": 1}')
    (src / "nested" / "extra.bin").write_bytes(b"extra")
    return src


def _build(src: Path, out: Path, **kw) -> EdgeBundle:
    kw.setdefault("name", "voice-agent-edge")
    kw.setdefault("version", "1.0.0")
    kw.setdefault("target", "gguf")
    kw.setdefault("base_model", "google/gemma-3-270m")
    return build_bundle(src, out, **kw)


# ===========================================================================
# export
# ===========================================================================


@pytest.mark.parametrize(
    ("target", "method", "expected_steps"),
    [
        ("onnx", "int8", ["validate", "convert_onnx", "quantize_int8", "write_manifest"]),
        ("onnx", "none", ["validate", "convert_onnx", "write_manifest"]),
        ("gguf", "q4_k_m", ["validate", "convert_gguf", "quantize_q4_k_m", "write_manifest"]),
        ("mlx", "int4", ["validate", "convert_mlx", "write_manifest"]),
    ],
)
def test_export_dry_run_manifest_per_target(tmp_path: Path, target, method, expected_steps):
    plan = ExportPlan(
        base_model="google/gemma-3-270m",
        target=target,
        out_dir=str(tmp_path / "out"),
        quantization={"method": method},
    )
    manifest = export(plan, dry_run=True)

    assert manifest["dry_run"] is True
    assert manifest["problems"] == []
    assert manifest["target"] == target
    assert manifest["quantization"] == {"method": method}
    assert [s["name"] for s in manifest["steps"]] == expected_steps
    assert {s["status"] for s in manifest["steps"]} == {"planned"}
    assert manifest["created_at"] > 0
    assert manifest["plan"]["base_model"] == "google/gemma-3-270m"
    # A dry run writes nothing and imports no converter.
    assert not (tmp_path / "out").exists()
    assert json.loads(json.dumps(manifest)) == manifest


def test_export_dry_run_lists_expected_files_and_dependencies(tmp_path: Path):
    onnx = export(
        ExportPlan("m", "onnx", str(tmp_path / "o"), quantization={"method": "int8"}), dry_run=True
    )
    assert onnx["files"] == ["model.onnx", "config.json", "tokenizer.json", "model-int8.onnx"]
    requires = {s["name"]: s.get("requires") for s in onnx["steps"]}
    assert requires["convert_onnx"] == ["optimum", "onnx"]
    assert requires["quantize_int8"] == ["onnxruntime"]

    gguf = export(
        ExportPlan("m", "gguf", str(tmp_path / "g"), quantization={"method": "q4_k_m"}),
        dry_run=True,
    )
    assert gguf["files"] == ["model-f16.gguf", "model-q4_k_m.gguf"]

    mlx = export(ExportPlan("m", "mlx", str(tmp_path / "x")), dry_run=True)
    assert mlx["files"] == ["weights.safetensors", "config.json", "tokenizer.json"]


def test_export_merges_an_adapter_when_one_is_given(tmp_path: Path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"lora")

    plan = ExportPlan("m", "gguf", str(tmp_path / "out"), adapter_path=str(adapter))
    names = [s["name"] for s in export(plan, dry_run=True)["steps"]]
    assert names[:2] == ["validate", "merge_adapter"]

    without = ExportPlan("m", "gguf", str(tmp_path / "out"), adapter_path=str(adapter),
                         merge_adapter=False)
    assert "merge_adapter" not in [s["name"] for s in export(without, dry_run=True)["steps"]]


@pytest.mark.parametrize(
    ("plan", "problem"),
    [
        (ExportPlan("", "gguf", "/tmp/o"), "base_model is empty"),
        (ExportPlan("m", "tflite", "/tmp/o"), "not in"),
        (ExportPlan("m", "gguf", ""), "out_dir is empty"),
        (ExportPlan("m", "onnx", "/tmp/o", quantization={"method": "q4_k_m"}), "not supported"),
        (ExportPlan("m", "gguf", "/tmp/o", quantization={"method": "fp3"}), "not in"),
        (ExportPlan("m", "gguf", "/tmp/o", adapter_path="/no/such/adapter"), "does not exist"),
    ],
)
def test_invalid_plans_are_reported_and_every_step_skipped(plan, problem):
    assert any(problem in p for p in plan.validate())
    manifest = export(plan, dry_run=True)
    assert manifest["problems"]
    assert {s["status"] for s in manifest["steps"]} == {"skipped"}


def test_export_plan_round_trips_and_reads_the_repository_config():
    plan = ExportPlan.from_dict(
        {"base_model": "m", "target": "mlx", "out_dir": "o", "quantization": {"method": "int4"}}
    )
    assert plan.method == "int4"
    assert ExportPlan.from_dict(plan.to_dict()) == plan

    from vmp.config import load_config

    configured = plan_from_config(load_config(ROOT / "configs" / "edge.toml"))
    assert configured.target in TARGETS
    assert configured.method in QUANT_METHODS
    # The configured adapter path is a build artefact, so it may be absent in a checkout;
    # nothing else about the shipped config may be invalid.
    assert all("adapter_path" in problem for problem in configured.validate())


# ===========================================================================
# bundle
# ===========================================================================


def test_build_bundle_copies_files_and_writes_a_manifest(tmp_path: Path, src_dir: Path):
    out = tmp_path / "bundle"
    bundle = _build(src_dir, out)

    assert sorted(bundle.files) == ["model-q4_k_m.gguf", "nested/extra.bin", "tokenizer.json"]
    assert (out / "nested" / "extra.bin").read_bytes() == b"extra"
    assert bundle.name == "voice-agent-edge"
    assert bundle.version == "1.0.0"
    assert bundle.manifest["schema"] == 1
    assert bundle.manifest["built_with_runtime"] == RUNTIME_VERSION
    assert bundle.manifest["required"] == sorted(bundle.files)
    assert bundle.files["model-q4_k_m.gguf"] == {
        "sha256": sha256_file(src_dir / "model-q4_k_m.gguf"),
        "size": 10,
    }

    on_disk = json.loads((out / MANIFEST_NAME).read_text())
    assert on_disk == bundle.manifest
    assert EdgeBundle.load(out).manifest == bundle.manifest


def test_build_bundle_dry_run_writes_nothing(tmp_path: Path, src_dir: Path):
    out = tmp_path / "bundle"
    bundle = _build(src_dir, out, dry_run=True)
    assert not out.exists()
    assert sorted(bundle.files) == ["model-q4_k_m.gguf", "nested/extra.bin", "tokenizer.json"]


def test_build_bundle_records_lineage_and_policy(tmp_path: Path, src_dir: Path):
    policy = EdgePolicy(offline_only=False, allowed_egress=["api.example.com"], fallback="cloud")
    bundle = _build(
        src_dir,
        tmp_path / "b",
        lineage={"artifact": "voice-sft", "version": "3", "git_sha": "abc123"},
        adapter_version="3",
        policy=policy,
    )
    assert bundle.manifest["lineage"]["git_sha"] == "abc123"
    assert bundle.manifest["adapter_version"] == "3"
    assert bundle.policy() == policy


def test_build_bundle_rejects_an_invalid_policy(tmp_path: Path, src_dir: Path):
    with pytest.raises(ValueError, match="invalid policy"):
        _build(src_dir, tmp_path / "b", policy=EdgePolicy(offline_only=True, fallback="cloud"))


def test_build_bundle_needs_an_existing_source(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _build(tmp_path / "missing", tmp_path / "b")


def test_verify_a_good_bundle(tmp_path: Path, src_dir: Path):
    bundle = _build(src_dir, tmp_path / "b")
    assert verify_bundle(tmp_path / "b") == (True, [])
    assert bundle.verify() == (True, [])


def test_verify_detects_a_tampered_file(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    (out / "model-q4_k_m.gguf").write_bytes(b"trojan-v1")  # same length class, new bytes

    ok, problems = verify_bundle(out)
    assert ok is False
    assert "checksum mismatch: model-q4_k_m.gguf" in problems


def test_verify_detects_a_deleted_file_and_an_unlisted_file(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    (out / "tokenizer.json").unlink()
    (out / "sneaky.so").write_bytes(b"payload")

    ok, problems = verify_bundle(out)
    assert ok is False
    assert "required file missing: tokenizer.json" in problems
    assert "unlisted file: sneaky.so" in problems
    # Without `strict` an extra file is tolerated but a missing one is not.
    ok, problems = verify_bundle(out, strict=False)
    assert ok is False
    assert not any("unlisted" in p for p in problems)


def test_verify_detects_a_tampered_manifest(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    del manifest["policy"]
    manifest["files"]["model-q4_k_m.gguf"]["sha256"] = "0" * 64
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))

    ok, problems = verify_bundle(out)
    assert ok is False
    assert "manifest missing key 'policy'" in problems
    assert "checksum mismatch: model-q4_k_m.gguf" in problems

    (out / MANIFEST_NAME).write_text("{ not json")
    ok, problems = verify_bundle(out)
    assert ok is False
    assert "not valid JSON" in problems[0]


def test_verify_missing_bundle_and_manifest(tmp_path: Path, src_dir: Path):
    ok, problems = verify_bundle(tmp_path / "nope")
    assert (ok, len(problems)) == (False, 1)
    assert "bundle directory not found" in problems[0]

    empty = tmp_path / "empty"
    empty.mkdir()
    assert verify_bundle(empty) == (False, [f"missing {MANIFEST_NAME}"])


def test_verify_checks_the_runtime_version(tmp_path: Path, src_dir: Path):
    _build(src_dir, tmp_path / "b", min_runtime_version="9.0.0")
    ok, problems = verify_bundle(tmp_path / "b")
    assert ok is False
    assert any("min_runtime_version 9.0.0" in p for p in problems)
    assert verify_bundle(tmp_path / "b", runtime_version="9.1.0") == (True, [])


def test_diff_bundles_lists_what_an_update_must_ship(tmp_path: Path, src_dir: Path):
    _build(src_dir, tmp_path / "v1")

    next_src = tmp_path / "export2"
    (next_src / "nested").mkdir(parents=True)
    (next_src / "model-q4_k_m.gguf").write_bytes(b"weights-v2-bigger")
    (next_src / "tokenizer.json").write_text('{"vocab": 1}')
    (next_src / "nested" / "extra.bin").write_bytes(b"extra")
    (next_src / "projector.bin").write_bytes(b"new-file")
    _build(next_src, tmp_path / "v2", version="2.0.0")

    diff = diff_bundles(tmp_path / "v1", tmp_path / "v2")
    assert diff["from"] == {"name": "voice-agent-edge", "version": "1.0.0"}
    assert diff["to"] == {"name": "voice-agent-edge", "version": "2.0.0"}
    assert diff["added"] == ["projector.bin"]
    assert diff["changed"] == ["model-q4_k_m.gguf"]
    assert diff["removed"] == []
    assert diff["unchanged"] == ["nested/extra.bin", "tokenizer.json"]
    assert diff["download_bytes"] == len(b"new-file") + len(b"weights-v2-bigger")
    assert diff["policy_changed"] is False


def test_diff_bundles_reports_removals_and_policy_changes(tmp_path: Path, src_dir: Path):
    a = _build(src_dir, tmp_path / "v1")
    (src_dir / "nested" / "extra.bin").unlink()
    b = _build(
        src_dir,
        tmp_path / "v2",
        version="2.0.0",
        policy=EdgePolicy(offline_only=False, allowed_egress=["api.example.com"]),
    )
    diff = diff_bundles(a, b)  # accepts loaded bundles as well as paths
    assert diff["removed"] == ["nested/extra.bin"]
    assert diff["added"] == []
    assert diff["policy_changed"] is True


# ===========================================================================
# policy
# ===========================================================================


def test_offline_only_policy_refuses_network_and_cloud_fallback():
    policy = EdgePolicy()
    assert policy.offline_only is True
    assert policy.validate() == []

    with pytest.raises(PolicyViolation, match="offline_only"):
        enforce(policy, Action(ACTION_NETWORK, "api.example.com"))
    with pytest.raises(PolicyViolation, match="cloud fallback refused"):
        enforce(policy, Action(ACTION_CLOUD_FALLBACK))
    with pytest.raises(PolicyViolation, match="must not be stored"):
        enforce(policy, {"kind": ACTION_STORE_AUDIO})

    allowed, reason = check(policy, Action(ACTION_NETWORK, "api.example.com"))
    assert allowed is False
    assert "offline_only" in reason


def test_connected_policy_allows_listed_hosts_only():
    policy = EdgePolicy(
        offline_only=False,
        allowed_egress=["api.example.com", "*.internal"],
        fallback="cloud",
        retention={"audio": "local"},
    )
    assert policy.validate() == []
    assert enforce(policy, Action(ACTION_NETWORK, "api.example.com"))
    assert enforce(policy, Action(ACTION_NETWORK, "metrics.internal"))
    assert enforce(policy, Action(ACTION_CLOUD_FALLBACK))
    assert enforce(policy, Action(ACTION_STORE_AUDIO))

    with pytest.raises(PolicyViolation, match="not in allowed_egress"):
        enforce(policy, Action(ACTION_NETWORK, "evil.example.org"))
    with pytest.raises(PolicyViolation):
        enforce(policy, Action(ACTION_NETWORK, None))


def test_cloud_fallback_needs_the_cloud_fallback_policy():
    policy = EdgePolicy(offline_only=False, allowed_egress=["api.example.com"], fallback="degrade")
    allowed, reason = check(policy, Action(ACTION_CLOUD_FALLBACK))
    assert allowed is False
    assert "not 'cloud'" in reason


def test_unknown_action_is_refused():
    allowed, reason = check(EdgePolicy(), Action("launch_missiles"))
    assert allowed is False
    assert "unknown action kind" in reason


@pytest.mark.parametrize(
    ("policy", "problem"),
    [
        (EdgePolicy(fallback="teleport"), "fallback"),
        (EdgePolicy(max_ttfa_ms=0), "max_ttfa_ms"),
        (EdgePolicy(retention={"audio": "cloud"}), "retention.audio"),
        (EdgePolicy(offline_only=True, allowed_egress=["a.com"]), "contradictory"),
        (EdgePolicy(offline_only=True, fallback="cloud"), "contradictory"),
        (EdgePolicy(offline_only=False, allowed_egress=["https://a.com"]), "must be a host"),
    ],
)
def test_policy_validation_problems(policy, problem):
    assert any(problem in p for p in policy.validate())


def test_policy_round_trip_and_config():
    from vmp.config import load_config

    policy = policy_from_config(load_config(ROOT / "configs" / "edge.toml"))
    assert policy.offline_only is True
    assert policy.fallback == "degrade"
    assert policy.retention == {"audio": "none"}
    assert policy.validate() == []
    assert EdgePolicy.from_dict(policy.to_dict()) == policy


# ===========================================================================
# runtime
# ===========================================================================


def test_local_segmenter_and_resolution():
    chunks = ["Hello there. ", "A second one! ", "A tail"]
    assert list(local_segmenter(chunks)) == ["Hello there.", "A second one!", "A tail"]
    # `vmp.serving.segment.segment` is importable here, so it wins.
    from vmp.serving.segment import segment

    assert resolve_segmenter() is segment


def test_edge_runtime_runs_one_turn_offline(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    sink = ListSink()
    llm = StubLLM()
    runtime = EdgeRuntime(out, StubSTT(), llm, StubTTS(), tracer=Tracer(sink))

    assert runtime.start() is True
    assert runtime.verified is True
    assert runtime.problems == []

    turn = runtime.turn(b"what time is it", session="s1")
    assert isinstance(turn, EdgeTurn)
    assert turn.transcript == "what time is it"
    assert turn.segments == ["Local answer here.", "A second sentence follows."]
    assert turn.assistant_text.strip() == " ".join(turn.segments)
    assert len(turn.audio) == 2
    assert turn.refused is False
    assert turn.fallback_used is False
    assert turn.ttfa_ms is not None and turn.ttfa_ms > 0

    # Stages come from the trace of this turn only.
    assert set(turn.stages) == {"stt", "llm", "segment.emit", "tts", "playback", "turn"}
    events = [r["event"] for r in sink.rows]
    assert events[0] == "turn.start"
    assert events[-1] == "turn.end"
    assert events.index("stt.end") < events.index("segment.emit.start")
    playback = next(r for r in sink.rows if r["event"] == "playback.end")
    assert playback["payload"]["response_ms"] == turn.ttfa_ms
    # The LLM saw the system prompt and the transcript.
    assert llm.seen[0][0]["role"] == "system"
    assert llm.seen[0][-1] == {"role": "user", "content": "what time is it"}


def test_edge_runtime_keeps_history_across_turns(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    llm = StubLLM()
    runtime = EdgeRuntime(out, StubSTT(), llm, StubTTS())
    runtime.start()

    runtime.turn(b"first question", session="s1")
    second = runtime.turn(b"second question", session="s1")
    assert second.turn == 2
    assert [m["role"] for m in llm.seen[1]] == ["system", "user", "assistant", "user"]
    assert runtime.turns == 2


def test_edge_runtime_health(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    runtime = EdgeRuntime(out, StubSTT(), StubLLM(), StubTTS())
    runtime.start()
    runtime.turn(b"hello", session="s1")

    health = runtime.health()
    assert health["status"] == "ok"
    assert health["bundle"] == "voice-agent-edge"
    assert health["bundle_version"] == "1.0.0"
    assert health["runtime_version"] == RUNTIME_VERSION
    assert health["verified"] is True
    assert health["problems"] == []
    assert health["turns"] == 1
    assert health["refusals"] == 0
    assert health["errors"] == 0
    assert health["offline_only"] is True
    assert health["last_ttfa_ms"] is not None
    assert health["uptime_s"] >= 0.0


def test_edge_runtime_refuses_to_start_on_a_tampered_bundle(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    (out / "model-q4_k_m.gguf").write_bytes(b"trojan")
    runtime = EdgeRuntime(out, StubSTT(), StubLLM(), StubTTS())

    with pytest.raises(RuntimeError, match="bundle verification failed"):
        runtime.start()

    assert runtime.start(require_verified=False) is False
    health = runtime.health()
    assert health["status"] == "degraded"
    assert any("checksum mismatch" in p for p in health["problems"])


def test_edge_runtime_degrades_when_the_local_model_fails(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    runtime = EdgeRuntime(out, StubSTT(), BrokenLLM(), StubTTS())
    runtime.start()

    turn = runtime.turn(b"hello", session="s1")
    assert turn.assistant_text == "I cannot answer that right now."
    assert turn.refused is False
    assert turn.fallback_used is False
    assert runtime.errors == 1
    assert runtime.health()["errors"] == 1


def test_edge_runtime_refuse_policy_marks_the_turn_refused(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out, policy=EdgePolicy(fallback="refuse"))
    sink = ListSink()
    runtime = EdgeRuntime(out, StubSTT(), BrokenLLM(), StubTTS(), tracer=Tracer(sink))
    runtime.start()

    turn = runtime.turn(b"hello", session="s1")
    assert turn.refused is True
    assert turn.assistant_text == ""
    assert runtime.refusals == 1
    turn_end = next(r for r in sink.rows if r["event"] == "turn.end")
    assert turn_end["payload"]["refused"] is True
    assert turn_end["payload"]["error"] == "RuntimeError"


def test_edge_runtime_cloud_fallback_is_refused_offline(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    runtime = EdgeRuntime(
        out,
        StubSTT(),
        BrokenLLM(),
        StubTTS(),
        policy=EdgePolicy(offline_only=True, fallback="degrade"),
        cloud=StubCloud(),
    )
    runtime.start()
    # `degrade` never reaches the network.
    assert runtime.turn(b"hi", session="s").assistant_text == "I cannot answer that right now."

    connected = EdgeRuntime(
        out,
        StubSTT(),
        BrokenLLM(),
        StubTTS(),
        policy=EdgePolicy(
            offline_only=False, allowed_egress=["api.example.com"], fallback="cloud"
        ),
        cloud=StubCloud(),
    )
    connected.start()
    turn = connected.turn(b"hi", session="s")
    assert turn.assistant_text == "Cloud answer."
    assert turn.fallback_used is True


def test_edge_runtime_cloud_fallback_blocked_by_egress_list(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    runtime = EdgeRuntime(
        out,
        StubSTT(),
        BrokenLLM(),
        StubTTS(),
        policy=EdgePolicy(offline_only=False, allowed_egress=["other.host"], fallback="cloud"),
        cloud=StubCloud(),
    )
    runtime.start()
    with pytest.raises(PolicyViolation, match="not in allowed_egress"):
        runtime.turn(b"hi", session="s")
    assert runtime.refusals == 1


def test_edge_runtime_flags_a_turn_slower_than_the_policy_budget(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out, policy=EdgePolicy(max_ttfa_ms=1))

    class SlowTTS(StubTTS):
        def synthesize(self, text: str) -> bytes:
            import time

            time.sleep(0.01)
            return super().synthesize(text)

    sink = ListSink()
    runtime = EdgeRuntime(out, StubSTT(), StubLLM(), SlowTTS(), tracer=Tracer(sink))
    runtime.start()
    turn = runtime.turn(b"hello", session="s1")

    assert turn.ttfa_ms > 1
    turn_end = next(r for r in sink.rows if r["event"] == "turn.end")
    assert turn_end["payload"]["over_max_ttfa"] is True


def test_edge_runtime_starts_lazily_on_the_first_turn(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    runtime = EdgeRuntime(out, StubSTT(), StubLLM(), StubTTS())
    assert runtime.bundle is None
    runtime.turn(b"hello", session="s1")
    assert runtime.bundle is not None
    assert runtime.verified is True


def test_edge_turn_to_dict_is_json_serialisable(tmp_path: Path, src_dir: Path):
    out = tmp_path / "b"
    _build(src_dir, out)
    runtime = EdgeRuntime(out, StubSTT(), StubLLM(), StubTTS())
    runtime.start()
    payload = runtime.turn(b"hello", session="s1").to_dict()
    assert json.loads(json.dumps(payload))["audio_bytes"] > 0
    assert payload["session"] == "s1"
