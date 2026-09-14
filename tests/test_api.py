"""HTTP and WebSocket API tests. Skipped unless `fastapi` (and `httpx`) are installed."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from vmp.serving.api import (
    LATENCY_BUCKETS_S,
    Metrics,
    check_bearer,
    create_app,
    parse_multipart,
    turn_to_json,
)
from vmp.serving.backends import EchoSTT, SilentTTS, TemplateLLM
from vmp.serving.runtime import (
    InMemorySessionStore,
    ListTraceSink,
    VoiceRuntime,
)
from vmp.types import Turn, Utterance


def _runtime(**kw) -> VoiceRuntime:
    return VoiceRuntime(EchoSTT(), TemplateLLM(), SilentTTS(), **kw)


@pytest.fixture
def client() -> TestClient:
    """App with authentication disabled."""
    app = create_app(_runtime(tracer=ListTraceSink()), InMemorySessionStore(), api_token="")
    with TestClient(app) as c:
        yield c


def _new_session(client: TestClient, **body) -> str:
    response = client.post("/v1/sessions", json=body)
    assert response.status_code == 201
    return response.json()["id"]


# -- units ------------------------------------------------------------------


def test_check_bearer_semantics():
    assert check_bearer(None, "") is True  # auth off
    assert check_bearer(None, None) is True
    assert check_bearer("Bearer tok", "tok") is True
    assert check_bearer("Bearer wrong", "tok") is False
    assert check_bearer("tok", "tok") is False
    assert check_bearer(None, "tok") is False


def test_parse_multipart_extracts_named_fields():
    boundary = "abc123"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
        "hello bytes\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="note"\r\n\r\n'
        "meta\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    fields = parse_multipart(body, f"multipart/form-data; boundary={boundary}")
    assert fields["audio"] == b"hello bytes"
    assert fields["note"] == b"meta"
    assert parse_multipart(b"plain", "text/plain") == {}


def test_turn_to_json_rounds_stages_and_adds_sentences():
    turn = Turn(
        session_id="s",
        turn=1,
        user=Utterance(id="u", text="hi"),
        assistant_text="Hello there.",
        stages={"llm": 1.23456},
    )
    payload = turn_to_json(turn, ["Hello there."], 512.5)
    assert payload["stages"] == {"llm": 1.235}
    assert payload["sentences"] == ["Hello there."]
    assert payload["audio_ms"] == 512.5
    assert payload["user"]["text"] == "hi"


def test_metrics_render_is_prometheus_exposition_format():
    m = Metrics()
    m.describe("vmp_demo_total", "demo counter")
    m.inc("vmp_demo_total", {"kind": "a"})
    m.inc("vmp_demo_total", {"kind": "a"}, value=2.0)
    m.inc("vmp_demo_total", {"kind": "b"})
    m.observe("vmp_demo_seconds", 0.03)
    text = m.render()
    lines = text.splitlines()

    assert text.endswith("\n")
    assert "# HELP vmp_demo_total demo counter" in lines
    assert "# TYPE vmp_demo_total counter" in lines
    assert 'vmp_demo_total{kind="a"} 3' in lines
    assert 'vmp_demo_total{kind="b"} 1' in lines
    assert "# TYPE vmp_demo_seconds histogram" in lines
    assert 'vmp_demo_seconds_bucket{le="+Inf"} 1' in lines
    assert "vmp_demo_seconds_count 1" in lines
    assert "vmp_demo_seconds_sum 0.03" in lines
    # Buckets are cumulative: nothing at 0.025, everything from 0.05 up.
    assert 'vmp_demo_seconds_bucket{le="0.025"} 0' in lines
    assert 'vmp_demo_seconds_bucket{le="0.05"} 1' in lines
    assert len([ln for ln in lines if "_bucket{" in ln]) == len(LATENCY_BUCKETS_S) + 1
    assert Metrics().render() == ""


# -- sessions ---------------------------------------------------------------


def test_create_and_get_session(client: TestClient):
    response = client.post("/v1/sessions", json={"system_prompt": "be brief", "meta": {"a": 1}})
    assert response.status_code == 201
    body = response.json()
    assert body["system_prompt"] == "be brief"
    assert body["created_at"] > 0

    got = client.get(f"/v1/sessions/{body['id']}")
    assert got.status_code == 200
    assert got.json()["id"] == body["id"]
    assert got.json()["turns"] == []
    assert got.json()["meta"] == {"a": 1}


def test_create_session_without_a_body(client: TestClient):
    assert client.post("/v1/sessions").status_code == 201


def test_create_session_rejects_bad_json(client: TestClient):
    response = client.post(
        "/v1/sessions", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json() == {"error": "invalid json"}


def test_unknown_session_is_404(client: TestClient):
    assert client.get("/v1/sessions/nope").status_code == 404
    assert client.get("/v1/sessions/nope").json() == {"error": "session not found"}
    assert client.post("/v1/sessions/nope/turns", json={"text": "hi"}).status_code == 404


def test_request_id_is_echoed(client: TestClient):
    assert client.get("/healthz", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"
    assert client.get("/healthz").headers["x-request-id"]


# -- turns ------------------------------------------------------------------


def test_text_turn_returns_sentences_stages_and_audio(client: TestClient):
    sid = _new_session(client)
    response = client.post(f"/v1/sessions/{sid}/turns", json={"text": "hello there"})
    assert response.status_code == 200
    body = response.json()

    assert body["turn"] == 1
    assert body["session_id"] == sid
    assert body["user"]["text"] == "hello there"
    assert body["assistant_text"]
    assert body["sentences"] == ["Sure.", "I can help with that."]
    assert body["assistant_text"] == " ".join(body["sentences"])
    assert body["audio_ms"] > 0
    assert "first_audio" in body["stages"]
    assert body["stages"]["turn"] >= body["stages"]["first_audio"]

    # The turn is persisted on the session.
    session = client.get(f"/v1/sessions/{sid}").json()
    assert len(session["turns"]) == 1
    assert session["turns"][0]["assistant_text"] == body["assistant_text"]

    second = client.post(f"/v1/sessions/{sid}/turns", json={"text": "and again"})
    assert second.json()["turn"] == 2


def test_turn_requires_non_empty_text(client: TestClient):
    sid = _new_session(client)
    assert client.post(f"/v1/sessions/{sid}/turns", json={}).status_code == 422
    assert client.post(f"/v1/sessions/{sid}/turns", json={"text": "  "}).status_code == 422
    bad = client.post(
        f"/v1/sessions/{sid}/turns",
        content=b"{oops",
        headers={"content-type": "application/json"},
    )
    assert bad.status_code == 400


def test_audio_turn_via_multipart(client: TestClient):
    sid = _new_session(client)
    response = client.post(
        f"/v1/sessions/{sid}/turns", files={"audio": ("clip.wav", b"spoken question")}
    )
    assert response.status_code == 200
    assert response.json()["user"]["text"] == "spoken question"

    missing = client.post(f"/v1/sessions/{sid}/turns", files={"other": ("x", b"y")})
    assert missing.status_code == 422


def test_request_size_limit():
    app = create_app(
        _runtime(), InMemorySessionStore(), api_token="", max_request_bytes=64
    )
    with TestClient(app) as c:
        sid = _new_session(c)
        response = c.post(f"/v1/sessions/{sid}/turns", json={"text": "x" * 500})
        assert response.status_code == 413


def test_turn_timeout_returns_504():
    class SlowLLM:
        def stream(self, messages):
            time.sleep(0.5)
            yield "Too late."

        def complete(self, messages):
            return "Too late."

    app = create_app(
        VoiceRuntime(EchoSTT(), SlowLLM(), SilentTTS()),
        InMemorySessionStore(),
        api_token="",
        turn_timeout_s=0.05,
    )
    with TestClient(app) as c:
        sid = _new_session(c)
        assert c.post(f"/v1/sessions/{sid}/turns", json={"text": "hi"}).status_code == 504


# -- auth -------------------------------------------------------------------


def test_bearer_auth_off_by_default(monkeypatch):
    monkeypatch.delenv("VMP_API_TOKEN", raising=False)
    app = create_app(_runtime(), InMemorySessionStore())
    with TestClient(app) as c:
        assert c.post("/v1/sessions").status_code == 201


def test_bearer_auth_from_environment(monkeypatch):
    monkeypatch.setenv("VMP_API_TOKEN", "s3cret")
    auth = {"authorization": "Bearer s3cret"}
    app = create_app(_runtime(), InMemorySessionStore())
    with TestClient(app) as c:
        assert c.post("/v1/sessions").status_code == 401
        assert c.post("/v1/sessions", headers={"authorization": "Bearer wrong"}).status_code == 401
        assert c.post("/v1/sessions", headers={"authorization": "s3cret"}).status_code == 401

        created = c.post("/v1/sessions", headers=auth)
        assert created.status_code == 201
        sid = created.json()["id"]
        assert c.get(f"/v1/sessions/{sid}").status_code == 401
        assert c.get(f"/v1/sessions/{sid}", headers=auth).status_code == 200
        turn = c.post(f"/v1/sessions/{sid}/turns", json={"text": "hi"}, headers=auth)
        assert turn.status_code == 200


def test_health_endpoints_stay_unauthenticated(monkeypatch):
    monkeypatch.setenv("VMP_API_TOKEN", "s3cret")
    app = create_app(_runtime(), InMemorySessionStore())
    with TestClient(app) as c:
        for path in ("/healthz", "/readyz", "/metrics"):
            assert c.get(path).status_code in (200, 503), path


def test_explicit_token_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("VMP_API_TOKEN", "from-env")
    app = create_app(_runtime(), InMemorySessionStore(), api_token="explicit")
    with TestClient(app) as c:
        stale = c.post("/v1/sessions", headers={"authorization": "Bearer from-env"})
        assert stale.status_code == 401
        fresh = c.post("/v1/sessions", headers={"authorization": "Bearer explicit"})
        assert fresh.status_code == 201


# -- health and metrics -----------------------------------------------------


def test_healthz(client: TestClient):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_each_backend(client: TestClient):
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"stt": True, "llm": True, "tts": True}}


def test_readyz_is_503_when_a_backend_is_not_ready():
    class NotReadyTTS(SilentTTS):
        def ready(self) -> bool:
            return False

    app = create_app(
        VoiceRuntime(EchoSTT(), TemplateLLM(), NotReadyTTS()), InMemorySessionStore(), api_token=""
    )
    with TestClient(app) as c:
        response = c.get("/readyz")
        assert response.status_code == 503
        assert response.json()["status"] == "not ready"
        assert response.json()["checks"]["tts"] is False


def test_readyz_treats_a_raising_probe_as_not_ready():
    class BoomLLM(TemplateLLM):
        def ready(self) -> bool:
            raise RuntimeError("probe failed")

    app = create_app(
        VoiceRuntime(EchoSTT(), BoomLLM(), SilentTTS()), InMemorySessionStore(), api_token=""
    )
    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 503


def test_metrics_exposition_format(client: TestClient):
    sid = _new_session(client)
    client.post(f"/v1/sessions/{sid}/turns", json={"text": "hello"})
    client.get("/v1/sessions/nope")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=0.0.4")
    text = response.text
    lines = text.splitlines()

    assert "# HELP vmp_turns_total Completed voice turns" in lines
    assert "# TYPE vmp_turns_total counter" in lines
    assert "vmp_turns_total 1" in lines
    assert "# TYPE vmp_http_requests_total counter" in lines
    assert 'vmp_http_requests_total{method="POST",path="/v1/sessions",status="201"} 1' in lines
    assert any('path="/v1/sessions/{session_id}",status="404"' in ln for ln in lines)
    # Route templates, not raw paths, keep the label cardinality bounded.
    assert sid not in text

    assert "# TYPE vmp_turn_stage_seconds histogram" in lines
    assert any(
        ln.startswith('vmp_turn_stage_seconds_bucket{stage="llm",le="+Inf"}') for ln in lines
    )
    assert any(ln.startswith("vmp_first_audio_seconds_count") for ln in lines)
    # Every sample line names a metric declared by a preceding TYPE line.
    types = {ln.split()[2] for ln in lines if ln.startswith("# TYPE ")}
    for line in lines:
        if line.startswith("#"):
            continue
        name = line.split("{")[0].split()[0]
        base = name.rsplit("_", 1)[0] if name.endswith(("_bucket", "_sum", "_count")) else name
        assert name in types or base in types, line


def test_metrics_can_be_shared_across_apps():
    metrics = Metrics()
    app = create_app(_runtime(), InMemorySessionStore(), metrics=metrics, api_token="")
    with TestClient(app) as c:
        c.get("/healthz")
    assert "vmp_http_requests_total" in metrics.render()


# -- websocket --------------------------------------------------------------


def test_websocket_streams_sentence_then_audio_then_done(client: TestClient):
    sid = _new_session(client)
    with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
        ws.send_json({"text": "hello there"})
        messages = []
        while True:
            message = ws.receive_json()
            messages.append(message)
            if message["type"] == "done":
                break

    kinds = [m["type"] for m in messages]
    assert kinds[0] == "sentence"
    assert kinds[-1] == "done"
    assert "audio" in kinds
    assert kinds.count("sentence") == kinds.count("audio") == 2

    sentences = [m for m in messages if m["type"] == "sentence"]
    audio = [m for m in messages if m["type"] == "audio"]
    assert [m["seq"] for m in sentences] == [1, 2]
    assert [m["seq"] for m in audio] == [1, 2]
    assert [m["text"] for m in sentences] == ["Sure.", "I can help with that."]
    # A sentence always precedes the audio for the same seq.
    for clip in audio:
        first_with_seq = next(m for m in messages if m["seq"] == clip["seq"])
        assert first_with_seq["type"] == "sentence"

    for clip in audio:
        assert clip["sample_rate"] == 24000
        assert clip["ms"] > 0
        assert clip["bytes_b64"]

    done = messages[-1]
    assert done["turn"] == 1
    assert done["text"] == " ".join(m["text"] for m in sentences)
    assert "first_audio" in done["stages"]


def test_websocket_handles_several_turns_and_bad_messages(client: TestClient):
    sid = _new_session(client)
    with client.websocket_connect(f"/v1/sessions/{sid}/stream") as ws:
        ws.send_json({"nope": 1})
        error = ws.receive_json()
        assert error == {"type": "error", "error": "field 'text' required"}

        turns = []
        for text in ("first question", "second question"):
            ws.send_json({"text": text})
            while True:
                message = ws.receive_json()
                if message["type"] == "done":
                    turns.append(message["turn"])
                    break
    assert turns == [1, 2]
    assert len(client.get(f"/v1/sessions/{sid}").json()["turns"]) == 2


def test_websocket_reports_an_unknown_session(client: TestClient):
    from starlette.websockets import WebSocketDisconnect

    with client.websocket_connect("/v1/sessions/nope/stream") as ws:
        assert ws.receive_json() == {"type": "error", "error": "session not found"}
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_websocket_requires_the_bearer_token(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("VMP_API_TOKEN", "s3cret")
    store = InMemorySessionStore()
    app = create_app(_runtime(), store)
    session = store.create()
    with TestClient(app) as c:
        with (
            pytest.raises(WebSocketDisconnect),
            c.websocket_connect(f"/v1/sessions/{session.id}/stream") as ws,
        ):
            ws.receive_json()
        with c.websocket_connect(
            f"/v1/sessions/{session.id}/stream", headers={"authorization": "Bearer s3cret"}
        ) as ws:
            ws.send_json({"text": "hello"})
            assert ws.receive_json()["type"] == "sentence"
