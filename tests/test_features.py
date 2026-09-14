from __future__ import annotations

import json

import pytest

from vmp.cli import main
from vmp.config import load_config
from vmp.features import (
    DEVICE_FEATURES,
    SESSION_FEATURES,
    Feature,
    FeatureView,
    InMemoryOnlineStore,
    JsonlOfflineStore,
    JsonlSource,
    KafkaSource,
    ListSource,
    ParquetOfflineStore,
    RedisOnlineStore,
    SessionFeatureUpdater,
    feast_source,
    get_view,
    latest_per_entity,
    list_views,
    materialize,
    point_in_time_join,
    register_view,
    unregister_view,
    views_from_config,
    write_feast_repo,
)
from vmp.features.offline import OfflineStore
from vmp.features.online import OnlineStore
from vmp.types import FeatureRow

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _view(ttl_s: float = 100.0) -> FeatureView:
    return FeatureView(
        name="v", entity="e", ttl_s=ttl_s, features=(Feature("x", "float"), Feature("s", "str"))
    )


def _rows() -> list[FeatureRow]:
    return [
        FeatureRow("a", 10.0, {"x": 1.0, "s": "one"}),
        FeatureRow("a", 20.0, {"x": 2.0, "s": "two"}),
        FeatureRow("a", 30.0, {"x": 3.0, "s": "three"}),
        FeatureRow("b", 25.0, {"x": 9.0, "s": "nine"}),
    ]


# --- views ----------------------------------------------------------------------


def test_registry_and_config_parity():
    names = [v.name for v in list_views()]
    assert names == ["device_features", "session_features", "speaker_features"]
    cfg = load_config(ROOT / "configs" / "features.toml")
    for v in views_from_config(cfg, register=False):
        assert v == get_view(v.name)
    assert SESSION_FEATURES.feature_names == (
        "turn_count",
        "avg_user_utterance_s",
        "avg_response_ms",
        "last_intent",
        "accent_profile",
    )
    assert DEVICE_FEATURES.entity == "device"


def test_view_validation_and_roundtrip():
    with pytest.raises(ValueError, match="dtype"):
        Feature("x", "bool")
    with pytest.raises(ValueError, match="ttl_s"):
        FeatureView("v", "e", 0, (Feature("x"),))
    with pytest.raises(ValueError, match="duplicate"):
        FeatureView("v", "e", 1, (Feature("x"), Feature("x")))
    v = _view()
    assert FeatureView.from_dict(json.loads(json.dumps(v.to_dict()))) == v
    with pytest.raises(KeyError, match="unknown feature view"):
        get_view("nope")
    register_view(v)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_view(_view(ttl_s=5))
        register_view(_view(ttl_s=5), replace=True)
        assert get_view("v").ttl_s == 5
    finally:
        assert unregister_view("v")
    assert not unregister_view("v")


# --- point-in-time --------------------------------------------------------------


def test_point_in_time_never_sees_future_rows():
    out = point_in_time_join(_rows(), [("a", 25.0), ("a", 20.0), ("a", 9.9)], ttl_s=None)
    assert out[0].values["x"] == 2.0  # 30.0 exists but is in the future
    assert out[1].values["x"] == 2.0  # equal timestamps are visible
    assert out[2].values == {}  # nothing known yet


def test_point_in_time_respects_ttl_and_feature_names():
    v = _view(ttl_s=5.0)
    reqs = [("a", 24.0), ("a", 26.0), ("zzz", 1.0)]
    out = point_in_time_join(_rows(), reqs, v.ttl_s, v.feature_names)
    assert out[0].values == {"x": 2.0, "s": "two"}
    assert out[1].values == {"x": None, "s": None}  # 20.0 is 6 s old, ttl is 5
    assert out[2].values == {"x": None, "s": None}
    assert [r.event_ts for r in out] == [24.0, 26.0, 1.0]


def test_point_in_time_unsorted_input_and_partial_values():
    rows = list(reversed(_rows()))
    rows.append(FeatureRow("a", 40.0, {"x": 4.0}))
    out = point_in_time_join(rows, [("a", 45.0)], None, ("x", "s"))
    assert out[0].values == {"x": 4.0, "s": None}


def test_latest_per_entity():
    latest = latest_per_entity(_rows())
    assert latest["a"].event_ts == 30.0 and latest["b"].event_ts == 25.0


# --- offline stores -------------------------------------------------------------


def test_jsonl_offline_store(tmp_path):
    v = _view(ttl_s=5.0)
    store = JsonlOfflineStore(tmp_path)
    assert isinstance(store, OfflineStore)
    assert store.read(v) == []
    assert store.write(v, _rows()[:2]) == 2
    assert store.write(v, _rows()[2:]) == 2
    assert len(store.read(v)) == 4
    assert [r.event_ts for r in store.read(v, start_ts=20.0, end_ts=25.0)] == [20.0, 25.0]
    pit = store.point_in_time(v, [("a", 31.0), ("b", 24.0)])
    assert pit[0].values == {"x": 3.0, "s": "three"}
    assert pit[1].values == {"x": None, "s": None}


def test_parquet_offline_store_lazy(tmp_path):
    store = ParquetOfflineStore(tmp_path)
    assert isinstance(store, OfflineStore)
    pytest.importorskip("pyarrow")
    v = _view()
    store.write(v, _rows())
    store.write(v, [FeatureRow("c", 1.0, {"x": 0.5, "s": "half"})])
    assert len(store.read(v)) == 5
    assert store.point_in_time(v, [("c", 2.0)])[0].values == {"x": 0.5, "s": "half"}


# --- online stores --------------------------------------------------------------


def test_in_memory_online_store_ttl():
    clock = {"now": 100.0}
    store = InMemoryOnlineStore(clock=lambda: clock["now"])
    assert isinstance(store, OnlineStore)
    store.put("v", "a", {"x": 1}, ts=100.0, ttl_s=10.0)
    store.put("v", "b", {"x": 2}, ts=100.0, ttl_s=None)
    assert store.get("v", "a") == {"x": 1}
    clock["now"] = 110.0
    assert store.get("v", "a") == {"x": 1}
    clock["now"] = 110.1
    assert store.get("v", "a") is None
    assert store.get("v", "b") == {"x": 2}
    assert store.get("v", "a", now=105.0) == {"x": 1}
    assert store.get_many("v", ["a", "b", "c"]) == {"a": None, "b": {"x": 2}, "c": None}
    assert store.entities("v") == ["a", "b"]
    assert store.delete("v", "b") and not store.delete("v", "b")
    assert store.get("other", "a") is None


def test_redis_online_store_is_lazy():
    pytest.importorskip("redis")
    with pytest.raises(Exception):  # noqa: B017 - no broker in the default run
        RedisOnlineStore("redis://127.0.0.1:1/0").get("v", "a")


# --- materialize ----------------------------------------------------------------


def test_materialize_and_dry_run(tmp_path):
    v = _view(ttl_s=50.0)
    offline = JsonlOfflineStore(tmp_path)
    offline.write(v, _rows())
    online = InMemoryOnlineStore(clock=lambda: 31.0)

    plan = materialize(v, offline, None, dry_run=True)
    assert plan["dry_run"] is True and plan["rows_read"] == 4 and plan["entities"] == 2
    assert plan["entities_written"] == 0
    assert len(online) == 0

    m = materialize(v, offline, online, start_ts=15.0, end_ts=25.0)
    assert m["rows_read"] == 2 and m["entities_written"] == 2
    assert online.get("v", "a") == {"x": 2.0, "s": "two"}  # 30.0 is outside the window
    assert online.timestamp("v", "a") == 20.0
    assert online.get("v", "a", now=71.0) is None  # ttl 50 from row ts 20

    with pytest.raises(ValueError, match="online store is required"):
        materialize(v, offline, None)
    with pytest.raises(ValueError, match="before"):
        materialize(v, offline, online, start_ts=5.0, end_ts=1.0)


# --- feast export ---------------------------------------------------------------


def test_feast_source_text(tmp_path):
    text = feast_source()
    assert "from feast import Entity, FeatureView, Field, FileSource" in text
    assert 'session = Entity(name="session", join_keys=["session_id"])' in text
    assert 'name="session_features"' in text
    assert "ttl=timedelta(seconds=3600)" in text
    assert 'Field(name="turn_count", dtype=Int64)' in text
    assert 'Field(name="accent_profile", dtype=String)' in text
    assert 'Field(name="wer_7d", dtype=Float32)' in text
    assert 'path="data/device_features.parquet"' in text
    compile(text, "features.py", "exec")

    m = write_feast_repo(tmp_path, dry_run=True)
    assert not (tmp_path / "feature_repo" / "features.py").exists()
    m = write_feast_repo(tmp_path, [SESSION_FEATURES], data_root="lake")
    assert m["views"] == ["session_features"] and m["entities"] == ["session"]
    written = (tmp_path / "feature_repo" / "features.py").read_text()
    assert 'path="lake/session_features.parquet"' in written


# --- stream ---------------------------------------------------------------------


def _turn(session: str, turn: int, t0: float, listen_ms: float, resp_ms: float, intent: str):
    base = {"session": session, "turn": turn, "span": "x", "seq": 0, "payload": {}}
    return [
        {**base, "ts": t0, "event": "listen.start", "ms": None},
        {**base, "ts": t0 + 1, "event": "listen.end", "ms": listen_ms},
        {**base, "ts": t0 + 2, "event": "stt.end", "ms": 200, "payload": {"accent": "en-IE"}},
        {**base, "ts": t0 + 3, "event": "llm.end", "ms": 300, "payload": {"intent": intent}},
        {**base, "ts": t0 + 4, "event": "playback.end", "payload": {"response_ms": resp_ms}},
        {**base, "ts": t0 + 5, "event": "turn.end", "ms": 900},
    ]


def test_stream_updater_math_and_online_push():
    online = InMemoryOnlineStore(clock=lambda: 1000.0)
    up = SessionFeatureUpdater(online=online)
    events = (
        _turn("s1", 1, 0.0, 1000.0, 400.0, "weather")
        + _turn("s2", 1, 10.0, 3000.0, 800.0, "timer")
        + _turn("s1", 2, 20.0, 2000.0, 600.0, "music")
    )
    out = up.consume(events[0])
    assert out is None
    summary = up.run(ListSource(events[1:]))
    assert summary["sessions"] == 2 and summary["turns"] == 3 and summary["rows"] == 3
    assert summary["events_seen"] == 18 and summary["events_used"] == 15

    f = up.features("s1")
    assert f == {
        "turn_count": 2,
        "avg_user_utterance_s": pytest.approx(1.5),
        "avg_response_ms": pytest.approx(500.0),
        "last_intent": "music",
        "accent_profile": "en-IE",
    }
    assert up.features("s2")["avg_user_utterance_s"] == pytest.approx(3.0)
    assert online.get("session_features", "s1") == f
    assert online.timestamp("session_features", "s1") == 25.0
    assert [r.values["turn_count"] for r in up.rows if r.entity_id == "s1"] == [1, 2]
    assert up.features("missing") is None


def test_stream_updater_ignores_malformed_and_writes_nothing_without_online(tmp_path):
    up = SessionFeatureUpdater()
    assert up.consume({"event": "turn.end"}) is None
    assert up.consume({"session": "s", "event": None}) is None
    assert up.consume({"session": "s", "event": "turn.end"}) == {
        "turn_count": 1,
        "avg_user_utterance_s": None,
        "avg_response_ms": None,
        "last_intent": None,
        "accent_profile": None,
    }
    p = tmp_path / "trace.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in _turn("j", 1, 0.0, 500.0, 100.0, "q")) + "\n")
    up2 = SessionFeatureUpdater()
    assert up2.run(JsonlSource(p))["turns"] == 1


def test_stream_rows_feed_point_in_time(tmp_path):
    up = SessionFeatureUpdater()
    up.run(_turn("s1", 1, 0.0, 1000.0, 400.0, "a") + _turn("s1", 2, 100.0, 3000.0, 800.0, "b"))
    store = JsonlOfflineStore(tmp_path)
    store.write(SESSION_FEATURES, up.rows)
    pit = store.point_in_time(SESSION_FEATURES, [("s1", 50.0), ("s1", 200.0)])
    assert pit[0].values["turn_count"] == 1 and pit[0].values["last_intent"] == "a"
    assert pit[1].values["turn_count"] == 2 and pit[1].values["last_intent"] == "b"


def test_kafka_source_is_lazy():
    try:
        KafkaSource("t", bootstrap_servers="127.0.0.1:1", max_messages=0)
    except ModuleNotFoundError as e:
        assert "confluent" in str(e)


# --- cli ------------------------------------------------------------------------


def test_cli_materialize_and_feast_export(tmp_path, capsys):
    offline = JsonlOfflineStore(tmp_path / "off")
    offline.write(SESSION_FEATURES, [FeatureRow("s1", 1.0, {"turn_count": 3})])
    cfg = str(ROOT / "configs" / "features.toml")
    argv = ["features", "materialize", "--view", "session_features", "--offline-root",
            str(tmp_path / "off"), "--config", cfg]
    assert main([*argv, "--dry-run"]) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["dry_run"] and m["rows_read"] == 1 and m["online_backend"] == "none"
    assert main(argv) == 0
    m = json.loads(capsys.readouterr().out)
    assert m["entities_written"] == 1 and m["online_backend"] == "memory"

    assert main(["features", "feast-export", "--out", str(tmp_path), "--config", cfg]) == 0
    m = json.loads(capsys.readouterr().out)
    assert set(m["views"]) == {"session_features", "speaker_features", "device_features"}
    assert (tmp_path / "feature_repo" / "features.py").exists()
