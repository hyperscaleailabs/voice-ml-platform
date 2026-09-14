"""Observability tests: tracer schema, metrics exposition, drift, SLOs and alert rules."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from vmp.observability.drift import DriftDetector, DriftReport, ks_statistic, psi
from vmp.observability.metrics import (
    DEFAULT_MS_BUCKETS,
    METRIC_STAGE_MS,
    METRIC_TTFA_MS,
    METRIC_TURN_TOTAL,
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    default_registry,
    install_standard_metrics,
    render_prometheus,
)
from vmp.observability.slo import (
    DEFAULT_SLOS,
    SLI,
    SLO,
    burn_rate,
    compute_sli,
    error_budget,
    evaluate_slos,
    percentile,
    slos_from_config,
    to_prometheus_rules_yaml,
)
from vmp.observability.trace import (
    TRACE_KEYS,
    JsonlSink,
    ListSink,
    Tracer,
    format_summary,
    group_by_turn,
    read_trace,
    summarise_turn,
    time_to_first_audio,
)

ROOT = Path(__file__).resolve().parents[1]


def _traced_turn(tracer: Tracer, session: str = "s1", turn: int = 1, ttfa: float = 900.0) -> None:
    """One turn's worth of spans, shaped like a real voice turn."""
    with tracer.span("turn", session, turn):
        with tracer.span("stt", session, turn) as payload:
            payload["chars"] = 12
        with tracer.span("llm", session, turn) as payload:
            payload["ttft_ms"] = 190.0
        for seq in (1, 2):
            with tracer.span("segment.emit", session, turn, seq=seq):
                pass
            with tracer.span("tts", session, turn, seq=seq):
                pass
            with tracer.span("playback", session, turn, seq=seq) as payload:
                if seq == 1:
                    payload["response_ms"] = ttfa


# ===========================================================================
# tracer
# ===========================================================================


def test_row_schema():
    sink = ListSink()
    tracer = Tracer(sink)
    with tracer.span("stt", "s1", 3, seq=2) as payload:
        payload["chars"] = 7

    start, end = sink.rows
    assert list(start) == list(TRACE_KEYS)
    assert start["event"] == "stt.start"
    assert start["ms"] is None
    assert start["payload"] == {}
    assert end["event"] == "stt.end"
    assert end["session"] == "s1"
    assert end["turn"] == 3
    assert end["seq"] == 2
    assert end["ms"] >= 0.0
    assert end["payload"] == {"chars": 7}
    assert start["span"] == end["span"]
    assert start["ts"] > 0
    assert json.loads(json.dumps(end)) == end


def test_span_ids_are_unique_and_ordered():
    tracer = Tracer(ListSink())
    ids = [tracer.new_span_id() for _ in range(5)]
    assert len(set(ids)) == 5
    prefixes = {i.split("-")[0] for i in ids}
    assert len(prefixes) == 1
    assert [i.split("-")[1] for i in ids] == [f"{n:06d}" for n in range(1, 6)]
    # A second tracer gets its own prefix.
    assert Tracer(ListSink()).new_span_id().split("-")[0] not in prefixes


def test_nested_spans_record_their_parent():
    sink = ListSink()
    tracer = Tracer(sink)
    with tracer.span("turn", "s1", 1):
        with tracer.span("llm", "s1", 1), tracer.span("segment.emit", "s1", 1, seq=1):
            pass
        tracer.event("note", "s1", 1, payload={"detail": "after the llm"})

    by_event = {row["event"]: row for row in sink.rows}
    assert "parent" not in by_event["turn.end"]["payload"]
    assert by_event["llm.start"]["payload"]["parent"] == by_event["turn.start"]["span"]
    assert by_event["segment.emit.end"]["payload"]["parent"] == by_event["llm.end"]["span"]
    # The context is restored when a span exits.
    assert by_event["note"]["payload"]["parent"] == by_event["turn.end"]["span"]


def test_span_records_an_error_and_reraises():
    sink = ListSink()
    tracer = Tracer(sink)
    with pytest.raises(ValueError), tracer.span("llm", "s1", 1):
        raise ValueError("boom")
    assert sink.rows[-1]["event"] == "llm.end"
    assert sink.rows[-1]["payload"]["error"] == "ValueError"


def test_spans_are_measured_with_an_injectable_clock():
    sink = ListSink()
    tracer = Tracer(sink, clock=lambda: 1234.5)
    with tracer.span("stt", "s1", 1):
        pass
    assert all(row["ts"] == 1234.5 for row in sink.rows)


def test_jsonl_sink_round_trips(tmp_path: Path):
    path = tmp_path / "nested" / "trace.jsonl"
    tracer = Tracer(JsonlSink(path, fsync=True))
    _traced_turn(tracer)

    rows = list(read_trace(path))
    assert rows[0]["event"] == "turn.start"
    assert rows[-1]["event"] == "turn.end"
    assert len(rows) == 18  # turn, stt, llm + two sentences x (segment, tts, playback)
    assert time_to_first_audio(rows) == 900.0

    with pytest.raises(FileNotFoundError):
        list(read_trace(tmp_path / "missing.jsonl"))


def test_list_sink_clear():
    sink = ListSink()
    Tracer(sink).event("note", "s", 1)
    assert len(sink.rows) == 1
    sink.clear()
    assert sink.rows == []


# -- summaries --------------------------------------------------------------


def test_summarise_turn_aggregates_per_stage():
    rows = [
        {"event": "stt.start", "ms": None},
        {"event": "stt.end", "ms": 100.0},
        {"event": "tts.end", "ms": 40.0},
        {"event": "tts.end", "ms": 60.0},
        {"event": "tts.end", "ms": 20.0},
        {"event": "turn.end", "ms": 500.0},
    ]
    summary = summarise_turn(rows)
    assert list(summary) == ["stt", "tts", "turn"]  # order of first appearance
    assert summary["stt"] == {"count": 1, "total_ms": 100.0, "first_ms": 100.0, "max_ms": 100.0}
    assert summary["tts"] == {"count": 3, "total_ms": 120.0, "first_ms": 40.0, "max_ms": 60.0}
    assert summarise_turn([]) == {}


def test_format_summary_reads_like_alpha_core():
    summary = summarise_turn(
        [
            {"event": "stt.end", "ms": 233.0},
            {"event": "llm.end", "ms": 192.0},
            {"event": "tts.end", "ms": 40.0},
            {"event": "tts.end", "ms": 60.0},
        ]
    )
    text = format_summary(summary)
    assert text == "stt 1x 233ms | llm 1x 192ms | tts 2x 100ms (first 40, max 60)"
    assert format_summary({}) == ""


def test_time_to_first_audio_takes_the_first_playback_only():
    sink = ListSink()
    tracer = Tracer(sink)
    _traced_turn(tracer, ttfa=1234.5)
    assert time_to_first_audio(sink.rows) == 1234.5

    playbacks = [r for r in sink.rows if r["event"] == "playback.end"]
    assert len(playbacks) == 2
    assert "response_ms" not in playbacks[1]["payload"]
    assert time_to_first_audio([]) is None
    assert time_to_first_audio([{"event": "playback.end", "payload": {}}]) is None


def test_group_by_turn_keeps_order():
    sink = ListSink()
    tracer = Tracer(sink)
    _traced_turn(tracer, "s1", 1, ttfa=900.0)
    _traced_turn(tracer, "s1", 2, ttfa=1100.0)
    _traced_turn(tracer, "s2", 1, ttfa=700.0)

    groups = group_by_turn(sink.rows)
    assert list(groups) == [("s1", 1), ("s1", 2), ("s2", 1)]
    assert [time_to_first_audio(rows) for rows in groups.values()] == [900.0, 1100.0, 700.0]
    assert all(rows[0]["event"] == "turn.start" for rows in groups.values())


# ===========================================================================
# metrics
# ===========================================================================


def test_counter_gauge_and_label_validation():
    counter = Counter("vmp_test_total", "test counter", ("stage",))
    counter.inc(labels={"stage": "llm"})
    counter.inc(2.0, labels={"stage": "llm"})
    counter.inc(labels={"stage": "tts"})
    assert counter.value({"stage": "llm"}) == 3.0
    assert counter.value({"stage": "tts"}) == 1.0

    with pytest.raises(ValueError, match="only increase"):
        counter.inc(-1.0, labels={"stage": "llm"})
    with pytest.raises(ValueError, match="expected labels"):
        counter.inc(labels={"wrong": "x"})
    with pytest.raises(ValueError):
        counter.inc()

    gauge = Gauge("vmp_test_gauge", "test gauge")
    gauge.set(5.0)
    gauge.inc(2.0)
    gauge.dec(3.0)
    assert gauge.value() == 4.0


def test_histogram_buckets_are_cumulative():
    histogram = Histogram("vmp_test_ms", "test histogram", buckets=(10.0, 100.0, 1000.0))
    for value in (5.0, 50.0, 500.0, 5000.0):
        histogram.observe(value)
    assert histogram.count() == 4
    assert histogram.sum() == 5555.0

    lines = histogram.render()
    assert lines[0] == "# HELP vmp_test_ms test histogram"
    assert lines[1] == "# TYPE vmp_test_ms histogram"
    assert 'vmp_test_ms_bucket{le="10"} 1' in lines
    assert 'vmp_test_ms_bucket{le="100"} 2' in lines
    assert 'vmp_test_ms_bucket{le="1000"} 3' in lines
    assert 'vmp_test_ms_bucket{le="+Inf"} 4' in lines
    assert "vmp_test_ms_count 4" in lines
    assert "vmp_test_ms_sum 5555" in lines

    with pytest.raises(ValueError, match="at least one bucket"):
        Histogram("vmp_empty", "no buckets", buckets=())


def test_registry_render_is_prometheus_exposition_format():
    registry = MetricsRegistry()
    registry.counter("vmp_b_total", "counter b", ("stage",)).inc(labels={"stage": "llm"})
    registry.gauge("vmp_a_ratio", "gauge a").set(0.25)
    registry.histogram("vmp_c_ms", "histogram c", buckets=(10.0,)).observe(5.0)

    text = render_prometheus(registry)
    lines = text.splitlines()
    assert text.endswith("\n")
    # Metrics are rendered in name order, each with HELP and TYPE first.
    assert lines[0] == "# HELP vmp_a_ratio gauge a"
    assert lines[1] == "# TYPE vmp_a_ratio gauge"
    assert lines[2] == "vmp_a_ratio 0.25"
    assert "# TYPE vmp_b_total counter" in lines
    assert 'vmp_b_total{stage="llm"} 1' in lines
    assert registry.names() == ["vmp_a_ratio", "vmp_b_total", "vmp_c_ms"]
    assert MetricsRegistry().render_prometheus() == ""


def test_registry_escapes_label_values():
    registry = MetricsRegistry()
    registry.counter("vmp_x_total", "x", ("path",)).inc(labels={"path": 'a"b\\c'})
    assert 'vmp_x_total{path="a\\"b\\\\c"} 1' in registry.render_prometheus()


def test_registry_rejects_a_type_change():
    registry = MetricsRegistry()
    registry.counter("vmp_dup", "a counter")
    with pytest.raises(TypeError, match="not a gauge"):
        registry.gauge("vmp_dup", "a gauge")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Counter("vmp_dup", "again"))


def test_standard_metrics_are_installed_and_idempotent():
    registry = install_standard_metrics(MetricsRegistry())
    install_standard_metrics(registry)
    assert METRIC_TURN_TOTAL in registry
    assert METRIC_STAGE_MS in registry
    assert isinstance(registry.get(METRIC_TTFA_MS), Histogram)
    assert registry.get(METRIC_TTFA_MS).buckets == DEFAULT_MS_BUCKETS

    registry.counter(METRIC_TURN_TOTAL, "", ("status",)).inc(labels={"status": "ok"})
    registry.histogram(METRIC_STAGE_MS, "", ("stage",)).observe(233.0, {"stage": "stt"})
    text = registry.render_prometheus()
    assert 'vmp_turn_total{status="ok"} 1' in text
    assert 'vmp_stage_ms_bucket{stage="stt",le="250"} 1' in text

    assert default_registry() is default_registry()
    assert METRIC_TURN_TOTAL in default_registry()


# ===========================================================================
# drift
# ===========================================================================


def test_psi_of_an_identical_distribution_is_zero():
    sample = [float(i % 10) for i in range(200)]
    assert psi(sample, list(sample)) == 0.0
    assert psi(sample, list(reversed(sample))) == 0.0  # order does not matter


def test_psi_grows_with_the_shift():
    reference = [float(i) / 100 for i in range(100)]
    small = [v + 0.05 for v in reference]
    large = [v + 2.0 for v in reference]
    assert psi(reference, small) < psi(reference, large)
    assert psi(reference, large) > 0.2  # "shifted" by the usual reading


def test_psi_on_a_known_two_bin_split():
    """Half the mass moves from one bin to the other: 0.4*ln(0.4/0.5)*2 by symmetry."""
    reference = [0.0] * 50 + [1.0] * 50
    current = [0.0] * 40 + [1.0] * 60
    expected = (0.4 - 0.5) * math.log(0.4 / 0.5) + (0.6 - 0.5) * math.log(0.6 / 0.5)
    assert psi(reference, current, bins=2) == pytest.approx(expected)


def test_psi_rejects_empty_samples_and_bad_bins():
    with pytest.raises(ValueError, match="non-empty"):
        psi([], [1.0])
    with pytest.raises(ValueError, match="bins"):
        psi([1.0], [1.0], bins=0)


def test_ks_statistic_on_known_distributions():
    assert ks_statistic([1, 2, 3, 4], [1, 2, 3, 4]) == 0.0
    # Disjoint supports: the CDFs are maximally apart.
    assert ks_statistic([0, 0, 0, 0], [1, 1, 1, 1]) == 1.0
    # Half of the second sample shifted past every value of the first.
    assert ks_statistic([0, 0, 0, 0], [0, 0, 1, 1]) == pytest.approx(0.5)
    assert ks_statistic([0.0, 1.0], [0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="non-empty"):
        ks_statistic([], [1.0])


def test_drift_detector_reports_a_stable_series():
    reference = [float(i % 10) for i in range(100)]
    report = DriftDetector().check(reference, list(reference), name="ttfa_ms")

    assert isinstance(report, DriftReport)
    assert report.drifted is False
    assert report.reasons == []
    assert report.psi == 0.0
    assert report.ks == 0.0
    assert report.name == "ttfa_ms"
    assert report.n_reference == report.n_current == 100
    assert report.mean_shift == 0.0
    assert json.loads(json.dumps(report.to_dict()))["drifted"] is False


def test_drift_detector_flags_a_shifted_series():
    reference = [float(i) for i in range(100)]
    current = [v + 50.0 for v in reference]
    report = DriftDetector().check(reference, current, name="wer")

    assert report.drifted is True
    assert any("psi" in reason for reason in report.reasons)
    assert any("ks" in reason for reason in report.reasons)
    assert report.reasons[0].startswith("wer: psi")
    assert report.mean_shift == pytest.approx(50.0)
    assert report.psi_threshold == 0.2
    assert report.ks_threshold == 0.1


def test_drift_detector_thresholds_are_configurable():
    reference = [float(i) for i in range(100)]
    current = [v + 5.0 for v in reference]
    detector = DriftDetector(psi_threshold=10.0, ks_threshold=1.0)
    assert detector.check(reference, current).drifted is False


def test_check_many_only_compares_shared_features():
    reference = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}
    current = {"a": [1.0, 2.0, 3.0], "c": [9.0]}
    reports = DriftDetector().check_many(reference, current)
    assert list(reports) == ["a"]
    assert reports["a"].drifted is False


# ===========================================================================
# SLO / SLI
# ===========================================================================


def test_percentile_interpolates():
    assert percentile([10.0], 95) == 10.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 0) == 1.0
    assert percentile([1, 2, 3, 4], 100) == 4.0
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_compute_sli_from_a_trace():
    sink = ListSink()
    tracer = Tracer(sink)
    for turn, ttfa in enumerate([900.0, 1100.0, 2000.0, 1500.0], 1):
        _traced_turn(tracer, "s1", turn, ttfa=ttfa)

    sli = compute_sli(sink.rows, wer=0.04)
    assert sli.n_turns == 4
    assert sli.values["ttfa_p50_ms"] == pytest.approx(1300.0)
    assert sli.values["ttfa_p95_ms"] == pytest.approx(1925.0)
    assert sli.values["error_rate"] == 0.0
    assert sli.values["wer"] == 0.04
    assert sli.sources["ttfa_p95_ms"].startswith("trace:")
    assert sli.sources["wer"] == "eval: golden set"
    assert sli.get("missing") is None
    assert json.loads(json.dumps(sli.to_dict()))["n_turns"] == 4


def test_compute_sli_counts_errored_and_truncated_turns():
    sink = ListSink()
    tracer = Tracer(sink)
    _traced_turn(tracer, "s1", 1, ttfa=900.0)
    with (
        pytest.raises(RuntimeError),
        tracer.span("turn", "s1", 2),
        tracer.span("llm", "s1", 2),
    ):
        raise RuntimeError("llm down")
    # A turn that started and never ended (the process died mid-turn).
    tracer._row("turn.start", "s1", 3, "span-x", None, None, {})

    sli = compute_sli(sink.rows)
    assert sli.n_turns == 3
    assert sli.values["error_rate"] == pytest.approx(2 / 3)


def test_compute_sli_of_an_empty_trace():
    sli = compute_sli([])
    assert sli.n_turns == 0
    assert sli.values == {}


def test_slo_met_and_burn_rate():
    latency = SLO("ttfa_p95", "ttfa_p95_ms", 3000.0)
    assert latency.met(2500.0) is True
    assert latency.met(3500.0) is False
    assert burn_rate(latency, 3000.0) == 1.0
    assert burn_rate(latency, 6000.0) == 2.0
    assert burn_rate(latency, 0.0) == 0.0

    availability = SLO("availability", "success_rate", 0.99, comparator=">=")
    assert availability.met(0.995) is True
    assert burn_rate(availability, 0.99) == pytest.approx(1.0)
    assert burn_rate(availability, 0.98) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="comparator"):
        burn_rate(SLO("x", "x", 1.0, comparator="~="), 1.0)
    assert burn_rate(SLO("zero", "errors", 0.0), 1.0) == math.inf


def test_error_budget_report():
    sli = SLI(values={"ttfa_p95_ms": 1500.0}, n_turns=10)
    slo = SLO("ttfa_p95", "ttfa_p95_ms", 3000.0)

    budget = error_budget(slo, sli)
    assert budget["measured"] == 1500.0
    assert budget["burn_rate"] == 0.5
    assert budget["consumed"] == 0.5
    assert budget["remaining"] == 0.5
    assert budget["met"] is True

    half_window = error_budget(slo, sli, elapsed_fraction=0.5)
    assert half_window["consumed"] == 0.25
    assert half_window["remaining"] == 0.75

    overspent = error_budget(slo, 6000.0)
    assert overspent["met"] is False
    assert overspent["remaining"] == -1.0


def test_error_budget_for_an_unmeasured_indicator():
    budget = error_budget(DEFAULT_SLOS[2], SLI(values={"ttfa_p95_ms": 1.0}))
    assert budget["measured"] is None
    assert budget["met"] is None
    assert budget["note"] == "indicator not measured"


def test_evaluate_slos_over_the_defaults():
    sli = SLI(values={"ttfa_p95_ms": 1200.0, "error_rate": 0.0, "wer": 0.03}, n_turns=20)
    budgets = evaluate_slos(DEFAULT_SLOS, sli)
    assert [b["slo"] for b in budgets] == ["ttfa_p95", "error_rate", "wer"]
    assert all(b["met"] for b in budgets)


def test_slos_from_the_repository_config():
    from vmp.config import load_config

    slos = slos_from_config(load_config(ROOT / "configs" / "slo.toml"))
    assert [s.name for s in slos] == [s.name for s in DEFAULT_SLOS]
    assert [s.to_dict() for s in slos] == [s.to_dict() for s in DEFAULT_SLOS]
    assert slos_from_config({}) == []


# -- alerting rules ---------------------------------------------------------


def test_prometheus_rules_contain_an_alert_per_slo():
    text = to_prometheus_rules_yaml(DEFAULT_SLOS)
    lines = text.splitlines()

    assert lines[0] == "groups:"
    assert lines[1] == "  - name: vmp-slo"
    assert lines[2] == "    rules:"
    for alert in ("VmpSloTtfaP95", "VmpSloErrorRate", "VmpSloWer"):
        assert f"      - alert: {alert}" in lines
    assert text.count("- alert:") == len(DEFAULT_SLOS)
    assert text.count("severity: page") == len(DEFAULT_SLOS)
    assert "          slo: ttfa_p95" in lines
    assert text.endswith("\n")


def test_prometheus_rules_expressions_and_windows():
    text = to_prometheus_rules_yaml(DEFAULT_SLOS, rate_window="1m", for_duration="5m")
    assert "histogram_quantile(0.95, sum(rate(vmp_ttfa_ms_bucket[1m])) by (le))) > 3000.0" in text
    assert "sum(rate(vmp_errors_total[1m])) / sum(rate(vmp_turn_total[1m]))) > 0.01" in text
    assert "        for: 5m" in text.splitlines()
    assert 'summary: "ttfa_p95_ms outside objective (<= 3000.0, window 30d)"' in text
    assert 'description: "target: golden-set WER at or below 5%"' in text


def test_prometheus_rules_quote_promql_label_matchers():
    """The WER expression carries `"` inside a double-quoted YAML scalar."""
    line = next(
        ln for ln in to_prometheus_rules_yaml(DEFAULT_SLOS).splitlines() if "vmp_wer" in ln
    )
    assert line == '        expr: "(max(vmp_wer{set=\\"golden\\"})) > 0.05"'
    scalar = line.split("expr: ", 1)[1]
    assert scalar.startswith('"') and scalar.endswith('"')
    assert json.loads(scalar) == '(max(vmp_wer{set="golden"})) > 0.05'


def test_prometheus_rules_flip_the_comparison_for_lower_bounds():
    text = to_prometheus_rules_yaml(
        [SLO("availability", "success_rate", 0.99, comparator=">=")], group="custom"
    )
    assert "  - name: custom" in text.splitlines()
    assert "        expr: \"(success_rate) < 0.99\"" in text.splitlines()
    assert "      - alert: VmpSloAvailability" in text.splitlines()
    assert "description" not in text  # no description was given
