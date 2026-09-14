"""Observability: stage tracer, metrics registry, drift detectors, SLOs.

Standard library only. OpenTelemetry and prometheus_client are optional adapters
imported lazily inside their classes.
"""

from __future__ import annotations

from vmp.observability.drift import DriftDetector, DriftReport, ks_statistic, psi
from vmp.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    PrometheusClientAdapter,
    default_registry,
    render_prometheus,
)
from vmp.observability.slo import (
    DEFAULT_SLOS,
    SLI,
    SLO,
    burn_rate,
    compute_sli,
    error_budget,
    to_prometheus_rules_yaml,
)
from vmp.observability.trace import (
    JsonlSink,
    ListSink,
    OtlpSink,
    Tracer,
    format_summary,
    read_trace,
    summarise_turn,
    time_to_first_audio,
)

__all__ = [
    "DEFAULT_SLOS",
    "SLI",
    "SLO",
    "Counter",
    "DriftDetector",
    "DriftReport",
    "Gauge",
    "Histogram",
    "JsonlSink",
    "ListSink",
    "MetricsRegistry",
    "OtlpSink",
    "PrometheusClientAdapter",
    "Tracer",
    "burn_rate",
    "compute_sli",
    "default_registry",
    "error_budget",
    "format_summary",
    "ks_statistic",
    "psi",
    "read_trace",
    "render_prometheus",
    "summarise_turn",
    "time_to_first_audio",
    "to_prometheus_rules_yaml",
]
