"""In-process metrics registry with Prometheus text exposition.

Counters, gauges and fixed-bucket histograms, keyed by label tuples. The
reference implementation is standard library only; `PrometheusClientAdapter`
mirrors writes into `prometheus_client` when that package is installed.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Iterable, Sequence
from typing import Any

LabelKey = tuple[tuple[str, str], ...]

# Standard metric names used across the platform.
METRIC_TURN_TOTAL = "vmp_turn_total"
METRIC_STAGE_MS = "vmp_stage_ms"
METRIC_TTFA_MS = "vmp_ttfa_ms"
METRIC_ERRORS_TOTAL = "vmp_errors_total"
METRIC_WER = "vmp_wer"
METRIC_LLM_TTFT_MS = "vmp_llm_ttft_ms"
METRIC_DRIFT_PSI = "vmp_drift_psi"
METRIC_EDGE_VERIFY_FAILED_TOTAL = "vmp_edge_bundle_verify_failed_total"
METRIC_EDGE_BUNDLE_INFO = "vmp_edge_bundle_info"

DEFAULT_MS_BUCKETS: tuple[float, ...] = (
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    3000.0,
    5000.0,
    10000.0,
)


def _label_key(labels: dict[str, str] | None) -> LabelKey:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _fmt_labels(key: LabelKey, extra: dict[str, str] | None = None) -> str:
    items = list(key)
    if extra:
        items.extend(sorted(extra.items()))
    if not items:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in items)
    return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_num(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = ()) -> None:
        self.name = name
        self.help = help
        self.labelnames = tuple(labelnames)
        self._lock = threading.Lock()

    def _check(self, labels: dict[str, str] | None) -> LabelKey:
        given = set(labels or {})
        if given != set(self.labelnames):
            raise ValueError(
                f"{self.name}: expected labels {sorted(self.labelnames)}, got {sorted(given)}"
            )
        return _label_key(labels)

    def render(self) -> list[str]:  # pragma: no cover - overridden
        return []


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help, labelnames)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("counters only increase")
        key = self._check(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(self._check(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, v in sorted(self._values.items()):
            lines.append(f"{self.name}{_fmt_labels(key)} {_fmt_num(v)}")
        return lines


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help: str, labelnames: Sequence[str] = ()) -> None:
        super().__init__(name, help, labelnames)
        self._values: dict[LabelKey, float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._check(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._check(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        self.inc(-amount, labels)

    def value(self, labels: dict[str, str] | None = None) -> float:
        return self._values.get(self._check(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        for key, v in sorted(self._values.items()):
            lines.append(f"{self.name}{_fmt_labels(key)} {_fmt_num(v)}")
        return lines


class Histogram(_Metric):
    """Cumulative fixed-bucket histogram (`le` buckets, `_sum`, `_count`)."""

    kind = "histogram"

    def __init__(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
        buckets: Iterable[float] = DEFAULT_MS_BUCKETS,
    ) -> None:
        super().__init__(name, help, labelnames)
        bs = sorted(float(b) for b in buckets)
        if not bs:
            raise ValueError("histogram needs at least one bucket")
        self.buckets: tuple[float, ...] = tuple(bs)
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = {}
        self._totals: dict[LabelKey, int] = {}

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._check(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * len(self.buckets))
            for i, b in enumerate(self.buckets):
                if value <= b:
                    counts[i] += 1
            self._sums[key] = self._sums.get(key, 0.0) + float(value)
            self._totals[key] = self._totals.get(key, 0) + 1

    def count(self, labels: dict[str, str] | None = None) -> int:
        return self._totals.get(self._check(labels), 0)

    def sum(self, labels: dict[str, str] | None = None) -> float:
        return self._sums.get(self._check(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key in sorted(self._counts):
            counts = self._counts[key]
            for b, c in zip(self.buckets, counts, strict=True):
                lines.append(f"{self.name}_bucket{_fmt_labels(key, {'le': _fmt_num(b)})} {c}")
            total = self._totals[key]
            lines.append(f"{self.name}_bucket{_fmt_labels(key, {'le': '+Inf'})} {total}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {_fmt_num(self._sums[key])}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {total}")
        return lines


class MetricsRegistry:
    """Holds metrics by name and renders them in Prometheus text format."""

    def __init__(self) -> None:
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def register(self, metric: _Metric) -> _Metric:
        with self._lock:
            if metric.name in self._metrics:
                raise ValueError(f"metric already registered: {metric.name}")
            self._metrics[metric.name] = metric
        return metric

    def get(self, name: str) -> _Metric:
        return self._metrics[name]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def counter(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Counter:
        m = self._metrics.get(name)
        if m is None:
            m = self.register(Counter(name, help, labelnames))
        if not isinstance(m, Counter):
            raise TypeError(f"{name} is a {m.kind}, not a counter")
        return m

    def gauge(self, name: str, help: str, labelnames: Sequence[str] = ()) -> Gauge:
        m = self._metrics.get(name)
        if m is None:
            m = self.register(Gauge(name, help, labelnames))
        if not isinstance(m, Gauge):
            raise TypeError(f"{name} is a {m.kind}, not a gauge")
        return m

    def histogram(
        self,
        name: str,
        help: str,
        labelnames: Sequence[str] = (),
        buckets: Iterable[float] = DEFAULT_MS_BUCKETS,
    ) -> Histogram:
        m = self._metrics.get(name)
        if m is None:
            m = self.register(Histogram(name, help, labelnames, buckets))
        if not isinstance(m, Histogram):
            raise TypeError(f"{name} is a {m.kind}, not a histogram")
        return m

    def render_prometheus(self) -> str:
        lines: list[str] = []
        for name in sorted(self._metrics):
            lines.extend(self._metrics[name].render())
        return "\n".join(lines) + ("\n" if lines else "")

    def names(self) -> list[str]:
        return sorted(self._metrics)


def install_standard_metrics(registry: MetricsRegistry) -> MetricsRegistry:
    """Create the platform's standard metrics on `registry` (idempotent)."""
    registry.counter(METRIC_TURN_TOTAL, "Voice turns completed", ("status",))
    registry.histogram(METRIC_STAGE_MS, "Per-stage latency in milliseconds", ("stage",))
    registry.histogram(METRIC_TTFA_MS, "Time to first audio in milliseconds")
    registry.counter(METRIC_ERRORS_TOTAL, "Errors by stage", ("stage",))
    registry.gauge(METRIC_WER, "Word error rate of the latest evaluation", ("set",))
    registry.histogram(METRIC_LLM_TTFT_MS, "LLM time to first token in milliseconds")
    registry.gauge(METRIC_DRIFT_PSI, "Population stability index per feature", ("feature",))
    registry.counter(
        METRIC_EDGE_VERIFY_FAILED_TOTAL, "Edge bundle verification failures", ("device",)
    )
    registry.gauge(METRIC_EDGE_BUNDLE_INFO, "Edge bundle in use (value 1)", ("version",))
    return registry


_DEFAULT: MetricsRegistry | None = None


def default_registry() -> MetricsRegistry:
    """Process-wide registry with the standard metrics installed."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = install_standard_metrics(MetricsRegistry())
    return _DEFAULT


def render_prometheus(registry: MetricsRegistry | None = None) -> str:
    return (registry or default_registry()).render_prometheus()


class PrometheusClientAdapter:
    """Mirrors a `MetricsRegistry` into `prometheus_client` objects (lazy import).

    `sync()` copies the current values: counters and gauges are set to the
    in-process value, histograms replay their sums and counts as bucket
    observations are not replayable exactly, so the adapter re-observes the
    midpoint of each bucket for the delta since the last sync.
    """

    def __init__(self, registry: MetricsRegistry, client_registry: Any = None) -> None:
        self.registry = registry
        self._client_registry = client_registry
        self._objects: dict[str, Any] = {}
        self._seen: dict[tuple[str, LabelKey], list[int]] = {}

    def _client(self) -> Any:
        import prometheus_client  # lazy

        return prometheus_client

    def _ensure(self, metric: _Metric) -> Any:
        if metric.name in self._objects:
            return self._objects[metric.name]
        pc = self._client()
        reg = self._client_registry or pc.REGISTRY
        kwargs: dict[str, Any] = {"registry": reg}
        if isinstance(metric, Histogram):
            obj = pc.Histogram(
                metric.name, metric.help, metric.labelnames, buckets=metric.buckets, **kwargs
            )
        elif isinstance(metric, Counter):
            obj = pc.Counter(metric.name, metric.help, metric.labelnames, **kwargs)
        else:
            obj = pc.Gauge(metric.name, metric.help, metric.labelnames, **kwargs)
        self._objects[metric.name] = obj
        return obj

    def sync(self) -> None:
        for name in self.registry.names():
            metric = self.registry.get(name)
            obj = self._ensure(metric)
            if isinstance(metric, Histogram):
                self._sync_histogram(metric, obj)
            elif isinstance(metric, Counter):
                for key, v in metric._values.items():
                    target = obj.labels(**dict(key)) if key else obj
                    current = target._value.get()
                    if v > current:
                        target.inc(v - current)
            elif isinstance(metric, Gauge):
                for key, v in metric._values.items():
                    target = obj.labels(**dict(key)) if key else obj
                    target.set(v)

    def _sync_histogram(self, metric: Histogram, obj: Any) -> None:
        for key, counts in metric._counts.items():
            target = obj.labels(**dict(key)) if key else obj
            prev = self._seen.get((metric.name, key), [0] * len(counts))
            lower = 0.0
            cumulative_prev = 0
            cumulative_now = 0
            for b, c, p in zip(metric.buckets, counts, prev, strict=True):
                delta = (c - cumulative_now) - (p - cumulative_prev)
                cumulative_now, cumulative_prev = c, p
                mid = (lower + b) / 2.0
                for _ in range(max(delta, 0)):
                    target.observe(mid)
                lower = b
            self._seen[(metric.name, key)] = list(counts)


__all__ = [
    "DEFAULT_MS_BUCKETS",
    "METRIC_DRIFT_PSI",
    "METRIC_EDGE_BUNDLE_INFO",
    "METRIC_EDGE_VERIFY_FAILED_TOTAL",
    "METRIC_ERRORS_TOTAL",
    "METRIC_LLM_TTFT_MS",
    "METRIC_STAGE_MS",
    "METRIC_TTFA_MS",
    "METRIC_TURN_TOTAL",
    "METRIC_WER",
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "PrometheusClientAdapter",
    "default_registry",
    "install_standard_metrics",
    "render_prometheus",
]
