"""The numbers the request path already computes, made observable.

Every response carries per-stage timings and a policy trace explaining why the
list looks the way it does. Until now all of it was computed, returned in the
body, and discarded — so the 100 ms latency budget was verified *offline* by
``retailgr bench`` and completely unobservable *online*. Knowing p99 on the
machine that ran the benchmark is not knowing it in a pod.

Three decisions worth stating, because each is the opposite of the obvious one:

**No dependency.** The registry here is about two hundred lines of counters
and histograms rendering Prometheus text format, rather than
``prometheus_client``. Same reason the broker and the online store have
in-process implementations: the default has to work with nothing installed,
and metrics that only exist when an optional package is present are metrics
nobody can rely on in a test. ``prometheus_client`` is a *dev* dependency —
``tests/test_metrics_export.py`` parses this module's output with the official
parser, so the format is verified against the reference implementation without
being tied to it.

**Buckets come from the budget.** The architecture document claims 100 ms for
the request and 5–35 ms per stage, and ``bench.CLAIMED_BUDGET_MS`` already
holds those figures. The histogram boundaries are built around them rather
than chosen by taste, which makes the SLO readable straight off the metric:
``retailgr_request_duration_seconds_bucket{stage="total",le="0.1"}`` *is* the
count of requests inside budget. An SLO expressed in a bucket that does not
exist is an SLO nobody can compute.

**No user labels, ever.** A ``user_id`` label turns one time series into one
per customer, which kills the Prometheus server and puts identifiers in a
system with no retention policy and a much wider audience than the lakehouse.
``tests/test_metrics_export.py`` asserts the cardinality is bounded by
construction.

The metric that matters most is the dullest: ``retailgr_requests_total`` split
by ``served_from``. Latency tells you the system is slow; the fallback rate
tells you it has stopped answering with the model and started answering with
a list of popular items, which looks completely healthy from the outside.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Derived from `bench.CLAIMED_BUDGET_MS`, in seconds, and deliberately
# straddling both the per-stage budgets (5–35 ms) and the total (100 ms) so
# every one of them lands on a bucket boundary rather than inside one.
DURATION_BUCKETS: tuple[float, ...] = (
    0.001,
    0.0025,
    0.005,  # filter and rerank budget
    0.01,  # context budget
    0.025,  # retrieval budget
    0.035,  # ranking budget
    0.05,
    0.1,  # the total budget: this bucket is the SLO
    0.25,
    0.5,
    1.0,
)

# How many items came back. 0 is the interesting one.
COUNT_BUCKETS: tuple[float, ...] = (0, 1, 5, 10, 20, 50, 100)


def _escape(value: str) -> str:
    return value.replace("\\", r"\\").replace('"', r"\"").replace("\n", r"\n")


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{_escape(str(value))}"' for key, value in sorted(labels.items()))
    return "{" + inner + "}"


@dataclass
class Counter:
    """A number that only goes up, per label combination."""

    name: str
    help: str
    labelnames: tuple[str, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(str(labels.get(name, "")) for name in self.labelnames)
        self.values[key] = self.values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        for key, value in sorted(self.values.items()):
            labels = dict(zip(self.labelnames, key, strict=True))
            lines.append(f"{self.name}_total{_labels(labels)} {value:g}")
        return lines


@dataclass
class Histogram:
    """Cumulative buckets, a sum and a count — the Prometheus shape."""

    name: str
    help: str
    buckets: tuple[float, ...] = DURATION_BUCKETS
    labelnames: tuple[str, ...] = ()
    counts: dict[tuple[str, ...], list[int]] = field(default_factory=dict)
    sums: dict[tuple[str, ...], float] = field(default_factory=dict)
    totals: dict[tuple[str, ...], int] = field(default_factory=dict)

    def observe(self, value: float, **labels: str) -> None:
        key = tuple(str(labels.get(name, "")) for name in self.labelnames)
        if key not in self.counts:
            self.counts[key] = [0] * len(self.buckets)
            self.sums[key] = 0.0
            self.totals[key] = 0
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[key][index] += 1
        self.sums[key] += value
        self.totals[key] += 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key in sorted(self.counts):
            labels = dict(zip(self.labelnames, key, strict=True))
            # Cumulative, which the observe loop above already maintains.
            for index, bound in enumerate(self.buckets):
                bucket_labels = {**labels, "le": _format_bound(bound)}
                lines.append(
                    f"{self.name}_bucket{_labels(bucket_labels)} {self.counts[key][index]}"
                )
            lines.append(f"{self.name}_bucket{_labels({**labels, 'le': '+Inf'})} "
                         f"{self.totals[key]}")
            lines.append(f"{self.name}_sum{_labels(labels)} {self.sums[key]:g}")
            lines.append(f"{self.name}_count{_labels(labels)} {self.totals[key]}")
        return lines


@dataclass
class Gauge:
    """A number that can go either way. Used here only for build info."""

    name: str
    help: str
    labelnames: tuple[str, ...] = ()
    values: dict[tuple[str, ...], float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        key = tuple(str(labels.get(name, "")) for name in self.labelnames)
        self.values[key] = value

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        for key, value in sorted(self.values.items()):
            labels = dict(zip(self.labelnames, key, strict=True))
            lines.append(f"{self.name}{_labels(labels)} {value:g}")
        return lines


def _format_bound(bound: float) -> str:
    if bound == math.inf:
        return "+Inf"
    # Prometheus compares `le` as a float but it is a label, so the string has
    # to be stable: the same bound must render identically every scrape or it
    # becomes a new series.
    return f"{bound:g}"


class Registry:
    """Everything this process measures, rendered on demand.

    Locked because uvicorn serves concurrently and a scrape can land in the
    middle of a request. The lock is held for microseconds around integer
    arithmetic; contention is not the concern, a torn read is.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Histogram | Gauge] = {}
        self._lock = threading.Lock()

    def register(self, metric: Counter | Histogram | Gauge):
        self._metrics[metric.name] = metric
        return metric

    def __getitem__(self, name: str):
        return self._metrics[name]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def series_count(self) -> int:
        """How many time series exist. The number that kills a Prometheus."""
        total = 0
        for metric in self._metrics.values():
            if isinstance(metric, Histogram):
                total += len(metric.counts) * (len(metric.buckets) + 3)
            else:
                total += len(metric.values)
        return total

    def render(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name in sorted(self._metrics):
                lines.extend(self._metrics[name].render())
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            for metric in self._metrics.values():
                if isinstance(metric, Histogram):
                    metric.counts.clear()
                    metric.sums.clear()
                    metric.totals.clear()
                else:
                    metric.values.clear()


# -- the serving metrics -------------------------------------------------------

REGISTRY = Registry()

REQUESTS = REGISTRY.register(
    Counter(
        "retailgr_requests",
        "Recommendation requests, by how the list was produced.",
        labelnames=("served_from", "ranker_used"),
    )
)
DURATION = REGISTRY.register(
    Histogram(
        "retailgr_request_duration_seconds",
        "Time in each stage of the request path. "
        'The le="0.1" bucket of stage="total" is the latency SLO.',
        buckets=DURATION_BUCKETS,
        labelnames=("stage",),
    )
)
POLICY_DROPPED = REGISTRY.register(
    Counter(
        "retailgr_policy_dropped",
        "Candidates removed by the policy layer, by reason.",
        labelnames=("reason",),
    )
)
ITEMS_RETURNED = REGISTRY.register(
    Histogram(
        "retailgr_items_returned",
        "How many items each response carried. Zero is the interesting value.",
        buckets=COUNT_BUCKETS,
        labelnames=(),
    )
)
BUILD = REGISTRY.register(
    Gauge(
        "retailgr_build_info",
        "Always 1. The labels are the payload: which bundle is serving.",
        labelnames=("model_version", "model_type", "variant", "ranker_served"),
    )
)
EXPOSURE_LOG = REGISTRY.register(
    Counter(
        "retailgr_exposure_log",
        "Writes to the exposure topic, by outcome. Without these there are "
        "no unbiased training labels later.",
        labelnames=("outcome",),
    )
)
BUILD_TIMESTAMP = REGISTRY.register(
    Gauge(
        "retailgr_build_timestamp_seconds",
        "When the serving bundle was created, as a unix timestamp.",
        labelnames=(),
    )
)

# The policy trace fields that are drops. The others (`promoted`, `pinned`,
# `candidates_in`, `items_out`) are not losses and are not counted here.
DROP_REASONS: tuple[str, ...] = (
    "dropped_out_of_stock",
    "dropped_ineligible",
    "dropped_purchased",
    "dropped_excluded",
    "dropped_no_sku",
    "capped_by_category",
)


def build_timestamp(created_at: Any) -> float | None:
    """`BundleManifest.created_at` as unix seconds, or None if unreadable.

    Separate and public because the staleness alert is only as good as this
    parse: a manifest whose timestamp silently fails to parse exports no
    series, `max()` over nothing returns nothing, and the alert quietly
    stops being able to fire. `tests/test_alerts.py` asserts a real
    manifest's `created_at` round-trips through here.
    """
    if not isinstance(created_at, str) or not created_at:
        return None
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Written by `datetime.now(timezone.utc)`, so a naive value means
        # something else produced it. UTC is the only defensible guess and
        # the alternative is discarding the timestamp entirely.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def observe_exposure(outcome: str) -> None:
    """Count one exposure-log attempt.

    `service._log_exposure` swallows every exception, with the comment that
    "the gap shows up as a drop in log volume, which is alertable". It was
    not alertable: nothing counted the writes, so a broker that started
    refusing them was invisible — the requests kept succeeding and the
    training labels quietly stopped arriving.

    Counted by outcome rather than as a bare total so the alert can be a
    *ratio*. A ratio is undefined when no exposures are attempted at all,
    which is the default deployment — `api.serve` builds the service without
    a logger — so the rule stays silent there instead of firing forever on a
    feature nobody switched on.
    """
    with REGISTRY.lock:
        EXPOSURE_LOG.inc(outcome=outcome)


def record_build(manifest: Any, ranker_served: bool) -> None:
    """Publish which bundle this process is serving.

    A build-info gauge is how you answer "did the alert start when we shipped
    the new model?" without correlating deploy logs by hand.

    The timestamp is a second gauge rather than another label on the first,
    because the freshness question is arithmetic — `time() - max(...)` — and
    a label is a string. An earlier alert here tried to detect a stale bundle
    with `changes()` over the build-info gauge, which counts changes in its
    *value*; that value is the constant 1, so the expression actually
    measured replica scaling, and a version deployed an hour ago had zero
    changes and fired as stale. Exporting the number the question is about
    was the fix.

    It is the *bundle's* creation time, not the process's start time. Using
    the process would reset freshness on every restart, which is the one
    event most likely to happen while the export pipeline is broken.
    """
    with REGISTRY.lock:
        BUILD.set(
            1,
            model_version=getattr(manifest, "model_version", "unknown"),
            model_type=getattr(manifest, "model_type", "unknown"),
            variant=getattr(manifest, "variant", "unknown"),
            ranker_served=str(bool(ranker_served)).lower(),
        )
        created = build_timestamp(getattr(manifest, "created_at", None))
        if created is not None:
            BUILD_TIMESTAMP.set(created)


def observe_response(response: Any, trace: Any = None) -> None:
    """Record one served request.

    Takes the response the service already built rather than instrumenting
    each stage separately, so the metrics cannot drift away from what the
    caller was actually told.
    """
    timings = getattr(response, "timings", None)
    with REGISTRY.lock:
        REQUESTS.inc(
            served_from=getattr(response, "served_from", "unknown"),
            ranker_used=str(bool(getattr(response, "ranker_used", False))).lower(),
        )
        if timings is not None:
            for stage in ("context", "retrieval", "filter", "ranking", "rerank", "total"):
                milliseconds = getattr(timings, f"{stage}_ms", None)
                if milliseconds is not None:
                    DURATION.observe(float(milliseconds) / 1000.0, stage=stage)
        ITEMS_RETURNED.observe(float(len(getattr(response, "items", []) or [])))

        if trace is not None:
            values = trace.as_dict() if hasattr(trace, "as_dict") else dict(trace)
            for reason in DROP_REASONS:
                count = values.get(reason, 0)
                if count:
                    POLICY_DROPPED.inc(float(count), reason=reason)


CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def render() -> str:
    """The exposition payload, in Prometheus text format."""
    return REGISTRY.render()
