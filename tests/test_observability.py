"""Metrics and trace propagation.

Two properties carry most of the weight here.

**Every exit path is instrumented.** `recommend` has four returns — the answer
and three fallbacks — and a fallback that is not counted is precisely the
failure the fallback rate exists to reveal. The service would look healthy in
every graph while serving a list of popular items to everyone. So there is a
structural test that no `return` in `recommend` bypasses `_finish`, not just a
behavioural one that happens to cover the paths someone thought of.

**The exposition format is verified against the reference implementation.**
This module hand-rolls Prometheus text format rather than depending on
`prometheus_client` at runtime, which is only defensible if the output is
checked against the real parser. It is — `prometheus_client` is a dev
dependency used for exactly that, the same arrangement as `kafka-python` and
the murmur2 vectors.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from retailgr.serving import metrics, tracing


@pytest.fixture(autouse=True)
def clean_registry():
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


# -- the exposition format ----------------------------------------------------


def test_the_output_parses_with_the_official_prometheus_parser():
    """The check that makes hand-rolling the format defensible."""
    parser = pytest.importorskip("prometheus_client.parser")

    metrics.REQUESTS.inc(served_from="model", ranker_used="false")
    metrics.DURATION.observe(0.004, stage="total")
    metrics.ITEMS_RETURNED.observe(20)
    metrics.BUILD.set(
        1, model_version="v1", model_type="hstu", variant="config", ranker_served="false"
    )

    families = {f.name: f for f in parser.text_string_to_metric_families(metrics.render())}
    assert "retailgr_requests" in families
    assert families["retailgr_requests"].type == "counter"
    assert families["retailgr_request_duration_seconds"].type == "histogram"
    assert families["retailgr_build_info"].type == "gauge"


def test_histogram_buckets_are_cumulative():
    """A non-cumulative histogram parses fine and makes every quantile wrong."""
    for value in (0.001, 0.02, 0.4):
        metrics.DURATION.observe(value, stage="total")

    counts = metrics.DURATION.counts[("total",)]
    assert counts == sorted(counts), counts
    assert counts[-1] <= metrics.DURATION.totals[("total",)]


def test_the_latency_slo_is_a_bucket_that_exists():
    """An SLO of "99% under 100 ms" is only computable if 0.1 is a boundary.
    Prometheus cannot interpolate a bucket that was never defined."""
    assert 0.1 in metrics.DURATION_BUCKETS
    for budget_ms in (5, 10, 25, 35):
        assert budget_ms / 1000 in metrics.DURATION_BUCKETS, budget_ms


def test_the_buckets_match_the_budget_the_benchmark_checks():
    """Two places state the budget; if they drift, the dashboard and the
    benchmark start disagreeing about whether the system is within it."""
    from retailgr.serving.bench import CLAIMED_BUDGET_MS, CLAIMED_TOTAL_MS

    for milliseconds in CLAIMED_BUDGET_MS.values():
        assert milliseconds / 1000 in metrics.DURATION_BUCKETS, milliseconds
    assert CLAIMED_TOTAL_MS / 1000 in metrics.DURATION_BUCKETS


def test_label_values_are_escaped():
    metrics.BUILD.set(
        1,
        model_version='we"ird\\value',
        model_type="hstu",
        variant="config",
        ranker_served="false",
    )
    rendered = metrics.render()
    assert r"\"" in rendered and r"\\" in rendered
    parser = pytest.importorskip("prometheus_client.parser")
    list(parser.text_string_to_metric_families(rendered))  # must still parse


def test_an_empty_registry_renders_valid_output():
    parser = pytest.importorskip("prometheus_client.parser")
    list(parser.text_string_to_metric_families(metrics.render()))


# -- cardinality --------------------------------------------------------------


def test_no_metric_is_labelled_by_anything_unbounded():
    """A `user_id` label is two failures at once: it multiplies the series
    count by the customer count, and it puts identifiers into a system with
    no retention policy and a far wider audience than the lakehouse."""
    forbidden = {"user_id", "session_id", "device_id", "sku", "request_id", "trace_id"}
    for name in ("retailgr_requests", "retailgr_request_duration_seconds",
                 "retailgr_policy_dropped", "retailgr_items_returned",
                 "retailgr_build_info"):
        labels = set(metrics.REGISTRY[name].labelnames)
        assert not (labels & forbidden), f"{name} is labelled by {labels & forbidden}"


def test_the_series_count_cannot_grow_with_traffic():
    """Ten thousand requests from ten thousand customers must produce the same
    number of series as ten."""
    for index in range(2000):
        metrics.REQUESTS.inc(served_from="model", ranker_used="false")
        metrics.DURATION.observe(0.001 * (index % 50), stage="total")
        metrics.POLICY_DROPPED.inc(reason="dropped_out_of_stock")
    after_many = metrics.REGISTRY.series_count()

    metrics.REGISTRY.reset()
    for _ in range(10):
        metrics.REQUESTS.inc(served_from="model", ranker_used="false")
        metrics.DURATION.observe(0.001, stage="total")
        metrics.POLICY_DROPPED.inc(reason="dropped_out_of_stock")
    assert metrics.REGISTRY.series_count() == after_many


# -- every exit path is counted -----------------------------------------------


def test_no_return_in_recommend_bypasses_the_instrumented_exit():
    """Structural, not behavioural.

    A behavioural test covers the paths someone remembered to write. This
    covers the one added next year: every `return` inside `recommend` has to
    go through `_finish`, which is where the metrics and the exposure log
    live. `recommend` has four exits, three of them fallbacks, and an
    uncounted fallback is invisible in exactly the situation the fallback
    rate exists to make visible.
    """
    from retailgr.serving.service import RecommendationService

    source = inspect.getsource(RecommendationService.recommend)
    tree = ast.parse(inspect.cleandoc(source))
    function = tree.body[0]

    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) >= 4, "recommend lost its fallback paths?"

    for node in returns:
        assert isinstance(node.value, ast.Call), ast.dump(node)
        assert isinstance(node.value.func, ast.Attribute), ast.dump(node)
        assert node.value.func.attr == "_finish", (
            f"a return at line {node.lineno} of recommend skips _finish"
        )


def test_a_fallback_is_counted_as_a_fallback():
    """The metric that matters: the system answering without the model looks
    perfectly healthy from the outside."""

    class _Response:
        served_from = "cold_start"
        ranker_used = False
        items: list = []
        timings = metrics  # any object; the getattr below finds no _ms fields

    metrics.observe_response(_Response())
    rendered = metrics.render()
    assert 'served_from="cold_start"' in rendered


def test_policy_drops_are_counted_by_reason_and_promotions_are_not():
    """`promoted` and `pinned` are not losses; counting them as drops would
    make the dashboard read as if the policy were discarding the catalogue."""
    from retailgr.serving.policy import PolicyTrace

    trace = PolicyTrace(
        candidates_in=100,
        dropped_out_of_stock=7,
        dropped_purchased=3,
        promoted=5,
        pinned=2,
        items_out=20,
    )

    class _Response:
        served_from = "model"
        ranker_used = True
        items = [{}] * 20
        timings = None

    metrics.observe_response(_Response(), trace)
    rendered = metrics.render()
    assert 'reason="dropped_out_of_stock"' in rendered
    assert 'reason="dropped_purchased"' in rendered
    assert "promoted" not in rendered
    assert "pinned" not in rendered


# -- trace propagation --------------------------------------------------------


def test_a_valid_traceparent_is_adopted():
    context = tracing.parse_traceparent(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )
    assert context is not None
    assert context.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert context.parent_id == "00f067aa0ba902b7"
    assert context.sampled is True
    assert context.source == "traceparent"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "garbage",
        "00-tooshort-00f067aa0ba902b7-01",
        # All-zero ids are invalid per the specification.
        "00-" + "0" * 32 + "-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01",
        # Version ff is forbidden.
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
    ],
)
def test_a_malformed_traceparent_is_ignored_not_fatal(header):
    """An upstream service sending a broken header is not a reason to fail a
    customer's request."""
    assert tracing.parse_traceparent(header) is None


def test_an_unknown_version_is_still_accepted():
    """Forward compatibility is in the specification: parse the fields we
    understand rather than rejecting a future version wholesale."""
    context = tracing.parse_traceparent(
        "02-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
    )
    assert context is not None
    assert context.sampled is False


def test_x_request_id_is_accepted_when_there_is_no_traceparent():
    context = tracing.context_from_headers({"x-request-id": "abc-123"})
    assert context.request_id == "abc-123"
    assert context.source == "x-request-id"


def test_traceparent_wins_over_x_request_id():
    context = tracing.context_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "x-request-id": "abc-123",
        }
    )
    assert context.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_a_request_with_no_headers_gets_a_new_trace():
    context = tracing.context_from_headers({})
    assert len(context.trace_id) == 32
    assert context.source == "generated"


def test_an_absurd_request_id_is_not_adopted():
    """An id is a key in the exposure log; an unbounded one from an untrusted
    caller is a storage problem with extra steps."""
    context = tracing.context_from_headers({"x-request-id": "x" * 5000})
    assert context.source == "generated"


def test_the_outgoing_header_is_well_formed():
    context = tracing.parse_traceparent(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )
    onward = tracing.format_traceparent(context)
    assert tracing.TRACEPARENT.match(onward)
    reparsed = tracing.parse_traceparent(onward)
    assert reparsed.trace_id == context.trace_id
    assert reparsed.parent_id != context.parent_id, "this hop needs its own span id"


def test_a_non_hex_request_id_still_produces_a_valid_outgoing_header():
    """`X-Request-Id` can be anything; `traceparent` cannot."""
    context = tracing.context_from_headers({"x-request-id": "not-hex-at-all"})
    assert tracing.TRACEPARENT.match(tracing.format_traceparent(context))


# -- end to end through the API -----------------------------------------------


def _client():
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from retailgr.serving.api import create_app
    from retailgr.serving.service import RecommendationResponse, StageTimings

    service = MagicMock()
    service.bundle.model_version = "test-version"
    service.bundle.manifest.model_type = "hstu"
    service.bundle.manifest.vocab_size = 945
    service.index.size = 900

    def recommend(context, limit=20, request_id=None):
        return RecommendationResponse(
            request_id=request_id or "generated-inside",
            model_version="test-version",
            items=[],
            served_from="model",
            timings=StageTimings(total_ms=3.0),
        )

    service.recommend.side_effect = recommend
    return TestClient(create_app(service), raise_server_exceptions=False)


def test_the_api_passes_the_callers_trace_id_to_the_service():
    """The defect this module exists for: the handler used to drop it."""
    client = _client()
    response = client.post(
        "/v1/recommendations",
        json={"user_id": "u1"},
        headers={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
    )
    assert response.status_code == 200
    assert response.json()["request_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_the_api_echoes_the_id_back():
    client = _client()
    response = client.post("/v1/recommendations", json={"user_id": "u1"})
    assert response.headers["x-request-id"] == response.json()["request_id"]
    assert tracing.TRACEPARENT.match(response.headers["traceparent"])


def test_the_metrics_endpoint_serves_text_not_json():
    client = _client()
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert not response.text.startswith('"'), "FastAPI JSON-encoded the payload"
    assert "# TYPE retailgr_requests counter" in response.text
