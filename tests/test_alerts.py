"""Alert rules, checked against the metrics that actually exist.

An alert rule is the one artifact in a system that is only exercised when
everything else has already gone wrong. It is never run in development, it
produces no output when it is correct, and it produces no output when it is
broken either. A rule naming a metric nobody exports looks exactly like a
rule that is quietly protecting you.

So three loops get closed here, in order of how much they cost when they are
open:

1. **Every metric name in a PromQL expression is one `metrics.py` emits**,
   down to the `_total` / `_bucket` / `_count` suffix and the label names.
   This is the check the manifest's own comment promises.
2. **Every label *value* in a selector is one the code can produce.**
   `served_from="retrieval_error"` matches nothing if the service spells it
   `retrieval_failed`, and matching nothing is silence, not an error.
3. **Every `le` is a real bucket boundary, rendered the way the exporter
   renders it.** `le` is a string label that Prometheus compares as a
   string. A rule asking for `le="0.10"` against a bucket exposed as
   `le="0.1"` selects an empty vector for ever.

The expressions are parsed with `promql-parser`, a binding over the same
grammar Prometheus uses, so "it parses" means it parses — not that it looked
plausible to a regex. What a parser cannot tell you is whether an expression
asks a sensible question: the staleness rule here used to be
`changes(sum by (model_version) (retailgr_build_info)[7d:1h]) == 0`, which
parses, and which counts changes in a gauge whose value is the constant 1.
It measured replica scaling, and a version deployed an hour ago had zero
changes and fired as stale. The tests at the end are the ones that would
have caught it: they assert the rules ask about numbers that move.

`kubernetes-validate` has no schema for `PrometheusRule` — it is a custom
resource, and its schema ships with the Prometheus operator rather than the
Kubernetes API. `test_deploy.py` therefore exempts this kind from its schema
sweep and asserts the exemption is covered here, so the gap cannot widen
silently into "nobody validates that file".
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from retailgr.serving import metrics

ALERTS_FILE = Path("deploy/k8s/50-alerts.yaml")

# Declared for `test_deploy.py`, which skips these kinds in its Kubernetes
# schema sweep. The two lists have to agree: a kind exempted there and absent
# here is a manifest nothing validates at all.
VALIDATED_KINDS = frozenset({"PrometheusRule"})


def _document() -> dict:
    documents = [d for d in yaml.safe_load_all(ALERTS_FILE.read_text(encoding="utf-8")) if d]
    assert len(documents) == 1, "expected exactly one document in the alerts manifest"
    return documents[0]


DOCUMENT = _document()
GROUPS = DOCUMENT["spec"]["groups"]
RULES = [(group["name"], rule) for group in GROUPS for rule in group["rules"]]
ALERTS = [(g, r) for g, r in RULES if "alert" in r]
RECORDS = [(g, r) for g, r in RULES if "record" in r]

ALERT_IDS = [r["alert"] for _, r in ALERTS]
RECORD_IDS = [r["record"] for _, r in RECORDS]
RULE_IDS = [r.get("alert") or r["record"] for _, r in RULES]


# -- what the exporter actually offers ----------------------------------------


def _exported_series() -> dict[str, set[str]]:
    """Every series name this process can emit, mapped to its label names.

    Read off the registry rather than listed by hand, so a metric renamed in
    `metrics.py` breaks the rules that name the old one instead of drifting
    away from them.
    """
    out: dict[str, set[str]] = {}
    for name in sorted(metrics.REGISTRY._metrics):
        metric = metrics.REGISTRY[name]
        labels = set(metric.labelnames)
        if isinstance(metric, metrics.Histogram):
            out[f"{name}_bucket"] = labels | {"le"}
            out[f"{name}_sum"] = set(labels)
            out[f"{name}_count"] = set(labels)
        elif isinstance(metric, metrics.Counter):
            out[f"{name}_total"] = set(labels)
        else:
            out[name] = set(labels)
    return out


EXPORTED = _exported_series()
RECORDED = {rule["record"] for _, rule in RECORDS}

# PromQL functions and keywords that sit in the same syntactic position as a
# metric name. Extracting names by regex would trip over every one of them,
# which is why the selectors below come out of the parser instead.
_DURATION = re.compile(r"^\d+(ms|s|m|h|d|w|y)$")


def _selectors(expression: str) -> list[tuple[str, dict[str, str]]]:
    """Every vector selector in an expression: (metric name, equality labels).

    The parser's printed form is stable and round-trips, so the selectors are
    recovered from it rather than by walking a node hierarchy the binding
    does not expose. Inequality matchers (`!=`, `=~`) are kept out of the
    label map on purpose: `served_from!="model"` says nothing about which
    values exist, so asserting against it would be asserting nothing.
    """
    promql_parser = pytest.importorskip("promql_parser")
    rendered = str(promql_parser.parse(expression))

    found: list[tuple[str, dict[str, str]]] = []
    for match in re.finditer(r"([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}", rendered):
        name, body = match.group(1), match.group(2)
        labels: dict[str, str] = {}
        for part in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(=~|!~|!=|=)\s*"([^"]*)"', body):
            key, operator, value = part.groups()
            if key == "__name__":
                name = value
                continue
            if operator == "=":
                labels[key] = value
        found.append((name, labels))
    return found


def test_the_parser_is_available_and_rejects_bad_promql():
    """Proof the parse checks below can fail.

    A parser that accepts anything reports every expression in the file as
    valid, which is the same outcome as not checking at all.
    """
    promql_parser = pytest.importorskip("promql_parser")
    with pytest.raises(ValueError):
        promql_parser.parse("sum(rate(retailgr_requests_total[5m])")  # unbalanced
    with pytest.raises(ValueError):
        promql_parser.parse('retailgr_requests_total{served_from="model"')


def test_the_alerts_manifest_exists_and_carries_rules():
    """A sweep over zero rules passes silently."""
    assert DOCUMENT["kind"] == "PrometheusRule"
    assert len(ALERTS) >= 8, f"only {len(ALERTS)} alerts"
    assert len(RECORDS) >= 1


# -- loop 1: the metric names are real ----------------------------------------


@pytest.mark.parametrize("group,rule", RULES, ids=RULE_IDS)
def test_every_expression_parses_as_promql(group, rule):
    promql_parser = pytest.importorskip("promql_parser")
    promql_parser.parse(rule["expr"])


@pytest.mark.parametrize("group,rule", RULES, ids=RULE_IDS)
def test_every_metric_a_rule_names_is_one_the_exporter_emits(group, rule):
    """The promise the manifest makes in its own header comment.

    This is the failure that looks most like coverage: the rule is in the
    repository, it is loaded by Prometheus, it is syntactically perfect, and
    it selects an empty vector on every evaluation because the metric is
    called something else.
    """
    for name, _ in _selectors(rule["expr"]):
        if ":" in name:  # a recording rule, checked separately below
            continue
        assert name in EXPORTED, (
            f"{rule.get('alert') or rule['record']} selects `{name}`, which "
            f"src/retailgr/serving/metrics.py does not export. "
            f"Exported: {sorted(EXPORTED)}"
        )


@pytest.mark.parametrize("group,rule", RULES, ids=RULE_IDS)
def test_every_label_a_rule_filters_on_exists_on_that_metric(group, rule):
    """A selector on a label the metric does not carry matches nothing.

    Same silence as a misspelled metric name, one level deeper and harder to
    see: `retailgr_requests_total{stage="total"}` is a perfectly well-formed
    request for a series that cannot exist, because `stage` belongs to the
    duration histogram and not to the request counter.
    """
    for name, labels in _selectors(rule["expr"]):
        if ":" in name:
            continue
        available = EXPORTED[name]
        for key in labels:
            assert key in available, (
                f"{rule.get('alert') or rule['record']} filters `{name}` on "
                f"`{key}`, which is not one of its labels {sorted(available)}"
            )


def test_every_recording_rule_an_alert_uses_is_defined_here():
    """Recording rules are the one reference Prometheus will not warn about.

    An alert on `retailgr:request_slo_failure_ratio:rate5m` when the rule is
    named `...:ratio5m` evaluates to nothing for ever, and the only symptom
    is an alert that never fires.
    """
    for _, rule in ALERTS:
        for name, _ in _selectors(rule["expr"]):
            if ":" not in name:
                continue
            assert name in RECORDED, (
                f"{rule['alert']} uses `{name}`, which no recording rule defines. "
                f"Defined: {sorted(RECORDED)}"
            )


def test_every_recording_rule_is_actually_used():
    """The other direction. A recording rule nobody reads is evaluated on
    every interval for ever and costs storage to answer no question."""
    used: set[str] = set()
    for _, rule in RULES:
        for name, _ in _selectors(rule["expr"]):
            if ":" in name and name != rule.get("record"):
                used.add(name)
    unused = RECORDED - used
    assert not unused, f"recording rules nothing reads: {sorted(unused)}"


# -- loop 2: the label values are ones the code can produce -------------------


def test_served_from_values_are_the_ones_the_service_sets():
    """`served_from` is the most important label in the file and the easiest
    to get wrong, because the values are string literals in `service.py` and
    string literals in YAML with nothing joining them."""
    source = Path("src/retailgr/serving/service.py").read_text(encoding="utf-8")
    produced = {"model"} | set(re.findall(r'_fallback_response\([^)]*?"([a-z_]+)"', source, re.S))
    assert {"cold_start", "retrieval_error", "policy_emptied"} <= produced, produced

    for _, rule in RULES:
        for name, labels in _selectors(rule["expr"]):
            if name != "retailgr_requests_total" or "served_from" not in labels:
                continue
            value = labels["served_from"]
            assert value in produced, (
                f"{rule.get('alert') or rule['record']} selects "
                f"served_from={value!r}; the service only ever sets {sorted(produced)}"
            )


def test_drop_reasons_in_the_rules_are_real_policy_trace_fields():
    from retailgr.serving.policy import PolicyTrace

    fields = set(PolicyTrace().as_dict())
    assert set(metrics.DROP_REASONS) <= fields, "DROP_REASONS has drifted from PolicyTrace"

    for _, rule in RULES:
        for name, labels in _selectors(rule["expr"]):
            if name != "retailgr_policy_dropped_total" or "reason" not in labels:
                continue
            assert labels["reason"] in metrics.DROP_REASONS, (
                f"{rule.get('alert') or rule['record']} selects "
                f"reason={labels['reason']!r}, which the exporter never emits"
            )


def test_boolean_labels_are_selected_in_the_case_the_exporter_writes_them():
    """`str(bool(x)).lower()` produces `true`, and a rule asking for `True`
    matches nothing. Prometheus label values are case-sensitive strings and
    Python's are capitalised, so this mismatch is the default outcome."""
    for _, rule in RULES:
        for _, labels in _selectors(rule["expr"]):
            for key in ("ranker_used", "ranker_served"):
                if key in labels:
                    assert labels[key] in {"true", "false"}, (
                        f"{rule.get('alert') or rule['record']} selects "
                        f"{key}={labels[key]!r}; the exporter writes 'true'/'false'"
                    )


# -- loop 3: `le` is a bucket that exists, spelled the way it is exposed ------


def _bucket_strings(buckets) -> set[str]:
    return {metrics._format_bound(bound) for bound in buckets}


def test_every_le_selector_is_a_boundary_the_exporter_renders_identically():
    """`le` is compared as a string.

    `le="0.10"` against a bucket exposed as `le="0.1"` is an empty vector, and
    a ratio whose numerator is an empty vector is not a small number — it is
    no result at all, so the alert neither fires nor reports that it cannot.
    """
    by_metric = {
        "retailgr_request_duration_seconds_bucket": _bucket_strings(metrics.DURATION_BUCKETS),
        "retailgr_items_returned_bucket": _bucket_strings(metrics.COUNT_BUCKETS),
    }
    seen = 0
    for _, rule in RULES:
        for name, labels in _selectors(rule["expr"]):
            if "le" not in labels:
                continue
            assert name in by_metric, f"unexpected histogram `{name}`"
            assert labels["le"] in by_metric[name] or labels["le"] == "+Inf", (
                f"{rule.get('alert') or rule['record']} asks for le={labels['le']!r} "
                f"on {name}; the exposed boundaries are {sorted(by_metric[name])}"
            )
            seen += 1
    assert seen >= 2, "expected the SLO and the empty-response rules to select a bucket"


def test_the_latency_slo_bucket_is_the_documented_budget():
    """The SLO is readable straight off the metric only if the number in the
    rule is the number in the architecture document."""
    from retailgr.serving.bench import CLAIMED_BUDGET_MS, CLAIMED_TOTAL_MS

    total = float(CLAIMED_TOTAL_MS) / 1000.0
    assert metrics._format_bound(total) == "0.1"
    assert total in metrics.DURATION_BUCKETS

    # The claim `metrics.py` makes in its own docstring: every per-stage
    # budget is a boundary too, so a stage's SLO is also a bucket you can
    # read rather than a quantile you have to interpolate.
    for stage, budget in CLAIMED_BUDGET_MS.items():
        assert float(budget) / 1000.0 in metrics.DURATION_BUCKETS, (
            f"the {stage} budget falls inside a bucket instead of on a boundary"
        )

    slo_rules = [r for _, r in RECORDS if "slo_failure_ratio" in r["record"]]
    assert slo_rules, "no SLO recording rule"
    for rule in slo_rules:
        les = [
            labels["le"]
            for _, labels in _selectors(rule["expr"])
            if "le" in labels
        ]
        assert les == ["0.1"], f"{rule['record']} measures against {les}, not the 100 ms budget"


def test_the_empty_response_rule_counts_responses_with_no_items():
    """`le="0"` on a count histogram is the number of zero-item responses —
    but only because 0 is an explicit boundary. Without it the lowest bucket
    would be `le="1"`, which also contains the responses that returned one
    item, and the alert would be measuring something else entirely."""
    assert 0 in metrics.COUNT_BUCKETS
    empty = [r for _, r in ALERTS if r["alert"] == "RetailGREmptyResponses"]
    assert empty, "RetailGREmptyResponses is gone"
    les = [labels["le"] for _, labels in _selectors(empty[0]["expr"]) if "le" in labels]
    assert les == ["0"]


# -- the rules ask about numbers that move ------------------------------------


def test_no_rule_applies_a_rate_or_changes_to_a_gauge():
    """The defect that produced this test.

    `changes(sum by (model_version) (retailgr_build_info)[7d:1h]) == 0` parses,
    reads convincingly, and is nonsense: build-info is pinned at 1, so the
    expression measures how often the replica count moved. Worse than not
    firing — it fires an hour after every successful deploy.

    `rate` and `changes` are for things that accumulate. Applying them to a
    constant gauge is always a mistake, so it is checked structurally rather
    than rule by rule.
    """
    gauges = {
        name
        for name in metrics.REGISTRY._metrics
        if isinstance(metrics.REGISTRY[name], metrics.Gauge)
    }
    assert gauges, "no gauges registered; this test would pass vacuously"

    pattern = re.compile(r"\b(rate|irate|increase|changes|delta|resets)\s*\(")
    for _, rule in RULES:
        expression = rule["expr"]
        for match in pattern.finditer(expression):
            # The argument runs to the matching close paren.
            depth, start = 0, match.end() - 1
            for index in range(start, len(expression)):
                depth += expression[index] == "("
                depth -= expression[index] == ")"
                if depth == 0:
                    argument = expression[start : index + 1]
                    break
            else:  # pragma: no cover - the parse test would have failed first
                raise AssertionError(f"unbalanced parentheses in {rule}")
            for gauge in gauges:
                assert gauge not in argument, (
                    f"{rule.get('alert') or rule['record']} applies "
                    f"{match.group(1)}() to the gauge `{gauge}`"
                )


def test_the_staleness_alert_reads_a_timestamp_that_the_exporter_sets():
    """Freshness needs a number that advances on its own.

    The bundle timestamp is the *bundle's*, not the process's: using the
    process's start time would reset freshness on every restart, and a
    restart is the event most likely to happen while the export pipeline is
    broken.
    """
    assert "retailgr_build_timestamp_seconds" in metrics.REGISTRY

    stale = [r for _, r in ALERTS if r["alert"] == "RetailGRBundleStale"]
    assert stale, "no staleness alert"
    names = {name for name, _ in _selectors(stale[0]["expr"])}
    assert "retailgr_build_timestamp_seconds" in names
    assert "time()" in stale[0]["expr"], "staleness has to be measured against now"


def test_a_real_manifest_timestamp_survives_the_round_trip():
    """The alert is only as good as this parse. A `created_at` that fails to
    parse exports no series, `max()` over nothing returns nothing, and the
    staleness alert silently loses the ability to fire."""
    import time

    from retailgr.serving.bundle import BundleManifest

    manifest = BundleManifest(
        model_version="test",
        model_type="hstu",
        variant="config",
        dataset="synthetic",
        vocab_size=945,
        embedding_dim=64,
        max_len=50,
    )
    parsed = metrics.build_timestamp(manifest.created_at)
    assert parsed is not None, f"{manifest.created_at!r} did not parse"
    assert abs(parsed - time.time()) < 120

    assert metrics.build_timestamp("not a timestamp") is None
    assert metrics.build_timestamp(None) is None
    # Naive and `Z`-suffixed forms both land on the same instant.
    assert metrics.build_timestamp("2026-01-01T00:00:00Z") == metrics.build_timestamp(
        "2026-01-01T00:00:00"
    )


def test_record_build_publishes_the_timestamp_alongside_the_build_info():
    from retailgr.serving.bundle import BundleManifest

    metrics.REGISTRY.reset()
    manifest = BundleManifest(
        model_version="v-under-test",
        model_type="hstu",
        variant="config",
        dataset="synthetic",
        vocab_size=945,
        embedding_dim=64,
        max_len=50,
    )
    metrics.record_build(manifest, ranker_served=False)
    rendered = metrics.render()
    try:
        assert 'retailgr_build_info{model_type="hstu"' in rendered
        assert "retailgr_build_timestamp_seconds " in rendered
    finally:
        metrics.REGISTRY.reset()


# -- the shape an operator depends on -----------------------------------------


@pytest.mark.parametrize("group,rule", ALERTS, ids=ALERT_IDS)
def test_every_alert_says_how_urgent_it_is_and_what_it_means(group, rule):
    labels = rule.get("labels", {})
    annotations = rule.get("annotations", {})
    assert labels.get("severity") in {"page", "ticket"}, rule["alert"]
    assert labels.get("slo"), f"{rule['alert']} has no slo label to group by"
    assert annotations.get("summary"), rule["alert"]
    assert len(annotations.get("description", "")) > 80, (
        f"{rule['alert']} has no description worth reading at 3am"
    )


@pytest.mark.parametrize("group,rule", ALERTS, ids=ALERT_IDS)
def test_anything_that_pages_a_human_tells_them_what_to_do(group, rule):
    """A page without a runbook is a page that ends in someone reading the
    PromQL to work out what the alert meant. Tickets are exempt: they are
    read at a desk, with the repository open."""
    if rule["labels"]["severity"] != "page":
        pytest.skip("tickets are read at a desk")
    assert rule["annotations"].get("runbook"), f"{rule['alert']} pages with no runbook"


@pytest.mark.parametrize("group,rule", ALERTS, ids=ALERT_IDS)
def test_every_alert_waits_before_firing(group, rule):
    """No `for:` means a single bad evaluation pages. Every alert here is a
    rate or a ratio over a window, and those are noisiest exactly when
    traffic is lowest."""
    assert "for" in rule, f"{rule['alert']} fires on one evaluation"
    assert _DURATION.match(rule["for"]), f"{rule['alert']} has for: {rule['for']!r}"


def test_alert_and_group_names_are_unique():
    """Two rules with one name is a silent overwrite in every dashboard and
    a silencing rule that suppresses more than its author meant."""
    assert len(ALERT_IDS) == len(set(ALERT_IDS)), ALERT_IDS
    assert len(RECORD_IDS) == len(set(RECORD_IDS)), RECORD_IDS
    names = [group["name"] for group in GROUPS]
    assert len(names) == len(set(names)), names


def test_every_group_declares_an_evaluation_interval():
    for group in GROUPS:
        assert _DURATION.match(group.get("interval", "")), group["name"]


def test_the_rule_carries_the_labels_the_operator_selects_on():
    """A PrometheusRule the operator's ruleSelector does not match is a file
    that is applied, accepted, and never loaded."""
    labels = DOCUMENT["metadata"]["labels"]
    assert labels.get("prometheus"), "no prometheus label; the operator will not pick this up"
    assert labels.get("role") == "alert-rules"
    assert DOCUMENT["metadata"]["namespace"] == "retailgr"


# -- the rules against a real scrape, not against the declarations -----------


def _scraped_series() -> dict[str, set[str]]:
    """Drive the exporter and read back what a Prometheus scrape would see.

    Every check above reads the registry's *declarations*. This one reads the
    rendered exposition, which is a different artifact: a histogram that
    declares a `stage` label and renders without it, or a counter whose
    `_total` suffix is dropped by a change to `Counter.render`, breaks every
    rule in the file while leaving all the declaration checks green.
    """
    parser = pytest.importorskip("prometheus_client.parser")

    from retailgr.serving.bundle import BundleManifest
    from retailgr.serving.policy import PolicyTrace

    metrics.REGISTRY.reset()

    class _Timings:
        context_ms, retrieval_ms, filter_ms = 4.0, 18.0, 2.0
        ranking_ms, rerank_ms, total_ms = 21.0, 3.0, 48.0

    class _Response:
        def __init__(self, served_from, ranker_used, items):
            self.served_from = served_from
            self.ranker_used = ranker_used
            self.items = [{}] * items
            self.timings = _Timings()

    metrics.record_build(
        BundleManifest(
            model_version="scrape-test",
            model_type="hstu",
            variant="config",
            dataset="synthetic",
            vocab_size=945,
            embedding_dim=64,
            max_len=50,
        ),
        ranker_served=True,
    )
    # One of each `served_from`, so the label the fallback alert selects on
    # is present with every value the service can produce.
    for served_from in ("model", "cold_start", "retrieval_error", "policy_emptied"):
        metrics.observe_response(
            _Response(served_from, served_from == "model", 20 if served_from == "model" else 0),
            PolicyTrace(candidates_in=50, dropped_out_of_stock=9, dropped_purchased=2),
        )

    # Every metric the registry declares has to be driven here, or the
    # "declared but never rendered" check below has nothing to compare
    # against — which is exactly how this fixture caught an exposure-log
    # counter that was registered and never incremented.
    metrics.observe_exposure("sent")
    metrics.observe_exposure("failed")

    out: dict[str, set[str]] = {}
    for family in parser.text_string_to_metric_families(metrics.render()):
        for sample in family.samples:
            out.setdefault(sample.name, set()).update(sample.labels)
    metrics.REGISTRY.reset()
    return out


SCRAPED = _scraped_series()


@pytest.mark.parametrize("group,rule", RULES, ids=RULE_IDS)
def test_every_selector_matches_a_series_a_real_scrape_produces(group, rule):
    """The end of the chain: these rules would select something.

    Name and labels are checked against the parsed exposition rather than the
    registry, so the assertion covers the render path too. Label *values* are
    deliberately not required to appear — `served_from="retrieval_error"` is
    supposed to match nothing when nothing is failing, and demanding it here
    would only prove the fixture produced it.
    """
    for name, labels in _selectors(rule["expr"]):
        if ":" in name:
            continue
        assert name in SCRAPED, (
            f"{rule.get('alert') or rule['record']} selects `{name}`, which a "
            f"real scrape of this exporter does not contain"
        )
        for key in labels:
            assert key in SCRAPED[name], (
                f"{rule.get('alert') or rule['record']} filters `{name}` on "
                f"`{key}`, which no scraped series of it carries"
            )


def test_everything_the_registry_declares_actually_renders():
    """Guards the test above, and is worth having on its own.

    Every series name derived from the registry has to appear in the parsed
    exposition, and vice versa. A metric that is declared and never rendered
    passes every declaration-level check in this file while being invisible
    to Prometheus; a series that renders under a name the registry does not
    imply is one no rule will ever be written against.
    """
    assert set(SCRAPED) == set(EXPORTED), {
        "declared but never rendered": sorted(set(EXPORTED) - set(SCRAPED)),
        "rendered but not declared": sorted(set(SCRAPED) - set(EXPORTED)),
    }
    assert "le" in SCRAPED["retailgr_request_duration_seconds_bucket"]
    assert "served_from" in SCRAPED["retailgr_requests_total"]


def test_the_fallback_rate_is_alerted_on():
    """The single most important rule in the file, asserted by name.

    Every fallback returns 200 with a plausible list of popular items, so
    latency, error rate and saturation all look perfect while the model has
    stopped participating. If this rule is ever deleted, the system's
    characteristic failure becomes invisible again.
    """
    fallback = [r for _, r in ALERTS if r["alert"] == "RetailGRServingFallbacks"]
    assert fallback, "the fallback-rate alert is gone"
    assert fallback[0]["labels"]["severity"] == "page"
    assert 'served_from!="model"' in fallback[0]["expr"]
