"""The deployment manifests, validated rather than believed.

A Kubernetes manifest is the easiest thing in a repository to get wrong
quietly. It is YAML, nothing type-checks it, and the feedback loop is a
cluster you may not have. So every document under `deploy/k8s/` is validated
here against the *real* Kubernetes API schemas — `kubernetes-validate` vendors
them, so this runs with no cluster, no kubectl and no network.

Schema validity is the floor, not the ceiling, and the tests below the
validation are the ones worth having: they assert the properties this
particular deployment needs, including the ones a schema is perfectly happy
to let you omit. A Deployment with no resource requests, no probes and a root
filesystem is valid YAML and a bad idea.

The last group closes the loop with `retailgr sizing`: the numbers in the
manifests must be the measured ones, so a future measurement that moves and a
manifest that does not is a test failure rather than a slow drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

MANIFEST_DIR = Path("deploy/k8s")
# The oldest version this is claimed to run on. Validating against the newest
# would pass on fields older clusters reject.
TARGET_VERSION = "1.29.0"

# Kinds the Kubernetes API does not define, so `kubernetes-validate` has no
# schema for them — `PrometheusRule` ships with the Prometheus operator, not
# with Kubernetes. Exempting them from the sweep below is unavoidable;
# leaving them unvalidated is not, so
# `test_exempt_kinds_are_validated_somewhere_else` asserts each one is
# covered by a module that does check it.
CUSTOM_RESOURCE_KINDS = frozenset({"PrometheusRule"})


def _documents() -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if document:
                out.append((f"{path.name}:{document['kind']}", document))
    return out


DOCUMENTS = _documents()
IDS = [name for name, _ in DOCUMENTS]


def _by_kind(kind: str) -> list[dict]:
    return [document for _, document in DOCUMENTS if document["kind"] == kind]


def _pod_specs() -> list[tuple[str, dict]]:
    """Every pod template in the directory, whatever wraps it."""
    out: list[tuple[str, dict]] = []
    for name, document in DOCUMENTS:
        spec = document.get("spec", {})
        if document["kind"] == "Deployment":
            out.append((name, spec["template"]["spec"]))
        elif document["kind"] == "CronJob":
            out.append((name, spec["jobTemplate"]["spec"]["template"]["spec"]))
    return out


# -- schema validity ----------------------------------------------------------


def test_the_manifest_directory_is_not_empty():
    """A validation suite over zero files passes, which is the failure mode
    this guards against."""
    assert len(DOCUMENTS) >= 8


@pytest.mark.parametrize("name,document", DOCUMENTS, ids=IDS)
def test_every_manifest_validates_against_the_kubernetes_schema(name, document):
    kubernetes_validate = pytest.importorskip("kubernetes_validate")
    if document["kind"] in CUSTOM_RESOURCE_KINDS:
        pytest.skip(f"{document['kind']} is a CRD; validated in tests/test_alerts.py")
    kubernetes_validate.validate(document, TARGET_VERSION, strict=True)


def test_exempt_kinds_are_validated_somewhere_else():
    """The exemption above is a hole unless something fills it.

    Without this, adding a kind to `CUSTOM_RESOURCE_KINDS` would be enough to
    make any manifest stop being checked at all — a one-line change that
    turns a failing test green and reads like housekeeping.
    """
    from tests import test_alerts

    assert CUSTOM_RESOURCE_KINDS <= test_alerts.VALIDATED_KINDS, (
        f"{sorted(CUSTOM_RESOURCE_KINDS - test_alerts.VALIDATED_KINDS)} are "
        "skipped by the schema sweep and validated by nothing"
    )


def test_no_manifest_is_skipped_by_every_check():
    """Each exempt kind must still appear in a file some test opens."""
    for kind in CUSTOM_RESOURCE_KINDS:
        assert any(document["kind"] == kind for _, document in DOCUMENTS), (
            f"{kind} is exempted from validation and no longer exists"
        )


def test_the_validator_rejects_a_manifest_it_should():
    """Proof the check above can fail. A validator that accepts anything
    reports green on every manifest ever written."""
    kubernetes_validate = pytest.importorskip("kubernetes_validate")
    broken = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": "broken"},
        "spec": {"replicas": "two", "selector": {}, "template": {}},
    }
    with pytest.raises(kubernetes_validate.ValidationError):
        kubernetes_validate.validate(broken, TARGET_VERSION, strict=True)


def test_strict_mode_catches_a_misspelled_field():
    """The failure that costs an afternoon: a typo the API server silently
    ignores, so the setting you thought you applied was never applied."""
    kubernetes_validate = pytest.importorskip("kubernetes_validate")
    document = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "typo"},
        "dat": {"key": "value"},  # should be `data`
    }
    with pytest.raises(kubernetes_validate.ValidationError):
        kubernetes_validate.validate(document, TARGET_VERSION, strict=True)


# -- the properties a schema will not enforce ---------------------------------


@pytest.mark.parametrize("name,pod", _pod_specs(), ids=[n for n, _ in _pod_specs()])
def test_every_container_declares_what_it_needs(name, pod):
    """No requests means the scheduler guesses, and it guesses wrong under
    load — the pod lands on a node with nothing left."""
    for container in pod["containers"] + pod.get("initContainers", []):
        requests = container.get("resources", {}).get("requests", {})
        assert "memory" in requests, f"{name}/{container['name']} has no memory request"
        assert "cpu" in requests, f"{name}/{container['name']} has no cpu request"


@pytest.mark.parametrize("name,pod", _pod_specs(), ids=[n for n, _ in _pod_specs()])
def test_no_container_runs_as_root_or_can_escalate(name, pod):
    assert pod["securityContext"]["runAsNonRoot"] is True
    for container in pod["containers"] + pod.get("initContainers", []):
        security = container.get("securityContext", {})
        assert security.get("allowPrivilegeEscalation") is False, container["name"]
        assert security.get("readOnlyRootFilesystem") is True, container["name"]
        assert security.get("capabilities", {}).get("drop") == ["ALL"], container["name"]


@pytest.mark.parametrize("name,pod", _pod_specs(), ids=[n for n, _ in _pod_specs()])
def test_no_container_carries_a_cpu_limit(name, pod):
    """Deliberate, and the opposite of the usual advice.

    A CPU limit throttles instead of killing. On a latency-budgeted request
    path that turns into p99 spikes which look exactly like a model problem
    and are not, and it also flattens the utilisation signal the autoscaler
    reads. Memory limits stay, because a memory limit is a kill switch and an
    unbounded leak takes the node with it.
    """
    for container in pod["containers"] + pod.get("initContainers", []):
        limits = container.get("resources", {}).get("limits", {})
        assert "cpu" not in limits, f"{name}/{container['name']} has a cpu limit"
        assert "memory" in limits, f"{name}/{container['name']} has no memory limit"


def test_the_serving_deployment_has_all_three_probes():
    """Each answers a different question, and using one for all three is how
    a rollout either stalls or ships a pod that cannot serve."""
    deployment = _by_kind("Deployment")[0]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(container)
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"


def test_the_rollout_never_drops_below_the_current_replica_count():
    deployment = _by_kind("Deployment")[0]
    strategy = deployment["spec"]["strategy"]["rollingUpdate"]
    assert strategy["maxUnavailable"] == 0


def test_a_disruption_budget_covers_the_serving_pods():
    """Without one, a node drain can take every replica at once."""
    budget = _by_kind("PodDisruptionBudget")[0]
    deployment = _by_kind("Deployment")[0]
    assert (
        budget["spec"]["selector"]["matchLabels"]
        == deployment["spec"]["selector"]["matchLabels"]
    )


def test_every_selector_actually_matches_its_pods():
    """The mistake that produces a Service with no endpoints and a Deployment
    that scales nothing: labels that drifted apart."""
    deployment = _by_kind("Deployment")[0]
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert deployment["spec"]["selector"]["matchLabels"].items() <= labels.items()
    service = _by_kind("Service")[0]
    assert service["spec"]["selector"].items() <= labels.items()


def test_the_autoscaler_points_at_the_deployment_that_exists():
    autoscaler = _by_kind("HorizontalPodAutoscaler")[0]
    target = autoscaler["spec"]["scaleTargetRef"]
    assert target["kind"] == "Deployment"
    assert target["name"] == _by_kind("Deployment")[0]["metadata"]["name"]


def test_the_autoscaler_scales_on_cpu_not_memory():
    """Memory here is ~97% framework and barely moves with load, so a
    memory target would never fire."""
    autoscaler = _by_kind("HorizontalPodAutoscaler")[0]
    names = {m["resource"]["name"] for m in autoscaler["spec"]["metrics"]}
    assert names == {"cpu"}


def test_the_autoscaler_floor_matches_the_disruption_budget():
    """minReplicas below minAvailable is a deadlock: the budget forbids the
    eviction the autoscaler has already made inevitable."""
    autoscaler = _by_kind("HorizontalPodAutoscaler")[0]
    budget = _by_kind("PodDisruptionBudget")[0]
    assert autoscaler["spec"]["minReplicas"] > budget["spec"]["minAvailable"]


def test_batch_jobs_do_not_overlap_themselves():
    for job in _by_kind("CronJob"):
        assert job["spec"]["concurrencyPolicy"] == "Forbid", job["metadata"]["name"]
        assert job["spec"]["jobTemplate"]["spec"]["template"]["spec"]["restartPolicy"] in {
            "Never",
            "OnFailure",
        }


def test_the_export_job_does_not_force_the_ranker_past_its_gate():
    """`--force-ranker` in a nightly CronJob would make overriding the
    offline gate the default behaviour of the system, which is the one thing
    the gate exists to prevent."""
    for job in _by_kind("CronJob"):
        for container in job["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"]:
            assert "--force-ranker" not in container.get("command", [])
            assert "--force-ranker" not in container.get("args", [])


# -- portability, applied to the manifests ------------------------------------


def test_no_manifest_names_a_cloud_provider():
    """The same check `retailgr audit` applies to the source, applied to the
    thing that would actually pin the deployment to one vendor."""
    forbidden = (
        "amazonaws.com",
        "eks.amazonaws.com",
        "azure.com",
        "azurecr.io",
        "gke.io",
        "googleapis.com",
        "container.googleapis.com",
        "csi.storage.gke.io",
        "ebs.csi.aws.com",
    )
    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8").lower()
        for needle in forbidden:
            assert needle not in text, f"{path.name} names {needle}"


def test_every_external_address_comes_from_the_configmap():
    """If an endpoint appeared inline in the Deployment, changing cloud would
    mean editing the workload rather than the configuration."""
    config = _by_kind("ConfigMap")[0]["data"]
    assert any("redis://" in value for value in config.values())
    assert any("9092" in value for value in config.values())

    deployment = _by_kind("Deployment")[0]
    rendered = yaml.safe_dump(deployment)
    for scheme in ("redis://", "http://", "https://"):
        assert scheme not in rendered, f"{scheme} appears inline in the Deployment"


# -- the manifests agree with the measurements --------------------------------


def _serving_resources() -> dict:
    deployment = _by_kind("Deployment")[0]
    return deployment["spec"]["template"]["spec"]["containers"][0]["resources"]


def test_the_memory_request_matches_what_sizing_measured():
    """The point of measuring: the number in the manifest is the number the
    process needs, and it stays that way because this fails if it drifts."""
    import json

    report = Path("artifacts/sizing.json")
    if not report.exists():
        pytest.skip("run `retailgr sizing` to produce artifacts/sizing.json")

    from retailgr.serving.sizing import recommended_resources

    advice = recommended_resources(json.loads(report.read_text(encoding="utf-8")))
    resources = _serving_resources()
    request = int(resources["requests"]["memory"].removesuffix("Mi"))
    limit = int(resources["limits"]["memory"].removesuffix("Mi"))

    # Within 15%: the manifest should track the measurement without being
    # regenerated for every megabyte of interpreter noise.
    assert abs(request - advice["memory_request_mib"]) / advice["memory_request_mib"] < 0.15
    assert abs(limit - advice["memory_limit_mib"]) / advice["memory_limit_mib"] < 0.15


def test_the_memory_limit_leaves_headroom_over_the_request():
    resources = _serving_resources()
    request = int(resources["requests"]["memory"].removesuffix("Mi"))
    limit = int(resources["limits"]["memory"].removesuffix("Mi"))
    assert limit >= request * 1.4


def test_the_startup_probe_budget_clears_the_measured_cold_start():
    """`startupProbe` decides how long a pod gets before the kubelet gives
    up. Under the cold start it never starts at all."""
    import json

    deployment = _by_kind("Deployment")[0]
    probe = deployment["spec"]["template"]["spec"]["containers"][0]["startupProbe"]
    budget = probe["periodSeconds"] * probe["failureThreshold"]

    report = Path("artifacts/sizing.json")
    measured = (
        json.loads(report.read_text(encoding="utf-8"))["cold_start"]["total_s"]
        if report.exists()
        else 1.1
    )
    assert budget >= measured * 3, f"{budget}s budget against a {measured}s cold start"


# -- the manifests refer to things that exist ---------------------------------


def _manifest_commands() -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for name, pod in _pod_specs():
        for container in pod["containers"] + pod.get("initContainers", []):
            command = list(container.get("command", [])) + list(container.get("args", []))
            if command:
                out.append((f"{name}/{container['name']}", command))
    return out


def test_every_command_a_manifest_runs_is_a_real_cli_subcommand():
    """The class of bug a schema validator cannot see.

    An earlier draft of these manifests had an init container running
    `retailgr fetch-bundle`. It validated cleanly against the Kubernetes API
    schema, read plausibly, and referred to a command that has never existed —
    it would have failed on first apply, in a cluster, at the worst possible
    moment. Cross-artifact references are where "looks finished" and "works"
    come apart, so they get their own check.
    """
    import argparse

    from retailgr.cli import build_parser

    parser = build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    known = set(subparsers[0].choices) if subparsers else set()
    assert known, "could not read the CLI's subcommands"

    for where, command in _manifest_commands():
        if command[0] != "retailgr":
            continue
        assert command[1] in known, f"{where} runs `retailgr {command[1]}`, which does not exist"


def test_every_probe_path_is_a_route_the_api_serves():
    """Same class of bug, other direction: a probe pointing at a path the app
    does not serve fails every pod into CrashLoopBackOff."""
    routes = _api_routes()
    for name, pod in _pod_specs():
        for container in pod["containers"]:
            for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
                spec = container.get(probe)
                if not spec or "httpGet" not in spec:
                    continue
                path = spec["httpGet"]["path"]
                assert path in routes, f"{name} {probe} hits {path}, which the API does not serve"


def _stub_service(index_size: int = 900):
    """A service double whose status fields are serialisable.

    A bare MagicMock returns mocks for `model_type` and `vocab_size`, the
    JSON encoder refuses them, and every health endpoint answers 500 — which
    looks like a broken probe and is a broken test.
    """
    from unittest.mock import MagicMock

    service = MagicMock()
    service.bundle.model_version = "test-version"
    service.bundle.manifest.model_type = "hstu"
    service.bundle.manifest.vocab_size = 945
    service.index.size = index_size
    return service


def _api_routes() -> set[str]:
    """The paths the FastAPI app actually exposes."""
    from retailgr.serving.api import create_app

    return {route.path for route in create_app(_stub_service()).routes}


def test_liveness_and_readiness_are_different_endpoints():
    """Pointing both at one handler loses the distinction that matters: a
    dependency blip should take a pod out of rotation, not restart it."""
    deployment = _by_kind("Deployment")[0]
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert (
        container["readinessProbe"]["httpGet"]["path"]
        != container["livenessProbe"]["httpGet"]["path"]
    )


def test_readiness_reports_not_ready_before_the_index_is_built():
    """The behaviour the probe depends on, asserted on the app rather than
    assumed from the manifest."""
    from fastapi.testclient import TestClient

    from retailgr.serving.api import create_app

    service = _stub_service(index_size=0)  # nothing loaded yet
    client = TestClient(create_app(service), raise_server_exceptions=False)
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 200  # alive, just not ready

    service.index.size = 900
    assert client.get("/readyz").status_code == 200
