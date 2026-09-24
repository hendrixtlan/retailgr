"""Take each dependency down and assert what the caller gets.

The module docstring of `serving/service.py` has always said:

    Every stage has a fallback. A recommender that returns nothing is worse
    than one that returns something stale, so a failure in retrieval or
    ranking falls back to the precomputed list in the online store rather
    than erroring.

Read from the code it looked true — three fallbacks, one instrumented exit,
an AST test proving every `return` goes through it. Measured by taking each
dependency down, two thirds of it was false:

* **A dead online store produced a 500 on every request.** Three store calls
  in the request path were unprotected, and worse, the fallback for
  *everything else* was itself a store call — so the recovery route ran
  through the component that had failed. The elaborate fallback machinery
  could not have run in the one outage it most needed to.
* **A ranker exception was a 500**, despite the sentence above naming
  ranking explicitly. `_rank` had no try/except at all.

Both are fixed and both are pinned here. The general lesson is the one worth
keeping: fallback code is only exercised in the outage it was written for,
so reading it proves nothing. This file is the only thing that makes those
claims checkable, which is why it injects failures rather than asserting
that handlers exist.

What is deliberately *not* claimed: that a dead store still produces good
recommendations. It produces a stale popular list, and only when the process
was healthy long enough to have cached one. A process that starts against a
dead store serves an empty 200 — which is a different failure, visible to
`RetailGREmptyResponses`, and much better than a 500.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from retailgr.online_store import InMemoryOnlineStore, ItemState, TailEvent
from retailgr.serving.policy import PolicyConfig, PolicyLayer, RequestContext
from retailgr.serving.retrieval import ExactRetrievalIndex
from retailgr.serving.service import RecommendationService, ServingConfig


class Down(Exception):
    """What a dead dependency raises."""


def _store(*broken: str) -> InMemoryOnlineStore:
    """A warm store whose named methods raise."""
    store = InMemoryOnlineStore()
    store.append_event("U1", TailEvent(sku="S1", token="S1", action="view", event_ts=1.0))
    store.append_event("U1", TailEvent(sku="S2", token="S2", action="click", event_ts=2.0))
    for sku in ("S1", "S2", "S3", "S4"):
        store.put_item_state(ItemState(sku=sku, available=True, stock_by_location={"L": 5}))
    store.put_global_fallback(["P1", "P2", "P3"])
    for name in broken:
        setattr(store, name, MagicMock(side_effect=Down(f"{name} is down")))
    return store


def _service(
    store: InMemoryOnlineStore,
    *,
    index=None,
    ranker_raises: bool = False,
    encoder_raises: bool = False,
) -> RecommendationService:
    bundle = MagicMock()
    bundle.model_version = "test"
    bundle.manifest.model_type = "hstu"
    bundle.manifest.vocab_size = 10
    bundle.has_ranker = ranker_raises
    bundle.id_by_token = {f"S{i}": i for i in range(1, 5)}
    bundle.token_by_id = {i: f"S{i}" for i in range(1, 5)}
    bundle.skus_by_token = {i: [f"S{i}"] for i in range(1, 5)}
    bundle.product_by_token = {i: f"P{i}" for i in range(1, 5)}
    # A real list, not a MagicMock: the policy checks stock per SKU, and a
    # mock here makes every candidate look out of stock, which would have
    # every test below passing for the wrong reason.
    bundle.skus_for = lambda token: [str(token)]
    bundle.product_for = lambda token: f"P{token}"
    bundle.ranker = MagicMock() if ranker_raises else None
    if ranker_raises:
        bundle.ranker.score_candidates.side_effect = Down("ranker is down")

    service = RecommendationService(
        bundle=bundle,
        index=index if index is not None else ExactRetrievalIndex(np.eye(5, 4, dtype=np.float32)),
        store=store,
        policy=PolicyLayer(PolicyConfig()),
        config=ServingConfig(),
    )
    service._user_vector = (
        MagicMock(side_effect=Down("encoder is down"))
        if encoder_raises
        else (lambda *a, **k: np.ones(4, dtype=np.float32))
    )
    return service


def _ask(service: RecommendationService):
    return service.recommend(RequestContext(user_id="U1", surface="home"), limit=5)


# -- the store: the dependency every other fallback used to run through ------


def test_a_dead_online_store_answers_instead_of_raising():
    """This was a 500 on every request.

    `_user_sequence` is a store read and it was not wrapped. Nothing else
    mattered: the three fallbacks below it were all store reads too.
    """
    response = _ask(_service(_store("user_tail", "item_states", "fallback")))
    assert response.served_from == "store_error"


def test_a_store_that_dies_after_startup_still_serves_items():
    """The realistic shape of the outage, and the one the cached list is for.

    A process that has been up serves a stale popular list. A process that
    started against a dead store has nothing cached and serves an empty 200 —
    honest, alertable, and still not a 500.
    """
    store = _store()
    service = _service(store)  # startup warms the last-resort list
    for name in ("user_tail", "item_states", "fallback"):
        setattr(store, name, MagicMock(side_effect=Down(f"{name} is down")))

    response = _ask(service)
    assert response.served_from == "store_error"
    assert len(response.items) == 3, "the cached cold-start list was not used"


def test_the_fallback_path_does_not_require_the_store_it_is_recovering_from():
    """The circularity that made the whole fallback design ineffective.

    Asserted on behaviour: break only `fallback`, leave the rest healthy,
    and a request that needs the fallback must still answer.
    """
    store = _store()
    service = _service(store)
    store.fallback = MagicMock(side_effect=Down("fallback is down"))
    store.user_tail = MagicMock(side_effect=Down("user_tail is down"))

    response = _ask(service)
    assert response.served_from == "store_error"
    assert len(response.items) == 3


def test_a_missing_inventory_read_fails_open_rather_than_emptying_the_page():
    """Deliberately the opposite of the safe-looking choice.

    `PolicyLayer` already reads a missing item state as available, so an
    empty map serves possibly-stale availability. Dropping every candidate
    for want of an inventory read would turn a cache blip into an empty page.
    """
    response = _ask(_service(_store("item_states")))
    assert response.served_from == "model"
    assert response.items


# -- retrieval, the encoder and the GPU --------------------------------------


def test_a_throwing_index_falls_back():
    index = MagicMock()
    index.search.side_effect = Down("ann is down")
    index.size = 5
    response = _ask(_service(_store(), index=index))
    assert response.served_from == "retrieval_error"
    assert response.items


def test_an_encoder_failure_falls_back():
    """Covers a GPU that has gone away mid-process: `torch.device("cuda")`
    does not raise when it is constructed, only when an operation runs, and
    that operation is inside the retrieval stage."""
    response = _ask(_service(_store(), encoder_raises=True))
    assert response.served_from == "retrieval_error"
    assert response.items


def test_auto_device_selection_does_not_require_a_gpu():
    from retailgr.models.sasrec import resolve_device

    assert resolve_device("auto").type in {"cuda", "cpu"}
    assert resolve_device("cpu").type == "cpu"


# -- the ranker --------------------------------------------------------------


def test_a_throwing_ranker_degrades_to_retrieval_order_not_to_a_500():
    """The fallback the module docstring promised and the code never had.

    Degrading to retrieval order and not to the popular list is the point: a
    bundle with no ranker serves retrieval order and that is a supported
    deployment, so a broken ranker should land in the same place. Losing the
    second stage is not losing the model.
    """
    response = _ask(_service(_store(), ranker_raises=True))
    assert response.served_from == "model", response.served_from
    assert response.items


def test_a_failed_ranker_is_reported_as_not_used():
    """Otherwise the response claims a ranker scored it and the exposure log
    records a ranking that never happened."""
    response = _ask(_service(_store(), ranker_raises=True))
    assert response.ranker_used is False


# -- the stream and the lakehouse --------------------------------------------


def test_a_dead_exposure_logger_does_not_fail_the_request():
    """Kafka being down must cost the log, not the customer."""
    logger = MagicMock()
    logger.send_recs_served.side_effect = Down("kafka is down")
    service = _service(_store())
    service.exposure_logger = logger

    response = _ask(service)
    assert response.items
    assert logger.send_recs_served.called


def test_the_request_path_never_touches_the_lakehouse():
    """Iceberg or its catalog being down is a batch problem by construction.

    Asserted structurally, because the guarantee is "serving does not import
    this" and a behavioural test would only prove it did not happen to on one
    request. A stale bundle is the real consequence, and
    `RetailGRBundleStale` is what reports it.
    """
    import ast
    from pathlib import Path

    forbidden = {"retailgr.io.tables", "retailgr.io.iceberg", "retailgr.spark_session"}
    # The request path only. `bench.py` and `sizing.py` live in this package
    # and are offline tools that bootstrap from the lakehouse on purpose —
    # exempting them by name keeps the check about the thing it is for.
    request_path = {
        "service.py", "api.py", "policy.py", "retrieval.py",
        "bundle.py", "factory.py", "metrics.py", "tracing.py",
    }
    offenders: list[str] = []
    for path in sorted(Path("src/retailgr/serving").rglob("*.py")):
        if path.name not in request_path:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name in forbidden:
                    offenders.append(f"{path}:{node.lineno} imports {name}")
    assert not offenders, offenders

    # And the stronger check: importing the request path must not drag a
    # Spark session in behind it. A lazy import inside a function would slip
    # past the AST walk above and still couple a serving pod to the cluster.
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import retailgr.serving.service, retailgr.serving.api, sys;"
            "print('pyspark' in sys.modules)",
        ],
        capture_output=True, text=True, cwd=".",
    )
    assert probe.stdout.strip() == "False", probe.stdout + probe.stderr


# -- the invariant behind all of the above -----------------------------------


@pytest.mark.parametrize(
    "broken",
    [
        (),
        ("user_tail",),
        ("item_states",),
        ("fallback",),
        ("user_tail", "item_states", "fallback"),
    ],
    ids=["healthy", "user_tail", "item_states", "fallback", "everything"],
)
def test_no_store_failure_escapes_as_an_exception(broken):
    """The invariant, parametrised, so a fourth store call added later is
    covered the moment someone adds it to this list — and so that the claim
    is about the request path rather than about three specific handlers."""
    response = _ask(_service(_store(*broken)))
    assert response.request_id
    assert response.served_from in {
        "model",
        "cold_start",
        "retrieval_error",
        "policy_emptied",
        "store_error",
    }


def test_every_store_call_in_the_request_path_is_guarded():
    """Structural, and the reason this file exists rather than three tests.

    A store call added to `recommend` without a guard reintroduces exactly
    the defect that was here: it will look fine in every test, in every
    staging environment, and in production right up until the store blips.
    """
    import ast
    import inspect

    source = inspect.cleandoc(inspect.getsource(RecommendationService.recommend))
    tree = ast.parse(source)
    function = tree.body[0]

    guarded: set[int] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                guarded.add(id(child))

    unguarded: list[str] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        # `self.store.<anything>(...)` or `self._user_sequence(...)`, which
        # is a store read wearing a different name.
        is_store = (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == "store"
        ) or func.attr == "_user_sequence"
        if is_store and id(node) not in guarded:
            unguarded.append(f"line {node.lineno}: .{func.attr}()")
    assert not unguarded, (
        f"unguarded store calls in recommend(): {unguarded}. A dead store "
        "must degrade, not raise — the fallbacks are store reads themselves."
    )
