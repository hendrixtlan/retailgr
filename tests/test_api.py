"""Tests for the HTTP edge and the request path end to end.

These build a tiny bundle on disk, fill an in-memory online store, and drive
the real FastAPI app. No Spark, no broker, no network — but the same
``RecommendationService`` and the same bundle format the served API uses, so a
break in the contract between export and serve shows up here.
"""

from __future__ import annotations

import numpy as np
import pytest

from retailgr.online_store import InMemoryOnlineStore, ItemState, TailEvent
from retailgr.serving.bundle import BundleManifest, export_bundle, load_bundle
from retailgr.serving.policy import PolicyConfig, PolicyLayer, RequestContext
from retailgr.serving.retrieval import build_index
from retailgr.serving.service import RecommendationService, ServingConfig

VOCAB = 12
DIM = 8


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    """Export a real bundle from a real (tiny, untrained) model."""
    from retailgr.models.hstu import HSTUModel

    model_config = {
        "hidden_dim": DIM,
        "num_blocks": 1,
        "num_heads": 2,
        "max_len": 8,
        "epochs": 1,
        "dropout": 0.0,
    }
    model = HSTUModel(VOCAB, model_config)

    token_by_id = {i: f"P{i}-black" for i in range(1, VOCAB)}
    category_by_id = {i: ("apparel" if i % 2 else "grocery") for i in range(1, VOCAB)}
    skus_by_token = {token: [f"{token}-S", f"{token}-M"] for token in token_by_id.values()}
    product_by_token = {token: token.split("-")[0] for token in token_by_id.values()}

    directory = tmp_path_factory.mktemp("bundle")
    export_bundle(
        directory,
        model=model,
        manifest=BundleManifest(
            model_version="test-v1",
            model_type="hstu",
            variant="config",
            dataset="synthetic",
            vocab_size=VOCAB,
            embedding_dim=DIM,
            max_len=8,
            model_config=model_config,
        ),
        token_by_id=token_by_id,
        category_by_id=category_by_id,
        skus_by_token=skus_by_token,
        product_by_token=product_by_token,
    )
    return directory


@pytest.fixture
def store():
    store = InMemoryOnlineStore(tail_length=10)
    for user in ("U1", "U2"):
        for index in range(1, 6):
            store.append_event(
                user,
                TailEvent(
                    sku=f"P{index}-black-M",
                    token=f"P{index}-black",
                    action="view",
                    event_ts=1_700_000_000 + index * 600,
                ),
            )
    for index in range(1, VOCAB):
        for size in ("S", "M"):
            store.put_item_state(
                ItemState(
                    sku=f"P{index}-black-{size}",
                    price=10.0 + index,
                    list_price=10.0 + index,
                    stock_by_location={"store-1": 5},
                )
            )
    store.put_global_fallback(["P9", "P10"])
    return store


@pytest.fixture
def service(bundle_dir, store):
    bundle = load_bundle(bundle_dir)
    return RecommendationService(
        bundle=bundle,
        index=build_index(bundle.item_embeddings),
        store=store,
        policy=PolicyLayer(PolicyConfig(max_per_category=10)),
        config=ServingConfig(retrieval_k=VOCAB, rank_k=VOCAB, default_limit=5),
    )


# -- the bundle round trip ----------------------------------------------------


def test_bundle_round_trips(bundle_dir):
    bundle = load_bundle(bundle_dir)
    assert bundle.manifest.model_version == "test-v1"
    assert bundle.item_embeddings.shape == (VOCAB, DIM)
    assert bundle.id_by_token["P1-black"] == 1
    # A token resolves to purchasable SKUs, which is what the policy needs.
    assert bundle.skus_for("P1-black") == ["P1-black-S", "P1-black-M"]


def test_bundle_embeddings_match_the_loaded_model(bundle_dir):
    """The exported index and the loaded encoder must be the same weights, or
    retrieval and ranking disagree about what an item is."""
    bundle = load_bundle(bundle_dir)
    live = bundle.model.net.item_embedding.weight.detach().cpu().numpy()
    np.testing.assert_allclose(bundle.item_embeddings, live, rtol=1e-6)


# -- the request path ---------------------------------------------------------


def test_recommend_returns_ranked_serveable_items(service):
    response = service.recommend(RequestContext("U1", store_id="store-1"), limit=5)
    assert response.served_from == "model"
    assert len(response.items) == 5
    scores = [item["score"] for item in response.items]
    assert scores == sorted(scores, reverse=True)
    for item in response.items:
        assert item["suggested_sku"]
        assert item["available_skus"]


def test_seen_items_are_not_recommended_back(service):
    """The user's own history must not come back as a recommendation when
    exclude_seen is on."""
    response = service.recommend(RequestContext("U1", store_id="store-1"), limit=5)
    returned = {item["style_color_id"] for item in response.items}
    seen = {f"P{i}-black" for i in range(1, 6)}
    assert not (returned & seen)


def test_unknown_user_gets_the_fallback_not_an_error(service):
    response = service.recommend(RequestContext("nobody"), limit=5)
    assert response.served_from == "cold_start"
    assert [item["product_id"] for item in response.items] == ["P9", "P10"]
    assert response.model_version.endswith("+fallback")


def test_out_of_stock_everywhere_falls_back_rather_than_returning_nothing(service, store):
    for index in range(1, VOCAB):
        for size in ("S", "M"):
            store.put_item_state(
                ItemState(sku=f"P{index}-black-{size}", stock_by_location={"store-1": 0})
            )
    response = service.recommend(RequestContext("U1", store_id="store-1"), limit=5)
    assert response.served_from == "policy_emptied"
    assert response.items  # the fallback list, not an empty page


def test_timings_are_recorded_for_every_stage(service):
    response = service.recommend(RequestContext("U1", store_id="store-1"), limit=5)
    timings = response.timings.as_dict()
    assert timings["total_ms"] > 0
    assert timings["retrieval_ms"] > 0
    # The sum of the stages cannot exceed the measured total.
    stages = sum(value for name, value in timings.items() if name != "total_ms")
    assert stages <= timings["total_ms"] + 1.0


def test_exposure_log_records_what_was_served(service):
    """Without this log there are no unbiased labels later."""
    from retailgr.streaming.broker import InMemoryBroker
    from retailgr.streaming.producer import EventProducer
    from retailgr.streaming.schemas import TOPIC_RECS_SERVED

    broker = InMemoryBroker()
    service.exposure_logger = EventProducer(broker)
    response = service.recommend(RequestContext("U1", store_id="store-1"), limit=5)

    logged = broker.records(TOPIC_RECS_SERVED)
    assert len(logged) == 1
    payload = logged[0].value
    assert payload["request_id"] == response.request_id
    assert payload["model_version"] == response.model_version
    assert len(payload["items"]) == len(response.items)
    assert payload["latency_ms"] > 0


def test_a_failing_exposure_logger_never_fails_the_request(service):
    class Broken:
        def send_recs_served(self, payload):
            raise RuntimeError("kafka is down")

    service.exposure_logger = Broken()
    response = service.recommend(RequestContext("U1", store_id="store-1"), limit=5)
    assert response.items


# -- HTTP ---------------------------------------------------------------------


@pytest.fixture
def client(service):
    from fastapi.testclient import TestClient

    from retailgr.serving.api import create_app

    return TestClient(create_app(service))


def test_health_reports_the_served_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-v1"
    assert body["index_size"] == VOCAB


def test_model_endpoint_exposes_the_manifest(client):
    """A served response must be traceable to the run that produced it."""
    body = client.get("/v1/model").json()
    assert body["model_type"] == "hstu"
    assert body["variant"] == "config"
    assert body["created_at"]


def test_recommendations_endpoint_returns_items(client):
    response = client.post(
        "/v1/recommendations",
        json={"user_id": "U1", "store_id": "store-1", "limit": 3, "surface": "home"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 3
    assert body["request_id"]
    assert body["timings"]["total_ms"] > 0


def test_request_without_an_identity_is_rejected(client):
    response = client.post("/v1/recommendations", json={"limit": 5})
    assert response.status_code == 422


def test_out_of_range_limit_is_rejected(client):
    assert client.post("/v1/recommendations", json={"user_id": "U1", "limit": 0}).status_code == 422
    assert (
        client.post("/v1/recommendations", json={"user_id": "U1", "limit": 500}).status_code == 422
    )


def test_device_id_works_when_there_is_no_user_id(client):
    response = client.post("/v1/recommendations", json={"device_id": "D1", "limit": 2})
    assert response.status_code == 200
