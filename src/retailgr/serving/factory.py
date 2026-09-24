"""Assemble a ``RecommendationService`` from config and an exported bundle.

Kept apart from ``api.py`` so that everything except the HTTP edge — the
latency benchmark, tests, a batch scorer — can build a service without
FastAPI or pydantic installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retailgr.config import Config
from retailgr.online_store import build_online_store
from retailgr.serving.bundle import load_bundle
from retailgr.serving.policy import PolicyConfig, PolicyLayer
from retailgr.serving.retrieval import build_index
from retailgr.serving.service import RecommendationService, ServingConfig


def build_service(
    cfg: Config,
    bundle_dir: str | Path,
    store: Any | None = None,
    exposure_logger: Any | None = None,
) -> RecommendationService:
    """Build the service the API and the benchmark both run."""
    bundle = load_bundle(Path(bundle_dir), device=str(cfg.get("serving.device", "cpu")))
    index = build_index(
        bundle.item_embeddings, kind=str(cfg.get("serving.retrieval_index", "exact"))
    )
    store = store if store is not None else build_online_store(cfg)
    policy = PolicyLayer(PolicyConfig.from_dict(cfg.get("policy")))
    serving_config = ServingConfig(
        retrieval_k=int(cfg.get("serving.retrieval_k", 800)),
        rank_k=int(cfg.get("serving.rank_k", 300)),
        default_limit=int(cfg.get("serving.default_limit", 20)),
        # bool or a per-surface map; ServingConfig.excludes_seen reads both.
        exclude_seen=cfg.get("serving.exclude_seen", True),
    )
    # Publish which bundle is serving before the first request, so a scrape
    # during cold start still says what this pod is.
    from retailgr.serving import metrics

    metrics.record_build(bundle.manifest, ranker_served=bundle.has_ranker)

    return RecommendationService(
        bundle=bundle,
        index=index,
        store=store,
        policy=policy,
        config=serving_config,
        exposure_logger=exposure_logger,
    )
