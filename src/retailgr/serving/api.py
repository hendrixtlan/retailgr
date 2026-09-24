"""The HTTP edge: a thin adapter over ``RecommendationService``.

Thin on purpose. Everything worth testing lives in ``service.py`` and
``policy.py``; this module only parses requests, calls one method, and
serialises the result. Swapping FastAPI for gRPC is a rewrite of this file and
nothing else.

Note the absence of ``from __future__ import annotations`` here: FastAPI reads
the handler's annotations at import time to find the request body, and with
postponed evaluation it sees a string it cannot resolve and treats the body as
a query parameter. The request model is defined at module level for the same
reason.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from retailgr.config import Config
from retailgr.serving import tracing
from retailgr.serving.factory import build_service
from retailgr.serving.policy import RequestContext
from retailgr.serving.service import RecommendationService

__all__ = ["RecommendationRequest", "build_service", "create_app", "serve"]


class RecommendationRequest(BaseModel):
    """One recommendation request."""

    user_id: str | None = None
    device_id: str | None = Field(
        default=None, description="Used before sign-in; stitched to user_id at login."
    )
    session_id: str | None = None
    surface: str = Field(default="home", description="home | pdp | cart | email")
    store_id: str | None = Field(
        default=None, description="Stock is checked at this location."
    )
    locale: str | None = None
    cart_skus: list[str] = Field(default_factory=list)
    purchased_skus: list[str] = Field(default_factory=list)
    session_categories: list[str] = Field(default_factory=list)
    price_affinity: float | None = Field(
        default=None, description="0-1; where this customer usually sits in the price band."
    )
    limit: int = Field(default=20, ge=1, le=100)


def create_app(service: RecommendationService):
    """A FastAPI app wrapping ``service``."""
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="RetailGR recommendations",
        version=service.bundle.model_version,
        description=(
            "Retrieval, ranking and a deterministic policy layer over an "
            "HSTU or SASRec encoder."
        ),
    )

    def _status() -> dict:
        return {
            "status": "ok",
            "model_version": service.bundle.model_version,
            "model_type": service.bundle.manifest.model_type,
            "vocab_size": service.bundle.manifest.vocab_size,
            "index_size": service.index.size,
        }

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness: is this process still the process?

        Deliberately cheap and deliberately *not* a dependency check. A
        liveness probe that fails when Redis is briefly unreachable restarts
        every replica at once and turns a dependency blip into an outage —
        the pod was fine, the thing it talks to was not, and killing the pod
        helps neither.
        """
        return _status()

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        """Readiness: can this replica answer a request right now?

        A different question from liveness, which is why it is a different
        endpoint. During the measured ~1.1 s cold start the process is alive
        and cannot serve; taking it out of the Service until the bundle and
        the index are up is the difference between a rolling deploy and a
        rolling outage.
        """
        ready = bool(service.bundle is not None and service.index.size > 0)
        return JSONResponse(
            {**_status(), "ready": ready},
            status_code=200 if ready else 503,
        )

    @app.get("/health", include_in_schema=False)
    def health() -> dict:
        """The original name, kept so an existing probe does not break."""
        return _status()

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        """Prometheus exposition, in text format.

        Plain text and not JSON because that is what a scraper reads, and
        `Response` rather than a return value because FastAPI would otherwise
        JSON-encode the string and hand Prometheus a quoted blob.
        """
        from retailgr.serving import metrics

        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)

    @app.get("/v1/model")
    def model_info() -> dict:
        """The manifest, so a served response can be traced to its training run.

        Scrubbed on the way out even though `export_bundle` already scrubs on
        the way in. Those are two different claims: this handler returns
        `asdict(service.bundle.manifest)`, which is an in-memory object, not
        the bytes of the file. A bundle built in-process, or loaded from a
        manifest written by an older version of the exporter, reaches here
        without having crossed the write boundary at all.

        This endpoint is why `privacy.py` exists. It used to return the
        manifest verbatim, the manifest carried `ranker_metrics`, and the
        discrimination diagnostic had kept the `user_ids` it needs to pair
        its samples — 200 real customer ids on an unauthenticated route.
        """
        from dataclasses import asdict

        from retailgr import privacy

        return privacy.scrub(asdict(service.bundle.manifest))

    @app.post("/v1/recommendations")
    def recommendations(body: RecommendationRequest, request: Request, response: Response) -> dict:
        identity = body.user_id or body.device_id
        if not identity:
            raise HTTPException(status_code=422, detail="user_id or device_id is required")

        # The caller's trace, if it brought one. This handler used to drop it
        # and let the service mint a uuid, which put every served list in the
        # exposure topic under an id nobody upstream had seen.
        trace = tracing.context_from_headers(request.headers)

        context = RequestContext(
            user_id=identity,
            surface=body.surface,
            store_id=body.store_id,
            locale=body.locale,
            cart_skus=tuple(body.cart_skus),
            purchased_skus=tuple(body.purchased_skus),
            session_categories=tuple(c.lower() for c in body.session_categories),
            price_affinity=body.price_affinity,
        )
        result = service.recommend(
            context, limit=body.limit, request_id=trace.request_id
        )
        # Echoed so a caller that did *not* send an id can still correlate,
        # and so a proxy in front of this can log the same value.
        response.headers[tracing.REQUEST_ID_HEADER] = result.request_id
        response.headers[tracing.TRACEPARENT_HEADER] = tracing.format_traceparent(trace)
        return result.as_dict()

    return app


def serve(
    cfg: Config,
    bundle_dir: "str | Path",
    host: str = "127.0.0.1",
    port: int = 8080,
    store: Any = None,
) -> None:  # pragma: no cover - runs a server
    import uvicorn

    service = build_service(cfg, bundle_dir, store=store)
    uvicorn.run(create_app(service), host=host, port=port, log_level="warning")
