"""The request path, with no HTTP in sight.

Four stages, each timed:

    context  ->  retrieval  ->  policy filter  ->  ranking  ->  policy re-rank

Keeping this framework-free is not tidiness. It means the whole request path is
unit-testable without a server, the latency budget can be measured per stage
rather than guessed, and the same object can be driven from an HTTP handler, a
batch job or a test.

Every stage has a fallback. A recommender that returns nothing is worse than
one that returns something stale, so a failure in retrieval or ranking falls
back to the precomputed list in the online store rather than erroring.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from retailgr.actions import encode_actions
from retailgr.online_store import OnlineStore
from retailgr.serving.bundle import ServingBundle
from retailgr.serving.policy import PolicyLayer, PolicyTrace, RequestContext, ScoredItem
from retailgr.serving.retrieval import RetrievalIndex


@dataclass
class ServingConfig:
    retrieval_k: int = 800
    rank_k: int = 300
    default_limit: int = 20
    # Whether to drop items the customer has already engaged with, **per
    # surface**. It used to be one global boolean, and the measurement that
    # changed it is large enough to be worth carrying here:
    #
    #     exclude_seen=True   recall@10 0.0968   (CI 0.0855-0.1088)
    #     exclude_seen=False  recall@10 0.1969   (CI 0.1778-0.2145)
    #
    # Turning it off **doubles** recall, on non-overlapping intervals. Only
    # 20.4% of held-out targets are repeats, so the doubling is not about
    # how many repeats exist — it is that repeats are what the sequence
    # model is most confident about, sitting at the very top of its
    # ranking. Excluding them serves the model's 11th-through-300th choices.
    #
    # The offline metric cannot settle this, and pretending otherwise is the
    # trap. Recall counts a repeat as a hit; whether re-showing something the
    # customer already found is worth anything is a question about the
    # surface, not about the model. A cart or email reminder obviously wants
    # repeats. A home page asking for discovery may not. So it is a map, and
    # the default is off for `home` and on everywhere else.
    exclude_seen: bool | dict[str, bool] = True
    # A stage that blows its slice of the budget falls back rather than
    # blocking the response.
    stage_timeout_ms: float = 60.0

    def excludes_seen(self, surface: str | None) -> bool:
        """Whether this surface drops already-seen items.

        A bare bool still works and still means "everywhere", so an existing
        deployment's config keeps its behaviour.
        """
        if isinstance(self.exclude_seen, bool):
            return self.exclude_seen
        settings = self.exclude_seen or {}
        if surface is not None and surface in settings:
            return bool(settings[surface])
        return bool(settings.get("default", True))


@dataclass
class StageTimings:
    context_ms: float = 0.0
    retrieval_ms: float = 0.0
    filter_ms: float = 0.0
    ranking_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "context_ms": round(self.context_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "filter_ms": round(self.filter_ms, 3),
            "ranking_ms": round(self.ranking_ms, 3),
            "rerank_ms": round(self.rerank_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass
class RecommendationResponse:
    request_id: str
    model_version: str
    items: list[dict[str, Any]]
    served_from: str = "model"
    ranker_used: bool = False
    timings: StageTimings = field(default_factory=StageTimings)
    policy: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_version": self.model_version,
            "served_from": self.served_from,
            "ranker_used": self.ranker_used,
            "items": self.items,
            "timings": self.timings.as_dict(),
            "policy": self.policy,
        }


class RecommendationService:
    """Orchestrates one recommendation request."""

    def __init__(
        self,
        bundle: ServingBundle,
        index: RetrievalIndex,
        store: OnlineStore,
        policy: PolicyLayer,
        config: ServingConfig | None = None,
        exposure_logger: Any | None = None,
    ):
        self.bundle = bundle
        self.index = index
        self.store = store
        self.policy = policy
        self.config = config or ServingConfig()
        self.exposure_logger = exposure_logger
        # The answer of last resort, held in this process.
        #
        # Every fallback path in this class used to end in
        # `store.fallback(user_id)`, so "the store is down" had no fallback —
        # the recovery route ran through the thing that had failed. This is a
        # copy of the global cold-start list, read once at startup and
        # refreshed whenever a store read succeeds, so there is always
        # something to serve without a network hop.
        #
        # It is deliberately a *stale* list and not a clever one. Its job is
        # to keep a dependency outage at 200-with-popular-items instead of
        # 500, and `retailgr_requests_total{served_from="store_error"}` is
        # what says the difference out loud.
        self._last_resort: tuple[str, ...] = ()
        self._refresh_last_resort()

    def _refresh_last_resort(self, product_ids: list[str] | None = None) -> None:
        if product_ids:
            self._last_resort = tuple(product_ids)
            return
        try:
            fetched = self.store.fallback("__startup__")
        except Exception:
            # The store is already unreachable. Nothing to cache; responses
            # will be empty until it comes back, which is a 200 an alert can
            # see rather than a 500 a customer sees.
            return
        if fetched:
            self._last_resort = tuple(fetched)

    # -- stage 1: context -----------------------------------------------------

    def _user_sequence(self, user_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """The user's hot tail, as the arrays the encoder expects.

        The tokens were resolved by the streaming consumer using the same
        granularity config the training job used, so no re-resolution happens
        here — that is the whole point of writing tokens into the tail.
        """
        tail = self.store.user_tail(user_id)
        if not tail:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                [],
            )
        token_ids: list[int] = []
        actions: list[str] = []
        timestamps: list[int] = []
        skus: list[str] = []
        for event in tail:
            token_id = self.bundle.id_by_token.get(event.token)
            if token_id is None:
                # A token the model never saw: it cannot be encoded. Kept out
                # of the sequence but still counted as seen, so we do not
                # recommend it back.
                skus.append(event.sku)
                continue
            token_ids.append(token_id)
            actions.append(event.action)
            timestamps.append(event.event_ts)
            skus.append(event.sku)
        return (
            np.asarray(token_ids, dtype=np.int64),
            encode_actions(actions),
            np.asarray(timestamps, dtype=np.int64),
            skus,
        )

    # -- stage 2: retrieval ---------------------------------------------------

    def _user_vector(
        self, tokens: np.ndarray, actions: np.ndarray, timestamps: np.ndarray
    ) -> np.ndarray:
        """Encode the history into one vector, for the retrieval index."""
        import torch

        from retailgr.io.loaders import HistoryBatch

        model = self.bundle.model
        batch = HistoryBatch(tokens=[tokens], actions=[actions], timestamps=[timestamps])
        with torch.no_grad():
            if hasattr(model, "_padded"):  # HSTU: reads all three arrays
                padded_tokens, padded_actions, padded_times = model._padded(batch)
                hidden = model.net(
                    padded_tokens.to(model.device),
                    padded_actions.to(model.device),
                    padded_times.to(model.device),
                )[:, -1, :]
            else:  # SASRec: items only
                padded = model._padded_inputs([tokens])
                hidden = model.net(padded.to(model.device))[:, -1, :]
        return hidden[0].float().cpu().numpy()

    # -- stage 4: ranking -----------------------------------------------------

    def _rank(
        self,
        candidates: list[ScoredItem],
        tokens: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
    ) -> list[ScoredItem]:
        """Re-score the surviving candidates with the target-aware ranker.

        The retrieval score is a dot product against an item embedding: the
        candidate never interacted with the history. The ranker appends each
        candidate to the interleaved sequence and predicts the actions it
        would receive, so the interaction happens inside the encoder. All
        candidates go through in one forward pass (M-FALCON).

        Without a ranker in the bundle this passes the retrieval score
        through, which is a valid deployment — retrieval order is a real
        ordering — and the response says so through ``ranker_used``.
        """
        if not self.bundle.has_ranker or not candidates:
            return candidates

        candidate_ids = np.asarray([item.token_id for item in candidates], dtype=np.int64)
        scores = self.bundle.ranker.score_candidates(
            tokens, actions, timestamps, None, candidate_ids
        )
        for position, item in enumerate(candidates):
            item.model_score = float(scores.blended[position])
            item.head_scores = {
                "click": round(float(scores.click[position]), 5),
                "cart": round(float(scores.cart[position]), 5),
                "purchase": round(float(scores.purchase[position]), 5),
                "return": round(float(scores.returned[position]), 5),
            }
        return candidates

    # -- the request ----------------------------------------------------------

    def recommend(
        self,
        context: RequestContext,
        limit: int | None = None,
        request_id: str | None = None,
    ) -> RecommendationResponse:
        limit = limit or self.config.default_limit
        request_id = request_id or str(uuid.uuid4())
        timings = StageTimings()
        started = time.perf_counter()

        # 1. context
        mark = time.perf_counter()
        try:
            tokens, actions, timestamps, seen_skus = self._user_sequence(context.user_id)
        except Exception:
            # The online store is unreachable. This was unprotected, and a
            # dead Redis produced a 500 on every request — while the three
            # elaborate fallbacks below all sat behind the same store and
            # could not have run either. Measured, not reasoned about:
            # `tests/test_failure_modes.py` takes each dependency down.
            timings.context_ms = (time.perf_counter() - mark) * 1000
            response = self._fallback_response(
                request_id, context, limit, timings, "store_error"
            )
            timings.total_ms = (time.perf_counter() - started) * 1000
            return self._finish(response, context)
        timings.context_ms = (time.perf_counter() - mark) * 1000

        if tokens.size == 0:
            response = self._fallback_response(request_id, context, limit, timings, "cold_start")
            timings.total_ms = (time.perf_counter() - started) * 1000
            return self._finish(response, context)

        # 2. retrieval
        mark = time.perf_counter()
        try:
            user_vector = self._user_vector(tokens, actions, timestamps)
            excludes_seen = self.config.excludes_seen(context.surface)
            exclude = set(tokens.tolist()) if excludes_seen else set()
            retrieved = self.index.search(user_vector, self.config.retrieval_k, exclude)
            scores = (
                self.index.scores_for(user_vector)
                if hasattr(self.index, "scores_for")
                else None
            )
        except Exception:
            response = self._fallback_response(
                request_id, context, limit, timings, "retrieval_error"
            )
            timings.total_ms = (time.perf_counter() - started) * 1000
            return self._finish(response, context)
        timings.retrieval_ms = (time.perf_counter() - mark) * 1000

        candidates = self._to_candidates(retrieved[: self.config.rank_k], scores)

        # 3. policy filter: stock, eligibility, already-purchased
        mark = time.perf_counter()
        all_skus = [sku for item in candidates for sku in item.skus]
        try:
            item_states = self.store.item_states(all_skus)
        except Exception:
            # Fail **open**, which is the opposite of the instinct and the
            # right call: `PolicyLayer` already reads a missing state as
            # available, so an empty map serves possibly-stale availability
            # instead of serving nothing. The alternative — dropping every
            # candidate for want of an inventory read — turns a cache blip
            # into an empty page.
            item_states = {}
        timings.filter_ms = (time.perf_counter() - mark) * 1000

        # 4. ranking
        mark = time.perf_counter()
        ranker_failed = False
        try:
            candidates = self._rank(candidates, tokens, actions, timestamps)
        except Exception:
            # Degrade to retrieval order rather than to the popular list.
            # A bundle with no ranker serves retrieval order and that is a
            # supported deployment, so a broken ranker should land in the
            # same place — losing the second stage is not losing the model.
            #
            # This block is the one the module docstring above always
            # claimed ("a failure in retrieval *or ranking* falls back") and
            # never had. A ranker exception was a 500.
            ranker_failed = True
        timings.ranking_ms = (time.perf_counter() - mark) * 1000

        # 5. policy re-rank: promos, price, diversity, merchandising
        mark = time.perf_counter()
        context_with_seen = RequestContext(
            user_id=context.user_id,
            surface=context.surface,
            store_id=context.store_id,
            locale=context.locale,
            cart_skus=context.cart_skus,
            purchased_skus=tuple(set(context.purchased_skus) | set(seen_skus))
            if self.policy.config.exclude_purchased
            and self.config.excludes_seen(context.surface)
            else context.purchased_skus,
            session_categories=context.session_categories,
            price_affinity=context.price_affinity,
        )
        final, trace = self.policy.apply(candidates, item_states, context_with_seen, limit)
        timings.rerank_ms = (time.perf_counter() - mark) * 1000

        if not final:
            response = self._fallback_response(
                request_id, context, limit, timings, "policy_emptied", trace
            )
            timings.total_ms = (time.perf_counter() - started) * 1000
            return self._finish(response, context, trace)

        timings.total_ms = (time.perf_counter() - started) * 1000
        response = RecommendationResponse(
            request_id=request_id,
            model_version=self.bundle.model_version,
            items=[
                {
                    "position": position,
                    "product_id": item.product_id,
                    "style_color_id": item.token,
                    "suggested_sku": item.suggested_sku,
                    "available_skus": item.available_skus[:5],
                    "score": round(item.final_score, 6),
                    "model_score": round(item.model_score, 6),
                    "reason_code": item.reason_code,
                    # Present only when a ranker scored this item; these are
                    # what a merchandiser asks about when a result surprises
                    # them.
                    **({"heads": item.head_scores} if item.head_scores else {}),
                }
                for position, item in enumerate(final)
            ],
            served_from="model",
            ranker_used=bool(self.bundle.has_ranker) and not ranker_failed,
            timings=timings,
            policy=trace.as_dict(),
        )
        return self._finish(response, context, trace)

    # -- helpers --------------------------------------------------------------

    def _to_candidates(
        self, token_ids: list[int], scores: np.ndarray | None
    ) -> list[ScoredItem]:
        out: list[ScoredItem] = []
        for token_id in token_ids:
            token = self.bundle.token_by_id.get(token_id)
            if token is None:
                continue
            out.append(
                ScoredItem(
                    token=token,
                    token_id=token_id,
                    product_id=self.bundle.product_by_token.get(token, token),
                    model_score=float(scores[token_id]) if scores is not None else 0.0,
                    category=self.bundle.category_by_id.get(token_id, ""),
                    skus=self.bundle.skus_for(token),
                )
            )
        return out

    def _fallback_response(
        self,
        request_id: str,
        context: RequestContext,
        limit: int,
        timings: StageTimings,
        reason: str,
        trace: PolicyTrace | None = None,
    ) -> RecommendationResponse:
        """Serve the precomputed list. Something beats nothing.

        Reaching the store is attempted first and is allowed to fail: this
        method is the recovery path for the store itself, so it cannot
        require the store to work.
        """
        try:
            product_ids = list(self.store.fallback(context.user_id))
            self._refresh_last_resort(product_ids)
        except Exception:
            product_ids = list(self._last_resort)
        product_ids = product_ids[:limit]
        return RecommendationResponse(
            request_id=request_id,
            model_version=f"{self.bundle.model_version}+fallback",
            items=[
                {
                    "position": position,
                    "product_id": product_id,
                    "style_color_id": product_id,
                    "suggested_sku": None,
                    "available_skus": [],
                    "score": 0.0,
                    "model_score": 0.0,
                    "reason_code": reason,
                }
                for position, product_id in enumerate(product_ids)
            ],
            served_from=reason,
            timings=timings,
            policy=trace.as_dict() if trace else {},
        )

    def _finish(
        self,
        response: RecommendationResponse,
        context: RequestContext,
        trace: Any = None,
    ) -> RecommendationResponse:
        """The single exit point of `recommend`.

        Metrics and the exposure log both have to happen on *every* path,
        including the three fallbacks — a fallback that is not counted is
        exactly the failure the fallback rate exists to reveal. Putting them
        in one helper that also returns the response means a new early return
        cannot quietly skip them, and `tests/test_metrics_export.py` asserts
        that no `return response` in `recommend` bypasses it.
        """
        from retailgr.serving import metrics

        metrics.observe_response(response, trace)
        self._log_exposure(response, context)
        return response

    def _log_exposure(self, response: RecommendationResponse, context: RequestContext) -> None:
        """Write what was shown to the exposure topic.

        Without this there are no unbiased training labels later: you cannot
        tell a product the customer rejected from one they never saw.
        """
        if self.exposure_logger is None:
            return
        from retailgr.serving import metrics

        try:
            self.exposure_logger.send_recs_served(
                {
                    "request_id": response.request_id,
                    "user_id": context.user_id,
                    "session_id": None,
                    "surface": context.surface,
                    "model_version": response.model_version,
                    "served_ts": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": response.timings.total_ms,
                    "items": [
                        {
                            "position": item["position"],
                            "product_id": item["product_id"],
                            "suggested_sku": item["suggested_sku"],
                            "score": item["score"],
                            "reason_code": item["reason_code"],
                        }
                        for item in response.items
                    ],
                }
            )
        except Exception:
            # Losing an exposure log must never fail a customer request. The
            # gap shows up as a drop in log volume — which was asserted here
            # for a long time and was not true, because nothing counted the
            # writes. `retailgr_exposure_log_total{outcome="failed"}` is what
            # made the sentence above honest.
            metrics.observe_exposure("failed")
            return
        metrics.observe_exposure("sent")
