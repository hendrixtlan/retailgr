"""The policy layer: the model proposes, the business disposes.

Everything here is deterministic and config-driven, deliberately. A learned
model cannot be trusted to respect a hard constraint — "never show what we
cannot ship" is not a soft preference to be traded off against relevance — so
constraints are enforced outside the model, where they can be read, audited
and changed without retraining.

Two kinds of rule, applied in this order:

1. **Filters** remove candidates outright: out of stock at the customer's
   location, not eligible in their region, already bought and not repeatable.
   A filter is a veto.
2. **Adjustments** reorder what survives: promo boost, price affinity,
   category caps for diversity, pinned and excluded slots. An adjustment is
   bounded, so relevance still decides most of the order.

Every drop and every boost is recorded in a ``PolicyTrace``, because "why did
the model show me this" is a question merchandisers ask daily and a system
that cannot answer it does not get trusted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from retailgr.online_store import ItemState


@dataclass
class PolicyConfig:
    """The knobs. Loaded from ``configs/policy.yaml``."""

    # Filters
    require_in_stock: bool = True
    exclude_purchased: bool = True
    excluded_products: tuple[str, ...] = ()
    # Adjustments, all multiplicative on the model score.
    promo_boost: float = 1.15
    price_affinity_weight: float = 0.10
    # Diversity: at most this many items per category in the final list.
    max_per_category: int = 3
    # Merchandising: products pinned to the front, in order.
    pinned_products: tuple[str, ...] = ()
    # Keep a boost from ever outranking a much more relevant item.
    max_total_boost: float = 1.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PolicyConfig:
        data = dict(data or {})
        for key in ("excluded_products", "pinned_products"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        return cls(**data)


@dataclass
class ScoredItem:
    """A candidate as it moves through the policy layer."""

    token: str
    token_id: int
    product_id: str
    model_score: float
    category: str = ""
    skus: list[str] = field(default_factory=list)
    suggested_sku: str | None = None
    available_skus: list[str] = field(default_factory=list)
    final_score: float = 0.0
    reason_code: str = "relevance"
    boosts: dict[str, float] = field(default_factory=dict)
    # Filled by the ranker: P(click), P(cart), P(purchase), P(return).
    head_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class PolicyTrace:
    """Why the list looks the way it does."""

    candidates_in: int = 0
    dropped_out_of_stock: int = 0
    dropped_ineligible: int = 0
    dropped_purchased: int = 0
    dropped_excluded: int = 0
    dropped_no_sku: int = 0
    capped_by_category: int = 0
    promoted: int = 0
    pinned: int = 0
    items_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates_in": self.candidates_in,
            "dropped_out_of_stock": self.dropped_out_of_stock,
            "dropped_ineligible": self.dropped_ineligible,
            "dropped_purchased": self.dropped_purchased,
            "dropped_excluded": self.dropped_excluded,
            "dropped_no_sku": self.dropped_no_sku,
            "capped_by_category": self.capped_by_category,
            "promoted": self.promoted,
            "pinned": self.pinned,
            "items_out": self.items_out,
        }


@dataclass
class RequestContext:
    """What the caller told us about this request."""

    user_id: str
    surface: str = "home"
    store_id: str | None = None
    locale: str | None = None
    cart_skus: tuple[str, ...] = ()
    purchased_skus: tuple[str, ...] = ()
    session_categories: tuple[str, ...] = ()
    price_affinity: float | None = None


class PolicyLayer:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    # -- stage 1: filters -----------------------------------------------------

    def filter_candidates(
        self,
        candidates: list[ScoredItem],
        item_states: dict[str, ItemState],
        context: RequestContext,
        trace: PolicyTrace,
    ) -> list[ScoredItem]:
        """Resolve each token to orderable SKUs and drop what cannot be served."""
        config = self.config
        purchased = set(context.purchased_skus)
        kept: list[ScoredItem] = []

        for item in candidates:
            if item.product_id in config.excluded_products:
                trace.dropped_excluded += 1
                continue

            if not item.skus:
                trace.dropped_no_sku += 1
                continue

            if config.require_in_stock:
                available = [
                    sku
                    for sku in item.skus
                    if (state := item_states.get(sku)) is None
                    or state.in_stock_at(context.store_id)
                ]
                if not available:
                    trace.dropped_out_of_stock += 1
                    continue
            else:
                available = list(item.skus)

            if config.exclude_purchased and purchased:
                available = [sku for sku in available if sku not in purchased]
                if not available:
                    trace.dropped_purchased += 1
                    continue

            item.available_skus = available
            item.suggested_sku = available[0]
            kept.append(item)

        return kept

    # -- stage 2: adjustments -------------------------------------------------

    def adjust_scores(
        self,
        candidates: list[ScoredItem],
        item_states: dict[str, ItemState],
        context: RequestContext,
        trace: PolicyTrace,
    ) -> list[ScoredItem]:
        """Apply bounded multiplicative adjustments to the model score."""
        config = self.config
        for item in candidates:
            multiplier = 1.0
            state = item_states.get(item.suggested_sku or "")

            if state is not None and state.promo_id:
                multiplier *= config.promo_boost
                item.boosts["promo"] = config.promo_boost
                item.reason_code = "promotion"
                trace.promoted += 1

            if (
                context.price_affinity is not None
                and state is not None
                and state.price
                and state.list_price
            ):
                # Reward items priced near what this customer usually pays.
                relative = min(state.price / max(state.list_price, 1e-6), 2.0)
                closeness = 1.0 - abs(relative - context.price_affinity)
                factor = 1.0 + config.price_affinity_weight * max(closeness, -1.0)
                multiplier *= factor
                item.boosts["price_affinity"] = round(factor, 4)

            if item.category and item.category in context.session_categories:
                # Session context: the customer is shopping this category now.
                multiplier *= 1.05
                item.boosts["session_category"] = 1.05

            multiplier = min(multiplier, config.max_total_boost)
            item.final_score = item.model_score * multiplier
        return candidates

    # -- stage 3: final ordering ---------------------------------------------

    def finalise(
        self, candidates: list[ScoredItem], limit: int, trace: PolicyTrace
    ) -> list[ScoredItem]:
        """Sort, cap per category for diversity, then apply pinned slots."""
        config = self.config
        ordered = sorted(candidates, key=lambda item: item.final_score, reverse=True)

        selected: list[ScoredItem] = []
        overflow: list[ScoredItem] = []
        per_category: dict[str, int] = defaultdict(int)

        for item in ordered:
            category = item.category or "_unknown"
            if config.max_per_category and per_category[category] >= config.max_per_category:
                trace.capped_by_category += 1
                overflow.append(item)
                continue
            per_category[category] += 1
            selected.append(item)
            if len(selected) >= limit:
                break

        # A cap must never return a short list: if diversity starved the
        # response, refill from what it held back.
        if len(selected) < limit:
            selected.extend(overflow[: limit - len(selected)])

        if config.pinned_products:
            by_product = {item.product_id: item for item in selected}
            pinned: list[ScoredItem] = []
            for product_id in config.pinned_products:
                item = by_product.pop(product_id, None)
                if item is not None:
                    item.reason_code = "merchandising"
                    pinned.append(item)
                    trace.pinned += 1
            selected = pinned + [item for item in selected if item.product_id in by_product]

        trace.items_out = len(selected[:limit])
        return selected[:limit]

    # -- the whole layer ------------------------------------------------------

    def apply(
        self,
        candidates: list[ScoredItem],
        item_states: dict[str, ItemState],
        context: RequestContext,
        limit: int,
    ) -> tuple[list[ScoredItem], PolicyTrace]:
        trace = PolicyTrace(candidates_in=len(candidates))
        kept = self.filter_candidates(candidates, item_states, context, trace)
        adjusted = self.adjust_scores(kept, item_states, context, trace)
        return self.finalise(adjusted, limit, trace), trace
