"""Tests for the serving path.

The policy layer gets the most attention here, because it is where the hard
constraints live. A model that ranks badly costs revenue; a policy layer that
leaks an out-of-stock item costs a cancelled order and a support call, and
"never show what we cannot ship" has to hold every time, not on average.
"""

from __future__ import annotations

import numpy as np
import pytest

from retailgr.online_store import InMemoryOnlineStore, ItemState, TailEvent
from retailgr.serving.policy import (
    PolicyConfig,
    PolicyLayer,
    PolicyTrace,
    RequestContext,
    ScoredItem,
)
from retailgr.serving.retrieval import ExactRetrievalIndex, build_index


def _item(token: str, score: float, category: str = "apparel", skus=None) -> ScoredItem:
    return ScoredItem(
        token=token,
        token_id=abs(hash(token)) % 1000 + 1,
        product_id=token.split("-")[0],
        model_score=score,
        category=category,
        skus=list(skus or [f"{token}-M"]),
    )


def _states(**by_sku: ItemState) -> dict[str, ItemState]:
    return dict(by_sku)


# -- retrieval ----------------------------------------------------------------


def test_exact_index_returns_the_true_top_k_in_order():
    embeddings = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.0], [0.0, 1.0], [0.5, 0.0]], dtype=np.float32
    )
    index = ExactRetrievalIndex(embeddings)
    assert index.search(np.array([1.0, 0.0]), k=3) == [1, 2, 4]


def test_exact_index_never_returns_the_padding_row():
    embeddings = np.array([[9.0, 9.0], [1.0, 0.0]], dtype=np.float32)
    index = ExactRetrievalIndex(embeddings)
    assert 0 not in index.search(np.array([1.0, 1.0]), k=2)


def test_exact_index_honours_exclusions():
    embeddings = np.array([[0.0], [3.0], [2.0], [1.0]], dtype=np.float32)
    index = ExactRetrievalIndex(embeddings)
    assert index.search(np.array([1.0]), k=2, exclude={1}) == [2, 3]


def test_index_rejects_a_mismatched_user_vector():
    index = ExactRetrievalIndex(np.zeros((4, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        index.search(np.zeros(4), k=2)


def test_build_index_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="unknown retrieval index"):
        build_index(np.zeros((3, 2), dtype=np.float32), kind="annoy")


# -- policy: filters are vetoes ----------------------------------------------


def test_out_of_stock_items_are_dropped_not_demoted():
    policy = PolicyLayer(PolicyConfig(require_in_stock=True))
    candidates = [_item("A", 9.0, skus=["A-M"]), _item("B", 1.0, skus=["B-M"])]
    states = _states(
        **{
            "A-M": ItemState(sku="A-M", stock_by_location={"s1": 0}),
            "B-M": ItemState(sku="B-M", stock_by_location={"s1": 4}),
        }
    )
    final, trace = policy.apply(candidates, states, RequestContext("U1", store_id="s1"), limit=10)
    # A scored nine times higher and still must not appear.
    assert [item.token for item in final] == ["B"]
    assert trace.dropped_out_of_stock == 1


def test_stock_is_checked_at_the_customers_own_store():
    policy = PolicyLayer(PolicyConfig(require_in_stock=True))
    states = _states(
        **{"A-M": ItemState(sku="A-M", stock_by_location={"s1": 5, "s2": 0})}
    )
    in_s1, _ = policy.apply(
        [_item("A", 1.0, skus=["A-M"])], states, RequestContext("U1", store_id="s1"), limit=5
    )
    in_s2, _ = policy.apply(
        [_item("A", 1.0, skus=["A-M"])], states, RequestContext("U1", store_id="s2"), limit=5
    )
    assert len(in_s1) == 1
    assert len(in_s2) == 0


def test_only_the_available_sizes_are_offered():
    """The token survives if any of its SKUs can ship, and the suggested SKU
    must be one of those."""
    policy = PolicyLayer(PolicyConfig(require_in_stock=True))
    candidates = [_item("A", 1.0, skus=["A-S", "A-M", "A-L"])]
    states = _states(
        **{
            "A-S": ItemState(sku="A-S", stock_by_location={"s1": 0}),
            "A-M": ItemState(sku="A-M", stock_by_location={"s1": 2}),
            "A-L": ItemState(sku="A-L", stock_by_location={"s1": 7}),
        }
    )
    final, _ = policy.apply(candidates, states, RequestContext("U1", store_id="s1"), limit=5)
    assert final[0].available_skus == ["A-M", "A-L"]
    assert final[0].suggested_sku in ("A-M", "A-L")


def test_already_purchased_items_are_dropped():
    policy = PolicyLayer(PolicyConfig(exclude_purchased=True, require_in_stock=False))
    context = RequestContext("U1", purchased_skus=("A-M",))
    final, trace = policy.apply(
        [_item("A", 5.0, skus=["A-M"]), _item("B", 1.0, skus=["B-M"])], {}, context, limit=5
    )
    assert [item.token for item in final] == ["B"]
    assert trace.dropped_purchased == 1


def test_excluded_products_are_dropped():
    policy = PolicyLayer(PolicyConfig(excluded_products=("A",), require_in_stock=False))
    final, trace = policy.apply(
        [_item("A", 5.0), _item("B", 1.0)], {}, RequestContext("U1"), limit=5
    )
    assert [item.token for item in final] == ["B"]
    assert trace.dropped_excluded == 1


def test_a_token_with_no_skus_is_dropped():
    """A model token that resolves to nothing orderable cannot be served."""
    policy = PolicyLayer(PolicyConfig(require_in_stock=False))
    orphan = _item("A", 5.0)
    orphan.skus = []
    final, trace = policy.apply([orphan, _item("B", 1.0)], {}, RequestContext("U1"), limit=5)
    assert [item.token for item in final] == ["B"]
    assert trace.dropped_no_sku == 1


# -- policy: adjustments are bounded ------------------------------------------


def test_promo_boosts_but_cannot_overturn_a_large_relevance_gap():
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, promo_boost=1.15))
    states = _states(**{"B-M": ItemState(sku="B-M", promo_id="SUMMER")})
    final, trace = policy.apply(
        [_item("A", 10.0, skus=["A-M"]), _item("B", 1.0, skus=["B-M"])],
        states,
        RequestContext("U1"),
        limit=5,
    )
    assert [item.token for item in final] == ["A", "B"]
    assert trace.promoted == 1
    assert final[1].reason_code == "promotion"


def test_promo_can_overturn_a_small_gap():
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, promo_boost=1.15))
    states = _states(**{"B-M": ItemState(sku="B-M", promo_id="SUMMER")})
    final, _ = policy.apply(
        [_item("A", 1.00, skus=["A-M"]), _item("B", 0.95, skus=["B-M"])],
        states,
        RequestContext("U1"),
        limit=5,
    )
    assert [item.token for item in final] == ["B", "A"]


def test_total_boost_is_capped():
    policy = PolicyLayer(
        PolicyConfig(require_in_stock=False, promo_boost=10.0, max_total_boost=1.5)
    )
    states = _states(**{"A-M": ItemState(sku="A-M", promo_id="X")})
    final, _ = policy.apply(
        [_item("A", 1.0, skus=["A-M"])], states, RequestContext("U1"), limit=5
    )
    assert final[0].final_score == pytest.approx(1.5)


# -- policy: diversity and merchandising --------------------------------------


def test_category_cap_limits_a_single_category():
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, max_per_category=2))
    candidates = [_item(f"A{i}", 10.0 - i, category="apparel") for i in range(5)]
    candidates += [_item("G1", 1.0, category="grocery")]
    final, trace = policy.apply(candidates, {}, RequestContext("U1"), limit=3)
    categories = [item.category for item in final]
    assert categories.count("apparel") == 2
    assert "grocery" in categories
    assert trace.capped_by_category >= 1


def test_the_cap_never_returns_a_short_list():
    """Diversity must not starve the response: if the cap leaves fewer items
    than asked for, the held-back ones come back."""
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, max_per_category=1))
    candidates = [_item(f"A{i}", 10.0 - i, category="apparel") for i in range(5)]
    final, _ = policy.apply(candidates, {}, RequestContext("U1"), limit=4)
    assert len(final) == 4


def test_pinned_products_come_first_and_are_labelled():
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, pinned_products=("B",)))
    final, trace = policy.apply(
        [_item("A", 10.0), _item("B", 0.1)], {}, RequestContext("U1"), limit=5
    )
    assert final[0].product_id == "B"
    assert final[0].reason_code == "merchandising"
    assert trace.pinned == 1


def test_session_category_gets_a_small_nudge():
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, max_per_category=10))
    context = RequestContext("U1", session_categories=("grocery",))
    final, _ = policy.apply(
        [_item("A", 1.00, category="apparel"), _item("G", 0.98, category="grocery")],
        {},
        context,
        limit=5,
    )
    assert [item.token for item in final] == ["G", "A"]


def test_limit_is_respected():
    policy = PolicyLayer(PolicyConfig(require_in_stock=False, max_per_category=100))
    candidates = [_item(f"A{i}", float(20 - i)) for i in range(20)]
    final, trace = policy.apply(candidates, {}, RequestContext("U1"), limit=5)
    assert len(final) == 5
    assert trace.items_out == 5


def test_policy_config_rejects_unknown_keys():
    """A typo in policy.yaml must fail loudly, not be silently ignored."""
    with pytest.raises(ValueError, match="unknown policy keys"):
        PolicyConfig.from_dict({"promo_bost": 2.0})


def test_policy_trace_serialises_every_counter():
    trace = PolicyTrace()
    assert set(trace.as_dict()) == {
        "candidates_in",
        "dropped_out_of_stock",
        "dropped_ineligible",
        "dropped_purchased",
        "dropped_excluded",
        "dropped_no_sku",
        "capped_by_category",
        "promoted",
        "pinned",
        "items_out",
    }


# -- the online store ---------------------------------------------------------


def test_tail_keeps_only_the_most_recent_events():
    store = InMemoryOnlineStore(tail_length=3)
    for index in range(6):
        store.append_event(
            "U1", TailEvent(sku=f"S{index}", token=f"T{index}", action="view", event_ts=index)
        )
    assert [event.token for event in store.user_tail("U1")] == ["T3", "T4", "T5"]


def test_item_states_reads_many_in_one_call():
    store = InMemoryOnlineStore()
    store.put_item_state(ItemState(sku="A", price=1.0))
    store.put_item_state(ItemState(sku="B", price=2.0))
    states = store.item_states(["A", "B", "MISSING"])
    assert set(states) == {"A", "B"}


def test_global_fallback_is_used_when_a_user_has_none():
    store = InMemoryOnlineStore()
    store.put_global_fallback(["P1", "P2"])
    assert store.fallback("unknown-user") == ["P1", "P2"]
    store.put_fallback("U1", ["P9"])
    assert store.fallback("U1") == ["P9"]
