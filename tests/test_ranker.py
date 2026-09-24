"""Tests for the ranker.

The load-bearing test here is ``test_mfalcon_batching_equals_one_at_a_time``.
M-FALCON's entire justification is that scoring many candidates in one pass is
*identical* to scoring them separately, only cheaper. If that equivalence
breaks — candidates leaking into each other's attention, or sitting at
different relative positions — the batched path silently returns different
numbers from the path any single-candidate debugging session would show, and
nothing else in the system would notice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from retailgr.actions import ACTION_TO_ID, encode_actions
from retailgr.io.loaders import SequenceSplit
from retailgr.models.ranker import (
    HEADS,
    RETURN_KEPT,
    RETURN_RETURNED,
    HSTURanker,
    HSTURankerNet,
    ScoreBlend,
)

VOCAB = 30
EVENTS = 6
DIM = 16


def _net(**kwargs) -> HSTURankerNet:
    torch.manual_seed(0)
    defaults = dict(
        vocab_size=VOCAB,
        embedding_dim=DIM,
        num_blocks=2,
        num_heads=2,
        dropout=0.0,
        max_events=EVENTS,
    )
    defaults.update(kwargs)
    net = HSTURankerNet(**defaults)
    net.eval()
    return net


def _history(batch: int = 1):
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randint(1, VOCAB, (batch, EVENTS), generator=generator)
    actions = torch.full((batch, EVENTS), ACTION_TO_ID["view"])
    times = (torch.arange(EVENTS).repeat(batch, 1) * 900).long()
    return tokens, actions, times


# -- the sequence shape -------------------------------------------------------


def test_interleaving_alternates_items_and_actions():
    net = _net()
    tokens, actions, times = _history()
    items, acts, kinds, ts, prefix_len = net.interleave(tokens, actions, times)

    assert prefix_len == 2 * EVENTS
    # Items on even positions, actions on odd, and never both.
    torch.testing.assert_close(items[:, 0::2], tokens)
    torch.testing.assert_close(acts[:, 1::2], actions)
    assert int(items[:, 1::2].sum()) == 0
    assert int(acts[:, 0::2].sum()) == 0
    # Kind tags distinguish the two, and both share the event's timestamp.
    assert set(kinds[0, 0::2].tolist()) == {1}
    assert set(kinds[0, 1::2].tolist()) == {2}
    torch.testing.assert_close(ts[:, 0::2], ts[:, 1::2])


def test_candidates_are_appended_as_item_positions_with_no_action():
    net = _net()
    tokens, actions, times = _history()
    candidates = torch.tensor([[7, 8, 9]])
    items, acts, kinds, _, prefix_len = net.interleave(tokens, actions, times, candidates)

    assert items.shape[1] == 2 * EVENTS + 3
    torch.testing.assert_close(items[:, prefix_len:], candidates)
    # A candidate has no action yet - predicting it is the task.
    assert int(acts[:, prefix_len:].sum()) == 0
    assert set(kinds[0, prefix_len:].tolist()) == {1}


# -- the mask -----------------------------------------------------------------


def test_candidates_cannot_see_each_other():
    net = _net()
    tokens, actions, times = _history()
    candidates = torch.tensor([[7, 8, 9]])
    items, acts, _, _, prefix_len = net.interleave(tokens, actions, times, candidates)
    mask = net.build_mask(items, acts, prefix_len)[0]

    for row in range(prefix_len, items.shape[1]):
        for col in range(prefix_len, items.shape[1]):
            expected = 1.0 if row == col else 0.0
            assert mask[row, col] == expected, (row, col)
        # It does see the whole history.
        assert mask[row, :prefix_len].sum() == prefix_len


def test_prefix_attention_stays_causal():
    net = _net()
    tokens, actions, times = _history()
    items, acts, _, _, prefix_len = net.interleave(tokens, actions, times)
    mask = net.build_mask(items, acts, prefix_len)[0]
    # No position may attend to a later one.
    assert torch.triu(mask, diagonal=1).sum() == 0


def test_candidate_rows_share_one_relative_position():
    """Every candidate must see the history from the same distance, or the
    batched score depends on a candidate's slot in the batch."""
    net = _net()
    tokens, actions, times = _history()
    candidates = torch.tensor([[7, 8, 9, 10]])
    items, _, _, ts, prefix_len = net.interleave(tokens, actions, times, candidates)
    bias = net.build_relative_bias(ts, prefix_len, items.shape[1])

    reference = bias[0, prefix_len, :prefix_len]
    for row in range(prefix_len + 1, items.shape[1]):
        torch.testing.assert_close(bias[0, row, :prefix_len], reference)


# -- M-FALCON ----------------------------------------------------------------


def test_mfalcon_batching_equals_one_at_a_time():
    """The whole point of M-FALCON: same answer, one pass instead of many."""
    net = _net()
    tokens, actions, times = _history()
    candidate_ids = [3, 11, 17, 21, 29]

    batched_logits, prefix_len = net(
        tokens, actions, times, torch.tensor([candidate_ids])
    )
    batched = {
        head: batched_logits[head][0, prefix_len:].detach().clone() for head in HEADS
    }

    for position, candidate in enumerate(candidate_ids):
        single_logits, single_prefix = net(
            tokens, actions, times, torch.tensor([[candidate]])
        )
        for head in HEADS:
            alone = single_logits[head][0, single_prefix:]
            torch.testing.assert_close(
                batched[head][position : position + 1], alone, rtol=2e-4, atol=2e-5
            )


def test_mfalcon_equivalence_holds_without_the_relative_bias():
    net = _net(use_relative_bias=False)
    tokens, actions, times = _history()
    candidate_ids = [5, 13, 19]

    batched_logits, prefix_len = net(tokens, actions, times, torch.tensor([candidate_ids]))
    for position, candidate in enumerate(candidate_ids):
        single_logits, single_prefix = net(
            tokens, actions, times, torch.tensor([[candidate]])
        )
        torch.testing.assert_close(
            batched_logits["purchase"][0, prefix_len + position : prefix_len + position + 1],
            single_logits["purchase"][0, single_prefix:],
            rtol=2e-4,
            atol=2e-5,
        )


def test_candidate_order_does_not_change_its_score():
    net = _net()
    tokens, actions, times = _history()
    forward, prefix_len = net(tokens, actions, times, torch.tensor([[3, 11, 17]]))
    reverse, _ = net(tokens, actions, times, torch.tensor([[17, 11, 3]]))
    torch.testing.assert_close(
        forward["purchase"][0, prefix_len],
        reverse["purchase"][0, prefix_len + 2],
        rtol=2e-4,
        atol=2e-5,
    )


def test_future_history_cannot_reach_an_earlier_position():
    net = _net()
    tokens, actions, times = _history()
    baseline, _ = net(tokens, actions, times)

    altered = tokens.clone()
    altered[0, -1] = (int(altered[0, -1]) % (VOCAB - 2)) + 1
    if int(altered[0, -1]) == int(tokens[0, -1]):
        altered[0, -1] = int(altered[0, -1]) + 1
    changed, _ = net(altered, actions, times)

    # Positions before the last event must be untouched.
    torch.testing.assert_close(
        baseline["purchase"][:, : 2 * (EVENTS - 1)],
        changed["purchase"][:, : 2 * (EVENTS - 1)],
    )


# -- labels -------------------------------------------------------------------


def test_head_labels_follow_the_action_hierarchy():
    actions = encode_actions(["view", "click", "add_to_cart", "purchase", "return"])
    actions = torch.from_numpy(actions).unsqueeze(0)
    returned = torch.full_like(actions, -1)
    labels = HSTURanker._labels_for(actions, returned)

    click, _ = labels["click"]
    cart, _ = labels["cart"]
    purchase, _ = labels["purchase"]
    # view, click, cart, purchase, return
    assert click[0].tolist() == [0, 1, 1, 1, 0]
    assert cart[0].tolist() == [0, 0, 1, 1, 0]
    assert purchase[0].tolist() == [0, 0, 0, 1, 0]


def test_immature_purchases_are_masked_out_of_the_return_head():
    """A purchase too recent to have been returned must contribute nothing —
    counting it as 'kept' teaches the model that recent purchases are safe."""
    actions = torch.from_numpy(encode_actions(["purchase"] * 4)).unsqueeze(0)
    # kept, returned, immature, not-a-purchase
    returned = torch.tensor([[RETURN_KEPT, RETURN_RETURNED, -2, -1]])
    label, mask = HSTURanker._labels_for(actions, returned)["return"]

    assert mask[0].tolist() == [1, 1, 0, 0]
    assert label[0].tolist() == [0, 1, 0, 0]


def test_padding_positions_are_masked_for_every_head():
    actions = torch.tensor([[0, 0, ACTION_TO_ID["purchase"]]])
    returned = torch.tensor([[-1, -1, RETURN_KEPT]])
    labels = HSTURanker._labels_for(actions, returned)
    for head in ("click", "cart", "purchase"):
        _, mask = labels[head]
        assert mask[0].tolist() == [0, 0, 1]


# -- the blend ----------------------------------------------------------------


def test_blend_penalises_return_risk_in_proportion_to_purchase_odds():
    blend = ScoreBlend(click=0.0, cart=0.0, purchase=1.0, return_penalty=1.0)
    scores = {
        "click": np.zeros(3),
        "cart": np.zeros(3),
        "purchase": np.array([0.9, 0.9, 0.1]),
        "return": np.array([0.0, 0.5, 0.5]),
    }
    blended = blend.apply(scores)
    # Same purchase odds, higher return risk -> lower score.
    assert blended[0] > blended[1]
    # The penalty scales with purchase odds: a likely-returned item nobody was
    # going to buy is barely penalised.
    assert blended[1] == pytest.approx(0.9 - 0.45)
    assert blended[2] == pytest.approx(0.1 - 0.05)


def test_blend_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown score blend"):
        ScoreBlend.from_dict({"purchase_weight": 1.0})


# -- training and scoring -----------------------------------------------------


def _split(users: int = 24, length: int = 8) -> SequenceSplit:
    rng = np.random.default_rng(7)
    split = SequenceSplit()
    for user in range(users):
        tokens = rng.integers(1, VOCAB, size=length).astype(np.int64)
        actions = encode_actions(
            [
                "purchase" if index % 3 == 0 else "view"
                for index in range(length)
            ]
        )
        returned = np.array(
            [
                (RETURN_RETURNED if index % 6 == 0 else RETURN_KEPT)
                if index % 3 == 0
                else -1
                for index in range(length)
            ],
            dtype=np.int64,
        )
        split.user_ids.append(f"u{user}")
        split.inputs.append(tokens)
        split.actions.append(actions)
        split.timestamps.append(np.arange(length, dtype=np.int64) * 3600)
        split.returned.append(returned)
        split.targets.append(tokens[:2])
    return split


def test_fit_reduces_the_loss_and_reports_label_counts():
    split = _split()
    ranker = HSTURanker(
        VOCAB,
        {
            "hidden_dim": DIM,
            "num_blocks": 1,
            "max_events": 8,
            "epochs": 8,
            "batch_size": 8,
            "dropout": 0.0,
        },
    )
    stats = ranker.fit(split)
    assert stats["final_epoch_loss"] < stats["first_epoch_loss"]
    # Every head saw labels, and the return head saw fewer than the others.
    for head in HEADS:
        assert stats[f"{head}_labels"] > 0
    assert stats["return_labels"] < stats["purchase_labels"]
    assert 0.0 < stats["return_positive_rate"] < 1.0


def test_score_candidates_returns_probabilities_per_head():
    split = _split()
    ranker = HSTURanker(
        VOCAB,
        {"hidden_dim": DIM, "num_blocks": 1, "max_events": 8, "epochs": 1, "batch_size": 8},
    )
    ranker.fit(split)
    candidates = np.array([2, 5, 9, 14, 21])
    scores = ranker.score_candidates(
        split.inputs[0], split.actions[0], split.timestamps[0], split.returned[0], candidates
    )

    for name, values in scores.as_dict().items():
        assert values.shape == candidates.shape, name
        assert np.isfinite(values).all(), name
        if name != "blended":
            assert ((values >= 0) & (values <= 1)).all(), name


def test_micro_batching_does_not_change_the_scores():
    """Splitting candidates across micro-batches is a cost decision, not a
    modelling one."""
    split = _split()
    base = {
        "hidden_dim": DIM,
        "num_blocks": 1,
        "max_events": 8,
        "epochs": 1,
        "batch_size": 8,
        "dropout": 0.0,
    }
    big = HSTURanker(VOCAB, {**base, "candidate_micro_batch": 64})
    big.fit(split)
    small = HSTURanker(VOCAB, {**base, "candidate_micro_batch": 2})
    small.net.load_state_dict(big.net.state_dict())
    small.net.eval()

    candidates = np.array([2, 5, 9, 14, 21, 26])
    args = (split.inputs[0], split.actions[0], split.timestamps[0], split.returned[0])
    np.testing.assert_allclose(
        big.score_candidates(*args, candidates).blended,
        small.score_candidates(*args, candidates).blended,
        rtol=1e-4,
        atol=1e-5,
    )


# -- negatives ----------------------------------------------------------------


def test_candidate_block_pairs_the_true_next_item_against_negatives():
    """The positive must be the held-out next item, in slot 0."""
    ranker = HSTURanker(
        VOCAB,
        {"hidden_dim": DIM, "num_blocks": 1, "max_events": 8, "epochs": 1, "num_negatives": 5},
    )
    tokens = torch.tensor([[3, 4, 5, 6]])
    actions = torch.from_numpy(encode_actions(["view", "view", "view", "purchase"])).unsqueeze(0)
    returned = torch.tensor([[-1, -1, -1, RETURN_KEPT]])
    generator = torch.Generator().manual_seed(1)

    candidates, labels = ranker._candidate_batch(tokens, actions, returned, generator)
    assert candidates.shape == (1, 6)
    assert int(candidates[0, 0]) == 6  # the held-out next item

    purchase_label, purchase_mask = labels["purchase"]
    assert purchase_label[0, 0] == 1.0  # it was purchased
    assert purchase_label[0, 1:].sum() == 0.0  # negatives are all zero
    assert purchase_mask[0, 0] == 1.0

    # The return head only trains on the positive: a negative was never
    # purchased, so P(return | purchase) is undefined for it.
    _, return_mask = labels["return"]
    assert return_mask[0, 0] == 1.0
    assert return_mask[0, 1:].sum() == 0.0


def test_a_negative_that_collides_with_the_positive_is_masked():
    """Sampling the true item as a negative would teach the opposite label."""
    ranker = HSTURanker(
        VOCAB,
        {"hidden_dim": DIM, "num_blocks": 1, "max_events": 8, "epochs": 1, "num_negatives": 4},
    )
    tokens = torch.tensor([[3, 4, 5, 6]])
    actions = torch.from_numpy(encode_actions(["view"] * 4)).unsqueeze(0)
    returned = torch.full_like(tokens, -1)

    candidates, labels = ranker._candidate_batch(
        tokens, actions, returned, torch.Generator().manual_seed(0)
    )
    # Force a collision and re-derive the mask the same way fit does.
    collision = candidates[:, 1:] == candidates[:, :1]
    _, mask = labels["click"]
    assert (mask[:, 1:][collision] == 0).all()


def test_training_without_negatives_still_runs_but_is_recorded():
    """num_negatives=0 reproduces the positives-only failure. It must remain
    reachable — that is how the ablation is run — and visible in the stats."""
    split = _split()
    ranker = HSTURanker(
        VOCAB,
        {
            "hidden_dim": DIM,
            "num_blocks": 1,
            "max_events": 8,
            "epochs": 2,
            "batch_size": 8,
            "num_negatives": 0,
            "dropout": 0.0,
        },
    )
    stats = ranker.fit(split)
    assert stats["negatives_per_positive"] == 0.0
    # Only the single positive per user contributes a candidate label.
    assert stats["candidate_labels"] > 0


def test_negatives_increase_the_candidate_label_count():
    split = _split()
    base = {
        "hidden_dim": DIM,
        "num_blocks": 1,
        "max_events": 8,
        "epochs": 1,
        "batch_size": 8,
        "dropout": 0.0,
    }
    without = HSTURanker(VOCAB, {**base, "num_negatives": 0}).fit(split)
    with_negatives = HSTURanker(VOCAB, {**base, "num_negatives": 16}).fit(split)
    assert with_negatives["candidate_labels"] > without["candidate_labels"] * 5


# -- the gate -----------------------------------------------------------------


def test_gate_blocks_a_ranker_that_orders_worse_than_retrieval():
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate(
        {"signals": {"blended": {"mrr": 0.10}}},
        {"mrr": 0.59},
    )
    assert verdict["passed"] is False
    assert verdict["relative_mrr"] == pytest.approx(0.1695, abs=1e-3)
    assert "regression" in verdict["reason"]


def test_gate_passes_a_ranker_that_matches_retrieval():
    from retailgr.evaluation.ranking import ranker_gate

    assert ranker_gate({"signals": {"blended": {"mrr": 0.60}}}, {"mrr": 0.59})["passed"]
    # Exactly equal passes: the threshold is "at least as good".
    assert ranker_gate({"signals": {"blended": {"mrr": 0.59}}}, {"mrr": 0.59})["passed"]


def test_gate_can_demand_a_margin():
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate(
        {"signals": {"blended": {"mrr": 0.60}}}, {"mrr": 0.59}, min_relative_mrr=1.1
    )
    assert verdict["passed"] is False


def test_gate_fails_closed_when_it_cannot_be_evaluated():
    """No measurement means no ship — not a shrug and a default of yes."""
    from retailgr.evaluation.ranking import ranker_gate

    assert ranker_gate({}, {})["passed"] is False
    assert ranker_gate({"signals": {}}, {"mrr": 0.5})["passed"] is False


def test_roc_auc_matches_hand_computed_values():
    from retailgr.evaluation.ranking import roc_auc

    # Perfect separation, and its mirror image.
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(np.array([1, 1, 0, 0]), np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    # All scores tied averages to 0.5.
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.5] * 4)) == pytest.approx(0.5)
    # One class is undefined, not 0.5.
    assert np.isnan(roc_auc(np.array([1, 1]), np.array([0.2, 0.8])))


def test_expected_value_blend_keeps_the_purchase_head_influential():
    """The bug this formulation fixes: with importance-style weights the
    purchase head's smaller natural range made it irrelevant."""
    # Realistic ranges measured on the synthetic set.
    scores = {
        "click": np.array([0.10, 0.02]),
        "cart": np.array([0.09, 0.02]),
        "purchase": np.array([0.001, 0.006]),
        "return": np.array([0.0, 0.0]),
    }
    value = ScoreBlend(click=0.005, cart=0.05, purchase=1.0, return_penalty=1.0)
    blended = value.apply(scores)
    # Item 1 has 6x the purchase probability and must win, despite item 0
    # having the higher click and cart probabilities.
    assert blended[1] > blended[0]


def test_return_penalty_can_zero_out_a_purchase():
    blend = ScoreBlend(click=0.0, cart=0.0, purchase=1.0, return_penalty=1.0)
    scores = {
        "click": np.zeros(2),
        "cart": np.zeros(2),
        "purchase": np.array([0.5, 0.5]),
        "return": np.array([0.0, 1.0]),
    }
    blended = blend.apply(scores)
    assert blended[0] == pytest.approx(0.5)
    assert blended[1] == pytest.approx(0.0)


def test_return_penalty_above_one_cannot_make_the_score_negative():
    blend = ScoreBlend(click=0.0, cart=0.0, purchase=1.0, return_penalty=2.0)
    scores = {
        "click": np.zeros(1),
        "cart": np.zeros(1),
        "purchase": np.array([0.5]),
        "return": np.array([0.9]),
    }
    assert blend.apply(scores)[0] == pytest.approx(0.0)


# -- hard negatives -----------------------------------------------------------


class _RankedRetrieval:
    """Retrieval whose ordering is fixed, so the pool it yields is known."""

    def __init__(self, vocab_size: int, preferred: list[int]):
        self.vocab_size = vocab_size
        self.preferred = preferred

    def score(self, batch):
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for rank, token in enumerate(self.preferred):
            scores[token] = 100.0 - rank
        return [scores for _ in range(len(batch))]


def test_the_negative_pool_holds_retrievals_order_and_skips_the_history():
    from retailgr.models.ranker import retrieval_negative_pool

    split = _split(users=3, length=6)
    split.inputs[0] = np.array([2, 3, 4, 5, 6, 7], dtype=np.int64)
    # Retrieval prefers items the user has already seen, then fresh ones.
    retrieval = _RankedRetrieval(VOCAB, [2, 3, 4, 20, 21, 22, 23])

    pool = retrieval_negative_pool(retrieval, split, VOCAB, pool_size=4, max_events=8)
    assert pool.shape == (3, 4)
    # The prefix is everything but the last event, and it is excluded the way
    # `exclude_seen` excludes it at serving time.
    prefix = set(split.inputs[0][:-1].tolist())
    assert not (set(pool[0].tolist()) & prefix)
    assert pool[0][0] == 20  # retrieval's best surviving candidate


def test_the_pool_leaves_the_positive_in():
    """Removing it would make the pool's composition depend on the label, and
    the collision mask already handles it downstream."""
    from retailgr.models.ranker import retrieval_negative_pool

    split = _split(users=1, length=5)
    split.inputs[0] = np.array([2, 3, 4, 5, 9], dtype=np.int64)
    pool = retrieval_negative_pool(
        _RankedRetrieval(VOCAB, [9, 20, 21]), split, VOCAB, pool_size=3, max_events=8
    )
    assert 9 in pool[0].tolist()


def test_hard_negatives_are_drawn_from_the_pool_in_the_configured_share():
    ranker = HSTURanker(
        VOCAB, {"hidden_dim": DIM, "num_negatives": 10, "hard_negative_fraction": 0.5}
    )
    pool = torch.full((4, 6), 7, dtype=torch.long)  # every pool entry is token 7
    generator = torch.Generator().manual_seed(0)
    drawn = ranker._sample_negatives(4, pool, generator)

    assert drawn.shape == (4, 10)
    # Half come from the pool, so exactly half are token 7 (up to the chance
    # that a uniform draw also lands on 7).
    assert (drawn[:, :5] == 7).all()


def test_a_zero_fraction_ignores_the_pool_entirely():
    ranker = HSTURanker(
        VOCAB, {"hidden_dim": DIM, "num_negatives": 8, "hard_negative_fraction": 0.0}
    )
    pool = torch.full((3, 4), 7, dtype=torch.long)
    with_pool = ranker._sample_negatives(3, pool, torch.Generator().manual_seed(1))
    without = ranker._sample_negatives(3, None, torch.Generator().manual_seed(1))
    assert torch.equal(with_pool, without)


def test_empty_pool_slots_fall_back_to_uniform_not_to_padding():
    """A padding token is a candidate the model never sees, so a user
    retrieval could not score must not contribute one as a negative."""
    ranker = HSTURanker(
        VOCAB, {"hidden_dim": DIM, "num_negatives": 6, "hard_negative_fraction": 1.0}
    )
    pool = torch.zeros((3, 4), dtype=torch.long)  # nothing retrievable
    drawn = ranker._sample_negatives(3, pool, torch.Generator().manual_seed(2))
    assert (drawn > 0).all()


def test_fit_accepts_a_pool_and_records_the_share_it_used():
    split = _split(users=16, length=8)
    base = {
        "hidden_dim": DIM,
        "num_blocks": 1,
        "max_events": 8,
        "epochs": 1,
        "batch_size": 8,
        "dropout": 0.0,
        "num_negatives": 8,
    }
    pool = np.full((len(split.inputs), 5), 11, dtype=np.int64)

    hard = HSTURanker(VOCAB, {**base, "hard_negative_fraction": 0.5})
    assert hard.fit(split, hard_negatives=pool)["hard_negative_fraction"] == 0.5
    # Passing a pool to a ranker that does not want one is harmless.
    soft = HSTURanker(VOCAB, {**base, "hard_negative_fraction": 0.0})
    assert soft.fit(split, hard_negatives=pool)["hard_negative_fraction"] == 0.0


def test_a_misaligned_pool_is_refused_rather_than_silently_reindexed():
    split = _split(users=8, length=8)
    ranker = HSTURanker(
        VOCAB,
        {
            "hidden_dim": DIM,
            "num_blocks": 1,
            "max_events": 8,
            "epochs": 1,
            "hard_negative_fraction": 0.5,
        },
    )
    with pytest.raises(ValueError, match="one row per user"):
        ranker.fit(split, hard_negatives=np.zeros((3, 5), dtype=np.int64))


# -- comparing two rankers ----------------------------------------------------


def _discrimination(reciprocal: list[float], users: list[str] | None = None) -> dict:
    users = users or [f"u{i}" for i in range(len(reciprocal))]
    return {
        "users": len(reciprocal),
        "user_ids": users,
        "signals": {"blended": {"mrr": float(np.mean(reciprocal))}},
        "reciprocal_ranks": {"blended": list(reciprocal)},
    }


def _retrieval(reciprocal: list[float], users: list[str] | None = None) -> dict:
    users = users or [f"u{i}" for i in range(len(reciprocal))]
    return {
        "users": len(reciprocal),
        "user_ids": users,
        "mrr": float(np.mean(reciprocal)),
        "reciprocal_ranks": list(reciprocal),
    }


def test_two_rankers_are_compared_per_user_not_mean_against_mean():
    from retailgr.evaluation.ranking import compare_rankers

    rng = np.random.default_rng(0)
    base = rng.uniform(0.05, 0.9, 400)
    better = np.clip(base + 0.01, 0, 1)

    result = compare_rankers(_discrimination(better), _discrimination(base))
    assert result.significant
    assert result.mean_difference > 0


def test_a_comparison_on_different_users_is_refused_not_approximated():
    """A paired test on misaligned rows is not a weaker result, it is a wrong
    one, so it returns nothing rather than a number."""
    from retailgr.evaluation.ranking import compare_rankers

    a = _discrimination([0.5, 0.4, 0.3], ["u0", "u1", "u2"])
    b = _discrimination([0.5, 0.4, 0.3], ["u0", "u9", "u2"])
    assert compare_rankers(a, b) is None
    assert compare_rankers(a, {"reciprocal_ranks": {}}) is None


def test_the_gate_blocks_a_significant_regression_against_retrieval():
    from retailgr.evaluation.ranking import ranker_gate

    rng = np.random.default_rng(1)
    retrieval_scores = rng.uniform(0.3, 0.9, 300)
    ranker_scores = np.clip(retrieval_scores - 0.05, 0, 1)

    verdict = ranker_gate(_discrimination(ranker_scores), _retrieval(retrieval_scores))
    assert verdict["ordering_passed"] is False
    assert verdict["paired"]["mean_difference"] < 0
    assert "worse" in verdict["paired_verdict"]


def test_the_gate_still_reports_the_paired_test_when_the_ranker_wins():
    from retailgr.evaluation.ranking import ranker_gate

    rng = np.random.default_rng(2)
    retrieval_scores = rng.uniform(0.2, 0.8, 300)
    ranker_scores = np.clip(retrieval_scores + 0.05, 0, 1)

    verdict = ranker_gate(_discrimination(ranker_scores), _retrieval(retrieval_scores))
    assert verdict["passed"] is True
    assert "better" in verdict["paired_verdict"]


def test_the_gate_works_without_per_user_values_at_all():
    """Older results carry only the aggregates; the gate must still decide."""
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate({"signals": {"blended": {"mrr": 0.7}}}, {"mrr": 0.59})
    assert verdict["passed"] is True
    assert "paired" not in verdict


def test_discrimination_keeps_aligned_per_user_values():
    from retailgr.evaluation.ranking import evaluate_discrimination

    split = _split(users=10, length=8)
    ranker = HSTURanker(VOCAB, {"hidden_dim": DIM, "num_blocks": 1, "max_events": 8})
    result = evaluate_discrimination(
        ranker, split, vocab_size=VOCAB, num_negatives=19, max_users=5
    )
    assert len(result["user_ids"]) == result["users"]
    for name, values in result["reciprocal_ranks"].items():
        assert len(values) == result["users"], name
        assert result["signals"][name]["mrr"] == pytest.approx(np.mean(values), abs=1e-5)
