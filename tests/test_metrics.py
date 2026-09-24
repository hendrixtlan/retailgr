"""Metrics are what the granularity decision rests on, so they are pinned to
hand-computed values."""

from __future__ import annotations

import math

import numpy as np

from retailgr.evaluation.metrics import (
    MetricAccumulator,
    hit_rate_at_k,
    ndcg_at_k,
    rank_from_scores,
    recall_at_k,
)


def test_recall_is_normalised_by_min_targets_k():
    ranked = [5, 3, 9, 1]
    # 2 of the 3 targets are inside the top 4.
    assert recall_at_k(ranked, {3, 9, 77}, 4) == 2 / 3
    # With k=2 only one hit fits, and the denominator becomes k.
    assert recall_at_k(ranked, {3, 9, 77}, 2) == 1 / 2


def test_ndcg_matches_the_hand_computed_value():
    ranked = [5, 3, 9, 1]
    targets = {3, 9}
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, targets, 4) == dcg / idcg


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([7, 8], {7, 8}, 2) == 1.0
    assert recall_at_k([7, 8], {7, 8}, 2) == 1.0


def test_no_hits_scores_zero():
    assert ndcg_at_k([1, 2], {7}, 2) == 0.0
    assert recall_at_k([1, 2], {7}, 2) == 0.0
    assert hit_rate_at_k([1, 2], {7}, 2) == 0.0


def test_empty_targets_are_nan_and_never_counted():
    assert math.isnan(recall_at_k([1, 2], [], 2))
    accumulator = MetricAccumulator([2], vocab_size=10)
    accumulator.update([1, 2], [])
    assert accumulator.results()["eval_users"] == 1.0
    assert "recall@2" not in accumulator.results()


def test_rank_from_scores_masks_padding_and_seen_items():
    scores = np.array([99.0, 1.0, 5.0, 3.0, 4.0], dtype=np.float32)
    # Index 0 is padding and must never be recommended, even with the top score.
    assert 0 not in rank_from_scores(scores, k=4)
    assert list(rank_from_scores(scores, k=2)) == [2, 4]
    # Excluding the best item promotes the next one.
    assert list(rank_from_scores(scores, k=2, exclude=[2])) == [4, 3]


def test_rank_handles_k_larger_than_vocabulary():
    scores = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    assert len(rank_from_scores(scores, k=50)) == 3


def test_accumulator_averages_and_reports_coverage():
    accumulator = MetricAccumulator([2], vocab_size=10)
    accumulator.update([1, 2], {1})  # recall 1.0
    accumulator.update([3, 4], {9})  # recall 0.0
    results = accumulator.results()
    assert results["recall@2"] == 0.5
    assert results["eval_users"] == 2.0
    # Four distinct tokens were recommended out of a vocabulary of ten.
    assert results["coverage@2"] == 0.4
