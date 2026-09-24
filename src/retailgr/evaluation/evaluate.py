"""Run a model over an evaluation split and collect metrics.

Metrics come back overall and broken down by the user's dominant category,
because the granularity question is answered per category, not globally.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np

from retailgr.evaluation.metrics import MetricAccumulator, rank_from_scores
from retailgr.io.loaders import SequenceSplit
from retailgr.models.base import Recommender


def _dominant_category(history: np.ndarray, category_by_id: Mapping[int, str]) -> str:
    labels = [category_by_id.get(int(token), "") for token in history]
    labels = [label for label in labels if label]
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def evaluate(
    model: Recommender,
    split: SequenceSplit,
    vocab_size: int,
    k_values: list[int],
    exclude_seen: bool = True,
    category_by_id: Mapping[int, str] | None = None,
    batch_size: int = 256,
    confidence: float = 0.95,
    resamples: int = 2000,
    with_intervals: bool = True,
) -> dict[str, object]:
    """Score every user in ``split`` and aggregate the ranking metrics.

    The result carries ``per_user`` alongside the means, so two models scored
    on the same split can be compared with a paired test. It is not meant for
    the run record — ``experiment`` strips it before serialising — because it
    is one number per user per metric.
    """
    if len(split) == 0:
        return {"overall": {"eval_users": 0.0}, "by_category": {}, "per_user": {}}

    max_k = max(k_values)
    overall = MetricAccumulator(k_values, vocab_size)
    per_category: dict[str, MetricAccumulator] = {}

    for start in range(0, len(split), batch_size):
        end = start + batch_size
        history_batch = split.batch(start, end)
        histories = history_batch.tokens
        targets = split.targets[start:end]
        scores = model.score(history_batch)

        for row, (history, target) in enumerate(zip(histories, targets, strict=True)):
            if target.size == 0:
                continue
            user_id = split.user_ids[start + row]
            exclude = history if exclude_seen else ()
            ranked = rank_from_scores(scores[row], max_k, exclude)
            overall.update(ranked, target, user_id=user_id)

            if category_by_id is not None:
                label = _dominant_category(history, category_by_id)
                bucket = per_category.setdefault(
                    label, MetricAccumulator(k_values, vocab_size, keep_per_user=False)
                )
                bucket.update(ranked, target, user_id=user_id)

    result: dict[str, object] = {
        "overall": overall.results(),
        "by_category": {name: acc.results() for name, acc in sorted(per_category.items())},
        "per_user": overall.per_user(),
        "user_ids": overall.user_ids,
    }
    if with_intervals:
        result["intervals"] = overall.intervals(confidence=confidence, resamples=resamples)
    return result
