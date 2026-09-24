"""Ranking metrics for retrieval evaluation.

Relevance is binary: a token the user actually interacted with inside the
evaluation window is relevant, everything else is not. Both metrics are
normalised by ``min(len(targets), k)``, so a user with 40 targets and k=10 can
still reach 1.0 - otherwise the score would be capped by the window length
rather than by the model's quality.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], targets: Iterable[int], k: int) -> float:
    target_set = set(targets)
    if not target_set:
        return float("nan")
    top_k = list(ranked)[:k]
    hits = sum(1 for item in top_k if item in target_set)
    return hits / min(len(target_set), k)


def ndcg_at_k(ranked: Sequence[int], targets: Iterable[int], k: int) -> float:
    target_set = set(targets)
    if not target_set:
        return float("nan")
    top_k = list(ranked)[:k]
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(top_k)
        if item in target_set
    )
    ideal_hits = min(len(target_set), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked: Sequence[int], targets: Iterable[int], k: int) -> float:
    target_set = set(targets)
    if not target_set:
        return float("nan")
    return 1.0 if any(item in target_set for item in list(ranked)[:k]) else 0.0


class MetricAccumulator:
    """Collects per-user metrics and the recommended-token set for coverage.

    Per-user values are retained, not just their running sum: a mean without
    its spread cannot be given a confidence interval, and every comparison in
    this project turns on differences small enough that the interval decides
    whether they exist. ``user_ids`` are kept too, so a paired comparison
    between two models can verify it is comparing the same users rather than
    trusting that two loops happened to skip the same rows.
    """

    def __init__(
        self, k_values: Sequence[int], vocab_size: int, keep_per_user: bool = True
    ):
        self.k_values = sorted(k_values)
        self.vocab_size = vocab_size
        self.keep_per_user = keep_per_user
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._per_user: dict[str, list[float]] = {}
        self._recommended: dict[int, set[int]] = {k: set() for k in self.k_values}
        self.user_ids: list[str] = []
        self.n_users = 0

    def _add(self, name: str, value: float) -> None:
        if self.keep_per_user:
            # NaN is retained here on purpose: it marks "undefined for this
            # user" and keeps every metric's list aligned with user_ids, which
            # is what makes the pairing sound. bootstrap_ci drops them.
            self._per_user.setdefault(name, []).append(value)
        if value != value:  # NaN
            return
        self._sums[name] = self._sums.get(name, 0.0) + value
        self._counts[name] = self._counts.get(name, 0) + 1

    def update(
        self, ranked: Sequence[int], targets: Iterable[int], user_id: str | None = None
    ) -> None:
        targets = list(targets)
        self.n_users += 1
        if self.keep_per_user:
            self.user_ids.append(user_id if user_id is not None else str(self.n_users - 1))
        for k in self.k_values:
            self._add(f"recall@{k}", recall_at_k(ranked, targets, k))
            self._add(f"ndcg@{k}", ndcg_at_k(ranked, targets, k))
            self._add(f"hit_rate@{k}", hit_rate_at_k(ranked, targets, k))
            self._recommended[k].update(list(ranked)[:k])

    def results(self) -> dict[str, float]:
        out = {
            name: self._sums[name] / self._counts[name]
            for name in sorted(self._sums)
            if self._counts.get(name)
        }
        for k in self.k_values:
            out[f"coverage@{k}"] = (
                len(self._recommended[k]) / self.vocab_size if self.vocab_size else 0.0
            )
        out["eval_users"] = float(self.n_users)
        return {name: round(float(value), 6) for name, value in out.items()}

    def per_user(self) -> dict[str, list[float]]:
        """Per-user values, one list per metric, aligned with ``user_ids``.

        Coverage is absent: it is a property of the whole recommendation set,
        not of any one user, so it has no per-user value and cannot be
        bootstrapped this way.
        """
        return dict(self._per_user)

    def intervals(
        self, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
    ) -> dict[str, dict[str, float]]:
        """Bootstrap CI for every per-user metric."""
        from retailgr.evaluation.stats import bootstrap_ci

        out: dict[str, dict[str, float]] = {}
        for name, values in self._per_user.items():
            interval = bootstrap_ci(
                values, confidence=confidence, resamples=resamples, seed=seed
            )
            if interval is not None:
                out[name] = interval.as_dict()
        return out


def rank_from_scores(scores: np.ndarray, k: int, exclude: Iterable[int] = ()) -> np.ndarray:
    """Top-``k`` indices of ``scores``, with ``exclude`` masked out.

    ``scores`` is indexed by token id; index 0 is the padding token and is
    always masked.
    """
    masked = np.asarray(scores, dtype=np.float32).copy()
    masked[0] = -np.inf
    excluded = np.fromiter(exclude, dtype=np.int64)
    if excluded.size:
        excluded = excluded[(excluded >= 0) & (excluded < masked.size)]
        masked[excluded] = -np.inf
    k = min(k, masked.size)
    # argpartition gives the top-k unordered; sort just those.
    candidates = np.argpartition(-masked, k - 1)[:k]
    return candidates[np.argsort(-masked[candidates])]
