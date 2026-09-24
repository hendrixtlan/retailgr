"""Popularity baseline.

Every recommender must beat this one before it is worth deploying. Recency
weighting is included because a flat all-time count is a weaker baseline than
what a merchandiser would actually put on the page.
"""

from __future__ import annotations

import numpy as np

from retailgr.io.loaders import HistoryBatch, SequenceSplit
from retailgr.models.base import Recommender


class PopularityModel(Recommender):
    name = "popularity"

    def __init__(self, vocab_size: int, recency_halflife: float = 0.0):
        """``recency_halflife`` is measured in events from the end of each
        history; 0 disables the weighting."""
        self.vocab_size = vocab_size
        self.recency_halflife = float(recency_halflife)
        self.scores = np.zeros(vocab_size, dtype=np.float32)

    def fit(self, train: SequenceSplit, val: SequenceSplit | None = None) -> dict[str, float]:
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for history in train.inputs:
            if history.size == 0:
                continue
            valid = history[(history > 0) & (history < self.vocab_size)]
            if valid.size == 0:
                continue
            if self.recency_halflife > 0:
                # Position 0 is the oldest event in the kept window.
                age = valid.size - 1 - np.arange(valid.size)
                weights = np.exp2(-age / self.recency_halflife)
            else:
                weights = np.ones(valid.size)
            np.add.at(counts, valid, weights)
        self.scores = counts.astype(np.float32)
        return {
            "nonzero_tokens": float((counts > 0).sum()),
            "total_weight": float(counts.sum()),
        }

    def score(self, batch: HistoryBatch) -> np.ndarray:
        # The same scores for everyone; personalisation happens only through
        # the seen-item filter applied by the caller.
        return np.tile(self.scores, (len(batch), 1))
