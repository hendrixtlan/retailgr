"""Common interface every model implements.

``score`` receives a :class:`HistoryBatch`, not a bare list of token arrays, so
that a model which reads actions or timestamps needs no special casing in the
evaluation loop. Item-only models simply ignore the extra arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from retailgr.io.loaders import HistoryBatch, SequenceSplit


class Recommender(ABC):
    """A model that turns user histories into ranked token ids."""

    name: str = "recommender"

    @abstractmethod
    def fit(self, train: SequenceSplit, val: SequenceSplit | None = None) -> dict[str, float]:
        """Train the model. Returns training diagnostics for the run report."""

    @abstractmethod
    def score(self, batch: HistoryBatch) -> np.ndarray:
        """Score every token for each history in ``batch``.

        Returns an array of shape ``(len(batch), vocab_size)``. Index 0 is the
        padding token and is masked out downstream.
        """
