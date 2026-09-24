"""Retrieval: a user vector, then the nearest items.

The encoder turns the user's history into one vector; retrieval finds the top-K
items whose embeddings score highest against it. Splitting the two is what
makes the stage swappable: an exact matmul is right for a few thousand items,
FAISS or a vector database for a few million, and the code above this module
does not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Candidate:
    token_id: int
    token: str
    score: float


class RetrievalIndex(Protocol):
    def search(self, user_vector: np.ndarray, k: int, exclude: set[int]) -> list[int]: ...

    @property
    def size(self) -> int: ...


class ExactRetrievalIndex:
    """Brute-force inner product over the item embedding table.

    For catalogs up to a few hundred thousand items this is not a compromise:
    one BLAS matmul is faster than an approximate index and returns the true
    top-K. Index 0 is the padding token and is never returned.
    """

    def __init__(self, item_embeddings: np.ndarray):
        if item_embeddings.ndim != 2:
            raise ValueError(f"expected a 2-D embedding table, got {item_embeddings.shape}")
        self.embeddings = np.ascontiguousarray(item_embeddings, dtype=np.float32)

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    def scores_for(self, user_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(user_vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"user vector has dim {vector.shape[0]}, index has {self.embeddings.shape[1]}"
            )
        return self.embeddings @ vector

    def search(self, user_vector: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
        scores = self.scores_for(user_vector).copy()
        scores[0] = -np.inf  # padding
        if exclude:
            indices = np.fromiter(
                (i for i in exclude if 0 <= i < scores.shape[0]), dtype=np.int64
            )
            if indices.size:
                scores[indices] = -np.inf
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-scores[top])] if np.isfinite(scores[i])]

    def search_with_scores(
        self, user_vector: np.ndarray, k: int, exclude: set[int] | None = None
    ) -> list[tuple[int, float]]:
        scores = self.scores_for(user_vector)
        ranked = self.search(user_vector, k, exclude)
        return [(token_id, float(scores[token_id])) for token_id in ranked]


class FaissRetrievalIndex:
    """FAISS-backed retrieval, for catalogs where the exact matmul stops fitting
    the latency budget.

    Kept behind the same interface and built only on request. Two things about
    it are measured rather than assumed, because the first time this class was
    ever executed it turned out to be wrong about both.

    **Recall.** An inverted-file index only searches the ``nprobe`` cells
    nearest the query. The original ``nprobe = nlist // 10`` returned **41% of
    the true top-20** on a 50,000-item catalogue — it silently discarded three
    candidates in five, and a candidate retrieval never returns is one the
    ranker cannot recover. Measured on that catalogue, at ``nlist = 100``:

        nprobe   10 -> 41% recall    25 -> 68%    50 -> 89%    75 -> 98%

    So the default is half the cells, not a tenth, and ``nprobe`` is a
    parameter rather than a buried heuristic. Use :func:`measure_recall` to
    tune it on your own embeddings; the numbers above are this project's, not
    yours.

    **When it is worth using at all.** Also measured, same hardware, recall
    held at or above 90%:

        50,000 items   exact 0.40 ms   faiss 0.41 ms   0.99x
        200,000        1.40 ms         1.21 ms         1.15x
        500,000        3.77 ms         2.79 ms         1.35x
        1,000,000      10.36 ms        6.46 ms         1.60x

    Below roughly 100,000 items the exact BLAS matmul is faster *and* perfect,
    so FAISS costs recall for nothing. This project's own vocabulary is under
    a thousand tokens, which is why ``exact`` is the default and this class
    exists for the catalog size it is documented for.
    """

    def __init__(
        self,
        item_embeddings: np.ndarray,
        nlist: int | None = None,
        nprobe: int | None = None,
    ):
        try:
            import faiss
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "faiss is not installed. Install the serving extra "
                "(pip install -e '.[serving-faiss]') or use the exact index."
            ) from error
        self.embeddings = np.ascontiguousarray(item_embeddings, dtype=np.float32)
        dimension = self.embeddings.shape[1]
        count = self.embeddings.shape[0]
        # FAISS's own guidance is a cell count around sqrt(n); 4*sqrt(n) keeps
        # cells small enough that probing half of them is still cheap.
        if nlist is None:
            nlist = max(1, min(65536, int(4 * np.sqrt(max(count, 1)))))
        self.nlist = int(nlist)

        if count < 4 * self.nlist:
            # Too few vectors to train a coarse quantiser; flat is correct,
            # exact, and at this size faster than the alternative anyway.
            self.index = faiss.IndexFlatIP(dimension)
            self.nprobe = self.nlist
        else:
            quantiser = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(
                quantiser, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(self.embeddings)
            self.nprobe = int(nprobe) if nprobe else max(1, self.nlist // 2)
            self.index.nprobe = self.nprobe
        self.index.add(self.embeddings)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def search(self, user_vector: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
        query = np.asarray(user_vector, dtype=np.float32).reshape(1, -1)
        # Over-fetch so exclusions cannot empty the list.
        fetch = min(self.size, k + len(exclude or ()) + 1)
        _, indices = self.index.search(query, fetch)
        excluded = exclude or set()
        out = [int(i) for i in indices[0] if i > 0 and int(i) not in excluded]
        return out[:k]


def build_index(
    item_embeddings: np.ndarray,
    kind: str = "exact",
    nlist: int | None = None,
    nprobe: int | None = None,
) -> RetrievalIndex:
    kind = kind.lower()
    if kind == "exact":
        return ExactRetrievalIndex(item_embeddings)
    if kind == "faiss":
        return FaissRetrievalIndex(item_embeddings, nlist=nlist, nprobe=nprobe)
    raise ValueError(f"unknown retrieval index '{kind}'; expected 'exact' or 'faiss'")


def measure_recall(
    index: RetrievalIndex,
    item_embeddings: np.ndarray,
    k: int = 20,
    queries: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """What share of the true top-``k`` does this index actually return?

    An approximate index trades recall for latency, and the exchange rate
    depends on the embeddings — how clustered they are, how many there are,
    what ``nprobe`` is set to. Inheriting someone else's ``nprobe`` is
    inheriting their exchange rate.

    Recall lost here cannot be recovered downstream: the ranker reorders the
    candidates it is given and has no way to ask for the one that was never
    returned. So this is the number to tune against, not latency alone.
    """
    exact = ExactRetrievalIndex(item_embeddings)
    rng = np.random.default_rng(seed)
    rows = rng.choice(item_embeddings.shape[0], size=min(queries, item_embeddings.shape[0]),
                      replace=False)

    scores: list[float] = []
    for row in rows:
        vector = item_embeddings[row]
        truth = set(exact.search(vector, k=k, exclude=set()))
        if not truth:
            continue
        found = set(index.search(vector, k=k, exclude=set()))
        scores.append(len(truth & found) / len(truth))
    if not scores:
        return {"queries": 0, "k": k, "recall": float("nan")}
    return {
        "queries": len(scores),
        "k": k,
        "recall": round(float(np.mean(scores)), 4),
        "worst_query_recall": round(float(np.min(scores)), 4),
    }
