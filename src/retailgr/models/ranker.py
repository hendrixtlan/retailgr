"""The HSTU ranker: multi-task, target-aware, and batched over candidates.

Retrieval answers "which few hundred items are plausible". Ranking answers
"of these, which will this customer click, buy, and keep". They are different
questions and the paper formulates them differently (Table 1 of
https://arxiv.org/abs/2402.17152):

    retrieval  x_i = (Phi_0, a_0), (Phi_1, a_1), ...      items and actions fused
               y_i = Phi_{i+1} if a_{i+1} is positive

    ranking    x_i = Phi_0, a_0, Phi_1, a_1, ...          items and actions INTERLEAVED
               y_i = a_0, none, a_1, none, ...            predict the action on the item

The interleaving is what makes ranking target-aware: a candidate item is
appended as its own position, and the hidden state there predicts the actions
that item would receive. The interaction between the candidate and the history
happens inside the encoder rather than after it, which is the thing a
late-fusion two-tower model cannot do.

Four heads:

    click     the customer engages past a view
    cart      it reaches the basket
    purchase  it is bought
    return    given a purchase, it comes back

The return head is the reason the sequence builder links purchases to their
returns. It is also the only head whose label arrives days late, so it trains
only on matured purchases.

**M-FALCON.** Scoring 300 candidates by running 300 forward passes over the
same history is 300x the work for no reason: the prefix computation is
identical every time. M-FALCON appends all candidates in one pass and shapes
the attention mask and the relative bias so each candidate sees exactly the
history it would have seen alone, and never the other candidates. The
equivalence is not an assumption here - ``tests/test_ranker.py`` asserts that
a batched score equals the one-at-a-time score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from retailgr.actions import ACTION_TO_ID, ACTION_VOCAB_SIZE
from retailgr.evaluation.calibration import HeadCalibrators
from retailgr.io.loaders import SequenceSplit
from retailgr.models.hstu import HSTULayer, RelativeBucketedTimeAndPositionBias
from retailgr.models.sasrec import resolve_device

# Head order is fixed: it is baked into exported bundles.
HEADS: tuple[str, ...] = ("click", "cart", "purchase", "return")

# Which actions count as a positive label for each head. An action not listed
# is a negative example, except for the return head, which has its own label.
HEAD_POSITIVE_ACTIONS: dict[str, tuple[str, ...]] = {
    "click": ("click", "add_to_cart", "purchase"),
    "cart": ("add_to_cart", "purchase"),
    "purchase": ("purchase",),
}

# Label values written by jobs.sequences.with_return_labels.
RETURN_KEPT = 0
RETURN_RETURNED = 1


@dataclass
class RankerScores:
    """Per-candidate head probabilities and the blended score.

    The four head arrays are the *calibrated* probabilities when the ranker
    carries a calibrator, because those are the numbers the blend used and
    therefore the numbers a response or a report should quote. ``raw`` keeps
    the model's own output so the two can be compared without re-running.
    """

    click: np.ndarray
    cart: np.ndarray
    purchase: np.ndarray
    returned: np.ndarray
    blended: np.ndarray
    raw: dict[str, np.ndarray] | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "click": self.click,
            "cart": self.cart,
            "purchase": self.purchase,
            "return": self.returned,
            "blended": self.blended,
        }


@dataclass
class ScoreBlend:
    """How the four heads become one number: expected value.

        score = v_purchase * P(purchase) * (1 - penalty * P(return))
              + v_cart     * P(cart)
              + v_click    * P(click)

    The weights are **relative business value**, not importance dials, and
    that distinction is the whole design. The heads do not share a scale and
    should not: on the synthetic set P(purchase) ranges over 0.0001-0.007
    while P(cart) reaches 0.11, because a purchase genuinely is rarer. A
    weighting that treats those as comparable (purchase 1.0, cart 0.3)
    collapses the blend into "0.3 x cart" and throws away the head that
    matters — measured, not theorised. Expressing the weights as value makes
    the rarity and the worth cancel: a purchase 50x rarer and 50x more
    valuable contributes as much as the cart signal.

    The return term multiplies the purchase term rather than subtracting on
    its own, because return risk only costs anything in proportion to how
    likely the purchase was. An item nobody would buy cannot be returned.
    """

    # Value of one click, cart-add and purchase, in the same unit.
    click: float = 0.005
    cart: float = 0.05
    purchase: float = 1.0
    # Share of a purchase's value destroyed by a certain return. 1.0 means a
    # returned purchase is worth nothing; above 1.0 charges the logistics too.
    return_penalty: float = 1.0

    def apply(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        kept = np.clip(1.0 - self.return_penalty * scores["return"], 0.0, None)
        return (
            self.purchase * scores["purchase"] * kept
            + self.cart * scores["cart"]
            + self.click * scores["click"]
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ScoreBlend:
        data = dict(data or {})
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown score blend keys: {sorted(unknown)}")
        return cls(**data)


def retrieval_negative_pool(
    retrieval_model: Any,
    split: SequenceSplit,
    vocab_size: int,
    pool_size: int = 300,
    max_events: int = 50,
) -> np.ndarray:
    """Retrieval's top candidates per user, to draw the ranker's negatives from.

    The ranker's job is to reorder retrieval's output, so the items it must
    learn to rank *down* are the ones retrieval ranks up. Drawing negatives
    uniformly from the vocabulary instead trains it on a question retrieval
    has already answered.

    Built from the same prefix the training step uses — the history with its
    last event held out, since that event is the positive — and excluding the
    prefix itself, which is what ``exclude_seen`` does at serving time. The
    positive may still appear in the pool; the collision mask in
    ``_candidate_batch`` handles that, and removing it here would make the
    pool's composition depend on the label.

    Returns ``(len(split), pool_size)``, zero-padded for users retrieval
    cannot score. Zero is the padding token, which
    :meth:`HSTURanker._sample_negatives` treats as "fall back to uniform".
    """
    from retailgr.evaluation.metrics import rank_from_scores
    from retailgr.io.loaders import HistoryBatch

    pool = np.zeros((len(split.inputs), pool_size), dtype=np.int64)
    for index, history in enumerate(split.inputs):
        keep = (history > 0) & (history < vocab_size)
        valid = history[keep]
        if valid.size < 2:
            continue
        prefix = valid[-max_events:][:-1]
        if prefix.size == 0:
            continue

        actions = split.actions[index]
        times = split.timestamps[index]
        batch = HistoryBatch(
            tokens=[prefix],
            actions=[
                actions[keep][-max_events:][:-1]
                if actions.size == history.size
                else np.zeros(prefix.size, dtype=np.int64)
            ],
            timestamps=[
                times[keep][-max_events:][:-1]
                if times.size == history.size
                else np.arange(prefix.size, dtype=np.int64)
            ],
        )
        ranked = rank_from_scores(retrieval_model.score(batch)[0], pool_size, prefix)
        pool[index, : ranked.size] = ranked
    return pool


class HSTURankerNet(nn.Module):
    """The encoder over an interleaved item/action sequence, plus four heads."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        attention_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
        max_events: int = 50,
        head_hidden: int = 64,
        num_time_buckets: int = 128,
        use_relative_bias: bool = True,
        use_temporal_bias: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_events = max_events
        # Interleaving doubles the sequence, and one candidate position is
        # appended on top.
        self.max_len = 2 * max_events + 1
        self.use_temporal_bias = use_temporal_bias
        attention_dim = attention_dim or max(1, embedding_dim // num_heads)
        hidden_dim = hidden_dim or max(1, embedding_dim // num_heads)

        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(ACTION_VOCAB_SIZE, embedding_dim, padding_idx=0)
        # Item positions and action positions are different kinds of token;
        # without this the encoder has to infer the alternation from position
        # alone.
        self.kind_embedding = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)

        self.relative_bias = (
            RelativeBucketedTimeAndPositionBias(self.max_len, num_buckets=num_time_buckets)
            if use_relative_bias
            else None
        )
        self.layers = nn.ModuleList(
            [
                HSTULayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dim=attention_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(embedding_dim, head_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_hidden, 1),
                )
                for name in HEADS
            }
        )

        for embedding in (self.item_embedding, self.action_embedding, self.kind_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].fill_(0)

    # -- sequence assembly ----------------------------------------------------

    def interleave(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        timestamps: torch.Tensor,
        candidates: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Build the interleaved sequence, optionally with candidates appended.

        ``tokens``/``actions``/``timestamps``: (B, N) history, left-padded.
        ``candidates``: (B, C) candidate item ids, or None for training.

        Returns ``(items, acts, kinds, times, prefix_len)`` each (B, L), where
        L = 2N for training and 2N + C when candidates are given.
        """
        batch, n = tokens.shape
        device = tokens.device
        prefix_len = 2 * n

        items = torch.zeros(batch, prefix_len, dtype=torch.long, device=device)
        acts = torch.zeros(batch, prefix_len, dtype=torch.long, device=device)
        kinds = torch.zeros(batch, prefix_len, dtype=torch.long, device=device)
        times = torch.zeros(batch, prefix_len, dtype=torch.long, device=device)

        # Even positions carry the item, odd positions carry the action taken
        # on it. Both share the event's timestamp.
        items[:, 0::2] = tokens
        acts[:, 1::2] = actions
        kinds[:, 0::2] = torch.where(tokens > 0, 1, 0)
        kinds[:, 1::2] = torch.where(actions > 0, 2, 0)
        times[:, 0::2] = timestamps
        times[:, 1::2] = timestamps

        if candidates is None:
            return items, acts, kinds, times, prefix_len

        count = candidates.shape[1]
        zeros = torch.zeros(batch, count, dtype=torch.long, device=device)
        # A candidate is an item position with no action yet - predicting that
        # action is the whole task.
        last_time = timestamps[:, -1:].expand(batch, count)
        return (
            torch.cat([items, candidates], dim=1),
            torch.cat([acts, zeros], dim=1),
            torch.cat([kinds, torch.where(candidates > 0, 1, 0)], dim=1),
            torch.cat([times, last_time], dim=1),
            prefix_len,
        )

    # -- attention mask -------------------------------------------------------

    def build_mask(
        self, items: torch.Tensor, acts: torch.Tensor, prefix_len: int
    ) -> torch.Tensor:
        """Causal within the prefix; each candidate sees the prefix and itself.

        Candidates must not attend to one another — that is what makes one
        batched pass equal to many single passes.
        """
        batch, length = items.shape
        device = items.device
        occupied = (items > 0) | (acts > 0)

        mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
        if length > prefix_len:
            candidate_rows = torch.arange(prefix_len, length, device=device)
            # Start from "prefix only", then let each candidate see itself.
            mask[candidate_rows, :] = False
            mask[candidate_rows.unsqueeze(1), torch.arange(prefix_len, device=device)] = True
            mask[candidate_rows, candidate_rows] = True

        mask = mask.unsqueeze(0) & occupied.unsqueeze(1) & occupied.unsqueeze(2)
        # A row that is entirely padding still needs a diagonal, or the layer
        # divides by a zero-weight sum.
        return mask.to(items.dtype if items.dtype.is_floating_point else torch.float32)

    def build_relative_bias(
        self, times: torch.Tensor, prefix_len: int, length: int
    ) -> torch.Tensor | None:
        """``rab`` for the interleaved sequence, with every candidate placed at
        the same position.

        If candidate j sat at position ``prefix_len + j`` it would see the
        history at a different relative distance than candidate 0, and the
        batched score would not match the single-candidate score. The paper
        handles this by "modifying attention masks and rab biases such that the
        attention operations performed for b_m candidates are exactly the
        same"; concretely, every candidate row reuses the row position
        ``prefix_len`` would have had.
        """
        if self.relative_bias is None:
            return None

        batch = times.size(0)
        use_time = self.use_temporal_bias and times is not None

        if length == prefix_len:
            bias = self.relative_bias(times if use_time else None, length=prefix_len)
            return bias.expand(batch, -1, -1) if bias.size(0) == 1 else bias

        # Only ``prefix_len + 1`` positions are ever distinct: the history,
        # plus one candidate slot. Every candidate reuses that slot's row, so
        # the bias table never has to grow with the candidate count.
        reduced_times = (
            torch.cat([times[:, :prefix_len], times[:, prefix_len : prefix_len + 1]], dim=1)
            if use_time
            else None
        )
        reduced = self.relative_bias(reduced_times, length=prefix_len + 1)
        if reduced.size(0) == 1:
            reduced = reduced.expand(batch, -1, -1)

        count = length - prefix_len
        bias = times.new_zeros((batch, length, length), dtype=reduced.dtype)
        bias[:, :prefix_len, :prefix_len] = reduced[:, :prefix_len, :prefix_len]
        # Every candidate sees the history exactly as a lone candidate would.
        bias[:, prefix_len:, :prefix_len] = reduced[
            :, prefix_len : prefix_len + 1, :prefix_len
        ].expand(-1, count, -1)
        # Candidate-to-candidate cells are masked out; only the diagonal (a
        # candidate attending to itself) survives.
        diagonal = torch.arange(prefix_len, length, device=times.device)
        bias[:, diagonal, diagonal] = reduced[:, prefix_len, prefix_len].unsqueeze(1)
        return bias

    # -- forward --------------------------------------------------------------

    def encode(
        self,
        items: torch.Tensor,
        acts: torch.Tensor,
        kinds: torch.Tensor,
        times: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        x = (
            self.item_embedding(items)
            + self.action_embedding(acts)
            + self.kind_embedding(kinds)
        )
        x = self.input_dropout(x)

        length = items.shape[1]
        mask = self.build_mask(items, acts, prefix_len)
        bias = self.build_relative_bias(times, prefix_len, length)
        # A constant, not the actual length. HSTU divides attention weights by
        # a fixed normaliser rather than a softmax denominator, and here that
        # constant has to stay constant: dividing by `prefix_len + C` would
        # make a candidate's score depend on how many other candidates shared
        # its forward pass, which is exactly the equivalence M-FALCON needs.
        # It also keeps training and scoring on the same scale.
        normaliser = float(self.max_len)
        for layer in self.layers:
            x = layer(x, mask, bias, normaliser)
        return self.output_norm(x)

    def head_logits(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits per head, at every position."""
        return {name: self.heads[name](hidden).squeeze(-1) for name in HEADS}

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        timestamps: torch.Tensor,
        candidates: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        items, acts, kinds, times, prefix_len = self.interleave(
            tokens, actions, timestamps, candidates
        )
        hidden = self.encode(items, acts, kinds, times, prefix_len)
        return self.head_logits(hidden), prefix_len


class HSTURanker:
    """Training and scoring for :class:`HSTURankerNet`."""

    name = "hstu_ranker"

    def __init__(self, vocab_size: int, config: dict[str, Any] | None = None):
        config = dict(config or {})
        self.vocab_size = vocab_size
        self.max_events = int(config.get("max_events", config.get("max_len", 50)))
        self.batch_size = int(config.get("batch_size", 64))
        self.epochs = int(config.get("epochs", 8))
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 0.0))
        self.seed = int(config.get("seed", 13))
        self.device = resolve_device(str(config.get("device", "auto")))
        self.micro_batch = int(config.get("candidate_micro_batch", 64))
        self.blend = ScoreBlend.from_dict(config.get("score_blend"))
        # Fitted, not configured: a calibrator is derived from held-out data
        # the same way the weights are, so it is attached after training by
        # ``evaluation.ranking.calibrate_heads`` and travels in the manifest.
        # Until then every head passes through unchanged, which is the
        # behaviour this class had before calibration existed.
        self.calibrators: HeadCalibrators = HeadCalibrators.from_dict(
            config.get("head_calibration")
        )
        # Sampled negatives at the candidate position. Without these the
        # ranker only ever scores items the user did interact with, learns
        # "given engagement, how deep does it go", and cannot order an
        # arbitrary candidate set — which is the whole job. Measured cost of
        # omitting them on the synthetic set: NDCG@10 down 89%.
        self.num_negatives = int(config.get("num_negatives", 32))
        # Share of those negatives drawn from retrieval's own candidates
        # instead of uniformly from the vocabulary.
        #
        # A uniform negative is almost always an item retrieval would never
        # have shown, so "positive vs 256 uniform" is solvable from an item
        # prior — which is the thing retrieval already does better. At serving
        # the ranker sees retrieval's top 300, where every candidate is
        # already plausible and the item prior tells it nothing. Training on
        # the easy version and deploying on the hard one is the standard
        # explanation for a second stage that measures well and reorders
        # badly, and here it is measurable: see the `resolution` column of the
        # calibration report.
        self.hard_negative_fraction = float(config.get("hard_negative_fraction", 0.0))
        self.candidate_loss_weight = float(config.get("candidate_loss_weight", 1.0))
        # Which event becomes the positive the ranker must pick out.
        #   next   -> the last event, as this model always did
        #   unseen -> a random event whose item is not already in the prefix,
        #             which is the only kind of item `exclude_seen` lets
        #             through at serving time. See `_unseen_cuts`.
        self.candidate_positive = str(config.get("candidate_positive", "next")).lower()
        if self.candidate_positive not in {"next", "unseen"}:
            raise ValueError(
                f"candidate_positive must be 'next' or 'unseen', "
                f"got {self.candidate_positive!r}"
            )
        # Weight on the teacher-forced loss — a scalar, or **per head**.
        #
        # Turning it down was the single biggest lever found on this model:
        # on this dataset the teacher-forced task is reproduced by a
        # five-number lookup on how many times the same item just repeated
        # (AUC 0.965 against the model's 0.986), so its gradient was buying
        # capacity for a generator artifact. Dropping it multiplies the
        # model's ordering signal on the *serving slate* by 2.85x, measured
        # over five seeds.
        #
        # It is per-head because a global zero was measured to be wrong. The
        # `return` head is the one whose teacher-forced task is not a funnel
        # artifact — "given a matured purchase, will it come back" is a real
        # prediction — and it is also the only head with no slate-shaped
        # label, because a slate yields at most one matured purchase per
        # user. With the history loss off everywhere it gets one label per
        # user per epoch and collapses to chance: AUC 0.873 -> 0.511. That
        # loss is invisible to the slate experiment, which is exactly why it
        # had to be looked for.
        weight = config.get("history_loss_weight", 1.0)
        if isinstance(weight, dict):
            self.history_loss_weight: dict[str, float] = {
                head: float(weight.get(head, 1.0)) for head in HEADS
            }
        else:
            self.history_loss_weight = dict.fromkeys(HEADS, float(weight))
        # Relative weights inside the multi-task loss. The return head sees
        # far fewer labelled positions, so it is upweighted to stop it being
        # drowned out.
        self.head_weights: dict[str, float] = {
            "click": 1.0,
            "cart": 1.0,
            "purchase": 1.0,
            "return": 2.0,
        }
        self.head_weights.update(config.get("head_weights") or {})

        torch.manual_seed(self.seed)
        self.net = HSTURankerNet(
            vocab_size=vocab_size,
            embedding_dim=int(config.get("hidden_dim", 64)),
            num_blocks=int(config.get("num_blocks", 2)),
            num_heads=int(config.get("num_heads", 2)),
            attention_dim=config.get("attention_dim"),
            hidden_dim=config.get("head_hidden_dim"),
            dropout=float(config.get("dropout", 0.2)),
            max_events=self.max_events,
            head_hidden=int(config.get("head_hidden", 64)),
            num_time_buckets=int(config.get("num_time_buckets", 128)),
            use_relative_bias=bool(config.get("use_relative_bias", True)),
            use_temporal_bias=bool(config.get("use_temporal_bias", True)),
        ).to(self.device)

    # -- data -----------------------------------------------------------------

    def _padded_history(
        self,
        split: SequenceSplit,
        indices: list[int],
        cuts: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Left-pad each user's last ``max_events`` events into dense arrays.

        ``cuts[row]`` truncates that user's history after position ``cut``,
        so the event at ``cut`` becomes the last one. `fit` holds the last
        event out as the candidate positive, so a cut is how the caller
        chooses *which* event the ranker is asked to discriminate. Without
        cuts the behaviour is unchanged: the whole history, positive last.
        """
        size = len(indices)
        n = self.max_events
        tokens = np.zeros((size, n), dtype=np.int64)
        actions = np.zeros((size, n), dtype=np.int64)
        times = np.zeros((size, n), dtype=np.int64)
        returned = np.full((size, n), -1, dtype=np.int64)

        for row, index in enumerate(indices):
            history = split.inputs[index]
            keep = (history > 0) & (history < self.vocab_size)
            valid = history[keep]
            if valid.size == 0:
                continue
            limit = None if cuts is None else cuts[row] + 1
            if limit is not None:
                valid = valid[:limit]
                if valid.size == 0:
                    continue
            tail = valid[-n:]
            offset = n - tail.size
            tokens[row, offset:] = tail

            row_actions = split.actions[index]
            if row_actions.size == history.size:
                kept_actions = row_actions[keep]
                actions[row, offset:] = (
                    kept_actions[:limit][-n:] if limit is not None else kept_actions[-n:]
                )
            row_times = split.timestamps[index]
            if row_times.size == history.size:
                kept_times = row_times[keep]
                selected = (
                    kept_times[:limit][-n:] if limit is not None else kept_times[-n:]
                )
                times[row, offset:] = selected
                if offset:
                    times[row, :offset] = selected[0]
            if index < len(split.returned):
                row_returned = split.returned[index]
                if row_returned.size == history.size:
                    kept_returned = row_returned[keep]
                    returned[row, offset:] = (
                        kept_returned[:limit][-n:] if limit is not None else kept_returned[-n:]
                    )

        return {
            "tokens": torch.from_numpy(tokens),
            "actions": torch.from_numpy(actions),
            "timestamps": torch.from_numpy(times),
            "returned": torch.from_numpy(returned),
        }

    @staticmethod
    def _labels_for(
        actions: torch.Tensor, returned: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Per-head ``(label, mask)`` at each event position.

        The mask is what keeps the return head honest: it is 1 only where the
        purchase has matured, so an unreturnable-yet purchase contributes
        nothing rather than counting as "kept".
        """
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        present = actions > 0

        for head, positives in HEAD_POSITIVE_ACTIONS.items():
            label = torch.zeros_like(actions, dtype=torch.float32)
            for action in positives:
                label = torch.where(
                    actions == ACTION_TO_ID[action], torch.ones_like(label), label
                )
            out[head] = (label, present.float())

        matured = (returned == RETURN_KEPT) | (returned == RETURN_RETURNED)
        return_label = (returned == RETURN_RETURNED).float()
        out["return"] = (return_label, matured.float())
        return out

    # -- training -------------------------------------------------------------

    def _sample_negatives(
        self,
        batch_size: int,
        pool: torch.Tensor | None,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """``num_negatives`` items to score against the true one.

        Mixed rather than purely hard, and the mix is the point. Pure hard
        negatives drop the easy global signal — that most of the catalogue is
        irrelevant — and they carry more false negatives, since retrieval's
        top-k is exactly where the items a customer would have liked but was
        never shown are concentrated. Keeping a uniform share bounds both.
        """
        uniform = torch.randint(
            1, self.vocab_size, (batch_size, self.num_negatives), generator=generator
        )
        if pool is None or self.hard_negative_fraction <= 0:
            return uniform

        hard_count = min(
            self.num_negatives, int(round(self.num_negatives * self.hard_negative_fraction))
        )
        if hard_count <= 0 or pool.size(1) == 0:
            return uniform

        picks = torch.randint(0, pool.size(1), (batch_size, hard_count), generator=generator)
        drawn = torch.gather(pool.cpu(), 1, picks)
        # A pool row can be short or empty for a user retrieval could not
        # score; those slots fall back to the uniform draw rather than to the
        # padding token, which would be a candidate the model never sees.
        drawn = torch.where(drawn > 0, drawn, uniform[:, :hard_count])
        return torch.cat([drawn, uniform[:, hard_count:]], dim=1)

    def _unseen_cuts(
        self, split: SequenceSplit, indices: list[int], generator: torch.Generator
    ) -> tuple[list[int], int]:
        """Positions whose item the customer has **not** already engaged with.

        This exists because of a measurement, and it is the most consequential
        one in this model's history. The ranker's positive was always the last
        event. On this data **52.5% of those positives are items already in
        the prefix**, and the split by action is not a detail:

            last action = view          34.4% already seen
            last action = add_to_cart  100.0% already seen
            last action = purchase     100.0% already seen
            last action = return       100.0% already seen

        A customer cannot buy something they never looked at, so a purchase
        positive is a repeat by construction. Meanwhile serving runs with
        `exclude_seen`, which removes every already-seen item from the slate
        before the ranker is consulted.

        So the `cart`, `purchase` and `return` heads were trained *entirely*
        on a class of item that serving filters out, and asked at request time
        to score a class of item they had never been trained on. That is why
        the purchase head orders the serving slate **below chance** (AUC
        0.465): the feature it learned — "this item repeats the prefix" — is
        anti-correlated with what survives the filter.

        Choosing an unseen cut makes the training positive the same kind of
        item the request path will actually present. It costs supervision:
        users whose every event after the first is a repeat have no such
        position, and they fall back to the last event. The count is reported
        so the trade is visible rather than assumed.
        """
        cuts: list[int] = []
        fallbacks = 0
        for index in indices:
            history = split.inputs[index]
            keep = (history > 0) & (history < self.vocab_size)
            valid = history[keep]
            if valid.size < 2:
                cuts.append(max(int(valid.size) - 1, 0))
                fallbacks += 1
                continue
            # A position is eligible when its item has not appeared before
            # it, and when at least one event precedes it to form a prefix.
            seen: set[int] = {int(valid[0])}
            eligible: list[int] = []
            for position in range(1, int(valid.size)):
                token = int(valid[position])
                if token not in seen:
                    eligible.append(position)
                seen.add(token)
            if not eligible:
                cuts.append(int(valid.size) - 1)
                fallbacks += 1
                continue
            pick = int(
                torch.randint(len(eligible), (1,), generator=generator).item()
            )
            cuts.append(eligible[pick])
        return cuts, fallbacks

    def _candidate_batch(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        returned: torch.Tensor,
        generator: torch.Generator,
        pool: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Build the candidate block: the true next item plus sampled negatives.

        This is the serving condition exactly — a prefix, and a set of
        candidates to order — so the ranker is trained on the task it will be
        asked to do. The true next item is the last event of the history,
        which the caller has already held out of the prefix.
        """
        batch_size = tokens.size(0)
        device = tokens.device
        positives = tokens[:, -1:]  # (B, 1), the held-out next item
        negatives = self._sample_negatives(batch_size, pool, generator).to(device)
        candidates = torch.cat([positives, negatives], dim=1)

        positive_actions = actions[:, -1:]
        positive_returned = returned[:, -1:]
        blank_actions = torch.zeros_like(negatives)

        positive_labels = self._labels_for(positive_actions, positive_returned)
        labels: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for head in HEADS:
            label, mask = positive_labels[head]
            # A negative was never shown to have any action, so every head
            # labels it 0 — except `return`, which is conditioned on a
            # purchase that never happened and is therefore masked out.
            if head == "return":
                negative_label = torch.zeros_like(blank_actions, dtype=torch.float32)
                negative_mask = torch.zeros_like(blank_actions, dtype=torch.float32)
            else:
                negative_label = torch.zeros_like(blank_actions, dtype=torch.float32)
                negative_mask = torch.ones_like(blank_actions, dtype=torch.float32)
            # A positive whose own action is padding carries no label either.
            labels[head] = (
                torch.cat([label, negative_label], dim=1),
                torch.cat([mask, negative_mask], dim=1),
            )
        # Collisions would teach the model that the true item is a negative.
        collision = candidates[:, 1:] == positives
        for head in HEADS:
            label, mask = labels[head]
            mask[:, 1:] = mask[:, 1:].masked_fill(collision, 0.0)
            labels[head] = (label, mask)
        return candidates, labels

    def fit(
        self,
        train: SequenceSplit,
        val: SequenceSplit | None = None,
        hard_negatives: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the ranker.

        ``hard_negatives`` is an optional ``(len(train), pool)`` array of
        retrieval's candidates per user, built by
        :func:`retrieval_negative_pool`. It only has an effect when
        ``hard_negative_fraction`` is above zero, so passing it is always
        safe and the config decides whether it is used.
        """
        usable = [
            index
            for index, history in enumerate(train.inputs)
            if ((history > 0) & (history < self.vocab_size)).sum() >= 2
        ]
        if not usable:
            raise ValueError("no usable histories for the ranker")

        pool_tensor: torch.Tensor | None = None
        if hard_negatives is not None and self.hard_negative_fraction > 0:
            pool_array = np.asarray(hard_negatives, dtype=np.int64)
            if pool_array.shape[0] != len(train.inputs):
                raise ValueError(
                    "hard_negatives must have one row per user in the split, "
                    f"got {pool_array.shape[0]} for {len(train.inputs)}"
                )
            pool_tensor = torch.from_numpy(pool_array)

        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        label_counts = dict.fromkeys(HEADS, 0)
        positive_counts = dict.fromkeys(HEADS, 0)
        candidate_labels_seen = 0
        cut_fallbacks = 0
        cuts_taken = 0
        history_loss: list[float] = []

        self.net.train()
        for epoch in range(self.epochs):
            order = torch.randperm(len(usable), generator=generator).tolist()
            epoch_loss, batches = 0.0, 0
            for start in range(0, len(order), self.batch_size):
                indices = [usable[i] for i in order[start : start + self.batch_size]]
                cuts = None
                if self.candidate_positive == "unseen":
                    cuts, fell_back = self._unseen_cuts(train, indices, generator)
                    cut_fallbacks += fell_back
                    cuts_taken += len(indices)
                batch = self._padded_history(train, indices, cuts=cuts)
                tokens = batch["tokens"].to(self.device)
                actions = batch["actions"].to(self.device)
                times = batch["timestamps"].to(self.device)
                returned = batch["returned"].to(self.device)

                # Hold out the last event: it becomes the positive candidate,
                # and the rest is the prefix. One forward pass then produces
                # both losses, because the candidate block attends to the
                # prefix while the prefix stays causal.
                prefix_tokens = tokens[:, :-1]
                prefix_actions = actions[:, :-1]
                prefix_times = times[:, :-1]
                prefix_returned = returned[:, :-1]

                candidates, candidate_labels = self._candidate_batch(
                    tokens,
                    actions,
                    returned,
                    generator,
                    pool=pool_tensor[indices] if pool_tensor is not None else None,
                )
                logits, prefix_len = self.net(
                    prefix_tokens, prefix_actions, prefix_times, candidates
                )
                observed = self._labels_for(prefix_actions, prefix_returned)

                loss = tokens.new_zeros((), dtype=torch.float32)
                for head in HEADS:
                    # (a) Teacher-forced: at each prefix item position 2i,
                    #     predict the action actually taken. This is what
                    #     calibrates the heads.
                    label, mask = observed[head]
                    head_logits = logits[head][:, :prefix_len][:, 0::2]
                    if head_logits.shape[1] != label.shape[1]:  # pragma: no cover
                        width = min(head_logits.shape[1], label.shape[1])
                        head_logits, label, mask = (
                            head_logits[:, :width],
                            label[:, :width],
                            mask[:, :width],
                        )
                    if mask.sum() > 0:
                        per_position = F.binary_cross_entropy_with_logits(
                            head_logits, label, reduction="none"
                        )
                        loss = loss + self.history_loss_weight[head] * self.head_weights[head] * (
                            (per_position * mask).sum() / mask.sum()
                        )
                        if epoch == 0:
                            label_counts[head] += int(mask.sum())
                            positive_counts[head] += int((label * mask).sum())

                    # (b) Discrimination: the true next item against sampled
                    #     negatives, at the candidate positions. This is what
                    #     makes the heads able to *order* anything.
                    candidate_label, candidate_mask = candidate_labels[head]
                    candidate_logits = logits[head][:, prefix_len:]
                    if candidate_mask.sum() > 0:
                        per_candidate = F.binary_cross_entropy_with_logits(
                            candidate_logits, candidate_label, reduction="none"
                        )
                        loss = loss + self.candidate_loss_weight * self.head_weights[head] * (
                            (per_candidate * candidate_mask).sum() / candidate_mask.sum()
                        )
                        if epoch == 0:
                            candidate_labels_seen += int(candidate_mask.sum())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            history_loss.append(epoch_loss / max(batches, 1))

        self.net.eval()
        stats: dict[str, float] = {
            "users_trained": float(len(usable)),
            "negatives_per_positive": float(self.num_negatives),
            "hard_negative_fraction": (
                self.hard_negative_fraction if pool_tensor is not None else 0.0
            ),
            "candidate_labels": float(candidate_labels_seen),
            "candidate_positive_unseen": 1.0 if self.candidate_positive == "unseen" else 0.0,
            "history_loss_weight_mean": float(
                sum(self.history_loss_weight.values()) / len(self.history_loss_weight)
            ),
            # How often no unseen position existed and the last event was
            # used anyway. A high share means this data is mostly repeats and
            # the serving filter is throwing away most of what it could show.
            "cut_fallback_share": (
                round(cut_fallbacks / cuts_taken, 4) if cuts_taken else 0.0
            ),
            "first_epoch_loss": round(history_loss[0], 5),
            "final_epoch_loss": round(history_loss[-1], 5),
            "epochs": float(self.epochs),
            "parameters": float(sum(p.numel() for p in self.net.parameters())),
        }
        for head in HEADS:
            stats[f"{head}_labels"] = float(label_counts[head])
            stats[f"{head}_positive_rate"] = (
                round(positive_counts[head] / label_counts[head], 5)
                if label_counts[head]
                else 0.0
            )
        return stats

    # -- scoring --------------------------------------------------------------

    @torch.no_grad()
    def score_candidates(
        self,
        tokens: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        returned: np.ndarray | None,
        candidates: np.ndarray,
    ) -> RankerScores:
        """Score ``candidates`` for one user, in micro-batches (M-FALCON)."""
        self.net.eval()
        split = SequenceSplit(
            user_ids=["_"],
            inputs=[np.asarray(tokens, dtype=np.int64)],
            targets=[np.zeros(0, dtype=np.int64)],
            actions=[np.asarray(actions, dtype=np.int64)],
            timestamps=[np.asarray(timestamps, dtype=np.int64)],
            returned=[
                np.asarray(returned, dtype=np.int64)
                if returned is not None
                else np.full(len(tokens), -1, dtype=np.int64)
            ],
        )
        batch = self._padded_history(split, [0])
        history_tokens = batch["tokens"].to(self.device)
        history_actions = batch["actions"].to(self.device)
        history_times = batch["timestamps"].to(self.device)

        candidates = np.asarray(candidates, dtype=np.int64).reshape(-1)
        collected: dict[str, list[np.ndarray]] = {name: [] for name in HEADS}

        for start in range(0, candidates.size, self.micro_batch):
            chunk = candidates[start : start + self.micro_batch]
            candidate_tensor = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            logits, prefix_len = self.net(
                history_tokens, history_actions, history_times, candidate_tensor
            )
            for name in HEADS:
                probabilities = torch.sigmoid(logits[name][0, prefix_len:])
                collected[name].append(probabilities.float().cpu().numpy())

        raw = {
            name: (
                np.concatenate(collected[name])
                if collected[name]
                else np.zeros(0, dtype=np.float32)
            )
            for name in HEADS
        }
        # Calibrate before blending, never after. The blend combines heads
        # whose scales differ by orders of magnitude, so a correction applied
        # to the blended number cannot recover the per-head weighting the
        # operator asked for — by then the heads have already been summed at
        # the wrong relative sizes.
        scores = self.calibrators.apply(raw)
        return RankerScores(
            click=scores["click"],
            cart=scores["cart"],
            purchase=scores["purchase"],
            returned=scores["return"],
            blended=self.blend.apply(scores),
            raw=raw,
        )
