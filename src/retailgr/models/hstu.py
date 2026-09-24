"""HSTU (Hierarchical Sequential Transduction Unit) in PyTorch.

From Zhai et al., "Actions Speak Louder than Words: Trillion-Parameter
Sequential Transducers for Generative Recommendations", ICML 2024
(https://arxiv.org/abs/2402.17152). This implementation follows the paper's
Equations (1)-(3) and was checked line by line against the authors' reference
code (https://github.com/meta-recsys/generative-recommenders, Apache-2.0),
specifically ``SequentialTransductionUnitJagged`` and
``RelativeBucketedTimeAndPositionBasedBias``.

    U(X), V(X), Q(X), K(X) = Split(phi1(f1(X)))                      (1)
    A(X)V(X) = phi2( Q(X)K(X)^T + rab^{p,t} ) V(X)                   (2)
    Y(X) = f2( Norm(A(X)V(X)) (*) U(X) )                             (3)

with phi1 = phi2 = SiLU, Norm = layer norm, and residual connections between
layers.

Three choices here are not interchangeable with a standard Transformer, and
each is load-bearing:

1. **No softmax.** Attention weights are ``silu(QK^T + rab) / N``, normalised
   by a constant sequence length rather than a data-dependent denominator. The
   paper's motivation is that the *number* of prior related events carries
   intensity of preference, which softmax normalises away. Their ablation puts
   HR@10 at .0617 with softmax against .0893 without.
2. **The causal mask is multiplicative and applied after the activation**, not
   as ``-inf`` before it. With no softmax there is nothing to renormalise, so
   masking means multiplying by zero.
3. **Actions are a modality, not metadata.** Following Table 1 of the paper,
   the retrieval formulation fuses ``(item, action)`` at each position, and a
   position only supervises the next item when that item's action was positive.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from retailgr.actions import ACTION_VOCAB_SIZE, positive_action_ids
from retailgr.io.loaders import HistoryBatch, SequenceSplit
from retailgr.models.base import Recommender
from retailgr.models.sasrec import resolve_device
from retailgr.models.training import EarlyStopping, EarlyStoppingConfig


class RelativeBucketedTimeAndPositionBias(nn.Module):
    """``rab^{p,t}``: a learned relative positional bias plus a learned bias
    over log-bucketed time gaps, shared across heads.

    The time bucketing reproduces the reference implementation:
    ``bucket = clamp((log(|dt|.clamp(min=1)) / divisor).long(), 0, num_buckets)``
    with ``divisor = 0.301``. The gaps are computed against the *next*
    timestamp, matching the causal setup there.
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int = 128,
        time_bucket_divisor: float = 0.301,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_buckets = num_buckets
        self.time_bucket_divisor = float(time_bucket_divisor)
        self.position_weight = nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0.0, std=0.02)
        )
        self.time_weight = nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0.0, std=0.02)
        )

    def _positional_bias(self, length: int | None = None) -> torch.Tensor:
        """Expand the ``2N-1`` parameters into an ``(1, L, L)`` Toeplitz matrix.

        The top-left ``L x L`` block of the full matrix is itself Toeplitz with
        the same relative distances, so a shorter sequence slices rather than
        needing its own parameters.
        """
        n = self.max_seq_len
        padded = F.pad(self.position_weight[: 2 * n - 1], [0, n]).repeat(n)
        padded = padded[..., :-n].reshape(1, n, 3 * n - 2)
        radius = (2 * n - 1) // 2
        full = padded[..., radius:-radius]
        if length is None or length == n:
            return full
        if length > n:
            raise ValueError(
                f"sequence length {length} exceeds the bias table's {n}; "
                "raise max_seq_len"
            )
        return full[:, :length, :length]

    def forward(
        self, timestamps: torch.Tensor | None, length: int | None = None
    ) -> torch.Tensor:
        """``timestamps``: ``(B, L)`` int64 seconds, or None for position only.

        ``L`` may be shorter than ``max_seq_len``: the ranker's interleaved
        sequence grows and shrinks with the number of candidates appended.
        """
        if timestamps is None:
            return self._positional_bias(length)

        batch, actual = timestamps.shape
        if length is not None and length != actual:
            raise ValueError(f"timestamps have length {actual}, expected {length}")
        positional = self._positional_bias(actual)

        # Extend by one so gaps are measured against the following event.
        extended = torch.cat([timestamps, timestamps[:, actual - 1 : actual]], dim=1)
        gaps = extended[:, 1:].unsqueeze(2) - extended[:, :-1].unsqueeze(1)
        buckets = torch.clamp(
            (torch.log(torch.abs(gaps).clamp(min=1.0)) / self.time_bucket_divisor).long(),
            min=0,
            max=self.num_buckets,
        ).detach()
        temporal = torch.index_select(
            self.time_weight, dim=0, index=buckets.reshape(-1)
        ).view(batch, actual, actual)
        return positional + temporal


class HSTULayer(nn.Module):
    """One HSTU layer: Equations (1), (2) and (3) plus a residual connection."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 2,
        attention_dim: int = 32,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.eps = eps
        # 1/sqrt(d_qk), as in the reference STULayer's attn_alpha.
        self.attention_alpha = 1.0 / (attention_dim**0.5)

        # f1: a single linear producing U, V (hidden_dim) and Q, K (attention_dim)
        # for every head in one matmul.
        uvqk_out = (hidden_dim * 2 + attention_dim * 2) * num_heads
        self.uvqk = nn.Linear(embedding_dim, uvqk_out, bias=True)
        nn.init.normal_(self.uvqk.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.uvqk.bias)

        # f2: back to the model dimension.
        self.output = nn.Linear(hidden_dim * num_heads, embedding_dim, bias=True)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_bias: torch.Tensor | None,
        normaliser: float | torch.Tensor,
    ) -> torch.Tensor:
        """``x``: (B, N, D). ``attention_mask``: (B, N, N) with 1 where
        attention is allowed. ``relative_bias``: (B, N, N) or (1, N, N).
        ``normaliser``: a scalar, or a (B, 1, 1, 1) tensor for per-sequence
        normalisation."""
        batch, seq_len, _ = x.shape
        heads, attn_dim, hidden = self.num_heads, self.attention_dim, self.hidden_dim

        # -- Equation (1): pre-norm, one projection, SiLU, split ---------------
        normed = F.layer_norm(x, normalized_shape=[self.embedding_dim], eps=self.eps)
        projected = F.silu(self.uvqk(normed))
        u, v, q, k = torch.split(
            projected,
            [hidden * heads, hidden * heads, attn_dim * heads, attn_dim * heads],
            dim=-1,
        )
        q = q.view(batch, seq_len, heads, attn_dim)
        k = k.view(batch, seq_len, heads, attn_dim)
        v = v.view(batch, seq_len, heads, hidden)

        # -- Equation (2): pointwise aggregated attention ---------------------
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) * self.attention_alpha
        if relative_bias is not None:
            scores = scores + relative_bias.unsqueeze(1)  # shared across heads
        # SiLU instead of softmax, then a constant normaliser. Masking comes
        # after the activation and is multiplicative: silu(0) != 0, so masking
        # before it would leak a constant from disallowed positions.
        weights = F.silu(scores) / normaliser
        weights = weights * attention_mask.unsqueeze(1)

        attended = torch.einsum("bhnm,bmhd->bnhd", weights, v)
        attended = attended.reshape(batch, seq_len, heads * hidden)

        # -- Equation (3): normalise, gate with U, project, residual ----------
        gated = u * F.layer_norm(
            attended, normalized_shape=[hidden * heads], eps=self.eps
        )
        projected_out = self.output(F.dropout(gated, p=self.dropout, training=self.training))
        return projected_out + x


class HSTUNet(nn.Module):
    """Item + action embeddings, then a stack of HSTU layers."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        attention_dim: int | None = None,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
        max_len: int = 50,
        num_time_buckets: int = 128,
        use_relative_bias: bool = True,
        use_temporal_bias: bool = True,
        use_actions: bool = True,
        normalise_by: str = "max_len",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.use_temporal_bias = use_temporal_bias
        self.use_actions = use_actions
        if normalise_by not in ("max_len", "sequence_len"):
            raise ValueError(
                f"normalise_by must be 'max_len' or 'sequence_len', got {normalise_by!r}"
            )
        self.normalise_by = normalise_by
        attention_dim = attention_dim or max(1, embedding_dim // num_heads)
        hidden_dim = hidden_dim or max(1, embedding_dim // num_heads)

        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # The action modality. Padding id 0 contributes nothing.
        self.action_embedding = nn.Embedding(ACTION_VOCAB_SIZE, embedding_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)

        self.relative_bias = (
            RelativeBucketedTimeAndPositionBias(max_len, num_buckets=num_time_buckets)
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

        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)
            self.action_embedding.weight[0].fill_(0)

    def forward(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tokens``/``actions``/``timestamps``: (B, N). Returns (B, N, D)."""
        batch, seq_len = tokens.shape
        valid = tokens > 0

        # Table 1, retrieval: each position is the fused (item, action) pair.
        x = self.item_embedding(tokens)
        if self.use_actions:
            x = x + self.action_embedding(actions)
        x = self.input_dropout(x)

        # Allowed: causal, and neither endpoint is padding.
        causal = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=tokens.device)
        )
        mask = causal.unsqueeze(0) & valid.unsqueeze(1) & valid.unsqueeze(2)
        mask = mask.to(x.dtype)

        relative_bias = None
        if self.relative_bias is not None:
            relative_bias = self.relative_bias(
                timestamps if (self.use_temporal_bias and timestamps is not None) else None
            )
            if relative_bias.size(0) == 1 and batch > 1:
                relative_bias = relative_bias.expand(batch, -1, -1)

        # The reference implementation divides by the padded window length.
        # That is faithful, but it attenuates users whose history is much
        # shorter than the window — common outside web-scale feeds — so
        # dividing by each user's own length is available as an option.
        normaliser: float | torch.Tensor
        if self.normalise_by == "sequence_len":
            lengths = valid.sum(dim=1).clamp(min=1).to(x.dtype)
            normaliser = lengths.view(batch, 1, 1, 1)
        else:
            normaliser = float(seq_len)
        for layer in self.layers:
            x = layer(x, mask, relative_bias, normaliser)
        return self.output_norm(x)

    def logits_for(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits, tied to the item embedding table."""
        return hidden @ self.item_embedding.weight.T


class HSTUModel(Recommender):
    """HSTU wired into the Stage 1 harness, for the retrieval task."""

    name = "hstu"

    def __init__(self, vocab_size: int, config: dict[str, Any] | None = None):
        config = dict(config or {})
        self.vocab_size = vocab_size
        self.max_len = int(config.get("max_len", 50))
        self.batch_size = int(config.get("batch_size", 128))
        self.epochs = int(config.get("epochs", 8))
        # Read once here so a typo in the block fails at construction, not
        # halfway through a run.
        self.early_stopping = EarlyStoppingConfig.from_config(config)
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 0.0))
        self.loss_kind = str(config.get("loss", "full_softmax"))
        self.num_negatives = int(config.get("num_negatives", 128))
        self.seed = int(config.get("seed", 13))
        self.device = resolve_device(str(config.get("device", "auto")))
        self.stride = int(config.get("stride", max(1, self.max_len // 2)))
        self.positive_actions = positive_action_ids(config.get("positive_actions"))
        # With no positive-action filter every next item is a target, which is
        # the item-only setup and makes the action modality the only difference
        # from SASRec.
        self.supervise_positive_only = bool(config.get("supervise_positive_only", True))

        torch.manual_seed(self.seed)
        self.net = HSTUNet(
            vocab_size=vocab_size,
            embedding_dim=int(config.get("hidden_dim", 64)),
            num_blocks=int(config.get("num_blocks", 2)),
            num_heads=int(config.get("num_heads", 2)),
            attention_dim=config.get("attention_dim"),
            hidden_dim=config.get("head_hidden_dim"),
            dropout=float(config.get("dropout", 0.2)),
            max_len=self.max_len,
            num_time_buckets=int(config.get("num_time_buckets", 128)),
            use_relative_bias=bool(config.get("use_relative_bias", True)),
            use_temporal_bias=bool(config.get("use_temporal_bias", True)),
            use_actions=bool(config.get("use_actions", True)),
            normalise_by=str(config.get("normalise_by", "max_len")),
        ).to(self.device)

    # -- data -----------------------------------------------------------------

    def _windows(self, split: SequenceSplit) -> dict[str, np.ndarray]:
        """Cut histories into fixed windows of tokens, actions and timestamps.

        Each window is ``max_len + 1`` long: the first ``max_len`` positions are
        the input, shifted by one they are the labels.
        """
        span = self.max_len + 1
        tokens_out: list[np.ndarray] = []
        actions_out: list[np.ndarray] = []
        times_out: list[np.ndarray] = []

        for tokens, actions, times in zip(
            split.inputs, split.actions, split.timestamps, strict=True
        ):
            keep = (tokens > 0) & (tokens < self.vocab_size)
            tokens_valid = tokens[keep]
            if tokens_valid.size < 2:
                continue
            actions_valid = actions[keep] if actions.size == tokens.size else np.zeros_like(
                tokens_valid
            )
            times_valid = (
                times[keep]
                if times.size == tokens.size
                else np.arange(tokens_valid.size, dtype=np.int64)
            )

            starts: list[int] = []
            if tokens_valid.size <= span:
                starts.append(0)
            else:
                cursor = tokens_valid.size - span
                while cursor >= 0:
                    starts.append(cursor)
                    cursor -= self.stride

            for start in starts:
                end = min(start + span, tokens_valid.size)
                window_tokens = tokens_valid[start:end]
                window_actions = actions_valid[start:end]
                window_times = times_valid[start:end]
                pad = span - window_tokens.size
                if pad > 0:
                    # Left-pad, so the newest event is always at the end.
                    window_tokens = np.concatenate(
                        [np.zeros(pad, dtype=np.int64), window_tokens]
                    )
                    window_actions = np.concatenate(
                        [np.zeros(pad, dtype=np.int64), window_actions]
                    )
                    first = window_times[0] if window_times.size else 0
                    window_times = np.concatenate(
                        [np.full(pad, first, dtype=np.int64), window_times]
                    )
                tokens_out.append(window_tokens)
                actions_out.append(window_actions)
                times_out.append(window_times)

        if not tokens_out:
            empty = np.zeros((0, span), dtype=np.int64)
            return {"tokens": empty, "actions": empty.copy(), "timestamps": empty.copy()}
        return {
            "tokens": np.stack(tokens_out),
            "actions": np.stack(actions_out),
            "timestamps": np.stack(times_out),
        }

    def _padded(self, batch: HistoryBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = len(batch)
        tokens = np.zeros((size, self.max_len), dtype=np.int64)
        actions = np.zeros((size, self.max_len), dtype=np.int64)
        times = np.zeros((size, self.max_len), dtype=np.int64)

        for row in range(size):
            history = batch.tokens[row]
            keep = (history > 0) & (history < self.vocab_size)
            valid = history[keep]
            if valid.size == 0:
                continue
            tail = valid[-self.max_len :]
            offset = self.max_len - tail.size
            tokens[row, offset:] = tail

            row_actions = batch.actions[row]
            if row_actions.size == history.size:
                actions[row, offset:] = row_actions[keep][-self.max_len :]
            row_times = batch.timestamps[row]
            if row_times.size == history.size:
                selected = row_times[keep][-self.max_len :]
                times[row, offset:] = selected
                if offset:
                    times[row, :offset] = selected[0]
        return (
            torch.from_numpy(tokens),
            torch.from_numpy(actions),
            torch.from_numpy(times),
        )

    # -- training -------------------------------------------------------------

    def _loss(
        self, hidden: torch.Tensor, labels: torch.Tensor, label_actions: torch.Tensor
    ) -> torch.Tensor:
        mask = labels > 0
        if self.supervise_positive_only and self.positive_actions:
            positive = torch.zeros_like(mask)
            for action_id in self.positive_actions:
                positive |= label_actions == action_id
            mask = mask & positive
        if not bool(mask.any()):
            return hidden.sum() * 0.0

        selected = hidden[mask]
        positives = labels[mask]

        if self.loss_kind == "sampled_softmax":
            negatives = torch.randint(
                1, self.vocab_size, (selected.size(0), self.num_negatives), device=selected.device
            )
            candidates = torch.cat([positives.unsqueeze(1), negatives], dim=1)
            embeddings = self.net.item_embedding(candidates)
            logits = torch.einsum("bd,bcd->bc", selected, embeddings)
            collision = candidates[:, 1:] == positives.unsqueeze(1)
            logits[:, 1:] = logits[:, 1:].masked_fill(collision, float("-inf"))
            target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            return F.cross_entropy(logits, target)

        logits = self.net.logits_for(selected)
        logits[:, 0] = float("-inf")
        return F.cross_entropy(logits, positives)

    def fit(self, train: SequenceSplit, val: SequenceSplit | None = None) -> dict[str, float]:
        windows = self._windows(train)
        if windows["tokens"].shape[0] == 0:
            raise ValueError("no training windows produced; check min_events_per_user")

        tokens = torch.from_numpy(windows["tokens"])
        actions = torch.from_numpy(windows["actions"])
        times = torch.from_numpy(windows["timestamps"])

        generator = torch.Generator().manual_seed(self.seed)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        supervised_positions = 0
        history: list[float] = []
        # With early stopping on and a validation split to watch, the epoch
        # count is a ceiling rather than a schedule. Without either, this is
        # exactly the fixed-epoch loop it always was.
        monitor = (
            EarlyStopping(self.early_stopping)
            if self.early_stopping.enabled and val is not None and len(val)
            else None
        )
        epoch_limit = self.early_stopping.max_epochs if monitor else self.epochs
        self.net.train()
        for epoch in range(epoch_limit):
            order = torch.randperm(tokens.size(0), generator=generator)
            epoch_loss, batches = 0.0, 0
            for start in range(0, tokens.size(0), self.batch_size):
                index = order[start : start + self.batch_size]
                batch_tokens = tokens[index].to(self.device)
                batch_actions = actions[index].to(self.device)
                batch_times = times[index].to(self.device)

                input_tokens = batch_tokens[:, :-1]
                input_actions = batch_actions[:, :-1]
                input_times = batch_times[:, :-1]
                labels = batch_tokens[:, 1:]
                label_actions = batch_actions[:, 1:]

                hidden = self.net(input_tokens, input_actions, input_times)
                loss = self._loss(hidden, labels, label_actions)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
                if epoch == 0:
                    supervised_positions += int((labels > 0).sum())
            history.append(epoch_loss / max(batches, 1))
            if monitor is not None and monitor.check(epoch, self, self.net, val):
                break

        stats: dict[str, Any] = {
            "train_windows": float(windows["tokens"].shape[0]),
            "label_positions": float(supervised_positions),
            "first_epoch_loss": round(history[0], 5),
            "final_epoch_loss": round(history[-1], 5),
            "epochs": float(len(history)),
            "train_loss_curve": [round(value, 5) for value in history],
            "parameters": float(sum(p.numel() for p in self.net.parameters())),
        }
        if monitor is not None:
            monitor.restore(self.net)
            stats.update(monitor.summary(epochs_run=len(history)))
        else:
            self.net.eval()
            stats["early_stopping"] = False
        return stats

    # -- inference ------------------------------------------------------------

    @torch.no_grad()
    def score(self, batch: HistoryBatch) -> np.ndarray:
        self.net.eval()
        out = np.zeros((len(batch), self.vocab_size), dtype=np.float32)
        for start in range(0, len(batch), self.batch_size):
            chunk = HistoryBatch(
                tokens=batch.tokens[start : start + self.batch_size],
                actions=batch.actions[start : start + self.batch_size],
                timestamps=batch.timestamps[start : start + self.batch_size],
            )
            tokens, actions, times = self._padded(chunk)
            hidden = self.net(
                tokens.to(self.device), actions.to(self.device), times.to(self.device)
            )[:, -1, :]
            out[start : start + len(chunk)] = self.net.logits_for(hidden).float().cpu().numpy()
        return out
