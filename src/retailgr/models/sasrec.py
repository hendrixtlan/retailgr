"""SASRec baseline in PyTorch.

A causal self-attention model over the user's item sequence, trained to predict
the next token. It is the reference point HSTU has to beat: same data, same
split, same metrics. It reads items only — no actions, no timestamps — which is
exactly the gap HSTU is meant to close.

Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation" (2018).
The sampled-softmax loss follows the setup used in the Generative Recommenders
paper rather than the original binary cross-entropy, so the comparison with
HSTU stays apples-to-apples.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from retailgr.io.loaders import HistoryBatch, SequenceSplit
from retailgr.models.base import Recommender
from retailgr.models.training import EarlyStopping, EarlyStoppingConfig


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


class _Block(nn.Module):
    """One transformer block: causal attention, then a pointwise feed-forward."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        # Rows that are entirely padding produce NaNs; zero them out.
        attended = torch.nan_to_num(attended)
        x = x + attended
        return x + self.ffn(self.ffn_norm(x))


class SASRecNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        max_len: int = 50,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.item_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_Block(hidden_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].fill_(0)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """``sequences``: (batch, seq_len) of token ids, 0 = padding."""
        batch, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        x = self.item_embedding(sequences) * math.sqrt(self.item_embedding.embedding_dim)
        x = self.dropout(x + self.position_embedding(positions))

        # Both masks are boolean, where True means "do not attend"; mixing a
        # float mask with a boolean one is deprecated in PyTorch.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequences.device), diagonal=1
        )
        padding_mask = sequences == 0
        for block in self.blocks:
            x = block(x, causal_mask, padding_mask)
        return self.output_norm(x)

    def logits_for(self, hidden: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits from hidden states (tied item embeddings)."""
        return hidden @ self.item_embedding.weight.T


class SASRecModel(Recommender):
    name = "sasrec"

    def __init__(self, vocab_size: int, config: dict[str, Any] | None = None):
        config = dict(config or {})
        self.vocab_size = vocab_size
        self.max_len = int(config.get("max_len", 50))
        self.batch_size = int(config.get("batch_size", 128))
        self.epochs = int(config.get("epochs", 8))
        self.early_stopping = EarlyStoppingConfig.from_config(config)
        self.learning_rate = float(config.get("learning_rate", 1e-3))
        self.weight_decay = float(config.get("weight_decay", 0.0))
        self.loss_kind = str(config.get("loss", "full_softmax"))
        self.num_negatives = int(config.get("num_negatives", 128))
        self.seed = int(config.get("seed", 13))
        self.device = resolve_device(str(config.get("device", "auto")))
        # Overlapping windows; on small datasets one sample per user is not
        # enough to fit even a small transformer.
        self.stride = int(config.get("stride", max(1, self.max_len // 2)))

        torch.manual_seed(self.seed)
        self.net = SASRecNet(
            vocab_size=vocab_size,
            hidden_dim=int(config.get("hidden_dim", 64)),
            num_blocks=int(config.get("num_blocks", 2)),
            num_heads=int(config.get("num_heads", 2)),
            dropout=float(config.get("dropout", 0.2)),
            max_len=self.max_len,
        ).to(self.device)

    # -- data -----------------------------------------------------------------

    def _windows(self, histories: list[np.ndarray]) -> np.ndarray:
        """Cut each history into overlapping fixed-length training windows."""
        samples: list[np.ndarray] = []
        span = self.max_len + 1  # inputs + the shifted labels
        for history in histories:
            valid = history[(history > 0) & (history < self.vocab_size)]
            if valid.size < 2:
                continue
            if valid.size <= span:
                padded = np.zeros(span, dtype=np.int64)
                padded[span - valid.size :] = valid
                samples.append(padded)
                continue
            start = valid.size - span
            while start >= 0:
                samples.append(valid[start : start + span].astype(np.int64))
                start -= self.stride
            if start + self.stride > 0:  # keep the oldest partial window
                padded = np.zeros(span, dtype=np.int64)
                head = valid[: start + self.stride]
                padded[span - head.size :] = head[-span:]
                if (padded > 0).sum() >= 2:
                    samples.append(padded)
        if not samples:
            return np.zeros((0, span), dtype=np.int64)
        return np.stack(samples)

    def _padded_inputs(self, histories: list[np.ndarray]) -> torch.Tensor:
        batch = np.zeros((len(histories), self.max_len), dtype=np.int64)
        for row, history in enumerate(histories):
            valid = history[(history > 0) & (history < self.vocab_size)]
            if valid.size == 0:
                continue
            tail = valid[-self.max_len :]
            batch[row, self.max_len - tail.size :] = tail
        return torch.from_numpy(batch)

    # -- training -------------------------------------------------------------

    def _loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        mask = labels > 0
        if not bool(mask.any()):
            return hidden.sum() * 0.0
        selected = hidden[mask]
        positives = labels[mask]

        if self.loss_kind == "sampled_softmax":
            negatives = torch.randint(
                1,
                self.vocab_size,
                (selected.size(0), self.num_negatives),
                device=selected.device,
            )
            candidates = torch.cat([positives.unsqueeze(1), negatives], dim=1)
            embeddings = self.net.item_embedding(candidates)
            logits = torch.einsum("bd,bcd->bc", selected, embeddings)
            # Do not let a sampled negative that equals the positive count
            # against it.
            collision = candidates[:, 1:] == positives.unsqueeze(1)
            logits[:, 1:] = logits[:, 1:].masked_fill(collision, float("-inf"))
            target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            return nn.functional.cross_entropy(logits, target)

        logits = self.net.logits_for(selected)
        logits[:, 0] = float("-inf")  # never predict padding
        return nn.functional.cross_entropy(logits, positives)

    def fit(self, train: SequenceSplit, val: SequenceSplit | None = None) -> dict[str, float]:
        samples = self._windows(train.inputs)
        if samples.shape[0] == 0:
            raise ValueError("no training windows produced; check min_events_per_user")

        generator = torch.Generator().manual_seed(self.seed)
        data = torch.from_numpy(samples)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        history: list[float] = []
        monitor = (
            EarlyStopping(self.early_stopping)
            if self.early_stopping.enabled and val is not None and len(val)
            else None
        )
        epoch_limit = self.early_stopping.max_epochs if monitor else self.epochs
        self.net.train()
        for epoch in range(epoch_limit):
            order = torch.randperm(data.size(0), generator=generator)
            epoch_loss, batches = 0.0, 0
            for start in range(0, data.size(0), self.batch_size):
                batch = data[order[start : start + self.batch_size]].to(self.device)
                inputs, labels = batch[:, :-1], batch[:, 1:]
                hidden = self.net(inputs)
                loss = self._loss(hidden, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batches += 1
            history.append(epoch_loss / max(batches, 1))
            if monitor is not None and monitor.check(epoch, self, self.net, val):
                break

        stats: dict[str, Any] = {
            "train_windows": float(samples.shape[0]),
            "first_epoch_loss": round(history[0], 5),
            "final_epoch_loss": round(history[-1], 5),
            "epochs": float(len(history)),
            "train_loss_curve": [round(value, 5) for value in history],
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
        histories = batch.tokens
        out = np.zeros((len(histories), self.vocab_size), dtype=np.float32)
        for start in range(0, len(histories), self.batch_size):
            chunk = histories[start : start + self.batch_size]
            inputs = self._padded_inputs(chunk).to(self.device)
            hidden = self.net(inputs)[:, -1, :]  # the last position predicts next
            logits = self.net.logits_for(hidden)
            out[start : start + len(chunk)] = logits.float().cpu().numpy()
        return out
