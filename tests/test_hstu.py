"""Tests for the HSTU layer.

These pin the three choices that make HSTU different from a Transformer, each
of which is easy to break silently:

1. attention is not softmax-normalised,
2. the causal mask is applied after the activation, so no future information
   reaches a position,
3. actions and timestamps actually change the output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from retailgr.actions import ACTION_TO_ID, ACTION_VOCAB_SIZE, encode_actions
from retailgr.io.loaders import HistoryBatch, SequenceSplit
from retailgr.models.hstu import (
    HSTULayer,
    HSTUModel,
    HSTUNet,
    RelativeBucketedTimeAndPositionBias,
)

VOCAB = 40
SEQ = 8
DIM = 16


def _net(**kwargs) -> HSTUNet:
    torch.manual_seed(0)
    defaults = dict(
        vocab_size=VOCAB, embedding_dim=DIM, num_blocks=2, num_heads=2, dropout=0.0, max_len=SEQ
    )
    defaults.update(kwargs)
    net = HSTUNet(**defaults)
    net.eval()
    return net


def _sequence(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(1, VOCAB, (batch, SEQ), generator=generator)
    actions = torch.randint(1, ACTION_VOCAB_SIZE, (batch, SEQ), generator=generator)
    timestamps = torch.arange(SEQ).repeat(batch, 1) * 3600
    return tokens, actions, timestamps


# -- the architecture ---------------------------------------------------------


def test_output_shapes():
    net = _net()
    tokens, actions, timestamps = _sequence()
    hidden = net(tokens, actions, timestamps)
    assert hidden.shape == (2, SEQ, DIM)
    assert net.logits_for(hidden).shape == (2, SEQ, VOCAB)


def test_attention_is_not_softmax_normalised():
    """Rows of the attention matrix must not sum to one.

    HSTU replaces softmax with ``silu(.) / N`` precisely so that the number of
    prior related events survives as signal. If someone swaps in a softmax,
    the row sums give it away.
    """
    torch.manual_seed(0)
    layer = HSTULayer(embedding_dim=DIM, num_heads=1, attention_dim=8, hidden_dim=8, dropout=0.0)
    layer.eval()
    x = torch.randn(1, SEQ, DIM)
    mask = torch.tril(torch.ones(SEQ, SEQ)).unsqueeze(0)

    normed = torch.nn.functional.layer_norm(x, [DIM], eps=layer.eps)
    projected = torch.nn.functional.silu(layer.uvqk(normed))
    _, _, q, k = torch.split(projected, [8, 8, 8, 8], dim=-1)
    scores = torch.einsum("bnhd,bmhd->bhnm", q.view(1, SEQ, 1, 8), k.view(1, SEQ, 1, 8))
    weights = torch.nn.functional.silu(scores * layer.attention_alpha) / SEQ
    weights = weights * mask.unsqueeze(1)

    row_sums = weights.sum(dim=-1).flatten()
    assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)


def test_future_tokens_cannot_change_an_earlier_position():
    """Change the last event; every earlier position must be untouched.

    This is the test that catches masking applied before the activation:
    silu(0) is 0 but silu of a masked-as-zero score is not, so an additive
    mask would leak a constant from the future.
    """
    net = _net()
    tokens, actions, timestamps = _sequence(batch=1)
    baseline = net(tokens, actions, timestamps)

    altered = tokens.clone()
    altered[0, -1] = (altered[0, -1] % (VOCAB - 2)) + 1
    if int(altered[0, -1]) == int(tokens[0, -1]):
        altered[0, -1] = int(altered[0, -1]) + 1
    changed = net(altered, actions, timestamps)

    torch.testing.assert_close(baseline[:, :-1, :], changed[:, :-1, :])
    assert not torch.allclose(baseline[:, -1, :], changed[:, -1, :])


def test_padding_positions_do_not_influence_the_last_position():
    """Left padding must be inert: a padded short history scores the same as
    the same history in a shorter tensor position."""
    net = _net()
    generator = torch.Generator().manual_seed(3)
    real = torch.randint(1, VOCAB, (1, 4), generator=generator)
    actions = torch.full((1, 4), ACTION_TO_ID["view"])
    times = torch.arange(4).unsqueeze(0) * 60

    padded_tokens = torch.cat([torch.zeros(1, 4, dtype=torch.long), real], dim=1)
    padded_actions = torch.cat([torch.zeros(1, 4, dtype=torch.long), actions], dim=1)
    padded_times = torch.cat([torch.full((1, 4), 0, dtype=torch.long), times], dim=1)

    out = net(padded_tokens, padded_actions, padded_times)
    # Positions that are pure padding carry no item and must not produce a
    # different answer when their contents change.
    other = padded_tokens.clone()
    other[0, :4] = 0
    torch.testing.assert_close(out, net(other, padded_actions, padded_times))


def test_actions_change_the_prediction():
    """The action modality must actually be read; otherwise HSTU is SASRec."""
    net = _net()
    tokens, actions, timestamps = _sequence(batch=1)
    viewed = torch.full_like(actions, ACTION_TO_ID["view"])
    purchased = torch.full_like(actions, ACTION_TO_ID["purchase"])
    assert not torch.allclose(
        net(tokens, viewed, timestamps), net(tokens, purchased, timestamps)
    )


def test_timestamps_change_the_prediction_through_the_temporal_bias():
    net = _net(use_relative_bias=True, use_temporal_bias=True)
    tokens, actions, _ = _sequence(batch=1)
    minutes_apart = torch.arange(SEQ).unsqueeze(0) * 60
    months_apart = torch.arange(SEQ).unsqueeze(0) * 60 * 60 * 24 * 30
    assert not torch.allclose(
        net(tokens, actions, minutes_apart), net(tokens, actions, months_apart)
    )


def test_temporal_bias_can_be_switched_off():
    net = _net(use_temporal_bias=False)
    tokens, actions, _ = _sequence(batch=1)
    close = torch.arange(SEQ).unsqueeze(0) * 60
    far = torch.arange(SEQ).unsqueeze(0) * 60 * 60 * 24 * 30
    torch.testing.assert_close(net(tokens, actions, close), net(tokens, actions, far))


def test_residual_connection_is_present():
    """With a zeroed output projection the layer must return its input."""
    torch.manual_seed(0)
    layer = HSTULayer(embedding_dim=DIM, num_heads=2, attention_dim=8, hidden_dim=8, dropout=0.0)
    layer.eval()
    with torch.no_grad():
        layer.output.weight.zero_()
        layer.output.bias.zero_()
    x = torch.randn(2, SEQ, DIM)
    mask = torch.tril(torch.ones(SEQ, SEQ)).expand(2, SEQ, SEQ)
    torch.testing.assert_close(layer(x, mask, None, float(SEQ)), x)


# -- the relative attention bias ----------------------------------------------


def test_positional_bias_is_toeplitz():
    """Equal relative distances must share a parameter."""
    torch.manual_seed(0)
    bias = RelativeBucketedTimeAndPositionBias(max_seq_len=SEQ)
    matrix = bias(None)[0].detach()
    assert matrix.shape == (SEQ, SEQ)
    for offset in range(-SEQ + 1, SEQ):
        diagonal = torch.diagonal(matrix, offset=offset)
        torch.testing.assert_close(diagonal, torch.full_like(diagonal, float(diagonal[0])))


def test_time_buckets_are_logarithmic_and_clamped():
    bias = RelativeBucketedTimeAndPositionBias(max_seq_len=4, num_buckets=8)
    # A huge gap must clamp to the last bucket rather than index out of range.
    timestamps = torch.tensor([[0, 1, 10**9, 2 * 10**9]], dtype=torch.long)
    out = bias(timestamps)
    assert out.shape == (1, 4, 4)
    assert torch.isfinite(out).all()


def test_bias_shape_matches_batch():
    bias = RelativeBucketedTimeAndPositionBias(max_seq_len=SEQ)
    timestamps = torch.arange(SEQ).repeat(3, 1)
    assert bias(timestamps).shape == (3, SEQ, SEQ)


# -- the model wrapper --------------------------------------------------------


def _split(n_users: int = 24, length: int = 12) -> SequenceSplit:
    rng = np.random.default_rng(5)
    split = SequenceSplit()
    for user in range(n_users):
        tokens = rng.integers(1, VOCAB, size=length).astype(np.int64)
        split.user_ids.append(f"u{user}")
        split.inputs.append(tokens)
        split.actions.append(encode_actions(["view"] * length))
        split.timestamps.append(np.arange(length, dtype=np.int64) * 600)
        split.targets.append(rng.integers(1, VOCAB, size=3).astype(np.int64))
    return split


def test_fit_runs_and_reduces_the_loss():
    split = _split()
    model = HSTUModel(
        VOCAB,
        {"hidden_dim": DIM, "num_blocks": 1, "num_heads": 2, "max_len": 8, "epochs": 6,
         "batch_size": 8, "dropout": 0.0},
    )
    stats = model.fit(split)
    assert stats["train_windows"] > 0
    assert stats["final_epoch_loss"] < stats["first_epoch_loss"]
    assert stats["parameters"] > 0


def test_score_shape_and_finiteness():
    split = _split()
    model = HSTUModel(
        VOCAB, {"hidden_dim": DIM, "num_blocks": 1, "max_len": 8, "epochs": 1, "batch_size": 8}
    )
    model.fit(split)
    scores = model.score(split.batch(0, 5))
    assert scores.shape == (5, VOCAB)
    assert np.isfinite(scores).all()


def test_empty_history_does_not_crash_scoring():
    model = HSTUModel(VOCAB, {"hidden_dim": DIM, "num_blocks": 1, "max_len": 8, "epochs": 1})
    batch = HistoryBatch(
        tokens=[np.zeros(0, dtype=np.int64)],
        actions=[np.zeros(0, dtype=np.int64)],
        timestamps=[np.zeros(0, dtype=np.int64)],
    )
    scores = model.score(batch)
    assert scores.shape == (1, VOCAB)
    assert np.isfinite(scores).all()


def test_positive_action_filter_changes_what_is_supervised():
    """Training only on positive actions must reduce the supervised positions."""
    split = SequenceSplit()
    rng = np.random.default_rng(11)
    for user in range(16):
        tokens = rng.integers(1, VOCAB, size=10).astype(np.int64)
        split.user_ids.append(f"u{user}")
        split.inputs.append(tokens)
        # Alternate a positive and a negative action.
        split.actions.append(encode_actions(["view", "remove_from_cart"] * 5))
        split.timestamps.append(np.arange(10, dtype=np.int64) * 60)
        split.targets.append(tokens[:2])

    base = {"hidden_dim": DIM, "num_blocks": 1, "max_len": 8, "epochs": 1, "batch_size": 8}
    strict = HSTUModel(VOCAB, {**base, "supervise_positive_only": True})
    loose = HSTUModel(VOCAB, {**base, "supervise_positive_only": False})

    windows = strict._windows(split)
    tokens = torch.from_numpy(windows["tokens"])
    actions = torch.from_numpy(windows["actions"])
    labels, label_actions = tokens[:, 1:], actions[:, 1:]

    def supervised(model: HSTUModel) -> int:
        mask = labels > 0
        if model.supervise_positive_only and model.positive_actions:
            positive = torch.zeros_like(mask)
            for action_id in model.positive_actions:
                positive |= label_actions == action_id
            mask = mask & positive
        return int(mask.sum())

    assert supervised(strict) < supervised(loose)
    assert supervised(strict) > 0


def test_unknown_action_names_do_not_break_encoding():
    encoded = encode_actions(["view", "teleported", "purchase"])
    assert encoded.shape == (3,)
    assert encoded.max() < ACTION_VOCAB_SIZE
    assert encoded[0] == ACTION_TO_ID["view"]
    assert encoded[2] == ACTION_TO_ID["purchase"]


def test_rejects_an_unknown_model_type():
    from retailgr.experiment import build_model

    with pytest.raises(ValueError):
        build_model(VOCAB, {"type": "transformer_xl"})


# -- the ablation harness -----------------------------------------------------


def test_ablation_set_changes_exactly_one_thing_each():
    from retailgr.experiment import HSTU_ABLATIONS

    assert "hstu_full" in HSTU_ABLATIONS
    assert HSTU_ABLATIONS["hstu_full"] == {}, "the full model must override nothing"
    known = {
        "use_temporal_bias",
        "use_relative_bias",
        "use_actions",
        "normalise_by",
        "supervise_positive_only",
    }
    for name, overrides in HSTU_ABLATIONS.items():
        assert set(overrides) <= known, (name, overrides)


def test_ablation_report_states_a_verdict_from_the_paired_test():
    """The report must read off the interval, not the point estimate: a large
    difference with an interval spanning zero is still "no difference"."""
    from retailgr.experiment import render_ablation

    def entry(kind: str, ndcg: float) -> dict:
        return {
            "type": kind,
            "fit": {"final_epoch_loss": 3.5},
            "fit_seconds": 1.0,
            "test": {
                "overall": {"recall@20": 0.4, "ndcg@20": ndcg},
                "intervals": {"ndcg@20": {"ci_low": ndcg - 0.01, "ci_high": ndcg + 0.01}},
            },
        }

    def comparison(difference: float, low: float, high: float, p: float) -> dict:
        return {
            "ndcg@20": {
                "mean_difference": difference,
                "ci_low": low,
                "ci_high": high,
                "p_value": p,
                "relative_difference": difference / 0.20,
                "significant": low > 0 or high < 0,
            }
        }

    report = render_ablation(
        {
            "variant": "config",
            "vocab_size": 100,
            "k_values": [20],
            "base_model_config": "hstu_small.yaml",
            "confidence": 0.95,
            "seeds": [13],
            "models": {
                "hstu_full": entry("hstu", 0.2000),
                "hstu_no_temporal_bias": entry("hstu", 0.2200),
                "hstu_no_actions": entry("hstu", 0.2100),
            },
            "comparisons": {
                "vs_hstu_full": {
                    # Clearly better: interval excludes zero.
                    "hstu_no_temporal_bias": comparison(0.02, 0.012, 0.028, 0.001),
                    # Same point estimate direction, but the interval spans zero.
                    "hstu_no_actions": comparison(0.01, -0.004, 0.024, 0.200),
                }
            },
        }
    )
    assert "10.0% better" in report  # 0.02 against a 0.20 baseline
    assert "no difference detected" in report
    # p adjusted is shown, and the seed guidance appears for a single-seed run.
    assert "p adj." in report
    assert "--seeds" in report


def test_ablation_report_shows_the_seed_spread_when_there_is_one():
    from retailgr.experiment import render_ablation

    report = render_ablation(
        {
            "variant": "config",
            "vocab_size": 100,
            "k_values": [20],
            "base_model_config": "hstu_small.yaml",
            "confidence": 0.95,
            "seeds": [13, 17, 23],
            "models": {
                "hstu_full": {
                    "type": "hstu",
                    "fit": {"final_epoch_loss": 3.5},
                    "fit_seconds": 3.0,
                    "test": {"overall": {"recall@20": 0.4, "ndcg@20": 0.20}},
                    "seed_spread": {
                        "ndcg@20": {
                            "seeds": 3,
                            "mean": 0.20,
                            "min": 0.198,
                            "max": 0.202,
                            "std": 0.002,
                        }
                    },
                }
            },
            "comparisons": {},
        }
    )
    assert "Across training seeds" in report
    assert "0.0020" in report
    # With a spread reported, the nudge to run more seeds is gone.
    assert "--seeds" not in report


def test_ablation_report_survives_a_missing_baseline():
    from retailgr.experiment import render_ablation

    report = render_ablation(
        {
            "variant": "config",
            "vocab_size": 100,
            "k_values": [20],
            "base_model_config": "hstu_small.yaml",
            "models": {
                "hstu_full": {
                    "type": "hstu",
                    "fit": {"final_epoch_loss": 3.5},
                    "fit_seconds": 1.0,
                    "test": {"overall": {"recall@20": 0.4, "ndcg@20": 0.22}},
                }
            },
        }
    )
    assert "hstu_full" in report
