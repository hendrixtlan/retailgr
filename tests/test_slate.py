"""The two changes that came out of asking why the ranker cannot order a slate.

The finding that produced them, in one line each:

**The teacher-forced metric was never about ranking.** The heads scored AUC
0.986 on history positions, and a five-number lookup table on how many times
the same item just repeated scores 0.965 — for the purchase head the table
*wins*, 0.9935 to 0.9909. That number was measuring the synthetic generator's
view -> cart -> purchase funnel.

**The training positive is a kind of item serving never shows.** 52.5% of
positives are already in the prefix, and 100% of the `add_to_cart`,
`purchase` and `return` ones are — a customer cannot buy something they never
looked at. `exclude_seen` removes exactly those before the ranker is
consulted, so three of four heads were trained entirely on a class of item
that never reaches them. The purchase head orders the slate *below* chance
(AUC 0.465) because the feature it learned is anti-correlated with what
survives the filter.

These tests pin the mechanics of the fixes. Whether the fixes *work* is
`retailgr slate-experiment`, which is a measurement and not an assertion.
"""

from __future__ import annotations

import numpy as np
import pytest

from retailgr.config import Config
from retailgr.slate_experiment import SLATE_VARIANTS, slate_auc

# -- AUC, which replaced resolution share as the headline ---------------------


def test_auc_is_half_for_a_scorer_that_carries_nothing():
    rng = np.random.default_rng(0)
    labels = (rng.random(4000) < 0.01).astype(float)
    assert abs(slate_auc(labels, rng.random(4000)) - 0.5) < 0.05


def test_auc_is_one_for_a_perfect_ordering_and_zero_for_its_reverse():
    labels = np.array([0, 0, 1, 1], dtype=float)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert slate_auc(labels, scores) == 1.0
    assert slate_auc(labels, -scores) == 0.0


def test_auc_averages_ties_rather_than_rewarding_them():
    """A model that outputs one constant is not a perfect ranker, and a
    tie-handling bug is the usual way it gets scored as one."""
    labels = np.array([0, 1, 0, 1], dtype=float)
    assert slate_auc(labels, np.zeros(4)) == 0.5


def test_auc_is_undefined_rather_than_zero_when_a_class_is_missing():
    """Returning 0.0 for "no positives in this slate" would drag every
    average down by the number of users retrieval missed entirely."""
    assert np.isnan(slate_auc(np.zeros(10), np.arange(10, dtype=float)))
    assert np.isnan(slate_auc(np.ones(10), np.arange(10, dtype=float)))


def test_auc_does_not_depend_on_the_scale_of_the_scores():
    """The whole reason for using it here: the heads are miscalibrated in
    scale by ~3x on the slate, and ordering is the question being asked."""
    rng = np.random.default_rng(1)
    labels = (rng.random(2000) < 0.05).astype(float)
    scores = rng.random(2000)
    assert slate_auc(labels, scores) == pytest.approx(slate_auc(labels, scores * 1e-4))


def test_the_resolution_metric_really_does_depend_on_the_base_rate():
    """Justifies replacing it, rather than asserting the replacement was wise.

    The same *relative* ordering, expressed at two base rates, produces
    resolution shares two orders of magnitude apart. That is how a real
    difference in ordering ability got reported as a 7000x collapse.
    """
    from retailgr.evaluation.calibration import calibration_report

    rng = np.random.default_rng(3)
    shares = []
    for base in (0.0022, 0.2436):
        n = 400_000
        # A scorer with a fixed, modest amount of signal at either base rate.
        signal = rng.random(n)
        labels = (rng.random(n) < base * (0.5 + signal)).astype(float)
        report = calibration_report(labels, base * (0.5 + signal), bins=10)
        shares.append(report.resolution_share)
    assert shares[1] > shares[0] * 20, shares


# -- the arms are distinct, and each changes something ------------------------


def test_every_arm_differs_from_the_control_in_exactly_one_way_or_both():
    control = SLATE_VARIANTS["control"]
    assert SLATE_VARIANTS["unseen"] != control
    assert SLATE_VARIANTS["no_history"] != control
    differences = {
        name: {k for k, v in arm.items() if control.get(k) != v}
        for name, arm in SLATE_VARIANTS.items()
    }
    assert differences["control"] == set()
    assert differences["unseen"] == {"candidate_positive"}
    assert differences["no_history"] == {"history_loss_weight"}
    assert differences["both"] == {"candidate_positive", "history_loss_weight"}


def test_the_ranker_rejects_an_unknown_candidate_positive():
    """A typo must not silently fall back to the old behaviour — that is a
    run that reports an arm's name and measures the control."""
    from retailgr.models.ranker import HSTURanker

    with pytest.raises(ValueError, match="candidate_positive"):
        HSTURanker(50, {"candidate_positive": "unsen", "hidden_dim": 8, "max_events": 4})


# -- unseen cuts pick a position whose item is new ----------------------------


def _ranker(vocab_size: int = 50):
    from retailgr.models.ranker import HSTURanker

    return HSTURanker(
        vocab_size,
        {
            "hidden_dim": 8,
            "num_blocks": 1,
            "num_heads": 1,
            "max_events": 10,
            "epochs": 1,
            "batch_size": 4,
            "num_negatives": 4,
            "candidate_positive": "unseen",
        },
    )


def _split(histories):
    from retailgr.io.loaders import SequenceSplit

    split = SequenceSplit()
    for index, tokens in enumerate(histories):
        tokens = np.asarray(tokens, dtype=np.int64)
        split.user_ids.append(f"U{index}")
        split.inputs.append(tokens)
        split.actions.append(np.ones(tokens.size, dtype=np.int64))
        split.timestamps.append(np.arange(tokens.size, dtype=np.int64))
        split.returned.append(np.full(tokens.size, -1, dtype=np.int64))
        split.targets.append(np.zeros(0, dtype=np.int64))
        split.target_actions.append(np.zeros(0, dtype=np.int64))
    return split


def test_an_unseen_cut_lands_on_an_item_not_already_in_the_prefix():
    import torch

    ranker = _ranker()
    # Position 3 is the only one whose item is new; 1, 2 and 4 are repeats.
    split = _split([[5, 5, 5, 9, 5]])
    generator = torch.Generator().manual_seed(0)
    for _ in range(10):
        cuts, fallbacks = ranker._unseen_cuts(split, [0], generator)
        assert cuts == [3], cuts
        assert fallbacks == 0


def test_a_history_of_one_repeated_item_falls_back_and_is_counted():
    """The cost of the arm, made visible rather than absorbed. A user with no
    new item has no serving-shaped positive to offer."""
    import torch

    ranker = _ranker()
    split = _split([[7, 7, 7, 7]])
    cuts, fallbacks = ranker._unseen_cuts(split, [0], torch.Generator().manual_seed(0))
    assert fallbacks == 1
    assert cuts == [3]


def test_the_cut_truncates_the_history_so_the_positive_ends_up_last():
    """`fit` holds out `tokens[:, -1]` as the positive, so a cut only works
    if the row is rebuilt to end there. Checked on the padded tensors rather
    than trusted."""
    ranker = _ranker()
    split = _split([[5, 5, 9, 5, 5]])
    batch = ranker._padded_history(split, [0], cuts=[2])
    tokens = batch["tokens"][0].numpy()
    real = tokens[tokens > 0]
    assert real.tolist() == [5, 5, 9], real
    assert real[-1] == 9


def test_without_cuts_the_history_is_unchanged():
    """The default has to stay exactly what it was, or every previously
    measured number becomes uncomparable."""
    ranker = _ranker()
    split = _split([[5, 5, 9, 5, 4]])
    tokens = ranker._padded_history(split, [0])["tokens"][0].numpy()
    assert tokens[tokens > 0].tolist() == [5, 5, 9, 5, 4]


def test_actions_and_timestamps_are_truncated_with_the_tokens():
    """A cut that trims tokens and not the parallel arrays silently shifts
    every action one position out of alignment, which would look like a
    model problem for a long time."""
    from retailgr.io.loaders import SequenceSplit

    split = SequenceSplit()
    split.user_ids.append("U0")
    split.inputs.append(np.array([5, 6, 7, 8], dtype=np.int64))
    split.actions.append(np.array([1, 2, 3, 4], dtype=np.int64))
    split.timestamps.append(np.array([10, 20, 30, 40], dtype=np.int64))
    split.returned.append(np.array([-1, -1, 0, 1], dtype=np.int64))
    split.targets.append(np.zeros(0, dtype=np.int64))
    split.target_actions.append(np.zeros(0, dtype=np.int64))

    ranker = _ranker()
    batch = ranker._padded_history(split, [0], cuts=[1])
    tokens = batch["tokens"][0].numpy()
    keep = tokens > 0
    assert tokens[keep].tolist() == [5, 6]
    assert batch["actions"][0].numpy()[keep].tolist() == [1, 2]
    assert batch["timestamps"][0].numpy()[keep].tolist() == [10, 20]
    assert batch["returned"][0].numpy()[keep].tolist() == [-1, -1]


def test_training_with_unseen_positives_runs_and_reports_the_fallback_share():
    ranker = _ranker()
    split = _split([[5, 6, 7, 8, 9], [3, 3, 4, 4, 5], [2, 2, 2, 2, 2]])
    stats = ranker.fit(split)
    assert stats["candidate_positive_unseen"] == 1.0
    # One of the three users has no new item after the first.
    assert stats["cut_fallback_share"] == pytest.approx(1 / 3, abs=0.01)


def test_the_history_loss_can_be_switched_off_without_the_model_failing():
    from retailgr.models.ranker import HSTURanker

    ranker = HSTURanker(
        50,
        {
            "hidden_dim": 8, "num_blocks": 1, "num_heads": 1, "max_events": 10,
            "epochs": 1, "batch_size": 4, "num_negatives": 4,
            "history_loss_weight": 0.0,
        },
    )
    stats = ranker.fit(_split([[5, 6, 7, 8, 9], [3, 4, 5, 6, 7]]))
    assert stats["history_loss_weight_mean"] == 0.0
    assert np.isfinite(stats["final_epoch_loss"])


def test_the_history_loss_can_be_kept_for_one_head_and_dropped_for_the_rest():
    """The arm a measurement produced rather than a guess.

    A global zero wins on the slate and takes the return head to chance with
    it — AUC 0.873 to 0.511 — and the slate cannot see that, because the
    return head has no slate-shaped label.
    """
    from retailgr.models.ranker import HEADS, HSTURanker

    ranker = HSTURanker(
        50,
        {
            "hidden_dim": 8, "num_blocks": 1, "num_heads": 1, "max_events": 10,
            "epochs": 1, "batch_size": 4, "num_negatives": 4,
            "history_loss_weight": {
                "click": 0.0, "cart": 0.0, "purchase": 0.0, "return": 1.0
            },
        },
    )
    assert ranker.history_loss_weight == {
        "click": 0.0, "cart": 0.0, "purchase": 0.0, "return": 1.0
    }
    stats = ranker.fit(_split([[5, 6, 7, 8, 9], [3, 4, 5, 6, 7]]))
    assert stats["history_loss_weight_mean"] == pytest.approx(0.25)
    assert np.isfinite(stats["final_epoch_loss"])

    # A scalar still means every head, so existing configs are unchanged.
    scalar = HSTURanker(50, {"hidden_dim": 8, "max_events": 4, "history_loss_weight": 0.5})
    assert scalar.history_loss_weight == dict.fromkeys(HEADS, 0.5)


def test_an_unlisted_head_keeps_full_weight_rather_than_silently_dropping_to_zero():
    """A map that names three heads must not switch the fourth off. That is
    the failure this design is most likely to produce and the hardest to
    notice — the head simply stops learning."""
    from retailgr.models.ranker import HSTURanker

    ranker = HSTURanker(
        50, {"hidden_dim": 8, "max_events": 4, "history_loss_weight": {"click": 0.0}}
    )
    assert ranker.history_loss_weight["click"] == 0.0
    assert ranker.history_loss_weight["return"] == 1.0
    assert ranker.history_loss_weight["purchase"] == 1.0


# -- exclude_seen is a surface decision, not a global one ---------------------


def test_exclude_seen_is_read_per_surface():
    from retailgr.serving.service import ServingConfig

    config = ServingConfig(
        exclude_seen={"default": True, "cart": False, "email": False}
    )
    assert config.excludes_seen("home") is True
    assert config.excludes_seen("cart") is False
    assert config.excludes_seen("email") is False
    # An unknown surface takes the default rather than guessing.
    assert config.excludes_seen("kiosk") is True
    assert config.excludes_seen(None) is True


def test_a_bare_boolean_still_means_everywhere():
    """Back-compatibility is load-bearing: an existing deployment's config
    must not change behaviour because this became a map."""
    from retailgr.serving.service import ServingConfig

    assert ServingConfig(exclude_seen=True).excludes_seen("cart") is True
    assert ServingConfig(exclude_seen=False).excludes_seen("home") is False


def test_the_shipped_config_turns_the_filter_off_for_the_reminder_surfaces():
    """Cart and email are reminders. The whole point of them is the item the
    customer already looked at, so filtering it out is the one behaviour
    those surfaces cannot have."""
    from retailgr.serving.service import ServingConfig

    cfg = Config.load()
    config = ServingConfig(exclude_seen=cfg.get("serving.exclude_seen", True))
    assert config.excludes_seen("cart") is False
    assert config.excludes_seen("email") is False
    assert config.excludes_seen("home") is True


def test_the_service_asks_the_surface_and_not_the_global_flag():
    """Structural: the request path must read `excludes_seen(surface)`. A
    direct read of `config.exclude_seen` would be truthy for the map and
    silently exclude on every surface."""
    import ast
    import inspect

    from retailgr.serving.service import RecommendationService

    source = inspect.cleandoc(inspect.getsource(RecommendationService.recommend))
    tree = ast.parse(source)
    direct = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "exclude_seen"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
    ]
    assert not direct, (
        "recommend() reads config.exclude_seen directly; a per-surface map is "
        "always truthy, so this would exclude on every surface"
    )
