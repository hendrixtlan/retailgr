"""Tests for calibration.

A miscalibrated head is invisible to every number the ranker report showed
before this module existed: AUC is invariant to exactly the transform that
causes it, and a rare-event Brier score is dominated by the rarity. So these
tests are mostly about detection — proving the measurement fires on a
distortion it is supposed to catch and stays quiet on one it is not — and
about the two invariants a calibrator must never break: it may change a
head's scale, and it may not change a head's order.

``test_calibration_restores_the_configured_business_weights`` is the one that
justifies the module. It reconstructs, with the probability ranges actually
measured on the synthetic set, the bug this project already shipped once: an
expected-value blend whose purchase term was being scaled into irrelevance.
"""

from __future__ import annotations

import numpy as np
import pytest

from retailgr.evaluation.calibration import (
    HeadCalibrators,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    calibration_report,
    calibrator_from_dict,
    fit_calibrator,
    logit,
    reliability_bins,
    sigmoid,
)
from retailgr.evaluation.ranking import roc_auc


def _rare_event(n: int = 40000, ceiling: float = 0.10, seed: int = 0):
    """Labels drawn from a known true probability, so calibration has a truth."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.0, ceiling, n)
    labels = (rng.uniform(size=n) < truth).astype(np.float64)
    return truth, labels


# -- measurement --------------------------------------------------------------


def test_a_perfectly_calibrated_forecast_measures_as_calibrated():
    truth, labels = _rare_event(ceiling=1.0)
    report = calibration_report(labels, truth)
    assert report.verdict() == "calibrated"
    assert report.bias_ratio == pytest.approx(1.0, abs=0.02)
    assert report.ece < 0.01
    assert report.brier_skill > 0


def test_systematic_overconfidence_is_detected_with_the_right_factor():
    """The failure mode that matters: right shape, wrong scale."""
    truth, labels = _rare_event()
    report = calibration_report(labels, truth / 20.0)
    assert "under-confident" in report.verdict()
    # The head believes the event is ~20x rarer than it is.
    assert report.bias_ratio == pytest.approx(0.05, abs=0.01)
    # And the error is comparable to the base rate itself.
    assert report.ece_relative > 0.5


def test_auc_cannot_see_what_calibration_sees():
    """The reason this module exists, stated as a test.

    A monotone rescaling leaves AUC bit-for-bit identical while moving the
    probabilities by a factor of twenty. Anything that judges the heads by AUC
    alone is blind to it.
    """
    truth, labels = _rare_event()
    distorted = truth / 20.0
    assert roc_auc(labels, distorted) == roc_auc(labels, truth)
    assert calibration_report(labels, distorted).ece > 10 * calibration_report(
        labels, truth
    ).ece


def test_brier_skill_exposes_a_head_that_loses_to_a_constant():
    """A raw Brier of 0.003 on a rare event looks excellent and can be worse
    than always predicting the base rate."""
    _, labels = _rare_event(ceiling=0.01)  # base rate ~0.005
    rng = np.random.default_rng(1)
    noise = rng.uniform(0.0, 0.05, labels.size)  # no signal, wrong scale

    report = calibration_report(labels, noise)
    assert report.brier < 0.01  # looks tiny
    assert report.brier_skill < 0  # and is worse than a constant


def test_murphy_decomposition_is_exact_for_the_binned_forecast():
    truth, labels = _rare_event()
    report = calibration_report(labels, truth / 3.0)
    reconstructed = report.reliability - report.resolution + report.uncertainty
    assert report.binned_brier == pytest.approx(reconstructed, abs=1e-12)
    # Reliability is the miscalibration a calibrator can fix; resolution is
    # the discrimination it cannot create.
    assert report.reliability > 0
    assert report.resolution > 0


def test_overdispersion_is_the_part_the_three_terms_cannot_see():
    """Murphy's terms describe a forecast grouped into bins. A head that
    shouts a range of numbers inside a bin whose outcomes are all alike is
    invisible to all three, and that is a real failure mode — this ranker's
    click head spread its predictions over three orders of magnitude on
    outcomes that varied by a factor of two."""
    rng = np.random.default_rng(5)
    labels = (rng.uniform(size=20000) < 0.01).astype(float)
    # Wildly varying predictions, all of them about the same event.
    shouty = rng.uniform(0.0001, 0.5, labels.size)
    calm = np.full(labels.size, 0.01)

    loud = calibration_report(labels, shouty)
    quiet = calibration_report(labels, calm)
    assert loud.overdispersion > 100 * abs(quiet.overdispersion)
    assert loud.brier > loud.binned_brier


def test_resolution_is_what_calibration_cannot_buy():
    """The number that decides whether calibrating is worth doing at all."""
    rng = np.random.default_rng(6)
    labels = (rng.uniform(size=30000) < 0.01).astype(float)
    # Predictions with the right shape but no relationship to the outcome.
    noise = rng.uniform(0.0001, 0.05, labels.size)

    before = calibration_report(labels, noise)
    calibrated = PlattCalibrator().fit(labels, noise).apply(noise)
    after = calibration_report(labels, calibrated)

    # Calibration moves the scale and leaves the knowledge exactly where it was.
    assert after.ece < before.ece
    assert after.resolution_share == pytest.approx(before.resolution_share, abs=1e-6)
    assert after.resolution_share < 0.01
    assert "uninformative" in after.verdict()


def test_a_head_with_real_resolution_is_not_called_uninformative():
    truth, labels = _rare_event(ceiling=1.0)
    report = calibration_report(labels, truth)
    assert report.resolution_share > 0.3
    assert report.verdict() == "calibrated"


def test_equal_width_bins_lose_the_structure_quantile_bins_keep():
    """Why ``strategy='quantile'`` is the default.

    With every prediction under 0.005, equal-width bins over [0, 1] put the
    whole distribution in one bucket and report a single average gap. That is
    not a reliability curve, it is a mean.
    """
    truth, labels = _rare_event()
    predictions = truth / 20.0
    uniform = reliability_bins(labels, predictions, bins=10, strategy="uniform")
    quantile = reliability_bins(labels, predictions, bins=10, strategy="quantile")
    assert len(uniform) == 1
    assert len(quantile) == 10
    # The quantile bins show the error growing across the range; one bin cannot.
    gaps = [abs(b.gap) for b in quantile]
    assert max(gaps) > 3 * min(gaps)


def test_bins_never_lose_or_duplicate_an_observation():
    truth, labels = _rare_event(n=5000)
    for strategy in ("quantile", "uniform"):
        table = reliability_bins(labels, truth, bins=10, strategy=strategy)
        assert sum(b.count for b in table) == labels.size


def test_binning_survives_a_head_that_predicts_one_value():
    """Quantile edges collapse when every prediction is identical."""
    labels = np.array([1.0, 0.0, 0.0, 1.0] * 50)
    report = calibration_report(labels, np.full(labels.size, 0.3))
    assert report is not None
    assert len(report.bins) == 1
    assert report.resolution == pytest.approx(0.0)  # a constant cannot resolve
    assert report.ece == pytest.approx(0.2, abs=1e-9)  # promised 0.3, got 0.5


def test_measurement_declines_rather_than_guessing():
    assert calibration_report([], []) is None
    # No positives at all: a ratio to a zero base rate is not a number.
    report = calibration_report(np.zeros(200), np.full(200, 0.01))
    assert not np.isfinite(report.bias_ratio)
    assert "undefined" in report.verdict() or "too few" in report.verdict()


def test_too_few_outcomes_is_reported_as_such_not_as_a_verdict():
    report = calibration_report(np.r_[np.ones(2), np.zeros(20)], np.linspace(0, 1, 22))
    assert report.verdict() == "too few outcomes to judge"


def test_brier_score_matches_the_definition():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)
    assert brier_score([1, 1, 0, 0], [0.5] * 4) == pytest.approx(0.25)
    assert np.isnan(brier_score([], []))


def test_labels_and_probabilities_must_align():
    with pytest.raises(ValueError, match="align"):
        calibration_report([1, 0, 1], [0.5, 0.5])


# -- Platt --------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,distort",
    [
        ("twenty times too small", lambda t: t / 20.0),
        ("eight times too large", lambda t: np.clip(t * 8, 0, 1)),
        ("logit squashed and shifted", lambda t: sigmoid(logit(t) * 0.4 - 1.0)),
    ],
)
def test_platt_repairs_a_monotone_distortion(name, distort):
    truth, labels = _rare_event()
    predictions = distort(truth)
    fit, held_out = slice(0, 20000), slice(20000, None)

    calibrator = PlattCalibrator().fit(labels[fit], predictions[fit])
    corrected = calibrator.apply(predictions[held_out])

    before = calibration_report(labels[held_out], predictions[held_out])
    after = calibration_report(labels[held_out], corrected)
    assert after.ece < before.ece / 5, name
    assert after.bias_ratio == pytest.approx(1.0, abs=0.1), name


def test_platt_does_not_diverge_on_a_near_singular_hessian():
    """Regression test for a real bug.

    When every prediction sits near zero the Newton weights ``p(1-p)`` are all
    tiny, the Hessian is near-singular, and an undamped step overshoots
    enormously — the first version of this fitted ``a = 6.6e12``, which
    saturates the sigmoid to exact 0 and 1 in float64 and destroys the very
    ordering the calibrator exists to preserve. The line search is what stops
    that, and this test is what notices if it is ever removed.
    """
    truth, labels = _rare_event()
    calibrator = PlattCalibrator().fit(labels, truth / 20.0)
    assert abs(calibrator.a) < 10
    assert np.isfinite(calibrator.b)
    corrected = calibrator.apply(truth / 20.0)
    assert np.unique(corrected).size > 1000  # not collapsed to 0/1


def test_platt_is_nearly_a_no_op_on_an_already_calibrated_head():
    truth, labels = _rare_event(ceiling=1.0)
    calibrator = PlattCalibrator().fit(labels, truth)
    assert calibrator.a == pytest.approx(1.0, abs=0.1)
    assert calibrator.b == pytest.approx(0.0, abs=0.1)


def test_platt_smoothing_keeps_the_fit_finite_on_separable_data():
    """Unsmoothed 0/1 targets on separable classes drive the slope to
    infinity chasing a perfect fit."""
    labels = np.r_[np.zeros(500), np.ones(50)]
    predictions = np.r_[np.linspace(0.001, 0.01, 500), np.linspace(0.5, 0.9, 50)]
    calibrator = PlattCalibrator().fit(labels, predictions)
    assert np.isfinite(calibrator.a) and abs(calibrator.a) < 1e3


# -- isotonic -----------------------------------------------------------------


def test_isotonic_pools_adjacent_violators():
    """Hand-checkable: the middle pair violates monotonicity and must pool."""
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    fitted = IsotonicCalibrator().fit(labels, predictions)
    # 0, 1, 0, 1 -> 0, then (1+0)/2 twice, then 1.
    assert list(np.round(fitted.y, 6)) == [0.0, 0.5, 0.5, 1.0]
    assert np.all(np.diff(fitted.y) >= 0)


def test_isotonic_pools_ties_before_fitting():
    """Two observations at the same predicted value cannot be separated by a
    monotone function; leaving them apart would invent a step inside a tie."""
    fitted = IsotonicCalibrator().fit([1.0, 0.0], [0.5, 0.5])
    assert fitted.x.size == 1
    assert fitted.y[0] == pytest.approx(0.5)


def test_isotonic_beats_platt_on_a_distortion_platt_cannot_express():
    truth, labels = _rare_event()
    # A power distortion is not a linear map of the logit, so the
    # two-parameter family cannot follow it.
    predictions = np.clip(truth**1.7 * 3, 1e-7, 1.0)
    fit, held_out = slice(0, 20000), slice(20000, None)

    platt = PlattCalibrator().fit(labels[fit], predictions[fit]).apply(predictions[held_out])
    iso = IsotonicCalibrator().fit(labels[fit], predictions[fit]).apply(predictions[held_out])
    assert (
        calibration_report(labels[held_out], iso).ece
        < calibration_report(labels[held_out], platt).ece
    )


def test_isotonic_extrapolates_flat_rather_than_inventing():
    fitted = IsotonicCalibrator().fit([0.0, 0.0, 1.0, 1.0], [0.2, 0.3, 0.7, 0.8])
    assert fitted.apply([0.0])[0] == pytest.approx(fitted.y[0])
    assert fitted.apply([1.0])[0] == pytest.approx(fitted.y[-1])


# -- the invariant: scale may change, order may not ---------------------------


def test_platt_preserves_a_head_s_ranking_bit_for_bit():
    """Strictly monotone, so every pairwise comparison survives untouched."""
    truth, labels = _rare_event()
    predictions = truth / 20.0
    calibrator = fit_calibrator(labels, predictions, kind="platt", min_positives=10)
    assert calibrator.kind == "platt"
    assert roc_auc(labels, calibrator.apply(predictions)) == pytest.approx(
        roc_auc(labels, predictions), abs=1e-9
    )


def test_isotonic_is_only_weakly_monotone_and_that_is_the_real_invariant():
    """Isotonic pools adjacent violators into flat segments, which turns
    distinct scores into ties — so it *can* move AUC, and expecting it not to
    would be expecting the wrong thing.

    What it can never do is invert a pair: if one candidate scored below
    another, it still does or they tie. That is the guarantee the blend needs
    and the one the fit is checked against.
    """
    truth, labels = _rare_event()
    predictions = truth / 20.0
    calibrator = fit_calibrator(labels, predictions, kind="isotonic", min_positives=10)
    assert calibrator.kind == "isotonic"

    distinct = np.unique(predictions)
    mapped = calibrator.apply(distinct)
    assert np.all(np.diff(mapped) >= 0)  # never inverts
    assert np.unique(mapped).size < distinct.size  # but does create ties


def test_fit_declines_when_there_are_too_few_outcomes():
    """Three positives can be fitted. The result is a correction derived from
    three coin flips and applied to every request."""
    rng = np.random.default_rng(2)
    labels = np.r_[np.ones(3), np.zeros(900)]
    assert fit_calibrator(labels, rng.uniform(size=903)).kind == "identity"


def test_fit_declines_a_calibrator_that_would_reverse_the_ranking():
    """A head whose scores run backwards can be made *perfectly calibrated*
    by a monotone-decreasing map. Accepting that would have calibration
    silently repair a modelling bug and report the head as healthy; refusing
    it leaves `identity` in the report, where someone can see it.
    """
    truth, labels = _rare_event()
    inverted = 0.01 - truth / 20.0  # ranks exactly backwards

    raw = PlattCalibrator().fit(labels, inverted)
    assert raw.a < 0  # the fit wants to flip it back
    assert calibration_report(labels, raw.apply(inverted)).ece < calibration_report(
        labels, inverted
    ).ece  # and calibration genuinely improves by flipping
    assert fit_calibrator(labels, inverted).kind == "identity"  # refused anyway


def test_the_slope_of_a_signal_free_head_is_just_noise():
    """Which is why the sign check cannot be left to chance: on a head with
    no signal the likelihood is nearly flat in the slope, so the fitted
    direction is whatever the sample happened to suggest."""
    _, labels = _rare_event()
    slopes = [
        PlattCalibrator()
        .fit(labels, np.random.default_rng(seed).uniform(0.0, 0.01, labels.size))
        .a
        for seed in range(6)
    ]
    assert min(slopes) < 0 < max(slopes)


def test_fit_declines_when_the_head_is_already_calibrated():
    """A two-parameter family will always shave something off the in-sample
    ECE. Installing a correction for that is a moving part in the request
    path in exchange for nothing."""
    truth, labels = _rare_event(ceiling=1.0)
    assert fit_calibrator(labels, truth, kind="platt").kind == "identity"
    # The fit itself is fine — it is just close enough to the identity that it
    # is not worth shipping.
    assert PlattCalibrator().fit(labels, truth).a == pytest.approx(1.0, abs=0.1)


def test_the_decision_to_install_is_made_out_of_sample():
    """Regression test for a rule that looked sound and was not.

    ECE on a *perfectly* calibrated head is not zero — it is however much
    noise a finite sample puts in the bins — and a two-parameter family can
    absorb a good share of that. So an in-sample "did ECE improve?" threshold
    reliably says yes on a head that needs no correction at all. It has to be
    asked on data the fit has not seen.
    """
    truth, labels = _rare_event(ceiling=1.0)
    inner, outer = slice(0, 28000), slice(28000, None)
    probe = PlattCalibrator().fit(labels[inner], truth[inner])

    in_sample_gain = 1 - (
        calibration_report(labels[inner], probe.apply(truth[inner])).ece
        / calibration_report(labels[inner], truth[inner]).ece
    )
    held_out_gain = 1 - (
        calibration_report(labels[outer], probe.apply(truth[outer])).ece
        / calibration_report(labels[outer], truth[outer]).ece
    )
    # In-sample the fit looks worth installing; held out it buys nothing.
    assert in_sample_gain > 0.1
    assert held_out_gain < 0.1
    assert fit_calibrator(labels, truth).kind == "identity"


def test_isotonic_falls_back_to_platt_when_positives_are_scarce():
    """Isotonic has as many parameters as it likes; with 40 positives it
    fits the validation noise."""
    rng = np.random.default_rng(4)
    labels = np.r_[np.ones(40), np.zeros(4000)]
    predictions = np.r_[rng.uniform(0.02, 0.2, 40), rng.uniform(0.0, 0.05, 4000)]
    assert fit_calibrator(labels, predictions, kind="isotonic").kind == "platt"


# -- what it is all for: the blend --------------------------------------------


def test_calibration_restores_the_configured_business_weights():
    """The bug this project already shipped once, and the fix, in one test.

    The heads do not share a scale: on the synthetic set P(purchase) topped
    out near 0.007 while P(cart) reached 0.11. ``ScoreBlend`` is an expected
    value, so those numbers are multiplied by business figures — and a
    purchase head that is 20x too small has its configured weight divided by
    20 before anything else happens. The operator asked for a blend dominated
    by purchases and got one dominated by cart-adds.
    """
    from retailgr.models.ranker import ScoreBlend

    truth, labels = _rare_event(ceiling=0.02, seed=11)  # base rate ~0.01
    squashed = truth / 20.0

    blend = ScoreBlend(click=0.005, cart=0.05, purchase=1.0, return_penalty=1.0)
    cart = np.full(truth.size, 0.09)
    click = np.full(truth.size, 0.10)
    zero = np.zeros(truth.size)

    def purchase_share(purchase: np.ndarray) -> float:
        total = blend.apply(
            {"purchase": purchase, "cart": cart, "click": click, "return": zero}
        )
        return float(np.mean(blend.purchase * purchase / total))

    # Uncalibrated: the purchase head, which carries 95% of the configured
    # value, contributes a few percent of the score.
    assert purchase_share(squashed) < 0.10

    calibrator = PlattCalibrator().fit(labels, squashed)
    # Calibrated: it contributes roughly what the weights asked for.
    assert purchase_share(calibrator.apply(squashed)) > 0.60


def test_head_calibrators_apply_per_head_and_leave_unknown_heads_alone():
    calibrators = HeadCalibrators(
        calibrators={"purchase": PlattCalibrator(a=1.0, b=3.0)}
    )
    out = calibrators.apply(
        {"purchase": np.array([0.001]), "cart": np.array([0.09])}
    )
    assert out["purchase"][0] > 0.001  # shifted up
    assert out["cart"][0] == pytest.approx(0.09)  # untouched
    assert not calibrators.is_identity


def test_head_calibrators_default_to_doing_nothing():
    assert HeadCalibrators.from_dict(None).is_identity
    assert HeadCalibrators.from_dict({}).apply({"cart": np.array([0.4])})["cart"][
        0
    ] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "calibrator",
    [
        PlattCalibrator(a=1.3, b=-0.7),
        IsotonicCalibrator(x=[0.1, 0.5, 0.9], y=[0.05, 0.4, 0.95]),
        IdentityCalibrator(),
    ],
)
def test_calibrators_survive_a_round_trip_through_the_manifest(calibrator):
    """A bundle is JSON. A calibrator that cannot make that trip intact would
    silently become the identity at serving time."""
    import json

    probe = np.linspace(0.01, 0.99, 25)
    restored = calibrator_from_dict(json.loads(json.dumps(calibrator.as_dict())))
    assert restored.kind == calibrator.kind
    assert np.allclose(restored.apply(probe), calibrator.apply(probe))


def test_head_calibrators_round_trip_with_their_provenance():
    import json

    original = HeadCalibrators(
        calibrators={"purchase": PlattCalibrator(a=0.9, b=2.1)},
        fitted_on={"purchase": {"source": "slate", "fit_positives": 312}},
    )
    restored = HeadCalibrators.from_dict(json.loads(json.dumps(original.as_dict())))
    assert restored.fitted_on["purchase"]["fit_positives"] == 312
    assert np.allclose(
        restored.apply({"purchase": np.array([0.002])})["purchase"],
        original.apply({"purchase": np.array([0.002])})["purchase"],
    )


# -- the distribution the numbers are measured on -----------------------------


class _FixedRetrieval:
    """A retrieval model whose ordering is known, so the slate is predictable."""

    def __init__(self, vocab_size: int, preferred: list[int]):
        self.vocab_size = vocab_size
        self.preferred = preferred

    def score(self, batch):
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for rank, token in enumerate(self.preferred):
            scores[token] = 100.0 - rank
        return [scores for _ in range(len(batch))]


def _tiny_ranker(vocab_size: int):
    from retailgr.models.ranker import HSTURanker

    return HSTURanker(
        vocab_size,
        {"hidden_dim": 16, "num_blocks": 1, "num_heads": 2, "max_events": 8, "dropout": 0.0},
    )


def _one_user_split(
    history: list[int],
    vocab_size: int,
    targets: list[int] | None = None,
    target_actions: list[str] | None = None,
):
    from retailgr.actions import encode_actions
    from retailgr.io.loaders import SequenceSplit

    targets = [history[-1]] if targets is None else targets
    target_actions = ["purchase"] * len(targets) if target_actions is None else target_actions

    split = SequenceSplit()
    split.user_ids.append("u0")
    split.inputs.append(np.array(history, dtype=np.int64))
    split.actions.append(encode_actions(["view"] * len(history)))
    split.timestamps.append(np.arange(len(history), dtype=np.int64) * 3600)
    split.returned.append(np.zeros(len(history), dtype=np.int64))
    split.targets.append(np.array(targets, dtype=np.int64))
    split.target_actions.append(encode_actions(target_actions))
    return split


def test_slate_collection_does_not_force_the_positive_into_the_candidate_set():
    """If retrieval misses the target, that user contributes only negatives —
    which is exactly the event that makes the true base rate lower than a
    forced-positive setup would report. Forcing it in would pin the base rate
    to 1/k regardless of how good retrieval is.
    """
    from retailgr.evaluation.ranking import collect_slate_probabilities

    vocab = 40
    split = _one_user_split([3, 4, 5], vocab, targets=[7])
    ranker = _tiny_ranker(vocab)

    hit = collect_slate_probabilities(
        ranker, _FixedRetrieval(vocab, [7, 10, 11, 12, 13]), split, vocab, candidate_k=5
    )
    miss = collect_slate_probabilities(
        ranker, _FixedRetrieval(vocab, [20, 21, 22, 23, 24]), split, vocab, candidate_k=5
    )
    assert hit["purchase"]["labels"].sum() == 1
    assert miss["purchase"]["labels"].sum() == 0
    assert miss["purchase"]["labels"].size == 5  # still scored, all negative


def test_slate_collection_excludes_the_history_the_way_serving_does():
    """And reports how much of the future that exclusion puts out of reach.

    This is not bookkeeping. On the synthetic set it turned out that *every*
    click, cart and purchase lands on an item the customer had already seen,
    so a slate built this way could not contain a single engagement positive
    — the base rate came out as exactly zero for every head. A measurement
    that cannot see its own blind spot reports that as "well calibrated".
    """
    from retailgr.evaluation.ranking import collect_slate_probabilities

    vocab = 40
    # Two targets: token 4 is already in the history, token 9 is not.
    split = _one_user_split([3, 4, 5], vocab, targets=[4, 9])
    collected = collect_slate_probabilities(
        _tiny_ranker(vocab),
        _FixedRetrieval(vocab, [3, 4, 5, 9, 10, 11]),
        split,
        vocab,
        candidate_k=3,
    )
    # 3, 4 and 5 are in the history, so the slate starts at 9.
    assert collected["purchase"]["labels"].size == 3
    assert collected["purchase"]["labels"].sum() == 1  # only the reachable one
    coverage = collected["_coverage"]
    assert coverage["target_items"] == 2
    assert coverage["reachable"] == 1
    assert coverage["excluded_as_seen"] == 1
    assert coverage["excluded_share"] == pytest.approx(0.5)


def test_slate_labels_differ_per_head_according_to_the_target_action():
    """A slate is one candidate set with four different label vectors on it.
    Without the target's action every head would share the label "engaged",
    which is not what the expected-value blend multiplies by money."""
    from retailgr.evaluation.ranking import collect_slate_probabilities

    vocab = 40
    split = _one_user_split(
        [3, 4, 5], vocab, targets=[7, 8], target_actions=["click", "purchase"]
    )
    collected = collect_slate_probabilities(
        _tiny_ranker(vocab), _FixedRetrieval(vocab, [7, 8, 10]), split, vocab, candidate_k=3
    )
    # A click counts for the click head only; a purchase counts for all three.
    assert collected["click"]["labels"].sum() == 2
    assert collected["cart"]["labels"].sum() == 1
    assert collected["purchase"]["labels"].sum() == 1


def test_the_return_head_has_no_slate_labels_at_all():
    """A return outcome exists only for a matured purchase, which the target
    arrays do not carry. Rather than labelling every unpurchased candidate
    "kept" — which would manufacture the base rate the measurement is meant
    to find — the slate yields nothing and the fit falls back to history
    positions, which the report names as a different distribution."""
    from retailgr.evaluation.ranking import collect_slate_probabilities

    vocab = 40
    split = _one_user_split([3, 4, 5], vocab, targets=[7])
    collected = collect_slate_probabilities(
        _tiny_ranker(vocab), _FixedRetrieval(vocab, [7, 10, 11]), split, vocab, candidate_k=3
    )
    assert collected["purchase"]["labels"].size == 3
    assert collected["return"]["labels"].size == 0


# -- the gate -----------------------------------------------------------------


def _calibration_with(bias: dict[str, float]) -> dict:
    return {
        "heads": {
            head: {"slate": {"bias_ratio": value, "base_rate": 0.01}}
            for head, value in bias.items()
        }
    }


def test_gate_blocks_a_well_ordering_ranker_whose_probabilities_are_wrong():
    """Ordering and scale are independent failures, and the blend needs both."""
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate(
        {"signals": {"blended": {"mrr": 0.70}}},
        {"mrr": 0.59},
        calibration=_calibration_with(
            {"click": 1.0, "cart": 1.1, "purchase": 0.05, "return": 1.0}
        ),
    )
    assert verdict["ordering_passed"] is True
    assert verdict["calibration_passed"] is False
    assert verdict["passed"] is False
    assert verdict["worst_calibrated_head"] == "purchase"
    assert verdict["worst_bias_ratio"] == pytest.approx(20.0)


def test_gate_passes_when_both_criteria_hold():
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate(
        {"signals": {"blended": {"mrr": 0.70}}},
        {"mrr": 0.59},
        calibration=_calibration_with(
            {"click": 1.0, "cart": 1.1, "purchase": 0.9, "return": 1.2}
        ),
    )
    assert verdict["passed"] is True


def test_gate_fails_closed_on_an_unmeasured_head():
    """An unmeasurable criterion blocks, the same rule the MRR check uses."""
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate(
        {"signals": {"blended": {"mrr": 0.70}}},
        {"mrr": 0.59},
        calibration=_calibration_with({"click": 1.0, "cart": 1.0, "purchase": 1.0}),
    )
    assert verdict["passed"] is False
    assert verdict["unmeasured_heads"] == ["return"]
    assert "unknown scale" in verdict["reason"]


def test_gate_judges_the_calibrated_numbers_when_there_are_any():
    """Once a correction is installed it is the corrected head that ships, so
    it is the corrected head the gate must judge."""
    from retailgr.evaluation.ranking import ranker_gate

    calibration = {
        "heads": {
            head: {
                "slate": {"bias_ratio": 0.05},
                "calibrated": {"bias_ratio": 1.02},
            }
            for head in ("click", "cart", "purchase", "return")
        }
    }
    verdict = ranker_gate(
        {"signals": {"blended": {"mrr": 0.70}}}, {"mrr": 0.59}, calibration=calibration
    )
    assert verdict["passed"] is True


def test_gate_without_calibration_behaves_exactly_as_before():
    """Existing bundles and callers must not start failing a check they never
    ran."""
    from retailgr.evaluation.ranking import ranker_gate

    verdict = ranker_gate({"signals": {"blended": {"mrr": 0.70}}}, {"mrr": 0.59})
    assert verdict["passed"] is True
    assert "calibration_passed" in verdict
    assert verdict["calibration_passed"] is True


# -- the control --------------------------------------------------------------


def test_retrieval_and_the_ranker_are_labelled_by_the_same_code():
    """Resolution is a property of a score *and its label*. If the control
    were labelled "any target" while the ranker was labelled "a target they
    purchased", the difference between the two numbers would be the labels,
    not the models — and it would read as a damning result about the ranker."""
    from retailgr.evaluation.ranking import collect_slate_probabilities, slate_head_labels

    vocab = 40
    split = _one_user_split(
        [3, 4, 5], vocab, targets=[7, 8], target_actions=["click", "purchase"]
    )
    candidates = np.array([7, 8, 10], dtype=np.int64)
    direct = slate_head_labels(candidates, split.targets[0], split.target_actions[0])

    collected = collect_slate_probabilities(
        _tiny_ranker(vocab), _FixedRetrieval(vocab, [7, 8, 10]), split, vocab, candidate_k=3
    )
    for head in ("click", "cart", "purchase"):
        assert np.array_equal(collected[head]["labels"], direct[head]), head


def test_slate_head_labels_separate_engagement_from_purchase():
    from retailgr.evaluation.ranking import slate_head_labels

    candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    targets = np.array([2, 3], dtype=np.int64)
    from retailgr.actions import encode_actions

    labels = slate_head_labels(candidates, targets, encode_actions(["view", "purchase"]))
    # A view is a target but qualifies for no head; a purchase qualifies for all.
    assert labels["click"].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert labels["purchase"].tolist() == [0.0, 0.0, 1.0, 0.0]
    # "any" keeps both, which is why it is not what the heads are scored on.
    assert labels["any"].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_the_control_measures_resolution_without_pretending_to_calibration():
    """Retrieval emits scores, not probabilities. Reporting an ECE or a bias
    ratio for them would be reporting a comparison between a rank and a
    rate."""
    from retailgr.evaluation.ranking import retrieval_slate_resolution

    vocab = 40
    split = _one_user_split([3, 4, 5], vocab, targets=[7, 9])
    result = retrieval_slate_resolution(
        _FixedRetrieval(vocab, [7, 9, 10, 11, 12]), split, vocab, candidate_k=5, max_users=5
    )
    assert result["users"] == 1
    click = result["heads"]["click"]
    assert "resolution_share" in click
    assert "brier_skill" in click
    for meaningless in ("bias_ratio", "ece", "ece_relative", "verdict", "mean_predicted"):
        assert meaningless not in click


def test_the_control_is_invariant_to_a_monotone_rescaling_of_the_scores():
    """It bins by rank percentile, and quantile bins do not move under a
    monotone map — so the number reported is the raw scores' resolution, not
    an artefact of the squashing."""
    from retailgr.evaluation.calibration import calibration_report

    rng = np.random.default_rng(12)
    raw = rng.normal(0, 5, 20000)
    labels = (rng.uniform(size=raw.size) < 1 / (1 + np.exp(-raw / 3)) * 0.02).astype(float)

    def resolution(scores):
        order = np.argsort(scores, kind="mergesort")
        percentile = np.empty(scores.size)
        percentile[order] = np.linspace(0.0, 1.0, scores.size)
        return calibration_report(labels, percentile).resolution_share

    assert resolution(raw) == pytest.approx(resolution(np.exp(raw / 10)), rel=1e-9)
    assert resolution(raw) == pytest.approx(resolution(raw * 100 + 7), rel=1e-9)
