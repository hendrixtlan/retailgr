"""Tests for the uncertainty machinery.

A bug here is worse than a bug in a model: it does not make the numbers wrong,
it makes wrong numbers look trustworthy. So these tests check the properties
that matter — that a known interval is recovered, that the paired test has the
power the pairing is supposed to buy, that it does not find differences that
are not there, and that NaN means "undefined" rather than "zero".
"""

from __future__ import annotations

import numpy as np
import pytest

from retailgr.evaluation.metrics import MetricAccumulator
from retailgr.evaluation.stats import (
    bootstrap_ci,
    describe_significance,
    paired_comparison,
    seed_spread,
)

# -- bootstrap ----------------------------------------------------------------


def test_interval_brackets_the_mean_and_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = bootstrap_ci(rng.normal(0.5, 0.2, 50), seed=1)
    large = bootstrap_ci(rng.normal(0.5, 0.2, 5000), seed=1)
    assert small.low < small.mean < small.high
    assert large.low < large.mean < large.high
    # Four decimal places of extra data must buy a tighter interval.
    assert large.half_width < small.half_width / 5


def test_interval_recovers_a_known_standard_error():
    """For a large sample the bootstrap half-width should land near
    1.96 * sd / sqrt(n)."""
    rng = np.random.default_rng(7)
    values = rng.normal(0.3, 0.1, 4000)
    interval = bootstrap_ci(values, seed=3)
    analytic = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    assert interval.half_width == pytest.approx(analytic, rel=0.15)


def test_constant_values_give_a_zero_width_interval():
    interval = bootstrap_ci([0.4] * 100, seed=0)
    assert interval.mean == pytest.approx(0.4)
    assert interval.half_width == pytest.approx(0.0)


def test_nan_values_are_dropped_not_treated_as_zero():
    """A user with no targets has an undefined metric. Counting it as 0 would
    drag every mean toward zero by however many such users there are."""
    values = [0.5, 0.5, float("nan"), 0.5]
    interval = bootstrap_ci(values, seed=0)
    assert interval.mean == pytest.approx(0.5)
    assert interval.n == 3


def test_empty_and_single_value_inputs():
    assert bootstrap_ci([]) is None
    assert bootstrap_ci([float("nan")]) is None
    single = bootstrap_ci([0.7])
    assert single.n == 1
    assert single.low == single.high == pytest.approx(0.7)


def test_confidence_level_is_respected():
    rng = np.random.default_rng(11)
    values = rng.normal(0.5, 0.2, 800)
    narrow = bootstrap_ci(values, confidence=0.80, seed=2)
    wide = bootstrap_ci(values, confidence=0.99, seed=2)
    assert narrow.half_width < wide.half_width


# -- paired comparison --------------------------------------------------------


def test_paired_test_finds_a_small_consistent_difference():
    """The point of pairing: a difference far smaller than the between-user
    spread is still detectable when it is consistent per user."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.4, 0.25, 600)  # huge spread across users
    better = base + 0.01  # tiny, but every single user improves

    comparison = paired_comparison(better, base, seed=5)
    assert comparison.significant
    assert comparison.mean_difference == pytest.approx(0.01, abs=1e-9)
    assert comparison.p_value < 0.01


def test_unpaired_intervals_would_have_missed_it():
    """The same data, compared as two independent means, overlaps completely —
    which is why this module exists."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.4, 0.25, 600)
    better = base + 0.01

    a = bootstrap_ci(better, seed=5)
    b = bootstrap_ci(base, seed=6)
    assert a.low < b.high and b.low < a.high  # the intervals overlap
    assert paired_comparison(better, base, seed=5).significant


def test_paired_test_reports_no_difference_when_there_is_none():
    rng = np.random.default_rng(4)
    base = rng.normal(0.4, 0.2, 500)
    noise = base + rng.normal(0.0, 0.05, 500)  # zero-mean perturbation

    comparison = paired_comparison(noise, base, seed=8)
    assert not comparison.significant
    assert comparison.p_value > 0.05
    assert comparison.verdict() == "no difference detected"


def test_paired_test_detects_a_regression_and_names_it():
    rng = np.random.default_rng(5)
    base = rng.normal(0.4, 0.1, 400)
    worse = base * 0.8

    comparison = paired_comparison(worse, base, seed=9)
    assert comparison.significant
    assert comparison.mean_difference < 0
    assert "worse" in comparison.verdict()
    assert comparison.relative_difference == pytest.approx(-0.2, rel=0.05)


def test_paired_test_needs_aligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        paired_comparison([0.1, 0.2, 0.3], [0.1, 0.2])


def test_paired_test_uses_pairwise_complete_rows():
    a = [0.5, float("nan"), 0.7, 0.9]
    b = [0.4, 0.3, float("nan"), 0.8]
    comparison = paired_comparison(a, b, seed=0)
    # Only rows 0 and 3 are usable in both.
    assert comparison.n == 2


def test_p_value_is_never_exactly_zero():
    """A finite number of resamples cannot demonstrate p = 0, and reporting it
    overstates the evidence."""
    base = np.zeros(200)
    better = np.ones(200)
    comparison = paired_comparison(better, base, resamples=200, seed=0)
    assert comparison.p_value > 0
    assert comparison.p_value <= 1 / 201 + 1e-12


def test_empty_paired_comparison_returns_none():
    assert paired_comparison([], []) is None
    assert paired_comparison([float("nan")], [0.5]) is None


def test_describe_significance_says_what_it_did_not_measure():
    rng = np.random.default_rng(2)
    base = rng.normal(0.4, 0.1, 300)
    text = describe_significance(paired_comparison(base + 0.05, base, seed=1))
    assert "training-seed" in text
    assert describe_significance(None).startswith("No paired")


# -- seed spread --------------------------------------------------------------


def test_seed_spread_summarises_runs():
    spread = seed_spread([0.40, 0.44, 0.42])
    assert spread["seeds"] == 3
    assert spread["mean"] == pytest.approx(0.42, abs=1e-6)
    assert spread["min"] == 0.40
    assert spread["max"] == 0.44
    assert spread["std"] > 0


def test_seed_spread_of_one_run_has_no_spread():
    spread = seed_spread([0.42])
    assert spread["std"] == 0.0
    assert spread["min"] == spread["max"]


def test_seed_spread_ignores_missing_runs():
    assert seed_spread([0.4, None if False else float("nan")])["seeds"] == 1
    assert seed_spread([]) is None


# -- the accumulator ---------------------------------------------------------


def test_accumulator_retains_per_user_values_aligned_with_user_ids():
    accumulator = MetricAccumulator([2], vocab_size=10)
    accumulator.update([1, 2], {1}, user_id="u1")
    accumulator.update([3, 4], {9}, user_id="u2")

    per_user = accumulator.per_user()
    assert accumulator.user_ids == ["u1", "u2"]
    assert per_user["recall@2"] == [1.0, 0.0]
    assert len(per_user["ndcg@2"]) == len(accumulator.user_ids)


def test_accumulator_keeps_nan_in_place_to_preserve_alignment():
    """An undefined metric must still occupy its row, or a paired comparison
    would silently line up different users."""
    accumulator = MetricAccumulator([2], vocab_size=10)
    accumulator.update([1, 2], {1}, user_id="u1")
    accumulator.update([1, 2], [], user_id="u2")  # no targets -> undefined

    values = accumulator.per_user()["recall@2"]
    assert len(values) == 2
    assert values[0] == 1.0
    assert np.isnan(values[1])
    # The mean still ignores it.
    assert accumulator.results()["recall@2"] == 1.0


def test_accumulator_intervals_cover_every_per_user_metric():
    accumulator = MetricAccumulator([5], vocab_size=100)
    rng = np.random.default_rng(0)
    for index in range(200):
        targets = set(rng.integers(1, 100, size=3).tolist())
        ranked = rng.integers(1, 100, size=5).tolist()
        accumulator.update(ranked, targets, user_id=f"u{index}")

    intervals = accumulator.intervals(resamples=200)
    assert {"recall@5", "ndcg@5", "hit_rate@5"} <= set(intervals)
    for bounds in intervals.values():
        assert bounds["ci_low"] <= bounds["mean"] <= bounds["ci_high"]
    # Coverage is a property of the whole set, not of one user.
    assert "coverage@5" not in intervals


def test_accumulator_can_skip_per_user_retention():
    accumulator = MetricAccumulator([2], vocab_size=10, keep_per_user=False)
    accumulator.update([1, 2], {1}, user_id="u1")
    assert accumulator.per_user() == {}
    assert accumulator.results()["recall@2"] == 1.0


# -- multiple comparisons -----------------------------------------------------


def test_holm_keeps_the_strongest_and_drops_the_marginal():
    from retailgr.evaluation.stats import holm_bonferroni

    # Four tests, as the head-to-head table produces.
    result = holm_bonferroni({"a": 0.001, "b": 0.016, "c": 0.033, "d": 0.4})
    assert result["a"]["significant"] is True
    # 0.033 with four tests in the family does not survive.
    assert result["c"]["significant"] is False
    assert result["d"]["significant"] is False
    # Adjustment can only make a p-value larger.
    for name, values in result.items():
        assert values["p_adjusted"] >= values["p_value"], name


def test_holm_is_monotone_in_the_original_order():
    from retailgr.evaluation.stats import holm_bonferroni

    result = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03})
    assert result["a"]["p_adjusted"] <= result["b"]["p_adjusted"]
    assert result["b"]["p_adjusted"] <= result["c"]["p_adjusted"]


def test_holm_with_one_test_changes_nothing():
    from retailgr.evaluation.stats import holm_bonferroni

    result = holm_bonferroni({"only": 0.04})
    assert result["only"]["p_adjusted"] == pytest.approx(0.04)
    assert result["only"]["significant"] is True


def test_holm_is_less_conservative_than_plain_bonferroni():
    from retailgr.evaluation.stats import holm_bonferroni

    # Plain Bonferroni would scale every p by 4; Holm only scales the largest
    # by 4, so a strong result keeps more of its evidence.
    result = holm_bonferroni({"a": 0.004, "b": 0.5, "c": 0.6, "d": 0.7})
    assert result["a"]["p_adjusted"] == pytest.approx(0.016)
    assert result["a"]["significant"] is True


def test_holm_caps_adjusted_values_at_one():
    from retailgr.evaluation.stats import holm_bonferroni

    result = holm_bonferroni({"a": 0.5, "b": 0.6, "c": 0.9})
    for values in result.values():
        assert values["p_adjusted"] <= 1.0


def test_holm_on_an_empty_family():
    from retailgr.evaluation.stats import holm_bonferroni

    assert holm_bonferroni({}) == {}
