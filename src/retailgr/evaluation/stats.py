"""Uncertainty on the numbers.

Every metric in this project is a mean over users, and a mean over ~1,100
users has a confidence interval wide enough to swallow most of the
differences worth arguing about. Reporting point estimates and then reasoning
about a 2% gap is how teams ship models that are indistinguishable from the
one they replaced.

Two kinds of uncertainty, and they are not interchangeable:

1. **User sampling** — would this difference hold on a different sample of
   users? Answered by a bootstrap over users, and for model-vs-model by a
   *paired* test, since both models saw the same users and the pairing removes
   the between-user variance that otherwise dominates.
2. **Training variance** — would this difference hold if we re-ran with a
   different seed? A bootstrap says nothing about this. It needs several
   training runs, which is what ``experiment.run_ablation``'s ``seeds``
   argument is for.

Reporting only the first while implying the second is a common and expensive
mistake, so the report labels which one it measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE = 0.95


@dataclass
class Interval:
    """A mean with its confidence interval."""

    mean: float
    low: float
    high: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "ci_low": round(self.low, 6),
            "ci_high": round(self.high, 6),
            "n": self.n,
            "confidence": self.confidence,
        }

    def format(self, digits: int = 4) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


@dataclass
class PairedComparison:
    """The difference between two models on the same users."""

    mean_a: float
    mean_b: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def significant(self) -> bool:
        """True when the interval for the difference excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    @property
    def relative_difference(self) -> float | None:
        if self.mean_b == 0:
            return None
        return self.mean_difference / self.mean_b

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "mean_difference": round(self.mean_difference, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "relative_difference": (
                round(self.relative_difference, 5)
                if self.relative_difference is not None
                else None
            ),
            "p_value": round(self.p_value, 5),
            "n": self.n,
            "significant": self.significant,
            "confidence": self.confidence,
        }

    def verdict(self, digits: int = 4) -> str:
        """One phrase a reader can act on."""
        if not self.significant:
            return "no difference detected"
        direction = "better" if self.mean_difference > 0 else "worse"
        relative = self.relative_difference
        magnitude = f"{abs(relative) * 100:.1f}%" if relative is not None else "-"
        return f"{magnitude} {direction}"

    def format(self, digits: int = 4) -> str:
        return (
            f"{self.mean_difference:+.{digits}f} "
            f"[{self.ci_low:+.{digits}f}, {self.ci_high:+.{digits}f}], "
            f"p={self.p_value:.3f}"
        )


def _clean(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def bootstrap_ci(
    values: Any,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval | None:
    """Percentile bootstrap CI for the mean of ``values``.

    ``values`` is one number per user. NaNs — users with no targets, whose
    metric is undefined rather than zero — are dropped, not counted as zero.
    """
    array = _clean(values)
    if array.size == 0:
        return None
    if array.size == 1:
        single = float(array[0])
        return Interval(single, single, single, 1, confidence)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Interval(
        mean=float(array.mean()),
        low=float(low),
        high=float(high),
        n=int(array.size),
        confidence=confidence,
    )


def paired_comparison(
    values_a: Any,
    values_b: Any,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> PairedComparison | None:
    """Compare two models measured on the same users, in the same order.

    Pairing matters: the spread *between users* is far larger than the spread
    between two models on one user, so comparing two independent intervals
    would call almost everything a tie. The per-user difference removes that
    nuisance variance.

    The p-value is a sign-flip permutation test: under the null hypothesis the
    two models are interchangeable, so the sign of each user's difference is a
    coin flip. That assumption is exactly right for a paired design and needs
    no normality.
    """
    array_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    array_b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if array_a.size != array_b.size:
        raise ValueError(
            f"paired comparison needs aligned values, got {array_a.size} and {array_b.size}"
        )

    usable = np.isfinite(array_a) & np.isfinite(array_b)
    array_a, array_b = array_a[usable], array_b[usable]
    if array_a.size == 0:
        return None

    differences = array_a - array_b
    observed = float(differences.mean())

    rng = np.random.default_rng(seed)
    if array_a.size == 1:
        low = high = observed
        p_value = 1.0
    else:
        # Bootstrap the mean difference.
        indices = rng.integers(0, differences.size, size=(resamples, differences.size))
        boot = differences[indices].mean(axis=1)
        alpha = (1 - confidence) / 2
        low, high = (float(v) for v in np.quantile(boot, [alpha, 1 - alpha]))

        # Sign-flip permutation test.
        signs = rng.choice((-1.0, 1.0), size=(resamples, differences.size))
        permuted = (differences * signs).mean(axis=1)
        # +1 in numerator and denominator: a p-value of exactly 0 overstates
        # what a finite number of resamples can show.
        p_value = float((np.abs(permuted) >= abs(observed)).sum() + 1) / (resamples + 1)

    return PairedComparison(
        mean_a=float(array_a.mean()),
        mean_b=float(array_b.mean()),
        mean_difference=observed,
        ci_low=low,
        ci_high=high,
        p_value=p_value,
        n=int(array_a.size),
        confidence=confidence,
    )


def seed_spread(values: Any) -> dict[str, Any] | None:
    """Summarise one metric across training seeds.

    This is the other variance, the one a bootstrap cannot see. With only a
    handful of seeds the right summary is the range, not a CI: three runs do
    not support a confidence statement, but they do show whether a claimed
    difference is larger than the noise between identical configurations.
    """
    array = _clean(values)
    if array.size == 0:
        return None
    return {
        "seeds": int(array.size),
        "mean": round(float(array.mean()), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        # ddof=1: these are a sample of possible runs, not the population.
        "std": round(float(array.std(ddof=1)), 6) if array.size > 1 else 0.0,
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, Any]:
    """Adjust a family of p-values for multiple comparisons, Holm's method.

    A report that shows eight tests and calls the ones under 0.05 significant
    is running eight chances to be fooled, not one: with eight true nulls the
    odds of at least one false positive are about 34%. Holm's step-down
    procedure controls that family-wise error rate and, unlike plain
    Bonferroni, loses almost no power doing it.

    This matters here because the reports present whole tables of comparisons
    — every variant against the baseline, every ablation against the full
    model — and the marginal ones (p around 0.03) are exactly the ones an
    adjustment changes its mind about.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        # Step-down: each p is scaled by the number of tests still in play,
        # and adjusted values are made monotone.
        scaled = min(1.0, (count - index) * p_value)
        running = max(running, scaled)
        adjusted[name] = running
    return {
        name: {
            "p_value": round(p_values[name], 5),
            "p_adjusted": round(adjusted[name], 5),
            "significant": bool(adjusted[name] <= alpha),
        }
        for name in p_values
    }


def describe_significance(comparison: PairedComparison | None) -> str:
    """A sentence for the report, honest about what was and was not shown."""
    if comparison is None:
        return "No paired comparison was possible."
    if comparison.significant:
        return (
            f"The difference is {comparison.format()} over {comparison.n} shared users, "
            "so the interval excludes zero. This covers user sampling only, not "
            "training-seed variance."
        )
    return (
        f"The difference is {comparison.format()} over {comparison.n} shared users, "
        "so the interval includes zero: on this data the two are "
        "indistinguishable, whatever the point estimates suggest."
    )
