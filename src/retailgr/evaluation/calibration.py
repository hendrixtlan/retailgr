"""Are the ranker's probabilities numbers, or just scores?

``ScoreBlend`` computes an expected value::

    score = v_purchase * P(purchase) * (1 - penalty * P(return))
          + v_cart * P(cart) + v_click * P(click)

Multiplying a probability by a euro amount only means something if the
probability is on the probability scale. AUC — the only thing the ranker
report measured until now — is invariant to any monotone transform of the
scores, so a head can have AUC 0.98 and still be wrong by a factor of twenty
in absolute terms. In a blend, that factor does not cancel: it silently
re-weights the heads relative to each other, and the business weights the
operator set are no longer the weights the system applies. That failure has
already happened once in this project, and it was found by accident rather
than by measurement. This module is the measurement.

**Which probabilities to calibrate.** The heads produce a number at two very
different kinds of position, and only one of them is what the blend consumes:

1. *Teacher-forced* positions — "this item is in the history; which action did
   the customer take on it?". A real conditional with a real base rate.
2. *Candidate* positions — "this item is being offered; will they act on it?".
   This is what serving scores and what the blend multiplies by money.

The candidate-position base rate seen during **training** is an artefact of
the sampling design: one true item against ``num_negatives`` uniform-random
ones gives a positive rate of 1/(1+N) regardless of what any real rate is.
Calibrating against that number would be calibrating to the sampler. Worse,
serving does not show uniform-random items — it shows retrieval's top-k, a
far harder and quite differently distributed set — so this is covariate shift,
not just prior shift, and no scalar prior correction repairs it.

So everything here is fitted and measured on **slates shaped like serving's**:
retrieval's top-k candidates, scored from a real prefix. That is the only
distribution on which a calibrated number is the number the blend needs.

**What a calibrated probability here does and does not mean.** The positive is
the item the customer actually interacted with next. Every other candidate is
labelled 0, including the ones they would have liked and never saw. So these
probabilities are calibrated to *observed* next-interaction, which is a lower
bound on relevance, and the gap is the usual missing-not-at-random problem of
recommender logs. A calibrated P(purchase) of 0.004 means "4 in 1000 slates
like this one had this item as the next purchase", not "4 in 1000 customers
would buy it". Fixing that needs exposure logs, which is a separate piece of
work; saying so is cheaper than pretending otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPSILON = 1e-7
DEFAULT_BINS = 10


def _as_arrays(labels: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size != p.size:
        raise ValueError(f"labels and probabilities must align, got {y.size} and {p.size}")
    usable = np.isfinite(y) & np.isfinite(p)
    return y[usable], np.clip(p[usable], 0.0, 1.0)


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(z: np.ndarray) -> np.ndarray:
    # The two-branch form avoids overflow in exp for large |z|.
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def brier_score(labels: Any, probabilities: Any) -> float:
    """Mean squared error of a probabilistic forecast.

    Unlike AUC this is a *proper* scoring rule: it is minimised only by the
    true probability, so it punishes miscalibration as well as bad ordering.
    On its own it is nearly unreadable for a rare event — see
    ``CalibrationReport.brier_skill`` for the version that is.
    """
    y, p = _as_arrays(labels, probabilities)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


# -- the reliability table ----------------------------------------------------


@dataclass
class ReliabilityBin:
    """One row of a reliability table: what was promised, what happened."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return self.mean_predicted - self.observed_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 6),
            "observed_rate": round(self.observed_rate, 6),
            "gap": round(self.gap, 6),
        }


def reliability_bins(
    labels: Any,
    probabilities: Any,
    bins: int = DEFAULT_BINS,
    strategy: str = "quantile",
) -> list[ReliabilityBin]:
    """Group predictions and compare the promise against the outcome.

    ``strategy="quantile"`` (equal mass) is the default and it is not a
    stylistic choice. These heads predict a rare event: on the synthetic set
    every P(purchase) lands between 0.0001 and 0.007. Equal-width bins over
    [0, 1] put all of it in the first bin, and a one-bin reliability curve
    cannot show miscalibration that varies across the range — it reports the
    average gap and calls it a day. Equal-mass bins put the resolution where
    the predictions actually are.

    ``strategy="uniform"`` is kept for the case where the absolute scale is
    the point, and for comparing against published numbers, which almost
    always use equal-width bins.
    """
    y, p = _as_arrays(labels, probabilities)
    if y.size == 0:
        return []
    bins = max(1, int(bins))

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, bins + 1))
        # Ties collapse bins; keeping the duplicates would create empty bins
        # with undefined means.
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([p.min(), p.min() + EPSILON])
    else:
        raise ValueError(f"unknown binning strategy '{strategy}'")

    # ``np.digitize`` with right=False puts the maximum in an overflow bin.
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, edges.size - 2)

    out: list[ReliabilityBin] = []
    for position in range(edges.size - 1):
        members = index == position
        count = int(members.sum())
        if count == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(edges[position]),
                upper=float(edges[position + 1]),
                count=count,
                mean_predicted=float(p[members].mean()),
                observed_rate=float(y[members].mean()),
            )
        )
    return out


@dataclass
class CalibrationReport:
    """Everything needed to decide whether a probability can be multiplied."""

    n: int
    positives: int
    base_rate: float
    mean_predicted: float
    brier: float
    brier_baseline: float
    ece: float
    mce: float
    reliability: float
    resolution: float
    uncertainty: float
    # The Brier score of the *binned* forecast — every prediction replaced by
    # its bin's mean. Murphy's identity is exact for this, and only
    # approximate for a continuous forecast; keeping both makes the identity
    # checkable and turns the difference between them into a measurement of
    # its own. See ``overdispersion``.
    binned_brier: float = 0.0
    bins: list[ReliabilityBin] = field(default_factory=list)

    @property
    def brier_skill(self) -> float:
        """Brier against the best constant forecast: always predict the base rate.

        For a rare event the raw Brier is dominated by the event's rarity — a
        purchase head with base rate 0.003 scores about 0.003 by predicting
        zero everywhere, which looks excellent and is worthless. The skill
        score makes the trivial baseline the zero point: below zero means the
        head is worse than a constant.
        """
        if self.brier_baseline <= 0:
            return float("nan")
        return 1.0 - self.brier / self.brier_baseline

    @property
    def bias_ratio(self) -> float:
        """Mean predicted over base rate: the single most legible number here.

        1.0 is right on average. 0.05 means the head believes the event is
        twenty times rarer than it is — which in a blend means its business
        weight is being divided by twenty before anything else happens.
        """
        if self.base_rate <= 0:
            return float("nan")
        return self.mean_predicted / self.base_rate

    @property
    def ece_relative(self) -> float:
        """ECE as a share of the base rate.

        An ECE of 0.003 is negligible for a coin flip and total for an event
        that happens three times in a thousand. Absolute ECE cannot be
        compared across these four heads; this can.
        """
        if self.base_rate <= 0:
            return float("nan")
        return self.ece / self.base_rate

    @property
    def resolution_share(self) -> float:
        """Resolution as a share of the uncertainty: how much the head *knows*.

        The one number calibration cannot improve and the one that decides
        whether calibrating is worth doing. A head whose predictions vary
        wildly while the outcomes behind them do not has resolution near
        zero, and correcting its scale produces something that is genuinely
        well calibrated and genuinely useless — a dressed-up constant.

        Reported as a share so it is comparable across heads with different
        base rates, and so the two distributions a head lives on can be put
        side by side.
        """
        if self.uncertainty <= 0:
            return float("nan")
        return self.resolution / self.uncertainty

    @property
    def overdispersion(self) -> float:
        """How much of the Brier score is spread *inside* the bins.

        Murphy's three terms describe the binned forecast. What they cannot
        see is a head that shouts a range of numbers within a bin whose
        outcomes are all the same, and that is a specific and common failure:
        predictions spanning three orders of magnitude over outcomes that
        span a factor of two. The gap between the raw and the binned Brier
        score measures exactly that, and a calibrator is very good at
        removing it.
        """
        return self.brier - self.binned_brier

    def verdict(self) -> str:
        """One phrase, in the terms the blend cares about."""
        if self.n < 50 or self.positives < 5:
            return "too few outcomes to judge"
        ratio = self.bias_ratio
        if not np.isfinite(ratio):
            return "no positives, calibration undefined"
        # Order matters: a head with no resolution is trivially calibratable,
        # and calling it "calibrated" without qualification would be the most
        # misleading thing this report could say.
        informative = np.isfinite(self.resolution_share) and self.resolution_share >= 0.01
        if 0.8 <= ratio <= 1.25 and self.ece_relative < 0.25:
            return "calibrated" if informative else "calibrated but uninformative"
        direction = "over" if ratio > 1 else "under"
        factor = ratio if ratio > 1 else (1.0 / ratio if ratio > 0 else float("inf"))
        suffix = "" if informative else ", and uninformative"
        return f"{direction}-confident by {factor:.1f}x{suffix}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "positives": self.positives,
            "base_rate": round(self.base_rate, 6),
            "mean_predicted": round(self.mean_predicted, 6),
            "bias_ratio": (
                round(self.bias_ratio, 4) if np.isfinite(self.bias_ratio) else None
            ),
            "brier": round(self.brier, 8),
            "binned_brier": round(self.binned_brier, 8),
            "overdispersion": round(self.overdispersion, 8),
            "brier_baseline": round(self.brier_baseline, 8),
            "brier_skill": (
                round(self.brier_skill, 5) if np.isfinite(self.brier_skill) else None
            ),
            "resolution_share": (
                round(self.resolution_share, 6)
                if np.isfinite(self.resolution_share)
                else None
            ),
            "ece": round(self.ece, 6),
            "ece_relative": (
                round(self.ece_relative, 4) if np.isfinite(self.ece_relative) else None
            ),
            "mce": round(self.mce, 6),
            "reliability": round(self.reliability, 8),
            "resolution": round(self.resolution, 8),
            "uncertainty": round(self.uncertainty, 8),
            "verdict": self.verdict(),
            "bins": [b.as_dict() for b in self.bins],
        }


def calibration_report(
    labels: Any,
    probabilities: Any,
    bins: int = DEFAULT_BINS,
    strategy: str = "quantile",
) -> CalibrationReport | None:
    """Measure how well ``probabilities`` predict ``labels`` in absolute terms.

    Reports the Murphy decomposition alongside the usual numbers::

        binned_brier = reliability - resolution + uncertainty

    ``reliability`` is the miscalibration (0 is perfect, lower is better) and
    ``resolution`` is how much the forecast actually varies with the outcome
    (higher is better). They answer different questions, and a single Brier
    number cannot tell "the probabilities are wrong" apart from "the
    probabilities are all the same". A calibrator fixes the first and cannot
    touch the second, so the split predicts in advance how much calibrating
    can possibly help — and, more usefully, says when it cannot help at all.

    The identity is against ``binned_brier``, not ``brier``, and the
    difference is deliberate. Murphy's three terms describe a forecast that
    has been grouped into bins; the raw Brier score also carries the spread
    *within* each bin, which is invisible to all three. Reporting both makes
    the identity checkable and names that spread — see ``overdispersion``.
    """
    y, p = _as_arrays(labels, probabilities)
    if y.size == 0:
        return None

    table = reliability_bins(y, p, bins=bins, strategy=strategy)
    base_rate = float(y.mean())
    weights = np.array([b.count for b in table], dtype=np.float64) / y.size
    gaps = np.array([abs(b.gap) for b in table], dtype=np.float64)
    predicted = np.array([b.mean_predicted for b in table], dtype=np.float64)
    observed = np.array([b.observed_rate for b in table], dtype=np.float64)

    reliability = float((weights * (predicted - observed) ** 2).sum()) if table else 0.0
    resolution = float((weights * (observed - base_rate) ** 2).sum()) if table else 0.0
    uncertainty = float(base_rate * (1.0 - base_rate))

    return CalibrationReport(
        n=int(y.size),
        positives=int(y.sum()),
        base_rate=base_rate,
        mean_predicted=float(p.mean()),
        brier=float(np.mean((p - y) ** 2)),
        binned_brier=reliability - resolution + uncertainty,
        # The constant forecast that a calibrated model must beat.
        brier_baseline=uncertainty,
        ece=float((weights * gaps).sum()) if table else float("nan"),
        mce=float(gaps.max()) if table else float("nan"),
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        bins=table,
    )


# -- calibrators --------------------------------------------------------------


class Calibrator:
    """A monotone map from a raw probability to a calibrated one.

    Monotone on purpose: a calibrator must not reorder a head's own
    candidates, so per-head AUC is invariant and calibration cannot smuggle in
    a ranking change. What it does change is the *blend*, because the blend
    combines heads whose scales were previously incomparable. That is the
    whole point, and ``tests/test_calibration.py`` pins both halves of it.
    """

    kind = "identity"

    def apply(self, probabilities: Any) -> np.ndarray:
        return np.asarray(probabilities, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


class IdentityCalibrator(Calibrator):
    """No correction. Used when a head has too few outcomes to fit on."""

    kind = "identity"


class PlattCalibrator(Calibrator):
    """``sigmoid(a * logit(p) + b)``, fitted by Newton's method on the NLL.

    Two parameters, so it survives the small validation sets the return head
    leaves behind, and it can only stretch and shift the logit — which is
    exactly the "right shape, wrong scale" failure a rare-event head has.

    Fitted against Platt's smoothed targets rather than 0/1. With a base rate
    near 1/300 the classes are close to separable in the tail, and unsmoothed
    targets drive ``a`` to infinity chasing a perfect fit on a handful of
    points. The smoothing is one line and it is the difference between a
    calibrator and an overfit.

    Newton needs a line search here, which is not the usual caveat about
    convergence speed. When every prediction sits near zero the weights
    ``p(1-p)`` are all tiny, the Hessian is near-singular, and an undamped
    step overshoots by orders of magnitude — the first version of this fitted
    ``a = 6.6e12`` on a well-behaved synthetic set, which saturates the
    sigmoid to exact 0 and 1 in float64 and destroys the ordering the
    calibrator is supposed to preserve. Backtracking on the objective costs a
    few evaluations and removes the failure mode.
    """

    kind = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    @staticmethod
    def _nll(z: np.ndarray, targets: np.ndarray, a: float, b: float) -> float:
        """Negative log-likelihood, in the form that does not overflow."""
        s = a * z + b
        # softplus(-s), stable for either sign.
        softplus = np.log1p(np.exp(-np.abs(s))) + np.maximum(-s, 0.0)
        return float((softplus + (1.0 - targets) * s).sum())

    def fit(
        self, labels: Any, probabilities: Any, iterations: int = 100, ridge: float = 1e-10
    ) -> PlattCalibrator:
        y, p = _as_arrays(labels, probabilities)
        if y.size == 0:
            return self
        z = logit(p)

        n_pos = float(y.sum())
        n_neg = float(y.size - n_pos)
        # Platt (1999) section 2.2.
        high = (n_pos + 1.0) / (n_pos + 2.0)
        low = 1.0 / (n_neg + 2.0)
        targets = np.where(y > 0, high, low)

        # Start from the constant forecast that already matches the base rate:
        # slope 1, intercept shifted so the mean prediction is right. Most of
        # the correction a rare-event head needs is exactly that shift, so
        # Newton starts near the answer instead of in the flat region.
        a = 1.0
        b = float(logit(np.array([targets.mean()]))[0] - z.mean())
        objective = self._nll(z, targets, a, b)

        for _ in range(iterations):
            predicted = sigmoid(a * z + b)
            residual = predicted - targets
            grad_a = float((residual * z).sum())
            grad_b = float(residual.sum())
            if max(abs(grad_a), abs(grad_b)) < 1e-9:
                break

            weight = predicted * (1.0 - predicted)
            h_aa = float((weight * z * z).sum()) + ridge
            h_ab = float((weight * z).sum())
            h_bb = float(weight.sum()) + ridge
            determinant = h_aa * h_bb - h_ab * h_ab
            if not np.isfinite(determinant) or abs(determinant) < 1e-300:
                break

            step_a = (h_bb * grad_a - h_ab * grad_b) / determinant
            step_b = (h_aa * grad_b - h_ab * grad_a) / determinant

            # Backtracking: accept the first step that actually improves the
            # objective. Without this the near-singular Hessian sends the fit
            # to infinity on its first iteration.
            scale = 1.0
            improved = False
            for _ in range(40):
                trial_a, trial_b = a - scale * step_a, b - scale * step_b
                trial = self._nll(z, targets, trial_a, trial_b)
                if np.isfinite(trial) and trial < objective - 1e-12:
                    a, b, objective, improved = trial_a, trial_b, trial, True
                    break
                scale *= 0.5
            if not improved:
                break

        self.a, self.b = float(a), float(b)
        return self

    def apply(self, probabilities: Any) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        return sigmoid(self.a * logit(p) + self.b)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "a": self.a, "b": self.b}


class IsotonicCalibrator(Calibrator):
    """Pool-adjacent-violators: the most flexible monotone map there is.

    Strictly better than Platt at fitting the training data, which is the
    problem — with a few thousand points and a handful of positives it will
    happily produce a step function that reproduces the validation noise. The
    fit keeps the knots and interpolates between them, and
    ``fit_calibrator`` only reaches for it when there are enough positives to
    support it.
    """

    kind = "isotonic"

    def __init__(self, x: Any = (), y: Any = ()):
        self.x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)

    def fit(self, labels: Any, probabilities: Any) -> IsotonicCalibrator:
        y, p = _as_arrays(labels, probabilities)
        if y.size == 0:
            return self

        order = np.argsort(p, kind="mergesort")
        xs, ys = p[order], y[order]

        # Pool equal x first: a monotone function cannot separate them, and
        # leaving them apart lets PAV create a step inside a tie.
        unique_x, inverse = np.unique(xs, return_inverse=True)
        sums = np.bincount(inverse, weights=ys, minlength=unique_x.size)
        counts = np.bincount(inverse, minlength=unique_x.size).astype(np.float64)

        # PAV over blocks of (total label, total weight, positions covered).
        totals: list[float] = []
        weights: list[float] = []
        spans: list[int] = []
        for index in range(unique_x.size):
            totals.append(float(sums[index]))
            weights.append(float(counts[index]))
            spans.append(1)
            # Merge backwards while the sequence of block means decreases.
            while (
                len(totals) > 1
                and totals[-2] / weights[-2] > totals[-1] / weights[-1]
            ):
                totals[-2:] = [totals[-2] + totals[-1]]
                weights[-2:] = [weights[-2] + weights[-1]]
                spans[-2:] = [spans[-2] + spans[-1]]

        # Expand the blocks back onto the x grid so ``apply`` can interpolate.
        fitted = np.empty(unique_x.size, dtype=np.float64)
        position = 0
        for total, weight, span in zip(totals, weights, spans, strict=True):
            fitted[position : position + span] = total / weight
            position += span

        self.x, self.y = unique_x, fitted
        return self

    def apply(self, probabilities: Any) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        if self.x.size == 0:
            return p
        if self.x.size == 1:
            return np.full(p.shape, float(self.y[0]))
        # Flat extrapolation outside the fitted range: an unseen probability
        # gets the nearest fitted rate rather than an invented one.
        return np.interp(p, self.x, self.y, left=self.y[0], right=self.y[-1])

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x": [round(float(v), 8) for v in self.x],
            "y": [round(float(v), 8) for v in self.y],
        }


def calibrator_from_dict(data: dict[str, Any] | None) -> Calibrator:
    if not data:
        return IdentityCalibrator()
    kind = data.get("kind", "identity")
    if kind == "platt":
        return PlattCalibrator(a=float(data.get("a", 1.0)), b=float(data.get("b", 0.0)))
    if kind == "isotonic":
        return IsotonicCalibrator(x=data.get("x", ()), y=data.get("y", ()))
    return IdentityCalibrator()


def _preserves_ordering(calibrator: Calibrator, probabilities: np.ndarray) -> bool:
    """Would this calibrator still let the head rank anything?

    Two ways a fitted calibrator fails this, and both are observed in
    practice:

    * **A negative slope.** A head with no real signal has a likelihood that
      is flat in the slope, so the fit picks up whatever sign the noise
      suggests. The resulting map is better calibrated and ranks backwards.
      The metrics cannot see this — ECE improves — so the check has to.
    * **Saturation.** A Platt fit with a large slope pushes the sigmoid to
      exactly 0 and 1 in float64 and every candidate ends up tied, which a
      constant forecast at the base rate also makes look well calibrated.

    Calibration is allowed to change the scale of a head and nothing else. If
    a fit wants to change the order, the head's problem was not calibration.
    """
    distinct = np.unique(probabilities)
    if distinct.size < 2:
        return True
    mapped = calibrator.apply(distinct)
    if not np.all(np.isfinite(mapped)):
        return False
    if np.unique(mapped).size < 2:
        return False
    return bool(np.all(np.diff(mapped) >= -1e-12))


def fit_calibrator(
    labels: Any,
    probabilities: Any,
    kind: str = "platt",
    min_positives: int = 10,
    bins: int = DEFAULT_BINS,
    min_improvement: float = 0.1,
) -> Calibrator:
    """Fit a calibrator, or decline to.

    Declining matters three times over.

    A head with three positives in the validation set can still be *fitted* —
    the arithmetic does not complain — and the result is a confident
    correction derived from three coin flips, applied to every request.

    A fit can succeed numerically and still break the head, by saturating or
    by inverting its order.

    And a fit can be simply pointless. A two-parameter family will nearly
    always shave *something* off the in-sample ECE, so "did it improve" is not
    a real question; ``min_improvement`` asks the one that is, which is
    whether it improved enough to be worth another moving part between the
    model and the request path. A head that is already calibrated should come
    back as the identity, and say so in the report.
    """
    y, p = _as_arrays(labels, probabilities)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives < min_positives or negatives < min_positives:
        return IdentityCalibrator()

    if kind == "identity":
        return IdentityCalibrator()

    def build(labels: np.ndarray, probabilities: np.ndarray) -> Calibrator:
        if kind == "isotonic" and float(labels.sum()) >= 10 * min_positives:
            # Isotonic has as many parameters as it likes; it needs an order
            # of magnitude more outcomes than Platt before it is fitting
            # signal rather than the validation set's noise.
            return IsotonicCalibrator().fit(labels, probabilities)
        return PlattCalibrator().fit(labels, probabilities)

    # Is the correction real, or is it fitting the bins' sampling noise?
    #
    # Measured in-sample this question has no teeth: ECE on a *perfectly*
    # calibrated head is not zero, it is however much noise 10 bins of finite
    # data produce, and a two-parameter family can absorb a good share of
    # that. An early version of this used an in-sample threshold and happily
    # installed a calibrator on a head that needed none. So the decision is
    # made on held-out data, the same discipline the rest of the project
    # applies to every other claim.
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(y.size)
    cut = int(y.size * 0.7)
    inner, outer = shuffled[:cut], shuffled[cut:]
    if (
        outer.size
        and float(y[inner].sum()) >= min_positives
        and float(y[outer].sum()) >= min_positives
    ):
        probe = build(y[inner], p[inner])
        before = calibration_report(y[outer], p[outer], bins=bins)
        after = calibration_report(y[outer], probe.apply(p[outer]), bins=bins)
        if before is None or after is None or before.ece <= 0:
            return IdentityCalibrator()
        if after.ece > before.ece * (1.0 - min_improvement):
            return IdentityCalibrator()

    # It generalises, so refit on everything and keep that.
    candidate = build(y, p)
    if not _preserves_ordering(candidate, p):
        return IdentityCalibrator()
    return candidate


@dataclass
class HeadCalibrators:
    """The fitted calibrator for each head, as carried in a bundle.

    This is fitted state, not configuration: it is derived from data the same
    way the weights are, so it travels with the weights in the manifest rather
    than sitting in a YAML file someone can edit into disagreement with the
    model it corrects.
    """

    calibrators: dict[str, Calibrator] = field(default_factory=dict)
    fitted_on: dict[str, Any] = field(default_factory=dict)

    def apply(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: (
                self.calibrators[name].apply(values)
                if name in self.calibrators
                else np.asarray(values, dtype=np.float64)
            )
            for name, values in scores.items()
        }

    @property
    def is_identity(self) -> bool:
        return all(c.kind == "identity" for c in self.calibrators.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibrators": {name: c.as_dict() for name, c in self.calibrators.items()},
            "fitted_on": self.fitted_on,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> HeadCalibrators:
        data = dict(data or {})
        return cls(
            calibrators={
                name: calibrator_from_dict(values)
                for name, values in (data.get("calibrators") or {}).items()
            },
            fitted_on=dict(data.get("fitted_on") or {}),
        )
