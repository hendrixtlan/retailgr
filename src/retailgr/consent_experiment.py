"""What consent costs the model, measured rather than assumed.

Every write-up of consent in a recommender stops at "we honour the flag".
Nobody publishes the second half, which is the half a product owner will
actually ask about: *how much worse do the recommendations get*. Without a
number, the conversation is a negotiation between someone who says privacy
is free and someone who says it is ruinous, and neither has evidence.

**The measurement.** Consent in this system removes whole users, not
individual events — one decision covers an account, and a user who opts out
of analytics contributes nothing at all. So the cost of an opt-in rate `r`
is the cost of training on a random `r` share of users, which this module
measures directly by subsampling the training split and re-fitting. The
evaluation split is held constant across every point: the question is "how
good is the model we can build", not "how good is it on the users who stayed",
and scoring each model on its own surviving population would answer the
second while looking like the first.

**The assumption, and which way it is wrong.** Subsampling uniformly assumes
consent is independent of behaviour. It is not. Privacy-conscious customers
differ from the average in ways that are hard to characterise and easy to
get backwards, so the uniform number is a **lower bound on the damage** — it
measures the cost of losing *some* users, not the cost of losing *these*
users. `--correlate` makes that concrete instead of leaving it as a caveat:
it opts users out in order of activity rather than at random, and the gap
between the two curves is the part of the cost that comes from *who* opted
out rather than *how many*.

That gap is the actual finding here. If it is small, the uniform estimate is
usable for planning. If it is large, then any consent forecast that does not
model who opts out is worthless, whatever its confidence interval says.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from retailgr.config import Config, load_model_config
from retailgr.evaluation.evaluate import evaluate
from retailgr.io.loaders import SequenceSplit, VariantData, load_variant

__all__ = ["DEFAULT_RATES", "render_consent_cost", "run_consent_cost", "subsample_users"]

# Opt-in rates to sweep. 1.0 is the control and is always included: a curve
# without its own baseline measures nothing, because the seed-to-seed spread
# of this model is large enough to swallow a small effect.
DEFAULT_RATES: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25)


def subsample_users(
    split: SequenceSplit,
    rate: float,
    *,
    seed: int,
    correlate: bool = False,
) -> SequenceSplit:
    """A split containing ``rate`` of the users, dropping the rest entirely.

    Whole users, never partial histories. Truncating a history would model
    something consent does not do and would flatter the result: half a
    sequence is still a training example, while an opted-out customer is
    not one at all.

    With ``correlate``, the users dropped are the *most active* rather than a
    random sample — a deliberately pessimistic stand-in for the fact that
    opting out is not independent of behaviour. The direction is a guess;
    the point is to show whether the answer depends on that guess, not to
    claim heavy shoppers are the ones who opt out.
    """
    count = len(split)
    if count == 0 or rate >= 1.0:
        return split

    keep_n = max(1, int(round(count * rate)))
    if correlate:
        # Longest histories leave first.
        activity = np.array([len(tokens) for tokens in split.inputs])
        # Stable tie-break, so this is reproducible for a given split.
        order = np.lexsort((np.arange(count), activity))
        keep = np.sort(order[:keep_n])
    else:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(count, size=keep_n, replace=False))

    out = SequenceSplit()
    for index in keep:
        out.user_ids.append(split.user_ids[index])
        out.inputs.append(split.inputs[index])
        out.targets.append(split.targets[index])
        out.actions.append(split.actions[index])
        out.timestamps.append(split.timestamps[index])
        out.returned.append(split.returned[index])
        if split.target_actions:
            out.target_actions.append(split.target_actions[index])
    return out


def _fit_and_score(
    data: VariantData,
    model_config: str,
    train: SequenceSplit,
    k_values: list[int],
    *,
    seed: int,
    exclude_seen: bool,
    resamples: int,
) -> dict[str, Any]:
    from retailgr.experiment import build_model

    model_cfg = dict(load_model_config(model_config))
    model_cfg["seed"] = seed
    _, model = build_model(data.vocab_size, model_cfg)

    started = time.time()
    fit_stats = model.fit(train, data.val)
    # Scored on the untouched test split, always. Evaluating each model on
    # its own surviving users would compare different questions and would
    # make consent look almost free, because the users who remain are the
    # ones the model was given.
    scored = evaluate(
        model,
        data.test,
        vocab_size=data.vocab_size,
        k_values=k_values,
        exclude_seen=exclude_seen,
        category_by_id=data.category_by_id,
        resamples=resamples,
    )
    scored.pop("per_user", None)
    scored.pop("user_ids", None)
    return {
        "train_users": len(train),
        "fit": fit_stats,
        "fit_seconds": round(time.time() - started, 2),
        "test": scored["overall"],
        "intervals": scored.get("intervals", {}),
    }


def run_consent_cost(
    cfg: Config,
    variant: str = "config",
    model_config: str = "hstu_small.yaml",
    rates: tuple[float, ...] = DEFAULT_RATES,
    seeds: tuple[int, ...] = (0, 1, 2),
    k_values: list[int] | None = None,
    correlate: bool = False,
) -> dict[str, Any]:
    """Fit at each opt-in rate and report the metric curve.

    Several seeds per rate, because this model's seed-to-seed spread is not
    small: a single run per point would produce a curve that mostly plots
    initialisation noise and would let any story be read off it.
    """
    k_values = k_values or [10]
    data = load_variant(cfg, variant)
    exclude_seen = bool(cfg.get("serving.exclude_seen", True))
    resamples = int(cfg.get("evaluation.bootstrap_resamples", 2000))

    points: list[dict[str, Any]] = []
    for rate in rates:
        runs: list[dict[str, Any]] = []
        for seed in seeds:
            train = subsample_users(
                data.train, rate, seed=seed * 1000 + int(rate * 100), correlate=correlate
            )
            runs.append(
                _fit_and_score(
                    data,
                    model_config,
                    train,
                    k_values,
                    seed=seed,
                    exclude_seen=exclude_seen,
                    resamples=resamples,
                )
            )
        metric_key = f"recall@{k_values[0]}"
        values = [float(run["test"][metric_key]) for run in runs]
        points.append(
            {
                "rate": rate,
                "train_users": runs[0]["train_users"],
                "metric": metric_key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
                "runs": runs,
            }
        )

    baseline = next((p for p in points if p["rate"] == 1.0), None)
    if baseline:
        for point in points:
            point["relative"] = (
                point["mean"] / baseline["mean"] if baseline["mean"] else float("nan")
            )
            # Is the drop larger than the noise this model has anyway? The
            # seed spread at full data is the only honest yardstick.
            spread = baseline["std"]
            point["beyond_seed_noise"] = bool(
                spread > 0 and (baseline["mean"] - point["mean"]) > 2 * spread
            )

    return {
        "variant": variant,
        "model_config": model_config,
        "correlate": correlate,
        "seeds": list(seeds),
        "k_values": k_values,
        "total_users": len(data.train),
        "points": points,
    }


def render_consent_cost(result: dict[str, Any]) -> str:
    """Markdown, with the caveat attached to the number rather than a footnote."""
    lines = [
        "# What consent costs",
        "",
        f"- Variant: `{result['variant']}`, model: `{result['model_config']}`",
        f"- Seeds per point: {result['seeds']}",
        f"- Training users at 100%: {result['total_users']}",
        f"- Opt-out selection: **{'most active first' if result['correlate'] else 'uniform'}**",
        "",
        "| Opt-in rate | Train users | Recall@10 | Seed spread | vs. 100% | Beyond noise |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for point in result["points"]:
        relative = point.get("relative")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{point['rate']:.0%}",
                    str(point["train_users"]),
                    f"{point['mean']:.4f}",
                    f"±{point['std']:.4f}",
                    "-" if relative is None else f"{relative:.3f}x",
                    "-" if relative is None else ("yes" if point["beyond_seed_noise"] else "no"),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "**Read the last column first.** This model's seed-to-seed spread is not",
        "small, and a difference inside it is not a difference. A row marked `no`",
        "means the measured drop at that opt-in rate is indistinguishable from",
        "re-running the same configuration with a different random seed.",
        "",
    ]
    if not result["correlate"]:
        lines += [
            "**This is a lower bound.** Users are dropped uniformly, which assumes",
            "consent is independent of behaviour. It is not: privacy-conscious",
            "customers differ from the average in ways that are hard to",
            "characterise and easy to get backwards. Run with `--correlate` to see",
            "how much of the answer depends on *who* opts out rather than how many.",
        ]
    else:
        lines += [
            "**The pessimistic direction.** Users are dropped most-active-first,",
            "which is a guess about who opts out, not a finding. Its value is the",
            "comparison with the uniform run: the gap between the two curves is the",
            "share of the cost that comes from *who* leaves rather than *how many*.",
        ]
    return "\n".join(lines) + "\n"
