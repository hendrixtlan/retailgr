"""Where each model actually converges, and what stopping at eight epochs cost.

Every comparison this repository reported was taken after exactly eight
epochs, because `fit(train, val)` accepted a validation split and read none
of it. This measures the thing that was assumed: for each model and seed,
train once with early stopping on the validation split and once with the old
fixed schedule, and score **both on test**.

The two runs share a seed, so up to epoch eight they are the same run — the
difference is purely what happened after, and that difference is the cost of
the arbitrary cut-off. It is reported per model because the comparison
between models is what was at risk: if one model converges at epoch 9 and
the other at epoch 40, eight epochs flattered the fast one and the reported
tie was a statement about learning speed rather than capacity.

The validation curve is recorded per epoch. It is not an unbiased estimate of
anything once it has been used for selection — the best point on it is the
maximum of N noisy looks — so the report quotes test for every comparison
and shows the curve only for its shape.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from retailgr.config import Config, load_model_config
from retailgr.evaluation.evaluate import evaluate
from retailgr.evaluation.stats import paired_comparison
from retailgr.io.loaders import load_variant

__all__ = ["render_convergence", "run_convergence"]


def _fit_and_test(
    data: Any,
    model_cfg: dict[str, Any],
    exclude_seen: bool,
    k_values: list[int],
) -> dict[str, Any]:
    from retailgr.experiment import build_model

    _, model = build_model(data.vocab_size, model_cfg, exclude_seen=exclude_seen)
    started = time.time()
    fit = model.fit(data.train, data.val)
    seconds = round(time.time() - started, 1)
    scored = evaluate(
        model,
        data.test,
        vocab_size=data.vocab_size,
        k_values=k_values,
        exclude_seen=exclude_seen,
        with_intervals=False,
    )
    overall = scored["overall"]
    return {
        "fit_seconds": seconds,
        "epochs": fit.get("epochs"),
        "best_epoch": fit.get("best_epoch"),
        "stopped_because": fit.get("stopped_because"),
        "val_curve": fit.get("val_curve", []),
        "train_loss_curve": fit.get("train_loss_curve", []),
        "test": {key: overall[key] for key in overall if "@" in key or key == "eval_users"},
        # Kept for the paired comparison between the two schedules.
        "per_user": scored.get("per_user", {}),
        "user_ids": scored.get("user_ids", []),
    }


def _pooled_per_user(runs: list[dict[str, Any]], schedule: str, metric: str) -> np.ndarray:
    """One value per test user: the metric averaged over seeds.

    Asserts the users line up across runs. Every run scores the same test
    split in the same order, and if that ever stopped being true the pairing
    would silently compare different customers.
    """
    reference = runs[0][schedule]["user_ids"]
    stacked = []
    for run in runs:
        if run[schedule]["user_ids"] != reference:
            raise ValueError("test users are not aligned across runs; pairing is invalid")
        stacked.append(np.asarray(run[schedule]["per_user"][metric], dtype=np.float64))
    return np.mean(np.vstack(stacked), axis=0)


def run_convergence(
    cfg: Config,
    variant: str = "config",
    model_configs: tuple[str, ...] = ("sasrec_small.yaml", "hstu_small.yaml"),
    seeds: tuple[int, ...] = (13, 17, 23),
    patience: int = 10,
    max_epochs: int = 150,
    fixed_epochs: int = 8,
    metric: str = "ndcg@10",
) -> dict[str, Any]:
    """Fixed eight epochs against early-stopped, per model and seed, on test."""
    data = load_variant(cfg, variant)
    exclude_seen = bool(cfg.get("evaluation.exclude_seen", True))
    k_values = sorted({10, int(metric.partition("@")[2])})

    models: dict[str, Any] = {}
    for path in model_configs:
        base = dict(load_model_config(path))
        name = str(base.get("name") or path)
        runs = []
        for seed in seeds:
            fixed_cfg = {**base, "seed": seed, "epochs": fixed_epochs}
            fixed_cfg.pop("early_stopping", None)
            stopped_cfg = {
                **base,
                "seed": seed,
                "early_stopping": {
                    "enabled": True,
                    "metric": metric,
                    "patience": patience,
                    "max_epochs": max_epochs,
                },
            }
            fixed = _fit_and_test(data, fixed_cfg, exclude_seen, k_values)
            stopped = _fit_and_test(data, stopped_cfg, exclude_seen, k_values)
            runs.append({"seed": seed, "fixed": fixed, "stopped": stopped})

        # Paired, per user, pooled over seeds: each user's metric averaged
        # across seeds for each schedule, then compared. Pooling first takes
        # the seed noise out of the per-user values before the pairing takes
        # the between-user noise out of the difference.
        pooled = {
            schedule: _pooled_per_user(runs, schedule, metric)
            for schedule in ("fixed", "stopped")
        }
        schedule_test = paired_comparison(pooled["stopped"], pooled["fixed"])

        fixed_values = [r["fixed"]["test"][metric] for r in runs]
        stopped_values = [r["stopped"]["test"][metric] for r in runs]
        best_epochs = [r["stopped"]["best_epoch"] for r in runs]
        models[name] = {
            "config": path,
            "runs": runs,
            "fixed_mean": float(np.mean(fixed_values)),
            "fixed_std": float(np.std(fixed_values, ddof=1)) if len(runs) > 1 else 0.0,
            "stopped_mean": float(np.mean(stopped_values)),
            "stopped_std": float(np.std(stopped_values, ddof=1)) if len(runs) > 1 else 0.0,
            "best_epochs": best_epochs,
            "gain": float(np.mean(stopped_values) - np.mean(fixed_values)),
            "relative_gain": (
                float(np.mean(stopped_values) / np.mean(fixed_values) - 1.0)
                if np.mean(fixed_values)
                else float("nan")
            ),
            "stopped_vs_fixed": schedule_test.as_dict() if schedule_test else None,
            "_pooled": pooled,
        }
        # The per-user arrays served the pairing and are dropped here rather
        # than left for the scrub: a run record has no use for them.
        for run in runs:
            for schedule in ("fixed", "stopped"):
                run[schedule].pop("per_user", None)
                run[schedule].pop("user_ids", None)

    # The comparison item 1 existed to put at risk: does the model-vs-model
    # verdict change once each model is trained to its own best epoch?
    names = list(models)
    between: dict[str, Any] = {}
    if len(names) == 2:
        a, b = names
        for schedule in ("fixed", "stopped"):
            comparison = paired_comparison(
                models[a]["_pooled"][schedule], models[b]["_pooled"][schedule]
            )
            between[schedule] = {
                "a": a,
                "b": b,
                **(comparison.as_dict() if comparison else {}),
            }
    for entry in models.values():
        entry.pop("_pooled", None)

    return {
        "variant": variant,
        "metric": metric,
        "seeds": list(seeds),
        "between_models": between,
        "patience": patience,
        "max_epochs": max_epochs,
        "fixed_epochs": fixed_epochs,
        "models": models,
    }


def render_convergence(result: dict[str, Any]) -> str:
    metric = result["metric"]
    lines = [
        "# Where the models converge",
        "",
        f"- Variant: `{result['variant']}`, seeds: {result['seeds']}",
        f"- Early stopping on validation `{metric}`, patience {result['patience']}, "
        f"cap {result['max_epochs']} epochs",
        "- Both schedules scored on **test**; validation only chose the epoch.",
        "",
        f"| Model | Best epoch per seed | {result['fixed_epochs']} epochs | Converged | Gain |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, entry in result["models"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    ", ".join(str(e) for e in entry["best_epochs"]),
                    f"{entry['fixed_mean']:.4f} ±{entry['fixed_std']:.4f}",
                    f"{entry['stopped_mean']:.4f} ±{entry['stopped_std']:.4f}",
                    f"{entry['relative_gain']:+.1%}",
                ]
            )
            + " |"
        )
    lines += ["", "## Converged against eight epochs, paired per user", ""]
    for name, entry in result["models"].items():
        test = entry.get("stopped_vs_fixed") or {}
        if not test:
            continue
        lines.append(
            f"- `{name}`: {test['mean_difference']:+.4f} "
            f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}], p={test['p_value']:.3f}"
        )

    between = result.get("between_models") or {}
    if between:
        lines += ["", "## The model comparison, at both schedules", ""]
        for schedule, test in between.items():
            if "mean_difference" not in test:
                continue
            lines.append(
                f"- {schedule}: `{test['a']}` − `{test['b']}` = "
                f"{test['mean_difference']:+.4f} "
                f"[{test['ci_low']:+.4f}, {test['ci_high']:+.4f}], p={test['p_value']:.3f}"
            )

    lines += [
        "",
        "The validation curve is not quoted as a result: once it chooses the "
        "epoch, its best point is the maximum of many noisy looks and is "
        "biased upwards. Test is touched once per run, after the choice.",
    ]
    return "\n".join(lines) + "\n"
