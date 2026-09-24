"""Does training on the serving distribution make the ranker order the slate?

The diagnosis this answers, in three measurements:

**1. The teacher-forced number was never about ranking.** The heads scored
0.81 resolution / AUC 0.986 on history positions, and a five-number lookup
table on *how many times the same item just repeated* scores AUC 0.965 — for
the purchase head the table wins outright, 0.9935 against 0.9909. That number
was measuring the synthetic generator's view -> cart -> purchase funnel, with
no user modelling involved.

**2. There is real signal on the serving slate.** Retrieval's own ordering of
its own candidates reaches AUC 0.714 there. So the slate is not unorderable,
and the ranker's 0.53 is not a dataset ceiling.

**3. The training positive is the wrong kind of item.** 52.5% of positives
are items already in the prefix, and by action it is stark: 100% of
`add_to_cart`, `purchase` and `return` positives are repeats, because a
customer cannot buy something they never looked at. Serving runs with
`exclude_seen`, which removes every already-seen item *before* the ranker is
consulted. Those three heads were trained entirely on a class of item that
never reaches them, which is why the purchase head orders the slate below
chance: the feature it learned is anti-correlated with what survives the
filter.

The arms below are separate, not stacked, so each effect is attributable:

    control      the model as it was
    unseen       positives restricted to items not already in the prefix
    no_history   the teacher-forced loss switched off
    both         unseen positives and no teacher-forced loss

`retrieval` is not an arm, it is the line to clear: a second stage that
orders worse than the first stage's own output has no reason to exist, which
is the same thing `ranker_gate` says with a different metric.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from retailgr.config import Config, load_model_config
from retailgr.evaluation.calibration import calibration_report
from retailgr.evaluation.ranking import HEADS, collect_slate_probabilities
from retailgr.io.loaders import load_variant

__all__ = ["SLATE_VARIANTS", "render_slate_experiment", "run_slate_experiment", "slate_auc"]

SLATE_VARIANTS: dict[str, dict[str, Any]] = {
    "control": {"candidate_positive": "next", "history_loss_weight": 1.0},
    "unseen": {"candidate_positive": "unseen", "history_loss_weight": 1.0},
    "no_history": {"candidate_positive": "next", "history_loss_weight": 0.0},
    "both": {"candidate_positive": "unseen", "history_loss_weight": 0.0},
    # The arm the first three produced. `no_history` wins on the slate and
    # takes the return head to chance with it (AUC 0.873 -> 0.511), a loss
    # the slate cannot see because the return head has no slate-shaped
    # label. Keeping the teacher-forced loss for that head alone is the only
    # arm here that was designed from a measurement rather than a guess.
    # A scalar sweep, holding composition fixed. The `control` vs
    # `return_only` comparison changed magnitude *and* which heads carry it,
    # so "less teacher-forced loss is better" cannot be read off it. These
    # scale all four heads together, which is the only way to separate the
    # two.
    "scale_50": {"candidate_positive": "next", "history_loss_weight": 0.5},
    "scale_25": {"candidate_positive": "next", "history_loss_weight": 0.25},
    "scale_10": {"candidate_positive": "next", "history_loss_weight": 0.1},
    "return_only": {
        "candidate_positive": "next",
        "history_loss_weight": {
            "click": 0.0, "cart": 0.0, "purchase": 0.0, "return": 1.0
        },
    },
}

# Heads that have a slate-shaped label. `return` needs a matured purchase,
# and a slate yields at most one of those per user, so it is not measurable
# here and is reported as absent rather than as zero.
SLATE_HEADS: tuple[str, ...] = ("click", "cart", "purchase")


def slate_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank AUC with ties averaged.

    AUC and not resolution share, because resolution is quadratic in
    (observed - base) and the slate base rate is ~0.2% against the
    teacher-forced set's 24%. The same ordering scores about 144x lower on
    the slate purely from that, which is how a real 50x difference in
    ordering ability got reported as a 7000x collapse.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(labels.size, dtype=float)
    ordered = scores[order]
    start = 0
    while start < labels.size:
        end = start
        while end + 1 < labels.size and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    positive = labels > 0.5
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _retrieval_slate_auc(labels: np.ndarray, candidate_k: int) -> float:
    """Retrieval's own ordering, scored on the same rows.

    The slate arrives in retrieval's ranked order, `candidate_k` rows per
    user, so position within the block *is* retrieval's score. No second
    model run is needed and none is possible — the collector does not keep
    the scores.
    """
    if labels.size == 0 or labels.size % candidate_k:
        return float("nan")
    position = np.tile(np.arange(candidate_k), labels.size // candidate_k)
    return slate_auc(labels, -position.astype(float))


def _fit_and_score(
    data: Any,
    retrieval: Any,
    ranker_cfg: dict[str, Any],
    pool: np.ndarray,
    candidate_k: int,
    max_users: int,
) -> dict[str, Any]:
    from retailgr.models.ranker import HSTURanker

    ranker = HSTURanker(data.vocab_size, ranker_cfg)
    started = time.time()
    fit_stats = ranker.fit(data.train, data.val, hard_negatives=pool)
    seconds = round(time.time() - started, 1)

    slate = collect_slate_probabilities(
        ranker,
        retrieval,
        data.test,
        vocab_size=data.vocab_size,
        candidate_k=candidate_k,
        max_users=max_users,
    )
    heads: dict[str, Any] = {}
    for head in HEADS:
        block = slate.get(head)
        if not block or block["labels"].size == 0 or block["labels"].sum() == 0:
            continue
        labels, probabilities = block["labels"], block["probabilities"]
        report = calibration_report(labels, probabilities, bins=10)
        heads[head] = {
            "n": int(labels.size),
            "positives": int(labels.sum()),
            "base_rate": round(float(labels.mean()), 6),
            "auc": round(slate_auc(labels, probabilities), 4),
            "resolution_share": round(report.resolution_share if report else 0.0, 6),
            # The gate has two criteria and this arm changes the loss that
            # the code comments credit with calibrating the heads. An arm
            # that orders better and predicts on the wrong scale still fails,
            # just on the other criterion, so both are recorded.
            "bias_ratio": round(report.bias_ratio if report else float("nan"), 4),
            "ece_relative": round(report.ece_relative if report else float("nan"), 4),
            "retrieval_auc": round(_retrieval_slate_auc(labels, candidate_k), 4),
        }
    # The return head has no slate-shaped label, so it is measured on the
    # only distribution where it exists. Without this an arm can look like a
    # clear win and be quietly trading a head away.
    from retailgr.evaluation.ranking import collect_teacher_forced_probabilities

    teacher_forced = collect_teacher_forced_probabilities(ranker, data.test, max_users=1000)
    block = teacher_forced.get("return") or {}
    labels = block.get("labels")
    return_auc = float("nan")
    if labels is not None and labels.size and 0 < labels.sum() < labels.size:
        return_auc = round(slate_auc(labels, block["probabilities"]), 4)

    return {
        "fit_seconds": seconds,
        "cut_fallback_share": fit_stats.get("cut_fallback_share", 0.0),
        "final_epoch_loss": fit_stats.get("final_epoch_loss"),
        "return_head_auc": return_auc,
        "heads": heads,
    }


def run_slate_experiment(
    cfg: Config,
    variant: str = "config",
    retrieval_config: str = "hstu_small.yaml",
    ranker_config: str = "ranker_small.yaml",
    variants: dict[str, dict[str, Any]] | None = None,
    seeds: tuple[int, ...] = (13, 17, 23),
    max_users: int = 400,
) -> dict[str, Any]:
    """Fit each arm at each seed and report slate AUC against retrieval's.

    One retrieval model and one candidate pool for every arm, so a difference
    is attributable to the training change and to nothing else. Several
    seeds because this model's seed spread is not small and a single run per
    arm would let any story be read off the table.
    """
    variants = variants or SLATE_VARIANTS
    data = load_variant(cfg, variant)

    from retailgr.spark_session import stop_spark

    stop_spark()

    from retailgr.experiment import build_model
    from retailgr.models.ranker import retrieval_negative_pool

    _, retrieval = build_model(data.vocab_size, load_model_config(retrieval_config))
    started = time.time()
    retrieval.fit(data.train, data.val)
    retrieval_seconds = round(time.time() - started, 1)

    base_cfg = load_model_config(ranker_config)
    candidate_k = int(cfg.get("serving.rank_k", 300))
    pool = retrieval_negative_pool(
        retrieval,
        data.train,
        vocab_size=data.vocab_size,
        pool_size=candidate_k,
        max_events=int(base_cfg.get("max_events", 50)),
    )

    entries: dict[str, Any] = {}
    for name, overrides in variants.items():
        runs = []
        for seed in seeds:
            ranker_cfg = {**base_cfg, **overrides, "seed": seed}
            runs.append(
                _fit_and_score(data, retrieval, ranker_cfg, pool, candidate_k, max_users)
            )
        summary: dict[str, Any] = {"overrides": overrides, "runs": runs, "heads": {}}
        for head in SLATE_HEADS:
            values = [r["heads"][head]["auc"] for r in runs if head in r["heads"]]
            if not values:
                continue
            biases = [r["heads"][head]["bias_ratio"] for r in runs if head in r["heads"]]
            summary["heads"][head] = {
                "auc_mean": round(float(np.mean(values)), 4),
                "auc_std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
                "values": values,
                "bias_ratio_mean": round(float(np.mean(biases)), 4),
                "retrieval_auc": runs[0]["heads"][head]["retrieval_auc"],
            }
        summary["cut_fallback_share"] = runs[0]["cut_fallback_share"]
        return_aucs = [r["return_head_auc"] for r in runs if not np.isnan(r["return_head_auc"])]
        summary["return_head_auc"] = (
            round(float(np.mean(return_aucs)), 4) if return_aucs else float("nan")
        )
        entries[name] = summary

    control = entries.get("control", {}).get("heads", {})
    for entry in entries.values():
        for head, block in entry["heads"].items():
            reference = control.get(head, {}).get("auc_mean")
            if reference is None:
                continue
            # Lift over chance, not the raw ratio: an AUC of 0.55 against
            # 0.53 is a 4% ratio and a 54% improvement in the only part that
            # carries information.
            block["over_chance"] = round(block["auc_mean"] - 0.5, 4)
            block["control_over_chance"] = round(reference - 0.5, 4)
            block["vs_control"] = (
                round((block["auc_mean"] - 0.5) / (reference - 0.5), 3)
                if abs(reference - 0.5) > 1e-9
                else float("nan")
            )
            spread = control.get(head, {}).get("auc_std", 0.0)
            block["beyond_seed_noise"] = bool(
                spread > 0 and abs(block["auc_mean"] - reference) > 2 * spread
            )

    return {
        "variant": variant,
        "candidate_k": candidate_k,
        "seeds": list(seeds),
        "max_users": max_users,
        "retrieval_config": retrieval_config,
        "retrieval_seconds": retrieval_seconds,
        "entries": entries,
    }


def render_slate_experiment(result: dict[str, Any]) -> str:
    """Markdown, with retrieval's own number on every row."""
    lines = [
        "# Ordering the serving slate",
        "",
        f"- Variant: `{result['variant']}`, candidate_k: {result['candidate_k']}",
        f"- Seeds per arm: {result['seeds']}, users: {result['max_users']}",
        "",
        "AUC, not resolution share. Resolution is quadratic in the deviation "
        "from the base rate, and the slate's base rate is ~0.2% against the "
        "teacher-forced set's 24% — the same ordering scores ~144x lower on "
        "the slate from that alone.",
        "",
    ]
    for head in SLATE_HEADS:
        rows = [
            (name, entry["heads"][head])
            for name, entry in result["entries"].items()
            if head in entry["heads"]
        ]
        if not rows:
            continue
        retrieval = rows[0][1]["retrieval_auc"]
        lines += [
            f"## {head}",
            "",
            f"Retrieval's own ordering of the same candidates: **AUC {retrieval}**. "
            "A second stage below this line has no reason to exist.",
            "",
            "| Arm | AUC | Seed spread | Over chance | vs control | Beyond noise | Scale |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
        for name, block in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{name}`",
                        f"{block['auc_mean']:.4f}",
                        f"±{block['auc_std']:.4f}",
                        f"{block.get('over_chance', 0):+.4f}",
                        f"{block.get('vs_control', float('nan')):.2f}x",
                        "yes" if block.get("beyond_seed_noise") else "no",
                        f"{block.get('bias_ratio_mean', float('nan')):.2f}x",
                    ]
                )
                + " |"
            )
        lines.append("")

    lines += [
        "## return",
        "",
        "No slate-shaped label exists for this head — a slate yields at most one",
        "matured purchase per user — so it is measured on history positions, the",
        "only distribution where it exists. An arm that wins above and loses here",
        "is trading a head away where the main table cannot see it.",
        "",
        "| Arm | Return head AUC |",
        "| --- | ---: |",
    ]
    for name, entry in result["entries"].items():
        lines.append(f"| `{name}` | {entry.get('return_head_auc', float('nan')):.4f} |")
    lines.append("")

    fallback = result["entries"].get("unseen", {}).get("cut_fallback_share")
    if fallback is not None:
        lines += [
            f"`unseen` fell back to the last event for **{fallback:.1%}** of users, "
            "who have no position whose item is new. That is supervision the arm "
            "gives up, and it is the cost of matching the serving distribution.",
            "",
        ]
    return "\n".join(lines) + "\n"
