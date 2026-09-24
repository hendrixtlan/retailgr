"""Can the ranker be given resolution on the distribution it serves?

Calibration produced one number that subsumed every other ranker finding in
this project: the heads explain 73–82% of the outcome variance on the
distribution they are *trained* on — "this item is in the customer's history,
what did they do with it?" — and 0.01–0.03% on the one they are *deployed* on
— "this item is in a slate we are about to show, will they act on it?". AUC
0.99 on the first question says nothing about the second, and the second is
the job.

The leading hypothesis is the negative sampler. The ranker learns its
candidate positions against ``num_negatives`` items drawn uniformly from the
vocabulary, almost none of which retrieval would ever have surfaced. So
"positive versus 256 uniform" is solvable from an item prior, and an item
prior is exactly what retrieval already provides. At serving the ranker is
handed retrieval's top 300, where every candidate is plausible and the prior
carries no information at all. It is trained on a question that has already
been answered and deployed on one it has never seen.

This module tests that hypothesis rather than assuming it. One retrieval
model is trained and shared, its own candidates become the pool the ranker's
negatives are drawn from, and several mixes are compared on the metric the
hypothesis is about — resolution on serving-shaped slates — with the paired
test and the multiple-comparison adjustment the rest of the project uses.

The result is reported whichever way it comes out. A negative result here is
worth more than another mechanism: it would rule out the cheap explanation
and point at the expensive one.
"""

from __future__ import annotations

import time
from typing import Any

from retailgr.config import Config, load_model_config
from retailgr.evaluation.ranking import (
    calibrate_heads,
    compare_against_retrieval,
    compare_rankers,
    evaluate_discrimination,
    evaluate_heads,
    evaluate_reranking,
    ranker_gate,
    retrieval_discrimination,
    retrieval_slate_resolution,
)
from retailgr.evaluation.stats import holm_bonferroni
from retailgr.experiment import build_model
from retailgr.io.loaders import load_variant
from retailgr.models.ranker import HSTURanker, retrieval_negative_pool

# The mixes to compare. ``uniform`` is the configuration every number in this
# repository so far was produced with, and it is the baseline every other row
# is tested against.
RANKER_VARIANTS: dict[str, dict[str, Any]] = {
    "uniform": {"hard_negative_fraction": 0.0},
    "half_hard": {"hard_negative_fraction": 0.5},
    "all_hard": {"hard_negative_fraction": 1.0},
}

BASELINE = "uniform"


def run_ranker_experiment(
    cfg: Config,
    variant: str = "config",
    retrieval_config: str = "hstu_small.yaml",
    ranker_config: str = "ranker_small.yaml",
    variants: dict[str, dict[str, Any]] | None = None,
    overrides: dict[str, Any] | None = None,
    calibration_users: int = 300,
    discrimination_users: int = 400,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Train one ranker per negative-sampling mix and compare them.

    Everything except the mix is held fixed — same retrieval model, same
    candidate pool, same seed, same data — so a difference is attributable to
    the sampler and nothing else.
    """
    variants = variants or RANKER_VARIANTS
    data = load_variant(cfg, variant)

    from retailgr.spark_session import stop_spark

    stop_spark()

    retrieval_cfg = load_model_config(retrieval_config)
    _, retrieval = build_model(data.vocab_size, retrieval_cfg)
    started = time.time()
    retrieval_fit = retrieval.fit(data.train, data.val)
    retrieval_seconds = round(time.time() - started, 2)

    base_cfg = {**load_model_config(ranker_config), **(overrides or {})}
    candidate_k = int(cfg.get("serving.rank_k", 300))

    # Built once from the shared retrieval model, so every variant draws from
    # exactly the same pool and the only thing that differs is how much of it
    # it uses.
    started = time.time()
    pool = retrieval_negative_pool(
        retrieval,
        data.train,
        vocab_size=data.vocab_size,
        pool_size=candidate_k,
        max_events=int(base_cfg.get("max_events", 50)),
    )
    pool_seconds = round(time.time() - started, 2)

    retrieval_scores = retrieval_discrimination(
        retrieval, data.test, vocab_size=data.vocab_size, max_users=discrimination_users
    )
    # The control. "The ranker explains 0.02% of the outcome variance on a
    # slate" only condemns the ranker if something else does better on the
    # same slate with the same labels — otherwise it condemns the task. The
    # obvious something else is the model that built the slate.
    retrieval_resolution = retrieval_slate_resolution(
        retrieval,
        data.test,
        vocab_size=data.vocab_size,
        candidate_k=candidate_k,
        max_users=calibration_users,
    )

    entries: dict[str, Any] = {}
    for name, override in variants.items():
        ranker_cfg = {**base_cfg, **override}
        ranker = HSTURanker(data.vocab_size, ranker_cfg)
        started = time.time()
        fit_stats = ranker.fit(data.train, data.val, hard_negatives=pool)
        fit_seconds = round(time.time() - started, 2)

        heads = evaluate_heads(ranker, data.test)
        # Calibration first: it is what makes the served blend the thing that
        # gets measured, and its `resolution` is the metric this experiment
        # is about.
        calibration = calibrate_heads(
            ranker,
            retrieval,
            fit_split=data.val,
            test_split=data.test,
            vocab_size=data.vocab_size,
            candidate_k=candidate_k,
            max_users=calibration_users,
        )
        ranker.calibrators = calibration.pop("calibrators")

        discrimination = evaluate_discrimination(
            ranker, data.test, vocab_size=data.vocab_size, max_users=discrimination_users
        )
        entries[name] = {
            "config": {k: ranker_cfg.get(k) for k in ("num_negatives", "epochs")}
            | dict(override),
            "fit": fit_stats,
            "fit_seconds": fit_seconds,
            "heads": heads,
            "calibration": calibration,
            "discrimination": discrimination,
            "reranking": evaluate_reranking(
                ranker,
                retrieval,
                data.test,
                vocab_size=data.vocab_size,
                candidate_k=candidate_k,
                metric_k=10,
                max_users=int(cfg.get("evaluation.rerank_users", 300)),
            ),
            "gate": ranker_gate(
                discrimination,
                retrieval_scores,
                min_relative_mrr=float(cfg.get("evaluation.ranker_gate_min_mrr", 1.0)),
                calibration=calibration,
                max_bias_ratio=float(cfg.get("evaluation.ranker_gate_max_bias", 4.0)),
            ),
            "vs_retrieval": (
                comparison.as_dict()
                if (
                    comparison := compare_against_retrieval(
                        discrimination, retrieval_scores, resamples=resamples
                    )
                )
                else None
            ),
        }

    # Every variant against the baseline, paired per user and adjusted as one
    # family — the same discipline the HSTU ablation uses, for the same
    # reason: a table of tests is a table of chances to be fooled.
    comparisons: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    if BASELINE in entries:
        for name, entry in entries.items():
            if name == BASELINE:
                continue
            result = compare_rankers(
                entry["discrimination"],
                entries[BASELINE]["discrimination"],
                confidence=confidence,
                resamples=resamples,
            )
            if result is None:
                continue
            comparisons[name] = result.as_dict() | {"verdict": result.verdict()}
            p_values[name] = result.p_value
    adjusted = holm_bonferroni(p_values) if p_values else {}

    return {
        "dataset": cfg.dataset_name,
        "variant": variant,
        "vocab_size": data.vocab_size,
        "candidate_k": candidate_k,
        "retrieval": {
            "config": retrieval_config,
            "fit": retrieval_fit,
            "fit_seconds": retrieval_seconds,
            "discrimination": retrieval_scores,
            "slate_resolution": retrieval_resolution,
        },
        "pool_seconds": pool_seconds,
        "baseline": BASELINE,
        "entries": entries,
        "comparisons": comparisons,
        "adjusted": adjusted,
        "confidence": confidence,
        "resamples": resamples,
    }


def _resolution(entry: dict[str, Any], head: str, where: str = "slate") -> float | None:
    report = ((entry.get("calibration") or {}).get("heads") or {}).get(head) or {}
    measured = report.get(where)
    return (measured or {}).get("resolution_share")


def render_ranker_experiment(result: dict[str, Any]) -> str:
    """Markdown report for the negative-sampling comparison."""
    entries = result.get("entries") or {}
    retrieval = (result.get("retrieval") or {}).get("discrimination") or {}
    baseline = result.get("baseline")

    lines = [
        "# Ranker: does the negative sampler explain the missing resolution?",
        "",
        f"- Dataset: `{result.get('dataset')}`, variant `{result.get('variant')}`, "
        f"vocabulary {result.get('vocab_size')}",
        f"- One shared retrieval model, its top {result.get('candidate_k')} used both "
        "as the negative pool and as the slate every number below is measured on",
        f"- Baseline: `{baseline}` — uniform negatives, the configuration every "
        "earlier number in this repository was produced with",
        "",
        "## The question",
        "",
        "The heads explain most of the outcome variance on the positions they are "
        "trained on and almost none on the slates they are deployed on. If the "
        "cause is that uniform negatives make the training task too easy, then "
        "drawing negatives from retrieval's own candidates should move "
        "**resolution on the slate** — and nothing else in this table is the "
        "point.",
        "",
        "## Resolution on the serving distribution",
        "",
    ]

    header = ["Variant", "click", "cart", "purchase", "click (history)", "purchase (history)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    # The control goes first, because without it none of the rows below mean
    # anything: a small number is only small relative to what is achievable.
    control = ((result.get("retrieval") or {}).get("slate_resolution") or {}).get("heads") or {}
    if control:

        def control_cell(head: str) -> str:
            value = (control.get(head) or {}).get("resolution_share")
            return f"**{value:.2%}**" if value is not None else "–"

        lines.append(
            "| "
            + " | ".join(
                [
                    "_retrieval, on its own slate_",
                    control_cell("click"),
                    control_cell("cart"),
                    control_cell("purchase"),
                    "–",
                    "–",
                ]
            )
            + " |"
        )
    for name, entry in entries.items():
        label = f"`{name}`" + (" _(baseline)_" if name == baseline else "")

        def cell(head: str, where: str = "slate", entry: dict[str, Any] = entry) -> str:
            value = _resolution(entry, head, where)
            return f"{value:.2%}" if value is not None else "–"

        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    cell("click"),
                    cell("cart"),
                    cell("purchase"),
                    cell("click", "teacher_forced"),
                    cell("purchase", "teacher_forced"),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Resolution is the share of the outcome's variance a head's predictions "
        "explain. Calibration cannot change it, so it is the one number here that "
        "measures what the model knows rather than how it expresses it. The two "
        "right-hand columns are the same heads on history positions, unchanged by "
        "this experiment and shown so the gap stays visible."
    )
    lines.append("")
    if control:
        lines.append(
            "**The first row is the control, and it is the one that makes the rest "
            "readable.** It is retrieval's own score, on the slate retrieval itself "
            "produced, against exactly the same per-head labels — the labelling code "
            "is shared between the two measurements precisely so this comparison "
            "cannot drift. Every number in this table is small in absolute terms, "
            "because most of the variance in *which one of 300 items a customer "
            "picks next* is irreducible. What matters is the ratio: if retrieval "
            "explains several times more of it than the model whose job is to "
            "reorder retrieval's output, the second stage is not adding knowledge, "
            "it is adding latency."
        )
        lines.append("")

    lines.append("## Ordering, against retrieval and against the baseline")
    lines.append("")
    header = [
        "Variant",
        "blended MRR",
        "vs retrieval",
        f"vs `{baseline}` (paired)",
        "p",
        "p adj",
        "Gate",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    lines.append(
        f"| _retrieval_ | {retrieval.get('mrr', float('nan')):.4f} | 1.00x | – | – | – | – |"
    )
    comparisons = result.get("comparisons") or {}
    adjusted = result.get("adjusted") or {}
    for name, entry in entries.items():
        signals = (entry.get("discrimination") or {}).get("signals") or {}
        mrr = (signals.get("blended") or {}).get("mrr")
        gate = entry.get("gate") or {}
        comparison = comparisons.get(name)
        stats = adjusted.get(name) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`" + (" _(baseline)_" if name == baseline else ""),
                    f"{mrr:.4f}" if mrr is not None else "–",
                    f"{gate.get('relative_mrr', float('nan')):.2f}x",
                    comparison["verdict"] if comparison else "–",
                    f"{comparison['p_value']:.3f}" if comparison else "–",
                    f"{stats['p_adjusted']:.3f}" if stats else "–",
                    "PASSED" if gate.get("passed") else "BLOCKED",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "The comparison against the baseline is paired per user and sign-flipped, "
        "and the whole column is Holm-adjusted as one family. Two rankers scoring "
        "the same users differ far less between themselves than the users differ "
        "from each other, so an unpaired reading of these MRRs would call every "
        "row a tie."
    )
    lines.append("")

    lines.append("## Re-ranking, the number that would actually ship")
    lines.append("")
    header = ["Variant", "retrieval NDCG@10", "re-ranked NDCG@10", "Δ"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for name, entry in entries.items():
        rerank = entry.get("reranking") or {}
        if not rerank.get("users"):
            continue
        delta = rerank.get("ndcg_delta_pct")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    f"{rerank.get('retrieval_ndcg@10', float('nan')):.4f}",
                    f"{rerank.get('reranked_ndcg@10', float('nan')):.4f}",
                    f"{delta:+.1f}%" if delta is not None else "–",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "One caveat this dataset cannot argue away: the vocabulary is "
        f"{result.get('vocab_size')} tokens and the slate is "
        f"{result.get('candidate_k')} of them, so a uniform negative already has "
        "roughly a one-in-three chance of being a candidate retrieval would have "
        "shown. Hard negatives have much less room to help here than they would "
        "on a real catalogue of millions, where a uniform draw is never a "
        "plausible item. A null result on this data is therefore weaker evidence "
        "against the hypothesis than a positive result would be for it."
    )
    return "\n".join(lines)
