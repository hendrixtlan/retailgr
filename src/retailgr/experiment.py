"""The deciding experiments, run on one harness.

Two questions, same pipeline and same time split:

1. Which token granularity wins, per category (all-SKU, all-style-colour,
   all-product, category config)?
2. Does HSTU beat SASRec, and does the popularity baseline stay beaten?

The output is one JSON run record plus a Markdown report.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retailgr import privacy
from retailgr.config import Config, load_model_config
from retailgr.evaluation.evaluate import evaluate
from retailgr.io.loaders import load_variant
from retailgr.jobs import ingest, sequences, silver
from retailgr.models.base import Recommender
from retailgr.models.hstu import HSTUModel
from retailgr.models.popularity import PopularityModel
from retailgr.models.sasrec import SASRecModel

DEFAULT_VARIANTS = ("sku", "style_color", "product", "config")
DEFAULT_MODEL_CONFIGS = ("sasrec_small.yaml", "hstu_small.yaml")

# A model config's `type` field selects the class.
MODEL_REGISTRY: dict[str, type[Recommender]] = {
    "sasrec": SASRecModel,
    "hstu": HSTUModel,
}


def build_model(
    vocab_size: int,
    model_cfg: dict[str, Any],
    exclude_seen: bool | None = None,
) -> tuple[str, Recommender]:
    """Instantiate the model a config asks for, with its reporting name.

    ``exclude_seen`` is how the pipeline's evaluation setting reaches early
    stopping. The validation metric must be computed the way the test metric
    will be — a model selected on recall with seen items included and then
    reported with them excluded was tuned for a different objective from the
    one it is judged on.
    """
    if exclude_seen is not None and (model_cfg.get("early_stopping") or {}):
        model_cfg = dict(model_cfg)
        model_cfg["early_stopping"] = {
            **model_cfg["early_stopping"],
            "exclude_seen": bool(exclude_seen),
        }
    kind = str(model_cfg.get("type") or "").lower()
    if kind not in MODEL_REGISTRY:
        raise ValueError(
            f"model config has type {kind!r}; expected one of {sorted(MODEL_REGISTRY)}"
        )
    name = str(model_cfg.get("name") or kind)
    return name, MODEL_REGISTRY[kind](vocab_size, model_cfg)


@dataclass
class RunRecord:
    started_at: str
    dataset: str
    warehouse_backend: str
    model_configs: list[str] = field(default_factory=list)
    pipeline_stats: dict[str, Any] = field(default_factory=dict)
    variants: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0


def build_pipeline(cfg: Config, skip_if_present: bool = False) -> dict[str, Any]:
    """Bronze and silver: run once, reused by every variant."""
    from retailgr.io.tables import table_exists
    from retailgr.spark_session import build_spark

    spark = build_spark(cfg)
    stats: dict[str, Any] = {}
    if skip_if_present and table_exists(spark, cfg, "silver.interactions"):
        stats["skipped"] = True
        return stats
    stats["bronze"] = ingest.run(spark, cfg)
    stats["silver"] = silver.run(spark, cfg)
    return stats


def build_variant_tables(cfg: Config, variants: list[str]) -> dict[str, Any]:
    from retailgr.spark_session import build_spark

    spark = build_spark(cfg)
    out: dict[str, Any] = {}
    for variant in variants:
        granularity = "config" if variant == "config" else variant
        result = sequences.run(spark, cfg, granularity=granularity, variant=variant)
        out[variant] = result.stats
    return out


def train_and_evaluate(
    cfg: Config,
    variant: str,
    model_configs: list[str] | tuple[str, ...] = DEFAULT_MODEL_CONFIGS,
    splits: tuple[str, ...] = ("val", "test"),
    confidence: float = 0.95,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Fit every model for one variant and score them on the given splits."""
    data = load_variant(cfg, variant)
    requested = [int(k) for k in cfg.get("evaluation.k_values", [10, 50, 200])]
    # A k at or above the vocabulary size is not a ranking metric any more:
    # every token fits in the list. Drop those instead of clamping them, which
    # would silently compare variants at different k.
    k_values = sorted({k for k in requested if k < data.vocab_size})
    dropped = sorted(set(requested) - set(k_values))
    if not k_values:
        k_values = [max(1, data.vocab_size // 10)]
    exclude_seen = bool(cfg.get("evaluation.exclude_seen", True))

    results: dict[str, Any] = {
        "vocab_size": data.vocab_size,
        "k_values": k_values,
        "k_values_dropped": dropped,
        "confidence": confidence,
        "models": {},
    }

    # Popularity is always present: it is the floor every model must clear.
    models: list[tuple[str, Recommender]] = [
        ("popularity", PopularityModel(data.vocab_size, recency_halflife=0.0))
    ]
    for path in model_configs:
        model_cfg = load_model_config(path)
        models.append(build_model(data.vocab_size, model_cfg, exclude_seen=exclude_seen))

    for name, model in models:
        started = time.time()
        fit_stats = model.fit(data.train, data.val)
        entry: dict[str, Any] = {
            # The report identifies the baseline by type, never by name: presets
            # are renamed freely (sasrec_small, sasrec_base, ...).
            "type": model.name,
            "fit": fit_stats,
            "fit_seconds": round(time.time() - started, 2),
        }
        for split_name in splits:
            split = getattr(data, split_name)
            entry[split_name] = evaluate(
                model,
                split,
                vocab_size=data.vocab_size,
                k_values=k_values,
                exclude_seen=exclude_seen,
                category_by_id=data.category_by_id,
                confidence=confidence,
                resamples=resamples,
            )
        results["models"][name] = entry

    # Paired comparisons against the reference model. Two independent
    # intervals would call nearly everything a tie; the per-user difference
    # removes the between-user variance that causes that.
    reference = _reference_model(results["models"])
    if reference:
        results["reference_model"] = reference
        results["comparisons"] = compare_models(
            results["models"], reference, k_values, confidence, resamples
        )

    # Per-user arrays are for the comparisons, not for the run record.
    for entry in results["models"].values():
        for split_name in splits:
            entry[split_name].pop("per_user", None)
            entry[split_name].pop("user_ids", None)

    return results


def _reference_model(models: dict[str, Any]) -> str | None:
    """The model everything else is compared against.

    SASRec when present, since it is the published baseline; otherwise
    popularity, which is the floor.
    """
    for name, entry in models.items():
        if entry.get("type") == "sasrec":
            return name
    return "popularity" if "popularity" in models else None


def compare_models(
    models: dict[str, Any],
    reference: str,
    k_values: list[int],
    confidence: float = 0.95,
    resamples: int = 2000,
    split_name: str = "test",
) -> dict[str, Any]:
    """Paired comparison of every model against ``reference``."""
    from retailgr.evaluation.stats import paired_comparison

    base = models.get(reference, {}).get(split_name, {})
    base_per_user = base.get("per_user") or {}
    base_users = base.get("user_ids") or []
    out: dict[str, Any] = {}

    for name, entry in models.items():
        if name == reference:
            continue
        other = entry.get(split_name, {})
        other_per_user = other.get("per_user") or {}
        other_users = other.get("user_ids") or []
        if not base_per_user or not other_per_user:
            continue
        # The pairing is only sound if the rows are the same users in the same
        # order. Two loops over the same split give that, but an assertion is
        # cheaper than a silently wrong significance claim.
        if base_users != other_users:
            out[name] = {"error": "evaluation users are not aligned; comparison skipped"}
            continue

        per_metric: dict[str, Any] = {}
        for k in k_values:
            for metric in (f"recall@{k}", f"ndcg@{k}", f"hit_rate@{k}"):
                if metric not in base_per_user or metric not in other_per_user:
                    continue
                comparison = paired_comparison(
                    other_per_user[metric],
                    base_per_user[metric],
                    confidence=confidence,
                    resamples=resamples,
                )
                if comparison is not None:
                    per_metric[metric] = comparison.as_dict()
        out[name] = per_metric
    return out


# Which HSTU components to switch off, one at a time. "Does this part of the
# architecture earn its place on my data?" is a question every adopter has, and
# on a small or synthetic dataset the answer is often no for the temporal bias.
HSTU_ABLATIONS: dict[str, dict[str, Any]] = {
    "hstu_full": {},
    "hstu_no_temporal_bias": {"use_temporal_bias": False},
    "hstu_no_rab": {"use_relative_bias": False, "use_temporal_bias": False},
    "hstu_no_actions": {"use_actions": False},
    "hstu_seq_normalised": {"normalise_by": "sequence_len"},
    "hstu_all_targets": {"supervise_positive_only": False},
}


def run_ablation(
    cfg: Config,
    variant: str = "config",
    base_model_config: str = "hstu_small.yaml",
    baseline_config: str | None = "sasrec_small.yaml",
    ablations: dict[str, dict[str, Any]] | None = None,
    seeds: list[int] | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
    fixed_epochs: int | None = None,
) -> dict[str, Any]:
    """Train one HSTU per ablation on a single granularity variant.

    ``fixed_epochs`` turns early stopping off and trains every arm for that
    many epochs — the schedule every ablation in this repository used before
    validation was read at all. It exists so the two schedules can be
    compared on the *same* data, which is the only comparison that isolates
    what convergence changed.

    Everything except the named override is held fixed, so a difference is
    attributable to that one component.

    ``seeds`` is the point of this command being separate from
    ``run_experiment``. A bootstrap interval answers "would this hold on other
    users"; it cannot answer "would this hold on another training run", and on
    a small dataset that second question is usually the bigger one. Passing
    several seeds trains each configuration that many times and reports the
    spread, which is the only honest way to tell a 2% architectural difference
    from a 2% difference between two runs of the same architecture.
    """
    ablations = ablations or HSTU_ABLATIONS
    seeds = list(seeds or [])
    data = load_variant(cfg, variant)
    requested = [int(k) for k in cfg.get("evaluation.k_values", [10, 50, 200])]
    k_values = sorted({k for k in requested if k < data.vocab_size}) or [
        max(1, data.vocab_size // 10)
    ]
    exclude_seen = bool(cfg.get("evaluation.exclude_seen", True))

    base_cfg = load_model_config(base_model_config)
    entries: dict[str, Any] = {}

    runs: list[tuple[str, dict[str, Any]]] = []
    if baseline_config:
        runs.append(("sasrec (baseline)", load_model_config(baseline_config)))
    for name, overrides in ablations.items():
        runs.append((name, {**base_cfg, **overrides}))
    if fixed_epochs is not None:
        runs = [
            (name, {**{k: v for k, v in c.items() if k != "early_stopping"},
                    "epochs": int(fixed_epochs)})
            for name, c in runs
        ]

    # One seed means one run per configuration, which is the default and is
    # explicitly not enough to claim a small difference.
    seed_list = seeds or [int(base_cfg.get("seed", 13))]

    for name, model_cfg in runs:
        per_seed: list[dict[str, Any]] = []
        for seed in seed_list:
            seeded_cfg = {**model_cfg, "seed": seed}
            _, model = build_model(data.vocab_size, seeded_cfg, exclude_seen=exclude_seen)
            started = time.time()
            fit_stats = model.fit(data.train, data.val)
            evaluation = evaluate(
                model,
                data.test,
                vocab_size=data.vocab_size,
                k_values=k_values,
                exclude_seen=exclude_seen,
                confidence=confidence,
                resamples=resamples,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "fit": fit_stats,
                    "fit_seconds": round(time.time() - started, 2),
                    "test": evaluation,
                }
            )

        first = per_seed[0]
        entry: dict[str, Any] = {
            "type": "sasrec" if "baseline" in name else "hstu",
            "overrides": {
                key: value for key, value in model_cfg.items() if base_cfg.get(key) != value
            }
            if "baseline" not in name
            else {},
            "fit": first["fit"],
            "fit_seconds": round(sum(run["fit_seconds"] for run in per_seed), 2),
            "test": first["test"],
            "seeds": seed_list,
        }
        if len(seed_list) > 1:
            from retailgr.evaluation.stats import seed_spread

            entry["seed_spread"] = {
                metric: seed_spread(
                    [run["test"]["overall"].get(metric) for run in per_seed]
                )
                for metric in (f"recall@{max(k_values)}", f"ndcg@{max(k_values)}")
            }
        entries[name] = entry

    # Paired comparison of each ablation against the full model, and against
    # the SASRec baseline. Same users, so the pairing is sound.
    reference = "hstu_full" if "hstu_full" in entries else None
    comparisons: dict[str, Any] = {}
    if reference:
        comparisons["vs_hstu_full"] = compare_models(
            entries, reference, k_values, confidence, resamples
        )
    if baseline_config and "sasrec (baseline)" in entries:
        comparisons["vs_sasrec"] = compare_models(
            entries, "sasrec (baseline)", k_values, confidence, resamples
        )

    for entry in entries.values():
        entry["test"].pop("per_user", None)
        entry["test"].pop("user_ids", None)

    return {
        "variant": variant,
        "vocab_size": data.vocab_size,
        "k_values": k_values,
        "base_model_config": base_model_config,
        "confidence": confidence,
        "seeds": seed_list,
        "models": entries,
        "comparisons": comparisons,
    }


def render_ablation(result: dict[str, Any]) -> str:
    """Markdown table for :func:`run_ablation`."""
    k = max(result["k_values"])
    seeds = result.get("seeds") or []
    confidence = result.get("confidence", 0.95)
    lines = [
        f"# HSTU ablation on variant `{result['variant']}`",
        "",
        f"- Base config: `{result['base_model_config']}`",
        f"- Token vocabulary: {result['vocab_size']}",
        # The dataset a number came from. The README once quoted an ablation
        # on 1,101 test users for weeks after the consent filter had cut the
        # pipeline's output to 943, and nothing on the page said which.
        f"- Test users: {_test_users(result)}",
        f"- Metric: @{k} on the test split",
        f"- Training seeds: {', '.join(str(s) for s in seeds)}"
        + (" (one run per configuration)" if len(seeds) < 2 else ""),
        "",
    ]

    header = ["Model", f"NDCG@{k}", f"{confidence:.0%} interval", "Final loss"]
    lines.append(_format_row(header))
    lines.append(_format_row(["---"] * len(header)))
    for name, entry in result["models"].items():
        overall = entry["test"]["overall"]
        intervals = entry["test"].get("intervals", {})
        bounds = intervals.get(f"ndcg@{k}")
        lines.append(
            _format_row(
                [
                    f"`{name}`",
                    f"{overall.get(f'ndcg@{k}', 0):.4f}",
                    f"[{bounds['ci_low']:.4f}, {bounds['ci_high']:.4f}]" if bounds else "-",
                    f"{entry['fit'].get('final_epoch_loss', float('nan')):.3f}",
                ]
            )
        )
    lines.append("")

    # -- what the pairing says ------------------------------------------------
    from retailgr.evaluation.stats import holm_bonferroni

    # Verdicts at the cutoff a customer sees *and* at the deep one, each
    # titled with its metric. This used to render only `max(k_values)`, i.e.
    # NDCG@200, under a column called "Difference" — and the README copied
    # that table as "the temporal bias costs 3.7%" with no cutoff attached.
    # At NDCG@10 the same comparison was +0.0003, p=0.740: nothing. A verdict
    # that cannot be quoted without its cutoff is the fix.
    cutoffs = sorted({min(result["k_values"]), k})
    for cutoff in cutoffs:
        for label, title in (
            ("vs_sasrec", "Against the SASRec baseline"),
            ("vs_hstu_full", "Against the full HSTU"),
        ):
            lines.extend(
                _paired_table(result, label, title, cutoff, confidence, holm_bonferroni)
            )

    # -- seed spread ----------------------------------------------------------
    return _render_ablation_tail(result, lines, k)


def _test_users(result: dict[str, Any]) -> str:
    """How many users the numbers are over, read from the results themselves."""
    for entry in (result.get("models") or {}).values():
        overall = ((entry or {}).get("test") or {}).get("overall") or {}
        if "eval_users" in overall:
            return str(int(overall["eval_users"]))
    for family in (result.get("comparisons") or {}).values():
        for metrics in (family or {}).values():
            for stats in (metrics or {}).values():
                if isinstance(stats, dict) and "n" in stats:
                    return str(int(stats["n"]))
    return "unknown"


def _paired_table(
    result: dict[str, Any],
    label: str,
    title: str,
    cutoff: int,
    confidence: float,
    holm_bonferroni: Any,
) -> list[str]:
    """One verdict table, titled with its cutoff. Each is its own test family."""
    lines: list[str] = []
    metric = f"ndcg@{cutoff}"
    comparisons = (result.get("comparisons") or {}).get(label) or {}
    rows = [
        (name, (metrics or {}).get(metric))
        for name, metrics in comparisons.items()
        if (metrics or {}).get(metric)
    ]
    if not rows:
        return lines
    adjusted = holm_bonferroni({name: stats["p_value"] for name, stats in rows})

    lines.append(f"## {title} — NDCG@{cutoff}, paired on the same users")
    lines.append("")
    header = [
        "Model", f"Δ NDCG@{cutoff}", f"{confidence:.0%} interval", "p", "p adj.", "Verdict"
    ]
    lines.append(_format_row(header))
    lines.append(_format_row(["---"] * len(header)))
    for name, stats in rows:
        family = adjusted.get(name, {})
        survives = family.get("significant", stats["significant"])
        relative = stats.get("relative_difference")
        if not survives:
            verdict = "no difference detected"
        elif relative is not None:
            verdict = (
                f"**{abs(relative) * 100:.1f}% "
                + ("better" if stats["mean_difference"] > 0 else "worse")
                + f" at @{cutoff}**"
            )
        else:
            verdict = "**differs**"
        lines.append(
            _format_row(
                [
                    f"`{name}`",
                    f"{stats['mean_difference']:+.4f}",
                    f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]",
                    f"{stats['p_value']:.3f}",
                    f"{family.get('p_adjusted', float('nan')):.3f}",
                    verdict,
                ]
            )
        )
    lines.append("")
    lines.append(
        f"`p adj.` is Holm-Bonferroni across the {len(rows)} comparisons in this "
        "table; the verdict uses it. A table of tests read at p<0.05 each is "
        "several chances to be fooled, not one."
    )
    lines.append("")
    return lines


def _render_ablation_tail(result: dict[str, Any], lines: list[str], k: int) -> str:
    """Seed spread and the closing note."""
    spreads = {
        name: entry["seed_spread"]
        for name, entry in result["models"].items()
        if entry.get("seed_spread")
    }
    if spreads:
        lines.append("## Across training seeds")
        lines.append("")
        header = ["Model", "Seeds", f"NDCG@{k} mean", "min", "max", "std"]
        lines.append(_format_row(header))
        lines.append(_format_row(["---"] * len(header)))
        for name, spread in spreads.items():
            values = spread.get(f"ndcg@{k}")
            if not values:
                continue
            lines.append(
                _format_row(
                    [
                        f"`{name}`",
                        str(values["seeds"]),
                        f"{values['mean']:.4f}",
                        f"{values['min']:.4f}",
                        f"{values['max']:.4f}",
                        f"{values['std']:.4f}",
                    ]
                )
            )
        lines.append("")
        lines.append(
            "This is the variance a bootstrap cannot see. If the spread between "
            "seeds of the *same* configuration is as large as the difference "
            "between two configurations, the difference is not a finding."
        )
        lines.append("")
    else:
        lines.append(
            "> Run with `--seeds 13 17 23` to separate an architectural difference "
            "from training noise. With one run per configuration, a small gap here "
            "is not evidence of anything."
        )
        lines.append("")

    lines.append(
        "Each row changes exactly one thing against the base config. A row that "
        "beats `hstu_full` *with an interval that excludes zero* means that "
        "component is not earning its place on this data — **at that cutoff**. "
        "A finding at NDCG@200 is a statement about positions 11-200, a fifth "
        "of a 945-item catalogue; it says nothing about the ten a customer sees "
        "unless the @10 table agrees."
    )
    return "\n".join(lines)


def run_experiment(
    cfg: Config,
    variants: list[str] | None = None,
    model_configs: list[str] | tuple[str, ...] | None = None,
    reuse_pipeline: bool = False,
) -> RunRecord:
    variants = list(variants or DEFAULT_VARIANTS)
    model_configs = list(model_configs or DEFAULT_MODEL_CONFIGS)
    started = time.time()
    record = RunRecord(
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=cfg.dataset_name,
        warehouse_backend=cfg.warehouse_backend,
        model_configs=list(model_configs),
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )

    record.pipeline_stats = build_pipeline(cfg, skip_if_present=reuse_pipeline)
    variant_stats = build_variant_tables(cfg, variants)

    # Spark's work is done; the training loop needs those cores and that heap.
    from retailgr.spark_session import stop_spark

    stop_spark()

    for variant in variants:
        entry: dict[str, Any] = {"data": variant_stats[variant]}
        entry.update(train_and_evaluate(cfg, variant, model_configs))
        record.variants[variant] = entry

    record.duration_seconds = round(time.time() - started, 2)
    return record


# -- reporting ----------------------------------------------------------------


def _format_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_report(record: RunRecord, k: int | None = None) -> str:
    """Render the run as a Markdown report."""
    lines: list[str] = []
    lines.append("# RetailGR - Stage 1 run report")
    lines.append("")
    lines.append(f"- Dataset: `{record.dataset}`")
    lines.append(f"- Warehouse backend: `{record.warehouse_backend}`")
    lines.append(f"- Started (UTC): {record.started_at}")
    lines.append(f"- Duration: {record.duration_seconds}s")
    lines.append("")

    # Pick the metric k: the largest k that EVERY variant reported. A k that
    # only some variants reached would compare them on different scales.
    per_variant_ks: list[set[int]] = []
    for variant in record.variants.values():
        variant_ks: set[int] = set()
        for model in variant.get("models", {}).values():
            for key in model.get("test", {}).get("overall", {}):
                if key.startswith("recall@"):
                    variant_ks.add(int(key.split("@")[1]))
        if variant_ks:
            per_variant_ks.append(variant_ks)
    shared_ks = set.intersection(*per_variant_ks) if per_variant_ks else set()
    if not shared_ks:
        lines.append("_No evaluation metrics shared by every variant were produced._")
        return "\n".join(lines)
    metric_k = k if k in shared_ks else max(shared_ks)

    dropped = {
        name: variant.get("k_values_dropped") or []
        for name, variant in record.variants.items()
        if variant.get("k_values_dropped")
    }
    if dropped:
        notes = ", ".join(f"`{name}`: {values}" for name, values in dropped.items())
        lines.append(
            f"> Some k values were dropped because they reach or exceed the vocabulary "
            f"size ({notes}). At such a k the model returns the whole catalog and the "
            f"metric stops discriminating."
        )
        lines.append("")

    lines.append("## Vocabulary and sparsity")
    lines.append("")
    lines.append(
        _format_row(["Variant", "Tokens", "Distinct SKUs", "SKUs per token", "Tokens < 5 events"])
    )
    lines.append(_format_row(["---"] * 5))
    for name, variant in record.variants.items():
        data = variant.get("data", {})
        share = data.get("sparse_token_share")
        lines.append(
            _format_row(
                [
                    f"`{name}`",
                    str(data.get("vocab_size", "-")),
                    str(data.get("distinct_skus", "-")),
                    str(data.get("compression_vs_sku", "-")),
                    f"{data.get('tokens_under_5_events', '-')}"
                    + (f" ({share:.1%})" if isinstance(share, float) else ""),
                ]
            )
        )
    lines.append("")

    confidence = next(
        (v.get("confidence") for v in record.variants.values() if v.get("confidence")), 0.95
    )
    lines.append(f"## Test metrics @{metric_k}")
    lines.append("")
    lines.append(
        f"Means with {confidence:.0%} bootstrap intervals over users. The interval "
        "covers user sampling only — not training-seed variance, which needs "
        "several runs (`ablate --seeds`)."
    )
    lines.append("")
    lines.append(
        _format_row(
            [
                "Variant",
                "Model",
                f"Recall@{metric_k}",
                f"NDCG@{metric_k}",
                f"Coverage@{metric_k}",
                "Users",
            ]
        )
    )
    lines.append(_format_row(["---"] * 6))
    for name, variant in record.variants.items():
        for model_name, model in variant.get("models", {}).items():
            test = model.get("test", {})
            overall = test.get("overall", {})
            if not overall:
                continue
            intervals = test.get("intervals", {})

            def cell(metric: str, intervals=intervals, overall=overall) -> str:
                point = overall.get(metric, 0.0)
                bounds = intervals.get(metric)
                if not bounds:
                    return f"{point:.4f}"
                return f"{point:.4f} [{bounds['ci_low']:.4f}, {bounds['ci_high']:.4f}]"

            lines.append(
                _format_row(
                    [
                        f"`{name}`",
                        model_name,
                        cell(f"recall@{metric_k}"),
                        cell(f"ndcg@{metric_k}"),
                        f"{overall.get(f'coverage@{metric_k}', 0):.4f}",
                        str(int(overall.get("eval_users", 0))),
                    ]
                )
            )
    lines.append("")

    # -- head-to-head: every sequence model against the SASRec baseline -------
    model_names: list[str] = []
    types: dict[str, str] = {}
    for variant in record.variants.values():
        for model_name, model in variant.get("models", {}).items():
            if model_name not in model_names:
                model_names.append(model_name)
            types.setdefault(model_name, str(model.get("type") or model_name))

    baseline_name = next((n for n in model_names if types.get(n) == "sasrec"), None)
    challengers = [
        n for n in model_names if n != baseline_name and types.get(n) != "popularity"
    ]
    challengers = [n for n in challengers if n != "popularity"]

    if baseline_name and challengers:
        lines.append(f"## Head to head vs {baseline_name}, NDCG@{metric_k}")
        lines.append("")
        # Every row in this table is a separate test, so the p-values are a
        # family and need adjusting before any of them is called significant.
        from retailgr.evaluation.stats import holm_bonferroni

        rows: list[tuple[str, str, dict[str, Any]]] = []
        for name, variant in record.variants.items():
            comparisons = variant.get("comparisons", {})
            for challenger in challengers:
                stats = (comparisons.get(challenger) or {}).get(f"ndcg@{metric_k}")
                if stats:
                    rows.append((name, challenger, stats))
        adjusted = holm_bonferroni(
            {f"{name}|{challenger}": stats["p_value"] for name, challenger, stats in rows}
        )

        header = [
            "Variant",
            "Challenger",
            "Difference",
            f"{confidence:.0%} interval",
            "p",
            "p adj.",
            "Verdict",
        ]
        lines.append(_format_row(header))
        lines.append(_format_row(["---"] * len(header)))
        for name, challenger, stats in rows:
            family = adjusted.get(f"{name}|{challenger}", {})
            survives = family.get("significant", stats["significant"])
            relative = stats.get("relative_difference")
            if not survives:
                verdict = "no difference detected"
            elif relative is not None:
                verdict = (
                    f"**{abs(relative) * 100:.1f}% "
                    + ("better" if stats["mean_difference"] > 0 else "worse")
                    + "**"
                )
            else:
                verdict = "**differs**"
            lines.append(
                _format_row(
                    [
                        f"`{name}`",
                        challenger,
                        f"{stats['mean_difference']:+.4f}",
                        f"[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}]",
                        f"{stats['p_value']:.3f}",
                        f"{family.get('p_adjusted', float('nan')):.3f}",
                        verdict,
                    ]
                )
            )
        lines.append("")
        lines.append(
            f"Paired against `{baseline_name}` on the same users, so the "
            "between-user spread — which is much larger than the gap between two "
            "models — is removed. That pairing is what gives these comparisons "
            "the power to resolve a 3% difference at all."
        )
        lines.append("")
        lines.append(
            "`p` is a sign-flip permutation test; `p adj.` is Holm-Bonferroni "
            f"across the {len(rows)} comparisons in this table, because showing "
            "several tests and reading the small ones is several chances to be "
            "fooled rather than one. The verdict column uses the adjusted value."
        )
        lines.append("")
        lines.append(
            "The interval is on the *difference*. When it includes zero the two "
            "models are indistinguishable on this data, whatever the point "
            "estimates look like. None of this covers training-seed variance — "
            "for that, `ablate --seeds`."
        )
        lines.append("")
        lines.append(
            "The presets are matched on hidden size, depth, heads, dropout, sequence "
            "length, loss, batch size, epochs and seed, so a real gap is the "
            "architecture and its extra modalities, not capacity."
        )
        lines.append("")

    # -- per-category breakdown, one table per sequence model ------------------
    for model_name in [m for m in model_names if m != "popularity"]:
        categories: set[str] = set()
        for variant in record.variants.values():
            by_category = (
                variant.get("models", {}).get(model_name, {}).get("test", {}).get("by_category", {})
            )
            categories.update(by_category)
        if not categories:
            continue
        lines.append(f"## {model_name} by category, Recall@{metric_k}")
        lines.append("")
        ordered = sorted(categories)
        lines.append(_format_row(["Variant", *ordered]))
        lines.append(_format_row(["---"] * (len(ordered) + 1)))
        for name, variant in record.variants.items():
            by_category = (
                variant.get("models", {}).get(model_name, {}).get("test", {}).get("by_category", {})
            )
            cells = [f"`{name}`"]
            for category in ordered:
                value = by_category.get(category, {}).get(f"recall@{metric_k}")
                cells.append(f"{value:.4f}" if isinstance(value, int | float) else "-")
            lines.append(_format_row(cells))
        lines.append("")

    lines.append(
        "For the granularity decision, read the per-category tables rather than the "
        "overall one: the winning token level is a per-category choice, and the "
        "overall number is dominated by whichever category has the most traffic."
    )
    return "\n".join(lines)


def save_run(record: RunRecord, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = record.started_at.replace(":", "").replace("-", "")
    json_path = output_dir / f"run_{stamp}.json"
    report_path = output_dir / f"run_{stamp}.md"
    privacy.write_json(json_path, asdict(record))
    report_path.write_text(render_report(record), encoding="utf-8")
    privacy.write_json(output_dir / "latest.json", asdict(record))
    (output_dir / "latest.md").write_text(render_report(record), encoding="utf-8")
    return {"json": json_path, "report": report_path}
