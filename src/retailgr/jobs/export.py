"""Train a model for serving and export it as a bundle.

The token-to-SKU mapping is built here, from the same
``silver.item_hierarchy`` table and the same ``GranularityResolver`` the
training job used. Rebuilding it at serving time from a different source is
how a serving path ends up recommending tokens it cannot resolve to anything
orderable.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from retailgr.config import Config, load_model_config
from retailgr.experiment import build_model
from retailgr.io.loaders import load_variant
from retailgr.jobs.sequences import resolver_for
from retailgr.serving.bundle import BundleManifest, export_bundle

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


def token_to_sku_mapping(
    spark: SparkSession, cfg: Config, granularity: str = "config"
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Map each model token to its SKUs and to a product id.

    Returns ``(skus_by_token, product_by_token)``.
    """
    from pyspark.sql import functions as F

    from retailgr.io.tables import read_table

    resolver = resolver_for(cfg, granularity)
    hierarchy = read_table(spark, cfg, "silver.item_hierarchy").withColumn(
        "token", resolver.token_column()
    )
    rows = (
        hierarchy.select("token", "sku", "product_id")
        .groupBy("token")
        .agg(
            F.collect_list("sku").alias("skus"),
            F.first("product_id", ignorenulls=True).alias("product_id"),
        )
        .collect()
    )

    skus_by_token: dict[str, list[str]] = defaultdict(list)
    product_by_token: dict[str, str] = {}
    for row in rows:
        token = str(row["token"])
        skus_by_token[token] = sorted({str(sku) for sku in row["skus"]})
        product_by_token[token] = str(row["product_id"] or token)
    return dict(skus_by_token), product_by_token


def run(
    spark: SparkSession,
    cfg: Config,
    output_dir: str | Path,
    variant: str = "config",
    model_config_path: str = "hstu_small.yaml",
    ranker_config_path: str | None = "ranker_small.yaml",
    model_version: str | None = None,
    evaluate_ranker: bool = True,
    force_ranker: bool = False,
) -> dict[str, Any]:
    """Train retrieval and (optionally) the ranker, then write one bundle."""
    import time

    data = load_variant(cfg, variant)
    # Read the hierarchy and release Spark before training: the trainer wants
    # the cores and the heap the driver JVM is holding.
    skus_by_token, product_by_token = token_to_sku_mapping(spark, cfg, variant)
    from retailgr.spark_session import stop_spark

    stop_spark()

    model_cfg = load_model_config(model_config_path)
    name, model = build_model(
        data.vocab_size,
        model_cfg,
        exclude_seen=bool(cfg.get("evaluation.exclude_seen", True)),
    )
    started = time.time()
    fit_stats = model.fit(data.train, data.val)
    retrieval_seconds = round(time.time() - started, 2)

    ranker = None
    ranker_cfg: dict[str, Any] | None = None
    ranker_report: dict[str, Any] = {}
    if ranker_config_path:
        from retailgr.models.ranker import HSTURanker

        ranker_cfg = load_model_config(ranker_config_path)
        ranker = HSTURanker(data.vocab_size, ranker_cfg)
        started = time.time()
        ranker_fit = ranker.fit(data.train, data.val)
        ranker_report = {
            "dataset": cfg.dataset_name,
            "variant": variant,
            "vocab_size": data.vocab_size,
            "fit": ranker_fit,
            "fit_seconds": round(time.time() - started, 2),
        }
        if evaluate_ranker:
            from retailgr.evaluation.ranking import (
                calibrate_heads,
                evaluate_discrimination,
                evaluate_heads,
                evaluate_reranking,
                ranker_gate,
                retrieval_discrimination,
            )

            ranker_report["heads"] = evaluate_heads(ranker, data.test)

            # Calibration comes first, and the order is load-bearing. Every
            # number below is a property of the blended score, and the blend
            # consumes the calibrated probabilities. Measuring the gate on the
            # uncalibrated blend and then shipping the calibrated one would
            # mean the gate approved a system that was never evaluated.
            uncalibrated = evaluate_discrimination(
                ranker, data.test, vocab_size=data.vocab_size
            )
            calibration = calibrate_heads(
                ranker,
                model,
                fit_split=data.val,
                test_split=data.test,
                vocab_size=data.vocab_size,
                candidate_k=int(cfg.get("serving.rank_k", 300)),
                max_users=int(cfg.get("evaluation.calibration_users", 400)),
                kind=str(cfg.get("evaluation.calibrator", "platt")),
            )
            calibrators = calibration.pop("calibrators")
            ranker.calibrators = calibrators
            ranker_report["calibration"] = calibration

            # Now the same diagnostic, on the ranker as it would actually
            # serve. Per-head numbers are unchanged by construction — a
            # calibrator is monotone — so any movement here is the blend
            # finally weighting heads that are on comparable scales.
            ranker_report["discrimination"] = evaluate_discrimination(
                ranker, data.test, vocab_size=data.vocab_size
            )
            ranker_report["discrimination_uncalibrated"] = uncalibrated
            # The same task, scored by retrieval. This is the comparison the
            # gate turns on.
            ranker_report["retrieval_discrimination"] = retrieval_discrimination(
                model, data.test, vocab_size=data.vocab_size
            )
            ranker_report["gate"] = ranker_gate(
                ranker_report["discrimination"],
                ranker_report["retrieval_discrimination"],
                min_relative_mrr=float(cfg.get("evaluation.ranker_gate_min_mrr", 1.0)),
                calibration=calibration,
                max_bias_ratio=float(cfg.get("evaluation.ranker_gate_max_bias", 4.0)),
            )
            # The question that decides whether the second stage ships: does
            # reordering retrieval's own candidates improve the metric?
            ranker_report["reranking"] = evaluate_reranking(
                ranker,
                model,
                data.test,
                vocab_size=data.vocab_size,
                candidate_k=int(cfg.get("serving.rank_k", 300)),
                metric_k=10,
                max_users=int(cfg.get("evaluation.rerank_users", 300)),
            )

    # The gate. A ranker that orders worse than retrieval is still exported
    # as a file — so the next run can compare against it — but it is not
    # wired into the request path, because ``load_bundle`` only loads a ranker
    # whose config the manifest carries. Shipping a regression with extra
    # latency should take an explicit override, not a default.
    gate = ranker_report.get("gate") or {}
    serve_ranker = bool(ranker_cfg) and (gate.get("passed", True) or force_ranker)
    if ranker_cfg and not serve_ranker:
        ranker_report["served"] = False
    # The calibrators only ship with the ranker they correct. A forced ranker
    # keeps them: overriding the gate should mean "ship it anyway", not "ship
    # it anyway *and* undo the one correction that was measured to help".
    head_calibration = (
        ranker.calibrators.as_dict() if (ranker is not None and serve_ranker) else None
    )

    manifest = BundleManifest(
        model_version=model_version or f"{name}-{variant}-{data.vocab_size}",
        model_type=model.name,
        variant=variant,
        dataset=cfg.dataset_name,
        vocab_size=data.vocab_size,
        embedding_dim=int(model_cfg.get("hidden_dim", 64)),
        max_len=int(model_cfg.get("max_len", 50)),
        model_config=model_cfg,
        metrics={"fit": fit_stats, "fit_seconds": retrieval_seconds},
        ranker_config=ranker_cfg if serve_ranker else None,
        ranker_metrics=ranker_report,
        head_calibration=head_calibration,
    )

    path = export_bundle(
        Path(output_dir),
        model=model,
        manifest=manifest,
        token_by_id=data.token_by_id,
        category_by_id=data.category_by_id,
        skus_by_token=skus_by_token,
        product_by_token=product_by_token,
        ranker=ranker,
    )
    return {
        "bundle": str(path),
        "model_version": manifest.model_version,
        "model_type": manifest.model_type,
        "vocab_size": data.vocab_size,
        "tokens_with_skus": len(skus_by_token),
        "fit": fit_stats,
        "ranker_served": serve_ranker,
        "ranker": ranker_report,
    }
