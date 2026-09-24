"""Command line entry point: ``python -m retailgr.cli <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from retailgr import privacy
from retailgr.config import Config

# Imported eagerly so --help lists the defaults; the model classes themselves
# are imported lazily inside the commands.
DEFAULT_MODEL_CONFIGS = ("sasrec_small.yaml", "hstu_small.yaml")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to pipeline.yaml")
    parser.add_argument("--granularity-config", default=None, help="path to granularity.yaml")
    parser.add_argument(
        "--dataset", default=None, help="override dataset.name (synthetic|rees46|hm)"
    )
    parser.add_argument(
        "--backend", default=None, help="override warehouse.backend (parquet|iceberg)"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override any pipeline.yaml value by dotted path, repeatable, e.g. "
            "--set clean.bot_events_per_day=100000 --set sequences.max_len=100"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch/BLAS thread budget; defaults to the available cores (1 when serving)",
    )


def _parse_scalar(text: str):
    """Read a CLI override value as YAML, so types survive."""
    import yaml

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_dotted(target: dict, dotted_key: str, value) -> None:
    parts = dotted_key.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set '{dotted_key}': '{part}' is not a mapping")
    node[parts[-1]] = value


def _load_config(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if getattr(args, "dataset", None):
        overrides.setdefault("dataset", {})["name"] = args.dataset
    if getattr(args, "backend", None):
        overrides.setdefault("warehouse", {})["backend"] = args.backend
    for item in getattr(args, "overrides", []) or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        _apply_dotted(overrides, key.strip(), _parse_scalar(raw.strip()))
    return Config.load(
        pipeline_path=args.config,
        granularity_path=args.granularity_config,
        overrides=overrides or None,
    )


def cmd_generate_data(args: argparse.Namespace) -> int:
    from retailgr.datasets.synthetic import SyntheticConfig, generate

    cfg = _load_config(args)
    params = cfg.get("dataset.synthetic", {}) or {}
    if args.users:
        params["n_users"] = args.users
    if args.products:
        params["n_products"] = args.products
    paths = generate(cfg.raw_path / "synthetic", SyntheticConfig(**params))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from retailgr.jobs import ingest
    from retailgr.spark_session import build_spark

    cfg = _load_config(args)
    stats = ingest.run(build_spark(cfg), cfg)
    print(json.dumps(privacy.scrub(stats), indent=2))
    return 0


def cmd_silver(args: argparse.Namespace) -> int:
    from retailgr.jobs import silver
    from retailgr.spark_session import build_spark

    cfg = _load_config(args)
    stats = silver.run(build_spark(cfg), cfg)
    print(json.dumps(privacy.scrub(stats), indent=2))
    return 0


def cmd_sequences(args: argparse.Namespace) -> int:
    from retailgr.jobs import sequences
    from retailgr.spark_session import build_spark

    cfg = _load_config(args)
    result = sequences.run(
        build_spark(cfg), cfg, granularity=args.granularity, variant=args.variant
    )
    print(json.dumps(privacy.scrub(result.stats), indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from retailgr.experiment import train_and_evaluate

    cfg = _load_config(args)
    results = train_and_evaluate(cfg, args.variant, args.model_configs)
    print(json.dumps(privacy.scrub(results), indent=2))
    return 0


def cmd_export_model(args: argparse.Namespace) -> int:
    from retailgr.jobs import export
    from retailgr.spark_session import build_spark

    cfg = _load_config(args)
    output = Path(args.output) if args.output else cfg.root / "artifacts" / "bundle"
    stats = export.run(
        build_spark(cfg),
        cfg,
        output_dir=output,
        variant=args.variant,
        model_config_path=args.model_config,
        ranker_config_path=None if args.no_ranker else args.ranker_config,
        model_version=args.model_version,
        evaluate_ranker=not args.skip_ranker_eval,
        force_ranker=args.force_ranker,
    )
    print(json.dumps(privacy.scrub(stats), indent=2))

    if stats.get("ranker"):
        from retailgr.evaluation.ranking import render_ranker_report

        report = render_ranker_report(stats["ranker"])
        report_dir = Path(args.output).parent if args.output else cfg.artifacts_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "ranker.md").write_text(report, encoding="utf-8")
        print()
        print(report)
        print()
        print(f"report: {report_dir / 'ranker.md'}")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from retailgr.jobs import bootstrap
    from retailgr.online_store import build_online_store
    from retailgr.spark_session import build_spark
    from retailgr.streaming.broker import build_broker

    cfg = _load_config(args)
    stats = bootstrap.run(
        build_spark(cfg),
        cfg,
        broker=build_broker(cfg),
        store=build_online_store(cfg),
        variant=args.variant,
        limit=args.limit,
        speed_factor=args.speed_factor,
    )
    print(json.dumps(privacy.scrub(stats), indent=2))
    if str(cfg.get("online_store.backend", "memory")).lower() == "memory":
        print(
            "\nNote: online_store.backend=memory, so this store lives only inside "
            "this process. Point it at Redis (online_store.backend=redis) for the "
            "state to outlive the command."
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - runs a server
    from retailgr.serving.api import serve

    cfg = _load_config(args)
    bundle = args.bundle or str(cfg.root / "artifacts" / "bundle")
    store = None
    if args.bootstrap:
        # An in-memory store is empty in a fresh process, so fill it first or
        # every request falls back to the cold-start list.
        from retailgr.jobs import bootstrap
        from retailgr.online_store import build_online_store
        from retailgr.spark_session import build_spark
        from retailgr.streaming.broker import build_broker

        store = build_online_store(cfg)
        stats = bootstrap.run(
            build_spark(cfg),
            cfg,
            broker=build_broker(cfg),
            store=store,
            variant=args.variant,
            limit=args.limit,
        )
        print(f"bootstrapped online store: {json.dumps(stats['store'])}")

    host = args.host or str(cfg.get("serving.host", "127.0.0.1"))
    port = args.port or int(cfg.get("serving.port", 8080))
    print(f"serving on http://{host}:{port} (bundle: {bundle})")
    serve(cfg, bundle, host=host, port=port, store=store)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from retailgr.serving.bench import render_bench, run_bench

    cfg = _load_config(args)
    bundle = args.bundle or str(cfg.root / "artifacts" / "bundle")
    result = run_bench(
        cfg,
        bundle_dir=bundle,
        variant=args.variant,
        requests=args.requests,
        limit=args.limit,
        warmup=args.warmup,
    )
    report = render_bench(result)
    print(report)
    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latency.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / "latency.json", result)
    print()
    print(f"report: {output_dir / 'latency.md'}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    from retailgr.experiment import HSTU_ABLATIONS, render_ablation, run_ablation

    cfg = _load_config(args)
    ablations = None
    if args.ablations:
        unknown = set(args.ablations) - set(HSTU_ABLATIONS)
        if unknown:
            raise SystemExit(
                f"unknown ablations {sorted(unknown)}; available: {sorted(HSTU_ABLATIONS)}"
            )
        ablations = {name: HSTU_ABLATIONS[name] for name in args.ablations}
    result = run_ablation(
        cfg,
        variant=args.variant,
        base_model_config=args.model_config,
        baseline_config=None if args.no_baseline else args.baseline_config,
        ablations=ablations,
        seeds=args.seeds,
        confidence=args.confidence,
        resamples=args.resamples,
        fixed_epochs=args.fixed_epochs,
    )
    report = render_ablation(result)
    print(report)

    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"ablation_{args.variant}.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / f"ablation_{args.variant}.json", result)
    print()
    print(f"report: {output_dir / f'ablation_{args.variant}.md'}")
    return 0


def cmd_slate_experiment(args: argparse.Namespace) -> int:
    from retailgr.slate_experiment import (
        SLATE_VARIANTS,
        render_slate_experiment,
        run_slate_experiment,
    )

    cfg = _load_config(args)
    variants = None
    if args.arms:
        unknown = set(args.arms) - set(SLATE_VARIANTS)
        if unknown:
            raise SystemExit(
                f"unknown arms {sorted(unknown)}; available: {sorted(SLATE_VARIANTS)}"
            )
        variants = {name: SLATE_VARIANTS[name] for name in args.arms}

    result = run_slate_experiment(
        cfg,
        variant=args.variant,
        retrieval_config=args.model_config,
        ranker_config=args.ranker_config,
        variants=variants,
        seeds=tuple(args.seeds),
        max_users=args.users,
    )
    report = render_slate_experiment(result)
    print(report)

    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slate.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / "slate.json", result)
    print(f"report: {output_dir / 'slate.md'}")
    return 0


def cmd_convergence(args: argparse.Namespace) -> int:
    from retailgr.convergence import render_convergence, run_convergence

    cfg = _load_config(args)
    result = run_convergence(
        cfg,
        variant=args.variant,
        model_configs=tuple(args.model_configs),
        seeds=tuple(args.seeds),
        patience=args.patience,
        max_epochs=args.max_epochs,
        fixed_epochs=args.fixed_epochs,
        metric=args.metric,
    )
    report = render_convergence(result)
    print(report)
    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "convergence.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / "convergence.json", result)
    print(f"report: {output_dir / 'convergence.md'}")
    return 0


def cmd_retention(args: argparse.Namespace) -> int:
    from retailgr.jobs import retention
    from retailgr.spark_session import build_spark

    cfg = _load_config(args)
    report = retention.run(build_spark(cfg), cfg, dry_run=args.dry_run)
    text = retention.render(report)
    print(text)

    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "retention.md").write_text(text, encoding="utf-8")
    privacy.write_json(output_dir / "retention.json", report)
    # Non-zero when the pass refused, so a CronJob surfaces it rather than
    # recording a success that deleted nothing.
    return 1 if report.get("refused") else 0


def cmd_forget(args: argparse.Namespace) -> int:
    from retailgr import erasure

    cfg = _load_config(args)
    if args.verify_only:
        result = erasure.verify(cfg, args.user_id, spark=None)
        print(json.dumps(privacy.scrub(result), indent=2))
        return 0 if result["clean"] else 1

    report = erasure.erase(cfg, args.user_id, dry_run=args.dry_run)
    # Named on the terminal, digested on disk: the operator typed the id a
    # moment ago, and a permanent file naming the person who asked to be
    # forgotten is a new record of them created by deleting them.
    print(erasure.render(report, reveal=True))

    settings = cfg.get("privacy.pseudonymisation", {}) or {}
    key = (
        privacy.pseudonymisation_key(settings.get("key"))
        if settings.get("enabled", False)
        else None
    )
    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"erasure_{report.subject_digest(key)}"
    (output_dir / f"{stem}.md").write_text(
        erasure.render(report, reveal=False, key=key), encoding="utf-8"
    )
    privacy.write_json(output_dir / f"{stem}.json", report.as_dict(key))
    # Non-zero when the identifier survived somewhere. An erasure command
    # that exits 0 having failed is the reason this whole module exists.
    return 0 if (args.dry_run or report.complete) else 1


def cmd_consent_cost(args: argparse.Namespace) -> int:
    from retailgr.consent_experiment import (
        DEFAULT_RATES,
        render_consent_cost,
        run_consent_cost,
    )

    cfg = _load_config(args)
    rates = tuple(args.rates) if args.rates else DEFAULT_RATES
    if 1.0 not in rates:
        # The control is not optional. Without it the curve has no baseline
        # and the seed spread has nothing to be compared against.
        rates = (1.0, *rates)

    result = run_consent_cost(
        cfg,
        variant=args.variant,
        model_config=args.model_config,
        rates=rates,
        seeds=tuple(args.seeds),
        correlate=args.correlate,
    )
    report = render_consent_cost(result)
    print(report)

    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "consent_cost_correlated" if args.correlate else "consent_cost"
    (output_dir / f"{stem}.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / f"{stem}.json", result)
    print(f"report: {output_dir / f'{stem}.md'}")
    return 0


def cmd_rank_experiment(args: argparse.Namespace) -> int:
    from retailgr.ranker_experiment import (
        RANKER_VARIANTS,
        render_ranker_experiment,
        run_ranker_experiment,
    )

    cfg = _load_config(args)
    variants = None
    if args.variants:
        unknown = set(args.variants) - set(RANKER_VARIANTS)
        if unknown:
            raise SystemExit(
                f"unknown variants {sorted(unknown)}; available: {sorted(RANKER_VARIANTS)}"
            )
        variants = {name: RANKER_VARIANTS[name] for name in args.variants}

    overrides: dict[str, object] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.num_negatives is not None:
        overrides["num_negatives"] = args.num_negatives

    result = run_ranker_experiment(
        cfg,
        variant=args.variant,
        retrieval_config=args.model_config,
        ranker_config=args.ranker_config,
        variants=variants,
        overrides=overrides or None,
        calibration_users=args.calibration_users,
        discrimination_users=args.discrimination_users,
        confidence=args.confidence,
        resamples=args.resamples,
    )
    report = render_ranker_experiment(result)
    print(report)

    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"ranker_negatives_{args.variant}"
    (output_dir / f"{name}.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / f"{name}.json", result)
    print()
    print(f"report: {output_dir / f'{name}.md'}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from retailgr.platform import audit, render_audit

    result = audit(args.source)
    report = render_audit(result)
    print(report)

    output_dir = Path(args.output) if args.output else Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "platform_audit.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / "platform_audit.json", result)
    print()
    print(f"report: {output_dir / 'platform_audit.md'}")
    # Fails the build: the point of an audit is that it can say no.
    return 0 if result["passed"] else 1


def cmd_sizing(args: argparse.Namespace) -> int:
    from retailgr.serving.sizing import measure, render_sizing

    cfg = _load_config(args)
    result = measure(
        args.bundle or (cfg.artifacts_dir / "bundle"),
        config_path=args.config or "configs/pipeline.yaml",
        requests=args.requests,
        repeats=args.repeats,
    )
    report = render_sizing(result)
    print(report)

    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sizing.md").write_text(report, encoding="utf-8")
    privacy.write_json(output_dir / "sizing.json", result)
    print()
    print(f"report: {output_dir / 'sizing.md'}")
    return 0 if "error" not in result else 1


def cmd_experiment(args: argparse.Namespace) -> int:
    from retailgr.experiment import render_report, run_experiment, save_run

    cfg = _load_config(args)
    record = run_experiment(
        cfg,
        variants=args.variants,
        model_configs=args.model_configs,
        reuse_pipeline=args.reuse_pipeline,
    )
    output_dir = Path(args.output) if args.output else cfg.artifacts_dir
    paths = save_run(record, output_dir)
    print(render_report(record))
    print()
    print(f"report: {paths['report']}")
    print(f"run record: {paths['json']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retailgr", description="RetailGR Stage 1 pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-data", help="write the synthetic dataset")
    _add_common(generate)
    generate.add_argument("--users", type=int, default=None)
    generate.add_argument("--products", type=int, default=None)
    generate.set_defaults(func=cmd_generate_data)

    ingest_parser = subparsers.add_parser("ingest", help="raw dataset -> bronze tables")
    _add_common(ingest_parser)
    ingest_parser.set_defaults(func=cmd_ingest)

    silver_parser = subparsers.add_parser("silver", help="bronze -> silver tables")
    _add_common(silver_parser)
    silver_parser.set_defaults(func=cmd_silver)

    sequences_parser = subparsers.add_parser("sequences", help="silver -> gold sequences")
    _add_common(sequences_parser)
    sequences_parser.add_argument(
        "--granularity",
        default="config",
        help="config | sku | style_color | product",
    )
    sequences_parser.add_argument("--variant", default=None, help="name for the output tables")
    sequences_parser.set_defaults(func=cmd_sequences)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="train the models for one variant and score them"
    )
    _add_common(evaluate_parser)
    evaluate_parser.add_argument("--variant", default="config")
    evaluate_parser.add_argument(
        "--model-configs",
        nargs="*",
        default=list(DEFAULT_MODEL_CONFIGS),
        help=f"default: {' '.join(DEFAULT_MODEL_CONFIGS)}",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    export_parser = subparsers.add_parser(
        "export-model", help="train on the gold tables and write a serving bundle"
    )
    _add_common(export_parser)
    export_parser.add_argument("--variant", default="config")
    export_parser.add_argument("--model-config", default="hstu_small.yaml")
    export_parser.add_argument("--ranker-config", default="ranker_small.yaml")
    export_parser.add_argument(
        "--no-ranker",
        action="store_true",
        help="export retrieval only; the API then serves retrieval order",
    )
    export_parser.add_argument(
        "--skip-ranker-eval",
        action="store_true",
        help="skip head AUC and the re-ranking comparison (they cost a few minutes)",
    )
    export_parser.add_argument(
        "--force-ranker",
        action="store_true",
        help=(
            "wire the ranker in even if it orders candidates worse than "
            "retrieval; the gate exists to stop exactly that"
        ),
    )
    export_parser.add_argument("--model-version", default=None)
    export_parser.add_argument("--output", default=None, help="bundle directory")
    export_parser.set_defaults(func=cmd_export_model)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="replay silver through the broker and materialise the online store",
    )
    _add_common(bootstrap_parser)
    bootstrap_parser.add_argument("--variant", default="config")
    bootstrap_parser.add_argument(
        "--limit", type=int, default=None, help="replay only the first N events"
    )
    bootstrap_parser.add_argument(
        "--speed-factor",
        type=float,
        default=None,
        help="wall-clock pacing: 3600 means one hour of event time per second",
    )
    bootstrap_parser.set_defaults(func=cmd_bootstrap)

    serve_parser = subparsers.add_parser("serve", help="run the recommendations API")
    _add_common(serve_parser)
    serve_parser.add_argument("--bundle", default=None)
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--variant", default="config")
    serve_parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="fill the online store before serving (needed for the in-memory store)",
    )
    serve_parser.add_argument("--limit", type=int, default=None, help="events to replay")
    serve_parser.set_defaults(func=cmd_serve)

    bench_parser = subparsers.add_parser(
        "bench", help="measure the serving latency budget, stage by stage"
    )
    _add_common(bench_parser)
    bench_parser.add_argument("--bundle", default=None)
    bench_parser.add_argument("--variant", default="config")
    bench_parser.add_argument("--requests", type=int, default=500)
    bench_parser.add_argument("--warmup", type=int, default=25)
    bench_parser.add_argument("--limit", type=int, default=20)
    bench_parser.add_argument("--output", default=None)
    bench_parser.set_defaults(func=cmd_bench)

    ablate_parser = subparsers.add_parser(
        "ablate", help="switch off HSTU components one at a time and compare"
    )
    _add_common(ablate_parser)
    ablate_parser.add_argument("--variant", default="config")
    ablate_parser.add_argument("--model-config", default="hstu_small.yaml")
    ablate_parser.add_argument("--baseline-config", default="sasrec_small.yaml")
    ablate_parser.add_argument(
        "--no-baseline", action="store_true", help="skip the SASRec reference row"
    )
    ablate_parser.add_argument(
        "--ablations",
        nargs="*",
        default=None,
        help="run only these ablations; default is all of them",
    )
    ablate_parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "train each configuration once per seed and report the spread. "
            "Without this, a small difference cannot be told from training noise"
        ),
    )
    ablate_parser.add_argument(
        "--confidence", type=float, default=0.95, help="bootstrap confidence level"
    )
    ablate_parser.add_argument(
        "--resamples", type=int, default=2000, help="bootstrap resamples"
    )
    ablate_parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=None,
        help="train every arm this many epochs with early stopping off, as every "
        "ablation here did before validation was read; for comparing schedules",
    )
    ablate_parser.add_argument("--output", default=None, help="where to write the report")
    ablate_parser.set_defaults(func=cmd_ablate)

    rank_parser = subparsers.add_parser(
        "rank-experiment",
        help="compare negative-sampling mixes for the ranker on slate resolution",
    )
    _add_common(rank_parser)
    rank_parser.add_argument("--variant", default="config")
    rank_parser.add_argument("--model-config", default="hstu_small.yaml")
    rank_parser.add_argument("--ranker-config", default="ranker_small.yaml")
    rank_parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="run only these mixes; default is uniform, half_hard, all_hard",
    )
    rank_parser.add_argument(
        "--epochs", type=int, default=None, help="override the ranker's epochs"
    )
    rank_parser.add_argument(
        "--num-negatives", type=int, default=None, help="override negatives per positive"
    )
    rank_parser.add_argument(
        "--calibration-users",
        type=int,
        default=300,
        help="users per split for the slate calibration measurement",
    )
    rank_parser.add_argument(
        "--discrimination-users",
        type=int,
        default=400,
        help="users for the paired ordering comparison",
    )
    rank_parser.add_argument(
        "--confidence", type=float, default=0.95, help="bootstrap confidence level"
    )
    rank_parser.add_argument(
        "--resamples", type=int, default=2000, help="bootstrap resamples"
    )
    rank_parser.add_argument("--name", default=None, help="report file stem")
    rank_parser.add_argument("--output", default=None, help="where to write the report")
    rank_parser.set_defaults(func=cmd_rank_experiment)

    slate_parser = subparsers.add_parser(
        "slate-experiment",
        help="can the ranker order retrieval's own candidates? arms against a control",
    )
    _add_common(slate_parser)
    slate_parser.add_argument("--variant", default="config")
    slate_parser.add_argument("--model-config", default="hstu_small.yaml")
    slate_parser.add_argument("--ranker-config", default="ranker_small.yaml")
    slate_parser.add_argument(
        "--arms", nargs="*", default=None,
        help="subset of: control unseen no_history both",
    )
    slate_parser.add_argument("--seeds", nargs="*", type=int, default=[13, 17, 23])
    slate_parser.add_argument("--users", type=int, default=400)
    slate_parser.add_argument("--output", default=None)
    slate_parser.set_defaults(func=cmd_slate_experiment)

    convergence_parser = subparsers.add_parser(
        "convergence",
        help="train with and without early stopping; report what the fixed schedule cost",
    )
    _add_common(convergence_parser)
    convergence_parser.add_argument("--variant", default="config")
    convergence_parser.add_argument(
        "--model-configs", nargs="*", default=["sasrec_small.yaml", "hstu_small.yaml"]
    )
    convergence_parser.add_argument("--seeds", nargs="*", type=int, default=[13, 17, 23])
    convergence_parser.add_argument("--patience", type=int, default=10)
    convergence_parser.add_argument("--max-epochs", type=int, default=150)
    convergence_parser.add_argument("--fixed-epochs", type=int, default=8)
    convergence_parser.add_argument("--metric", default="ndcg@10")
    convergence_parser.add_argument("--output", default=None)
    convergence_parser.set_defaults(func=cmd_convergence)

    retention_parser = subparsers.add_parser(
        "retention", help="prune each layer past its window and expire old snapshots"
    )
    _add_common(retention_parser)
    retention_parser.add_argument(
        "--dry-run", action="store_true", help="report what would be pruned, change nothing"
    )
    retention_parser.add_argument("--output", default=None)
    retention_parser.set_defaults(func=cmd_retention)

    forget_parser = subparsers.add_parser(
        "forget",
        help="erase one customer from every store this code controls, and verify it",
    )
    _add_common(forget_parser)
    forget_parser.add_argument("--user-id", required=True)
    forget_parser.add_argument(
        "--dry-run", action="store_true", help="report what would be removed, change nothing"
    )
    forget_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="search every store for the identifier without deleting anything",
    )
    forget_parser.add_argument("--output", default=None)
    forget_parser.set_defaults(func=cmd_forget)

    consent_parser = subparsers.add_parser(
        "consent-cost",
        help="measure what honouring consent costs the model, at several opt-in rates",
    )
    _add_common(consent_parser)
    consent_parser.add_argument("--variant", default="config")
    consent_parser.add_argument("--model-config", default="hstu_small.yaml")
    consent_parser.add_argument(
        "--rates",
        nargs="*",
        type=float,
        default=None,
        help="opt-in rates to sweep; 1.0 is added if missing, as the control",
    )
    consent_parser.add_argument(
        "--seeds", nargs="*", type=int, default=[0, 1, 2],
        help="seeds per rate; one run per point plots initialisation noise",
    )
    consent_parser.add_argument(
        "--correlate",
        action="store_true",
        help="drop the most active users instead of a random sample, to show "
        "how much of the answer depends on *who* opts out",
    )
    consent_parser.add_argument("--output", default=None)
    consent_parser.set_defaults(func=cmd_consent_cost)

    audit_parser = subparsers.add_parser(
        "audit", help="check the portability claim: no vendor SDKs, no fixed endpoints"
    )
    audit_parser.add_argument(
        "--source", default="src/retailgr", help="package root to audit"
    )
    audit_parser.add_argument("--output", default=None, help="where to write the report")
    audit_parser.set_defaults(func=cmd_audit)

    sizing_parser = subparsers.add_parser(
        "sizing", help="measure what a serving pod needs: memory, cold start, throughput"
    )
    _add_common(sizing_parser)
    sizing_parser.add_argument("--bundle", default=None, help="bundle directory")
    sizing_parser.add_argument("--requests", type=int, default=200)
    sizing_parser.add_argument(
        "--repeats", type=int, default=3, help="fresh interpreters to median over"
    )
    sizing_parser.add_argument("--output", default=None, help="where to write the report")
    sizing_parser.set_defaults(func=cmd_sizing)

    experiment_parser = subparsers.add_parser(
        "experiment", help="run every granularity variant and write the report"
    )
    _add_common(experiment_parser)
    experiment_parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="default: sku style_color product config",
    )
    experiment_parser.add_argument(
        "--model-configs",
        nargs="*",
        default=list(DEFAULT_MODEL_CONFIGS),
        help=f"default: {' '.join(DEFAULT_MODEL_CONFIGS)}",
    )
    experiment_parser.add_argument("--output", default=None, help="where to write the report")
    experiment_parser.add_argument(
        "--reuse-pipeline",
        action="store_true",
        help="skip bronze/silver when the tables already exist",
    )
    experiment_parser.set_defaults(func=cmd_experiment)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Pin the thread budget before torch loads: on a small machine the
    # library defaults oversubscribe the cores and everything slows down.
    from retailgr.compute import set_thread_budget

    set_thread_budget(
        threads=getattr(args, "threads", None), serving=args.command in ("serve", "bench")
    )
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
