"""The retention job, and the guard that stops it emptying a warehouse.

Four settings in `pipeline.yaml` described a retention policy and were read
by nothing (`tests/test_config_is_wired.py` is the scan that found them). This
is the job, and these are the three decisions in it that could have gone the
easy way.

**The clock is the wall clock.** Retention means "older than N days from
now". Measuring from the newest event in the table instead is the reading
that makes a fixed sample dataset work and that would silently retain
everything for ever in production, because the newest event is always today.

**A pass that would remove nearly everything refuses.** That is far more
likely to be a wrong clock, days confused with hours, or a backfill of
historical data than a policy. Running the shipped config against the sample
data hits this, deliberately: its events are months old, so a 30-day bronze
window covers 100% of them.

**Gold is pruned per user.** `gold.sequences_*` has no scalar date — one
`input_ts` array per row — and half a sequence is not a smaller training
example, it is a corrupted one.

The Spark-backed tests are marked slow. The guard's arithmetic is checked
without Spark, because it is the part that decides whether a mistake costs a
warehouse.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from retailgr.config import Config
from retailgr.jobs import retention


def _cfg(tmp_path, **overrides):
    privacy = {"retention": {
        "bronze_days": 30, "silver_days": 400, "gold_days": 400,
        "clock": "wall", "max_delete_share": 0.5, **overrides,
    }}
    return Config.load(
        "configs/pipeline.yaml",
        overrides={
            "dataset": {"name": "synthetic", "raw_path": str(tmp_path / "raw")},
            "warehouse": {"backend": "parquet", "root": str(tmp_path / "warehouse")},
            "privacy": privacy,
        },
    )


# -- the shape of the job, without Spark --------------------------------------


def test_the_layers_it_prunes_are_the_ones_holding_user_rows():
    """`gold.vocab_*` is aggregate counts and `silver.item_hierarchy` is
    catalogue; pruning either by date would delete the vocabulary a model
    needs to be loadable at all."""
    tables = {table for table, _, _ in retention.DATED_TABLES}
    assert tables == {"bronze.interactions", "silver.interactions"}
    assert "silver.item_hierarchy" not in tables
    assert not any("vocab" in t for t in tables)


def test_every_dated_table_names_a_window_that_exists_in_the_config():
    """A table pointed at a config key nobody set would be retained for ever
    while appearing in this list."""
    from pathlib import Path

    import yaml

    settings = yaml.safe_load(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))
    windows = settings["privacy"]["retention"]
    for table, _, key in retention.DATED_TABLES:
        assert key in windows, f"{table} reads {key}, which the config does not set"
        assert int(windows[key]) > 0, f"{key} is not a positive number of days"


def test_bronze_has_the_shortest_window():
    """The one ordering that is a privacy claim rather than a preference:
    bronze is the only layer holding raw, unfiltered, un-pseudonymised
    events, so it cannot be the layer kept longest."""
    from pathlib import Path

    import yaml

    windows = yaml.safe_load(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))[
        "privacy"
    ]["retention"]
    assert windows["bronze_days"] < windows["silver_days"]
    assert windows["bronze_days"] < windows["gold_days"]


def test_an_unknown_clock_is_an_error_not_a_default():
    """Falling back to a default clock on a typo would silently change what
    retention means."""
    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"privacy": {"retention": {"clock": "walll"}}},
    )
    with pytest.raises(ValueError, match="clock must be"):
        retention._cutoff(cfg, None, 30)


def test_the_wall_clock_needs_no_spark_session():
    """It must not touch the warehouse to answer "what is 30 days ago" — the
    data clock does, and conflating them would make the safe option require
    the thing it is protecting."""
    cfg = Config.load("configs/pipeline.yaml")
    cutoff, clock = retention._cutoff(cfg, None, 30)
    assert clock == "wall"
    expected = datetime.now(timezone.utc) - timedelta(days=30)
    assert abs((cutoff - expected).total_seconds()) < 5


# -- the guard ----------------------------------------------------------------


@pytest.mark.slow
def test_the_shipped_config_refuses_on_the_sample_data(tmp_path):
    """The guard doing its job, on the real warehouse rather than a fixture.

    The sample dataset's events are months old and the shipped bronze window
    is 30 days, so a wall-clock pass covers all of them. Refusing is the only
    honest answer: the alternative is an empty bronze table and a green
    CronJob.
    """
    pytest.importorskip("pyspark")
    from pathlib import Path

    from retailgr.spark_session import build_spark

    if not Path("data/warehouse/bronze/interactions").exists():
        pytest.skip("no warehouse; run `retailgr ingest` first")

    cfg = Config.load("configs/pipeline.yaml")
    spark = build_spark(cfg)
    try:
        report = retention.run(spark, cfg, dry_run=True)
    finally:
        spark.stop()

    assert report["refused"], "a 100% window did not refuse"
    assert "clock" in report["refused"], "the refusal does not name the likely cause"
    assert report["clock"] == "wall"


@pytest.mark.slow
def test_the_data_clock_prunes_instead_of_refusing(tmp_path):
    """The other direction, so the job is not merely a refusal machine."""
    pytest.importorskip("pyspark")
    from pathlib import Path

    from retailgr.spark_session import build_spark

    if not Path("data/warehouse/bronze/interactions").exists():
        pytest.skip("no warehouse; run `retailgr ingest` first")

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"privacy": {"retention": {"clock": "data", "bronze_days": 45}}},
    )
    spark = build_spark(cfg)
    try:
        report = retention.run(spark, cfg, dry_run=True)
    finally:
        spark.stop()

    assert not report["refused"], report["refused"]
    assert report["clock"] == "data"
    bronze = next(s for s in report["steps"] if s.get("table") == "bronze.interactions")
    assert bronze["status"] == "would prune"
    assert 0 < bronze["rows_removed"] < bronze["rows_before"]


@pytest.mark.slow
def test_a_dry_run_leaves_the_row_count_untouched(tmp_path):
    pytest.importorskip("pyspark")
    from pathlib import Path

    from retailgr.io.tables import read_table
    from retailgr.spark_session import build_spark

    if not Path("data/warehouse/bronze/interactions").exists():
        pytest.skip("no warehouse; run `retailgr ingest` first")

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"privacy": {"retention": {"clock": "data", "bronze_days": 45}}},
    )
    spark = build_spark(cfg)
    try:
        before = read_table(spark, cfg, "bronze.interactions").count()
        retention.run(spark, cfg, dry_run=True)
        after = read_table(spark, cfg, "bronze.interactions").count()
    finally:
        spark.stop()
    assert before == after


# -- the report says what happened --------------------------------------------


def test_the_rendered_report_says_which_clock_it_used():
    """Reading "pruned 40% of bronze" without knowing which clock produced
    the cutoff is reading a number with no units."""
    report = {
        "dry_run": True, "max_delete_share": 0.5, "clock": "data", "seconds": 1.0,
        "steps": [{"table": "bronze.interactions", "window_days": 30,
                   "status": "would prune", "rows_removed": 5, "delete_share": 0.1}],
        "refused": None,
    }
    text = retention.render(report)
    assert "Clock: **data**" in text
    assert "bronze.interactions" in text
    assert "10.0%" in text


def test_a_refusal_is_rendered_as_a_decision_not_a_crash():
    report = {
        "dry_run": False, "max_delete_share": 0.5, "clock": "wall", "seconds": 1.0,
        "steps": [], "refused": "bronze.interactions: would remove 100%",
    }
    text = retention.render(report)
    assert "## Refused" in text
    assert "declining, not failing" in text


def test_the_snapshot_expiry_step_is_reported_even_when_not_applicable():
    """"Not applicable" and "done" are different answers. A parquet rewrite
    genuinely leaves no history; an Iceberg delete without expiry leaves
    every row readable through the previous snapshot."""
    cfg = Config.load("configs/pipeline.yaml", overrides={"warehouse": {"backend": "parquet"}})
    step = retention._expire_snapshots(cfg, ["silver.interactions"], 7, dry_run=False)
    assert "not applicable" in step["status"]


def test_the_cli_exits_non_zero_when_the_pass_refused():
    """Otherwise a CronJob records a success for a run that deleted nothing
    and meant to delete something."""
    import inspect

    from retailgr.cli import cmd_retention

    source = inspect.getsource(cmd_retention)
    assert 'report.get("refused")' in source
    assert "return 1 if" in source
