"""Enforce the retention windows, which until now enforced nothing.

`configs/pipeline.yaml` carried `privacy.retention.bronze_days: 30` with a
paragraph explaining why bronze is the shortest window — it is the only layer
holding raw, unfiltered, un-pseudonymised events — and **no code read it.**
Nor `silver_days`, `gold_days` or `snapshot_expiry_days`. Four settings that
read as a policy and were documentation of an intention, which is worse than
their absence: someone reading that file would reasonably conclude bronze is
pruned at thirty days.

Three decisions here, each of which could have gone the easy way:

**The clock is the wall clock, and that is why this job refuses to run on the
sample data.** Retention means "older than N days from now", and the
alternative — measuring from the newest event in the table — is the reading
that makes a demo work and a production deployment quietly keep everything
for ever, because the newest event is always today. The synthetic dataset's
events are months old, so a 30-day bronze window covers all of them; the
guard below is what turns that into a refusal rather than an empty warehouse.
`clock: data` exists for exactly that case and has to be asked for.

**A pass that would delete almost everything refuses.** Removing 100% of a
table is far more likely to be a misconfiguration — wrong clock, days
confused with hours, a backfill of historical data — than an intended
policy. `max_delete_share` fails the job loudly instead. The same reasoning
as the consent filter: both silent options are wrong, so neither is taken.

**Gold is pruned per user, not per row.** `gold.sequences_*` has no scalar
date: it has `input_ts`, one array per user. So the unit is a customer whose
*last* event has aged out, which is also the only reading that means anything
— half a sequence is not half a training example.

Snapshot expiry is not an optional extra step. In Iceberg a delete writes a
new snapshot and every earlier one stays readable by id, so a retention pass
without expiry moves rows out of the current view and nowhere else. That is
the same trap `erasure.py` was built around, and it applies here for the same
reason.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from retailgr.config import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

__all__ = ["DATED_TABLES", "render", "run"]

# (table, date column, config key for the window)
DATED_TABLES: tuple[tuple[str, str, str], ...] = (
    ("bronze.interactions", "event_date", "bronze_days"),
    ("silver.interactions", "event_date", "silver_days"),
)

GOLD_VARIANTS: tuple[str, ...] = ("config", "product", "sku", "style_color")


class RetentionRefused(Exception):
    """Raised when a pass would delete more than the configured share.

    A distinct type so a scheduled run can be told apart from a crash: this
    is the job working correctly and declining, not failing.
    """


def _cutoff(cfg: Config, spark: SparkSession, days: int) -> tuple[datetime, str]:
    """The date before which rows are past their window, and which clock said so."""
    clock = str(cfg.get("privacy.retention.clock", "wall")).lower()
    if clock == "wall":
        return datetime.now(timezone.utc) - timedelta(days=days), "wall"
    if clock != "data":
        raise ValueError(
            f"privacy.retention.clock must be 'wall' or 'data', got {clock!r}"
        )

    # Measured from the newest event in bronze. Only defensible for a fixed
    # sample dataset, and it has to be asked for, because in production it
    # silently retains everything for ever — the newest event is always now.
    from pyspark.sql import functions as F

    from retailgr.io.tables import read_table

    newest = (
        read_table(spark, cfg, "bronze.interactions")
        .select(F.max("event_ts").alias("newest"))
        .first()
    )
    if newest is None or newest["newest"] is None:
        return datetime.now(timezone.utc) - timedelta(days=days), "wall (bronze empty)"
    latest = newest["newest"]
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest - timedelta(days=days), "data"


def _prune_dated(
    spark: SparkSession,
    cfg: Config,
    table: str,
    column: str,
    cutoff: datetime,
    max_share: float,
    dry_run: bool,
) -> dict[str, Any]:
    from pyspark.sql import functions as F

    from retailgr.io.tables import _table_location, read_table, write_table

    step: dict[str, Any] = {"table": table, "cutoff": cutoff.date().isoformat()}
    if cfg.warehouse_backend != "iceberg" and not Path(_table_location(cfg, table)).exists():
        step["status"] = "absent"
        return step

    try:
        rows = read_table(spark, cfg, table)
    except Exception:  # pragma: no cover - table missing in this backend
        step["status"] = "absent"
        return step
    if column not in rows.columns:
        step["status"] = f"no {column} column"
        return step

    total = rows.count()
    keep = rows.filter(F.col(column) >= F.lit(cutoff.date()))
    kept = keep.count()
    removed = total - kept
    step.update({"rows_before": total, "rows_removed": removed, "rows_after": kept})
    if total == 0:
        step["status"] = "empty"
        return step

    share = removed / total
    step["delete_share"] = round(share, 4)
    if share > max_share:
        raise RetentionRefused(
            f"{table}: the window would remove {removed}/{total} rows "
            f"({share:.1%}), over the {max_share:.0%} limit. That is far more "
            "likely to be a misconfiguration than a policy — check "
            "privacy.retention.clock (the sample dataset's events are months "
            "old, so a wall-clock window covers all of them) and that the "
            "window is in days. Raise privacy.retention.max_delete_share to "
            "proceed deliberately."
        )
    if dry_run:
        step["status"] = "would prune"
        return step
    if removed == 0:
        step["status"] = "nothing to prune"
        return step

    # Materialise before the overwrite: the write truncates the directory the
    # reader is still streaming from.
    keep.cache()
    keep.count()
    write_table(spark, cfg, table, keep, mode="overwrite")
    keep.unpersist()
    step["status"] = "pruned"
    return step


def _prune_sequences(
    spark: SparkSession,
    cfg: Config,
    variant: str,
    cutoff: datetime,
    max_share: float,
    dry_run: bool,
) -> dict[str, Any]:
    """Drop users whose last event has aged out.

    Per user because `gold.sequences_*` has no scalar date — `input_ts` is one
    array per row — and because a partially-pruned sequence is not a smaller
    training example, it is a corrupted one.
    """
    from pyspark.sql import functions as F

    from retailgr.io.tables import _table_location, read_table, write_table

    table = f"gold.sequences_{variant}"
    step: dict[str, Any] = {"table": table, "cutoff": cutoff.date().isoformat()}
    if cfg.warehouse_backend != "iceberg" and not Path(_table_location(cfg, table)).exists():
        step["status"] = "absent"
        return step
    try:
        rows = read_table(spark, cfg, table)
    except Exception:  # pragma: no cover
        step["status"] = "absent"
        return step
    if "input_ts" not in rows.columns:
        step["status"] = "no input_ts column"
        return step

    boundary = int(cutoff.timestamp())
    last_event = F.array_max(F.col("input_ts"))
    total = rows.count()
    keep = rows.filter(last_event >= F.lit(boundary))
    kept = keep.count()
    removed = total - kept
    step.update({"users_before": total, "users_removed": removed, "users_after": kept})
    if total == 0:
        step["status"] = "empty"
        return step

    share = removed / total
    step["delete_share"] = round(share, 4)
    if share > max_share:
        raise RetentionRefused(
            f"{table}: the window would remove {removed}/{total} users "
            f"({share:.1%}), over the {max_share:.0%} limit."
        )
    if dry_run:
        step["status"] = "would prune"
        return step
    if removed == 0:
        step["status"] = "nothing to prune"
        return step

    keep.cache()
    keep.count()
    write_table(spark, cfg, table, keep, mode="overwrite", partition_by=["split"])
    keep.unpersist()
    step["status"] = "pruned"
    return step


def _expire_snapshots(cfg: Config, tables: list[str], days: int, dry_run: bool) -> dict[str, Any]:
    """Drop superseded Iceberg snapshots older than the window.

    Without this the pruning above is cosmetic: the removed rows stay
    readable through any earlier snapshot, which this repository's own
    `read_arrow(..., snapshot_id=...)` will happily return.
    """
    step: dict[str, Any] = {"step": "expire_snapshots", "days": days, "tables": {}}
    if cfg.warehouse_backend != "iceberg":
        step["status"] = "not applicable: parquet rewrites in place"
        return step

    from retailgr.io import iceberg as iceberg_io

    catalog = iceberg_io.catalog_for(cfg)
    older_than = datetime.now(timezone.utc) - timedelta(days=days)
    for table in tables:
        if not iceberg_io.table_exists(cfg, table):
            continue
        namespace, name = iceberg_io._namespace_and_name(table)
        handle = catalog.load_table(f"{namespace}.{name}")
        before = len(handle.metadata.snapshots)
        if dry_run:
            step["tables"][table] = {"snapshots": before, "status": "would expire"}
            continue
        handle.maintenance.expire_snapshots().older_than(older_than).commit()
        handle.refresh()
        step["tables"][table] = {
            "snapshots_before": before,
            "snapshots_after": len(handle.metadata.snapshots),
        }
    step.setdefault("status", "expired")
    return step


def run(spark: SparkSession, cfg: Config, dry_run: bool = False) -> dict[str, Any]:
    """Prune every layer past its window, then expire the snapshots."""
    started = time.time()
    settings = cfg.get("privacy.retention", {}) or {}
    max_share = float(settings.get("max_delete_share", 0.5))

    report: dict[str, Any] = {"dry_run": dry_run, "max_delete_share": max_share, "steps": []}
    refused: str | None = None

    try:
        for table, column, key in DATED_TABLES:
            days = int(settings.get(key, 0) or 0)
            if days <= 0:
                report["steps"].append({"table": table, "status": "no window configured"})
                continue
            cutoff, clock = _cutoff(cfg, spark, days)
            report.setdefault("clock", clock)
            step = _prune_dated(spark, cfg, table, column, cutoff, max_share, dry_run)
            step["window_days"] = days
            report["steps"].append(step)

        gold_days = int(settings.get("gold_days", 0) or 0)
        if gold_days > 0:
            cutoff, clock = _cutoff(cfg, spark, gold_days)
            report.setdefault("clock", clock)
            for variant in GOLD_VARIANTS:
                step = _prune_sequences(spark, cfg, variant, cutoff, max_share, dry_run)
                step["window_days"] = gold_days
                report["steps"].append(step)
    except RetentionRefused as error:
        # Not a crash. The job looked at what the window covers and declined,
        # which is the only honest answer when the alternative is emptying a
        # table on a configuration nobody re-read.
        refused = str(error)

    report["refused"] = refused
    if refused is None:
        expiry_days = int(settings.get("snapshot_expiry_days", 0) or 0)
        if expiry_days > 0:
            tables = [t for t, _, _ in DATED_TABLES] + [
                f"gold.sequences_{v}" for v in GOLD_VARIANTS
            ]
            report["steps"].append(_expire_snapshots(cfg, tables, expiry_days, dry_run))

    report["seconds"] = round(time.time() - started, 2)
    return report


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Retention",
        "",
        f"- Mode: {'dry run' if report['dry_run'] else 'applied'}",
        f"- Clock: **{report.get('clock', 'n/a')}**",
        f"- Refuses above: {report['max_delete_share']:.0%} of a table",
        f"- Duration: {report['seconds']:.1f}s",
        "",
        "| Layer | Window | Status | Removed | Share |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for step in report["steps"]:
        if step.get("step") == "expire_snapshots":
            continue
        removed = step.get("rows_removed", step.get("users_removed", "-"))
        share = step.get("delete_share")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{step.get('table', step.get('step'))}`",
                    f"{step.get('window_days', '-')}d",
                    str(step.get("status", "-")),
                    str(removed),
                    "-" if share is None else f"{share:.1%}",
                ]
            )
            + " |"
        )

    if report.get("refused"):
        lines += [
            "",
            "## Refused",
            "",
            report["refused"],
            "",
            "This is the job declining, not failing. A pass that removes almost "
            "everything is far more likely to be a misconfiguration than a "
            "policy, and an empty warehouse is a much more expensive way to "
            "find that out.",
        ]

    expiry = next((s for s in report["steps"] if s.get("step") == "expire_snapshots"), None)
    if expiry:
        lines += ["", "## Snapshot expiry", "", f"- {expiry.get('status')}"]
        for table, detail in (expiry.get("tables") or {}).items():
            lines.append(f"- `{table}`: {detail}")
        lines += [
            "",
            "Without this the pruning above is cosmetic: in Iceberg a delete "
            "writes a new snapshot and every earlier one stays readable by id.",
        ]
    return "\n".join(lines) + "\n"
