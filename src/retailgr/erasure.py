"""Deleting a customer, and proving it happened.

Most "right to erasure" implementations in a data platform are a `DELETE`
against one table and a ticket marked done. This one is built around the
observation that in a lakehouse that is usually **worse than doing nothing**,
because it produces a confident record of a deletion that did not occur.
Three specific ways that happens here, all of them found by reading this
repository rather than in the abstract:

**1. Iceberg keeps every prior snapshot readable.** `Table.delete()` writes a
new snapshot without the rows. The old snapshot still exists, still resolves
by id, and `read_arrow(..., snapshot_id=...)` in this very codebase will hand
it back. A delete without snapshot expiry removes the customer from the
current view of the table and from nowhere else.

**2. The pipeline re-derives everything from `data/raw/`.** Delete from
bronze, silver and gold, then run `retailgr ingest` — and the customer is
back, because the CSV the whole warehouse is built from was never touched.
Erasure that the next scheduled job undoes is not erasure.

**3. The online store is rebuilt from silver.** `retailgr bootstrap` replays
silver into Redis, so clearing the cache first and the lakehouse second
leaves a window in which a replay restores the cache. The order is forced:
source, then lakehouse, then cache. Never the reverse.

So the deliverable here is not `forget`. It is `verify`, which searches every
store for the literal identifier afterwards and reports where it survived.
`forget` without `verify` is a claim; together they are a measurement.

**What cannot be erased, stated plainly rather than omitted:**

- *Kafka.* A log is append-only. `interactions.v1` is keyed by `user_id`, so
  a compacted topic could take a tombstone — but it is configured
  `cleanup_policy: delete`, and `recs.served.v1` carries `user_id` in the
  value while being keyed by `request_id`, so no key-targeted delete can
  reach it at all. Both rely on the 7-day retention window, and during that
  window the data is still there. That is a real limitation with a real
  bound, and the bound is what makes it defensible.
- *The trained model.* Erasure removes the training data, not the parameters
  fitted to it. The encoder has an item embedding table and no per-user
  vectors, so nothing here is *about* one customer — but their behaviour
  contributed to the weights and removing that contribution is machine
  unlearning, which is a research area and not a function call. What this
  system can honestly say is that the model is retrained from data the
  customer is no longer in, so their influence ends at the next export. It
  cannot say the current model has forgotten them.
- *Backups and object-store versioning*, if the deployment has them. Out of
  scope for this code and in scope for whoever operates it; naming it here
  is better than leaving it to be discovered during an audit.
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from retailgr import privacy
from retailgr.config import Config

__all__ = [
    "ErasureReport",
    "UNERASABLE",
    "erase",
    "target_identifiers",
    "verify",
]

# Tables that hold per-user rows, in the order they must be cleared. Each
# entry is (table, column). `gold.vocab_*` is aggregate counts and
# `silver.item_hierarchy` is catalogue, so neither is here.
USER_TABLES: tuple[tuple[str, str], ...] = (
    ("bronze.interactions", "user_id"),
    ("silver.interactions", "user_id"),
)

GOLD_VARIANTS: tuple[str, ...] = ("config", "product", "sku", "style_color")

# Stated in the report every time, because a report that lists only what it
# did invites the reader to assume the rest was nothing.
UNERASABLE: tuple[dict[str, str], ...] = (
    {
        "store": "kafka",
        "why": (
            "An append-only log. interactions.v1 is keyed by user_id but is "
            "cleanup_policy=delete, not compact, so a tombstone would not "
            "remove anything; recs.served.v1 carries user_id in the value "
            "while being keyed by request_id, so no key-targeted delete can "
            "reach it."
        ),
        "bound": "Both topics have retention_ms of 7 days.",
    },
    {
        "store": "model_weights",
        "why": (
            "Erasure removes training data, not the parameters fitted to it. "
            "The encoder holds item embeddings and no per-user vectors, so "
            "nothing in the weights is about one customer — but their "
            "behaviour shaped them, and removing that contribution is "
            "machine unlearning, not a delete."
        ),
        "bound": (
            "The next export trains on data they are no longer in, so their "
            "influence ends at that bundle. The currently-serving bundle "
            "still carries it."
        ),
    },
    {
        "store": "backups",
        "why": (
            "Whatever the deployment keeps outside this code: object-store "
            "versioning, filesystem snapshots, a nightly dump."
        ),
        "bound": "Unknown to this code. Named so it is not assumed to be nothing.",
    },
)


@dataclass
class ErasureReport:
    """What was removed, from where, and what survived."""

    user_id: str
    identifiers: list[str]
    dry_run: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    unerasable: list[dict[str, str]] = field(default_factory=lambda: list(UNERASABLE))
    seconds: float = 0.0

    @property
    def complete(self) -> bool:
        """True only when the verification pass found nothing anywhere."""
        return bool(self.verification) and not self.verification.get("survived")

    def subject_digest(self, key: bytes | None = None) -> str:
        """A handle for the erased subject that is not their identifier.

        The first version of this wrote `user_id` straight into the receipt,
        and `tests/test_privacy.py` caught it: an erasure that ends by
        creating a new permanent file naming the person who asked to be
        forgotten has not entirely understood the request.

        A digest keeps receipts distinguishable — an auditor can match this
        one to the ticket that produced it — without the file being a record
        of the person. **Unkeyed it is a correlation handle and not a
        protection**: the identifier space here is a million candidates and
        a laptop reverses a bare SHA-256 over it in seconds. When a
        pseudonymisation key is configured this uses it, and then it is
        genuinely unsearchable without the key. Saying which of the two you
        have is the difference between a safeguard and the appearance of one.
        """
        if key is not None:
            return privacy.pseudonymise(self.user_id, key, length=16)
        return hashlib.sha256(self.user_id.encode("utf-8")).hexdigest()[:16]

    def as_dict(self, key: bytes | None = None) -> dict[str, Any]:
        return {
            "subject_digest": self.subject_digest(key),
            "subject_digest_keyed": key is not None,
            "identifiers_targeted": len(self.identifiers),
            "dry_run": self.dry_run,
            "complete": self.complete,
            "steps": self.steps,
            "verification": self.verification,
            "unerasable": self.unerasable,
            "seconds": round(self.seconds, 2),
        }


def target_identifiers(cfg: Config, user_id: str) -> list[str]:
    """Every string that identifies this customer in some layer.

    Silver and gold hold a pseudonym when pseudonymisation is on, and bronze
    and the raw file hold the original. Deleting only one of them looks like
    a complete erasure from whichever layer was checked. The pseudonym is
    recomputed from the id rather than looked up, which is exactly why
    :func:`privacy.pseudonymise` is deterministic.
    """
    identifiers = [user_id]
    settings = cfg.get("privacy.pseudonymisation", {}) or {}
    if bool(settings.get("enabled", False)):
        key = privacy.pseudonymisation_key(settings.get("key"))
        identifiers.append(
            privacy.pseudonymise(user_id, key, length=int(settings.get("length", 32)))
        )
    return identifiers


# -- the raw file -------------------------------------------------------------


def _erase_from_raw(cfg: Config, identifiers: list[str], dry_run: bool) -> dict[str, Any]:
    """Rewrite the source CSV without this customer.

    First, and not optional. Every table below is derived from this file, so
    an erasure that skips it is undone by the next `retailgr ingest` — and
    it will be undone silently, by a scheduled job, long after the ticket
    was closed.
    """
    path = cfg.raw_path / cfg.dataset_name / "interactions.csv"
    step: dict[str, Any] = {"step": "raw", "path": str(path)}
    if not path.exists():
        step["status"] = "absent"
        return step

    targets = set(identifiers)
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "user_id" not in fieldnames:
            step["status"] = "no user_id column"
            return step
        kept = [row for row in reader if row.get("user_id") not in targets]

    total = sum(1 for _ in open(path, encoding="utf-8")) - 1
    step["rows_before"] = total
    step["rows_removed"] = total - len(kept)
    if dry_run:
        step["status"] = "would rewrite"
        return step

    # Write beside the original and replace atomically: a crash halfway
    # through an erasure must not leave a truncated source of truth.
    temporary = path.with_suffix(".csv.erasing")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)
    step["status"] = "rewritten"
    return step


# -- the lakehouse ------------------------------------------------------------


def _tables_to_clear(cfg: Config) -> list[tuple[str, str]]:
    tables = list(USER_TABLES)
    for variant in GOLD_VARIANTS:
        tables.append((f"gold.sequences_{variant}", "user_id"))
    return tables


def _erase_from_iceberg(
    cfg: Config, table: str, column: str, identifiers: list[str], dry_run: bool
) -> dict[str, Any]:
    from retailgr.io import iceberg as iceberg_io

    step: dict[str, Any] = {"step": table, "backend": "iceberg"}
    if not iceberg_io.table_exists(cfg, table):
        step["status"] = "absent"
        return step

    catalog = iceberg_io.catalog_for(cfg)
    namespace, name = iceberg_io._namespace_and_name(table)
    handle = catalog.load_table(f"{namespace}.{name}")
    quoted = ", ".join(f"'{identifier}'" for identifier in identifiers)
    predicate = f"{column} in ({quoted})"

    before = len(handle.scan(row_filter=predicate).to_arrow())
    step["rows_matched"] = before
    if dry_run:
        step["status"] = "would delete"
        return step

    if before:
        handle.delete(delete_filter=predicate)
    step["status"] = "deleted"
    step["snapshots_after_delete"] = len(handle.metadata.snapshots)
    return step


def _erase_from_parquet(
    cfg: Config, spark: Any, table: str, column: str, identifiers: list[str], dry_run: bool
) -> dict[str, Any]:
    from pyspark.sql import functions as F

    from retailgr.io.tables import TABLE_PARTITIONS, _table_location, read_table, write_table

    step: dict[str, Any] = {"step": table, "backend": "parquet"}
    if not Path(_table_location(cfg, table)).exists():
        step["status"] = "absent"
        return step

    events = read_table(spark, cfg, table)
    if column not in events.columns:
        step["status"] = f"no {column} column"
        return step

    matched = events.filter(F.col(column).isin(identifiers)).count()
    step["rows_matched"] = matched
    if dry_run:
        step["status"] = "would rewrite"
        return step

    kept = events.filter(~F.col(column).isin(identifiers))
    # The partition columns come from the write that created the table, not
    # from TABLE_PARTITIONS: that map has a `gold.sequences` key which no
    # real table name ever matches — the tables are `gold.sequences_config`
    # and friends — so looking partitions up there would silently
    # de-partition every gold table on the way through an erasure.
    partitions = [
        c for c in TABLE_PARTITIONS.get(table, ()) if c in kept.columns
    ] or _observed_partitions(cfg, table, kept.columns)
    # `.cache()` before the overwrite: the write truncates the directory the
    # reader is still streaming from, and an unmaterialised plan would find
    # its own input gone halfway through.
    kept.cache()
    kept.count()
    write_table(spark, cfg, table, kept, mode="overwrite", partition_by=partitions)
    kept.unpersist()
    step["status"] = "rewritten"
    step["rows_remaining"] = read_table(spark, cfg, table).count()
    return step


def _observed_partitions(cfg: Config, table: str, columns: list[str]) -> list[str]:
    """Partition columns read off the directory layout that exists.

    Parquet partitioning shows up as `column=value` directory names, so the
    table itself says how it was written. Asking it beats trusting a lookup
    table that has a key nothing matches.
    """
    from retailgr.io.tables import _table_location

    root = Path(_table_location(cfg, table))
    if not root.exists():
        return []
    for child in root.iterdir():
        if child.is_dir() and "=" in child.name:
            name = child.name.split("=", 1)[0]
            if name in columns:
                return [name]
    return []


def _expire_snapshots(cfg: Config, tables: list[str], dry_run: bool) -> dict[str, Any]:
    """Remove the snapshots that still contain the deleted rows.

    Without this the erasure is cosmetic. `Table.delete()` writes a new
    snapshot; every earlier one still resolves by id, and this repository's
    own `read_arrow(..., snapshot_id=...)` will read it back. A customer
    deleted from the current view and readable in the previous one has not
    been deleted.

    Expiry is by *age*, and the age is normally days — which means a fresh
    erasure leaves the rows readable until the window passes. For an actual
    erasure request that is not good enough, so `erase` passes an age of
    zero and takes every snapshot but the current one.
    """
    from retailgr.io import iceberg as iceberg_io

    step: dict[str, Any] = {"step": "expire_snapshots", "tables": {}}
    if cfg.warehouse_backend != "iceberg":
        step["status"] = "not applicable: parquet rewrites in place"
        return step

    catalog = iceberg_io.catalog_for(cfg)
    for table in tables:
        if not iceberg_io.table_exists(cfg, table):
            continue
        namespace, name = iceberg_io._namespace_and_name(table)
        handle = catalog.load_table(f"{namespace}.{name}")
        before = len(handle.metadata.snapshots)
        if dry_run:
            step["tables"][table] = {"snapshots": before, "status": "would expire"}
            continue
        # Everything strictly older than now, which is everything except the
        # snapshot the table currently points at.
        handle.maintenance.expire_snapshots().older_than(
            datetime.now(timezone.utc) + timedelta(seconds=1)
        ).commit()
        handle.refresh()
        step["tables"][table] = {
            "snapshots_before": before,
            "snapshots_after": len(handle.metadata.snapshots),
        }
    step.setdefault("status", "expired")
    return step


# -- the cache ----------------------------------------------------------------


def _erase_from_online_store(cfg: Config, user_id: str, dry_run: bool) -> dict[str, Any]:
    from retailgr.online_store import build_online_store

    step: dict[str, Any] = {"step": "online_store"}
    store = build_online_store(cfg)
    try:
        had_tail = bool(store.user_tail(user_id))
        step["had_tail"] = had_tail
        if dry_run:
            step["status"] = "would forget"
            return step
        step["keys_removed"] = store.forget(user_id)
        step["status"] = "forgotten"
    finally:
        store.close()
    return step


# -- verification -------------------------------------------------------------


def verify(cfg: Config, user_id: str, spark: Any = None) -> dict[str, Any]:
    """Search every store for the identifier and report where it survives.

    A **literal substring** search over the files, not a structured query.
    That is the point: a structured query asks the table whether it contains
    the row, using the same code path that just deleted it, and inherits
    every assumption that code makes. Reading the bytes asks a different
    question — is this string anywhere on disk — and it catches the cases
    the structured query cannot see, including a derived `session_id` built
    by string-concatenating the user id, and an Iceberg data file that is
    unreferenced by the current snapshot but still present.
    """
    identifiers = target_identifiers(cfg, user_id)
    survived: list[dict[str, Any]] = []
    checked: list[str] = []

    # 1. The raw source.
    raw = cfg.raw_path / cfg.dataset_name / "interactions.csv"
    checked.append(str(raw))
    if raw.exists():
        hits = _count_in_file(raw, identifiers)
        if hits:
            survived.append({"where": str(raw), "matches": hits})

    # 2. Every file under the warehouse. Parquet is binary but the string
    #    values are stored uncompressed often enough that a byte search
    #    finds them; when it does not, the structured check below covers it.
    warehouse = cfg.warehouse_root
    checked.append(str(warehouse))
    if warehouse.exists():
        for path in sorted(warehouse.rglob("*")):
            if not path.is_file():
                continue
            hits = _count_in_file(path, identifiers)
            if hits:
                survived.append({"where": str(path), "matches": hits})

    # 3. The structured question as well, because a byte search over a
    #    dictionary-encoded or compressed Parquet page can miss a value that
    #    a reader would still return.
    if spark is not None:
        from pyspark.sql import functions as F

        from retailgr.io.tables import _table_location, read_table

        for table, column in _tables_to_clear(cfg):
            if cfg.warehouse_backend != "iceberg" and not Path(
                _table_location(cfg, table)
            ).exists():
                continue
            checked.append(table)
            try:
                rows = read_table(spark, cfg, table)
            except Exception:  # pragma: no cover - table missing in this backend
                continue
            if column not in rows.columns:
                continue
            count = rows.filter(F.col(column).isin(identifiers)).count()
            if count:
                survived.append({"where": f"{table} (query)", "matches": count})

    # 4. The cache.
    from retailgr.online_store import build_online_store

    store = build_online_store(cfg)
    try:
        checked.append("online_store")
        remaining = len(store.user_tail(user_id)) + len(
            store.fallback(user_id) if _has_own_fallback(store, user_id) else []
        )
        if remaining:
            survived.append({"where": "online_store", "matches": remaining})
    finally:
        store.close()

    return {
        "identifiers_searched": len(identifiers),
        "checked": checked,
        "survived": survived,
        "clean": not survived,
    }


def _has_own_fallback(store: Any, user_id: str) -> bool:
    """Whether this user has a fallback of their own.

    `fallback()` falls through to the shared cold-start list, so asking it
    directly would report every erased user as still present — and the
    "erasure incomplete" would be the global list, which is about the
    catalogue and not about them.
    """
    fallbacks = getattr(store, "_fallbacks", None)
    if isinstance(fallbacks, dict):
        return user_id in fallbacks
    client = getattr(store, "client", None)
    if client is not None and hasattr(store, "_fallback_key"):
        return bool(client.exists(store._fallback_key(user_id)))
    return False


def _count_in_file(path: Path, identifiers: list[str]) -> int:
    """How many times any identifier appears in a file, by whatever route works.

    This started as a plain byte search and **the plain byte search was
    wrong**, in the direction that matters: Spark writes Parquet with Snappy
    by default, so the string `U000068` appears nowhere in the bytes of a
    file holding nineteen of that customer's rows. Searching 130 Parquet
    files for a user who is unambiguously in them returned zero hits.

    A verifier that reports "clean" for data that is present is worse than
    no verifier, because it converts an incomplete erasure into a signed
    statement that it was complete. So Parquet is decoded and its string
    columns are scanned for real, and the byte search is kept only for the
    formats where it is sound — CSV, JSON, Avro, logs, and any stray text.

    Reading the *files* rather than querying the tables is deliberate and
    survives here: a file on disk that the current snapshot no longer
    references is invisible to a query and entirely visible to this.
    """
    if path.suffix == ".parquet":
        return _count_in_parquet(path, identifiers)
    try:
        blob = path.read_bytes()
    except OSError:  # pragma: no cover - unreadable file
        return 0
    return sum(blob.count(identifier.encode("utf-8")) for identifier in identifiers)


def _count_in_parquet(path: Path, identifiers: list[str]) -> int:
    """Decode a Parquet file and count identifier occurrences in its columns.

    Every string-ish column, not only `user_id`: a derived `session_id` can
    embed the user id by construction — two of the dataset adapters build
    one by concatenating it — so a scan restricted to the column named
    `user_id` would certify an erasure that left the identifier sitting in
    the next column along.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency here
        return 0
    try:
        table = pq.read_table(path)
    except Exception:  # pragma: no cover - not a readable parquet file
        return 0

    wanted = set(identifiers)
    total = 0
    for name in table.schema.names:
        column = table.column(name)
        import pyarrow as pa

        if not pa.types.is_string(column.type) and not pa.types.is_large_string(column.type):
            continue
        for value in column.to_pylist():
            if value is None:
                continue
            if value in wanted:
                total += 1
            elif any(identifier in value for identifier in identifiers):
                # Substring, so a session_id built as `{user_id}#{date}`
                # counts. This is the one place a substring match is right:
                # the question is whether this exact customer survives, not
                # whether something looks like an identifier.
                total += 1
    return total


# -- the whole thing ----------------------------------------------------------


def erase(
    cfg: Config,
    user_id: str,
    *,
    dry_run: bool = False,
    spark: Any = None,
) -> ErasureReport:
    """Remove ``user_id`` from every store this code controls, then verify.

    The order is not a preference. Raw first, because every table is derived
    from it and the next ingest would restore the customer. Lakehouse
    second. Cache last, because `retailgr bootstrap` rebuilds the cache
    *from silver* — clearing Redis before silver leaves a window where a
    replay puts the customer straight back.
    """
    started = time.time()
    identifiers = target_identifiers(cfg, user_id)
    report = ErasureReport(user_id=user_id, identifiers=identifiers, dry_run=dry_run)

    report.steps.append(_erase_from_raw(cfg, identifiers, dry_run))

    tables = _tables_to_clear(cfg)
    if cfg.warehouse_backend == "iceberg":
        for table, column in tables:
            report.steps.append(_erase_from_iceberg(cfg, table, column, identifiers, dry_run))
        report.steps.append(_expire_snapshots(cfg, [t for t, _ in tables], dry_run))
    else:
        owns_spark = spark is None
        if owns_spark:
            from retailgr.spark_session import build_spark

            spark = build_spark(cfg)
        try:
            for table, column in tables:
                report.steps.append(
                    _erase_from_parquet(cfg, spark, table, column, identifiers, dry_run)
                )
        finally:
            if owns_spark:
                spark.stop()
                spark = None

    report.steps.append(_erase_from_online_store(cfg, user_id, dry_run))

    if not dry_run:
        owns_spark = spark is None
        if owns_spark:
            from retailgr.spark_session import build_spark

            spark = build_spark(cfg)
        try:
            report.verification = verify(cfg, user_id, spark=spark)
        finally:
            if owns_spark:
                spark.stop()

    report.seconds = time.time() - started
    return report


def render(report: ErasureReport, *, reveal: bool = False, key: bytes | None = None) -> str:
    """Markdown receipt, including what could not be done.

    ``reveal`` names the subject. It is true for what goes to the terminal —
    the operator typed the id a moment ago and needs to see which erasure
    they are reading — and false for what goes to disk, because a receipt
    that ends up as a permanent file naming the person who asked to be
    forgotten is a new record of them, created by the act of deleting them.
    """
    subject = report.user_id if reveal else report.subject_digest(key)
    lines = [
        f"# Erasure: {subject}",
        "",
        f"- Identifiers targeted: {len(report.identifiers)}",
        f"- Mode: {'dry run' if report.dry_run else 'applied'}",
        f"- Duration: {report.seconds:.1f}s",
        "",
        "## Steps",
        "",
        "| Step | Status | Rows |",
        "| --- | --- | ---: |",
    ]
    for step in report.steps:
        rows = step.get("rows_removed", step.get("rows_matched", step.get("keys_removed", "-")))
        lines.append(
            f"| `{step.get('step')}` | {step.get('status', '-')} | {rows} |"
        )

    lines += ["", "## Verification", ""]
    if report.dry_run:
        lines.append("Not run: a dry run changes nothing, so there is nothing to verify.")
    elif report.complete:
        checked = len(report.verification.get("checked", []))
        lines.append(
            f"Searched {checked} locations for the literal identifier and found "
            "**no occurrences**. The search reads bytes rather than querying the "
            "tables, so it does not inherit the assumptions of the code that "
            "just deleted them."
        )
    else:
        lines.append("**Incomplete.** The identifier survives here:")
        lines.append("")
        for entry in report.verification.get("survived", []):
            lines.append(f"- `{entry['where']}` — {entry['matches']} occurrences")

    lines += ["", "## What this could not erase", ""]
    for item in report.unerasable:
        lines += [f"**{item['store']}** — {item['why']}", "", f"*Bound:* {item['bound']}", ""]
    return "\n".join(lines) + "\n"
