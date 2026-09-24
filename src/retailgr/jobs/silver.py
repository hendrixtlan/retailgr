"""Silver: consent, pseudonymisation, dedupe, bots, short histories, hierarchy.

This is where most of the damage gets done in a real pipeline, so each filter
reports how many rows it removed. Those counts go into the run report.

Two of the steps are here rather than anywhere else, and both placements are
arguable enough to be worth stating:

**Consent is enforced on the way out of bronze, not on the way in.** Bronze
is the as-received landing zone; an event that arrived is a fact, and
rewriting the landing zone to match a policy destroys the ability to answer
"what did we actually receive". So bronze keeps everything under a short
retention and silver is the first layer with a policy applied. The cost is
that raw, unfiltered events exist for the length of the bronze window, which
is a real exposure and is why that window is configured rather than assumed.

**Pseudonymisation happens on the same boundary.** Silver and everything
downstream hold an HMAC of the identifier rather than the identifier. That
is a narrower win than it sounds and the module that implements it says so:
a pseudonym attached to a fifty-event purchase history is re-identifiable by
anyone holding a second copy of those purchases. What it does buy is that a
warehouse extract does not *name* customers, and that erasure gains a second
lever — destroy the key and the mapping is gone.

The pseudonym is deterministic, which is what keeps joins working and what
makes erasure possible at all: the pseudonym for a customer asking to be
deleted is recomputed from their id, rather than looked up in a mapping
table that would be the most sensitive asset in the system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from retailgr import privacy
from retailgr.config import Config
from retailgr.io.tables import read_table, write_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


def run(spark: SparkSession, cfg: Config) -> dict[str, int]:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    clean = cfg.get("clean", {}) or {}
    events = read_table(spark, cfg, "bronze.interactions")
    catalog = read_table(spark, cfg, "bronze.catalog")

    stats: dict[str, int] = {"input_events": events.count()}

    # 0. Consent, before anything else. Filtering first means every count
    #    below is a count over data this pipeline is allowed to process,
    #    rather than a count that has to be mentally adjusted afterwards.
    events, consent_stats = _apply_consent(cfg, events)
    stats.update(consent_stats)

    # 1. Deduplicate on the producer's idempotency key.
    events = events.dropDuplicates(["event_id"])
    stats["after_dedupe"] = events.count()

    # 2. Drop unwanted event types.
    drop_types = [str(t) for t in (clean.get("drop_event_types") or [])]
    if drop_types:
        events = events.filter(~F.col("event_type").isin(drop_types))
        stats["after_event_type_filter"] = events.count()

    # 3. Drop events whose SKU is not in the catalog: they cannot be served.
    hierarchy = catalog.select(
        "sku", "style_color_id", "product_id", "category", "brand", "attributes", "list_price"
    ).dropDuplicates(["sku"])
    events = events.join(F.broadcast(hierarchy.select("sku")), on="sku", how="inner")
    stats["after_catalog_join"] = events.count()

    # 4. Bot filter: users with an implausible number of events in one day.
    per_day = events.groupBy("user_id", "event_date").count()
    bots = (
        per_day.filter(F.col("count") > int(clean.get("bot_events_per_day", 500)))
        .select("user_id")
        .distinct()
    )
    n_bots = bots.count()
    if n_bots:
        events = events.join(F.broadcast(bots), on="user_id", how="left_anti")
    stats["bot_users_removed"] = n_bots

    # 5. Users that are too short to learn from, or too long to be one person.
    per_user = events.groupBy("user_id").count()
    keep_users = per_user.filter(
        (F.col("count") >= int(clean.get("min_events_per_user", 5)))
        & (F.col("count") <= int(clean.get("max_events_per_user", 2000)))
    ).select("user_id")
    events = events.join(F.broadcast(keep_users), on="user_id", how="inner")
    stats["after_user_filters"] = events.count()

    # 6. Deterministic ordering key for ties within the same timestamp.
    order = Window.partitionBy("user_id").orderBy(
        F.col("event_ts").asc(), F.col("event_id").asc()
    )
    events = events.withColumn("event_rank", F.row_number().over(order))

    # 7. Pseudonymise. Last, because every filter above is expressed in terms
    #    of `user_id` and rewriting the column first would only mean reading
    #    the same code with an unreadable key in it. The mapping is
    #    deterministic, so nothing downstream loses a join.
    events, pseudonym_stats = _pseudonymise(cfg, events)
    stats.update(pseudonym_stats)

    write_table(spark, cfg, "silver.interactions", events)
    write_table(spark, cfg, "silver.item_hierarchy", hierarchy)

    stats["silver_events"] = events.count()
    stats["silver_items"] = hierarchy.count()
    stats["silver_users"] = events.select("user_id").distinct().count()
    return stats


def _apply_consent(cfg: Config, events):
    """Drop events the customer did not agree to have analysed.

    Refuses to run rather than guess when enforcement is on and the source
    carries no consent at all. Both silent alternatives are wrong in a way
    that is hard to notice later: keeping everything turns an un-updated
    producer into a blanket opt-in, and dropping everything produces an
    empty warehouse that looks like a broken join. A public research dataset
    genuinely has no consent signal, so that case is configuration
    (`privacy.consent.enforce: false`) and not an exception to be swallowed.
    """
    from pyspark.sql import functions as F

    settings = cfg.get("privacy.consent", {}) or {}
    enforce = bool(settings.get("enforce", True))
    consumer = str(settings.get("consumer", "analytics"))
    if not enforce:
        return events, {"consent_enforced": 0}

    if "consent" not in events.columns:
        raise ValueError(
            "privacy.consent.enforce is on but bronze.interactions has no "
            "`consent` column. Re-run ingest with an adapter that emits one, "
            "or set privacy.consent.enforce: false for a dataset that has no "
            "consent signal (a public research dataset does not)."
        )

    populated = events.filter(F.col("consent").isNotNull()).limit(1).count()
    if populated == 0:
        raise ValueError(
            "privacy.consent.enforce is on and every `consent` value is null, "
            "so this run would drop 100% of events. That is the correct "
            "reading of the data and almost certainly not the intended one. "
            "Either the producer is not sending consent yet, or this dataset "
            "has none — set privacy.consent.enforce: false deliberately."
        )

    before = events.count()
    events = events.filter(privacy.consent_column(consumer))
    after = events.count()
    return events, {
        "consent_enforced": 1,
        "consent_dropped_events": before - after,
        "after_consent": after,
    }


def _pseudonymise(cfg: Config, events):
    """Replace `user_id` with a keyed HMAC of itself.

    Implemented with Spark's own `hmac` where available and a two-stage
    `sha2` HMAC construction otherwise, rather than a Python UDF: this runs
    over every row of the largest table in the warehouse, and a UDF would
    serialise all of it through the interpreter to compute a hash that the
    JVM can do in place.

    The key never reaches a column, a log line or the run stats. What the
    stats carry is the *prefix* of the key's own digest, which is enough to
    tell two runs apart — "was this table written with the key we still
    have?" is a question erasure needs answered and identity of the key is
    not recoverable from it.
    """
    import hashlib


    settings = cfg.get("privacy.pseudonymisation", {}) or {}
    if not bool(settings.get("enabled", False)):
        return events, {"pseudonymised": 0}

    key = privacy.pseudonymisation_key(settings.get("key"))
    length = int(settings.get("length", 32))
    events = events.withColumn("user_id", privacy.pseudonym_column("user_id", key, length))
    return events, {
        "pseudonymised": 1,
        "pseudonym_key_fingerprint": hashlib.sha256(key).hexdigest()[:12],
    }
