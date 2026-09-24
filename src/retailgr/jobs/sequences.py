"""Gold: user sequences and the time-based split, at a chosen token granularity.

Run this once per granularity variant. Each run writes its own vocabulary and
sequence tables, which is what makes the SKU-vs-product comparison a fair one:
the split, the cleaning and the evaluation code are identical, only the token
mapping changes.

Split design - global time cutoffs, not per-user leave-one-out:

    |<----------- train ----------->|<-- val -->|<-- test -->|
    0                              T1          T2          Tmax

    train : every event before T1.
    val   : history before T1, targets inside [T1, T2).
    test  : history before T2, targets inside [T2, Tmax].

A per-user leave-one-out split leaks the future into training, because one
user's held-out event may sit before another user's training events. The
cutoff version is what production actually faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from retailgr.config import Config
from retailgr.granularity import GranularityResolver, uniform_resolver
from retailgr.io.tables import read_table, write_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

PAD_ID = 0


@dataclass
class SequenceTables:
    """Where one variant's outputs landed."""

    variant: str
    sequences_table: str
    vocab_table: str
    stats: dict[str, Any]


def resolver_for(cfg: Config, granularity: str) -> GranularityResolver:
    """``config`` uses configs/granularity.yaml; anything else is a flat level."""
    if granularity == "config":
        return GranularityResolver(cfg.granularity)
    return uniform_resolver(granularity)


def _sorted_event_arrays(events: DataFrame, max_len: int) -> DataFrame:
    """One row per user: token ids, actions, timestamps and return labels.

    ``input_returned`` is the ranker's fourth head, and it is the one label
    that cannot be read off the event itself: a return arrives days later as
    its own event. See :func:`with_return_labels`.
    """
    from pyspark.sql import functions as F

    packed = events.groupBy("user_id").agg(
        F.sort_array(
            F.collect_list(
                F.struct(
                    F.col("event_rank").alias("rank"),
                    F.col("token_id").alias("token_id"),
                    F.col("event_type").alias("action"),
                    F.col("event_ts").cast("long").alias("ts"),
                    F.col("returned").alias("returned"),
                )
            )
        ).alias("events")
    )
    # Keep only the most recent max_len events.
    trimmed = packed.withColumn(
        "events",
        F.when(
            F.size("events") > max_len,
            F.slice(F.col("events"), F.size("events") - max_len + 1, max_len),
        ).otherwise(F.col("events")),
    )
    return (
        trimmed.withColumn("input_tokens", F.transform("events", lambda e: e["token_id"]))
        .withColumn("input_actions", F.transform("events", lambda e: e["action"]))
        .withColumn("input_ts", F.transform("events", lambda e: e["ts"]))
        .withColumn("input_returned", F.transform("events", lambda e: e["returned"]))
        .drop("events")
    )


# Label values for ``input_returned``. -1 is not "false": a purchase too
# recent to have been returned yet carries no information, and training the
# return head on it would teach the model that recent purchases are safe.
RETURN_LABEL_NOT_APPLICABLE = -1  # not a purchase
RETURN_LABEL_IMMATURE = -2  # a purchase inside the return window
RETURN_LABEL_KEPT = 0
RETURN_LABEL_RETURNED = 1


def with_return_labels(
    events: DataFrame, return_window_days: int, max_event_ts: int
) -> DataFrame:
    """Attach ``returned`` to every event, by linking purchases to returns.

    A return event names the order it reverses (the producer rejects one that
    does not), so the join key is ``user_id + order_id + sku``.

    The maturity rule is the part that is easy to get wrong. A purchase made
    two days before the end of the data has not had time to be returned, so
    labelling it "kept" would bias the head toward believing recent purchases
    are safe. Those rows are marked immature and excluded from the return
    loss instead.
    """
    from pyspark.sql import functions as F

    returns = (
        events.filter(F.col("event_type") == "return")
        .select(
            F.col("user_id"),
            F.col("order_id"),
            F.col("sku"),
            F.lit(1).alias("was_returned"),
        )
        .filter(F.col("order_id").isNotNull() & (F.col("order_id") != ""))
        .dropDuplicates(["user_id", "order_id", "sku"])
    )

    maturity_cutoff = max_event_ts - return_window_days * 24 * 3600
    joined = events.join(
        F.broadcast(returns), on=["user_id", "order_id", "sku"], how="left"
    )
    return joined.withColumn(
        "returned",
        F.when(F.col("event_type") != "purchase", F.lit(RETURN_LABEL_NOT_APPLICABLE))
        .when(F.col("was_returned") == 1, F.lit(RETURN_LABEL_RETURNED))
        .when(
            F.col("event_ts").cast("long") > F.lit(maturity_cutoff),
            F.lit(RETURN_LABEL_IMMATURE),
        )
        .otherwise(F.lit(RETURN_LABEL_KEPT)),
    ).drop("was_returned")


def run(
    spark: SparkSession,
    cfg: Config,
    granularity: str = "config",
    variant: str | None = None,
) -> SequenceTables:
    from pyspark.sql import functions as F

    variant = variant or granularity
    resolver = resolver_for(cfg, granularity)
    max_len = int(cfg.get("sequences.max_len", 200))
    split_cfg = cfg.get("split", {}) or {}
    train_frac = float(split_cfg.get("train_frac", 0.8))
    val_frac = float(split_cfg.get("val_frac", 0.1))
    min_input = int(split_cfg.get("min_input_events", 3))
    target_types = [str(t) for t in (split_cfg.get("target_event_types") or [])]

    events = read_table(spark, cfg, "silver.interactions")
    hierarchy = read_table(spark, cfg, "silver.item_hierarchy")

    # Attach the hierarchy and resolve each event to its model token.
    events = events.join(F.broadcast(hierarchy), on="sku", how="inner").withColumn(
        "token", resolver.token_column()
    )
    events = events.withColumn("event_ts_unix", F.col("event_ts").cast("long"))

    # -- time cutoffs ---------------------------------------------------------
    bounds = events.select(
        F.min("event_ts_unix").alias("min_ts"), F.max("event_ts_unix").alias("max_ts")
    ).first()

    # Return labels for the ranker's fourth head. Done before the split so a
    # purchase in the training window can still be matched to a return that
    # arrives in the validation window - the label is about the purchase, and
    # withholding evidence that exists would understate return risk.
    events = with_return_labels(
        events,
        return_window_days=int(cfg.get("sequences.return_window_days", 30)),
        max_event_ts=int(bounds["max_ts"]),
    )
    # Exact percentiles, not approximate ones, and the difference is not
    # academic. `approxQuantile(..., relativeError=0.001)` returns a value
    # that depends on how the data is *laid out* — the algorithm summarises
    # per partition and merges — so the same events read from a different
    # file arrangement give a different cutoff.
    #
    # Measured on this project's synthetic set: the Parquet tables
    # (partitioned by event_date) and the Iceberg tables (unpartitioned) put
    # the train/val boundary 1,586 seconds apart, which moved four users out
    # of training and into validation. Every metric downstream shifts by a
    # little, and nothing anywhere says why — two runs on identical data stop
    # being comparable because the storage changed.
    #
    # An exact percentile is a function of the values alone. On a dataset too
    # large to sort, `sequences.quantile_error` trades that determinism back
    # for speed, deliberately and in one place.
    error = float(cfg.get("sequences.quantile_error", 0.0))
    probabilities = [train_frac, train_frac + val_frac]
    if error > 0:
        quantiles = events.approxQuantile("event_ts_unix", probabilities, error)
    else:
        row = events.select(
            F.percentile(F.col("event_ts_unix").cast("double"), probabilities).alias("q")
        ).first()
        quantiles = list(row["q"])
    t1, t2 = int(quantiles[0]), int(quantiles[1])
    if not (bounds["min_ts"] < t1 < t2 <= bounds["max_ts"]):
        raise ValueError(
            f"degenerate time split (t1={t1}, t2={t2}); the dataset may be too small or "
            "concentrated in time"
        )

    train_events = events.filter(F.col("event_ts_unix") < t1)
    val_window = events.filter(
        (F.col("event_ts_unix") >= t1) & (F.col("event_ts_unix") < t2)
    )
    test_window = events.filter(F.col("event_ts_unix") >= t2)
    history_for_test = events.filter(F.col("event_ts_unix") < t2)

    # -- vocabulary, built from training events only ---------------------------
    # A token first seen after the cutoff is a cold-start item: it cannot be
    # retrieved by an id-based model, and counting it would flatter the metrics.
    from pyspark.sql import Window

    vocab = (
        train_events.groupBy("token")
        .agg(
            F.count("*").alias("train_events"),
            F.first("category", ignorenulls=True).alias("category"),
        )
        .withColumn(
            "token_id",
            F.row_number().over(
                Window.orderBy(F.col("train_events").desc(), F.col("token").asc())
            ),
        )
        .select("token_id", "token", "category", "train_events")
    )
    vocab_table = f"gold.vocab_{variant}"
    write_table(spark, cfg, vocab_table, vocab, partition_by=[])
    vocab = read_table(spark, cfg, vocab_table).cache()
    vocab_size = vocab.count()

    def with_ids(frame: DataFrame) -> DataFrame:
        return frame.join(F.broadcast(vocab.select("token", "token_id")), on="token", how="inner")

    train_events = with_ids(train_events)
    history_for_test = with_ids(history_for_test)

    def targets_of(window: DataFrame) -> DataFrame:
        """Target tokens for one window, with the action that goes with each.

        The actions are what let the ranker's heads be *calibrated* rather
        than merely ranked: a target is only a positive for the purchase head
        if the customer actually purchased it. Without them, calibration can
        only ask "did they engage at all", which is not what the
        expected-value blend multiplies by money.

        Several events on one item collapse to the most committed one, by
        ``ACTION_COMMITMENT`` rather than by action id — a max over ids would
        read ``remove_from_cart`` as stronger than ``add_to_cart``.
        """
        from retailgr.actions import ACTION_COMMITMENT, ACTION_TO_ID

        selected = window
        if target_types:
            selected = selected.filter(F.col("event_type").isin(target_types))
        in_vocab = with_ids(selected)

        commitment = F.create_map(
            *[
                item
                for name, rank in ACTION_COMMITMENT.items()
                for item in (F.lit(name), F.lit(rank))
            ]
        )
        ranked = in_vocab.withColumn(
            "commitment", F.coalesce(commitment[F.col("event_type")], F.lit(-1))
        )
        per_token = ranked.groupBy("user_id", "token_id").agg(
            F.max("commitment").alias("commitment")
        )
        action_id = F.create_map(
            *[
                item
                for name, rank in ACTION_COMMITMENT.items()
                for item in (F.lit(rank), F.lit(ACTION_TO_ID[name]))
            ]
        )
        per_token = per_token.withColumn(
            "action_id", F.coalesce(action_id[F.col("commitment")], F.lit(0))
        )
        # array_sort on a struct orders by its first field, so the two arrays
        # come out aligned and in the same token order as before.
        paired = per_token.groupBy("user_id").agg(
            F.array_sort(F.collect_list(F.struct("token_id", "action_id"))).alias("pairs")
        )
        return paired.select(
            "user_id",
            F.transform("pairs", lambda p: p["token_id"]).alias("target_tokens"),
            F.transform("pairs", lambda p: p["action_id"]).alias("target_actions"),
        )

    val_targets = targets_of(val_window)
    test_targets = targets_of(test_window)

    # -- histories ------------------------------------------------------------
    train_hist = _sorted_event_arrays(train_events, max_len).filter(
        F.size("input_tokens") >= min_input
    )
    test_hist = _sorted_event_arrays(history_for_test, max_len).filter(
        F.size("input_tokens") >= min_input
    )

    empty_targets = F.array().cast("array<int>")

    train_rows = (
        train_hist.withColumn("target_tokens", empty_targets)
        .withColumn("target_actions", empty_targets)
        .withColumn("split", F.lit("train"))
    )
    val_rows = (
        train_hist.join(val_targets, on="user_id", how="inner")
        .withColumn("split", F.lit("val"))
    )
    test_rows = (
        test_hist.join(test_targets, on="user_id", how="inner")
        .withColumn("split", F.lit("test"))
    )

    columns = [
        "user_id",
        "input_tokens",
        "input_actions",
        "input_ts",
        "input_returned",
        "target_tokens",
        "target_actions",
        "split",
    ]
    sequences = (
        train_rows.select(*columns)
        .unionByName(val_rows.select(*columns))
        .unionByName(test_rows.select(*columns))
    )

    sequences_table = f"gold.sequences_{variant}"
    write_table(spark, cfg, sequences_table, sequences, partition_by=["split"])

    # -- stats for the report -------------------------------------------------
    persisted = read_table(spark, cfg, sequences_table)
    per_split = {
        row["split"]: row["users"]
        for row in persisted.groupBy("split").agg(F.count("*").alias("users")).collect()
    }
    sparse_tokens = vocab.filter(F.col("train_events") < 5).count()
    distinct_skus = events.select("sku").distinct().count()

    # How much the return head actually has to learn from. A pipeline with no
    # matured returns trains that head on nothing, which is worth knowing
    # before reading its AUC.
    purchases = train_events.filter(F.col("event_type") == "purchase")
    return_counts = purchases.groupBy("returned").count().collect()
    by_label = {int(row["returned"]): int(row["count"]) for row in return_counts}
    matured = by_label.get(RETURN_LABEL_KEPT, 0) + by_label.get(RETURN_LABEL_RETURNED, 0)

    stats = {
        "variant": variant,
        "granularity": resolver.describe(),
        "vocab_size": vocab_size,
        "distinct_skus": distinct_skus,
        "compression_vs_sku": round(distinct_skus / vocab_size, 3) if vocab_size else None,
        "tokens_under_5_events": sparse_tokens,
        "sparse_token_share": round(sparse_tokens / vocab_size, 4) if vocab_size else None,
        "train_users": per_split.get("train", 0),
        "val_users": per_split.get("val", 0),
        "test_users": per_split.get("test", 0),
        "cutoff_train_end_unix": t1,
        "cutoff_val_end_unix": t2,
        "train_purchases": sum(by_label.values()),
        "purchases_matured": matured,
        "purchases_returned": by_label.get(RETURN_LABEL_RETURNED, 0),
        "purchases_immature": by_label.get(RETURN_LABEL_IMMATURE, 0),
        "return_rate": (
            round(by_label.get(RETURN_LABEL_RETURNED, 0) / matured, 4) if matured else None
        ),
    }
    vocab.unpersist()
    return SequenceTables(
        variant=variant, sequences_table=sequences_table, vocab_table=vocab_table, stats=stats
    )
