"""Bring the online store up to date by replaying the stream into it.

This is the join between the two paths. The same silver table that trained the
model is replayed as Kafka events; the session-state consumer reads them back
and materialises the online store. Nothing here is a demo shortcut — it is the
backfill a real deployment runs before switching traffic to a new cluster, and
the reason it is worth having as a command is that it exercises producer,
schema, broker, consumer and store in one go.

The nightly fallback list is computed here too, from training popularity, so a
cold-start user or a timed-out model tier still gets something sensible.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from retailgr.config import Config
from retailgr.granularity import GranularityResolver
from retailgr.io.loaders import load_variant
from retailgr.jobs.sequences import resolver_for
from retailgr.online_store import OnlineStore
from retailgr.streaming.broker import Broker
from retailgr.streaming.consumer import SessionStateConsumer
from retailgr.streaming.producer import EventProducer
from retailgr.streaming.replay import read_silver_events, replay_catalog, replay_events
from retailgr.streaming.schemas import TOPIC_SPECS

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


def _catalog_rows(spark: SparkSession, cfg: Config) -> list[dict[str, Any]]:
    from retailgr.io.tables import read_table

    frame = read_table(spark, cfg, "silver.item_hierarchy")
    return [row.asDict(recursive=True) for row in frame.collect()]


def compute_fallback(
    cfg: Config, variant: str, resolver: GranularityResolver, top_n: int = 50
) -> list[str]:
    """The global fallback list: most popular tokens in the training window.

    Popularity is computed from the same gold tables the model trained on, so
    the fallback cannot reference an item the model has never seen.
    """
    data = load_variant(cfg, variant)
    counts: Counter[int] = Counter()
    for history in data.train.inputs:
        counts.update(int(token) for token in history if token > 0)
    return [data.token_by_id[token_id] for token_id, _ in counts.most_common(top_n)]


def run(
    spark: SparkSession,
    cfg: Config,
    broker: Broker,
    store: OnlineStore,
    variant: str = "config",
    limit: int | None = None,
    speed_factor: float | None = None,
) -> dict[str, Any]:
    """Replay silver through the broker and materialise the online store."""
    broker.create_topics(TOPIC_SPECS)
    resolver = resolver_for(cfg, variant)

    # 1. Catalog, price and inventory first: the session-state consumer needs a
    #    catalog record to resolve a SKU to the right model token.
    catalog_rows = _catalog_rows(spark, cfg)
    catalog_count = replay_catalog(broker, catalog_rows)

    # 2. The interaction stream.
    producer = EventProducer(broker)
    rows = read_silver_events(spark, cfg, limit=limit)
    replay_stats = replay_events(producer, rows, speed_factor=speed_factor)

    # 3. Read it all back and build the online state.
    consumer = SessionStateConsumer(
        broker,
        store,
        resolver,
        tail_length=int(cfg.get("online_store.tail_length", 50)),
    )
    session_stats = consumer.run()

    # 4. The fallback list, for cold starts and timed-out model tiers.
    fallback = compute_fallback(cfg, variant, resolver)
    if hasattr(store, "put_global_fallback"):
        store.put_global_fallback(fallback)

    return {
        "catalog_rows": catalog_count,
        "replay": replay_stats.as_dict(),
        "session_state": session_stats.as_dict(),
        "fallback_size": len(fallback),
        "store": store.stats() if hasattr(store, "stats") else {},
    }
