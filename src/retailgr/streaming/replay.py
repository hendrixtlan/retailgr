"""Replay a dataset through the event topic.

This is the step that makes the online and offline paths share one definition
of an event: instead of a separate "streaming demo" with its own fake data, the
*same* silver table that trained the model is replayed as Kafka messages. If
the replay produces events the consumer cannot turn back into an equivalent
bronze table, the two paths have drifted, and that is worth finding out here
rather than in production.

Two speeds:

* ``as_fast_as_possible`` — a backfill. Used by tests and by the bootstrap of
  the online store.
* ``speed_factor`` — wall-clock pacing, where one hour of event time takes
  ``3600 / speed_factor`` seconds. Used to watch the system behave under a
  realistic arrival pattern.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from retailgr.config import Config
from retailgr.streaming.broker import Broker
from retailgr.streaming.producer import EventProducer
from retailgr.streaming.schemas import TOPIC_CATALOG, TOPIC_INVENTORY, TOPIC_PRICING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


@dataclass
class ReplayStats:
    events_read: int = 0
    events_produced: int = 0
    events_rejected: int = 0
    catalog_rows: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "events_read": self.events_read,
            "events_produced": self.events_produced,
            "events_rejected": self.events_rejected,
            "catalog_rows": self.catalog_rows,
            "seconds": round(self.seconds, 2),
            "events_per_second": (
                round(self.events_produced / self.seconds, 1) if self.seconds > 0 else None
            ),
        }


def _row_to_message(row: dict[str, Any]) -> dict[str, Any]:
    """Turn a silver row into an interaction message."""
    event_ts = row.get("event_ts")
    return {
        "event_id": str(row["event_id"]),
        "user_id": str(row["user_id"]),
        "session_id": str(row.get("session_id") or row["event_id"]),
        "event_type": str(row["event_type"]),
        "sku": str(row["sku"]),
        "style_color_id": _optional_str(row.get("style_color_id")),
        "product_id": _optional_str(row.get("product_id")),
        "event_ts": event_ts.isoformat() if hasattr(event_ts, "isoformat") else str(event_ts),
        "ingest_ts": None,
        "price": None if row.get("price") is None else float(row["price"]),
        "quantity": None if row.get("quantity") is None else int(row["quantity"]),
        "order_id": _optional_str(row.get("order_id")),
        "return_reason": _optional_str(row.get("return_reason")),
        "channel": None,
        "device": None,
        "surface": None,
        "store_id": None,
        "locale": None,
        "schema_version": 1,
    }


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def read_silver_events(
    spark: SparkSession, cfg: Config, limit: int | None = None
) -> list[dict[str, Any]]:
    """Read silver interactions in event-time order, as plain dicts."""
    from pyspark.sql import functions as F

    from retailgr.io.tables import read_table

    frame = read_table(spark, cfg, "silver.interactions").orderBy(
        F.col("event_ts").asc(), F.col("event_id").asc()
    )
    if limit:
        frame = frame.limit(limit)
    return [row.asDict(recursive=True) for row in frame.collect()]


def replay_events(
    producer: EventProducer,
    rows: Iterable[dict[str, Any]],
    speed_factor: float | None = None,
    skip_invalid: bool = True,
    progress_every: int = 0,
) -> ReplayStats:
    """Produce ``rows`` as interaction events.

    ``speed_factor`` of None replays as fast as the broker accepts. A value of
    3600 means one hour of event time per wall-clock second.
    """
    stats = ReplayStats()
    started = time.time()
    first_event_ts: float | None = None

    for row in rows:
        stats.events_read += 1
        message = _row_to_message(row)

        if speed_factor:
            event_seconds = _event_seconds(row.get("event_ts"))
            if event_seconds is not None:
                if first_event_ts is None:
                    first_event_ts = event_seconds
                target = (event_seconds - first_event_ts) / speed_factor
                drift = target - (time.time() - started)
                if drift > 0:
                    time.sleep(drift)

        try:
            producer.send_interaction(message)
            stats.events_produced += 1
        except Exception:
            if not skip_invalid:
                raise
            stats.events_rejected += 1

        if progress_every and stats.events_produced % progress_every == 0:
            print(f"    produced {stats.events_produced} events")

    producer.flush()
    stats.seconds = time.time() - started
    return stats


def _event_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def replay_catalog(
    broker: Broker, catalog_rows: Iterable[dict[str, Any]]
) -> int:
    """Publish catalog, pricing and inventory state to the compacted topics.

    These are keyed by SKU and compacted, so replaying them is idempotent:
    the topic ends up holding current state per SKU regardless of how many
    times this runs.
    """
    count = 0
    for row in catalog_rows:
        sku = str(row["sku"])
        broker.produce(
            TOPIC_CATALOG,
            {
                "sku": sku,
                "style_color_id": _optional_str(row.get("style_color_id")) or sku,
                "product_id": _optional_str(row.get("product_id")) or sku,
                "category": _optional_str(row.get("category")),
                "brand": _optional_str(row.get("brand")),
                "attributes": row.get("attributes") or {},
            },
            key=sku,
        )
        list_price = row.get("list_price")
        broker.produce(
            TOPIC_PRICING,
            {
                "sku": sku,
                "list_price": None if list_price is None else float(list_price),
                "current_price": None if list_price is None else float(list_price),
                "promo_id": None,
            },
            key=sku,
        )
        broker.produce(
            TOPIC_INVENTORY,
            {"sku": sku, "location": "default", "stock": 10, "available": True},
            key=f"{sku}#default",
        )
        count += 1
    broker.flush()
    return count
