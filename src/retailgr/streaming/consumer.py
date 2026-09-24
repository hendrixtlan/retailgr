"""Consumers: the two things that read the interaction topic.

1. ``BronzeSink`` — the offline path. Micro-batches events into
   ``bronze.interactions``, deduplicating on ``event_id`` so an at-least-once
   delivery becomes an effectively-once table. This is the consumer that keeps
   the lakehouse the long-term copy of the stream.

2. ``SessionStateConsumer`` — the online path. Maintains each user's hot tail
   and live item state in the online store, which is what the serving API
   reads at request time. Sub-second freshness is the whole point: an event
   produced now must influence the next request.

Both run off the same topic and the same schema, which is what stops the two
paths from drifting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from retailgr.config import Config
from retailgr.granularity import GranularityResolver
from retailgr.online_store import ItemState, OnlineStore, TailEvent
from retailgr.streaming.broker import Broker, Record
from retailgr.streaming.schemas import (
    TOPIC_CATALOG,
    TOPIC_INTERACTIONS,
    TOPIC_INVENTORY,
    TOPIC_PRICING,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


def _parse_timestamp(value: Any) -> int:
    """Event timestamps arrive as ISO strings on the wire; seconds here."""
    if value is None:
        return 0
    if isinstance(value, int | float):
        return int(value)
    try:
        return int(datetime.fromisoformat(str(value)).timestamp())
    except ValueError:
        return 0


# -- offline path -------------------------------------------------------------


@dataclass
class SinkStats:
    records_consumed: int = 0
    records_written: int = 0
    duplicates_dropped: int = 0
    batches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "records_consumed": self.records_consumed,
            "records_written": self.records_written,
            "duplicates_dropped": self.duplicates_dropped,
            "batches": self.batches,
        }


class BronzeSink:
    """Writes the interaction topic into ``bronze.interactions``.

    Delivery is at-least-once, so the sink dedupes on ``event_id`` inside each
    batch and relies on the silver job's own dedupe across batches. That is the
    same guarantee an Iceberg exactly-once sink gives, without needing one.
    """

    def __init__(
        self,
        broker: Broker,
        spark: SparkSession,
        cfg: Config,
        group_id: str = "retailgr-bronze-sink",
        batch_size: int = 5000,
    ):
        self.broker = broker
        self.spark = spark
        self.cfg = cfg
        self.group_id = group_id
        self.batch_size = batch_size
        self.stats = SinkStats()
        self._seen_event_ids: set[str] = set()

    def _write_batch(self, rows: list[dict[str, Any]], first_batch: bool) -> None:
        from pyspark.sql import functions as F

        from retailgr.io.tables import write_table

        if not rows:
            return
        frame = (
            self.spark.createDataFrame(rows)
            .withColumn("event_ts", F.to_timestamp("event_ts"))
            .withColumn("price", F.col("price").cast("double"))
            .withColumn("quantity", F.col("quantity").cast("int"))
            .withColumn("event_date", F.to_date("event_ts"))
            .select(
                "event_id",
                "user_id",
                "session_id",
                "event_type",
                "sku",
                "event_ts",
                "price",
                "quantity",
                "order_id",
                "return_reason",
                "event_date",
            )
        )
        # The first batch of a run replaces the table; later ones append.
        write_table(
            self.spark,
            self.cfg,
            "bronze.interactions",
            frame,
            mode="overwrite" if first_batch else "append",
        )
        self.stats.records_written += len(rows)
        self.stats.batches += 1

    def run(self, max_records: int | None = None, timeout: float = 1.0) -> SinkStats:
        batch: list[dict[str, Any]] = []
        first_batch = True

        for record in self.broker.consume(
            [TOPIC_INTERACTIONS],
            group_id=self.group_id,
            max_records=max_records,
            timeout=timeout,
        ):
            self.stats.records_consumed += 1
            event_id = str(record.value.get("event_id"))
            if event_id in self._seen_event_ids:
                self.stats.duplicates_dropped += 1
                continue
            self._seen_event_ids.add(event_id)
            batch.append(
                {
                    "event_id": event_id,
                    "user_id": str(record.value["user_id"]),
                    "session_id": str(record.value.get("session_id") or ""),
                    "event_type": str(record.value["event_type"]),
                    "sku": str(record.value["sku"]),
                    "event_ts": str(record.value["event_ts"]),
                    "price": record.value.get("price"),
                    "quantity": record.value.get("quantity") or 1,
                    "order_id": str(record.value.get("order_id") or ""),
                    "return_reason": str(record.value.get("return_reason") or ""),
                }
            )
            if len(batch) >= self.batch_size:
                self._write_batch(batch, first_batch)
                first_batch = False
                batch = []

        if batch:
            self._write_batch(batch, first_batch)
        # Offsets advance only after the data is durable.
        self.broker.commit(self.group_id)
        return self.stats


# -- online path --------------------------------------------------------------


@dataclass
class SessionStats:
    interactions: int = 0
    catalog_updates: int = 0
    price_updates: int = 0
    inventory_updates: int = 0
    unknown_skus: int = 0
    tokens_resolved: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "interactions": self.interactions,
            "catalog_updates": self.catalog_updates,
            "price_updates": self.price_updates,
            "inventory_updates": self.inventory_updates,
            "unknown_skus": self.unknown_skus,
            "tokens_resolved": self.tokens_resolved,
        }


@dataclass
class _CatalogEntry:
    style_color_id: str
    product_id: str
    category: str | None
    attributes: dict[str, str] = field(default_factory=dict)


class SessionStateConsumer:
    """Keeps the online store current: user tails and live item state.

    The model token for each event is resolved here with the *same*
    ``GranularityResolver`` the Spark job uses, so the sequence the serving
    path builds is the sequence the model was trained on. Resolving it at read
    time instead would be the classic place for training/serving skew to creep
    in.
    """

    def __init__(
        self,
        broker: Broker,
        store: OnlineStore,
        resolver: GranularityResolver,
        group_id: str = "retailgr-session-state",
        tail_length: int = 50,
        on_event: Callable[[Record], None] | None = None,
    ):
        self.broker = broker
        self.store = store
        self.resolver = resolver
        self.group_id = group_id
        self.tail_length = tail_length
        self.on_event = on_event
        self.stats = SessionStats()
        self._catalog: dict[str, _CatalogEntry] = {}

    # -- catalog / price / inventory
    def _handle_catalog(self, value: dict[str, Any]) -> None:
        sku = str(value["sku"])
        self._catalog[sku] = _CatalogEntry(
            style_color_id=str(value.get("style_color_id") or sku),
            product_id=str(value.get("product_id") or sku),
            category=value.get("category"),
            attributes={
                str(k): str(v) for k, v in (value.get("attributes") or {}).items()
            },
        )
        self.stats.catalog_updates += 1

    def _handle_pricing(self, value: dict[str, Any]) -> None:
        sku = str(value["sku"])
        state = self.store.item_state(sku) or ItemState(sku=sku)
        state.list_price = value.get("list_price")
        state.price = value.get("current_price", value.get("list_price"))
        state.promo_id = value.get("promo_id")
        self.store.put_item_state(state)
        self.stats.price_updates += 1

    def _handle_inventory(self, value: dict[str, Any]) -> None:
        sku = str(value["sku"])
        state = self.store.item_state(sku) or ItemState(sku=sku)
        location = str(value.get("location") or "default")
        stock = int(value.get("stock") or 0)
        state.stock_by_location = {**state.stock_by_location, location: stock}
        state.available = bool(value.get("available", stock > 0))
        self.store.put_item_state(state)
        self.stats.inventory_updates += 1

    # -- interactions
    def _resolve_token(self, sku: str) -> str:
        entry = self._catalog.get(sku)
        if entry is None:
            # No catalog record yet: the SKU is its own token. Counted, because
            # a growing number here means the CDC topic is behind.
            self.stats.unknown_skus += 1
            return sku
        token = self.resolver.token_for(
            {
                "sku": sku,
                "style_color_id": entry.style_color_id,
                "product_id": entry.product_id,
                "category": entry.category,
                "attributes": entry.attributes,
            }
        )
        self.stats.tokens_resolved += 1
        return token

    def _handle_interaction(self, value: dict[str, Any]) -> None:
        sku = str(value["sku"])
        user_id = str(value["user_id"])
        self.store.append_event(
            user_id,
            TailEvent(
                sku=sku,
                token=self._resolve_token(sku),
                action=str(value["event_type"]),
                event_ts=_parse_timestamp(value.get("event_ts")),
                session_id=str(value.get("session_id") or ""),
            ),
            max_length=self.tail_length,
        )
        self.stats.interactions += 1

    def run(self, max_records: int | None = None, timeout: float = 1.0) -> SessionStats:
        handlers = {
            TOPIC_CATALOG: self._handle_catalog,
            TOPIC_PRICING: self._handle_pricing,
            TOPIC_INVENTORY: self._handle_inventory,
            TOPIC_INTERACTIONS: self._handle_interaction,
        }
        # Catalog, price and inventory first: an interaction whose SKU has no
        # catalog record yet cannot be resolved to the right token.
        ordered_topics = [TOPIC_CATALOG, TOPIC_PRICING, TOPIC_INVENTORY, TOPIC_INTERACTIONS]

        for topic in ordered_topics:
            for record in self.broker.consume(
                [topic], group_id=self.group_id, max_records=max_records, timeout=timeout
            ):
                handlers[topic](record.value)
                if self.on_event is not None:
                    self.on_event(record)
        self.broker.commit(self.group_id)
        return self.stats
