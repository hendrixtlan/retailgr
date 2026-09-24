"""Tests for the streaming path.

The in-process broker is a faithful double of the three Kafka guarantees this
pipeline relies on — partitioning by key, order within a partition, offsets
that only advance on commit — so these tests check behaviour, not plumbing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from retailgr.actions import ACTION_TYPES
from retailgr.granularity import GranularityResolver, uniform_resolver
from retailgr.online_store import InMemoryOnlineStore, ItemState
from retailgr.streaming.broker import InMemoryBroker, partition_for
from retailgr.streaming.consumer import SessionStateConsumer
from retailgr.streaming.producer import (
    EventProducer,
    InteractionEvent,
    SchemaViolation,
    validate_interaction,
)
from retailgr.streaming.replay import replay_catalog, replay_events
from retailgr.streaming.schemas import (
    TOPIC_CATALOG,
    TOPIC_INTERACTIONS,
    TOPIC_SPECS,
    interaction_spark_schema,
)

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def _event(user: str = "U1", sku: str = "P1-black-M", **kwargs) -> InteractionEvent:
    defaults = dict(
        user_id=user,
        event_type="view",
        sku=sku,
        session_id="S1",
        event_ts=NOW,
    )
    defaults.update(kwargs)
    return InteractionEvent(**defaults)


# -- the broker double --------------------------------------------------------


def test_same_key_always_lands_in_the_same_partition():
    """This is the guarantee the sequence builder depends on: one user's
    events must stay ordered, which only holds if they share a partition."""
    assert partition_for("U1", 12) == partition_for("U1", 12)
    assert len({partition_for(f"U{i}", 12) for i in range(200)}) > 1


def test_order_is_preserved_within_a_partition():
    broker = InMemoryBroker(default_partitions=4)
    producer = EventProducer(broker)
    for index in range(10):
        producer.send_interaction(
            _event(sku=f"P{index}", event_ts=NOW + timedelta(seconds=index))
        )

    records = [r for r in broker.records(TOPIC_INTERACTIONS) if r.key == "U1"]
    skus = [r.value["sku"] for r in records]
    assert skus == [f"P{i}" for i in range(10)]


def test_offsets_advance_only_on_commit():
    broker = InMemoryBroker(default_partitions=2)
    producer = EventProducer(broker)
    for index in range(6):
        producer.send_interaction(_event(user=f"U{index}", sku=f"P{index}"))

    first = list(broker.consume([TOPIC_INTERACTIONS], group_id="g1"))
    assert len(first) == 6

    # No commit yet, so a fresh read sees everything again.
    assert len(list(broker.consume([TOPIC_INTERACTIONS], group_id="g1"))) == 6

    broker.commit("g1")
    assert list(broker.consume([TOPIC_INTERACTIONS], group_id="g1")) == []

    # A different group has its own offsets.
    assert len(list(broker.consume([TOPIC_INTERACTIONS], group_id="g2"))) == 6


def test_topic_specs_are_created_with_their_partition_counts():
    broker = InMemoryBroker(default_partitions=1)
    broker.create_topics(TOPIC_SPECS)
    assert broker.partition_count(TOPIC_INTERACTIONS) == TOPIC_SPECS[TOPIC_INTERACTIONS][
        "partitions"
    ]


def test_compacted_topics_are_declared_compacted():
    """Catalog, price and inventory must be compacted: they are current state,
    not a log, and a delete policy would silently lose items."""
    for topic in ("catalog.cdc.v1", "pricing.cdc.v1", "inventory.v1"):
        assert TOPIC_SPECS[topic]["cleanup_policy"] == "compact"
    assert TOPIC_SPECS[TOPIC_INTERACTIONS]["cleanup_policy"] == "delete"


# -- the schema ---------------------------------------------------------------


def test_valid_event_passes_and_gets_ids_filled_in():
    message = _event().to_message()
    validate_interaction(message)
    assert message["event_id"]
    assert message["ingest_ts"] is not None
    # The client clock and the server clock are both recorded.
    assert message["event_ts"] != message["ingest_ts"]


def test_unknown_event_type_is_rejected():
    with pytest.raises(SchemaViolation, match="event_type"):
        validate_interaction(_event(event_type="teleported").to_message())


def test_missing_required_field_is_rejected():
    message = _event().to_message()
    message["user_id"] = ""
    with pytest.raises(SchemaViolation, match="required"):
        validate_interaction(message)


def test_field_not_in_the_schema_is_rejected():
    """A producer inventing a field is how schemas rot; catch it at the edge."""
    message = _event().to_message()
    message["loyalty_tier"] = "gold"
    with pytest.raises(SchemaViolation, match="not in the schema"):
        validate_interaction(message)


def test_negative_price_and_quantity_are_rejected():
    with pytest.raises(SchemaViolation, match="price"):
        validate_interaction(_event(price=-1.0).to_message())
    with pytest.raises(SchemaViolation, match="quantity"):
        validate_interaction(_event(quantity=-3).to_message())


def test_a_return_must_name_the_order_it_reverses():
    with pytest.raises(SchemaViolation, match="order_id"):
        validate_interaction(_event(event_type="return").to_message())
    validate_interaction(_event(event_type="return", order_id="O123").to_message())


def test_producer_can_skip_invalid_events_in_a_batch():
    broker = InMemoryBroker()
    producer = EventProducer(broker)
    events = [_event(sku="P1"), _event(sku="P2", event_type="nonsense"), _event(sku="P3")]
    sent = producer.send_interactions(events, skip_invalid=True)
    assert sent == 2
    assert producer.rejected == 1
    assert broker.topic_size(TOPIC_INTERACTIONS) == 2


def test_spark_schema_is_derived_from_the_avro_schema():
    """Deriving it is what stops bronze from silently missing a new field."""
    ddl = interaction_spark_schema()
    for name in ("event_id string", "user_id string", "event_ts timestamp", "price double"):
        assert name in ddl
    assert "event_type string" in ddl  # the Avro enum becomes a string


def test_every_action_type_is_in_the_topic_enum():
    from retailgr.streaming.schemas import INTERACTION_AVRO_SCHEMA

    field = next(
        f for f in INTERACTION_AVRO_SCHEMA["fields"] if f["name"] == "event_type"
    )
    assert tuple(field["type"]["symbols"]) == ACTION_TYPES


# -- the session-state consumer ----------------------------------------------


def _catalog_rows() -> list[dict]:
    return [
        {
            "sku": "P1-black-S",
            "style_color_id": "P1-black",
            "product_id": "P1",
            "category": "apparel",
            "brand": "b",
            "attributes": {"size": "S"},
            "list_price": 30.0,
        },
        {
            "sku": "P1-black-M",
            "style_color_id": "P1-black",
            "product_id": "P1",
            "category": "apparel",
            "brand": "b",
            "attributes": {"size": "M"},
            "list_price": 30.0,
        },
        {
            "sku": "G9",
            "style_color_id": "G9",
            "product_id": "G9",
            "category": "grocery",
            "brand": "b",
            "attributes": {},
            "list_price": 5.0,
        },
    ]


def _bootstrap_store(resolver: GranularityResolver, events: list[InteractionEvent]):
    broker = InMemoryBroker(default_partitions=2)
    broker.create_topics(TOPIC_SPECS)
    replay_catalog(broker, _catalog_rows())
    producer = EventProducer(broker)
    for event in events:
        producer.send_interaction(event)

    store = InMemoryOnlineStore(tail_length=10)
    consumer = SessionStateConsumer(broker, store, resolver, tail_length=10)
    stats = consumer.run()
    return broker, store, consumer, stats


def test_consumer_resolves_tokens_with_the_training_granularity():
    """The tail must carry the token the model was trained on, not the SKU.

    Resolving it at request time from a different config is the classic
    training/serving skew, so the consumer does it with the same resolver the
    Spark job uses.
    """
    resolver = GranularityResolver(
        {"default": "sku", "by_category": {"apparel": {"level": "style_color"}}}
    )
    _, store, _, stats = _bootstrap_store(
        resolver,
        [
            _event(sku="P1-black-S", event_ts=NOW),
            _event(sku="P1-black-M", event_ts=NOW + timedelta(minutes=1)),
            _event(sku="G9", event_ts=NOW + timedelta(minutes=2)),
        ],
    )
    tail = store.user_tail("U1")
    assert [e.token for e in tail] == ["P1-black", "P1-black", "G9"]
    # Both sizes collapse to one token, but the SKU is still there to serve.
    assert [e.sku for e in tail] == ["P1-black-S", "P1-black-M", "G9"]
    assert stats.unknown_skus == 0
    assert stats.tokens_resolved == 3


def test_consumer_counts_skus_it_has_no_catalog_record_for():
    """A rising count here means the CDC topic is lagging, which is worth
    alerting on rather than hiding."""
    broker = InMemoryBroker()
    broker.create_topics(TOPIC_SPECS)
    producer = EventProducer(broker)
    producer.send_interaction(_event(sku="NEVER-SEEN"))

    store = InMemoryOnlineStore()
    consumer = SessionStateConsumer(broker, store, uniform_resolver("sku"))
    stats = consumer.run()
    assert stats.unknown_skus == 1
    # The event is still recorded: the SKU becomes its own token.
    assert store.user_tail("U1")[0].token == "NEVER-SEEN"


def test_price_and_inventory_reach_the_online_store():
    _, store, _, stats = _bootstrap_store(uniform_resolver("sku"), [_event(sku="G9")])
    state = store.item_state("G9")
    assert state is not None
    assert state.list_price == 5.0
    assert state.stock_by_location == {"default": 10}
    assert stats.price_updates == 3
    assert stats.inventory_updates == 3


def test_tail_is_trimmed_to_its_limit():
    events = [
        _event(sku="G9", event_ts=NOW + timedelta(minutes=index)) for index in range(25)
    ]
    _, store, _, _ = _bootstrap_store(uniform_resolver("sku"), events)
    assert len(store.user_tail("U1")) == 10


def test_replay_preserves_event_count_and_is_idempotent_for_the_catalog():
    broker = InMemoryBroker()
    broker.create_topics(TOPIC_SPECS)

    rows = [
        {
            "event_id": f"E{i}",
            "user_id": "U1",
            "session_id": "S1",
            "event_type": "view",
            "sku": "G9",
            "event_ts": NOW + timedelta(seconds=i),
            "price": 5.0,
            "quantity": 1,
            "order_id": "",
            "return_reason": "",
            "style_color_id": "G9",
            "product_id": "G9",
        }
        for i in range(5)
    ]
    stats = replay_events(EventProducer(broker), rows)
    assert stats.events_produced == 5
    assert stats.events_rejected == 0
    assert broker.topic_size(TOPIC_INTERACTIONS) == 5

    # The catalog topic is compacted and keyed by SKU, so replaying twice is
    # state, not duplication, once compaction runs.
    replay_catalog(broker, _catalog_rows())
    replay_catalog(broker, _catalog_rows())
    keys = {r.key for r in broker.records(TOPIC_CATALOG)}
    assert keys == {"P1-black-S", "P1-black-M", "G9"}


def test_item_state_availability_rules():
    # No inventory feed at all: do not hide the item.
    assert ItemState(sku="X").in_stock_at("store-1") is True
    # Explicitly unavailable wins over stock.
    assert ItemState(sku="X", available=False, stock_by_location={"s": 5}).in_stock_at("s") is False
    # Stock is per location.
    state = ItemState(sku="X", stock_by_location={"s1": 3, "s2": 0})
    assert state.in_stock_at("s1") is True
    assert state.in_stock_at("s2") is False
    # No location given: anywhere in the network counts.
    assert state.in_stock_at(None) is True


# -- the in-process broker partitions the way Kafka does ----------------------


# Taken from `kafka-python`'s implementation of the Java client's murmur2.
# Hardcoded rather than computed, because this is a *core* guarantee of the
# in-process broker and a clean `make install` does not include a Kafka
# client — the comparison test below skipped silently on a fresh environment,
# which meant the claim went unchecked exactly where it most needed checking.
KAFKA_MURMUR2_VECTORS = {
    "": 275646681,
    "a": 2731586172,
    "ab": 316155434,
    "abc": 479470107,
    "abcd": 2971317748,
    "abcde": 461995741,
    "abcdef": 1870650108,
    "user-0000001": 125997462,
    "u1": 745337584,
    "ünïcode": 3406400901,
    "S-12345": 1640692110,
    "order-99999999": 1239859780,
}


def test_the_hash_matches_kafkas_known_values():
    """Always runs, with no optional dependency. The vectors cover each
    remainder of the 4-byte block loop — 0, 1, 2 and 3 trailing bytes — which
    is where a murmur2 transcription goes wrong."""
    from retailgr.streaming.broker import murmur2

    for key, expected in KAFKA_MURMUR2_VECTORS.items():
        assert murmur2(key.encode("utf-8")) == expected, key


def test_the_partitioner_is_kafkas_hash_not_a_stand_in_for_it():
    """Verified against `kafka-python`'s implementation, which needs no broker.

    This used to be `zlib.crc32`, and crc32 satisfies the guarantee the
    pipeline depends on — one key always lands in one partition — while
    answering a different question wrongly. The local broker is also used to
    ask "will my keys hot-spot across twelve partitions?", and a hash Kafka
    does not use makes that answer unrelated to the cluster's.
    """
    reference = pytest.importorskip("kafka.partitioner.default")

    from retailgr.streaming.broker import murmur2

    keys = ["a", "ab", "abc", "abcd", "ünïcode", ""] + [f"user-{i:07d}" for i in range(500)]
    for key in keys:
        assert murmur2(key.encode("utf-8")) == reference.murmur2(key.encode("utf-8")), key


def test_the_partition_chosen_is_the_partition_kafka_would_choose():
    reference = pytest.importorskip("kafka.partitioner.default")

    from retailgr.streaming.broker import partition_for

    for partitions in (1, 4, 12, 64):
        for index in range(300):
            key = f"user-{index:07d}"
            expected = (reference.murmur2(key.encode("utf-8")) & 0x7FFFFFFF) % partitions
            assert partition_for(key, partitions) == expected, (key, partitions)


def test_keys_spread_across_partitions_rather_than_piling_up():
    """The property the hash exists for, and the one crc32 could not be
    trusted to predict."""
    import collections

    from retailgr.streaming.broker import partition_for

    counts = collections.Counter(
        partition_for(f"user-{index:07d}", 12) for index in range(3000)
    )
    assert len(counts) == 12, "some partition never gets a key"
    # No partition takes more than 1.5x the even share.
    assert max(counts.values()) < (3000 / 12) * 1.5
