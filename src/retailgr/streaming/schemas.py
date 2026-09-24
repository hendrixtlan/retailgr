"""Event schemas — the contract between producers and every consumer.

The schema lives here, in one place, and both the Kafka path and the lakehouse
path are generated from it. That is deliberate: a recommender's most common
production failure is the online path and the offline path disagreeing about
what an event is, and the cheapest guard against it is refusing to write the
shape down twice.

The Avro schema below is the source of truth for the *shape*:
``interaction_spark_schema()`` derives the matching Spark type from it, so
bronze cannot drift from the topic. That derivation is load-bearing and it is
tested.

**It is not the wire format.** This module used to open by saying "Avro is the
wire format (Schema Registry holds the canonical copy)", and it was not true:
``broker.py`` serialises values as JSON — its own docstring says so plainly —
and nothing in this repository contacts a Schema Registry.
``avro_schema_json()`` below exists to register these schemas and is called by
nothing yet. Two docstrings in one package disagreed, and the wrong one was in
the module named ``schemas``, which is where anybody would look first.

What swapping it in actually takes: the two ``_serialise``/``_deserialise``
hooks in ``broker.py``, a ``confluent-kafka`` Avro serializer pointed at
``streaming.schema_registry_url``, and registering each schema under
``<topic>-value``. Until then the JSON payloads match these field names
exactly, which is what keeps the offline and online paths agreeing — but they
carry no schema id, so a producer on an older schema is caught by a consumer
raising on a missing field rather than by the registry refusing the write.
"""

from __future__ import annotations

import json
from typing import Any

from retailgr.actions import ACTION_TYPES

SCHEMA_NAMESPACE = "com.retailgr.events"
INTERACTION_SCHEMA_VERSION = 1

# -- topics -------------------------------------------------------------------

TOPIC_INTERACTIONS = "interactions.v1"
TOPIC_CATALOG = "catalog.cdc.v1"
TOPIC_PRICING = "pricing.cdc.v1"
TOPIC_INVENTORY = "inventory.v1"
TOPIC_RECS_SERVED = "recs.served.v1"

# partitions, replication, cleanup policy, retention (ms; None = broker default)
TOPIC_SPECS: dict[str, dict[str, Any]] = {
    TOPIC_INTERACTIONS: {
        "partitions": 12,
        "replication": 1,
        "cleanup_policy": "delete",
        # The lakehouse is the long-term copy; the topic only needs a replay
        # window wide enough to survive a consumer outage.
        "retention_ms": 7 * 24 * 3600 * 1000,
        "key": "user_id",
    },
    TOPIC_CATALOG: {
        "partitions": 6,
        "replication": 1,
        "cleanup_policy": "compact",
        "retention_ms": None,
        "key": "sku",
    },
    TOPIC_PRICING: {
        "partitions": 6,
        "replication": 1,
        "cleanup_policy": "compact",
        "retention_ms": None,
        "key": "sku",
    },
    TOPIC_INVENTORY: {
        "partitions": 6,
        "replication": 1,
        "cleanup_policy": "compact",
        "retention_ms": None,
        "key": "sku_location",
    },
    TOPIC_RECS_SERVED: {
        "partitions": 12,
        "replication": 1,
        "cleanup_policy": "delete",
        "retention_ms": 7 * 24 * 3600 * 1000,
        "key": "request_id",
    },
}

# -- Avro ---------------------------------------------------------------------

INTERACTION_AVRO_SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "Interaction",
    "namespace": SCHEMA_NAMESPACE,
    "doc": "One customer interaction with one SKU.",
    "fields": [
        {"name": "event_id", "type": "string", "doc": "UUID; the idempotency key."},
        {"name": "user_id", "type": "string"},
        {
            "name": "session_id",
            "type": "string",
            "doc": "Producer-assigned; used for session context at serving time.",
        },
        {
            "name": "event_type",
            "type": {"type": "enum", "name": "EventType", "symbols": list(ACTION_TYPES)},
        },
        {"name": "sku", "type": "string", "doc": "The unit of inventory and price."},
        {
            "name": "style_color_id",
            "type": ["null", "string"],
            "default": None,
            "doc": "Parent of the SKU; resolved from the catalog if absent.",
        },
        {"name": "product_id", "type": ["null", "string"], "default": None},
        {
            "name": "event_ts",
            "type": {"type": "long", "logicalType": "timestamp-micros"},
            "doc": "Client clock.",
        },
        {
            "name": "ingest_ts",
            "type": ["null", {"type": "long", "logicalType": "timestamp-micros"}],
            "default": None,
            "doc": "Server clock, stamped on receipt; the two differ and both matter.",
        },
        {"name": "price", "type": ["null", "double"], "default": None},
        {"name": "quantity", "type": ["null", "int"], "default": 1},
        {"name": "order_id", "type": ["null", "string"], "default": None},
        {
            "name": "return_reason",
            "type": ["null", "string"],
            "default": None,
            "doc": "Set on return events; feeds the return-risk head.",
        },
        {
            "name": "consent",
            "type": ["null", "string"],
            "default": None,
            "doc": (
                "Comma-separated purposes the customer granted at collection "
                "time: any of service, analytics, personalisation. Null or "
                "absent denies everything except service — a producer that "
                "has not been updated must not become a silent opt-in. "
                "Withdrawal is not represented here: this records what was "
                "agreed when the event was collected, and a later withdrawal "
                "is handled by erasure, which removes the event rather than "
                "rewriting its history."
            ),
        },
        {"name": "channel", "type": ["null", "string"], "default": None},
        {"name": "device", "type": ["null", "string"], "default": None},
        {"name": "surface", "type": ["null", "string"], "default": None},
        {"name": "store_id", "type": ["null", "string"], "default": None},
        {"name": "locale", "type": ["null", "string"], "default": None},
        {
            "name": "schema_version",
            "type": "int",
            "default": INTERACTION_SCHEMA_VERSION,
        },
    ],
}

RECS_SERVED_AVRO_SCHEMA: dict[str, Any] = {
    "type": "record",
    "name": "RecsServed",
    "namespace": SCHEMA_NAMESPACE,
    "doc": "What the API actually showed. Without this there are no unbiased labels.",
    "fields": [
        {"name": "request_id", "type": "string"},
        {"name": "user_id", "type": "string"},
        {"name": "session_id", "type": ["null", "string"], "default": None},
        {"name": "surface", "type": "string"},
        {"name": "model_version", "type": "string"},
        {"name": "served_ts", "type": {"type": "long", "logicalType": "timestamp-micros"}},
        {"name": "latency_ms", "type": ["null", "double"], "default": None},
        {
            "name": "items",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "ServedItem",
                    "fields": [
                        {"name": "position", "type": "int"},
                        {"name": "product_id", "type": "string"},
                        {"name": "suggested_sku", "type": ["null", "string"], "default": None},
                        {"name": "score", "type": "double"},
                        {"name": "reason_code", "type": ["null", "string"], "default": None},
                    ],
                },
            },
        },
    ],
}

SCHEMAS: dict[str, dict[str, Any]] = {
    TOPIC_INTERACTIONS: INTERACTION_AVRO_SCHEMA,
    TOPIC_RECS_SERVED: RECS_SERVED_AVRO_SCHEMA,
}


def avro_schema_json(topic: str) -> str:
    """The schema as it *would* be registered, under ``<topic>-value``.

    Nothing calls this yet — the wire format is JSON (see the module
    docstring). It is kept because it is the exact payload a registry needs
    and because the schema it serialises is the source of truth for the Spark
    type either way, so the two cannot drift apart while waiting.
    """
    if topic not in SCHEMAS:
        raise KeyError(f"no Avro schema for topic '{topic}'; known: {sorted(SCHEMAS)}")
    return json.dumps(SCHEMAS[topic], indent=2, sort_keys=False)


# -- Spark --------------------------------------------------------------------

_AVRO_TO_SPARK = {
    "string": "string",
    "int": "int",
    "long": "bigint",
    "double": "double",
    "boolean": "boolean",
}


def _spark_type(avro_type: Any) -> str:
    """Translate one Avro field type into a Spark DDL type."""
    if isinstance(avro_type, str):
        return _AVRO_TO_SPARK[avro_type]
    if isinstance(avro_type, list):
        # A union with null is just a nullable column in Spark.
        non_null = [t for t in avro_type if t != "null"]
        if len(non_null) != 1:
            raise ValueError(f"cannot map union {avro_type} to a Spark type")
        return _spark_type(non_null[0])
    if isinstance(avro_type, dict):
        logical = avro_type.get("logicalType")
        if logical in ("timestamp-micros", "timestamp-millis"):
            return "timestamp"
        if avro_type.get("type") == "enum":
            return "string"
        if avro_type.get("type") == "array":
            return f"array<{_spark_type(avro_type['items'])}>"
        if avro_type.get("type") == "record":
            fields = ", ".join(
                f"{f['name']}: {_spark_type(f['type'])}" for f in avro_type["fields"]
            )
            return f"struct<{fields}>"
        return _spark_type(avro_type["type"])
    raise ValueError(f"unsupported Avro type: {avro_type!r}")


def interaction_spark_schema() -> str:
    """Spark DDL for the interaction topic, derived from the Avro schema.

    Deriving it means a field added to the topic cannot be silently missing
    from bronze.
    """
    fields = ", ".join(
        f"{field['name']} {_spark_type(field['type'])}"
        for field in INTERACTION_AVRO_SCHEMA["fields"]
    )
    return fields


def spark_schema_for(topic: str) -> str:
    schema = SCHEMAS[topic]
    return ", ".join(
        f"{field['name']} {_spark_type(field['type'])}" for field in schema["fields"]
    )
