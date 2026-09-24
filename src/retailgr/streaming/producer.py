"""Producing interaction events, with the schema enforced on the way out.

Validation happens here rather than in the consumer on purpose. A malformed
event that reaches the topic is already a problem for every consumer and for
the lakehouse; rejecting it at the producer keeps one bad client from
poisoning the stream.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from retailgr.actions import ACTION_TYPES
from retailgr.streaming.broker import Broker
from retailgr.streaming.schemas import (
    INTERACTION_AVRO_SCHEMA,
    INTERACTION_SCHEMA_VERSION,
    TOPIC_INTERACTIONS,
    TOPIC_RECS_SERVED,
)


class SchemaViolation(ValueError):
    """An event that does not match the registered schema."""


@dataclass
class InteractionEvent:
    """One interaction, in the shape the topic expects."""

    user_id: str
    event_type: str
    sku: str
    session_id: str = ""
    event_ts: datetime | None = None
    event_id: str = ""
    style_color_id: str | None = None
    product_id: str | None = None
    price: float | None = None
    quantity: int | None = 1
    order_id: str | None = None
    return_reason: str | None = None
    channel: str | None = None
    device: str | None = None
    surface: str | None = None
    store_id: str | None = None
    locale: str | None = None
    schema_version: int = INTERACTION_SCHEMA_VERSION
    ingest_ts: datetime | None = field(default=None)

    def to_message(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_id"] = self.event_id or str(uuid.uuid4())
        payload["session_id"] = self.session_id or payload["event_id"]
        ts = self.event_ts or datetime.now(timezone.utc)
        payload["event_ts"] = ts.isoformat()
        payload["ingest_ts"] = (self.ingest_ts or datetime.now(timezone.utc)).isoformat()
        return payload


_REQUIRED_FIELDS = [
    field_def["name"]
    for field_def in INTERACTION_AVRO_SCHEMA["fields"]
    if not isinstance(field_def["type"], list) and "default" not in field_def
]
_ALLOWED_FIELDS = {field_def["name"] for field_def in INTERACTION_AVRO_SCHEMA["fields"]}


def validate_interaction(message: dict[str, Any]) -> None:
    """Check a message against the registered interaction schema."""
    missing = [name for name in _REQUIRED_FIELDS if message.get(name) in (None, "")]
    if missing:
        raise SchemaViolation(f"interaction is missing required fields: {missing}")

    unknown = set(message) - _ALLOWED_FIELDS
    if unknown:
        raise SchemaViolation(f"interaction has fields not in the schema: {sorted(unknown)}")

    if message["event_type"] not in ACTION_TYPES:
        raise SchemaViolation(
            f"event_type '{message['event_type']}' is not one of {list(ACTION_TYPES)}"
        )

    quantity = message.get("quantity")
    if quantity is not None and int(quantity) < 0:
        raise SchemaViolation(f"quantity must not be negative, got {quantity}")

    price = message.get("price")
    if price is not None and float(price) < 0:
        raise SchemaViolation(f"price must not be negative, got {price}")

    if message["event_type"] == "return" and not message.get("order_id"):
        # A return with no order cannot be attributed, so the return-risk head
        # would learn from it without knowing what was returned.
        raise SchemaViolation("a return event must carry the order_id it reverses")


class EventProducer:
    """Publishes interactions and served-recommendation logs."""

    def __init__(self, broker: Broker, validate: bool = True):
        self.broker = broker
        self.validate = validate
        self.produced = 0
        self.rejected = 0

    def send_interaction(self, event: InteractionEvent | dict[str, Any]) -> dict[str, Any]:
        message = event.to_message() if isinstance(event, InteractionEvent) else dict(event)
        if self.validate:
            validate_interaction(message)
        # Keyed by user so one user's events stay ordered inside a partition,
        # which is what the sequence builder downstream assumes.
        self.broker.produce(TOPIC_INTERACTIONS, message, key=message["user_id"])
        self.produced += 1
        return message

    def send_interactions(
        self, events: list[InteractionEvent | dict[str, Any]], skip_invalid: bool = False
    ) -> int:
        """Produce a batch. With ``skip_invalid`` a bad event is counted and
        dropped instead of aborting the batch."""
        sent = 0
        for event in events:
            try:
                self.send_interaction(event)
                sent += 1
            except SchemaViolation:
                if not skip_invalid:
                    raise
                self.rejected += 1
        return sent

    def send_recs_served(self, payload: dict[str, Any]) -> None:
        """The exposure log. Every served response goes here."""
        self.broker.produce(TOPIC_RECS_SERVED, payload, key=payload["request_id"])

    def flush(self, timeout: float = 10.0) -> None:
        self.broker.flush(timeout=timeout)
