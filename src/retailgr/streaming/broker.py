"""The broker seam: real Kafka, or an in-process double with the same contract.

Everything above this module — the producer, the replay job, the bronze sink,
the session-state materialiser — is written against ``Broker`` and does not
know which one it has. That is what lets the whole streaming path be tested
without a broker, and run against real Kafka by changing one config value.

The in-process implementation is a *faithful* double, not a stub: it partitions
by key hash, preserves order within a partition, and tracks per-group offsets.
Those are the three properties the pipeline actually depends on — keying by
``user_id`` is what makes a user's events arrive in order — so a test that
passes here is testing something real.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Record:
    """One message, as a consumer sees it."""

    topic: str
    partition: int
    offset: int
    key: str | None
    value: dict[str, Any]
    timestamp_ms: int


class Broker(Protocol):
    """The subset of Kafka this project uses."""

    def create_topics(self, specs: dict[str, dict[str, Any]]) -> None: ...

    def produce(self, topic: str, value: dict[str, Any], key: str | None = None) -> None: ...

    def flush(self, timeout: float = 10.0) -> None: ...

    def consume(
        self,
        topics: list[str],
        group_id: str,
        max_records: int | None = None,
        timeout: float = 1.0,
        from_beginning: bool = True,
    ) -> Iterator[Record]: ...

    def commit(self, group_id: str) -> None: ...

    def close(self) -> None: ...


def murmur2(data: bytes) -> int:
    """The hash Kafka's default partitioner uses, from the Java client.

    Vendored in about twenty lines rather than imported, because the
    in-process broker must work with no Kafka client installed — that is the
    whole point of it — and this is the one piece of the client's behaviour
    the double cannot approximate without changing what it predicts.
    ``tests/test_streaming.py`` asserts it agrees with ``kafka-python``'s
    implementation whenever that is installed.
    """
    length = len(data)
    seed = 0x9747B28C
    m = 0x5BD1E995
    r = 24

    h = seed ^ length
    rounded = length & ~0x3

    for block in range(0, rounded, 4):
        k = (
            (data[block] & 0xFF)
            | ((data[block + 1] & 0xFF) << 8)
            | ((data[block + 2] & 0xFF) << 16)
            | (data[block + 3] << 24)
        )
        k = (k * m) & 0xFFFFFFFF
        k ^= (k & 0xFFFFFFFF) >> r
        k = (k * m) & 0xFFFFFFFF
        h = (h * m) & 0xFFFFFFFF
        h ^= k

    remaining = length & 0x3
    if remaining == 3:
        h ^= (data[rounded + 2] & 0xFF) << 16
    if remaining >= 2:
        h ^= (data[rounded + 1] & 0xFF) << 8
    if remaining >= 1:
        h ^= data[rounded] & 0xFF
        h = (h * m) & 0xFFFFFFFF

    h ^= (h & 0xFFFFFFFF) >> 13
    h = (h * m) & 0xFFFFFFFF
    h ^= (h & 0xFFFFFFFF) >> 15
    # Returned unsigned. Kafka's Java implementation returns a signed int and
    # its partitioner masks the sign bit off with `& 0x7fffffff`, so the two
    # conventions choose the same partition — but returning the unsigned value
    # makes this function byte-comparable against `kafka-python`'s, which is
    # the reference the tests check it against.
    return h & 0xFFFFFFFF


def partition_for(key: str | None, partitions: int) -> int:
    """Which partition Kafka would put this key in.

    Not "in spirit" — the same hash. This used to be ``zlib.crc32``, which
    satisfies the guarantee the pipeline depends on (one key always lands in
    one partition, so a customer's events never overtake each other) and
    quietly fails a second thing the in-process broker is used for.

    A double that hashes differently from Kafka cannot answer "will my keys
    hot-spot?". For a retail event stream keyed by ``user_id`` over twelve
    partitions, skew is an operational question people ask of the local run
    and then act on, and the crc32 answer had no relationship to the one the
    cluster would give. Matching murmur2 makes the local partition layout the
    real layout.

    A null key round-robins in Kafka; here it lands in partition 0, which
    keeps the double deterministic. Every topic this project produces to has
    a key, so that difference does not arise — and unlike the hash, it is a
    difference in a case the pipeline never exercises.
    """
    if key is None:
        return 0
    return (murmur2(key.encode("utf-8")) & 0x7FFFFFFF) % partitions


# -- in-process ---------------------------------------------------------------


@dataclass
class _Partition:
    records: list[Record] = field(default_factory=list)


class InMemoryBroker:
    """An in-process broker for tests and for running the pipeline with no
    infrastructure. Thread-safe, so a producer and a consumer can run in
    different threads the way they do in production."""

    def __init__(self, default_partitions: int = 4):
        self.default_partitions = default_partitions
        self._topics: dict[str, list[_Partition]] = {}
        self._specs: dict[str, dict[str, Any]] = {}
        # group -> topic -> partition -> next offset to read
        self._offsets: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._pending: dict[str, dict[str, dict[int, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._lock = threading.RLock()

    # -- admin
    def create_topics(self, specs: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            for topic, spec in specs.items():
                partitions = int(spec.get("partitions", self.default_partitions))
                self._specs[topic] = dict(spec)
                if topic not in self._topics:
                    self._topics[topic] = [_Partition() for _ in range(partitions)]

    def _ensure(self, topic: str) -> list[_Partition]:
        if topic not in self._topics:
            self._topics[topic] = [_Partition() for _ in range(self.default_partitions)]
        return self._topics[topic]

    # -- produce
    def produce(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        with self._lock:
            partitions = self._ensure(topic)
            index = partition_for(key, len(partitions))
            target = partitions[index]
            target.records.append(
                Record(
                    topic=topic,
                    partition=index,
                    offset=len(target.records),
                    key=key,
                    value=value,
                    timestamp_ms=int(time.time() * 1000),
                )
            )

    def flush(self, timeout: float = 10.0) -> None:  # noqa: ARG002 - nothing buffered
        return None

    # -- consume
    def consume(
        self,
        topics: list[str],
        group_id: str,
        max_records: int | None = None,
        timeout: float = 1.0,  # noqa: ARG002 - the log is already complete
        from_beginning: bool = True,
    ) -> Iterator[Record]:
        produced = 0
        for topic in topics:
            with self._lock:
                partitions = self._ensure(topic)
                offsets = self._offsets[group_id][topic]
                snapshot = [(index, list(part.records)) for index, part in enumerate(partitions)]
            for index, records in snapshot:
                start = offsets.get(index, 0 if from_beginning else len(records))
                for record in records[start:]:
                    yield record
                    produced += 1
                    # Read position advances as records are handed out; it
                    # becomes durable only on commit.
                    self._pending[group_id][topic][index] = record.offset + 1
                    if max_records is not None and produced >= max_records:
                        return

    def commit(self, group_id: str) -> None:
        with self._lock:
            for topic, partitions in self._pending[group_id].items():
                self._offsets[group_id][topic].update(partitions)
            self._pending[group_id].clear()

    def close(self) -> None:
        return None

    # -- inspection, for tests
    def topic_size(self, topic: str) -> int:
        with self._lock:
            return sum(len(part.records) for part in self._ensure(topic))

    def records(self, topic: str) -> list[Record]:
        """Every record, partition by partition. Order holds within a
        partition, which is the guarantee Kafka actually gives."""
        with self._lock:
            return [record for part in self._ensure(topic) for record in part.records]

    def partition_count(self, topic: str) -> int:
        with self._lock:
            return len(self._ensure(topic))


# -- real Kafka ---------------------------------------------------------------


class KafkaBroker:
    """Real Kafka over ``kafka-python``.

    Values are JSON on the wire here rather than Avro-with-Schema-Registry.
    The schema in ``schemas.py`` is still the contract and is still validated
    on produce; swapping the serializer for ``confluent-kafka``'s Avro
    serializer is a change to the two ``_serialise``/``_deserialise`` hooks
    below and nothing else.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        client_id: str = "retailgr",
        acks: str | int = "all",
        linger_ms: int = 20,
        compression: str | None = "gzip",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.acks = acks
        self.linger_ms = linger_ms
        self.compression = compression
        self._producer = None
        self._consumers: dict[str, Any] = {}

    # -- serialisation hooks
    @staticmethod
    def _serialise(value: dict[str, Any]) -> bytes:
        return json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")

    @staticmethod
    def _deserialise(payload: bytes) -> dict[str, Any]:
        return json.loads(payload.decode("utf-8"))

    def _require_kafka(self):
        try:
            import kafka  # noqa: F401
        except ImportError as error:  # pragma: no cover - depends on the env
            raise RuntimeError(
                "kafka-python is not installed. Install the streaming extra "
                "(pip install -e '.[streaming]') or use the in-process broker "
                "with streaming.broker=memory."
            ) from error
        return kafka

    def create_topics(self, specs: dict[str, dict[str, Any]]) -> None:
        kafka = self._require_kafka()
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError

        admin = KafkaAdminClient(
            bootstrap_servers=self.bootstrap_servers, client_id=f"{self.client_id}-admin"
        )
        topics = []
        for topic, spec in specs.items():
            configs = {"cleanup.policy": spec.get("cleanup_policy", "delete")}
            if spec.get("retention_ms"):
                configs["retention.ms"] = str(spec["retention_ms"])
            topics.append(
                NewTopic(
                    name=topic,
                    num_partitions=int(spec.get("partitions", 1)),
                    replication_factor=int(spec.get("replication", 1)),
                    topic_configs=configs,
                )
            )
        try:
            admin.create_topics(topics)
        except TopicAlreadyExistsError:
            pass
        except kafka.errors.KafkaError as error:  # pragma: no cover
            # Existing topics with different settings are not worth failing on.
            if "already exists" not in str(error).lower():
                raise
        finally:
            admin.close()

    @property
    def producer(self):
        if self._producer is None:
            self._require_kafka()
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                acks=self.acks,
                linger_ms=self.linger_ms,
                compression_type=self.compression,
                # Exactly-once-ish: the broker dedupes producer retries, and
                # consumers still dedupe on event_id downstream.
                enable_idempotence=True,
                value_serializer=self._serialise,
                key_serializer=lambda k: None if k is None else str(k).encode("utf-8"),
            )
        return self._producer

    def produce(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        self.producer.send(topic, value=value, key=key)

    def flush(self, timeout: float = 10.0) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=timeout)

    def _consumer_for(self, topics: list[str], group_id: str, from_beginning: bool):
        key = f"{group_id}:{','.join(sorted(topics))}"
        if key not in self._consumers:
            self._require_kafka()
            from kafka import KafkaConsumer

            self._consumers[key] = KafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                client_id=self.client_id,
                auto_offset_reset="earliest" if from_beginning else "latest",
                enable_auto_commit=False,
                value_deserializer=self._deserialise,
                key_deserializer=lambda k: None if k is None else k.decode("utf-8"),
            )
        return self._consumers[key]

    def consume(
        self,
        topics: list[str],
        group_id: str,
        max_records: int | None = None,
        timeout: float = 1.0,
        from_beginning: bool = True,
    ) -> Iterator[Record]:
        consumer = self._consumer_for(topics, group_id, from_beginning)
        produced = 0
        while True:
            batches = consumer.poll(timeout_ms=int(timeout * 1000), max_records=500)
            if not batches:
                return
            for partition, messages in batches.items():
                for message in messages:
                    yield Record(
                        topic=partition.topic,
                        partition=partition.partition,
                        offset=message.offset,
                        key=message.key,
                        value=message.value,
                        timestamp_ms=message.timestamp,
                    )
                    produced += 1
                    if max_records is not None and produced >= max_records:
                        return

    def commit(self, group_id: str) -> None:
        for key, consumer in self._consumers.items():
            if key.startswith(f"{group_id}:"):
                consumer.commit()

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close()
            self._producer = None
        for consumer in self._consumers.values():
            consumer.close()
        self._consumers.clear()


def build_broker(cfg) -> Broker:
    """Build the broker named by ``streaming.broker`` in pipeline.yaml."""
    kind = str(cfg.get("streaming.broker", "memory")).lower()
    if kind == "memory":
        return InMemoryBroker(
            default_partitions=int(cfg.get("streaming.memory_partitions", 4))
        )
    if kind == "kafka":
        return KafkaBroker(
            bootstrap_servers=str(cfg.get("streaming.bootstrap_servers", "localhost:9092")),
            client_id=str(cfg.get("streaming.client_id", "retailgr")),
            linger_ms=int(cfg.get("streaming.linger_ms", 20)),
        )
    raise ValueError(f"unknown streaming.broker '{kind}'; expected 'memory' or 'kafka'")
