"""One contract per swap point, run against every implementation of it.

The platform audit checks that the backends have the right *shape*. This
checks that they have the right *behaviour*, and it does so with a single body
of assertions rather than one suite per implementation. That is the whole
design: if the Kafka broker is tested by different assertions from the
in-memory one, "swappable" means "both exist", not "either will do".

Backends whose infrastructure is absent skip rather than fail — there is no
Kafka, Redis or FAISS in this environment — but they skip *out of the same
test*, so the moment `make up` is running they are covered by the identical
assertions with no new code.

`test_the_suite_reports_what_it_could_not_verify` is the one that keeps this
honest. A contract suite that quietly skips three of six backends and prints
a row of green dots is the platform equivalent of reporting AUC on the easy
distribution: it measures what is convenient and implies the rest.
"""

from __future__ import annotations

import shutil
import socket
import tempfile
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# What each optional backend needs, and where to look for it. Kept here rather
# than read from pipeline.yaml so the probe cannot be redirected at something
# that happens to answer.
KAFKA_BOOTSTRAP = ("localhost", 9092)
REDIS_ADDRESS = ("localhost", 6379)
PROBE_TIMEOUT = 0.4

# Three states, and keeping them apart is the point. "Verified" means the
# real client against the real service. "Substituted" means the project's own
# code ran, against something standing in for the service — which checks the
# half of the integration this repository wrote and none of the half it did
# not. Rolling the second into the first would be the sort of quiet
# overclaiming this whole project exists to avoid.
VERIFIED: dict[str, set[str]] = {}
SUBSTITUTED: dict[str, dict[str, str]] = {}
SKIPPED: dict[str, dict[str, str]] = {}
# How a verified backend was reached, when that is worth saying out loud.
NOTES: dict[str, dict[str, str]] = {}


def _reachable(address: tuple[str, int]) -> bool:
    """Is something listening? A fast, bounded probe — never a hang."""
    try:
        with socket.create_connection(address, timeout=PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def _verified(point: str, backend: str) -> None:
    VERIFIED.setdefault(point, set()).add(backend)


def _substituted(point: str, backend: str, note: str) -> None:
    SUBSTITUTED.setdefault(point, {})[backend] = note


def _note(point: str, backend: str, detail: str) -> None:
    NOTES.setdefault(point, {})[backend] = detail


def _require(point: str, backend: str, module: str | None, address: tuple[str, int] | None):
    """Skip with a reason that names what was missing, and record it."""
    if module:
        try:
            __import__(module)
        except ImportError:
            reason = f"`{module}` is not installed"
            SKIPPED.setdefault(point, {})[backend] = reason
            pytest.skip(reason)
    if address and not _reachable(address):
        reason = f"nothing listening on {address[0]}:{address[1]} (`make up` starts it)"
        SKIPPED.setdefault(point, {})[backend] = reason
        pytest.skip(reason)
    _verified(point, backend)


# -- the broker ---------------------------------------------------------------


@pytest.fixture(params=["memory", "kafka"])
def broker(request):
    from retailgr.streaming.broker import InMemoryBroker, KafkaBroker

    if request.param == "memory":
        _verified("broker", "memory")
        instance: Any = InMemoryBroker(default_partitions=4)
    else:
        _require("broker", "kafka", "kafka", KAFKA_BOOTSTRAP)
        instance = KafkaBroker(
            bootstrap_servers=f"{KAFKA_BOOTSTRAP[0]}:{KAFKA_BOOTSTRAP[1]}",
            client_id="retailgr-contract",
        )
    try:
        yield instance
    finally:
        instance.close()


def _topic(broker, name: str = "contract.events.v1", partitions: int = 4) -> str:
    # Real Kafka keeps topics between runs, so the name carries the test's
    # identity rather than relying on a clean broker.
    unique = f"{name}.{id(broker) % 100000}"
    broker.create_topics({unique: {"partitions": partitions, "cleanup_policy": "delete"}})
    return unique


def test_contract_a_produced_record_comes_back_intact(broker):
    topic = _topic(broker)
    broker.produce(topic, {"user_id": "u1", "sku": "S-1"}, key="u1")
    broker.flush()

    records = list(broker.consume([topic], group_id="g1", max_records=1, timeout=5.0))
    assert len(records) == 1
    assert records[0].topic == topic
    assert records[0].key == "u1"
    assert records[0].value["sku"] == "S-1"


def test_contract_a_key_always_lands_in_one_partition(broker):
    """The ordering guarantee everything downstream assumes: one customer's
    events never overtake each other."""
    topic = _topic(broker)
    for index in range(12):
        broker.produce(topic, {"user_id": "u1", "n": index}, key="u1")
    broker.flush()

    records = list(broker.consume([topic], group_id="g2", max_records=12, timeout=5.0))
    assert len(records) == 12
    assert len({record.partition for record in records}) == 1
    assert [record.value["n"] for record in records] == list(range(12))


def test_contract_offsets_advance_only_on_commit(broker):
    topic = _topic(broker)
    for index in range(4):
        broker.produce(topic, {"n": index}, key=f"k{index}")
    broker.flush()

    first = list(broker.consume([topic], group_id="g3", max_records=4, timeout=5.0))
    assert len(first) == 4
    broker.commit("g3")
    again = list(broker.consume([topic], group_id="g3", max_records=4, timeout=1.0))
    assert again == []


def test_contract_creating_a_topic_twice_is_not_an_error(broker):
    topic = _topic(broker)
    broker.create_topics({topic: {"partitions": 4, "cleanup_policy": "delete"}})


# -- the online store ---------------------------------------------------------


def _free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="session")
def embedded_redis():
    """A real ``redis-server``, started for the test session.

    No Docker daemon here, and the Apache-style download hosts are blocked —
    but ``redislite`` ships a compiled redis-server in a PyPI wheel, and PyPI
    is reachable. So the Redis backend can be exercised the way it will
    actually run: this project's code, the real ``redis-py`` client, a real
    TCP socket, a real server.

    That matters more than it sounds. Until this existed the Redis path had
    never executed at all, and the last time an unexecuted path in this
    repository was finally run — FAISS — it turned out to be returning 41% of
    the results it should have.
    """
    redislite = pytest.importorskip("redislite")
    directory = tempfile.mkdtemp(prefix="retailgr-redis-")
    port = _free_port()
    server = redislite.Redis(
        str(Path(directory) / "contract.rdb"), serverconfig={"port": str(port)}
    )
    try:
        yield {"port": port, "version": server.info()["redis_version"]}
    finally:
        server.shutdown()
        shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(params=["memory", "redis"])
def store(request):
    from retailgr.online_store import InMemoryOnlineStore, RedisOnlineStore

    if request.param == "memory":
        _verified("online_store", "memory")
        yield InMemoryOnlineStore(tail_length=50)
        return

    # Three tiers, most real first. Which one ran is recorded, because
    # "the Redis backend works" and "our Redis code parses its own keys back"
    # are different claims and only one of them is worth putting in a README.
    if _reachable(REDIS_ADDRESS):
        _require("online_store", "redis", "redis", REDIS_ADDRESS)
        instance: Any = RedisOnlineStore(
            url=f"redis://{REDIS_ADDRESS[0]}:{REDIS_ADDRESS[1]}/9", tail_length=50
        )
    else:
        embedded = request.getfixturevalue("embedded_redis")
        _verified("online_store", "redis")
        _note(
            "online_store",
            "redis",
            f"real redis-server {embedded['version']} started by redislite, "
            "over TCP, with the project's own client",
        )
        instance = RedisOnlineStore(
            url=f"redis://127.0.0.1:{embedded['port']}/9", tail_length=50
        )

    try:
        instance.client.flushdb()
        yield instance
    finally:
        instance.close()


def _event(sku: str, timestamp: int):
    from retailgr.online_store import TailEvent

    return TailEvent(sku=sku, token=sku, action="view", event_ts=timestamp)


def test_contract_a_tail_reads_back_in_order(store):
    user = f"u-{id(store)}"
    for index in range(5):
        store.append_event(user, _event(f"S-{index}", 1000 + index))
    tail = store.user_tail(user)
    assert [event.sku for event in tail] == [f"S-{i}" for i in range(5)]


def test_contract_a_tail_is_capped(store):
    user = f"cap-{id(store)}"
    for index in range(12):
        store.append_event(user, _event(f"S-{index}", 1000 + index), max_length=4)
    tail = store.user_tail(user)
    assert len(tail) == 4
    assert [event.sku for event in tail] == [f"S-{i}" for i in range(8, 12)]


def test_contract_item_state_round_trips(store):
    from retailgr.online_store import ItemState

    sku = f"S-{id(store)}"
    store.put_item_state(
        ItemState(sku=sku, price=19.99, stock_by_location={"w1": 3})
    )
    state = store.item_state(sku)
    assert state is not None
    assert state.price == pytest.approx(19.99)
    assert state.stock_by_location["w1"] == 3


def test_contract_a_missing_key_is_empty_not_an_error(store):
    """Every read path in the request handler depends on this: an unknown
    customer is a cold start, not a 500."""
    assert store.item_state("no-such-sku") is None
    assert store.user_tail("no-such-user") == []
    assert store.item_states(["no-such-sku"]) == {}


def test_contract_many_item_states_in_one_call(store):
    from retailgr.online_store import ItemState

    skus = [f"M-{id(store)}-{index}" for index in range(3)]
    for sku in skus:
        store.put_item_state(ItemState(sku=sku, price=1.0))
    fetched = store.item_states([*skus, "absent"])
    assert set(fetched) == set(skus)


def test_contract_an_empty_batch_is_empty_not_an_error(store):
    """`MGET` with no keys is a protocol error in Redis, so a store that
    passes the list straight through breaks on the request where the policy
    layer filtered every candidate out — which is a request that happens."""
    assert store.item_states([]) == {}


def test_contract_absent_fields_come_back_absent(store):
    """The serialisation trap: a `None` that survives a JSON round trip as
    the string "None" is still falsy in none of the places that matter, and
    prices it through as a value."""
    from retailgr.online_store import ItemState

    sku = f"N-{id(store)}"
    store.put_item_state(ItemState(sku=sku, price=None, list_price=None, promo_id=None))
    state = store.item_state(sku)
    assert state.price is None
    assert state.list_price is None
    assert state.promo_id is None
    assert state.available is True  # the default, not a dropped field


def test_contract_types_survive_the_round_trip(store):
    """An in-memory store hands back the object it was given; a Redis store
    hands back whatever JSON made of it. The policy layer compares prices and
    counts stock, so a float arriving as a string is a silent wrong answer
    rather than an error."""
    from retailgr.online_store import ItemState

    sku = f"T-{id(store)}"
    store.put_item_state(
        ItemState(sku=sku, price=9.5, stock_by_location={"w1": 3}, available=False)
    )
    state = store.item_state(sku)
    assert isinstance(state.price, float)
    assert isinstance(state.stock_by_location["w1"], int)
    assert state.available is False
    # And the behaviour that depends on all three.
    assert state.in_stock_at("w1") is False


def test_contract_fallbacks_round_trip(store):
    user = f"fb-{id(store)}"
    store.put_fallback(user, ["P-1", "P-2"])
    assert store.fallback(user) == ["P-1", "P-2"]


# -- the retrieval index ------------------------------------------------------


@pytest.fixture(params=["exact", "faiss_flat", "faiss_ivf"])
def index(request):
    """Both FAISS code paths, not just the one a small fixture happens to hit.

    `FaissRetrievalIndex` falls back to a flat index below `4 * nlist`
    vectors, so the obvious fixture — a few dozen embeddings — silently tests
    an exact index wearing FAISS's name and never touches the inverted file
    that is the only reason to install FAISS at all. The approximate path has
    the recall trade-off, so it is the one the overlap assertion below is
    actually about.
    """
    from retailgr.serving.retrieval import FaissRetrievalIndex, build_index

    rng = np.random.default_rng(0)

    if request.param == "exact":
        _verified("retrieval_index", "exact")
        embeddings = rng.normal(size=(64, 16)).astype(np.float32)
        embeddings[0] = 0.0  # the padding row
        return build_index(embeddings, kind="exact"), embeddings

    _require("retrieval_index", request.param, "faiss", None)
    if request.param == "faiss_flat":
        embeddings = rng.normal(size=(64, 16)).astype(np.float32)
        embeddings[0] = 0.0
        return build_index(embeddings, kind="faiss"), embeddings

    # Enough vectors, and a small enough nlist, that the coarse quantiser is
    # built and trained rather than skipped.
    embeddings = rng.normal(size=(800, 16)).astype(np.float32)
    embeddings[0] = 0.0
    return FaissRetrievalIndex(embeddings, nlist=8), embeddings


def test_contract_search_returns_k_valid_ids(index):
    built, embeddings = index
    results = built.search(embeddings[5], k=10, exclude=set())
    assert len(results) == 10
    assert all(0 < identifier < embeddings.shape[0] for identifier in results)
    assert len(set(results)) == 10


def test_contract_the_padding_row_is_never_returned(index):
    built, embeddings = index
    results = built.search(embeddings[5], k=20, exclude=set())
    assert 0 not in results


def test_contract_exclusions_are_honoured(index):
    built, embeddings = index
    excluded = {3, 5, 7, 11}
    results = built.search(embeddings[5], k=10, exclude=excluded)
    assert not (set(results) & excluded)


def test_contract_an_approximate_index_keeps_most_of_the_true_top_k(index):
    """Where the contract deliberately differs — and the check that found a
    real defect the first time it ran.

    An approximate index is *not* behaviourally identical to an exact one, and
    demanding equality would be the wrong contract: it trades recall for
    speed, which is the entire reason to use it. But the trade has a floor,
    because recall lost at retrieval is unrecoverable — the ranker reorders
    what it is handed and cannot ask for the candidate that was never
    returned.

    The shipped configuration failed this at 35%. `nprobe` was a tenth of the
    cells, so the index searched a tenth of the catalogue; on 50,000 items
    that returned 41% of the true top-20. The floor is what stops that coming
    back the next time someone tunes for latency.
    """
    from retailgr.serving.retrieval import measure_recall

    built, embeddings = index
    result = measure_recall(built, embeddings, k=20, queries=60)
    assert result["recall"] >= 0.80, result


# -- what was, and was not, verified ------------------------------------------


def test_the_suite_reports_what_it_could_not_verify():
    """Runs last, and prints the honest summary.

    This test never fails on a missing backend — an environment without Kafka
    is not a broken environment. It fails if a swap point had *no* backend
    verified at all, which would mean the contract proved nothing about it.
    """
    from retailgr.platform import SWAP_POINTS

    lines = []
    for point in SWAP_POINTS:
        if point.name == "warehouse":
            continue  # function-level backend, covered by the pipeline tests
        verified = sorted(VERIFIED.get(point.name, ()))
        lines.append(f"{point.name}: verified {verified or 'nothing'}")
        for backend, detail in sorted(NOTES.get(point.name, {}).items()):
            lines.append(f"    {backend} via {detail}")
        for backend, note in sorted(SUBSTITUTED.get(point.name, {}).items()):
            lines.append(f"    {backend} PARTLY verified — {note}")
        for backend, reason in sorted(SKIPPED.get(point.name, {}).items()):
            lines.append(f"    {backend} NOT verified — {reason}")
        assert verified, f"no backend of '{point.name}' was verified by the contract"
    print("\n" + "\n".join(lines))


def test_a_substituted_backend_is_never_counted_as_verified():
    """The distinction the summary rests on, asserted rather than trusted.

    fakeredis running the contract is a real and useful result — it is the
    first time any of `RedisOnlineStore`'s key layout and pipelining has
    executed — and it is not the same claim as "the Redis backend works".
    """
    substituted = {
        backend for backends in SUBSTITUTED.values() for backend in backends
    }
    for point, backends in VERIFIED.items():
        overlap = backends & substituted
        assert not overlap, f"'{point}' counts {overlap} as verified and substituted"


def test_the_redis_namespace_isolates_two_environments(embedded_redis):
    """`RedisOnlineStore`'s docstring promises "keys are namespaced so one
    Redis can hold several environments". That promise had never been
    executed — this is the test that makes it a fact rather than a comment.
    """
    from retailgr.online_store import ItemState, RedisOnlineStore

    url = f"redis://127.0.0.1:{embedded_redis['port']}/3"
    staging = RedisOnlineStore(url=url, namespace="staging")
    production = RedisOnlineStore(url=url, namespace="production")
    try:
        staging.put_item_state(ItemState(sku="SHARED", price=1.0))
        assert staging.item_state("SHARED") is not None
        assert production.item_state("SHARED") is None

        staging.put_fallback("u1", ["P-staging"])
        assert production.fallback("u1") == []
    finally:
        staging.client.flushdb()
        staging.close()
        production.close()
