"""Erasure, and the two ways it silently does not happen.

Both failures in this file were found by running the code, not by reading
it, and both produce the same symptom: a green result for an erasure that
did not occur. That symptom is worse than a red one, because a confident
record of a deletion is exactly what an audit relies on.

**The verifier was blind.** It searched every file's bytes for the
identifier. Spark writes Parquet with Snappy, so the string `U000068`
appears nowhere in the bytes of a file holding nineteen of that customer's
rows: 130 files searched, zero hits, customer present. The first run of the
real erasure reported "no occurrences" and the only reason that answer was
correct is that the raw CSV is plain text and caught it. Fixed by decoding
Parquet instead of grepping it, and the test below is the one that would
have caught it — it asserts the verifier finds a user who was *not* erased.

**Iceberg keeps the deleted rows readable.** `Table.delete()` writes a new
snapshot without them; every earlier snapshot still resolves by id, and this
repository's own `read_arrow(..., snapshot_id=...)` hands them straight
back. The test below does exactly that, so "snapshot expiry is required" is
a demonstrated fact rather than a line in a docstring.

The ordering tests matter for the same reason. The pipeline re-derives
everything from `data/raw/`, and the online store is rebuilt from silver, so
an erasure that runs in the wrong order is undone by the next scheduled job
— quietly, and long after the ticket was closed.
"""

from __future__ import annotations

import csv

import pytest

from retailgr import erasure, privacy
from retailgr.config import Config

# -- the verifier can tell present from absent --------------------------------


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "user_id", "sku", "consent"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def raw_cfg(tmp_path):
    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={
            "dataset": {"name": "synthetic", "raw_path": str(tmp_path / "raw")},
            "warehouse": {"backend": "parquet", "root": str(tmp_path / "warehouse")},
            "online_store": {"backend": "memory"},
        },
    )
    _write_csv(
        cfg.raw_path / "synthetic" / "interactions.csv",
        [
            {"event_id": "E1", "user_id": "U000001", "sku": "S1", "consent": "service,analytics"},
            {"event_id": "E2", "user_id": "U000002", "sku": "S2", "consent": "service"},
            {"event_id": "E3", "user_id": "U000001", "sku": "S3", "consent": "service,analytics"},
        ],
    )
    return cfg


def test_the_verifier_finds_a_user_who_was_never_erased(raw_cfg):
    """The test that gives every "clean" result below its meaning.

    A verifier that always reports clean passes every erasure test ever
    written and certifies every incomplete deletion.
    """
    result = erasure.verify(raw_cfg, "U000001")
    assert not result["clean"]
    assert any("interactions.csv" in entry["where"] for entry in result["survived"])


def test_the_verifier_reports_clean_for_a_user_who_is_not_there(raw_cfg):
    assert erasure.verify(raw_cfg, "U999999")["clean"] is True


def test_the_verifier_sees_inside_compressed_parquet(tmp_path):
    """The defect this module was rewritten around.

    Searching the *bytes* of a Snappy-compressed Parquet file for a string
    that is unambiguously in one of its columns returns nothing. The byte
    search was the whole verifier, so the first real erasure run reported
    success on the strength of the raw CSV alone.
    """
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    # Shaped like a real warehouse file, because the shape is what does the
    # hiding. A toy file with nineteen copies of one value is *not* a
    # reproduction: the string lands in the column statistics and in a
    # dictionary page too small to compress, and a byte search finds it —
    # which is how a byte-search verifier can pass a unit test and still be
    # blind against the files it was written for. Many distinct users make
    # the dictionary page big enough for Snappy to bite, and the string
    # disappears exactly as it does in `data/warehouse`.
    users = [f"U{i:06d}" for i in range(3000) for _ in range(5)]
    path = tmp_path / "part-0.snappy.parquet"
    pq.write_table(pa.table({"user_id": users}), path, compression="snappy")

    # The old implementation, reproduced so the claim is checked rather than
    # asserted: the bytes really do not contain the string.
    assert path.read_bytes().count(b"U000068") == 0

    assert erasure._count_in_file(path, ["U000068"]) == 5


def test_the_verifier_catches_an_identifier_hiding_in_a_derived_column(tmp_path):
    """Two of the dataset adapters build `session_id` as `{user_id}#{date}`.

    A scan restricted to the column called `user_id` would certify an
    erasure that left the identifier sitting in the next column along.
    """
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    path = tmp_path / "derived.parquet"
    pq.write_table(
        pa.table({"user_id": ["U000002"], "session_id": ["U000001#2026-01-01"]}),
        path,
        compression="snappy",
    )
    assert erasure._count_in_file(path, ["U000001"]) == 1


# -- erasure removes the source, not only the derivatives ---------------------


def test_erasure_rewrites_the_raw_file_that_everything_is_derived_from(raw_cfg):
    """Otherwise the next `retailgr ingest` restores the customer.

    This is the step most likely to be left out, because every visible table
    is downstream of it and clearing those looks complete.
    """
    report = erasure.erase(raw_cfg, "U000001")
    rows = list(csv.DictReader(open(raw_cfg.raw_path / "synthetic" / "interactions.csv")))
    assert [row["user_id"] for row in rows] == ["U000002"]
    assert report.complete


def test_a_dry_run_changes_nothing(raw_cfg):
    before = (raw_cfg.raw_path / "synthetic" / "interactions.csv").read_text()
    report = erasure.erase(raw_cfg, "U000001", dry_run=True)
    after = (raw_cfg.raw_path / "synthetic" / "interactions.csv").read_text()
    assert before == after
    assert report.dry_run
    # And it does not claim success, because nothing was verified.
    assert not report.complete


def test_the_report_names_what_it_could_not_erase(raw_cfg):
    """A report listing only what it did invites the reader to assume the
    rest was nothing. Kafka, the trained weights and backups are all real
    residue and all three are named every time."""
    report = erasure.erase(raw_cfg, "U000001", dry_run=True)
    stores = {item["store"] for item in report.unerasable}
    assert {"kafka", "model_weights", "backups"} <= stores
    for item in report.unerasable:
        assert item["bound"], f"{item['store']} states no bound on the residue"


def test_erasure_targets_the_pseudonym_as_well_as_the_id(tmp_path):
    """Silver and gold hold a pseudonym; bronze and raw hold the original.

    Deleting only one of them is a complete erasure from whichever layer
    happened to be checked.
    """
    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={
            "dataset": {"name": "synthetic", "raw_path": str(tmp_path / "raw")},
            "warehouse": {"backend": "parquet", "root": str(tmp_path / "warehouse")},
            "privacy": {
                "pseudonymisation": {
                    "enabled": True,
                    "key": "a-test-key-that-is-long-enough",
                    "length": 32,
                }
            },
        },
    )
    identifiers = erasure.target_identifiers(cfg, "U000001")
    assert len(identifiers) == 2
    assert identifiers[0] == "U000001"
    assert identifiers[1] == privacy.pseudonymise(
        "U000001", b"a-test-key-that-is-long-enough", length=32
    )


def test_without_pseudonymisation_there_is_only_one_identifier(raw_cfg):
    assert erasure.target_identifiers(raw_cfg, "U000001") == ["U000001"]


# -- Iceberg: the delete is cosmetic until the snapshots go -------------------


@pytest.fixture
def iceberg_cfg(tmp_path):
    pytest.importorskip("pyiceberg")
    return Config.load(
        "configs/pipeline.yaml",
        overrides={
            "dataset": {"name": "synthetic", "raw_path": str(tmp_path / "raw")},
            "warehouse": {
                "backend": "iceberg",
                "root": str(tmp_path / "warehouse"),
                "iceberg": {
                    "engine": "pyiceberg",
                    "catalog_name": "retailgr",
                    "uri": "",
                    "warehouse": "",
                },
            },
            "online_store": {"backend": "memory"},
        },
    )


def _seed_iceberg(cfg, table="silver.interactions"):
    import pyarrow as pa

    from retailgr.io import iceberg as iceberg_io

    rows = pa.table(
        {
            "user_id": ["U000001", "U000002", "U000001"],
            "sku": ["S1", "S2", "S3"],
            "event_date": ["2026-01-01"] * 3,
        }
    )
    iceberg_io.write_arrow(cfg, table, rows, mode="overwrite")
    return rows


def test_an_iceberg_delete_leaves_the_rows_readable_in_the_previous_snapshot(iceberg_cfg):
    """The reason snapshot expiry is not optional.

    This is the whole argument for `_expire_snapshots`, demonstrated rather
    than asserted: after a delete the customer is gone from the table and
    still returned by a read of the snapshot before it — using this
    repository's own time-travel API, which any analyst can call.
    """
    from retailgr.io import iceberg as iceberg_io

    _seed_iceberg(iceberg_cfg)
    catalog = iceberg_cfg and iceberg_io.catalog_for(iceberg_cfg)
    handle = catalog.load_table("silver.interactions")
    before_snapshot = handle.metadata.current_snapshot_id

    handle.delete(delete_filter="user_id in ('U000001')")
    handle.refresh()

    current = handle.scan().to_arrow().column("user_id").to_pylist()
    assert "U000001" not in current, "the delete did not take effect at all"

    historical = iceberg_io.read_arrow(
        iceberg_cfg, "silver.interactions", snapshot_id=before_snapshot
    )
    assert "U000001" in historical.column("user_id").to_pylist(), (
        "expected the deleted rows to survive in the previous snapshot — if "
        "this fails, pyiceberg's semantics changed and the expiry step "
        "should be re-justified rather than simply kept"
    )


def test_erasure_expires_the_snapshots_that_still_hold_the_rows(iceberg_cfg):
    """The fix, checked end to end on the real format."""
    from retailgr.io import iceberg as iceberg_io

    _seed_iceberg(iceberg_cfg)
    catalog = iceberg_io.catalog_for(iceberg_cfg)
    before_snapshot = catalog.load_table("silver.interactions").metadata.current_snapshot_id

    report = erasure.erase(iceberg_cfg, "U000001")

    handle = catalog.load_table("silver.interactions")
    handle.refresh()
    assert "U000001" not in handle.scan().to_arrow().column("user_id").to_pylist()

    remaining = {snapshot.snapshot_id for snapshot in handle.metadata.snapshots}
    assert before_snapshot not in remaining, (
        "the pre-delete snapshot survived, so the rows are still readable by "
        "id and the erasure is cosmetic"
    )

    expiry = next(step for step in report.steps if step["step"] == "expire_snapshots")
    assert expiry["status"] == "expired"


def test_the_expiry_step_is_skipped_on_parquet_and_says_so(raw_cfg):
    """Not applicable is a different answer from done, and the report has to
    distinguish them — a parquet rewrite genuinely leaves no history."""
    step = erasure._expire_snapshots(raw_cfg, ["silver.interactions"], dry_run=False)
    assert "not applicable" in step["status"]


# -- the online store ---------------------------------------------------------


def test_forgetting_a_user_does_not_take_the_shared_cold_start_list(raw_cfg):
    """`__global__` used to live at `{ns}:fallback:__global__`, inside the
    per-user keyspace, one wildcard away from being deleted by any cleanup.

    The symptom would have been empty recommendations for every new customer
    while every per-user path kept working — which is the hardest kind of
    outage to attribute.
    """
    from retailgr.online_store import InMemoryOnlineStore, TailEvent

    store = InMemoryOnlineStore()
    store.append_event("U000001", TailEvent(sku="S1", token=1, action="view", event_ts=1.0))
    store.put_fallback("U000001", ["P1"])
    store.put_global_fallback(["P-POPULAR"])

    assert store.forget("U000001") == 2
    assert store.user_tail("U000001") == []
    assert store.fallback("U000001") == ["P-POPULAR"]
    assert store.fallback("U999999") == ["P-POPULAR"]


def test_forget_reports_zero_when_there_was_nothing_to_remove():
    """"Nothing was there" and "done" are different answers, and only one of
    them means the cache was already clean."""
    from retailgr.online_store import InMemoryOnlineStore

    assert InMemoryOnlineStore().forget("U000001") == 0


def test_the_redis_store_puts_the_global_fallback_outside_the_user_keyspace():
    """Checked on the key strings, because that is where the hazard was."""
    from retailgr.online_store import RedisOnlineStore

    store = RedisOnlineStore(namespace="retailgr")
    user_prefix = store._fallback_key("")
    assert not store._global_fallback_key().startswith(user_prefix)


def test_every_online_store_implements_forget():
    """The Protocol gained a method; an implementation that missed it would
    fail at erasure time, in production, on a legal deadline."""
    from retailgr.online_store import InMemoryOnlineStore, RedisOnlineStore

    for implementation in (InMemoryOnlineStore, RedisOnlineStore):
        assert callable(getattr(implementation, "forget", None)), implementation


def test_redis_tails_carry_a_ttl_when_retention_is_configured():
    """The tail was bounded by *count* and not by time, so a customer who
    stopped shopping kept their last fifty events indefinitely."""
    redislite = pytest.importorskip("redislite")

    from retailgr.online_store import RedisOnlineStore, TailEvent

    server = redislite.Redis()
    try:
        store = RedisOnlineStore(namespace="ttltest", ttl_seconds=1234)
        store._client = server
        store.append_event("U000001", TailEvent(sku="S1", token=1, action="view", event_ts=1.0))
        ttl = server.ttl(store._tail_key("U000001"))
        assert 0 < ttl <= 1234

        store.put_global_fallback(["P-POPULAR"])
        # The shared list must not expire: cold-start responses would go
        # empty exactly when the rest of the system is least able to help.
        assert server.ttl(store._global_fallback_key()) == -1
    finally:
        server.shutdown()


def test_no_ttl_is_set_when_retention_is_not_configured():
    """Opt-in, because silently expiring a production cache on upgrade is
    its own incident."""
    redislite = pytest.importorskip("redislite")

    from retailgr.online_store import RedisOnlineStore, TailEvent

    server = redislite.Redis()
    try:
        store = RedisOnlineStore(namespace="nottl", ttl_seconds=None)
        store._client = server
        store.append_event("U000001", TailEvent(sku="S1", token=1, action="view", event_ts=1.0))
        assert server.ttl(store._tail_key("U000001")) == -1
    finally:
        server.shutdown()
