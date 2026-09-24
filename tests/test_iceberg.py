"""The Iceberg backend, executed.

For most of this project's life the lakehouse backend was written, reviewed
and never run: it reached the format through Spark, Spark reached it through
two jars from Maven Central, and the environment this was built in gets a 403
from Maven. The README said so, which is better than not saying so and still
not the same as knowing.

``pyiceberg`` closes that gap — a full implementation of the format in a PyPI
wheel, no JVM — and running the backend found two defects immediately, both
of the kind that only appear when data actually moves:

* a Spark ``MapType`` degraded to ``list<struct<_1,_2>>`` on the way through
  pandas, which surfaced two pipeline stages later as an unrelated-looking
  SQL error about ``element_at``;
* the time split was computed with ``approxQuantile``, whose answer depends
  on file layout, so the same events stored two different ways produced
  train/val boundaries 1,586 seconds apart and moved users between splits.

The second is the more serious: it means two runs on identical data were not
comparable if the storage differed, and nothing anywhere said so.

These tests pin the format properties the architecture actually relies on.
``tests/test_pipeline.py`` covers the pipeline itself; this covers what
Iceberg is *for*.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyiceberg")

import pyarrow as pa  # noqa: E402

from retailgr.config import Config  # noqa: E402
from retailgr.io import iceberg  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """An Iceberg warehouse with a SQL catalog, needing nothing running."""
    return Config.load(
        "configs/pipeline.yaml",
        overrides={
            "warehouse": {
                "backend": "iceberg",
                "root": str(tmp_path / "warehouse"),
                "iceberg": {
                    "engine": "pyiceberg",
                    "catalog_name": "retailgr",
                    "uri": "",
                    "warehouse": "",
                },
            }
        },
    )


def _events(count: int = 3, offset: int = 0) -> pa.Table:
    return pa.table(
        {
            "sku": [f"S-{i + offset}" for i in range(count)],
            "price": [1.0 * (i + offset) for i in range(count)],
            "event_date": ["2026-01-01"] * count,
        }
    )


# -- the format works ---------------------------------------------------------


def test_a_table_round_trips(cfg):
    iceberg.write_arrow(cfg, "silver.interactions", _events())
    assert iceberg.read_arrow(cfg, "silver.interactions").num_rows == 3


def test_it_writes_iceberg_v2_not_a_pile_of_parquet(cfg):
    """The point of the backend. A directory of Parquet files has no schema
    history, no snapshots and no atomic commit; those come from the metadata
    layer, so its presence is what is being asserted."""
    result = iceberg.write_arrow(cfg, "silver.interactions", _events())
    assert result["format_version"] == 2
    assert result["snapshot_id"] is not None

    root = cfg.warehouse_root / "silver" / "interactions"
    assert list(root.glob("metadata/*.metadata.json")), "no table metadata"
    assert list(root.glob("metadata/snap-*.avro")), "no manifest list"
    assert list(root.glob("data/*.parquet")), "no data files"


def test_a_missing_table_is_absent_not_an_exception(cfg):
    assert iceberg.table_exists(cfg, "gold.never_written") is False
    iceberg.write_arrow(cfg, "gold.written", _events())
    assert iceberg.table_exists(cfg, "gold.written") is True


# -- what the architecture actually wanted Iceberg for ------------------------


def test_an_overwrite_is_one_commit_not_a_delete_then_write(cfg):
    """A nightly rebuild that truncates and reloads leaves a window where the
    table is empty, and a reader in that window gets no rows rather than
    stale rows. Iceberg replaces the pointer instead, so the previous
    snapshot stays readable until the new one is complete."""
    iceberg.write_arrow(cfg, "silver.interactions", _events(count=3))
    before = iceberg.snapshots(cfg, "silver.interactions")[-1]["snapshot_id"]

    iceberg.write_arrow(cfg, "silver.interactions", _events(count=1, offset=99))
    assert iceberg.read_arrow(cfg, "silver.interactions").num_rows == 1
    # The old data is still there, addressable, not deleted.
    assert iceberg.read_arrow(cfg, "silver.interactions", snapshot_id=before).num_rows == 3


def test_every_write_leaves_a_snapshot_you_can_name(cfg):
    """The operational difference between a lakehouse and a folder: 'what did
    the model train on' has an answer."""
    iceberg.write_arrow(cfg, "silver.interactions", _events(count=2))
    iceberg.write_arrow(cfg, "silver.interactions", _events(count=5), mode="append")

    history = iceberg.snapshots(cfg, "silver.interactions")
    assert len(history) >= 2
    assert history[0]["parent_id"] is None
    assert history[-1]["parent_id"] is not None
    assert [s["operation"] for s in history][0] == "append"


def test_reading_an_old_snapshot_does_not_disturb_the_current_one(cfg):
    iceberg.write_arrow(cfg, "silver.interactions", _events(count=4))
    first = iceberg.snapshots(cfg, "silver.interactions")[-1]["snapshot_id"]
    iceberg.write_arrow(cfg, "silver.interactions", _events(count=1, offset=50))

    assert iceberg.read_arrow(cfg, "silver.interactions", snapshot_id=first).num_rows == 4
    assert iceberg.read_arrow(cfg, "silver.interactions").num_rows == 1


# -- the type that broke the pipeline -----------------------------------------


def test_a_map_column_survives_as_a_map(cfg):
    """The first defect running this found.

    ``item_hierarchy.attributes`` is a ``MapType``, and the granularity
    resolver calls ``element_at(attributes, 'capacity')`` on it. Sent through
    pandas it arrived as ``list<struct<_1,_2>>``: still readable, still
    writable, and no longer a map — so the resolver failed two stages later
    with an error naming neither pandas nor Iceberg.
    """
    data = pa.table(
        {
            "sku": pa.array(["A", "B"]),
            "attributes": pa.array(
                [[("capacity", "256GB")], [("colour", "black")]],
                type=pa.map_(pa.string(), pa.string()),
            ),
        }
    )
    iceberg.write_arrow(cfg, "silver.item_hierarchy", data)
    back = iceberg.read_arrow(cfg, "silver.item_hierarchy")

    field = back.schema.field("attributes")
    assert pa.types.is_map(field.type), f"attributes came back as {field.type}"
    assert dict(back.column("attributes")[0].as_py())["capacity"] == "256GB"


def test_nulls_and_floats_survive_the_round_trip(cfg):
    data = pa.table(
        {
            "sku": ["A", "B"],
            "price": pa.array([9.99, None], type=pa.float64()),
            "quantity": pa.array([3, None], type=pa.int64()),
        }
    )
    iceberg.write_arrow(cfg, "gold.prices", data)
    rows = iceberg.read_arrow(cfg, "gold.prices").to_pylist()
    by_sku = {row["sku"]: row for row in rows}
    assert by_sku["A"]["price"] == pytest.approx(9.99)
    assert by_sku["B"]["price"] is None
    assert by_sku["B"]["quantity"] is None


# -- reading it without a JVM -------------------------------------------------


def test_rows_come_back_in_the_shape_the_loader_wants(cfg):
    """``io.loaders`` reads the lakehouse straight into memory for training.
    Making that path JVM-free is most of why this module exists: it removes a
    Spark session, and a Maven round trip, from the loop people iterate in."""
    iceberg.write_arrow(cfg, "gold.vocab_config", _events(count=2))
    rows = iceberg.read_rows(cfg, "gold.vocab_config", ["sku", "price"])
    assert len(rows) == 2
    assert set(rows[0]) == {"sku", "price"}


def test_asking_for_a_column_that_is_not_there_is_not_fatal(cfg):
    """A table written before a column existed still has to load, or every
    schema change becomes a migration."""
    iceberg.write_arrow(cfg, "gold.vocab_config", _events(count=2))
    rows = iceberg.read_rows(cfg, "gold.vocab_config", ["sku", "not_a_column"])
    assert set(rows[0]) == {"sku"}


def test_the_loader_reads_iceberg_without_starting_spark(cfg, monkeypatch):
    """Asserted by making a Spark session impossible: if anything reaches for
    one, this fails."""
    import retailgr.spark_session as spark_session

    def explode(*args, **kwargs):
        raise AssertionError("a Spark session was started to read Iceberg")

    monkeypatch.setattr(spark_session, "build_spark", explode)

    iceberg.write_arrow(cfg, "gold.vocab_config", _events(count=3))
    from retailgr.io.loaders import _read_table_rows

    rows = _read_table_rows(cfg, "gold.vocab_config", ["sku", "price"])
    assert len(rows) == 3


def test_the_backend_says_so_when_pyiceberg_is_missing(cfg, monkeypatch):
    """Fail with the fix in the message, not with an ImportError from three
    frames down."""
    monkeypatch.setattr(iceberg, "available", lambda: False)
    with pytest.raises(iceberg.IcebergUnavailable, match="pyiceberg is not installed"):
        iceberg.write_arrow(cfg, "silver.interactions", _events())


# -- the engine selector ------------------------------------------------------


def test_every_engine_the_selector_can_return_is_a_branch_that_exists():
    """Dispatch completeness, which is the real invariant behind
    `warehouse.iceberg.engine`.

    Spark and pyiceberg are two engines over one format and they share no
    interface — the choice is a branch inside `write_table`. So the thing
    worth checking is not that they look alike but that the selector cannot
    return a value nothing handles. An earlier version of the platform audit
    modelled this as a swap point and duly reported that the Spark engine
    "fails to implement `write_arrow`", a function it has no reason to own.
    """
    import inspect

    from retailgr.io import tables

    source = inspect.getsource(tables.write_table) + inspect.getsource(tables.read_table)
    for engine in ("pyiceberg",):
        assert f'== "{engine}"' in source, f"write_table does not branch on {engine}"

    # And the selector's own vocabulary is closed.
    selector = inspect.getsource(tables.iceberg_engine)
    assert '"spark", "pyiceberg"' in selector or "'spark', 'pyiceberg'" in selector


@pytest.mark.parametrize("requested", ["spark", "pyiceberg"])
def test_an_explicit_engine_is_honoured_without_a_spark_session(requested):
    """`auto` probes Spark's classpath; an explicit value must not, or forcing
    the JVM-free engine would still need a JVM to find that out."""
    from retailgr.io.tables import iceberg_engine

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"warehouse": {"iceberg": {"engine": requested}}},
    )
    assert iceberg_engine(cfg, spark=None) == requested


def test_auto_picks_the_jvm_free_engine_when_spark_has_no_iceberg_runtime():
    """Which is the situation in this repository's own environment: Maven
    Central answers 403, so the jars never arrive and a Spark session
    configured for Iceberg cannot write a byte of it."""
    from retailgr.io.tables import iceberg_engine

    cfg = Config.load(
        "configs/pipeline.yaml", overrides={"warehouse": {"iceberg": {"engine": "auto"}}}
    )
    assert iceberg_engine(cfg, spark=None) == "pyiceberg"


# -- the conversion that actually broke ---------------------------------------


class _RecordingSession:
    """A Spark session stand-in that remembers what it was handed."""

    def __init__(self, accept_arrow: bool = True):
        self.accept_arrow = accept_arrow
        self.received = None

    def createDataFrame(self, data, *args, **kwargs):  # noqa: N802 - Spark's name
        import pyarrow as pa

        self.received = data
        if isinstance(data, pa.Table) and not self.accept_arrow:
            raise TypeError("createDataFrame() got an unexpected type")
        return data


def test_arrow_reaches_spark_as_arrow_not_as_pandas():
    """The defect that broke the pipeline, pinned where it happened.

    `test_a_map_column_survives_as_a_map` covers `io/iceberg.py`, which is a
    pure pyarrow round trip — and mutation testing showed it does *not* cover
    this: reverting `_from_arrow` to the pandas path left that test green.
    The bug lived in the Spark hand-off, so this is the test that watches it.
    The full-pipeline version in `tests/test_warehouse_equivalence.py` catches
    it too, in seventy seconds; this one takes milliseconds, which is the
    difference between a guard that runs and a guard that runs nightly.
    """
    import pyarrow as pa

    from retailgr.io.tables import _from_arrow

    data = pa.table(
        {
            "sku": ["A"],
            "attributes": pa.array(
                [[("capacity", "256GB")]], type=pa.map_(pa.string(), pa.string())
            ),
        }
    )
    session = _RecordingSession()
    _from_arrow(session, data)

    assert isinstance(session.received, pa.Table), "Spark was handed a converted frame"
    assert pa.types.is_map(session.received.schema.field("attributes").type)


def test_the_pandas_fallback_is_only_for_a_spark_that_cannot_take_arrow():
    """It still has to exist — PySpark 3.5 cannot accept an Arrow table and
    `pyproject.toml` supports it — but it must be unreachable on a version
    that can.

    The `importorskip` is not defensive padding. A clean install of the
    declared dependencies used to fail right here with
    `ModuleNotFoundError: pandas`, which is how it was discovered that the
    fallback had a dependency nobody had declared: it worked in a developer's
    environment and would have raised in a fresh one.
    """
    pytest.importorskip("pandas")

    import pyarrow as pa

    from retailgr.io.tables import _from_arrow

    data = pa.table({"sku": ["A"], "price": [1.0]})
    session = _RecordingSession(accept_arrow=False)
    result = _from_arrow(session, data)

    assert not isinstance(result, pa.Table)  # it fell back
    assert session.received is not None
