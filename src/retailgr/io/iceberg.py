"""Apache Iceberg without a JVM.

The Iceberg backend in this project reached the format through Spark, which
means through two jars pulled from Maven Central at runtime. That is fine on a
laptop with open network and useless anywhere else — and "anywhere else"
turned out to include the environment this repository was built in, where
Maven Central answers 403. The result was a backend that had never once
executed: the single situation this project treats as a finding rather than a
footnote.

``pyiceberg`` is a full Python implementation of the format. No JVM, no jars,
no Maven — a wheel from PyPI. That makes two engines over one format:

    format   Apache Iceberg v2, on object storage or a local path
    engines  Spark (distributed writes, the batch jobs)
             pyiceberg (no JVM: reads for training, writes that fit in memory)

Which is worth more than a workaround. The training path loads tables into
memory anyway, so making it read Iceberg directly removes a Spark dependency
from the loop people iterate in — and it means an air-gapped cluster with no
Maven mirror can still produce and consume the lakehouse.

The catalog is a SQL catalog by default (SQLite locally, any SQLAlchemy URL
otherwise) because it needs nothing running. Point ``warehouse.iceberg.uri``
at a REST catalog and the same code talks to that instead; the format on disk
is identical either way, which is the property Iceberg exists to provide.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from retailgr.config import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa
    from pyiceberg.catalog import Catalog


class IcebergUnavailable(RuntimeError):
    """pyiceberg is not installed."""


def available() -> bool:
    """Can this process reach Iceberg without a JVM?"""
    try:
        import pyiceberg  # noqa: F401
    except ImportError:
        return False
    return True


def _require() -> None:
    if not available():
        raise IcebergUnavailable(
            "pyiceberg is not installed. Install the lakehouse extra "
            "(pip install -e '.[lakehouse]') or use warehouse.backend=parquet."
        )


def catalog_for(cfg: Config) -> Catalog:
    """Open the catalog named by ``warehouse.iceberg``.

    A REST URI is used as given. Anything else falls back to a SQL catalog
    beside the warehouse root, which needs no service at all — the difference
    is where the table *pointers* live, not the table format, so a table
    written through one is readable through the other.
    """
    _require()
    from pyiceberg.catalog import load_catalog
    from pyiceberg.catalog.sql import SqlCatalog

    name = str(cfg.get("warehouse.iceberg.catalog_name", "retailgr"))
    uri = str(cfg.get("warehouse.iceberg.uri", "") or "")
    warehouse = str(cfg.get("warehouse.iceberg.warehouse", "") or "")

    if uri.startswith(("http://", "https://")):
        properties: dict[str, str] = {"type": "rest", "uri": uri}
        if warehouse:
            properties["warehouse"] = warehouse
        endpoint = str(cfg.get("warehouse.iceberg.s3_endpoint", "") or "")
        if endpoint:
            properties["s3.endpoint"] = endpoint
        return load_catalog(name, **properties)

    root = Path(cfg.warehouse_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    catalog_uri = uri or f"sqlite:///{root / 'catalog.db'}"
    location = warehouse if warehouse.startswith(("s3://", "file://")) else f"file://{root}"
    return SqlCatalog(name, uri=catalog_uri, warehouse=location)


def _namespace_and_name(table: str) -> tuple[str, str]:
    layer, _, name = table.partition(".")
    return layer, name


def write_arrow(
    cfg: Config,
    table: str,
    data: pa.Table,
    mode: str = "overwrite",
    partition_by: Any = (),
) -> dict[str, Any]:
    """Write an Arrow table as ``table``.

    ``overwrite`` replaces the table's contents in one commit rather than
    deleting and rewriting — an Iceberg overwrite is atomic, so a reader
    mid-write sees the previous snapshot instead of a half-written table.
    That property is the reason this project wanted Iceberg in the first
    place, and ``tests/test_iceberg.py`` asserts it rather than trusting it.
    """
    _require()
    catalog = catalog_for(cfg)
    namespace, name = _namespace_and_name(table)
    catalog.create_namespace_if_not_exists(namespace)

    identifier = f"{namespace}.{name}"
    try:
        iceberg_table = catalog.load_table(identifier)
    except Exception:
        iceberg_table = catalog.create_table(identifier, schema=data.schema)
        # A brand-new table has nothing to overwrite.
        mode = "append"

    if mode == "append":
        iceberg_table.append(data)
    else:
        iceberg_table.overwrite(data)

    iceberg_table = catalog.load_table(identifier)
    return {
        "table": identifier,
        "rows": data.num_rows,
        "snapshot_id": iceberg_table.metadata.current_snapshot_id,
        "snapshots": len(iceberg_table.metadata.snapshots),
        "format_version": iceberg_table.metadata.format_version,
    }


def read_arrow(cfg: Config, table: str, snapshot_id: int | None = None) -> pa.Table:
    """Read ``table`` as Arrow, optionally as of an earlier snapshot."""
    _require()
    catalog = catalog_for(cfg)
    namespace, name = _namespace_and_name(table)
    iceberg_table = catalog.load_table(f"{namespace}.{name}")
    scan = (
        iceberg_table.scan(snapshot_id=snapshot_id)
        if snapshot_id is not None
        else iceberg_table.scan()
    )
    return scan.to_arrow()


def read_rows(cfg: Config, table: str, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Read ``table`` as row dicts, the shape ``io.loaders`` wants."""
    data = read_arrow(cfg, table)
    if columns:
        wanted = [c for c in columns if c in data.schema.names]
        data = data.select(wanted)
    return data.to_pylist()


def table_exists(cfg: Config, table: str) -> bool:
    if not available():
        return False
    namespace, name = _namespace_and_name(table)
    try:
        catalog_for(cfg).load_table(f"{namespace}.{name}")
    except Exception:
        return False
    return True


def snapshots(cfg: Config, table: str) -> list[dict[str, Any]]:
    """The table's history, newest last.

    Exposed because it is the operational difference between a lakehouse and
    a directory of Parquet: every write is a snapshot you can read, compare
    against, or roll back to.
    """
    _require()
    catalog = catalog_for(cfg)
    namespace, name = _namespace_and_name(table)
    iceberg_table = catalog.load_table(f"{namespace}.{name}")
    return [
        {
            "snapshot_id": snapshot.snapshot_id,
            "parent_id": snapshot.parent_snapshot_id,
            "timestamp_ms": snapshot.timestamp_ms,
            "operation": (snapshot.summary.operation.value if snapshot.summary else None),
            "records": int((snapshot.summary or {}).get("total-records", 0) or 0),
        }
        for snapshot in iceberg_table.metadata.snapshots
    ]
