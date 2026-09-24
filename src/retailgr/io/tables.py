"""Read and write warehouse tables through one interface.

The pipeline code never says "Parquet" or "Iceberg"; it says
``write_table(spark, cfg, "silver.interactions", df)``. Swapping the backend is
a config change, which is what keeps Stage 1 runnable on a laptop and Stage 2
runnable on the real lakehouse.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from retailgr.config import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

# Tables the Stage 1 pipeline knows about, with their partition columns.
TABLE_PARTITIONS: dict[str, tuple[str, ...]] = {
    "bronze.interactions": ("event_date",),
    "bronze.catalog": (),
    "silver.interactions": ("event_date",),
    "silver.item_hierarchy": (),
    "gold.sequences": ("split",),
}


def _table_location(cfg: Config, table: str) -> str:
    layer, name = table.split(".", 1)
    return str(cfg.warehouse_root / layer / name)


def _qualified_name(cfg: Config, table: str) -> str:
    catalog = cfg.get("warehouse.iceberg.catalog_name", "retailgr")
    return f"{catalog}.{table}"


def _spark_has_iceberg(spark: SparkSession) -> bool:
    """Is Iceberg's Spark runtime actually on the classpath?

    Worth asking directly rather than finding out from a stack trace halfway
    through a write. The jars are pulled from Maven at session start, and a
    cluster with no Maven mirror — or an HTTP proxy that answers 403, which is
    the case in the environment this project was built in — gets a Spark
    session that looks configured for Iceberg and cannot write a byte of it.
    """
    try:
        spark._jvm.java.lang.Class.forName(  # type: ignore[union-attr]
            "org.apache.iceberg.spark.SparkCatalog"
        )
    except Exception:
        return False
    return True


def iceberg_engine(cfg: Config, spark: SparkSession | None = None) -> str:
    """Which engine reaches the Iceberg format: ``spark`` or ``pyiceberg``.

    Two engines, one format. Spark writes at scale; pyiceberg needs no JVM at
    all, which is what lets the training loop read the lakehouse directly and
    what lets a machine with no Maven access produce it. ``auto`` prefers
    Spark when its runtime is present, because the batch jobs already hold a
    distributed DataFrame, and falls back rather than failing.
    """
    from retailgr.io import iceberg as pyiceberg_io

    requested = str(cfg.get("warehouse.iceberg.engine", "auto")).lower()
    if requested in {"spark", "pyiceberg"}:
        return requested
    if spark is not None and _spark_has_iceberg(spark):
        return "spark"
    return "pyiceberg" if pyiceberg_io.available() else "spark"


def _to_arrow(df: DataFrame):
    """A Spark DataFrame as an Arrow table, whichever PySpark this is."""
    if hasattr(df, "toArrow"):
        return df.toArrow()
    import pyarrow as pa

    return pa.Table.from_pandas(df.toPandas(), preserve_index=False)


def _from_arrow(spark: SparkSession, data) -> DataFrame:
    """An Arrow table as a Spark DataFrame, without losing its types.

    Going via pandas loses them, and not always loudly. On the catalog table
    it fails outright — ``CANNOT_INFER_TYPE_FOR_FIELD attributes`` — because
    pandas hands Spark a column of bare dicts with nothing to infer from. On
    a simpler frame it is worse than failing: a ``struct<colour, size>``
    arrives as a ``map<string, string>``, which reads fine, writes fine, and
    is a different schema from the one the table declares.

    PySpark 4 accepts an Arrow table directly and keeps the schema. The
    pandas path stays only as a fallback for older versions, where the
    ambiguity is at least a known one.
    """
    try:
        return spark.createDataFrame(data)
    except TypeError:  # pragma: no cover - PySpark < 4
        return spark.createDataFrame(data.to_pandas())


def write_table(
    spark: SparkSession,
    cfg: Config,
    table: str,
    df: DataFrame,
    mode: str = "overwrite",
    partition_by: Sequence[str] | None = None,
) -> None:
    """Persist ``df`` as ``table`` using the configured backend."""
    partitions = list(partition_by if partition_by is not None else TABLE_PARTITIONS.get(table, ()))
    # Only partition by columns the frame actually has.
    partitions = [c for c in partitions if c in df.columns]

    if cfg.warehouse_backend == "iceberg":
        if iceberg_engine(cfg, spark) == "pyiceberg":
            from retailgr.io import iceberg as pyiceberg_io

            pyiceberg_io.write_arrow(cfg, table, _to_arrow(df), mode=mode)
            return

        layer = table.split(".", 1)[0]
        catalog = cfg.get("warehouse.iceberg.catalog_name", "retailgr")
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{layer}")
        writer = df.writeTo(_qualified_name(cfg, table)).using("iceberg")
        if partitions:
            from pyspark.sql import functions as F

            writer = writer.partitionedBy(*[F.col(c) for c in partitions])
        if mode == "append":
            writer.append()
        else:
            writer.createOrReplace()
        return

    location = _table_location(cfg, table)
    Path(location).parent.mkdir(parents=True, exist_ok=True)
    writer = df.write.mode(mode).format("parquet")
    if partitions:
        writer = writer.partitionBy(*partitions)
    writer.save(location)


def read_table(spark: SparkSession, cfg: Config, table: str) -> DataFrame:
    """Load ``table`` from the configured backend."""
    if cfg.warehouse_backend == "iceberg":
        if iceberg_engine(cfg, spark) == "pyiceberg":
            from retailgr.io import iceberg as pyiceberg_io

            # Back through Spark so callers keep the DataFrame they expect.
            # `io.loaders` skips this hop entirely and reads Arrow directly,
            # which is the point of having the engine at all.
            return _from_arrow(spark, pyiceberg_io.read_arrow(cfg, table))
        return spark.table(_qualified_name(cfg, table))
    location = _table_location(cfg, table)
    if not Path(location).exists():
        raise FileNotFoundError(
            f"table '{table}' not found at {location}. Run the earlier pipeline stage first."
        )
    return spark.read.parquet(location)


def table_exists(spark: SparkSession, cfg: Config, table: str) -> bool:
    if cfg.warehouse_backend == "iceberg":
        if iceberg_engine(cfg, spark) == "pyiceberg":
            from retailgr.io import iceberg as pyiceberg_io

            return pyiceberg_io.table_exists(cfg, table)
        try:
            spark.table(_qualified_name(cfg, table)).schema  # noqa: B018
            return True
        except Exception:  # pragma: no cover - depends on catalog state
            return False
    return Path(_table_location(cfg, table)).exists()
