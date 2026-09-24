"""Spark session builder for both warehouse backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from retailgr.config import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


def build_spark(cfg: Config) -> SparkSession:
    """Create a SparkSession configured for the warehouse backend in ``cfg``.

    With backend ``parquet`` no extra jars are needed, so this works on a plain
    laptop or in CI. With backend ``iceberg`` the Iceberg runtime and the AWS
    bundle are pulled from Maven and the REST catalog is registered.
    """
    from pyspark.sql import SparkSession

    spark_cfg = cfg.get("spark", {}) or {}
    builder = (
        SparkSession.builder.appName(spark_cfg.get("app_name", "retailgr"))
        .master(spark_cfg.get("master", "local[*]"))
        .config("spark.driver.memory", spark_cfg.get("driver_memory", "4g"))
        .config("spark.sql.shuffle.partitions", str(spark_cfg.get("shuffle_partitions", 16)))
        .config("spark.sql.session.timeZone", "UTC")
        # Keep the local run quiet and reproducible.
        .config("spark.ui.showConsoleProgress", "false")
    )

    local_dir = spark_cfg.get("local_dir")
    if local_dir:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        builder = builder.config("spark.local.dir", local_dir)

    if cfg.warehouse_backend == "iceberg" and str(
        (cfg.get("warehouse.iceberg", {}) or {}).get("engine", "auto")
    ).lower() != "pyiceberg":
        ice = cfg.get("warehouse.iceberg", {}) or {}
        catalog = ice.get("catalog_name", "retailgr")
        packages = ",".join(ice.get("spark_packages", []))
        if packages:
            # Resolved from Maven when the session starts. If Maven is
            # unreachable the JVM does not fall back, it exits before
            # returning a port — the whole session dies and the error says
            # nothing about Iceberg. That is why `engine: pyiceberg` skips
            # this block entirely rather than trying and recovering.
            builder = builder.config("spark.jars.packages", packages)
        builder = (
            builder.config(
                "spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
            )
            .config(
                f"spark.sql.catalog.{catalog}",
                "org.apache.iceberg.spark.SparkCatalog",
            )
            .config(f"spark.sql.catalog.{catalog}.type", "rest")
            .config(f"spark.sql.catalog.{catalog}.uri", ice.get("uri"))
            .config(f"spark.sql.catalog.{catalog}.warehouse", ice.get("warehouse"))
            .config(
                f"spark.sql.catalog.{catalog}.io-impl",
                "org.apache.iceberg.aws.s3.S3FileIO",
            )
            .config(f"spark.sql.catalog.{catalog}.s3.endpoint", ice.get("s3_endpoint"))
            .config(f"spark.sql.catalog.{catalog}.s3.path-style-access", "true")
            .config("spark.sql.defaultCatalog", catalog)
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(spark_cfg.get("log_level", "WARN"))
    return spark


def stop_spark() -> None:
    """Shut down the active session and free the JVM.

    Worth doing before a long training run: the driver JVM holds a couple of
    gigabytes and its heartbeat threads keep competing for cores that the
    trainer needs. Calling ``build_spark`` afterwards creates a fresh session.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:  # pragma: no cover - pyspark not installed
        return
    active = SparkSession.getActiveSession() or SparkSession._instantiatedSession
    if active is not None:
        active.stop()
