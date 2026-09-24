"""Bronze: land the raw dataset in the warehouse, unchanged apart from schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from retailgr.config import Config
from retailgr.datasets.adapters import load_dataset
from retailgr.io.tables import write_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession


def run(spark: SparkSession, cfg: Config) -> dict[str, int]:
    from pyspark.sql import functions as F

    interactions, catalog = load_dataset(spark, cfg)

    interactions = interactions.withColumn("event_date", F.to_date("event_ts"))
    write_table(spark, cfg, "bronze.interactions", interactions)
    write_table(spark, cfg, "bronze.catalog", catalog)

    counts = {
        "bronze.interactions": interactions.count(),
        "bronze.catalog": catalog.count(),
    }
    return counts
