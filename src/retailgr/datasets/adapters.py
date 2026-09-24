"""Dataset adapters: turn a public dataset into the canonical bronze schema.

Adding a retailer's own export means adding one adapter here; nothing
downstream changes.

Canonical interactions:
    event_id, user_id, session_id, event_type, sku, event_ts,
    price, quantity, order_id, return_reason

Canonical catalog:
    sku, style_color_id, product_id, category, brand, attributes<map>, list_price
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from retailgr.config import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

EVENT_TYPES = ("view", "click", "add_to_cart", "remove_from_cart", "purchase", "return")


def _empty_attributes():
    from pyspark.sql import functions as F

    return F.create_map().cast("map<string,string>")


def _attributes_map(*pairs: tuple[str, object]):
    """Build a ``map<string,string>`` from (name, column) pairs, dropping blanks."""
    from pyspark.sql import functions as F

    entries = []
    for name, column in pairs:
        value = F.when(
            (column.isNotNull()) & (F.trim(column.cast("string")) != F.lit("")),
            column.cast("string"),
        )
        entries.extend([F.lit(name), value])
    if not entries:
        return _empty_attributes()
    # map_filter drops the keys whose value stayed null.
    return F.map_filter(F.create_map(*entries), lambda _, v: v.isNotNull())


# -- synthetic ----------------------------------------------------------------


def load_synthetic(spark: SparkSession, cfg: Config) -> tuple[DataFrame, DataFrame]:
    from pyspark.sql import functions as F

    from retailgr.datasets.synthetic import SyntheticConfig, generate

    output_dir = cfg.raw_path / "synthetic"
    interactions_path = output_dir / "interactions.csv"
    if not interactions_path.exists():
        params = cfg.get("dataset.synthetic", {}) or {}
        generate(output_dir, SyntheticConfig(**params))

    interactions = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(interactions_path))
        .withColumn("event_ts", F.to_timestamp("event_ts"))
        .withColumn("price", F.col("price").cast("double"))
        .withColumn("quantity", F.col("quantity").cast("int"))
        .withColumn("order_id", F.coalesce(F.col("order_id").cast("string"), F.lit("")))
        .withColumn("return_reason", F.coalesce(F.col("return_reason").cast("string"), F.lit("")))
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "sku",
            "event_ts",
            "price",
            "quantity",
            "order_id",
            "return_reason",
            "consent",
        )
    )

    catalog = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(str(output_dir / "catalog.csv"))
        .withColumn(
            "attributes",
            _attributes_map(
                ("size", F.col("size")),
                ("color", F.col("color")),
                ("capacity", F.col("capacity")),
            ),
        )
        .select(
            F.col("sku").cast("string").alias("sku"),
            F.col("style_color_id").cast("string").alias("style_color_id"),
            F.col("product_id").cast("string").alias("product_id"),
            F.lower(F.col("category").cast("string")).alias("category"),
            F.col("brand").cast("string").alias("brand"),
            "attributes",
            F.col("list_price").cast("double").alias("list_price"),
        )
    )
    return interactions, catalog


# -- REES46 -------------------------------------------------------------------


def load_rees46(spark: SparkSession, cfg: Config) -> tuple[DataFrame, DataFrame]:
    """REES46 eCommerce behaviour data (Kaggle).

    Columns: event_time, event_type, product_id, category_id, category_code,
    brand, price, user_id, user_session. There are no size or colour variants,
    so SKU == style-colour == product; the granularity config still applies and
    simply resolves to the same token.
    """
    from pyspark.sql import functions as F

    pattern = str(cfg.raw_path / (cfg.get("dataset.rees46.glob") or "rees46/*.csv"))
    raw = spark.read.option("header", True).option("inferSchema", True).csv(pattern)

    events = (
        raw.withColumn("event_ts", F.to_timestamp(F.regexp_replace("event_time", " UTC$", "")))
        .withColumn("sku", F.col("product_id").cast("string"))
        .withColumn("user_id", F.col("user_id").cast("string"))
        .withColumn("session_id", F.coalesce(F.col("user_session").cast("string"), F.lit("")))
        .withColumn(
            "event_type",
            F.when(F.col("event_type") == "cart", F.lit("add_to_cart"))
            .when(F.col("event_type") == "remove_from_cart", F.lit("remove_from_cart"))
            .otherwise(F.col("event_type").cast("string")),
        )
        .withColumn(
            "event_id",
            F.sha2(
                F.concat_ws(
                    "|",
                    F.col("user_id"),
                    F.col("sku"),
                    F.col("event_type"),
                    F.col("event_ts").cast("string"),
                ),
                256,
            ),
        )
        .withColumn("quantity", F.lit(1).cast("int"))
        .withColumn("order_id", F.lit(""))
        .withColumn("return_reason", F.lit(""))
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "sku",
            "event_ts",
            F.col("price").cast("double").alias("price"),
            "quantity",
            "order_id",
            "return_reason",
        )
    )

    catalog = (
        raw.select("product_id", "category_code", "brand", "price")
        .withColumn("sku", F.col("product_id").cast("string"))
        .groupBy("sku")
        .agg(
            F.first("category_code", ignorenulls=True).alias("category_code"),
            F.first("brand", ignorenulls=True).alias("brand"),
            F.avg("price").cast("double").alias("list_price"),
        )
        .withColumn("style_color_id", F.col("sku"))
        .withColumn("product_id", F.col("sku"))
        .withColumn(
            "category",
            F.lower(
                F.coalesce(
                    F.split(F.col("category_code"), "\\.").getItem(0), F.lit("unknown")
                )
            ),
        )
        .withColumn("attributes", _empty_attributes())
        .select(
            "sku",
            "style_color_id",
            "product_id",
            "category",
            F.col("brand").cast("string").alias("brand"),
            "attributes",
            "list_price",
        )
    )
    return events, catalog


# -- H&M ----------------------------------------------------------------------


def load_hm(spark: SparkSession, cfg: Config) -> tuple[DataFrame, DataFrame]:
    """H&M Personalized Fashion Recommendations (Kaggle).

    ``article_id`` is a product in one colour, ``product_code`` is the product
    across colours. That makes this dataset the cleanest public test of the
    style-colour vs product question. Only purchases are recorded.
    """
    from pyspark.sql import functions as F

    transactions_path = cfg.raw_path / (
        cfg.get("dataset.hm.transactions") or "hm/transactions_train.csv"
    )
    articles_path = cfg.raw_path / (cfg.get("dataset.hm.articles") or "hm/articles.csv")

    transactions = spark.read.option("header", True).option("inferSchema", True).csv(
        str(transactions_path)
    )
    articles = spark.read.option("header", True).option("inferSchema", True).csv(
        str(articles_path)
    )

    events = (
        transactions.withColumn("event_ts", F.to_timestamp("t_dat"))
        .withColumn("sku", F.col("article_id").cast("string"))
        .withColumn("user_id", F.col("customer_id").cast("string"))
        # The dataset has no sessions; one shopping day is the closest proxy.
        .withColumn("session_id", F.concat_ws("#", F.col("user_id"), F.col("t_dat")))
        .withColumn("event_type", F.lit("purchase"))
        .withColumn(
            "event_id",
            F.sha2(
                F.concat_ws(
                    "|", F.col("user_id"), F.col("sku"), F.col("t_dat"), F.col("sales_channel_id")
                ),
                256,
            ),
        )
        .withColumn("quantity", F.lit(1).cast("int"))
        .withColumn("order_id", F.col("session_id"))
        .withColumn("return_reason", F.lit(""))
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "sku",
            "event_ts",
            F.col("price").cast("double").alias("price"),
            "quantity",
            "order_id",
            "return_reason",
        )
    )

    catalog = (
        articles.withColumn("sku", F.col("article_id").cast("string"))
        .withColumn("style_color_id", F.col("sku"))
        .withColumn("product_id", F.col("product_code").cast("string"))
        .withColumn("category", F.lit("apparel"))
        .withColumn(
            "attributes",
            _attributes_map(
                ("color", F.col("colour_group_name")),
                ("product_type", F.col("product_type_name")),
                ("department", F.col("department_name")),
            ),
        )
        .select(
            "sku",
            "style_color_id",
            "product_id",
            "category",
            F.lit("hm").alias("brand"),
            "attributes",
            F.lit(None).cast("double").alias("list_price"),
        )
    )
    return events, catalog


# -- MovieLens ----------------------------------------------------------------
#
# MovieLens has no item variants, so the granularity question degenerates
# (SKU == style-colour == product) and every variant resolves to the same
# token. What it does have is *real* timestamps spanning years and an explicit
# rating per interaction, which is what makes it the right dataset for the
# other question: does HSTU's temporal bias and action modality pay off when
# time and sentiment actually carry signal?
#
# Ratings map onto the action vocabulary as follows. This is a modelling
# choice, stated here rather than buried:
#
#     rating >= 4  -> purchase          (sought out and liked: strong positive)
#     rating == 3  -> view              (consumed, lukewarm: weak positive)
#     rating <= 2  -> remove_from_cart  (explicit rejection: negative)
#
# The negative class is real - about 17% of ML-1M ratings are 1 or 2 - so the
# positive-target filter from Table 1 of the HSTU paper has something to do.

RATING_TO_ACTION_SQL = """
CASE WHEN rating >= 4 THEN 'purchase'
     WHEN rating = 3 THEN 'view'
     ELSE 'remove_from_cart' END
"""

# MovieLens records ratings, not sessions. Users rate in sittings, so a gap
# longer than this starts a new session. Derived, not measured - named so it
# can be argued with.
SESSION_GAP_SECONDS = 30 * 60


def _sessionise(events: DataFrame) -> DataFrame:
    """Add ``session_id`` by splitting each user's history on time gaps."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    ordered = Window.partitionBy("user_id").orderBy("event_ts_unix")
    with_gap = events.withColumn(
        "previous_ts", F.lag("event_ts_unix").over(ordered)
    ).withColumn(
        "new_session",
        F.when(
            F.col("previous_ts").isNull()
            | ((F.col("event_ts_unix") - F.col("previous_ts")) > SESSION_GAP_SECONDS),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    running = Window.partitionBy("user_id").orderBy("event_ts_unix").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    return (
        with_gap.withColumn("session_index", F.sum("new_session").over(running))
        .withColumn(
            "session_id", F.concat_ws("#", F.col("user_id"), F.col("session_index"))
        )
        .drop("previous_ts", "new_session", "session_index")
    )


def _movielens_frames(
    raw: DataFrame, item_prefix: str, catalog: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Shared shaping for both MovieLens sizes."""
    from pyspark.sql import functions as F

    events = (
        raw.withColumn("user_id", F.concat(F.lit("U"), F.col("user_id").cast("string")))
        .withColumn("sku", F.concat(F.lit(item_prefix), F.col("item_id").cast("string")))
        .withColumn("event_ts_unix", F.col("timestamp").cast("long"))
        .withColumn("event_ts", F.col("event_ts_unix").cast("timestamp"))
        .withColumn("event_type", F.expr(RATING_TO_ACTION_SQL))
    )
    events = _sessionise(events)
    events = (
        events.withColumn(
            "event_id",
            F.sha2(
                F.concat_ws("|", F.col("user_id"), F.col("sku"), F.col("event_ts_unix")), 256
            ),
        )
        .withColumn("quantity", F.lit(1).cast("int"))
        .withColumn("order_id", F.lit(""))
        .withColumn("return_reason", F.lit(""))
        # The rating itself is kept as the "price" slot's numeric companion is
        # not meaningful here, so price stays null rather than faked.
        .withColumn("price", F.lit(None).cast("double"))
        .select(
            "event_id",
            "user_id",
            "session_id",
            "event_type",
            "sku",
            "event_ts",
            "price",
            "quantity",
            "order_id",
            "return_reason",
        )
    )
    return events, catalog


def load_ml1m(spark: SparkSession, cfg: Config) -> tuple[DataFrame, DataFrame]:
    """MovieLens-1M: 1,000,209 ratings, 6,040 users, timestamps over ~3 years.

    Read from the pre-split files published with the NCF paper's code, which
    together are the complete dataset. Item ids there are re-indexed, so no
    genre metadata can be joined; every item lands in one category and the
    granularity variants all collapse to the same tokens. Use ``ml100k`` when
    the per-category breakdown matters.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        IntegerType,
        LongType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("user_id", IntegerType(), False),
            StructField("item_id", IntegerType(), False),
            StructField("rating", IntegerType(), False),
            StructField("timestamp", LongType(), False),
        ]
    )
    paths = [
        str(cfg.raw_path / (cfg.get("dataset.ml1m.train") or "ml1m/ml-1m.train.rating")),
        str(cfg.raw_path / (cfg.get("dataset.ml1m.test") or "ml1m/ml-1m.test.rating")),
    ]
    raw = spark.read.option("sep", "\t").schema(schema).csv(paths)

    catalog = (
        raw.select("item_id")
        .distinct()
        .withColumn("sku", F.concat(F.lit("M"), F.col("item_id").cast("string")))
        .withColumn("style_color_id", F.col("sku"))
        .withColumn("product_id", F.col("sku"))
        .withColumn("category", F.lit("movies"))
        .withColumn("attributes", _empty_attributes())
        .select(
            "sku",
            "style_color_id",
            "product_id",
            "category",
            F.lit("movielens").alias("brand"),
            "attributes",
            F.lit(None).cast("double").alias("list_price"),
        )
    )
    return _movielens_frames(raw, "M", catalog)


def load_ml100k(spark: SparkSession, cfg: Config) -> tuple[DataFrame, DataFrame]:
    """MovieLens-100K: 100,000 ratings with real genres.

    Smaller than ML-1M but it carries genre labels, so the per-category
    breakdown in the report is real rather than a single bucket. The first
    listed genre becomes the category.
    """
    from pyspark.sql import functions as F

    inter_path = str(
        cfg.raw_path / (cfg.get("dataset.ml100k.interactions") or "ml100k/ml-100k.inter")
    )
    item_path = str(cfg.raw_path / (cfg.get("dataset.ml100k.items") or "ml100k/ml-100k.item"))

    raw = (
        spark.read.option("sep", "\t")
        .option("header", True)
        .option("inferSchema", True)
        .csv(inter_path)
        .withColumnRenamed("user_id:token", "user_id")
        .withColumnRenamed("item_id:token", "item_id")
        .withColumnRenamed("rating:float", "rating")
        .withColumnRenamed("timestamp:float", "timestamp")
    )
    items = (
        spark.read.option("sep", "\t")
        .option("header", True)
        .option("inferSchema", True)
        .csv(item_path)
        .withColumnRenamed("item_id:token", "item_id")
        .withColumnRenamed("movie_title:token_seq", "title")
        .withColumnRenamed("release_year:token", "release_year")
        .withColumnRenamed("class:token_seq", "genres")
    )

    catalog = (
        items.withColumn("sku", F.concat(F.lit("M"), F.col("item_id").cast("string")))
        .withColumn("style_color_id", F.col("sku"))
        .withColumn("product_id", F.col("sku"))
        .withColumn(
            "category",
            F.lower(
                F.coalesce(F.split(F.col("genres"), " ").getItem(0), F.lit("unknown"))
            ),
        )
        .withColumn(
            "attributes",
            _attributes_map(
                ("genres", F.col("genres")),
                ("release_year", F.col("release_year")),
            ),
        )
        .select(
            "sku",
            "style_color_id",
            "product_id",
            "category",
            F.lit("movielens").alias("brand"),
            "attributes",
            F.lit(None).cast("double").alias("list_price"),
        )
    )
    return _movielens_frames(raw, "M", catalog)


ADAPTERS: dict[str, Callable[[SparkSession, Config], tuple[DataFrame, DataFrame]]] = {
    "synthetic": load_synthetic,
    "rees46": load_rees46,
    "hm": load_hm,
    "ml1m": load_ml1m,
    "ml100k": load_ml100k,
}


def load_dataset(spark: SparkSession, cfg: Config) -> tuple[DataFrame, DataFrame]:
    name = cfg.dataset_name
    if name not in ADAPTERS:
        raise ValueError(f"unknown dataset '{name}'. Available: {sorted(ADAPTERS)}")
    interactions, catalog = ADAPTERS[name](spark, cfg)
    return _with_consent_column(interactions), catalog


def _with_consent_column(interactions: DataFrame) -> DataFrame:
    """Guarantee a ``consent`` column so every layer downstream has one shape.

    Adapters for public datasets have nothing to put here — MovieLens and
    the H&M competition data carry no consent signal, because they were not
    collected through a consent flow that anyone recorded. Filling in a
    plausible value would be inventing a legal fact, so the column is null
    and `privacy.consented` denies everything except `service`.

    Null is therefore the correct value *and* the value that would silently
    empty the training set. That tension is resolved in `jobs/silver.py`,
    which refuses to run with enforcement on and no consent data rather than
    quietly dropping every row or quietly keeping them.
    """
    from pyspark.sql import functions as F

    if "consent" in interactions.columns:
        return interactions.withColumn("consent", F.col("consent").cast("string"))
    return interactions.withColumn("consent", F.lit(None).cast("string"))
