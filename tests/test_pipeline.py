"""End-to-end pipeline test on a tiny synthetic dataset.

Slow (it starts Spark), so it is marked and can be skipped with
``pytest -m "not slow"``. It is the test that catches the expensive mistakes:
leakage across the time cutoff, tokens that escape the vocabulary, and splits
that silently come out empty.
"""

from __future__ import annotations

import shutil

import pytest

from retailgr.config import Config
from retailgr.granularity import GranularityResolver
from retailgr.io.loaders import load_variant
from retailgr.io.tables import read_table
from retailgr.jobs import ingest, sequences, silver

pytestmark = pytest.mark.slow

SMALL_DATASET = {
    "dataset": {"synthetic": {"n_users": 200, "n_products": 60, "days": 40, "seed": 3}},
    "clean": {"min_events_per_user": 3},
    "sequences": {"max_len": 50},
    "evaluation": {"k_values": [5, 20]},
}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Config:
    """Build the whole pipeline once in a throwaway directory."""
    root = tmp_path_factory.mktemp("retailgr")
    overrides = dict(SMALL_DATASET)
    overrides["warehouse"] = {"backend": "parquet", "root": str(root / "warehouse")}
    overrides["dataset"] = dict(overrides["dataset"], name="synthetic", raw_path=str(root / "raw"))
    overrides["spark"] = {"master": "local[2]", "shuffle_partitions": 2}
    cfg = Config.load(overrides=overrides)
    yield cfg
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def spark(workspace: Config):
    from retailgr.spark_session import build_spark

    session = build_spark(workspace)
    yield session


@pytest.fixture(scope="module")
def built(workspace: Config, spark):
    ingest.run(spark, workspace)
    stats = silver.run(spark, workspace)
    result = sequences.run(spark, workspace, granularity="config", variant="config")
    return stats, result


def test_silver_keeps_events_and_drops_nothing_unexpected(built):
    stats, _ = built
    assert stats["silver_events"] > 0
    assert stats["silver_users"] > 0
    # Synthetic event ids are unique, so dedupe must be a no-op. The
    # baseline is what *enters* dedupe, which is no longer the raw input:
    # the consent filter runs first, deliberately, so every count below it
    # is a count over data this pipeline is allowed to process rather than
    # one the reader has to adjust in their head.
    entering_dedupe = stats.get("after_consent", stats["input_events"])
    assert stats["after_dedupe"] == entering_dedupe


def test_the_consent_filter_removed_something_and_not_everything(built):
    """A consent step that drops nothing is not enforcing, and one that
    drops everything has misread the field. Both look like success in a
    stats dict nobody reads closely."""
    stats, _ = built
    assert stats["consent_enforced"] == 1, "the fixture is not exercising enforcement"
    dropped = stats["consent_dropped_events"]
    assert 0 < dropped < stats["input_events"], dropped
    assert stats["after_consent"] == stats["input_events"] - dropped


def test_every_split_is_populated(built):
    _, result = built
    assert result.stats["train_users"] > 0
    assert result.stats["val_users"] > 0
    assert result.stats["test_users"] > 0


def test_token_vocabulary_is_smaller_than_the_sku_catalog(built):
    _, result = built
    # The default config collapses sizes, so tokens must be fewer than SKUs.
    assert result.stats["vocab_size"] < result.stats["distinct_skus"]
    assert result.stats["compression_vs_sku"] > 1.0


def test_no_future_events_leak_into_the_training_history(workspace, spark, built):
    """Every val input event must sit strictly before the training cutoff."""
    from pyspark.sql import functions as F

    _, result = built
    t1 = result.stats["cutoff_train_end_unix"]
    frame = read_table(spark, workspace, result.sequences_table).filter(F.col("split") == "val")
    latest = frame.select(F.max(F.array_max("input_ts")).alias("latest")).first()["latest"]
    assert latest is not None
    assert latest < t1


def test_test_history_stops_at_the_second_cutoff(workspace, spark, built):
    from pyspark.sql import functions as F

    _, result = built
    t2 = result.stats["cutoff_val_end_unix"]
    frame = read_table(spark, workspace, result.sequences_table).filter(F.col("split") == "test")
    latest = frame.select(F.max(F.array_max("input_ts")).alias("latest")).first()["latest"]
    assert latest < t2


def test_loaded_tokens_all_live_inside_the_vocabulary(workspace, built):
    _, result = built
    data = load_variant(workspace, "config")
    for split in (data.train, data.val, data.test):
        for row in split.inputs:
            assert row.size == 0 or (row.min() >= 1 and row.max() < data.vocab_size)
        for row in split.targets:
            assert row.size == 0 or (row.min() >= 1 and row.max() < data.vocab_size)


def test_spark_and_python_resolvers_agree(workspace, spark):
    """The Spark expression and the pure-Python resolver must not drift."""

    resolver = GranularityResolver(workspace.granularity)
    hierarchy = read_table(spark, workspace, "silver.item_hierarchy")
    with_token = hierarchy.withColumn("token", resolver.token_column())
    rows = with_token.select(
        "sku", "style_color_id", "product_id", "category", "attributes", "token"
    ).limit(200).collect()
    assert rows
    for row in rows:
        item = row.asDict(recursive=True)
        expected = resolver.token_for(item)
        assert item["token"] == expected, item


def test_sku_variant_produces_one_token_per_sku(workspace, spark):
    result = sequences.run(spark, workspace, granularity="sku", variant="sku_test")
    # Every training token is a SKU, so the vocabulary cannot exceed the catalog.
    assert result.stats["vocab_size"] <= result.stats["distinct_skus"]
    assert result.stats["compression_vs_sku"] >= 1.0


def test_metrics_stay_inside_their_range(workspace, built):
    import numpy as np

    from retailgr.evaluation.evaluate import evaluate
    from retailgr.models.popularity import PopularityModel

    data = load_variant(workspace, "config")
    model = PopularityModel(data.vocab_size)
    model.fit(data.train)
    results = evaluate(model, data.test, data.vocab_size, [5, 20], exclude_seen=True)
    overall = results["overall"]
    for name, value in overall.items():
        if name == "eval_users":
            continue
        assert np.isfinite(value), name
        assert 0.0 <= value <= 1.0, (name, value)
    assert overall["eval_users"] > 0
    # Recall can only grow with k.
    assert overall["recall@20"] >= overall["recall@5"]


def test_sequence_model_beats_a_random_ranker(workspace, built):
    """The harness must be able to tell a model that uses history from one that
    does not. Random is measured, not assumed: on this generator a user's
    cluster affinity - not global popularity - carries the signal, so a
    theoretical 'k / vocab_size' would be the wrong yardstick."""
    import numpy as np

    from retailgr.config import load_model_config
    from retailgr.evaluation.evaluate import evaluate
    from retailgr.models.base import Recommender
    from retailgr.models.sasrec import SASRecModel

    class RandomModel(Recommender):
        name = "random"

        def __init__(self, vocab_size: int, seed: int = 0):
            self.vocab_size = vocab_size
            self.rng = np.random.default_rng(seed)

        def fit(self, train, val=None):
            return {}

        def score(self, batch):
            return self.rng.random((len(batch), self.vocab_size)).astype(np.float32)

    data = load_variant(workspace, "config")
    k = 20

    random_model = RandomModel(data.vocab_size, seed=11)
    random_recall = evaluate(random_model, data.test, data.vocab_size, [k])["overall"][
        f"recall@{k}"
    ]

    model_cfg = load_model_config("sasrec_small.yaml")
    model_cfg.update({"epochs": 6, "max_len": 30, "hidden_dim": 32})
    sasrec = SASRecModel(data.vocab_size, model_cfg)
    sasrec.fit(data.train)
    sasrec_recall = evaluate(sasrec, data.test, data.vocab_size, [k])["overall"][f"recall@{k}"]

    assert sasrec_recall > 1.3 * random_recall, (sasrec_recall, random_recall)


def test_hstu_trains_on_the_same_tables_as_sasrec(workspace, built):
    """HSTU must run end to end off the gold tables, reading the action and
    timestamp arrays the sequence builder wrote."""
    from retailgr.config import load_model_config
    from retailgr.evaluation.evaluate import evaluate
    from retailgr.models.hstu import HSTUModel

    data = load_variant(workspace, "config")
    # The loader must have populated all three parallel arrays.
    assert len(data.train.actions) == len(data.train.inputs)
    assert len(data.train.timestamps) == len(data.train.inputs)
    assert data.train.actions[0].size == data.train.inputs[0].size
    assert data.train.timestamps[0].size == data.train.inputs[0].size
    # Real action ids, not the zero fallback.
    assert max(int(a.max()) for a in data.train.actions if a.size) > 0

    model_cfg = load_model_config("hstu_small.yaml")
    model_cfg.update({"epochs": 4, "max_len": 30, "hidden_dim": 32})
    model = HSTUModel(data.vocab_size, model_cfg)
    stats = model.fit(data.train)
    assert stats["train_windows"] > 0

    results = evaluate(model, data.test, data.vocab_size, [20], exclude_seen=True)
    recall = results["overall"]["recall@20"]
    assert 0.0 <= recall <= 1.0
