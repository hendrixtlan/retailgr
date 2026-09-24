"""Two storage backends, one answer.

`warehouse.backend` is advertised as a swap: Parquet on a laptop, Iceberg on
the lakehouse, same pipeline either way. Nothing had ever checked that the
answer was the same, and running it found that it was not — not because the
data changed, but because `approxQuantile` computes the time split from a
*summary per partition*, and the two backends lay the data out differently.
The train/val boundary landed 1,586 seconds apart, and four users changed
split.

That is the worst shape a bug can take here: two runs on identical data stop
being comparable, every metric shifts a little, and nothing says why. So the
determinism is asserted directly, and the whole pipeline is run through both
backends and diffed.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from retailgr.config import Config
from retailgr.jobs import ingest, sequences, silver

pytestmark = pytest.mark.slow

pytest.importorskip("pyiceberg")

SMALL = {
    "dataset": {"synthetic": {"n_users": 150, "n_products": 50, "days": 40, "seed": 5}},
    "clean": {"min_events_per_user": 3},
    "sequences": {"max_len": 50},
    "spark": {"master": "local[2]", "shuffle_partitions": 2},
}


def _config(root, backend: str, raw) -> Config:
    overrides = dict(SMALL)
    overrides["dataset"] = dict(overrides["dataset"], name="synthetic", raw_path=str(raw))
    overrides["warehouse"] = {
        "backend": backend,
        "root": str(root),
        "iceberg": {"engine": "pyiceberg", "catalog_name": "retailgr", "uri": "", "warehouse": ""},
    }
    return Config.load(overrides=overrides)


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    """Run the pipeline twice over identical raw data, once per backend."""
    from retailgr.datasets.synthetic import SyntheticConfig, generate
    from retailgr.spark_session import build_spark, stop_spark

    root = tmp_path_factory.mktemp("equivalence")
    raw = root / "raw"
    generate(raw / "synthetic", SyntheticConfig(**SMALL["dataset"]["synthetic"]))

    results = {}
    for backend in ("parquet", "iceberg"):
        cfg = _config(root / backend, backend, raw)
        spark = build_spark(cfg)
        ingest.run(spark, cfg)
        silver.run(spark, cfg)
        result = sequences.run(spark, cfg, granularity="config", variant="config")
        results[backend] = (cfg, result.stats)
        stop_spark()

    yield results
    shutil.rmtree(root, ignore_errors=True)


def test_both_backends_choose_the_same_cutoffs(both):
    parquet = both["parquet"][1]
    iceberg = both["iceberg"][1]
    assert parquet["cutoff_train_end_unix"] == iceberg["cutoff_train_end_unix"]
    assert parquet["cutoff_val_end_unix"] == iceberg["cutoff_val_end_unix"]


def test_both_backends_put_the_same_users_in_the_same_splits(both):
    parquet = both["parquet"][1]
    iceberg = both["iceberg"][1]
    for key in ("train_users", "val_users", "test_users", "vocab_size"):
        assert parquet[key] == iceberg[key], key


def _digest(rows: list[dict], keys: list[str]) -> str:
    items = sorted(
        json.dumps({k: row[k] for k in keys if k in row}, sort_keys=True, default=str)
        for row in rows
    )
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def test_the_gold_tables_are_identical_not_merely_similar(both):
    """Row counts matching is weak evidence; the contents are what training
    reads."""
    from retailgr.io.loaders import _read_table_rows

    for table, keys in (
        ("gold.vocab_config", ["token_id", "token", "category"]),
        (
            "gold.sequences_config",
            ["user_id", "split", "input_tokens", "target_tokens", "target_actions"],
        ),
    ):
        digests = {
            backend: _digest(_read_table_rows(cfg, table, keys), keys)
            for backend, (cfg, _) in both.items()
        }
        assert digests["parquet"] == digests["iceberg"], f"{table} differs across backends"


def test_the_loaded_splits_match_too(both):
    """One level up from the tables: what `load_variant` hands the trainer."""
    from retailgr.io.loaders import load_variant

    loaded = {backend: load_variant(cfg, "config") for backend, (cfg, _) in both.items()}
    parquet, iceberg = loaded["parquet"], loaded["iceberg"]
    assert parquet.vocab_size == iceberg.vocab_size
    assert parquet.token_by_id == iceberg.token_by_id
    for split in ("train", "val", "test"):
        a = getattr(parquet, split)
        b = getattr(iceberg, split)
        assert sorted(a.user_ids) == sorted(b.user_ids), split
