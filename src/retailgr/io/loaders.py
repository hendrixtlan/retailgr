"""Load gold tables out of the warehouse and into memory for training.

Stage 1 fits the sequences in RAM on purpose: the point is a fast loop over
data and model choices, not a distributed trainer. A production HSTU run
replaces this with a sharded reader over the same tables.

Each history carries three parallel arrays — tokens, action ids and timestamps.
Item-only models use the first; HSTU uses all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retailgr.actions import encode_actions
from retailgr.config import Config


@dataclass
class HistoryBatch:
    """A slice of user histories handed to a model for scoring."""

    tokens: list[np.ndarray]
    actions: list[np.ndarray]
    timestamps: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class SequenceSplit:
    """One split, as plain Python lists.

    ``returned`` is the ranker's delayed label: for each position, whether
    that purchase came back. See ``jobs.sequences`` for the label values.
    """

    user_ids: list[str] = field(default_factory=list)
    inputs: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    timestamps: list[np.ndarray] = field(default_factory=list)
    returned: list[np.ndarray] = field(default_factory=list)
    # The action taken on each target, aligned with ``targets``. Needed to
    # calibrate a head against the outcome it actually predicts: "did they
    # engage" and "did they buy" are different labels on the same slate.
    target_actions: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.user_ids)

    def batch(self, start: int = 0, end: int | None = None) -> HistoryBatch:
        stop = len(self.user_ids) if end is None else end
        return HistoryBatch(
            tokens=self.inputs[start:stop],
            actions=self.actions[start:stop],
            timestamps=self.timestamps[start:stop],
        )


@dataclass
class VariantData:
    variant: str
    vocab_size: int  # includes the padding id at index 0
    token_by_id: dict[int, str]
    category_by_id: dict[int, str]
    train: SequenceSplit
    val: SequenceSplit
    test: SequenceSplit


def _parquet_path(cfg: Config, table: str) -> Path:
    layer, name = table.split(".", 1)
    return cfg.warehouse_root / layer / name


def _read_table_rows(cfg: Config, table: str, columns: list[str]) -> list[dict[str, Any]]:
    """Read a warehouse table as a list of row dicts."""
    if cfg.warehouse_backend == "iceberg":
        from retailgr.io import iceberg as pyiceberg_io

        # No JVM if we can avoid one. This function's whole job is to get the
        # table into memory, and starting a Spark session to do that costs
        # seconds per run and a Maven round trip the training loop has no use
        # for. pyiceberg reads the same format directly.
        if pyiceberg_io.available() and str(
            cfg.get("warehouse.iceberg.engine", "auto")
        ).lower() in {"auto", "pyiceberg"}:
            return pyiceberg_io.read_rows(cfg, table, columns)

        from retailgr.io.tables import read_table
        from retailgr.spark_session import build_spark

        spark = build_spark(cfg)
        frame = read_table(spark, cfg, table).select(*columns)
        return [row.asDict(recursive=True) for row in frame.collect()]

    import pyarrow.dataset as ds

    path = _parquet_path(cfg, table)
    if not path.exists():
        raise FileNotFoundError(f"table '{table}' not found at {path}; run the pipeline first")
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    wanted = [c for c in columns if c in available]
    table_data = dataset.to_table(columns=wanted)
    return table_data.to_pylist()


def load_variant(cfg: Config, variant: str) -> VariantData:
    """Load the vocabulary and the three splits for one granularity variant."""
    vocab_rows = _read_table_rows(cfg, f"gold.vocab_{variant}", ["token_id", "token", "category"])
    token_by_id = {int(row["token_id"]): str(row["token"]) for row in vocab_rows}
    category_by_id = {int(row["token_id"]): str(row.get("category") or "") for row in vocab_rows}
    # Index 0 is reserved for padding, so the embedding table is max_id + 1.
    vocab_size = (max(token_by_id) + 1) if token_by_id else 1

    rows = _read_table_rows(
        cfg,
        f"gold.sequences_{variant}",
        [
            "user_id",
            "input_tokens",
            "input_actions",
            "input_ts",
            "input_returned",
            "target_tokens",
            "target_actions",
            "split",
        ],
    )

    buckets: dict[str, SequenceSplit] = {
        name: SequenceSplit() for name in ("train", "val", "test")
    }
    for row in rows:
        split = str(row["split"])
        if split not in buckets:
            continue
        tokens = np.asarray(row["input_tokens"] or [], dtype=np.int64)
        actions = encode_actions(row.get("input_actions"))
        timestamps = np.asarray(row.get("input_ts") or [], dtype=np.int64)
        returned = np.asarray(
            row.get("input_returned") if row.get("input_returned") is not None else [],
            dtype=np.int64,
        )
        # The arrays are written together by the sequence builder; if a table
        # predates one of them, fall back to neutral values rather than
        # failing. -1 means "not a purchase", so a missing column disables the
        # return head instead of teaching it something false.
        if actions.size != tokens.size:
            actions = np.zeros(tokens.size, dtype=np.int64)
        if timestamps.size != tokens.size:
            timestamps = np.arange(tokens.size, dtype=np.int64)
        if returned.size != tokens.size:
            returned = np.full(tokens.size, -1, dtype=np.int64)

        bucket = buckets[split]
        bucket.user_ids.append(str(row["user_id"]))
        bucket.inputs.append(tokens)
        bucket.actions.append(actions)
        bucket.timestamps.append(timestamps)
        bucket.returned.append(returned)
        target_tokens = np.asarray(row["target_tokens"] or [], dtype=np.int64)
        target_actions = np.asarray(row.get("target_actions") or [], dtype=np.int64)
        if target_actions.size != target_tokens.size:
            # A table written before target actions existed. 0 is the padding
            # id, which every head treats as "no qualifying action", so the
            # heads simply go uncalibrated rather than being fed a guess.
            target_actions = np.zeros(target_tokens.size, dtype=np.int64)
        bucket.targets.append(target_tokens)
        bucket.target_actions.append(target_actions)

    return VariantData(
        variant=variant,
        vocab_size=vocab_size,
        token_by_id=token_by_id,
        category_by_id=category_by_id,
        train=buckets["train"],
        val=buckets["val"],
        test=buckets["test"],
    )
