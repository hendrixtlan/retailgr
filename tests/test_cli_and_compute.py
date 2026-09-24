"""Tests for the config override mechanism and the thread budget.

The ``--set`` flag exists because cleaning thresholds tuned for a retail event
stream are wrong for other data: MovieLens power users rate hundreds of films
in a day and the bot filter throws them out. Getting the parsing wrong there
silently changes what the pipeline trains on, so it is worth testing.
"""

from __future__ import annotations

import pytest

from retailgr.cli import _apply_dotted, _parse_scalar, build_parser
from retailgr.compute import available_cores, set_thread_budget

# -- override parsing ---------------------------------------------------------


def test_scalars_keep_their_types():
    assert _parse_scalar("100000") == 100000
    assert _parse_scalar("0.85") == pytest.approx(0.85)
    assert _parse_scalar("true") is True
    assert _parse_scalar("false") is False
    assert _parse_scalar("[a, b]") == ["a", "b"]
    assert _parse_scalar("parquet") == "parquet"


def test_dotted_paths_build_nested_mappings():
    target: dict = {}
    _apply_dotted(target, "clean.bot_events_per_day", 100000)
    _apply_dotted(target, "clean.max_events_per_user", 50)
    _apply_dotted(target, "warehouse.root", "/tmp/x")
    assert target == {
        "clean": {"bot_events_per_day": 100000, "max_events_per_user": 50},
        "warehouse": {"root": "/tmp/x"},
    }


def test_setting_through_a_scalar_is_refused():
    target: dict = {"clean": 5}
    with pytest.raises(ValueError, match="not a mapping"):
        _apply_dotted(target, "clean.threshold", 1)


def test_overrides_reach_the_loaded_config():
    from retailgr.cli import _load_config

    parser = build_parser()
    args = parser.parse_args(
        [
            "silver",
            "--dataset",
            "ml1m",
            "--set",
            "clean.bot_events_per_day=100000",
            "--set",
            "sequences.max_len=77",
        ]
    )
    cfg = _load_config(args)
    assert cfg.dataset_name == "ml1m"
    assert cfg.get("clean.bot_events_per_day") == 100000
    assert cfg.get("sequences.max_len") == 77
    # Untouched values keep their file defaults.
    assert cfg.get("evaluation.exclude_seen") is True


def test_malformed_override_is_rejected():
    from retailgr.cli import _load_config

    args = build_parser().parse_args(["silver", "--set", "nonsense"])
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        _load_config(args)


def test_every_subcommand_accepts_the_common_flags():
    parser = build_parser()
    for command in (
        "ingest",
        "silver",
        "sequences",
        "evaluate",
        "experiment",
        "ablate",
        "export-model",
        "bootstrap",
        "bench",
    ):
        args = parser.parse_args([command, "--set", "spark.master=local[1]"])
        assert args.overrides == ["spark.master=local[1]"]


# -- thread budget ------------------------------------------------------------


def test_available_cores_is_at_least_one():
    assert available_cores() >= 1


def test_thread_budget_is_applied_and_returned():
    applied = set_thread_budget(threads=2)
    assert applied == 2
    import torch

    assert torch.get_num_threads() == 2


def test_serving_defaults_to_a_single_thread():
    """Serving wants one thread per request and many processes, not one
    process fighting itself for cores."""
    assert set_thread_budget(serving=True) == 1


def test_thread_budget_is_clamped_to_at_least_one():
    assert set_thread_budget(threads=0) == 1
    assert set_thread_budget(threads=-4) == 1
