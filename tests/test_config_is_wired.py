"""Every configuration key is read by some code.

This file exists because of four settings that were not. `pipeline.yaml`
carried a `privacy.retention` block with a paragraph explaining why bronze is
the shortest window — it is the only layer holding raw, un-pseudonymised
events — and `bronze_days`, `silver_days`, `gold_days` and
`snapshot_expiry_days` were read by nothing at all. The Redis TTL beside them
*was* wired, which is what made the block look finished.

That is worse than the settings being absent. Someone reading that file would
reasonably conclude bronze is pruned at thirty days, and the only way to find
out otherwise is to go looking for the job. A config key is an interface: it
promises the system behaves differently when you change it.

The same scan found `streaming.schema_registry_url`, pointing at a service
nothing in this repository contacts, under a comment claiming the Avro
serializer uses it — while the broker's own docstring correctly says the wire
format is JSON.

The check has to allow for blocks consumed wholesale — `PolicyConfig.from_dict(cfg.get("policy"))`,
`SyntheticConfig(**params)`, the granularity rules — so it looks for the
dotted path, or for the parent block being fetched *and* the leaf name
appearing as a string or a dataclass field. That is loose enough to have
false negatives; it is not loose enough to have missed the retention block,
which is the bar it was built to clear.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

CONFIGS = ("configs/pipeline.yaml", "configs/granularity.yaml")

# Keys that are deliberately read by nothing in this repository. Each needs a
# reason, and the reason has to be about the key rather than about the effort
# of wiring it — an allowlist with "TODO" in it is the thing this test exists
# to prevent.
DELIBERATELY_UNREAD: dict[str, str] = {
    "streaming.schema_registry_url": (
        "The wire format is JSON, not Avro-with-Schema-Registry — see "
        "broker.py's _serialise. The Avro schema in schemas.py is still "
        "load-bearing because the Spark type is derived from it, but no code "
        "contacts a registry. Kept because it is the endpoint a deployment "
        "would need the moment the serializer is swapped, and because "
        "docker-compose.yml starts the service."
    ),
}


def _leaves(node, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                out.extend(_leaves(value, path))
            else:
                out.append(path)
    return out


def _config_keys() -> list[str]:
    keys: list[str] = []
    for path in CONFIGS:
        keys += _leaves(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    return sorted(set(keys))


# Blocks whose *keys are data* rather than config names: a surface, a
# product category. Their leaf names cannot be looked for in the source
# because they are supplied by whoever writes the config. What has to be
# read is the block itself, and — where the block has a fixed inner shape —
# the names of those inner fields.
DATA_KEYED_BLOCKS: tuple[str, ...] = (
    "serving.exclude_seen",  # keys are surface names
    "by_category",  # keys are product categories
)


def _source() -> str:
    """Only `src/`.

    Including `tests/` here was a self-referential bug caught by this file's
    own negative control: a key named in a test and nowhere else counted as
    read, and the test naming it was *this* one. A config key read only by a
    test is not wired into anything.
    """
    return "\n".join(p.read_text(encoding="utf-8") for p in Path("src").rglob("*.py"))


KEYS = _config_keys()
SOURCE = _source()


def _reads_name(name: str, source: str) -> bool:
    return bool(
        re.search(rf'["\']{re.escape(name)}["\']', source)
        or re.search(rf"\b{re.escape(name)}\b\s*[:=]", source)
        or re.search(rf"\.{re.escape(name)}\b", source)
    )


def _is_read(key: str, source: str) -> bool:
    for block in DATA_KEYED_BLOCKS:
        if not key.startswith(block + "."):
            continue
        if block not in source:
            return False
        # Drop the data segment; anything after it is a fixed inner field and
        # does need a reader.
        rest = key[len(block) + 1 :].split(".")[1:]
        return all(_reads_name(part, source) for part in rest)

    if key in source:
        return True
    parts = key.split(".")
    parent, leaf = ".".join(parts[:-1]), parts[-1]
    if not parent or parent not in source:
        return False
    # The parent block is fetched wholesale. Accept the leaf as a quoted
    # string (`settings.get("enforce")`) or as a dataclass field / attribute
    # (`cfg.consent_analytics_rate`, `n_users: int = 3000`).
    return _reads_name(leaf, source)


def test_the_scan_found_keys_at_all():
    """A scan over zero keys passes for ever."""
    assert len(KEYS) > 50, len(KEYS)
    assert "privacy.retention.bronze_days" in KEYS


def test_every_config_key_is_read_by_some_code():
    orphans = [
        key for key in KEYS if key not in DELIBERATELY_UNREAD and not _is_read(key, SOURCE)
    ]
    assert not orphans, (
        f"these config keys are read by no code: {orphans}. A key is an "
        "interface — it promises the system behaves differently when you "
        "change it. Wire it, delete it, or add it to DELIBERATELY_UNREAD "
        "with a reason about the key."
    )


@pytest.mark.parametrize("key", sorted(DELIBERATELY_UNREAD))
def test_the_exemptions_are_still_unread_and_still_explained(key):
    """An exemption that becomes wired should lose its exemption, and one
    whose reason has rotted should be re-argued rather than inherited."""
    assert key in KEYS, f"{key} is exempted and no longer exists in the config"
    reason = DELIBERATELY_UNREAD[key]
    assert len(reason) > 80, f"{key}'s exemption has no real reason"
    for word in ("todo", "later", "for now", "temporar"):
        assert word not in reason.lower(), f"{key}'s reason is a deferral, not a reason"


def test_the_check_can_actually_fail():
    """Proof the matcher does not accept everything, which would make the
    test above green for every config ever written."""
    assert not _is_read("privacy.retention.invented_key_nobody_reads", SOURCE)
    assert _is_read("privacy.retention.bronze_days", SOURCE)


# -- the retention settings specifically, since they are why this file exists -


@pytest.mark.parametrize(
    "key",
    [
        "privacy.retention.bronze_days",
        "privacy.retention.silver_days",
        "privacy.retention.gold_days",
        "privacy.retention.snapshot_expiry_days",
        "privacy.retention.online_store_ttl_seconds",
        "privacy.retention.clock",
        "privacy.retention.max_delete_share",
    ],
)
def test_each_retention_setting_reaches_code_that_uses_it(key):
    """Named one by one rather than covered by the sweep above: these four
    were the defect, and a parametrised list makes a regression name itself."""
    leaf = key.split(".")[-1]
    retention = Path("src/retailgr/jobs/retention.py").read_text(encoding="utf-8")
    store = Path("src/retailgr/online_store.py").read_text(encoding="utf-8")
    assert leaf in retention or leaf in store, f"{key} is not read by any job"
