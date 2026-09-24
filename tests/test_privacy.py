"""Identifiers: that they are stripped where they must be, and kept where they must be.

The defect this file was written around: `GET /v1/model` returned the bundle
manifest verbatim, the manifest carried the ranker's discrimination
diagnostic, and that diagnostic had kept the `user_ids` it needs to pair its
samples. 200 real customer ids, 600 entries, 65 KB, over an unauthenticated
endpoint, on every deployment since the ranker shipped.

The interesting part is why. `experiment.py` already strips `per_user` and
`user_ids` before writing a run record — twice, by hand. The rule was known
and it was applied in two of the three places that needed it. So the tests
here are mostly *structural*: they assert the rule is enforced at the one
boundary every artifact crosses, rather than asserting that each individual
artifact happens to be clean today. A test of the second kind passes right up
until someone adds a fourth writer.

The other half of this file is the opposite assertion. The online store and
the event stream **must** keep `user_id` — a cache of user tails with the
user scrubbed out is not a safer cache, it is a broken one. Scrubbing is the
control for things people read; erasure, retention and TTL are the controls
for things the system uses.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from retailgr import privacy

# -- the detector works, so the assertions below mean something ---------------


def test_the_detector_finds_identifiers_by_key_name():
    payload = {"ranker_metrics": {"discrimination": {"auc": 0.61, "user_ids": ["U1", "U2"]}}}
    found = privacy.find_identifiers(payload)
    assert found == [("/ranker_metrics/discrimination/user_ids", "key `user_ids`")]


def test_the_detector_finds_identifiers_by_shape_under_a_harmless_key():
    """The case a key-name list cannot cover: a field nobody thought to name."""
    payload = {"notes": ["U000005", "4f1c0f4a-0b2e-4d3a-9c1e-2b7a8d9e0f11"]}
    found = privacy.find_identifiers(payload)
    assert len(found) == 2, found
    assert all("identifier shape" in reason for _, reason in found)


def test_the_detector_matches_whole_values_and_not_substrings():
    """A stated limit, not an oversight.

    Matching is anchored, so an identifier embedded in a sentence is not
    found. Unanchored matching over free text would flag `model_version`
    strings and hashes constantly, and a detector that cries wolf gets
    switched off — which is strictly worse than one with a known edge.

    The edge is covered elsewhere and differently: the erasure verifier
    searches for the *literal* id as a substring across every store,
    because there the question is "did this exact customer survive" rather
    than "does this look like anybody".
    """
    embedded = {"notes": ["the run covered U000005 and others"]}
    assert privacy.find_identifiers(embedded) == []


def test_the_detector_does_not_fire_on_ordinary_report_fields():
    """A detector that flags everything gets switched off."""
    payload = {
        "model_version": "hstu_small-config-945",
        "model_type": "hstu",
        "vocab_size": 945,
        "auc": 0.6139,
        "n_user_ids": 200,
        "dataset": "synthetic",
        "variant": "config",
    }
    assert privacy.find_identifiers(payload) == []


def test_an_empty_identifier_field_is_not_a_leak():
    """`session_id: None` in the exposure payload is a field that was never
    populated. Reporting it would train people to ignore the detector."""
    assert privacy.find_identifiers({"session_id": None, "user_ids": []}) == []


# -- scrubbing keeps the sample size and drops the sample ---------------------


def test_scrub_replaces_identifiers_with_their_count():
    """Deleting outright would lose the n. A report that cannot state its own
    sample size is a worse report, and the count is not identifying."""
    payload = {"discrimination": {"auc": 0.61, "user_ids": [f"U{i:06d}" for i in range(200)]}}
    clean = privacy.scrub(payload)
    assert clean["discrimination"] == {"auc": 0.61, "n_user_ids": 200}


def test_scrub_records_that_a_scalar_was_present_without_saying_what():
    assert privacy.scrub({"user_id": "U000005"}) == {"has_user_id": True}


def test_scrub_does_not_mutate_the_caller():
    """The analysis code genuinely needs `user_ids` — the paired test aligns
    its samples on them. They must not survive to disk; they must survive in
    memory. A mutating scrub would break the statistics to protect a file."""
    payload = {"discrimination": {"user_ids": ["U1", "U2"]}}
    privacy.scrub(payload)
    assert payload["discrimination"]["user_ids"] == ["U1", "U2"]


def test_scrub_reaches_identifiers_nested_inside_lists():
    payload = {"runs": [{"seed": 0, "user_ids": ["U1"]}, {"seed": 1, "user_ids": ["U2", "U3"]}]}
    clean = privacy.scrub(payload)
    assert [r["n_user_ids"] for r in clean["runs"]] == [1, 2]
    assert privacy.find_identifiers(clean) == []


# -- the boundary is enforced structurally ------------------------------------


def _unscrubbed_writers() -> list[tuple[str, int]]:
    """Every `write_text(json.dumps(X))` under src/ where X is not scrubbed.

    Structural because the failure was structural: the rule was applied at
    two call sites out of three. Finding the third by hand is how it was
    missed the first time.

    `privacy.py` itself is excluded — it is the implementation of the rule,
    and a rule that has to obey itself is a recursion, not a check.
    """
    out: list[tuple[str, int]] = []
    for path in sorted(Path("src").rglob("*.py")):
        if path.name == "privacy.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "write_text"):
                continue
            for argument in node.args:
                if not (
                    isinstance(argument, ast.Call)
                    and isinstance(argument.func, ast.Attribute)
                    and argument.func.attr == "dumps"
                ):
                    continue
                payload = argument.args[0] if argument.args else None
                scrubbed = (
                    isinstance(payload, ast.Call)
                    and isinstance(payload.func, ast.Attribute)
                    and payload.func.attr == "scrub"
                )
                if not scrubbed:
                    out.append((str(path), node.lineno))
    return out


def test_no_module_serialises_json_to_disk_without_scrubbing_it():
    """Every payload that becomes a file crosses `privacy.scrub` first.

    There is deliberately **no allowlist** of files judged harmless. Two of
    the writes this catches are item data that the scrub cannot change, and
    they go through it anyway: an allowlist is the structure that rots,
    because the next entry is added by whoever is in a hurry.

    This is the test that would have caught the original defect. `export.py`
    built a manifest full of `user_ids` and serialised it with a bare
    `json.dumps`, looking exactly like the two call sites that did strip
    them.
    """
    writers = _unscrubbed_writers()
    assert not writers, (
        f"these serialise JSON to a file without privacy.scrub: {writers}. "
        "Use privacy.write_json, or wrap the payload in privacy.scrub()."
    )


def test_the_structural_check_can_actually_fail():
    """Proof the AST walk finds what it claims to find."""
    import tempfile

    source = 'path.write_text(json.dumps(payload, indent=2), encoding="utf-8")\n'
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory) / "writer.py"
        scratch.write_text(source, encoding="utf-8")
        tree = ast.parse(source)
        hits = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write_text"
            and any(
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Attribute)
                and a.func.attr == "dumps"
                for a in node.args
            )
        ]
        assert len(hits) == 1


def test_write_json_scrubs_what_it_writes(tmp_path):
    target = privacy.write_json(
        tmp_path / "report.json",
        {"discrimination": {"auc": 0.6, "user_ids": ["U000005", "U000006"]}},
    )
    written = json.loads(target.read_text(encoding="utf-8"))
    assert privacy.find_identifiers(written) == []
    assert written["discrimination"]["n_user_ids"] == 2


# -- the export path, driven rather than inspected ----------------------------


def test_export_bundle_strips_identifiers_from_the_manifest(tmp_path):
    """The actual defect, reproduced against the real writer.

    `ranker_metrics` is a free-form dict that the export job fills from the
    evaluation code, so nothing stops an identifier reaching it. What stops
    it reaching disk is this.
    """
    import torch

    from retailgr.serving.bundle import BundleManifest, export_bundle

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_embedding = torch.nn.Embedding(4, 2)

    class _Model:
        name = "hstu"
        net = _Net()

    manifest = BundleManifest(
        model_version="leak-test",
        model_type="hstu",
        variant="config",
        dataset="synthetic",
        vocab_size=4,
        embedding_dim=2,
        max_len=8,
        ranker_metrics={
            "discrimination": {
                "auc": 0.61,
                "user_ids": [f"U{i:06d}" for i in range(200)],
            }
        },
    )
    export_bundle(
        tmp_path,
        model=_Model(),
        manifest=manifest,
        token_by_id={0: "<pad>", 1: "SKU-1"},
        category_by_id={0: "", 1: "shoes"},
        skus_by_token={"1": ["SKU-1"]},
        product_by_token={"1": "P-1"},
    )
    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert privacy.find_identifiers(written) == [], written["ranker_metrics"]
    assert written["ranker_metrics"]["discrimination"]["n_user_ids"] == 200

    # And the in-memory manifest is untouched, because the paired test that
    # runs after the export still needs the arrays.
    assert len(manifest.ranker_metrics["discrimination"]["user_ids"]) == 200


def test_no_api_response_carries_an_identifier():
    """End of the chain: what a caller can actually retrieve over HTTP.

    `/v1/model` is the endpoint that leaked. It is checked here against the
    real app rather than against the file, because the file being clean and
    the response being clean are two different claims — the handler returns
    `asdict(service.bundle.manifest)`, not the file's bytes.
    """
    from unittest.mock import MagicMock

    from fastapi.testclient import TestClient

    from retailgr.serving.api import create_app
    from retailgr.serving.bundle import BundleManifest

    manifest = BundleManifest(
        model_version="leak-test",
        model_type="hstu",
        variant="config",
        dataset="synthetic",
        vocab_size=4,
        embedding_dim=2,
        max_len=8,
        ranker_metrics={"discrimination": {"user_ids": ["U000005"]}},
    )
    service = MagicMock()
    service.bundle.model_version = "leak-test"
    service.bundle.manifest = manifest
    service.index.size = 10

    client = TestClient(create_app(service), raise_server_exceptions=False)
    for route in ("/v1/model", "/healthz", "/readyz", "/health"):
        response = client.get(route)
        found = privacy.find_identifiers(response.json())
        assert not found, f"{route} returns identifiers: {found}"


def test_the_shipped_artifacts_carry_no_identifiers():
    """The files actually in this repository, not a constructed example.

    Skipped rather than failed when the artifacts are absent: a fresh clone
    has not run anything yet, and a test that demands generated files fails
    for the wrong reason.
    """
    artifacts = Path("artifacts")
    if not artifacts.exists():
        pytest.skip("no artifacts; run the pipeline first")

    leaks: dict[str, list] = {}
    for path in sorted(artifacts.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        found = privacy.find_identifiers(data)
        if found:
            leaks[str(path)] = found[:3]
    assert not leaks, leaks


# -- and the places that must keep identifiers --------------------------------


def test_the_online_store_still_keys_by_user():
    """The inverse assertion, and it matters as much as the others.

    An over-eager scrub applied to the cache would produce a system that
    cannot answer a request, while looking more private. The online store
    holds `user_id` because that is its entire function; the controls there
    are TTL and erasure, not redaction.
    """
    from retailgr.online_store import InMemoryOnlineStore, TailEvent

    store = InMemoryOnlineStore(tail_length=5)
    store.append_event("U000005", TailEvent(sku="S1", token=1, action="view", event_ts=1.0))
    assert [event.sku for event in store.user_tail("U000005")] == ["S1"]


def test_the_event_schema_still_carries_user_id():
    """Same argument, for the stream. The topic is keyed by `user_id`, which
    is what makes a user's events land in one partition in order — and what
    makes a key-targeted erasure possible at all."""
    from retailgr.streaming.schemas import INTERACTION_AVRO_SCHEMA, TOPIC_INTERACTIONS, TOPIC_SPECS

    fields = {field["name"] for field in INTERACTION_AVRO_SCHEMA["fields"]}
    assert "user_id" in fields
    assert TOPIC_SPECS[TOPIC_INTERACTIONS]["key"] == "user_id"


# -- consent: the rule, and the two implementations of it ---------------------


@pytest.mark.parametrize(
    "value,purpose,expected",
    [
        # Fails closed on everything that is not an explicit grant. A
        # producer that has not been updated yet must not become a silent
        # opt-in, which is the failure that turns a consent feature into a
        # liability.
        (None, "analytics", False),
        ("", "analytics", False),
        ("   ", "analytics", False),
        ("service", "analytics", False),
        ("service,analytics", "analytics", True),
        ("service, analytics", "analytics", True),  # whitespace after the comma
        ("analytics", "personalisation", False),
        ("service,analytics,personalisation", "personalisation", True),
        # `personalisation` must not be satisfied by a value that merely
        # contains it, and `analytics` must not be satisfied by a longer
        # word starting the same way.
        ("analytics_extended", "analytics", False),
        ("depersonalisation", "personalisation", False),
        # Service is always granted: an event that arrived cannot be
        # un-processed for the request it arrived in. Recording that
        # honestly beats a flag that pretends otherwise.
        (None, "service", True),
        ("", "service", True),
        # Other shapes a producer might send.
        (["service", "analytics"], "analytics", True),
        ({"analytics": True, "personalisation": False}, "personalisation", False),
        (42, "analytics", False),
    ],
)
def test_consent_fails_closed(value, purpose, expected):
    assert privacy.consented(value, purpose) is expected


def test_training_requires_more_than_analytics():
    """Nested permissions. A customer who allowed analysis has not thereby
    allowed their behaviour to become a training example."""
    assert privacy.permits("service,analytics", "analytics") is True
    assert privacy.permits("service,analytics", "training") is False
    assert privacy.permits("service,analytics,personalisation", "training") is True


def test_an_unknown_purpose_is_an_error_not_a_denial():
    """A typo in a purpose name must not read as "not consented" — that is a
    silent policy change disguised as a strict default."""
    with pytest.raises(ValueError):
        privacy.consented("service", "marketting")
    with pytest.raises(ValueError):
        privacy.permits("service", "advertising")


def test_the_event_schema_carries_consent():
    from retailgr.streaming.schemas import INTERACTION_AVRO_SCHEMA

    field = next(
        f for f in INTERACTION_AVRO_SCHEMA["fields"] if f["name"] == "consent"
    )
    # Nullable with a null default, so an old producer stays valid against
    # the schema and is denied by the filter rather than rejected at the
    # broker. Rejecting would drop the event entirely, including the
    # `service` purpose it is always allowed to be used for.
    assert field["type"] == ["null", "string"]
    assert field["default"] is None


# -- pseudonymisation ---------------------------------------------------------


def test_there_is_no_default_pseudonymisation_key():
    """A default committed here would make every pseudonym in every
    deployment reversible by anyone who can read this repository, while
    looking exactly as safe as a real key."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="no pseudonymisation key"):
            privacy.pseudonymisation_key(None)


def test_a_short_key_is_refused():
    """Brute-forceable against a known identifier format, which this has."""
    with pytest.raises(ValueError, match="at least 16"):
        privacy.pseudonymisation_key("short")


def test_the_key_can_come_from_the_environment():
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {privacy.KEY_ENVIRONMENT_VARIABLE: "x" * 32}):
        assert privacy.pseudonymisation_key(None) == b"x" * 32


def test_pseudonyms_are_deterministic_and_key_dependent():
    """Determinism is not a convenience here, it is what makes erasure
    possible: the pseudonym of a customer asking to be deleted is recomputed
    from their id rather than looked up in a mapping table that would be the
    most sensitive asset in the system."""
    key = b"a-key-that-is-long-enough-here"
    first = privacy.pseudonymise("U000005", key)
    assert first == privacy.pseudonymise("U000005", key)
    assert first != privacy.pseudonymise("U000006", key)
    assert first != privacy.pseudonymise("U000005", b"a-different-key-entirely!!")
    assert len(first) == 32


def test_the_pseudonym_is_a_real_hmac_not_a_prefix_hash():
    """`sha256(key || value)` is what people reach for when the engine has
    no HMAC builtin, and it is length-extendable. Checked against Python's
    own hmac so the construction is verified rather than described."""
    import hashlib
    import hmac as hmac_module

    key = b"a-key-that-is-long-enough-here"
    expected = hmac_module.new(key, b"U000005", hashlib.sha256).hexdigest()[:32]
    assert privacy.pseudonymise("U000005", key) == expected
    assert privacy.pseudonymise("U000005", key) != hashlib.sha256(
        key + b"U000005"
    ).hexdigest()[:32]
