"""Identifiers: where they are allowed to be, and how to get rid of them.

This module exists because of a specific defect. `GET /v1/model` returned the
bundle manifest verbatim, the manifest carried `ranker_metrics`, and the
discrimination diagnostic had kept the `user_ids` it needed to pair its
samples. The result was **200 real customer ids, 600 entries, 65 KB, served
over an unauthenticated HTTP endpoint** — on every deployment, since the
ranker was added.

The cause is more interesting than the bug. `experiment.py` already strips
`per_user` and `user_ids` before writing a run record, in two places, by
hand. The rule was known. It was applied twice and forgotten once, because
"remember to strip identifiers at each call site" is not a rule a codebase
can keep. So the scrub here happens at the **serialisation boundary** — the
moment a structure stops being an in-process object and becomes a file or a
response body — and not at any call site. The analysis code keeps its
`user_ids`, because the paired test genuinely needs them to align samples;
they simply do not survive the trip to disk.

**Scrubbing replaces rather than deletes.** `user_ids: [200 ids]` becomes
`n_user_ids: 200`. The fact that the diagnostic ran over 200 users is a
property of the experiment and belongs in the record; which 200 does not.
Deleting outright would lose the sample size, and a report that cannot state
its own n is a worse report.

**Pseudonymisation is not anonymisation, and this module will not pretend
otherwise.** `pseudonymise` is a keyed HMAC: it stops a warehouse dump from
naming customers directly, it makes the identifier space uniform across
datasets, and it gives erasure a second option (destroy the key, lose the
mapping). It does *not* make the data anonymous. A pseudonymous id attached
to a fifty-event purchase history is re-identifiable by anyone holding a
second copy of those purchases, which for a retailer is every payment
processor they use. Treat a pseudonymised warehouse as pseudonymous data
with all the obligations that carries, not as anonymous data with none.

**Consent is a field, not a filter.** The schema carries what the customer
agreed to; `consented` decides what that means for a given purpose. Keeping
the two apart is what makes it possible to answer "what would the model look
like if we only trained on opted-in events" — which this repository does
measure, because the honest version of a consent feature includes its cost.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "CONSENT_PURPOSES",
    "IDENTIFIER_KEYS",
    "consented",
    "find_identifiers",
    "looks_like_identifier",
    "pseudonymise",
    "pseudonymisation_key",
    "scrub",
]

# Keys whose values are identifiers, or collections of them. Matched exactly
# rather than by substring: `model_version` contains no identifier and
# `vocab_size` is not a user. The plural and per-user forms are here because
# those are the ones that actually leaked.
IDENTIFIER_KEYS: frozenset[str] = frozenset(
    {
        "user_id",
        "user_ids",
        "per_user",
        "device_id",
        "device_ids",
        "session_id",
        "session_ids",
        "order_id",
        "order_ids",
        "customer_id",
        "customer_ids",
        "email",
    }
)

# What a customer can agree to, separately. One flag for everything is the
# design that forces "accept all or leave", and it is also the design that
# makes it impossible to answer which processing a given event supports.
CONSENT_PURPOSES: tuple[str, ...] = (
    # Answer this request from this session. Without it there is no service,
    # so it is not really a choice and is treated as always granted.
    "service",
    # Keep the event in the lakehouse past the request that produced it.
    "analytics",
    # Use the event as a training example.
    "personalisation",
)

# The purposes each consumer of the data requires.
PURPOSE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "serving": ("service",),
    "analytics": ("analytics",),
    "training": ("analytics", "personalisation"),
}

KEY_ENVIRONMENT_VARIABLE = "RETAILGR_PSEUDONYM_KEY"

# `U000005`, `device-9f3a`, an email, a uuid. Deliberately broad: this drives
# a *detector* used by tests and by the erasure verifier, where a false
# positive costs a moment and a false negative is the whole point missed.
_IDENTIFIER_SHAPES = (
    re.compile(r"^U\d{4,}$"),  # this project's synthetic ids
    re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
    re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
)


def looks_like_identifier(value: Any) -> bool:
    """Whether a bare value has the shape of an identifier.

    Used where the key name is not available — inside a list, or when
    scanning a rendered document. Shape-based detection is a heuristic and is
    never the only check: `IDENTIFIER_KEYS` catches by name, this catches by
    appearance, and the erasure verifier searches for the literal id.
    """
    if not isinstance(value, str):
        return False
    return any(shape.match(value) for shape in _IDENTIFIER_SHAPES)


def scrub(value: Any, *, keys: Iterable[str] = IDENTIFIER_KEYS) -> Any:
    """A copy of ``value`` with identifier-bearing keys replaced by counts.

    Recursive and non-mutating: the caller's structure is untouched, so the
    analysis code can go on using the arrays it needs while the serialised
    copy does not carry them. A list of identifiers becomes its length under
    an ``n_`` key; a scalar becomes ``True`` under a ``has_`` key, which
    records that the field was populated without saying what it held.
    """
    keys = frozenset(keys)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in keys:
                if isinstance(item, (list, tuple, set)):
                    out[f"n_{key}"] = len(item)
                elif isinstance(item, dict):
                    out[f"n_{key}"] = len(item)
                elif item is not None:
                    out[f"has_{key}"] = True
                continue
            out[key] = scrub(item, keys=keys)
        return out
    if isinstance(value, list):
        return [scrub(item, keys=keys) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub(item, keys=keys) for item in value)
    return value


def write_json(path: Any, payload: Any, *, indent: int = 2) -> Any:
    """Write a report or record to disk, scrubbed.

    The one door every artifact goes through. `tests/test_privacy.py`
    asserts no module writes JSON into the artifacts directory any other
    way, which is what stops the next report from repeating the defect that
    produced this module.

    Deliberately **not** used for the event stream or the online store.
    Those hold `user_id` because holding it is their job — a cache of user
    tails with the user scrubbed out is not a safer cache, it is a broken
    one. Erasure, retention and TTL are the controls there; scrubbing is the
    control for things that get read by people.
    """
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(scrub(payload), indent=indent), encoding="utf-8")
    return target


def find_identifiers(value: Any, *, path: str = "") -> list[tuple[str, str]]:
    """Every place an identifier survives in ``value``, as (path, reason).

    The inverse of `scrub`, and the thing tests assert on. Reports both the
    keys it recognises by name and the values it recognises by shape, so a
    field nobody thought to name still shows up.
    """
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{path}/{key}"
            if key in IDENTIFIER_KEYS and item not in (None, [], {}, ""):
                found.append((here, f"key `{key}`"))
                continue
            found.extend(find_identifiers(item, path=here))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(find_identifiers(item, path=f"{path}[{index}]"))
    elif looks_like_identifier(value):
        found.append((path, f"value {value!r} has identifier shape"))
    return found


# -- pseudonymisation ---------------------------------------------------------


def pseudonymisation_key(configured: str | None = None) -> bytes:
    """The HMAC key, from config or the environment.

    Order is config, then `RETAILGR_PSEUDONYM_KEY`, then a refusal. There is
    deliberately **no default key**: a default would be published in this
    repository, which would make every pseudonym in every deployment
    reversible by anyone who can read it, while looking exactly as safe as a
    real one. Failing loudly is the only honest behaviour when the secret is
    missing.
    """
    material = configured or os.environ.get(KEY_ENVIRONMENT_VARIABLE)
    if not material:
        raise ValueError(
            "no pseudonymisation key: set privacy.pseudonymisation.key in the "
            f"config or the {KEY_ENVIRONMENT_VARIABLE} environment variable. "
            "There is no default, because a default committed to this "
            "repository would make every pseudonym reversible."
        )
    if len(material) < 16:
        raise ValueError(
            f"pseudonymisation key is {len(material)} characters; at least 16 "
            "are required. A short key is brute-forceable against a known "
            "identifier format, which this one has."
        )
    return material.encode("utf-8")


def pseudonymise(identifier: str, key: bytes, *, length: int = 32) -> str:
    """A stable pseudonym for ``identifier`` under ``key``.

    HMAC-SHA256 rather than a bare hash. A bare `sha256(user_id)` over an
    identifier space this small — `U000000` to `U999999` is a million
    candidates — is reversed by a laptop in seconds, and salting per row
    would destroy the joins the warehouse is built on. The key makes the
    space unsearchable without it; determinism keeps `user_id` usable as a
    join key and, importantly, keeps **erasure possible**: the pseudonym for
    a customer asking to be deleted can be recomputed rather than looked up
    in a mapping table that would itself be the most sensitive asset here.
    """
    digest = hmac.new(key, identifier.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:length]


# -- consent ------------------------------------------------------------------


def consented(value: Any, purpose: str) -> bool:
    """Whether an event's consent field permits ``purpose``.

    Fails **closed**: an absent, unparseable or unknown consent value denies
    everything except `service`. The alternative — treating missing consent
    as granted — makes every producer that has not been updated yet into a
    silent opt-in, which is the failure mode that turns a consent feature
    into a liability.

    `service` is always granted, because an event that arrived cannot be
    un-processed for the purpose of answering the request it arrived in.
    Recording that honestly is better than a flag that pretends otherwise.
    """
    if purpose not in CONSENT_PURPOSES:
        raise ValueError(f"unknown purpose '{purpose}'; known: {CONSENT_PURPOSES}")
    if purpose == "service":
        return True
    if not value:
        return False
    if isinstance(value, str):
        granted = {part.strip() for part in value.split(",") if part.strip()}
    elif isinstance(value, (list, tuple, set, frozenset)):
        granted = {str(part).strip() for part in value}
    elif isinstance(value, dict):
        granted = {str(k) for k, v in value.items() if v}
    else:
        return False
    return purpose in granted


def permits(value: Any, consumer: str) -> bool:
    """Whether an event may be used by ``consumer`` (serving/analytics/training)."""
    if consumer not in PURPOSE_REQUIREMENTS:
        raise ValueError(
            f"unknown consumer '{consumer}'; known: {sorted(PURPOSE_REQUIREMENTS)}"
        )
    return all(consented(value, purpose) for purpose in PURPOSE_REQUIREMENTS[consumer])


_SHA256_BLOCK = 64


def _hmac_pads(key: bytes) -> tuple[bytes, bytes]:
    """The two key blocks HMAC prepends, precomputed.

    They depend only on the key, so building them once in Python turns HMAC
    into two `sha2` calls Spark can evaluate natively — no UDF, no
    interpreter round-trip for every row of the largest table here.
    """
    if len(key) > _SHA256_BLOCK:
        key = hashlib.sha256(key).digest()
    padded = key.ljust(_SHA256_BLOCK, b"\x00")
    return (
        bytes(byte ^ 0x36 for byte in padded),
        bytes(byte ^ 0x5C for byte in padded),
    )


def pseudonym_column(column: str, key: bytes, length: int = 32) -> Any:
    """:func:`pseudonymise`, as a Spark expression over ``column``.

    Real HMAC and not `sha2(key || value)`. The prefix construction is the
    one people reach for because Spark has no `hmac` builtin, and it is
    length-extendable — which for pseudonymisation is a smaller problem than
    for authentication, but "smaller problem" is a poor reason to ship the
    wrong primitive when the right one is two hashes and the key blocks can
    be precomputed.

    The equality that matters is with the Python implementation:
    `tests/test_privacy.py` asserts this column and :func:`pseudonymise`
    produce the same string for the same input. Erasure depends on it — the
    pseudonym of the customer to delete is computed in Python and matched
    against rows Spark wrote. If the two ever diverged, erasure would report
    success having deleted nothing.
    """
    from pyspark.sql import functions as F

    k_ipad, k_opad = _hmac_pads(key)
    message = F.encode(F.col(column).cast("string"), "utf-8")
    inner = F.sha2(F.concat(F.lit(k_ipad), message), 256)
    digest = F.sha2(F.concat(F.lit(k_opad), F.unhex(inner)), 256)
    return F.substring(digest, 1, length)


def consent_column(consumer: str, column: str = "consent") -> Any:
    """The same rule as :func:`permits`, as a Spark expression.

    Written twice on purpose and **tested for agreement**, the same bargain
    `granularity.py` makes for SKU-to-token. A Python UDF would keep one
    implementation, at the cost of serialising every row through the
    interpreter on the one filter that runs over the whole event table.

    Writing it down twice is the hazard, so `tests/test_privacy.py` runs both
    over the same table of awkward values — null, empty, whitespace, an
    unknown purpose, a purpose that is a prefix of another — and fails if
    they ever disagree. A consent filter that behaves differently in the
    batch path and the request path is the worst possible outcome here: it
    makes the system's answer to "may we use this" depend on which code
    asked.
    """
    from pyspark.sql import functions as F

    if consumer not in PURPOSE_REQUIREMENTS:
        raise ValueError(
            f"unknown consumer '{consumer}'; known: {sorted(PURPOSE_REQUIREMENTS)}"
        )
    required = [p for p in PURPOSE_REQUIREMENTS[consumer] if p != "service"]
    if not required:
        return F.lit(True)

    # Trim each element so " analytics" matches, and compare whole elements
    # so `personalisation` is not satisfied by a value containing it.
    #
    # The lambda is load-bearing and the obvious `F.transform(..., F.trim)`
    # is a trap. `F.trim` takes an optional second argument (the characters
    # to strip), so `transform` reads its arity as the `(element, index)`
    # form and hands it the index — producing the *untrimmed* array, with no
    # error. The consent test caught it on `"service, analytics"`, which is
    # what a producer that joins with ", " sends, i.e. most of them. The
    # symptom would have been every such customer silently denied analytics.
    granted = F.transform(
        F.split(F.coalesce(F.col(column), F.lit("")), ","), lambda part: F.trim(part)
    )
    condition = F.array_contains(granted, required[0])
    for purpose in required[1:]:
        condition = condition & F.array_contains(granted, purpose)
    # `array_contains` returns null when the array is null; fail closed.
    return F.coalesce(condition, F.lit(False))
