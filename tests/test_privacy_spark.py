"""The consent rule and the pseudonym, written twice and checked for agreement.

`privacy.py` implements both in Python and again as Spark expressions. That
is the same bargain `granularity.py` makes for SKU-to-token, and for the same
reason: a Python UDF keeps one implementation at the cost of serialising
every row of the largest table in the warehouse through the interpreter, on
the one filter that runs over all of it.

Writing it down twice is the hazard the whole `schemas.py` module exists to
avoid elsewhere, so it does not get to pass unguarded here. Both failures it
could produce are silent:

- **A consent filter that disagrees between the batch path and the request
  path** makes the system's answer to "may we use this event" depend on
  which code asked. Nothing errors; the two paths simply hold different
  data, and the difference shows up as an unreproducible metric months later.
- **A pseudonym that differs between Spark and Python breaks erasure.**
  Silver rows are written by Spark; the pseudonym of the customer to delete
  is computed in Python and matched against them. If the two ever diverged,
  `retailgr forget` would match nothing, delete nothing, and report success
  — which is precisely the failure mode `erasure.py` was built to prevent.

These need a Spark session, so they are marked slow and skipped by default.
That is a real gap: the check that matters most runs least often. It is run
by `make test-all`, and the agreement is also asserted at the vector level
against Python's own `hmac` module in `test_privacy.py`, which runs always.
"""

from __future__ import annotations

import pytest

from retailgr import privacy
from retailgr.config import Config

pytestmark = pytest.mark.slow

pytest.importorskip("pyspark")


@pytest.fixture(scope="module")
def spark():
    from retailgr.spark_session import build_spark

    session = build_spark(Config.load())
    yield session
    session.stop()


# Values chosen to break a careless implementation rather than to pass a
# careful one: null, empty, whitespace, a purpose that is a prefix of
# another, a purpose that contains another, and the comma-space a real
# producer emits.
CONSENT_CASES = [
    None,
    "",
    "   ",
    "service",
    "analytics",
    "service,analytics",
    "service, analytics",
    "service,analytics,personalisation",
    "personalisation",
    "analytics_extended",
    "depersonalisation",
    "SERVICE,ANALYTICS",
    ",,,",
    "analytics,",
]


@pytest.mark.parametrize("consumer", ["serving", "analytics", "training"])
def test_the_spark_consent_filter_agrees_with_the_python_one(spark, consumer):
    """Row by row, over every awkward value, for every consumer."""
    rows = [(index, value) for index, value in enumerate(CONSENT_CASES)]
    frame = spark.createDataFrame(rows, "id int, consent string")

    kept = {
        row["id"]
        for row in frame.filter(privacy.consent_column(consumer)).select("id").collect()
    }
    expected = {
        index for index, value in rows if privacy.permits(value, consumer)
    }
    disagreements = {
        CONSENT_CASES[index]
        for index in (kept ^ expected)
    }
    assert kept == expected, (
        f"consumer={consumer}: Spark and Python disagree on {disagreements}"
    )


def test_serving_consent_keeps_everything(spark):
    """`service` is always granted, so the serving filter is a pass-through.

    Asserted rather than assumed: a filter that silently dropped rows here
    would empty the request path's view of its own data.
    """
    frame = spark.createDataFrame(
        [(index, value) for index, value in enumerate(CONSENT_CASES)],
        "id int, consent string",
    )
    assert frame.filter(privacy.consent_column("serving")).count() == len(CONSENT_CASES)


def test_the_consent_filter_is_not_vacuous(spark):
    """Proof the parametrised test above can fail.

    A filter that kept everything would agree with a Python function that
    also kept everything, and the pair would pass while enforcing nothing.
    """
    frame = spark.createDataFrame(
        [(index, value) for index, value in enumerate(CONSENT_CASES)],
        "id int, consent string",
    )
    kept = frame.filter(privacy.consent_column("training")).count()
    assert 0 < kept < len(CONSENT_CASES), kept


IDENTIFIERS = [
    "U000005",
    "U999999",
    "",
    "a b,c",
    "device-9f3a-2b",
    "U000005 ",  # trailing space: a different customer as far as a join knows
    "ünïcødé-id",
    "x" * 300,
]


def test_the_spark_pseudonym_equals_the_python_one(spark):
    """The equality erasure depends on.

    Silver is written by Spark; `retailgr forget` computes the pseudonym in
    Python and matches against those rows. A divergence here would make
    every erasure match nothing, delete nothing, and report success.
    """
    key = b"a-test-key-that-is-long-enough"
    frame = spark.createDataFrame([(value,) for value in IDENTIFIERS], "user_id string")
    got = (
        frame.withColumn("p", privacy.pseudonym_column("user_id", key, 32))
        .select("user_id", "p")
        .collect()
    )
    for row in got:
        assert row["p"] == privacy.pseudonymise(row["user_id"], key, length=32), row["user_id"]


def test_the_spark_pseudonym_depends_on_the_key(spark):
    """Otherwise the column is a plain hash wearing a key's name."""
    frame = spark.createDataFrame([("U000005",)], "user_id string")
    one = privacy.pseudonym_column("user_id", b"key-one-long-enough!!", 32)
    two = privacy.pseudonym_column("user_id", b"key-two-long-enough!!", 32)
    assert frame.withColumn("p", one).first()["p"] != frame.withColumn("p", two).first()["p"]


def test_a_key_longer_than_the_hash_block_still_agrees(spark):
    """HMAC hashes a key longer than 64 bytes before padding it. Getting that
    branch wrong produces a column that works, is stable, and does not match
    Python — which is the worst of the three outcomes."""
    key = b"k" * 200
    frame = spark.createDataFrame([("U000005",)], "user_id string")
    got = frame.withColumn("p", privacy.pseudonym_column("user_id", key, 32)).first()["p"]
    assert got == privacy.pseudonymise("U000005", key, length=32)


def test_silver_refuses_to_run_when_consent_is_enforced_and_entirely_absent(spark, tmp_path):
    """The one case where both silent options are wrong.

    Keeping everything turns an un-updated producer into a blanket opt-in.
    Dropping everything produces an empty warehouse that reads like a broken
    join and gets debugged for a day. Refusing is the only answer that tells
    the operator what decision they are actually making.
    """
    from retailgr.jobs.silver import _apply_consent

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"privacy": {"consent": {"enforce": True, "consumer": "analytics"}}},
    )
    frame = spark.createDataFrame([("U1", None), ("U2", None)], "user_id string, consent string")
    with pytest.raises(ValueError, match="every `consent` value is null"):
        _apply_consent(cfg, frame)


def test_silver_refuses_when_the_column_is_missing_entirely(spark):
    from retailgr.jobs.silver import _apply_consent

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"privacy": {"consent": {"enforce": True}}},
    )
    frame = spark.createDataFrame([("U1",)], "user_id string")
    with pytest.raises(ValueError, match="no `consent` column"):
        _apply_consent(cfg, frame)


def test_disabling_enforcement_is_a_deliberate_configuration(spark):
    """A public research dataset genuinely has no consent signal. That is
    configuration, not an exception to swallow — and the stats say which
    mode the run was in, so a report cannot be read as enforced when it
    was not."""
    from retailgr.jobs.silver import _apply_consent

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={"privacy": {"consent": {"enforce": False}}},
    )
    frame = spark.createDataFrame([("U1", None)], "user_id string, consent string")
    kept, stats = _apply_consent(cfg, frame)
    assert kept.count() == 1
    assert stats["consent_enforced"] == 0


def test_the_pseudonymisation_step_reports_a_key_fingerprint_and_not_the_key(spark):
    """"Was this table written with the key we still have" is a question
    erasure needs answered. The key itself must not be the way it is."""
    from retailgr.jobs.silver import _pseudonymise

    cfg = Config.load(
        "configs/pipeline.yaml",
        overrides={
            "privacy": {
                "pseudonymisation": {
                    "enabled": True,
                    "key": "a-test-key-that-is-long-enough",
                    "length": 32,
                }
            }
        },
    )
    frame = spark.createDataFrame([("U000005",)], "user_id string")
    out, stats = _pseudonymise(cfg, frame)

    assert out.first()["user_id"] != "U000005"
    assert stats["pseudonymised"] == 1
    fingerprint = stats["pseudonym_key_fingerprint"]
    assert "a-test-key-that-is-long-enough" not in fingerprint
    assert len(fingerprint) == 12
