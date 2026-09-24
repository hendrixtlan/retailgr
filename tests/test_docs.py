"""Does the README describe the code that exists?

Documentation drift is the failure this project is most exposed to, because
the README is not decoration here — it carries the measured numbers and the
claims the whole thing is judged on. Three kinds of drift are mechanical
enough to catch, and this session produced examples of two of them:

* a command that was removed but still named as an instruction;
* a file or directory that moved;
* a count that quietly stopped matching.

What this cannot check is whether a *number* in the prose is still the number
a fresh run produces. That one needs the run, and `artifacts/` is where it
lands.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pytest

README = Path("README.md")

# Commands the README names in order to say they do NOT exist. Removing one
# from the code and leaving the explanation behind is correct; silently
# instructing someone to run it is not.
KNOWN_ABSENT = {"fetch-bundle"}


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_every_cli_command_the_readme_names_exists(readme):
    from retailgr.cli import build_parser

    subparsers = [
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    known = set(subparsers[0].choices)

    named = set(re.findall(r"retailgr ([a-z][a-z-]+)", readme))
    missing = named - known - KNOWN_ABSENT
    assert not missing, f"README instructs `retailgr {missing}`, which does not exist"


def test_every_absent_command_is_described_as_absent(readme):
    """The other half: if `fetch-bundle` ever comes back, this list is stale
    and the exemption above would hide a real command from the check."""
    from retailgr.cli import build_parser

    subparsers = [
        action
        for action in build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    for command in KNOWN_ABSENT:
        assert command not in set(subparsers[0].choices), (
            f"`{command}` exists now; remove it from KNOWN_ABSENT so it is checked"
        )


def test_every_make_target_the_readme_shows_exists(readme):
    """Only targets inside a shell code fence — `make those numbers` is
    English, and a checker that cannot tell the difference gets ignored."""
    targets = set(re.findall(r"^([a-z][a-z0-9-]*):", Path("Makefile").read_text(), re.M))

    shown: set[str] = set()
    for block in re.findall(r"```bash\n(.*?)```", readme, re.S):
        shown |= set(re.findall(r"^\s*make ([a-z][a-z0-9-]*)", block, re.M))

    assert shown, "no `make` invocations found in the README; has the quick start moved?"
    missing = shown - targets
    assert not missing, f"README shows `make {missing}`, which the Makefile does not define"


def test_every_path_the_readme_names_exists(readme):
    paths = set(
        re.findall(r"`((?:src/|tests/|deploy/|configs/|docker/|scripts/)[\w./-]+)`", readme)
    )
    assert len(paths) > 10, "the README stopped naming files; this check is no longer doing work"
    missing = sorted(path for path in paths if not Path(path).exists())
    assert not missing, f"README names paths that do not exist: {missing}"


def test_the_test_count_has_not_drifted_far(readme):
    """Not exact — a number that fails on every added test gets deleted
    rather than fixed. Far enough out and it is misinformation."""
    claimed = re.search(r"tests/\s+(\d+) tests", readme)
    assert claimed, "the README no longer states a test count"

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", collected.stdout)
    assert match, collected.stdout[-500:]

    stated, actual = int(claimed.group(1)), int(match.group(1))
    drift = abs(stated - actual) / actual
    assert drift < 0.1, f"README says {stated} tests, the suite has {actual}"


def test_the_unverified_list_names_kafka_and_not_the_two_that_now_run(readme):
    """The claim most likely to go stale in the flattering direction: three
    backends were unverified, two now run, and the paragraph saying so has to
    keep up."""
    section = readme[readme.index("What is still not executed") :][:1200]
    assert "Kafka" in section
    assert "redis" not in section.lower(), "Redis runs now; the unverified list still names it"
    assert "Iceberg" not in section, "Iceberg runs now; the unverified list still names it"
