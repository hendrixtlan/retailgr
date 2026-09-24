"""Tests for the platform audit.

An audit that passes is worthless until you have seen it fail. Every check
here is exercised twice: once against a violation it must catch, and once
against the legitimate form of the same thing, which it must let through. The
second half is the harder one — a lock-in scanner that flags every string
containing a colon would "pass" this project only by making the check
meaningless.

The last test runs the audit against the real source tree, which is the one
that turns the README's architecture sentence into something CI can fail on.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from typing import Protocol

import pytest

from retailgr.platform import (
    SWAP_POINTS,
    audit,
    audit_lazy_imports,
    audit_swap_point,
    protocol_methods,
    render_audit,
    scan_hardcoded_endpoints,
    scan_vendor_imports,
)


def _module(tmp_path: Path, source: str, name: str = "sample.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return tmp_path


# -- 1. no proprietary dependency ---------------------------------------------


@pytest.mark.parametrize(
    "line,vendor",
    [
        ("import boto3", "AWS"),
        ("from google.cloud import storage", "Google Cloud"),
        ("import azure.storage.blob", "Azure"),
        ("from snowflake.connector import connect", "Snowflake"),
    ],
)
def test_a_vendor_sdk_import_is_caught(tmp_path, line, vendor):
    root = _module(tmp_path, f"{line}\n")
    findings = scan_vendor_imports(root)
    assert len(findings) == 1
    assert findings[0]["vendor"] == vendor


def test_a_lazy_vendor_import_is_still_a_vendor_import(tmp_path):
    """Hiding it inside a function changes when it loads, not what it ties
    the project to."""
    root = _module(
        tmp_path,
        """
        def upload():
            import boto3
            return boto3.client("s3")
        """,
    )
    assert len(scan_vendor_imports(root)) == 1


def test_open_components_are_not_mistaken_for_vendor_lock_in(tmp_path):
    """pyspark, pyarrow and a Kafka client are the open components the
    architecture is built on. A scanner that flags them has misunderstood the
    claim it is checking."""
    root = _module(
        tmp_path,
        """
        import pyarrow.dataset as ds
        import pyspark.sql.functions as F
        import confluent_kafka
        import redis
        """,
    )
    assert scan_vendor_imports(root) == []


# -- 2. no hardcoded endpoint -------------------------------------------------


def test_a_fixed_endpoint_is_caught(tmp_path):
    root = _module(
        tmp_path,
        """
        def connect():
            return open_socket("kafka.prod.internal:9092")
        """,
    )
    findings = scan_hardcoded_endpoints(root)
    assert len(findings) == 1
    assert findings[0]["value"] == "kafka.prod.internal:9092"


@pytest.mark.parametrize(
    "value",
    ["https://s3.eu-west-1.amazonaws.com", "10.4.2.11", "redis.internal:6379"],
)
def test_every_shape_of_fixed_address_is_caught(tmp_path, value):
    root = _module(tmp_path, f'ENDPOINT = "{value}"\n')
    assert [f["value"] for f in scan_hardcoded_endpoints(root)] == [value]


def test_a_configured_default_is_allowed(tmp_path):
    """The property being checked is *overridability*, not absence. An address
    written as a config default is one `--set` away from being someone
    else's."""
    root = _module(
        tmp_path,
        """
        def build(cfg):
            return Broker(cfg.get("streaming.bootstrap_servers", "localhost:9092"))
        """,
    )
    assert scan_hardcoded_endpoints(root) == []


def test_a_parameter_default_is_allowed(tmp_path):
    """Every caller can replace it, which is exactly how the configured value
    reaches `KafkaBroker`."""
    root = _module(
        tmp_path,
        """
        class KafkaBroker:
            def __init__(self, bootstrap_servers: str = "kafka.example.com:9092"):
                self.bootstrap_servers = bootstrap_servers
        """,
    )
    assert scan_hardcoded_endpoints(root) == []


def test_an_address_in_a_docstring_is_documentation_not_configuration(tmp_path):
    root = _module(
        tmp_path,
        '''
        """Run the API on http://127.0.0.1:8080 and the catalog on
        http://localhost:8181."""

        def serve():
            """Point a browser at http://localhost:9001 for the console."""
            return None
        ''',
    )
    assert scan_hardcoded_endpoints(root) == []


def test_loopback_bind_addresses_are_not_endpoints(tmp_path):
    """`0.0.0.0` is what a server binds to, not somewhere it connects."""
    root = _module(tmp_path, 'HOST = "0.0.0.0"\nOTHER = "127.0.0.1"\n')
    assert scan_hardcoded_endpoints(root) == []


# -- 3. optional clients load on demand ---------------------------------------


def test_an_eagerly_imported_optional_client_is_caught(tmp_path):
    """A top-level `import redis` turns an optional backend into a required
    dependency of the whole package."""
    root = _module(tmp_path, "import redis\n\n\nclass Store:\n    pass\n")
    findings = audit_lazy_imports(root)
    assert len(findings) == 1
    assert findings[0]["module"] == "redis"


def test_the_same_import_inside_a_function_is_fine(tmp_path):
    root = _module(
        tmp_path,
        """
        class Store:
            def __init__(self, url):
                import redis

                self.client = redis.Redis.from_url(url)
        """,
    )
    assert audit_lazy_imports(root) == []


def test_importing_retailgr_needs_none_of_the_optional_clients():
    """The property the previous two tests exist to protect, checked on the
    real package by actually importing it.

    Run in a subprocess because this one is about what a *fresh* interpreter
    loads: inside the test session those modules may already be in
    ``sys.modules`` for entirely unrelated reasons, and asserting on that
    would prove nothing.
    """
    import subprocess
    import sys

    probe = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ("
        "  'retailgr.online_store', 'retailgr.streaming.broker',"
        "  'retailgr.serving.retrieval', 'retailgr.serving.factory')];"
        "print(sorted(m for m in ('confluent_kafka','redis','faiss') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert loaded == "[]", f"importing retailgr pulled in {loaded}"
    assert audit_lazy_imports(Path("src/retailgr")) == []


# -- 4. the swap points are real ----------------------------------------------


class _Example(Protocol):
    def read(self, key: str, default: int = 0) -> int: ...

    def close(self) -> None: ...


def test_protocol_methods_are_discovered_with_their_signatures():
    methods = protocol_methods(_Example)
    assert set(methods) == {"read", "close"}
    assert "default" in methods["read"].parameters


def test_a_backend_missing_a_method_does_not_conform():
    class Incomplete:
        def read(self, key: str, default: int = 0) -> int:
            return default

    point = _swap_point_for(Incomplete)
    entry = audit_swap_point(point)["implementations"]["candidate"]
    assert entry["conforms"] is False
    assert entry["missing"] == ["close"]


def test_a_backend_with_an_incompatible_signature_does_not_conform():
    """The failure Python will not report until that code path runs: the
    method exists, and calling it the way the protocol says blows up."""

    class Narrower:
        def read(self, key: str, default: int) -> int:  # default is now required
            return default

        def close(self) -> None:
            return None

    entry = audit_swap_point(_swap_point_for(Narrower))["implementations"]["candidate"]
    assert entry["conforms"] is False
    assert "default" in entry["mismatched"]["read"]


def test_a_backend_demanding_an_extra_argument_does_not_conform():
    class Demanding:
        def read(self, key: str, default: int = 0, *, tenant: str) -> int:
            return default

        def close(self) -> None:
            return None

    entry = audit_swap_point(_swap_point_for(Demanding))["implementations"]["candidate"]
    assert entry["conforms"] is False
    assert "tenant" in entry["mismatched"]["read"]


def test_a_conforming_backend_passes_including_extra_optional_arguments():
    class Generous:
        def read(self, key: str, default: int = 0, retries: int = 3) -> int:
            return default

        def close(self) -> None:
            return None

    entry = audit_swap_point(_swap_point_for(Generous))["implementations"]["candidate"]
    assert entry["conforms"] is True


def test_kwargs_absorbs_the_protocol():
    class Flexible:
        def read(self, key: str, **kwargs) -> int:
            return 0

        def close(self) -> None:
            return None

    entry = audit_swap_point(_swap_point_for(Flexible))["implementations"]["candidate"]
    assert entry["conforms"] is True


def _swap_point_for(implementation: type):
    """A swap point whose single implementation is the class under test.

    ``SwapPoint`` addresses implementations as ``"module:Name"``, the way the
    real registry does, so a class defined inside a test function has to be
    published on this module before it can be resolved. Doing it here rather
    than declaring six classes at module scope keeps each case next to the
    assertion that explains it.
    """
    from retailgr.platform import SwapPoint

    setattr(inspect.getmodule(_swap_point_for), implementation.__name__, implementation)
    return SwapPoint(
        name="example",
        setting="example.backend",
        protocol=f"{__name__}:_Example",
        default="candidate",
        implementations={"candidate": f"{__name__}:{implementation.__name__}"},
    )


# -- the real thing -----------------------------------------------------------


def test_the_project_passes_its_own_platform_audit():
    """The architecture sentence, as a test. If someone adds an `import
    boto3` to the serving path or pins a bucket URL, this is what says so."""
    result = audit(Path("src/retailgr"))
    assert result["passed"], result["failures"]


def test_every_advertised_backend_conforms_to_its_protocol():
    """Including the ones no test can instantiate here. Kafka, Redis and
    FAISS are absent from this environment, so an integration test would skip
    them and report green — which is how a backend stays broken until someone
    selects it in production."""
    result = audit(Path("src/retailgr"))
    checked = 0
    for point in result["swap_points"]:
        for name, entry in point["implementations"].items():
            if not entry.get("checked"):
                continue
            checked += 1
            assert entry["conforms"], (point["name"], name, entry)
    # The four class-based backends plus their in-memory counterparts.
    assert checked >= 6


def test_the_audit_covers_every_swap_point_the_config_can_select():
    settings = {point.setting for point in SWAP_POINTS}
    assert settings == {
        "warehouse.backend",
        "streaming.broker",
        "online_store.backend",
        "serving.index",
    }


def test_the_report_names_what_it_did_not_establish():
    """A platform audit that reads as "portable, proven" would be the most
    misleading document in the repository."""
    text = render_audit(audit(Path("src/retailgr")))
    assert "PASSED" in text
    assert "does **not** establish" in text
    assert "any particular cloud" in text


# -- the dependencies are declared, not inherited -----------------------------


def _third_party_imports() -> dict[str, set[str]]:
    """Every non-stdlib, non-first-party module this repository imports."""
    import ast
    import sys

    found: dict[str, set[str]] = {}
    stdlib = set(sys.stdlib_module_names)
    for root in ("src", "tests", "scripts"):
        for path in Path(root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    head = name.split(".")[0]
                    # `tests` is first-party too: `test_deploy.py` imports
                    # `test_alerts.VALIDATED_KINDS` so the CRD exemption in
                    # one file has to be covered by the other.
                    if head in stdlib or head in {"retailgr", "tests"}:
                        continue
                    found.setdefault(head, set()).add(str(path))
    return found


# Distributions whose import name differs from their package name.
DISTRIBUTION_ALIASES = {
    "kafka-python": "kafka",
    "faiss-cpu": "faiss",
    "pyyaml": "yaml",
    "kubernetes-validate": "kubernetes_validate",
}


def _declared_imports() -> set[str]:
    import re

    import tomllib

    project = tomllib.load(open("pyproject.toml", "rb"))["project"]
    specs = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        specs.extend(group)

    declared = set()
    for spec in specs:
        name = re.split(r"[<>=\[; ]", spec)[0].strip().lower()
        declared.add(DISTRIBUTION_ALIASES.get(name, name.replace("-", "_")))
    return declared


def test_every_third_party_import_is_declared():
    """Inheriting a dependency is not the same as having one.

    `api.py` imports pydantic at module level and pydantic was never declared
    — it worked because FastAPI happens to pull it in, which is a fact about
    FastAPI's packaging rather than about this project. The day that changes,
    the failure arrives as an ImportError in a serving pod.
    """
    undeclared = sorted(set(_third_party_imports()) - _declared_imports())
    assert not undeclared, f"imported but not declared in pyproject.toml: {undeclared}"


def test_the_audit_is_looking_at_something():
    """A scan that finds no imports would pass the test above forever."""
    found = _third_party_imports()
    assert len(found) >= 10, f"only found {sorted(found)}"
    assert {"numpy", "torch", "pytest"} <= set(found)
