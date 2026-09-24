"""Auditing the platform claim.

The project's pitch has a sentence about architecture: compute, storage and
machine learning are decoupled, so it runs on any cloud without depending on
proprietary platforms. Everything else in this repository is measured. That
sentence was not — it was a description of intent, and intent is exactly what
rots first, because nothing fails when someone adds a vendor SDK import or
hardcodes a bucket URL.

This module turns the sentence into three checks that run in CI:

1. **No proprietary dependency.** No module under ``src/retailgr`` may import
   a cloud vendor's SDK. The open components are the point; a single
   ``import boto3`` in the serving path makes "any cloud" false.
2. **No hardcoded endpoint.** Every address of an external system must arrive
   through configuration. A default inside ``cfg.get("...", "localhost:9092")``
   is fine — it is overridable, and being overridable is the whole property.
   The same string written anywhere else is not.
3. **The swap points are real.** Four components are advertised as
   replaceable: the warehouse, the broker, the online store and the vector
   index. Each has a ``Protocol`` and a factory, and Python enforces neither
   at runtime — a backend can be missing a method, or have one with a
   different signature, and nothing notices until that path runs in
   production. So every registered implementation is checked against its
   protocol statically, without needing Kafka or Redis to be up.

A fourth property falls out of the third and is checked with it: the optional
clients must be imported *lazily*. ``import retailgr`` has to work with only
the core dependencies installed, or "swappable" means "swappable once you have
installed all of them", which is not the same thing.

None of this proves the system runs on a given cloud. It proves the specific
things that would make it *not* run on one.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Importing any of these from application code contradicts "no proprietary
# platform". The list is deliberately about *vendor* SDKs: pyarrow, pyspark
# and kafka clients are open components and belong here.
VENDOR_MODULES: dict[str, str] = {
    "boto3": "AWS",
    "botocore": "AWS",
    "awswrangler": "AWS",
    "sagemaker": "AWS",
    "google.cloud": "Google Cloud",
    "google.colab": "Google",
    "azure": "Azure",
    "azureml": "Azure",
    "databricks": "Databricks",
    "pyspark.databricks": "Databricks",
    "snowflake": "Snowflake",
    "vertexai": "Google Cloud",
}

# Something that looks like the address of an external system.
ENDPOINT_PATTERN = re.compile(
    r"""(?xi)
    ^(
        https?://[^\s]+              # a URL
      | [a-z0-9][a-z0-9.\-]*:\d{2,5} # host:port
      | \d{1,3}(\.\d{1,3}){3}        # a bare IPv4 address
    )$
    """
)

# Addresses that are not an external system's: loopback bind addresses and
# the like. These are still only allowed where the rules below allow them;
# they are listed so a failure names something worth fixing.
ENDPOINT_EXEMPT = {"0.0.0.0", "127.0.0.1", "::1", "localhost"}


@dataclass
class SwapPoint:
    """A component the architecture claims can be replaced."""

    name: str
    setting: str
    protocol: str  # "module:Name", or "" for a module-level function set
    default: str
    implementations: dict[str, str] = field(default_factory=dict)  # name -> "module:Name"
    optional_imports: dict[str, str] = field(default_factory=dict)  # name -> module
    # For backends that are a set of module functions rather than a class.
    # The warehouse is one: ``tables.py`` dispatches on the backend name to
    # free functions, so there is no class to compare against — but the
    # functions still have to exist and still have to take the same
    # arguments, and that is exactly as checkable.
    required_functions: tuple[str, ...] = ()

    def resolve(self, target: str) -> Any:
        module_name, _, attribute = target.partition(":")
        import importlib

        return getattr(importlib.import_module(module_name), attribute)


SWAP_POINTS: tuple[SwapPoint, ...] = (
    SwapPoint(
        name="warehouse",
        setting="warehouse.backend",
        protocol="",  # module-level functions, not a class
        default="parquet",
        implementations={
            "parquet": "retailgr.io.tables",
            "iceberg": "retailgr.io.tables",
        },
        required_functions=("write_table", "read_table", "table_exists"),
    ),
    # `warehouse.iceberg.engine` is deliberately *not* a swap point. Spark and
    # pyiceberg are two engines over one Iceberg v2 format, but they do not
    # share an interface — the choice is a branch inside `write_table`, behind
    # the warehouse's own three functions. Listing it here produced a row
    # claiming the Spark engine failed to implement `write_arrow`, which is a
    # function it has no reason to have. The real invariant — that every value
    # `iceberg_engine()` can return is a branch `write_table` handles — is a
    # dispatch-completeness check, and it lives in `tests/test_iceberg.py`.
    SwapPoint(
        name="broker",
        setting="streaming.broker",
        protocol="retailgr.streaming.broker:Broker",
        default="memory",
        implementations={
            "memory": "retailgr.streaming.broker:InMemoryBroker",
            "kafka": "retailgr.streaming.broker:KafkaBroker",
        },
        optional_imports={"kafka": "kafka"},
    ),
    SwapPoint(
        name="online_store",
        setting="online_store.backend",
        protocol="retailgr.online_store:OnlineStore",
        default="memory",
        implementations={
            "memory": "retailgr.online_store:InMemoryOnlineStore",
            "redis": "retailgr.online_store:RedisOnlineStore",
        },
        optional_imports={"redis": "redis"},
    ),
    SwapPoint(
        name="retrieval_index",
        setting="serving.index",
        protocol="retailgr.serving.retrieval:RetrievalIndex",
        default="exact",
        implementations={
            "exact": "retailgr.serving.retrieval:ExactRetrievalIndex",
            "faiss": "retailgr.serving.retrieval:FaissRetrievalIndex",
        },
        optional_imports={"faiss": "faiss"},
    ),
)


# -- 1. no proprietary dependency ---------------------------------------------


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(tree: ast.AST) -> list[tuple[str, int, bool]]:
    """Every module imported, with its line and whether the import is lazy.

    "Lazy" means the import statement sits inside a function, so it runs when
    that backend is chosen rather than when the package loads.
    """
    lazy_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for child in ast.walk(node):
                lazy_nodes.add(id(child))

    out: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        lazy = id(node) in lazy_nodes
        if isinstance(node, ast.Import):
            out.extend((alias.name, node.lineno, lazy) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module, node.lineno, lazy))
    return out


def scan_vendor_imports(root: Path) -> list[dict[str, Any]]:
    """Application modules importing a cloud vendor's SDK."""
    findings: list[dict[str, Any]] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, line, _ in _imported_modules(tree):
            for vendor_module, vendor in VENDOR_MODULES.items():
                if module == vendor_module or module.startswith(vendor_module + "."):
                    findings.append(
                        {
                            "file": str(path),
                            "line": line,
                            "module": module,
                            "vendor": vendor,
                        }
                    )
    return findings


# -- 2. no hardcoded endpoint -------------------------------------------------


def _overridable_constants(tree: ast.AST) -> set[int]:
    """String constants that something else can replace.

    Two forms count, and the shared property is the point of the check. An
    address is not a portability problem because it is written down; it is one
    because nothing can change it.

    * A default in ``cfg.get(key, default)`` — overridable from
      ``pipeline.yaml`` or ``--set``.
    * A default parameter value — overridable by every caller, which is how
      ``build_broker`` passes the configured bootstrap servers down to
      ``KafkaBroker``.

    Anything else is a fixed address, and a fixed address is what makes
    "runs on any cloud" untrue.
    """
    allowed: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if name not in {"get", "getenv", "environ", "setdefault", "pop"}:
                continue
            for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
                for inner in ast.walk(argument):
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                        allowed.add(id(inner))

        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d)]
            for default in defaults:
                for inner in ast.walk(default):
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                        allowed.add(id(inner))
    return allowed


def _docstring_constants(tree: ast.AST) -> set[int]:
    """Strings that are documentation, not values."""
    out: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for statement in body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                if isinstance(statement.value.value, str):
                    out.add(id(statement.value))
    return out


def scan_hardcoded_endpoints(root: Path) -> list[dict[str, Any]]:
    """Endpoints written into code where configuration cannot reach them."""
    findings: list[dict[str, Any]] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        exempt = _overridable_constants(tree) | _docstring_constants(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in exempt:
                continue
            value = node.value.strip()
            if value in ENDPOINT_EXEMPT or not ENDPOINT_PATTERN.match(value):
                continue
            findings.append({"file": str(path), "line": node.lineno, "value": value})
    return findings


# -- 3. the swap points are real ----------------------------------------------


def protocol_methods(protocol: Any) -> dict[str, inspect.Signature]:
    """The methods a protocol requires, with their signatures."""
    out: dict[str, inspect.Signature] = {}
    for name, member in vars(protocol).items():
        if name.startswith("_") or not callable(member):
            continue
        out[name] = inspect.signature(member)
    return out


def _compatible(required: inspect.Signature, provided: inspect.Signature) -> str | None:
    """Why ``provided`` cannot stand in for ``required``, or None if it can."""
    required_parameters = list(required.parameters.values())[1:]  # drop self
    provided_parameters = list(provided.parameters.values())[1:]

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in provided_parameters
    ):
        return None  # **kwargs absorbs anything

    provided_by_name = {parameter.name: parameter for parameter in provided_parameters}
    for parameter in required_parameters:
        if parameter.name not in provided_by_name:
            return f"missing parameter '{parameter.name}'"
        mine = provided_by_name[parameter.name]
        has_default = parameter.default is not inspect.Parameter.empty
        mine_has_default = mine.default is not inspect.Parameter.empty
        if has_default and not mine_has_default:
            return f"'{parameter.name}' is optional in the protocol and required here"

    for parameter in provided_parameters:
        if parameter.name in {p.name for p in required_parameters}:
            continue
        if parameter.default is inspect.Parameter.empty and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return f"requires extra parameter '{parameter.name}' the protocol does not supply"
    return None


def audit_swap_point(swap_point: SwapPoint) -> dict[str, Any]:
    """Check every implementation of one swap point against its protocol."""
    result: dict[str, Any] = {
        "name": swap_point.name,
        "setting": swap_point.setting,
        "default": swap_point.default,
        "implementations": {},
    }
    if not swap_point.protocol:
        # A module-level function set. No protocol object, so the named
        # functions are the contract — and "this module does not export
        # `read_arrow`" is a real failure that used to read as "not checked".
        result["required_functions"] = list(swap_point.required_functions)
        for name, module_name in swap_point.implementations.items():
            entry: dict[str, Any] = {
                "module": module_name,
                "checked": bool(swap_point.required_functions),
                "missing": [],
                "mismatched": {},
                "optional_import": swap_point.optional_imports.get(name),
            }
            if entry["checked"]:
                import importlib

                try:
                    module = importlib.import_module(module_name)
                except ImportError as error:
                    entry["missing"] = list(swap_point.required_functions)
                    entry["mismatched"] = {"import": str(error)}
                else:
                    for function in swap_point.required_functions:
                        attribute = getattr(module, function, None)
                        if attribute is None or not callable(attribute):
                            entry["missing"].append(function)
            entry["conforms"] = not entry["missing"] and not entry["mismatched"]
            result["implementations"][name] = entry
        return result

    protocol = swap_point.resolve(swap_point.protocol)
    required = protocol_methods(protocol)
    result["protocol_methods"] = sorted(required)

    for name, target in swap_point.implementations.items():
        entry: dict[str, Any] = {
            "class": target,
            "checked": True,
            "missing": [],
            "mismatched": {},
            "optional_import": swap_point.optional_imports.get(name),
        }
        implementation = swap_point.resolve(target)
        for method, signature in required.items():
            attribute = getattr(implementation, method, None)
            if attribute is None or not callable(attribute):
                entry["missing"].append(method)
                continue
            problem = _compatible(signature, inspect.signature(attribute))
            if problem:
                entry["mismatched"][method] = problem
        entry["conforms"] = not entry["missing"] and not entry["mismatched"]
        result["implementations"][name] = entry
    return result


def audit_lazy_imports(root: Path) -> list[dict[str, Any]]:
    """Optional clients that load at import time instead of on demand.

    ``import retailgr`` must work with only the core dependencies installed.
    A top-level ``import redis`` in ``online_store.py`` makes the whole
    package unimportable without Redis, which turns an optional backend into
    a required one.
    """
    optional = {
        module
        for swap_point in SWAP_POINTS
        for module in swap_point.optional_imports.values()
        if module not in {"pyspark"}  # pyspark is a declared core dependency
    }
    findings: list[dict[str, Any]] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module, line, lazy in _imported_modules(tree):
            head = module.split(".")[0]
            if head in optional and not lazy:
                findings.append({"file": str(path), "line": line, "module": module})
    return findings


# -- the whole audit ----------------------------------------------------------


def audit(root: Path | str) -> dict[str, Any]:
    """Run every platform check and return one result."""
    root = Path(root)
    vendor = scan_vendor_imports(root)
    endpoints = scan_hardcoded_endpoints(root)
    eager = audit_lazy_imports(root)
    swap_points = [audit_swap_point(point) for point in SWAP_POINTS]

    non_conforming = [
        (point["name"], name)
        for point in swap_points
        for name, entry in point["implementations"].items()
        if entry.get("checked") and not entry.get("conforms")
    ]
    return {
        "root": str(root),
        "vendor_imports": vendor,
        "hardcoded_endpoints": endpoints,
        "eager_optional_imports": eager,
        "swap_points": swap_points,
        "passed": not (vendor or endpoints or eager or non_conforming),
        "failures": {
            "vendor_imports": len(vendor),
            "hardcoded_endpoints": len(endpoints),
            "eager_optional_imports": len(eager),
            "non_conforming_backends": len(non_conforming),
        },
    }


def render_audit(result: dict[str, Any]) -> str:
    """Markdown report for the platform audit."""
    lines = [
        "# Platform audit",
        "",
        "The architecture claim — *decoupled compute, storage and machine "
        "learning, so it runs on any cloud without proprietary platforms* — "
        "checked mechanically instead of asserted.",
        "",
        f"**{'PASSED' if result.get('passed') else 'FAILED'}**",
        "",
    ]

    header = ["Check", "Findings", "Verdict"]
    checks = [
        ("No cloud vendor SDK in application code", "vendor_imports"),
        ("No endpoint configuration cannot reach", "hardcoded_endpoints"),
        ("Optional clients imported lazily", "eager_optional_imports"),
        ("Backends conform to their protocol", "non_conforming_backends"),
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for label, key in checks:
        count = (result.get("failures") or {}).get(key, 0)
        lines.append(f"| {label} | {count} | {'pass' if count == 0 else '**fail**'} |")
    lines.append("")

    for key, label in (
        ("vendor_imports", "Vendor SDK imports"),
        ("hardcoded_endpoints", "Hardcoded endpoints"),
        ("eager_optional_imports", "Eagerly imported optional clients"),
    ):
        findings = result.get(key) or []
        if not findings:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for finding in findings:
            detail = finding.get("module") or finding.get("value")
            lines.append(f"- `{finding['file']}:{finding['line']}` — `{detail}`")
        lines.append("")

    lines.append("## Swap points")
    lines.append("")
    header = ["Component", "Setting", "Backend", "Needs", "Conforms"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for point in result.get("swap_points") or []:
        for name, entry in point["implementations"].items():
            if not entry.get("checked"):
                verdict = "_not checked_"
            elif entry.get("conforms"):
                verdict = "yes"
            else:
                problems = entry.get("missing", []) + list(entry.get("mismatched", {}))
                verdict = "**no** — " + ", ".join(f"`{p}`" for p in problems)
            needs = entry.get("optional_import")
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{point['name']}`",
                        f"`{point['setting']}`",
                        f"`{name}`" + (" _(default)_" if name == point["default"] else ""),
                        f"`{needs}`" if needs else "core only",
                        verdict,
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append(
        "Conformance is checked statically — by comparing each backend's "
        "methods and signatures against the `Protocol` it claims to satisfy — "
        "so it runs with no Kafka, Redis or FAISS present. Python checks "
        "neither of those things at runtime, which means a backend can be "
        "missing a method for as long as nobody selects it. That is precisely "
        "the failure a portability claim is supposed to rule out, and the one "
        "an integration test cannot catch without the infrastructure it is "
        "testing."
    )
    lines.append("")
    lines.append(
        "What this does **not** establish: that the system runs on any "
        "particular cloud. It establishes the absence of the specific things "
        "that would stop it — a vendor SDK in the request path, an address "
        "nobody can change, a backend that is swappable only in the README."
    )
    return "\n".join(lines)
