"""What does a serving pod actually need?

Every `resources:` block in a Kubernetes manifest is a number someone chose.
Usually they chose it by copying another manifest, and the cluster then either
wastes the difference or gets OOMKilled under load. Both failures are quiet.

This measures the four numbers that decide the block, on the bundle that is
actually going to be served:

* **Bundle bytes** — what the init container pulls and what the image or
  volume has to hold.
* **Resident memory** — measured as a delta from a clean interpreter, so the
  answer is what *this* costs rather than what Python plus PyTorch costs.
  Reported both ways, because the pod pays for both.
* **Cold start** — import, load, and first scored request. This is what a
  readiness probe's `initialDelaySeconds` has to clear, and what a horizontal
  autoscaler pays every time it adds a replica.
* **Throughput** — requests per second per core, which is what turns a
  traffic figure into a replica count.

None of it is a guess, and where a number depends on the machine this ran on,
the report says so.
"""

from __future__ import annotations

import gc
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _rss_bytes() -> int:
    """Resident set size of this process.

    ``ru_maxrss`` is in kilobytes on Linux and bytes on macOS. Reading
    ``/proc`` when it exists avoids both the unit question and the "max"
    part — the peak is not what the pod holds once it is warm.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def directory_bytes(path: Path) -> dict[str, Any]:
    """Size of a bundle, file by file."""
    path = Path(path)
    files = {
        entry.name: entry.stat().st_size for entry in sorted(path.iterdir()) if entry.is_file()
    }
    return {"total_bytes": sum(files.values()), "files": files}


def seed_store(service: Any, users: int = 200, tail: int = 12, seed: int = 0) -> list[str]:
    """Populate the online store from the bundle's own vocabulary.

    Deliberately not ``jobs.bootstrap``, which needs Spark. A Spark driver in
    the measuring process would put a JVM's worth of py4j allocation into the
    RSS figure that is supposed to describe a serving pod — and a serving pod
    has no Spark in it.

    Synthetic tails are enough here because what is being measured is the cost
    of the request *path*, not the quality of its answer: retrieval, the
    policy layer and the ranker all do the same work whatever the history
    contains.
    """
    import numpy as np

    from retailgr.online_store import TailEvent

    rng = np.random.default_rng(seed)
    tokens = [token for token_id, token in service.bundle.token_by_id.items() if token_id > 0]
    if not tokens:
        return []

    user_ids: list[str] = []
    now = int(time.time())
    for index in range(users):
        user_id = f"sizing-{index}"
        for step in range(tail):
            token = tokens[int(rng.integers(0, len(tokens)))]
            skus = service.bundle.skus_for(token)
            service.store.append_event(
                user_id,
                TailEvent(
                    sku=skus[0] if skus else token,
                    token=token,
                    action="view",
                    event_ts=now - (tail - step) * 60,
                ),
            )
        user_ids.append(user_id)
    return user_ids


_PROBE = """
import json, time
start = time.perf_counter()
from retailgr.serving.sizing import _rss_bytes
interpreter_rss = _rss_bytes()

from retailgr.config import Config
from retailgr.serving.factory import build_service
from retailgr.serving.service import RequestContext
from retailgr.serving.sizing import seed_store
import gc
imported = time.perf_counter()
imported_rss = _rss_bytes()

cfg = Config.load({config!r})
service = build_service(cfg, {bundle!r})
loaded = time.perf_counter()
# Torch loads lazily inside build_service, so this is the first point at
# which the framework's own memory is on the books.
loaded_rss = _rss_bytes()

users = seed_store(service)
service.recommend(RequestContext(user_id=users[0] if users else "cold"), limit=10)
served = time.perf_counter()

for index in range(25):
    service.recommend(RequestContext(user_id=users[index % len(users)]), limit=10)
gc.collect()
warm = _rss_bytes()

requests = {requests}
t0 = time.perf_counter()
for index in range(requests):
    service.recommend(RequestContext(user_id=users[index % len(users)]), limit=10)
elapsed = time.perf_counter() - t0

print("SIZING " + json.dumps({{
    "import_s": imported - start,
    "load_s": loaded - imported,
    "first_request_s": served - loaded,
    "total_s": served - start,
    "interpreter_rss_bytes": interpreter_rss,
    "imported_rss_bytes": imported_rss,
    "loaded_rss_bytes": loaded_rss,
    "warm_rss_bytes": warm,
    "requests": requests,
    "seconds": elapsed,
    "requests_per_second": requests / elapsed if elapsed else None,
    "users_seeded": len(users),
    "ranker_served": bool(getattr(service.bundle, "has_ranker", False)),
}}))
"""


def _probe(bundle_dir: Path, config_path: Path, requests: int) -> dict[str, Any] | None:
    """Run one measurement in a fresh interpreter.

    Fresh each time, because cold start measured inside a process that has
    already imported torch answers a question no pod ever asks, and resident
    memory measured in the test session includes the test session.
    """
    import json as _json

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE.format(config=str(config_path), bundle=str(bundle_dir), requests=requests),
        ],
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )
    if completed.returncode != 0:
        return None
    for line in reversed(completed.stdout.strip().splitlines()):
        if line.startswith("SIZING "):
            return _json.loads(line[len("SIZING ") :])
    return None


def measure(
    bundle_dir: Path | str,
    config_path: Path | str = "configs/pipeline.yaml",
    requests: int = 200,
    repeats: int = 3,
) -> dict[str, Any]:
    """Everything a `resources:` block needs."""
    bundle_dir, config_path = Path(bundle_dir), Path(config_path)
    samples = [
        sample
        for _ in range(repeats)
        if (sample := _probe(bundle_dir, config_path, requests)) is not None
    ]
    if not samples:
        return {
            "bundle": directory_bytes(bundle_dir),
            "error": "the serving probe did not complete; is the bundle exported?",
        }

    def median(key: str) -> float:
        values = sorted(float(sample[key]) for sample in samples if sample.get(key) is not None)
        return values[len(values) // 2]

    cores = os.cpu_count() or 1
    throughput = median("requests_per_second")
    gc.collect()
    return {
        "bundle": directory_bytes(bundle_dir),
        "runs": len(samples),
        "cold_start": {
            "runs": len(samples),
            "import_s": round(median("import_s"), 3),
            "load_s": round(median("load_s"), 3),
            "first_request_s": round(median("first_request_s"), 3),
            "total_s": round(median("total_s"), 3),
        },
        "footprint": {
            "interpreter_rss_bytes": int(median("interpreter_rss_bytes")),
            "imported_rss_bytes": int(median("imported_rss_bytes")),
            "loaded_rss_bytes": int(median("loaded_rss_bytes")),
            "warm_rss_bytes": int(median("warm_rss_bytes")),
            "requests": requests,
            "requests_per_second": round(throughput, 1),
            "cores_available": cores,
            "requests_per_second_per_core": round(throughput / cores, 1),
            "ranker_served": bool(samples[0].get("ranker_served")),
            "users_seeded": int(samples[0].get("users_seeded", 0)),
        },
        "python": sys.version.split()[0],
    }


def _mib(value: float) -> str:
    return f"{value / 1024 / 1024:.0f} MiB"


def recommended_resources(result: dict[str, Any]) -> dict[str, Any]:
    """Turn the measurements into the numbers a manifest can carry.

    The request is what the pod needs when warm; the limit leaves headroom
    for the allocator and for a burst, because a memory limit is a kill
    switch rather than a target. Neither is a round number someone liked.
    """
    footprint = result.get("footprint") or {}
    warm = float(footprint.get("warm_rss_bytes") or 0)
    cold = float((result.get("cold_start") or {}).get("total_s") or 0)

    # 25% headroom on request, 2x on limit: the allocator does not return
    # freed pages promptly and a burst of concurrent requests allocates.
    request_bytes = warm * 1.25
    limit_bytes = warm * 2.0
    return {
        "memory_request_mib": int(request_bytes / 1024 / 1024) + 1,
        "memory_limit_mib": int(limit_bytes / 1024 / 1024) + 1,
        "measured_warm_mib": round(warm / 1024 / 1024, 1),
        # The readiness probe must not fire before the process can answer.
        "readiness_initial_delay_s": max(5, int(cold * 2) + 1),
        "measured_cold_start_s": cold,
        "requests_per_second_per_core": footprint.get("requests_per_second_per_core"),
    }


def render_sizing(result: dict[str, Any]) -> str:
    """Markdown report for the deployment footprint."""
    bundle = result.get("bundle") or {}
    cold = result.get("cold_start") or {}
    footprint = result.get("footprint") or {}
    advice = recommended_resources(result)

    lines = [
        "# Deployment footprint",
        "",
        "Measured, on the bundle that would actually be served. Every "
        "`resources:` block in `deploy/k8s/` carries these numbers rather "
        "than a figure copied from another manifest.",
        "",
        "## The bundle",
        "",
        f"**{_mib(bundle.get('total_bytes', 0))}** total.",
        "",
        "| File | Size |",
        "| --- | --- |",
    ]
    for name, size in (bundle.get("files") or {}).items():
        lines.append(f"| `{name}` | {size / 1024:,.0f} KiB |")
    lines.append("")

    lines.append("## Memory")
    lines.append("")
    lines.append("| Measurement | Value |")
    lines.append("| --- | --- |")
    steps = [
        ("A bare interpreter", "interpreter_rss_bytes"),
        ("After importing the serving stack", "imported_rss_bytes"),
        ("After loading the bundle (torch loads here, lazily)", "loaded_rss_bytes"),
        ("Warm, after 25 requests", "warm_rss_bytes"),
    ]
    previous = 0
    for label, key in steps:
        value = footprint.get(key, 0)
        delta = f" _(+{_mib(value - previous)})_" if previous else ""
        emphasis = "**" if key == "warm_rss_bytes" else ""
        lines.append(f"| {label} | {emphasis}{_mib(value)}{emphasis}{delta} |")
        previous = value
    lines.append("")
    bundle_mib = (bundle.get("total_bytes", 0)) / 1024 / 1024
    framework = footprint.get("loaded_rss_bytes", 0) - footprint.get("imported_rss_bytes", 0)
    lines.append(
        f"Worth reading the third row before anyone tries to shrink the model "
        f"to fit a pod: the bundle on disk is {bundle_mib:.1f} MiB and loading "
        f"it costs {_mib(framework)}, because that step is where PyTorch "
        "actually initialises. Almost all of this pod is framework, and none "
        "of that part gets smaller by training a smaller model."
    )
    lines.append("")

    lines.append("## Cold start")
    lines.append("")
    if cold.get("error"):
        lines.append(f"Not measured: {cold['error']}")
    else:
        lines.append("| Phase | Seconds |")
        lines.append("| --- | --- |")
        lines.append(f"| Import | {cold.get('import_s', 0):.2f} |")
        lines.append(f"| Load the bundle | {cold.get('load_s', 0):.2f} |")
        lines.append(f"| First request | {cold.get('first_request_s', 0):.2f} |")
        lines.append(f"| **Total** | **{cold.get('total_s', 0):.2f}** |")
        lines.append("")
        lines.append(
            f"Median of {cold.get('runs', 0)} fresh interpreters. This is what a "
            "readiness probe has to clear and what an autoscaler pays for every "
            "replica it adds."
        )
    lines.append("")

    lines.append("## Throughput")
    lines.append("")
    lines.append(
        f"{footprint.get('requests_per_second', 0):,.0f} req/s on "
        f"{footprint.get('cores_available', 0)} cores — "
        f"**{footprint.get('requests_per_second_per_core', 0):,.0f} req/s per core**, "
        f"{'with' if footprint.get('ranker_served') else 'without'} the ranker in the path."
    )
    lines.append("")
    lines.append(
        "In-process, so it excludes HTTP framing and the network. It is the "
        "ceiling the request path imposes, not a number to promise anyone."
    )
    lines.append("")

    lines.append("## What the manifests should say")
    lines.append("")
    lines.append("```yaml")
    lines.append("resources:")
    lines.append("  requests:")
    lines.append(f"    memory: {advice['memory_request_mib']}Mi   # measured warm "
                 f"{advice['measured_warm_mib']} MiB + 25%")
    lines.append("  limits:")
    lines.append(f"    memory: {advice['memory_limit_mib']}Mi   # 2x warm: a limit is a "
                 "kill switch, not a target")
    lines.append("readinessProbe:")
    lines.append(f"  initialDelaySeconds: {advice['readiness_initial_delay_s']}   "
                 f"# measured cold start {advice['measured_cold_start_s']}s, doubled")
    lines.append("```")
    lines.append("")
    lines.append(
        "No CPU limit is suggested. A CPU limit throttles rather than kills, "
        "and throttling a latency-budgeted request path trades a clean "
        "autoscaling signal for p99 spikes that look like a model problem."
    )
    return "\n".join(lines)
