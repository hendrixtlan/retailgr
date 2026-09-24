"""Measure the serving latency budget instead of asserting it.

The architecture document claims a p99 under 100 ms and splits that across
stages. A claimed budget is worth very little; this measures the real thing on
real requests, stage by stage, so the numbers in the document can be checked
rather than believed.

What this does and does not measure: it drives
``RecommendationService.recommend`` in-process, so it captures context
assembly, retrieval, the policy layer and the model forward pass, but not HTTP
parsing, TLS or network. Those add a few milliseconds at the edge and are the
easiest part of the budget to hold.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

from retailgr.config import Config
from retailgr.serving.policy import RequestContext


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    # Nearest-rank percentile: no interpolation, so a reported p99 is a
    # request that actually happened.
    index = min(len(ordered) - 1, max(0, int(round(fraction * len(ordered) + 0.5)) - 1))
    return ordered[index]


def run_bench(
    cfg: Config,
    bundle_dir: str | Path,
    variant: str = "config",
    requests: int = 500,
    limit: int = 20,
    warmup: int = 25,
) -> dict[str, Any]:
    """Bootstrap the online store, then time ``requests`` recommendations."""
    from retailgr.jobs import bootstrap
    from retailgr.online_store import build_online_store
    from retailgr.serving.factory import build_service
    from retailgr.spark_session import build_spark
    from retailgr.streaming.broker import build_broker

    store = build_online_store(cfg)
    bootstrap_stats = bootstrap.run(
        build_spark(cfg), cfg, broker=build_broker(cfg), store=store, variant=variant
    )
    service = build_service(cfg, bundle_dir, store=store)

    # Time real users, not synthetic ids: a user with no history takes the
    # cold-start path and would flatter the numbers.
    user_ids = _users_with_history(store, needed=requests + warmup)
    if not user_ids:
        raise RuntimeError(
            "no users with history in the online store; run bootstrap with more events"
        )

    for index in range(warmup):
        service.recommend(
            RequestContext(user_id=user_ids[index % len(user_ids)]), limit=limit
        )

    totals: list[float] = []
    stages: dict[str, list[float]] = {
        "context_ms": [],
        "retrieval_ms": [],
        "filter_ms": [],
        "ranking_ms": [],
        "rerank_ms": [],
    }
    served_from: dict[str, int] = {}
    items_returned: list[int] = []

    started = time.perf_counter()
    for index in range(requests):
        user_id = user_ids[index % len(user_ids)]
        response = service.recommend(RequestContext(user_id=user_id), limit=limit)
        totals.append(response.timings.total_ms)
        for name in stages:
            stages[name].append(getattr(response.timings, name))
        served_from[response.served_from] = served_from.get(response.served_from, 0) + 1
        items_returned.append(len(response.items))
    wall_seconds = time.perf_counter() - started

    return {
        "bundle": str(bundle_dir),
        "model_version": service.bundle.model_version,
        "model_type": service.bundle.manifest.model_type,
        "variant": variant,
        "dataset": cfg.dataset_name,
        "requests": requests,
        "warmup": warmup,
        "limit": limit,
        "index_size": service.index.size,
        "retrieval_k": service.config.retrieval_k,
        "rank_k": service.config.rank_k,
        "users_available": len(user_ids),
        "throughput_rps": round(requests / wall_seconds, 1) if wall_seconds else None,
        "served_from": served_from,
        "mean_items_returned": round(statistics.fmean(items_returned), 2),
        "total_ms": _summary(totals),
        "stages": {name: _summary(values) for name, values in stages.items()},
        "bootstrap": bootstrap_stats,
    }


def _users_with_history(store: Any, needed: int) -> list[str]:
    if hasattr(store, "_tails"):
        users = [user for user, tail in store._tails.items() if len(tail) >= 3]
        return users[:needed] if users else []
    return []


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(_percentile(values, 0.50), 3),
        "p95": round(_percentile(values, 0.95), 3),
        "p99": round(_percentile(values, 0.99), 3),
        "max": round(max(values), 3),
    }


# The per-stage budget claimed in the architecture document, in milliseconds.
CLAIMED_BUDGET_MS = {
    "context_ms": 10.0,
    "retrieval_ms": 25.0,
    "filter_ms": 5.0,
    "ranking_ms": 35.0,
    "rerank_ms": 5.0,
}
CLAIMED_TOTAL_MS = 100.0


def render_bench(result: dict[str, Any]) -> str:
    """Markdown report, measured against the claimed budget."""
    lines = [
        "# Serving latency",
        "",
        f"- Model: `{result['model_type']}` (`{result['model_version']}`)",
        f"- Dataset: `{result['dataset']}`, variant `{result['variant']}`",
        f"- Retrieval index: {result['index_size']} items, "
        f"top-{result['retrieval_k']} then {result['rank_k']} to the ranker",
        f"- {result['requests']} requests over {result['users_available']} real user "
        f"histories, after {result['warmup']} warm-up calls",
        f"- Throughput: {result['throughput_rps']} req/s, single process, single thread",
        "",
        "In-process measurement: it covers context assembly, retrieval, the model",
        "forward pass and the policy layer, but not HTTP or network.",
        "",
        "## Per stage",
        "",
    ]
    header = ["Stage", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Budget (ms)", "Verdict"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for name, summary in result["stages"].items():
        budget = CLAIMED_BUDGET_MS.get(name)
        p99 = summary.get("p99", float("nan"))
        verdict = "-" if budget is None else ("within" if p99 <= budget else "over")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name.removesuffix('_ms')}`",
                    f"{summary.get('p50', 0):.3f}",
                    f"{summary.get('p95', 0):.3f}",
                    f"{p99:.3f}",
                    "-" if budget is None else f"{budget:.0f}",
                    verdict,
                ]
            )
            + " |"
        )

    total = result["total_ms"]
    verdict = "within" if total.get("p99", 1e9) <= CLAIMED_TOTAL_MS else "over"
    lines.append(
        "| "
        + " | ".join(
            [
                "**end to end**",
                f"**{total.get('p50', 0):.3f}**",
                f"**{total.get('p95', 0):.3f}**",
                f"**{total.get('p99', 0):.3f}**",
                f"**{CLAIMED_TOTAL_MS:.0f}**",
                f"**{verdict}**",
            ]
        )
        + " |"
    )
    lines.append("")

    served = result.get("served_from", {})
    if served:
        parts = ", ".join(f"{name}: {count}" for name, count in sorted(served.items()))
        lines.append(f"Responses by path: {parts}.")
        if set(served) - {"model"}:
            lines.append("")
            lines.append(
                "Anything other than `model` means a stage fell back. A fallback is "
                "cheap to serve, so a high fallback rate makes these numbers look "
                "better than the system is - check this line before trusting the "
                "table above."
            )
    lines.append("")
    lines.append(
        f"Mean items returned: {result['mean_items_returned']} of {result['limit']} requested."
    )
    return "\n".join(lines)
