"""Generate a synthetic retail dataset with a real SKU hierarchy.

Public e-commerce datasets rarely carry size or colour variants, which is
exactly what the SKU-vs-product question is about. This generator produces a
catalog that does: products split into style-colours, style-colours split into
sizes, plus categories where the SKU *is* the product (grocery).

The interaction stream has sequential structure on purpose - a user drifts
inside an affinity cluster - so a sequence model has something to learn that a
popularity baseline cannot reproduce.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# category -> (share of catalog, variant style)
CATEGORIES = {
    "apparel": 0.30,
    "footwear": 0.18,
    "electronics": 0.14,
    "home": 0.18,
    "grocery": 0.20,
}

APPAREL_SIZES = ["XS", "S", "M", "L", "XL"]
FOOTWEAR_SIZES = ["24", "25", "26", "27", "28", "29"]
CAPACITIES = ["128GB", "256GB", "512GB"]
COLORS = ["black", "white", "navy", "olive", "red"]
BRANDS = [f"brand_{i:02d}" for i in range(12)]

RETURN_REASONS = ["too_small", "too_large", "fit", "damaged", "changed_mind"]


@dataclass
class SyntheticConfig:
    n_users: int = 3000
    n_products: int = 500
    days: int = 60
    seed: int = 7
    n_clusters: int = 12
    # Share of users who grant each purpose. `analytics` is the broader
    # consent and `personalisation` is nested inside it: a customer who will
    # not be analysed is not then separately personalised, so a generator
    # that samples them independently would produce a combination that does
    # not occur and make the measured cost of consent look smaller than it
    # is. The defaults are illustrative, not measured from anything — the
    # point of `retailgr consent-cost` is that the rate is a dial whose
    # effect you can look up rather than guess.
    consent_analytics_rate: float = 0.85
    consent_personalisation_rate: float = 0.70


def _build_catalog(rng: np.random.Generator, cfg: SyntheticConfig) -> list[dict]:
    """One row per SKU, carrying its style-colour and product parents."""
    rows: list[dict] = []
    category_names = list(CATEGORIES)
    category_probs = np.array([CATEGORIES[c] for c in category_names], dtype=float)
    category_probs /= category_probs.sum()

    for product_index in range(cfg.n_products):
        category = str(rng.choice(category_names, p=category_probs))
        product_id = f"P{product_index:05d}"
        brand = str(rng.choice(BRANDS))
        cluster = int(rng.integers(0, cfg.n_clusters))
        base_price = float(np.round(rng.uniform(5, 400), 2))

        if category in ("apparel", "footwear"):
            sizes = APPAREL_SIZES if category == "apparel" else FOOTWEAR_SIZES
            colors = list(rng.choice(COLORS, size=int(rng.integers(1, 4)), replace=False))
            for color in colors:
                style_color_id = f"{product_id}-{color}"
                for size in sizes:
                    rows.append(
                        {
                            "sku": f"{style_color_id}-{size}",
                            "style_color_id": style_color_id,
                            "product_id": product_id,
                            "category": category,
                            "brand": brand,
                            "size": size,
                            "color": color,
                            "capacity": "",
                            "list_price": base_price,
                            "cluster": cluster,
                        }
                    )
        elif category == "electronics":
            color = str(rng.choice(COLORS))
            for capacity in CAPACITIES:
                style_color_id = f"{product_id}-{color}"
                price = base_price * (1 + 0.25 * CAPACITIES.index(capacity))
                rows.append(
                    {
                        "sku": f"{style_color_id}-{capacity}",
                        "style_color_id": style_color_id,
                        "product_id": product_id,
                        "category": category,
                        "brand": brand,
                        "size": "",
                        "color": color,
                        "capacity": capacity,
                        "list_price": float(np.round(price, 2)),
                        "cluster": cluster,
                    }
                )
        elif category == "home":
            colors = list(rng.choice(COLORS, size=int(rng.integers(1, 3)), replace=False))
            for color in colors:
                style_color_id = f"{product_id}-{color}"
                rows.append(
                    {
                        "sku": style_color_id,
                        "style_color_id": style_color_id,
                        "product_id": product_id,
                        "category": category,
                        "brand": brand,
                        "size": "",
                        "color": color,
                        "capacity": "",
                        "list_price": base_price,
                        "cluster": cluster,
                    }
                )
        else:  # grocery: the SKU is the product
            rows.append(
                {
                    "sku": product_id,
                    "style_color_id": product_id,
                    "product_id": product_id,
                    "category": category,
                    "brand": brand,
                    "size": "",
                    "color": "",
                    "capacity": "",
                    "list_price": base_price,
                    "cluster": cluster,
                }
            )
    return rows


def _sample_events(
    rng: np.random.Generator, cfg: SyntheticConfig, catalog: list[dict]
) -> list[dict]:
    n_skus = len(catalog)
    clusters = np.array([row["cluster"] for row in catalog])
    categories = np.array([row["category"] for row in catalog])
    prices = np.array([row["list_price"] for row in catalog])
    sizes = np.array([row["size"] for row in catalog])

    # Long-tail popularity over SKUs.
    popularity = rng.pareto(1.2, size=n_skus) + 1.0
    popularity /= popularity.sum()

    cluster_members = {c: np.where(clusters == c)[0] for c in range(cfg.n_clusters)}

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events: list[dict] = []
    event_counter = 0

    for user_index in range(cfg.n_users):
        user_id = f"U{user_index:06d}"
        # Each user favours two clusters; this is the signal a sequence model
        # can pick up and a global popularity list cannot.
        favourites = rng.choice(cfg.n_clusters, size=2, replace=False)
        cluster_weights = np.full(cfg.n_clusters, 0.02)
        cluster_weights[favourites] = [0.6, 0.36]
        cluster_weights /= cluster_weights.sum()

        # A stable preferred size per user, used only for sized categories.
        preferred = {
            "apparel": str(rng.choice(APPAREL_SIZES)),
            "footwear": str(rng.choice(FOOTWEAR_SIZES)),
        }
        price_affinity = float(rng.uniform(0.3, 1.0))

        n_sessions = 1 + int(rng.poisson(3.5))
        for _ in range(n_sessions):
            session_id = f"S{event_counter:09d}"
            session_start = start + timedelta(
                seconds=float(rng.uniform(0, cfg.days * 24 * 3600))
            )
            cluster = int(rng.choice(cfg.n_clusters, p=cluster_weights))
            candidates = cluster_members[cluster]
            if candidates.size == 0:
                continue

            n_views = 1 + int(rng.poisson(4))
            cursor = session_start
            for _ in range(n_views):
                # Blend cluster affinity, popularity and price fit.
                weights = popularity[candidates].copy()
                fit = np.exp(-np.abs(prices[candidates] / prices.max() - price_affinity) * 3)
                weights = weights * fit
                # Sized categories: the user mostly looks at their own size.
                candidate_sizes = sizes[candidates]
                preferred_sizes = np.array(
                    [preferred.get(str(c), "") for c in categories[candidates]]
                )
                size_match = np.where(
                    (candidate_sizes == "") | (candidate_sizes == preferred_sizes), 1.0, 0.25
                )
                weights = weights * size_match
                if weights.sum() <= 0:
                    continue
                weights = weights / weights.sum()
                sku_index = int(rng.choice(candidates, p=weights))
                row = catalog[sku_index]

                cursor = cursor + timedelta(seconds=float(rng.uniform(20, 400)))
                event_counter += 1
                events.append(
                    {
                        "event_id": f"E{event_counter:010d}",
                        "user_id": user_id,
                        "session_id": session_id,
                        "event_type": "view",
                        "sku": row["sku"],
                        "event_ts": cursor.isoformat(),
                        "price": row["list_price"],
                        "quantity": 1,
                        "order_id": "",
                        "return_reason": "",
                    }
                )

                if rng.random() < 0.22:
                    cursor = cursor + timedelta(seconds=float(rng.uniform(5, 90)))
                    event_counter += 1
                    events.append(
                        {
                            "event_id": f"E{event_counter:010d}",
                            "user_id": user_id,
                            "session_id": session_id,
                            "event_type": "add_to_cart",
                            "sku": row["sku"],
                            "event_ts": cursor.isoformat(),
                            "price": row["list_price"],
                            "quantity": 1,
                            "order_id": "",
                            "return_reason": "",
                        }
                    )

                    if rng.random() < 0.45:
                        order_id = f"O{event_counter:010d}"
                        cursor = cursor + timedelta(seconds=float(rng.uniform(30, 600)))
                        event_counter += 1
                        events.append(
                            {
                                "event_id": f"E{event_counter:010d}",
                                "user_id": user_id,
                                "session_id": session_id,
                                "event_type": "purchase",
                                "sku": row["sku"],
                                "event_ts": cursor.isoformat(),
                                "price": row["list_price"],
                                "quantity": 1,
                                "order_id": order_id,
                                "return_reason": "",
                            }
                        )

                        # Returns land days later and carry a reason, which is
                        # what lets the ranker learn return risk.
                        wrong_size = row["size"] not in ("", preferred.get(row["category"], ""))
                        return_probability = 0.25 if wrong_size else 0.06
                        if rng.random() < return_probability:
                            reason = (
                                str(rng.choice(["too_small", "too_large", "fit"]))
                                if wrong_size
                                else str(rng.choice(["damaged", "changed_mind"]))
                            )
                            event_counter += 1
                            events.append(
                                {
                                    "event_id": f"E{event_counter:010d}",
                                    "user_id": user_id,
                                    "session_id": session_id,
                                    "event_type": "return",
                                    "sku": row["sku"],
                                    "event_ts": (
                                        cursor + timedelta(days=float(rng.uniform(1, 14)))
                                    ).isoformat(),
                                    "price": row["list_price"],
                                    "quantity": 1,
                                    "order_id": order_id,
                                    "return_reason": reason,
                                }
                            )
    events.sort(key=lambda e: e["event_ts"])
    _stamp_consent(rng, cfg, events)
    return events


def _stamp_consent(
    rng: np.random.Generator, cfg: SyntheticConfig, events: list[dict]
) -> None:
    """Attach a consent string to every event, decided once per user.

    Per user and not per event, because that is how a consent banner
    behaves: one decision covers the session and usually the account. Making
    it per event would scatter a user's history across the filter and hide
    the effect that actually matters — consent removes *whole users*, which
    is what makes it expensive for a sequence model. Half a history is not
    half a training example; below `min_events_per_user` it is none.

    What this deliberately does not model is consent changing over time. A
    withdrawal is not a value in this field; it is a request to remove the
    events, and it is handled by `retailgr forget`.
    """
    decisions: dict[str, str] = {}
    for event in events:
        user_id = event["user_id"]
        choice = decisions.get(user_id)
        if choice is None:
            granted = ["service"]
            if rng.random() < cfg.consent_analytics_rate:
                granted.append("analytics")
                # Nested, not independent: personalisation is a narrower
                # permission than analytics and cannot be granted without it.
                if rng.random() < (
                    cfg.consent_personalisation_rate / max(cfg.consent_analytics_rate, 1e-9)
                ):
                    granted.append("personalisation")
            choice = ",".join(granted)
            decisions[user_id] = choice
        event["consent"] = choice


def generate(output_dir: Path, cfg: SyntheticConfig) -> dict[str, Path]:
    """Write ``interactions.csv`` and ``catalog.csv`` under ``output_dir``."""
    rng = np.random.default_rng(cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = _build_catalog(rng, cfg)
    events = _sample_events(rng, cfg, catalog)

    catalog_path = output_dir / "catalog.csv"
    catalog_fields = [
        "sku",
        "style_color_id",
        "product_id",
        "category",
        "brand",
        "size",
        "color",
        "capacity",
        "list_price",
    ]
    with open(catalog_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=catalog_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(catalog)

    interactions_path = output_dir / "interactions.csv"
    event_fields = [
        "event_id",
        "user_id",
        "session_id",
        "event_type",
        "sku",
        "event_ts",
        "price",
        "quantity",
        "order_id",
        "return_reason",
        "consent",
    ]
    with open(interactions_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)

    return {"catalog": catalog_path, "interactions": interactions_path}
