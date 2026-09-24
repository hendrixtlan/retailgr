"""Resolve the model token for an item, per category.

The SKU is the system of record; this module decides only what the model treats
as one item. Everything here has a pure-Python implementation (used by the unit
tests and by the serving-side resolver) and a matching Spark expression, so the
offline and online paths cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import Column

LEVELS = ("sku", "style_color", "product")

LEVEL_COLUMNS = {
    "sku": "sku",
    "style_color": "style_color_id",
    "product": "product_id",
}


@dataclass(frozen=True)
class GranularityRule:
    """How one category maps a SKU to a model token."""

    level: str
    plus_attributes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown granularity level '{self.level}', expected one of {LEVELS}")


def _parse_rule(raw: Any) -> GranularityRule:
    if isinstance(raw, str):
        return GranularityRule(level=raw)
    if isinstance(raw, Mapping):
        return GranularityRule(
            level=str(raw.get("level", "sku")),
            plus_attributes=tuple(raw.get("plus_attributes", ()) or ()),
        )
    raise ValueError(f"cannot read granularity rule from {raw!r}")


class GranularityResolver:
    """Applies ``configs/granularity.yaml`` to items."""

    def __init__(self, config: Mapping[str, Any]):
        self.default = _parse_rule(config.get("default", "sku"))
        self.by_category: dict[str, GranularityRule] = {
            str(category).lower(): _parse_rule(rule)
            for category, rule in (config.get("by_category", {}) or {}).items()
        }
        size_affinity = config.get("size_affinity", {}) or {}
        self.size_affinity_categories = {
            str(c).lower() for c in (size_affinity.get("enabled_for", []) or [])
        }
        self.excluded_return_reasons = {
            str(r).lower() for r in (size_affinity.get("exclude_return_reasons", []) or [])
        }

    def rule_for(self, category: str | None) -> GranularityRule:
        if category is None:
            return self.default
        return self.by_category.get(str(category).lower(), self.default)

    def token_for(self, item: Mapping[str, Any]) -> str:
        """Token for one item row (``sku``, ``style_color_id``, ``product_id``,
        ``category`` and optionally ``attributes``)."""
        rule = self.rule_for(item.get("category"))
        base = item.get(LEVEL_COLUMNS[rule.level])
        if base is None:
            # A catalog missing the coarser id falls back to the SKU rather
            # than dropping the event.
            base = item["sku"]
        token = str(base)
        if rule.plus_attributes:
            attributes = item.get("attributes") or {}
            parts = [
                f"{name}={attributes[name]}"
                for name in rule.plus_attributes
                if attributes.get(name) is not None
            ]
            if parts:
                token = "|".join([token, *parts])
        return token

    def uses_size_affinity(self, category: str | None) -> bool:
        return str(category or "").lower() in self.size_affinity_categories

    def counts_as_kept_size(self, return_reason: str | None) -> bool:
        """True when a purchase tells us the size fit the customer."""
        if return_reason is None:
            return True
        return str(return_reason).lower() not in self.excluded_return_reasons

    # -- Spark ----------------------------------------------------------------

    def token_column(self) -> Column:
        """The same mapping as :meth:`token_for`, as a Spark column expression.

        Expects columns ``sku``, ``style_color_id``, ``product_id``,
        ``category`` and ``attributes`` (``map<string,string>``).
        """
        from pyspark.sql import functions as F

        def _base_for(rule: GranularityRule) -> Column:
            column = F.col(LEVEL_COLUMNS[rule.level])
            base = F.coalesce(column.cast("string"), F.col("sku").cast("string"))
            for name in rule.plus_attributes:
                value = F.element_at(F.col("attributes"), F.lit(name))
                base = F.when(
                    value.isNotNull(),
                    F.concat(base, F.lit("|"), F.lit(f"{name}="), value),
                ).otherwise(base)
            return base

        category = F.lower(F.coalesce(F.col("category").cast("string"), F.lit("")))
        expression = _base_for(self.default)
        # Build the chain in reverse so the first matching category wins.
        for name, rule in self.by_category.items():
            expression = F.when(category == F.lit(name), _base_for(rule)).otherwise(expression)
        return expression

    def describe(self) -> dict[str, str]:
        """Human-readable summary, written into the run report."""
        summary = {"_default": self.default.level}
        for category, rule in sorted(self.by_category.items()):
            label = rule.level
            if rule.plus_attributes:
                label += " + " + ",".join(rule.plus_attributes)
            summary[category] = label
        return summary


def uniform_resolver(level: str) -> GranularityResolver:
    """A resolver that applies one level to every category.

    This is what the SKU-vs-product experiment compares against the
    category-aware config.
    """
    return GranularityResolver({"default": level, "by_category": {}})
