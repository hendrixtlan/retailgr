"""The token mapping is the decision Stage 1 exists to make, so it is tested
in isolation from Spark."""

from __future__ import annotations

import pytest

from retailgr.granularity import GranularityResolver, uniform_resolver

CONFIG = {
    "default": "sku",
    "by_category": {
        "apparel": {"level": "style_color"},
        "electronics": {"level": "product", "plus_attributes": ["capacity"]},
        "grocery": "sku",
    },
    "size_affinity": {
        "enabled_for": ["apparel", "footwear"],
        "exclude_return_reasons": ["too_small", "fit"],
    },
}

TSHIRT = {
    "sku": "P001-navy-M",
    "style_color_id": "P001-navy",
    "product_id": "P001",
    "category": "apparel",
    "attributes": {"size": "M", "color": "navy"},
}
PHONE = {
    "sku": "P900-black-256GB",
    "style_color_id": "P900-black",
    "product_id": "P900",
    "category": "electronics",
    "attributes": {"capacity": "256GB", "color": "black"},
}
MILK = {
    "sku": "P500",
    "style_color_id": "P500",
    "product_id": "P500",
    "category": "grocery",
    "attributes": {},
}


def test_apparel_collapses_sizes_but_keeps_colour():
    resolver = GranularityResolver(CONFIG)
    small = dict(TSHIRT, sku="P001-navy-S", attributes={"size": "S", "color": "navy"})
    red = dict(TSHIRT, sku="P001-red-M", style_color_id="P001-red")

    assert resolver.token_for(TSHIRT) == resolver.token_for(small) == "P001-navy"
    assert resolver.token_for(red) == "P001-red"


def test_electronics_keeps_capacity_but_collapses_colour():
    resolver = GranularityResolver(CONFIG)
    other_colour = dict(PHONE, sku="P900-white-256GB", style_color_id="P900-white")
    bigger = dict(PHONE, sku="P900-black-512GB", attributes={"capacity": "512GB"})

    assert resolver.token_for(PHONE) == "P900|capacity=256GB"
    assert resolver.token_for(other_colour) == "P900|capacity=256GB"
    assert resolver.token_for(bigger) == "P900|capacity=512GB"


def test_unlisted_category_falls_back_to_the_default():
    resolver = GranularityResolver(CONFIG)
    unknown = dict(MILK, category="pet_supplies")
    assert resolver.token_for(unknown) == unknown["sku"]
    assert resolver.token_for(MILK) == "P500"


def test_missing_parent_id_falls_back_to_the_sku():
    resolver = GranularityResolver(CONFIG)
    broken = dict(TSHIRT, style_color_id=None)
    assert resolver.token_for(broken) == broken["sku"]


def test_missing_attribute_is_skipped_not_rendered_as_none():
    resolver = GranularityResolver(CONFIG)
    no_capacity = dict(PHONE, attributes={"color": "black"})
    assert resolver.token_for(no_capacity) == "P900"


def test_uniform_resolver_ignores_categories():
    for level, expected in (
        ("sku", TSHIRT["sku"]),
        ("style_color", TSHIRT["style_color_id"]),
        ("product", TSHIRT["product_id"]),
    ):
        assert uniform_resolver(level).token_for(TSHIRT) == expected


def test_size_affinity_rules():
    resolver = GranularityResolver(CONFIG)
    assert resolver.uses_size_affinity("apparel") is True
    assert resolver.uses_size_affinity("grocery") is False
    # A purchase returned because it did not fit says nothing about size.
    assert resolver.counts_as_kept_size(None) is True
    assert resolver.counts_as_kept_size("damaged") is True
    assert resolver.counts_as_kept_size("too_small") is False


def test_unknown_level_is_rejected_early():
    with pytest.raises(ValueError):
        GranularityResolver({"default": "sku_color"})
