"""The action vocabulary — the second modality HSTU reads.

SASRec sees only which items a user touched. HSTU sees *what the user did* at
each position, which is the point of "Actions Speak Louder than Words": a view
that went nowhere and a purchase are different evidence about the same item.

Ids are fixed here rather than derived from the data so that they stay stable
across granularity variants, datasets and re-runs. Index 0 is padding.
"""

from __future__ import annotations

import numpy as np

# Order defines the ids; never reorder, only append.
ACTION_TYPES: tuple[str, ...] = (
    "view",
    "click",
    "add_to_cart",
    "remove_from_cart",
    "purchase",
    "return",
)

PAD_ACTION_ID = 0
ACTION_TO_ID: dict[str, int] = {name: i + 1 for i, name in enumerate(ACTION_TYPES)}
ID_TO_ACTION: dict[int, str] = {i: name for name, i in ACTION_TO_ID.items()}
UNKNOWN_ACTION_ID = len(ACTION_TYPES) + 1
ACTION_VOCAB_SIZE = len(ACTION_TYPES) + 2  # padding + known actions + unknown

# Actions that make the *next* item a supervised target.
#
# Table 1 of the paper defines retrieval targets as "Phi_i if a_i is positive,
# otherwise undefined". Which actions count as positive is a product decision,
# not a property of the architecture, so it is configurable. The default treats
# everything except an explicit rejection as positive, which keeps training
# supervision aligned with the default evaluation target set (all event types).
DEFAULT_POSITIVE_ACTIONS: tuple[str, ...] = (
    "view",
    "click",
    "add_to_cart",
    "purchase",
)


def encode_actions(actions: list[str] | None) -> np.ndarray:
    """Map action names to ids, unknown names to a single reserved id."""
    if not actions:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(
        [ACTION_TO_ID.get(str(a), UNKNOWN_ACTION_ID) for a in actions], dtype=np.int64
    )


def positive_action_ids(names: list[str] | tuple[str, ...] | None = None) -> set[int]:
    selected = tuple(names) if names else DEFAULT_POSITIVE_ACTIONS
    return {ACTION_TO_ID[name] for name in selected if name in ACTION_TO_ID}


# How committed each action is, for collapsing several events on one item into
# the strongest thing the customer did with it.
#
# This is not the id order and must not be derived from it. Ids follow the
# funnel, so `remove_from_cart` (4) outranks `add_to_cart` (3) and `return` (6)
# outranks `purchase` (5) — taking a max over ids would read a cancelled cart
# as a stronger signal than a completed one. Commitment is its own scale.
ACTION_COMMITMENT: dict[str, int] = {
    "remove_from_cart": 0,
    "return": 1,
    "view": 2,
    "click": 3,
    "add_to_cart": 4,
    "purchase": 5,
}
