"""The online store: the only state the serving path reads at request time.

It holds exactly what must be fresh and nothing else:

* the tail of each user's history (the last N events) — written by the
  streaming consumer within seconds of the event,
* live item state: price, active promo, stock per location — written from the
  compacted CDC topics,
* a precomputed fallback list per user — written nightly from the lakehouse,
  and served when a model tier times out.

Redis is the production choice; the in-memory implementation is what tests and
a laptop use. Both satisfy ``OnlineStore``, so the serving code never knows
which it has.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# How many recent events the serving path needs. The model reads a longer
# history from its own window; this is the hot tail that must be fresh.
DEFAULT_TAIL_LENGTH = 50


@dataclass
class TailEvent:
    """One event in a user's hot tail."""

    sku: str
    token: str
    action: str
    event_ts: int
    session_id: str = ""


@dataclass
class ItemState:
    """What the policy layer needs to know about an item right now."""

    sku: str
    price: float | None = None
    list_price: float | None = None
    promo_id: str | None = None
    stock_by_location: dict[str, int] = field(default_factory=dict)
    available: bool = True

    def in_stock_at(self, location: str | None) -> bool:
        if not self.available:
            return False
        if not self.stock_by_location:
            # No inventory feed for this item: treat availability as the only
            # signal rather than silently hiding the whole catalog.
            return True
        if location is None:
            return any(count > 0 for count in self.stock_by_location.values())
        return self.stock_by_location.get(location, 0) > 0


class OnlineStore(Protocol):
    def append_event(self, user_id: str, event: TailEvent, max_length: int = ...) -> None: ...

    def user_tail(self, user_id: str) -> list[TailEvent]: ...

    def put_item_state(self, state: ItemState) -> None: ...

    def item_state(self, sku: str) -> ItemState | None: ...

    def item_states(self, skus: list[str]) -> dict[str, ItemState]: ...

    def put_fallback(self, user_id: str, product_ids: list[str]) -> None: ...

    def fallback(self, user_id: str) -> list[str]: ...

    def forget(self, user_id: str) -> int: ...

    def close(self) -> None: ...


class InMemoryOnlineStore:
    """Thread-safe in-process store, used by tests and local runs."""

    def __init__(self, tail_length: int = DEFAULT_TAIL_LENGTH):
        self.tail_length = tail_length
        self._tails: dict[str, deque[TailEvent]] = {}
        self._items: dict[str, ItemState] = {}
        self._fallbacks: dict[str, list[str]] = {}
        self._global_fallback: list[str] = []
        self._lock = threading.RLock()

    def append_event(self, user_id: str, event: TailEvent, max_length: int | None = None) -> None:
        limit = max_length or self.tail_length
        with self._lock:
            tail = self._tails.get(user_id)
            if tail is None or tail.maxlen != limit:
                tail = deque(tail or (), maxlen=limit)
                self._tails[user_id] = tail
            tail.append(event)

    def user_tail(self, user_id: str) -> list[TailEvent]:
        with self._lock:
            return list(self._tails.get(user_id, ()))

    def put_item_state(self, state: ItemState) -> None:
        with self._lock:
            self._items[state.sku] = state

    def item_state(self, sku: str) -> ItemState | None:
        with self._lock:
            return self._items.get(sku)

    def item_states(self, skus: list[str]) -> dict[str, ItemState]:
        with self._lock:
            return {sku: self._items[sku] for sku in skus if sku in self._items}

    def put_fallback(self, user_id: str, product_ids: list[str]) -> None:
        with self._lock:
            self._fallbacks[user_id] = list(product_ids)

    def fallback(self, user_id: str) -> list[str]:
        with self._lock:
            return list(self._fallbacks.get(user_id) or self._global_fallback)

    def put_global_fallback(self, product_ids: list[str]) -> None:
        """Used when a user has no precomputed list — a cold start."""
        with self._lock:
            self._global_fallback = list(product_ids)

    def forget(self, user_id: str) -> int:
        """Remove everything this store holds about ``user_id``.

        Returns how many things were removed, which is what makes the
        erasure report able to say "nothing was there" rather than "done".
        The two are different answers and only one of them means the cache
        was already clean.
        """
        with self._lock:
            removed = 0
            if self._tails.pop(user_id, None) is not None:
                removed += 1
            if self._fallbacks.pop(user_id, None) is not None:
                removed += 1
            return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "users_with_tail": len(self._tails),
                "items_with_state": len(self._items),
                "users_with_fallback": len(self._fallbacks),
            }

    def close(self) -> None:
        return None


class RedisOnlineStore:
    """Redis-backed store. Keys are namespaced so one Redis can hold several
    environments.

    Tails are Redis lists trimmed on write, item state is a JSON string per
    SKU, and fallbacks are JSON lists. All three are O(1) reads, which is what
    keeps the store inside its 10 ms slice of the latency budget.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        namespace: str = "retailgr",
        tail_length: int = DEFAULT_TAIL_LENGTH,
        socket_timeout: float = 0.05,
        ttl_seconds: int | None = None,
    ):
        self.url = url
        self.namespace = namespace
        self.tail_length = tail_length
        self.socket_timeout = socket_timeout
        # Per-user keys expire; item state and the global fallback do not.
        # The tail was bounded by *count* and not by time, so a customer who
        # stopped shopping two years ago kept their last fifty events
        # indefinitely — a retention policy of "until the disk fills".
        self.ttl_seconds = ttl_seconds
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis
            except ImportError as error:  # pragma: no cover - depends on the env
                raise RuntimeError(
                    "redis is not installed. Install the serving extra "
                    "(pip install -e '.[serving]') or use online_store.backend=memory."
                ) from error
            self._client = redis.Redis.from_url(
                self.url,
                decode_responses=True,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_timeout,
            )
        return self._client

    def _tail_key(self, user_id: str) -> str:
        return f"{self.namespace}:tail:{user_id}"

    def _item_key(self, sku: str) -> str:
        return f"{self.namespace}:item:{sku}"

    def _fallback_key(self, user_id: str) -> str:
        return f"{self.namespace}:fallback:{user_id}"

    def _global_fallback_key(self) -> str:
        """The cold-start list, on a prefix of its own.

        It used to live at `{ns}:fallback:__global__`, inside the per-user
        keyspace. That is a trap with one exit: any sweep over
        `{ns}:fallback:*` — an erasure, a retention pass, a cleanup script
        someone writes at 2am — would take out the list every cold-start
        request falls back to, and the symptom would be empty
        recommendations for new users while every per-user path kept working.
        """
        return f"{self.namespace}:fallback_global"

    def append_event(self, user_id: str, event: TailEvent, max_length: int | None = None) -> None:
        limit = max_length or self.tail_length
        key = self._tail_key(user_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps(asdict(event)))
        pipe.ltrim(key, -limit, -1)
        if self.ttl_seconds:
            # Refreshed on every write, so the window is "this long since we
            # last heard from them" rather than "this long since we first
            # did" — an active customer is never dropped mid-session.
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def user_tail(self, user_id: str) -> list[TailEvent]:
        raw = self.client.lrange(self._tail_key(user_id), 0, -1)
        return [TailEvent(**json.loads(item)) for item in raw]

    def put_item_state(self, state: ItemState) -> None:
        self.client.set(self._item_key(state.sku), json.dumps(asdict(state)))

    def item_state(self, sku: str) -> ItemState | None:
        raw = self.client.get(self._item_key(sku))
        return None if raw is None else ItemState(**json.loads(raw))

    def item_states(self, skus: list[str]) -> dict[str, ItemState]:
        if not skus:
            return {}
        # One round trip for the whole candidate set; per-SKU gets would blow
        # the latency budget on their own.
        raw = self.client.mget([self._item_key(sku) for sku in skus])
        out: dict[str, ItemState] = {}
        for sku, payload in zip(skus, raw, strict=True):
            if payload:
                out[sku] = ItemState(**json.loads(payload))
        return out

    def put_fallback(self, user_id: str, product_ids: list[str]) -> None:
        self.client.set(
            self._fallback_key(user_id),
            json.dumps(product_ids),
            ex=self.ttl_seconds or None,
        )

    def fallback(self, user_id: str) -> list[str]:
        raw = self.client.get(self._fallback_key(user_id)) or self.client.get(
            self._global_fallback_key()
        )
        return json.loads(raw) if raw else []

    def put_global_fallback(self, product_ids: list[str]) -> None:
        # No TTL and no per-user prefix: this list is about the catalogue,
        # not about a person, and it is the answer of last resort. Expiring
        # it would make cold-start responses empty at the exact moment the
        # rest of the system is least able to help.
        self.client.set(self._global_fallback_key(), json.dumps(product_ids))

    def forget(self, user_id: str) -> int:
        """Delete this user's keys. Returns how many actually existed.

        Targeted `delete` rather than a `scan` over a pattern: the pattern
        form is what makes a cleanup accidentally take the shared key with
        it, and `__global__` used to sit squarely in the blast radius.
        """
        keys = [self._tail_key(user_id), self._fallback_key(user_id)]
        return int(self.client.delete(*keys))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def build_online_store(cfg: Any) -> OnlineStore:
    """Build the store named by ``online_store.backend`` in pipeline.yaml."""
    kind = str(cfg.get("online_store.backend", "memory")).lower()
    tail_length = int(cfg.get("online_store.tail_length", DEFAULT_TAIL_LENGTH))
    if kind == "memory":
        return InMemoryOnlineStore(tail_length=tail_length)
    if kind == "redis":
        ttl = cfg.get("privacy.retention.online_store_ttl_seconds", None)
        return RedisOnlineStore(
            url=str(cfg.get("online_store.url", "redis://localhost:6379/0")),
            namespace=str(cfg.get("online_store.namespace", "retailgr")),
            tail_length=tail_length,
            ttl_seconds=int(ttl) if ttl else None,
        )
    raise ValueError(f"unknown online_store.backend '{kind}'; expected 'memory' or 'redis'")
