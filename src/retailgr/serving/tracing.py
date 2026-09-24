"""Keeping the caller's trace id instead of inventing a new one.

The plumbing for this was almost complete and the last step was missing.
``RecommendationService.recommend`` takes a ``request_id``; the response
carries it; the exposure topic has a ``request_id`` field so a served list can
be joined back to the request that produced it. And the HTTP handler called
``service.recommend(context, limit=...)`` with no id at all, so every request
minted a fresh uuid4 and whatever id the caller was already carrying went in
the bin.

The effect is specific: a trace that crosses three services arrives here and
stops. The recommendation is in the exposure log under an id nobody upstream
has ever seen, so "why did this customer get this list at 14:32" is answerable
only by timestamp and luck.

**W3C Trace Context, parsed here rather than imported.** ``traceparent`` is a
fixed 55-character string with four dash-separated fields, and reading it is
about thirty lines. OpenTelemetry is the obvious dependency and the wrong one
for this: it pulls an SDK, an exporter and a configuration surface into a
serving image to extract a substring, and — like the broker and the online
store — the default here has to work with nothing installed. The ids produced
are the standard's, so any OTel collector downstream joins on them without
knowing this module exists.

What this deliberately does *not* do is emit spans. Propagating an id is what
makes the logs and the exposure topic join up; span export is a collector, a
protocol and an operational commitment, and claiming half of it under the name
"tracing" would be worse than having none.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

# version "-" trace-id(32 hex) "-" parent-id(16 hex) "-" flags(2 hex)
TRACEPARENT = re.compile(
    r"^(?P<version>[0-9a-f]{2})-"
    r"(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<parent_id>[0-9a-f]{16})-"
    r"(?P<flags>[0-9a-f]{2})$"
)

# All-zero ids are explicitly invalid in the specification.
INVALID_TRACE_ID = "0" * 32
INVALID_PARENT_ID = "0" * 16

TRACEPARENT_HEADER = "traceparent"
REQUEST_ID_HEADER = "x-request-id"


@dataclass(frozen=True)
class TraceContext:
    """What this request should be filed under."""

    trace_id: str
    parent_id: str | None = None
    sampled: bool = False
    # Where the id came from, so a metric or a log line can say whether the
    # caller supplied one. A service that thinks it is propagating and is
    # silently generating looks identical from the inside.
    source: str = "generated"

    @property
    def request_id(self) -> str:
        return self.trace_id


def parse_traceparent(value: str | None) -> TraceContext | None:
    """Read a W3C ``traceparent``, or return None if it is not one.

    Returns None rather than raising: a malformed header from an upstream
    service is not a reason to fail a customer's request, it is a reason to
    start a new trace and carry on.
    """
    if not value:
        return None
    match = TRACEPARENT.match(value.strip().lower())
    if not match:
        return None
    trace_id = match.group("trace_id")
    parent_id = match.group("parent_id")
    if trace_id == INVALID_TRACE_ID or parent_id == INVALID_PARENT_ID:
        return None
    # Version ff is forbidden; anything else unknown is forward-compatible and
    # the specification says to accept the fields we understand.
    if match.group("version") == "ff":
        return None
    return TraceContext(
        trace_id=trace_id,
        parent_id=parent_id,
        sampled=bool(int(match.group("flags"), 16) & 0x01),
        source="traceparent",
    )


def new_trace() -> TraceContext:
    return TraceContext(trace_id=uuid.uuid4().hex, source="generated")


def context_from_headers(headers: object) -> TraceContext:
    """The trace this request belongs to, in order of preference.

    1. ``traceparent`` — the standard, and the only one that also carries the
       parent span and the sampling decision.
    2. ``X-Request-Id`` — what most load balancers and older services send.
       Accepted because refusing it would throw away a real id on the grounds
       that it is the wrong shape.
    3. A new one.
    """
    get = getattr(headers, "get", None)
    if get is None:
        return new_trace()

    parsed = parse_traceparent(get(TRACEPARENT_HEADER))
    if parsed is not None:
        return parsed

    supplied = (get(REQUEST_ID_HEADER) or "").strip()
    if supplied and len(supplied) <= 200:
        return TraceContext(trace_id=supplied, source="x-request-id")
    return new_trace()


def format_traceparent(context: TraceContext, span_id: str | None = None) -> str:
    """The header to send onward, so the chain continues past this service."""
    trace_id = context.trace_id if len(context.trace_id) == 32 else uuid.uuid4().hex
    span = span_id or uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{span}-{'01' if context.sampled else '00'}"
