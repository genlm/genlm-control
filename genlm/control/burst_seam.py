"""The burst's serving seam: the ContextVars a burst writes and a potential reads.

Two variables, one protocol. A driver parks a row on ``_burst_row`` and answers its
logits reads from the engine's warm batch; ``_burst_lane_sums`` hands a potential its
banked per-token sums instead of re-scoring. This module imports nothing from
``potential`` or ``sampler`` so both sides can depend on it.
"""

import contextlib
import contextvars
from typing import NamedTuple


class LaneSums(NamedTuple):
    """Banked per-token sums a burst boundary serves in place of scoring: ``prefix`` for
    the live rows, ``complete`` for the terminated ones. Each is ``{potential: values}``,
    positional against the contexts the caller scores."""

    prefix: dict
    complete: dict


# The :class:`LaneSums` in scope at a burst boundary, else ``None``.
_burst_lane_sums: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_lane_sums", default=None
)

# A parked row's seat in the burst, bound for the whole of one row's step.
# Set only by the burst; ``None`` everywhere else.
_burst_row: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_row", default=None
)


@contextlib.contextmanager
def burst_lane_sums(prefix, complete):
    """Inject each leaf's banked sums for one boundary."""
    token = _burst_lane_sums.set(LaneSums(prefix, complete))
    try:
        yield
    finally:
        _burst_lane_sums.reset(token)


@contextlib.contextmanager
def burst_row(seat):
    """Bind ``seat`` for one row's step."""
    token = _burst_row.set(seat)
    try:
        yield
    finally:
        _burst_row.reset(token)


async def burst_serve(context):
    """Park until the burst delivers this row's warm for ``context``, returning it as a
    ``{potential: LazyWeights}`` override each leaf reads for itself; ``None`` outside a
    burst.

    Every read at one context length is one decode step (target and proposal share the
    step's warm); a read past a draw parks, with a context ending in the token just
    drawn."""
    seat = _burst_row.get()
    return await seat.next_warm(context) if seat is not None else None
