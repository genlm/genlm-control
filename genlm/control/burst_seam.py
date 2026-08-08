"""The burst's serving seam: the ContextVars a burst writes and a potential reads.

Three variables, one protocol. A driver parks a row on ``_burst_row`` and answers its
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


# Per-burst override: {potential: LazyWeights} a potential's ``logw_next`` returns for
# itself instead of computing. Written by ``burst_serve``, read by ``PromptedLLM``.
_burst_logw_next_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_logw_next", default=None
)

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
    """Bind ``seat`` for one row's step, over a fresh warm-override scope.

    The warm scope must be per STEP even though the row's coroutine spans the whole
    burst: ``burst_serve`` sets it without a reset token, so the previous step's warm
    would otherwise still be readable at the top of the next."""
    tok_seat = _burst_row.set(seat)
    tok_warm = _burst_logw_next_overrides.set(None)
    try:
        yield
    finally:
        _burst_logw_next_overrides.reset(tok_warm)
        _burst_row.reset(tok_seat)


async def burst_serve(context):
    """Park until the burst delivers this row's warm for ``context``, then inject it.

    A no-op outside a burst. Every read at one context length is one decode step (target
    and proposal share the step's warm); a read past a draw parks, with a context ending
    in the token just drawn."""
    seat = _burst_row.get()
    if seat is not None:
        _burst_logw_next_overrides.set(await seat.next_warm(context))
